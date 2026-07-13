"""Generate SfM camera poses aligned to the LiDAR world frame.

Automates the July-2026 campaign pipeline that produced the champion
``sfm_viewmats.npz`` (the single biggest splat-quality lever, ~+4 dB):

  A. extract every camera frame from the IMAGE bag to portrait JPEGs,
     keyed by the exact bag timestamp process_splat.py trains on;
  B. pycolmap SIFT + sequential matching (overlap 20, quadratic) +
     GLOMAP global mapping (incremental fallback) with the factory
     OPENCV_FISHEYE calibration as a shared-camera prior;
  C. sim(3)-align (Umeyama, iterative median trimming) the SfM poses to
     the LiDAR odometry trajectory — residuals ARE the LiDAR drift;
  D. loop closure: spatially-close frame pairs from the aligned poses
     (the sequential graph has zero cross-pass matches, so revisited
     objects drift a few cm between passes — the "blurred arm" failure),
     matched into a copy of the database, then a second global mapping
     and final alignment.

Output: ``--output`` npz with ``names``, ``bag_ts_ns``, ``viewmats``
(world→cam, LiDAR world frame) ready for ``process_splat.py --sfm-poses``,
plus ``fisheye_params`` — the bundle-adjusted OPENCV_FISHEYE intrinsics
(fx,fy,cx,cy,k1..k4, portrait frame) for ``--fisheye-intrinsics``, so the
undistortion uses the same lens model the poses were optimised under.

Needs an interpreter with pycolmap (with GLOMAP bindings), rosbags, scipy
and cv2 — the server locates one via lidar_jobs._sfm_python(). CPU-bound;
a ~900-frame 20 MP scan takes on the order of an hour or two.
"""

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

# Reuse the bag/calibration/pose helpers from the existing pipeline stages.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import process_pointcloud as ppc  # noqa: E402
from process_splat import _interp_pose  # noqa: E402

# Loop-closure pair generation (stage D) — values from the campaign run that
# fixed the cross-pass drift on scan 20260527212832.
MIN_FRAME_GAP = 25  # closer pairs are already covered by sequential overlap
MAX_DIST_M = 2.0  # camera centres within 2 m
MAX_ANGLE_DEG = 60.0  # optical axes within 60 deg (else no shared surface)
NEIGHBOURS = 8  # nearest spatial neighbours kept per image


def progress(pct: int, msg: str):
    print(f"PROGRESS:{pct}:{msg}", flush=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Stage A: extract portrait frames ─────────────────────────────────


def extract_frames(image_bag: Path, camera: str, images_dir: Path, ts_csv: Path):
    """Write every camera frame as a portrait JPEG named ``NNNN.jpg`` and a
    CSV mapping name → bag timestamp.

    Uses the same reader as the point-cloud/splat stages (ppc.read_images),
    so the bag timestamps here are EXACTLY the keys build_camera_frames will
    look up in the output npz — no clock reconciliation needed.

    Frames already on disk are kept (cheap resume after an interrupted run),
    but the CSV is always rewritten from the full bag pass.
    """
    import cv2

    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (bag_ts, img) in enumerate(ppc.read_images(image_bag, camera, rot="ccw")):
        name = f"{i:04d}.jpg"
        dst = images_dir / name
        if not dst.exists():
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append((name, int(bag_ts)))
        if (i + 1) % 100 == 0:
            progress(
                2 + min(10, (i + 1) // 100), f"SfM poses: extracted {i + 1} frames…"
            )
    with open(ts_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "bag_ts_ns"])
        w.writerows(rows)
    log(f"extracted {len(rows)} frames to {images_dir}")
    return dict(rows)


# ── Stage B: SfM (features / matching / mapping) ─────────────────────


def fisheye_prior(scan: Path, camera: str):
    """Factory OPENCV_FISHEYE params (fx fy cx cy k1-k4) as the SfM prior."""
    calib = ppc.load_calibration(scan / "calibration", camera)
    K, D = calib["K"], calib["D"]
    return [K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0], D[1], D[2], D[3]]


def extract_and_match(db: Path, images_dir: Path, params):
    import pycolmap

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV_FISHEYE"
    reader.camera_params = ",".join(str(p) for p in params)

    progress(14, "SfM poses: extracting SIFT features (CPU, slow)…")
    pycolmap.extract_features(
        database_path=db,
        image_path=images_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
    )
    log("feature extraction done")

    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 20
    pairing.quadratic_overlap = True
    pairing.loop_detection = False
    progress(30, "SfM poses: sequential matching (overlap 20)…")
    pycolmap.match_sequential(database_path=db, pairing_options=pairing)
    log("matching done")


def run_mapping(db: Path, images_dir: Path, sparse_dir: Path, label: str):
    """GLOMAP global mapping with incremental fallback; returns the best
    reconstruction (most registered images) or None."""
    import pycolmap

    sparse_dir.mkdir(parents=True, exist_ok=True)
    recs = None
    try:
        log(f"{label}: global mapping (GLOMAP)…")
        recs = pycolmap.global_mapping(
            database_path=db, image_path=images_dir, output_path=sparse_dir
        )
    except Exception as e:
        log(f"{label}: global mapping failed ({e!r}); falling back to incremental")
    if not recs:
        log(f"{label}: incremental mapping…")
        recs = pycolmap.incremental_mapping(
            database_path=db, image_path=images_dir, output_path=sparse_dir
        )
    if not recs:
        return None
    best_id, best = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    for rid, rec in recs.items():
        log(
            f"{label}: model {rid}: {rec.num_reg_images()} registered, "
            f"{rec.num_points3D()} pts, mean reproj "
            f"{rec.compute_mean_reprojection_error():.2f} px"
        )
    return best


def poses_from_rec(rec):
    """name → (R world→cam 3x3, camera centre 3) plus refined fisheye params."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    out = {}
    for img in rec.images.values():
        p = img.cam_from_world()
        q, t = p.rotation.quat, p.translation  # quat is x,y,z,w (scipy order)
        R = Rotation.from_quat(q).as_matrix()
        out[img.name] = (R, -R.T @ np.asarray(t))
    cam = next(iter(rec.cameras.values()))
    return out, np.asarray(cam.params, dtype=float)


# ── Stage C: sim(3) alignment to the LiDAR trajectory ────────────────


def umeyama(src, dst):
    """Least-squares sim(3): dst ≈ s·R·src + t.  Returns s, R, t."""
    import numpy as np

    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    t = mu_d - s * R @ mu_s
    return s, R, t


def align_to_lidar(sfm, ts_by_name, traj_path: Path, T_lidar_cam, cam_time_offset):
    """Fit sim(3) SfM→LiDAR-world and return per-frame rigid world→cam
    viewmats in the LiDAR frame, with SfM-failure frames dropped.

    Returns (names, viewmats, export_mask, stats_str) — names/viewmats are the
    EXPORTED (kept) frames only.
    """
    import numpy as np

    names = sorted(n for n in sfm if n in ts_by_name)
    if len(names) < 10:
        raise RuntimeError(f"only {len(names)} SfM poses have timestamps")

    traj = np.load(traj_path)
    traj_ts = traj["ts"].astype(np.int64)
    poses = traj["poses"].astype(np.float64)
    order = np.argsort(traj_ts)
    traj_ts, poses = traj_ts[order], poses[order]
    offset_ns = int(round(cam_time_offset * 1e9))

    def lidar_vm(ts_ns):
        P, _, _ = _interp_pose(traj_ts, poses, ts_ns + offset_ns)
        return T_lidar_cam @ np.linalg.inv(P)

    c_sfm = np.array([sfm[n][1] for n in names])
    lidar_vms = np.array([lidar_vm(ts_by_name[n]) for n in names])
    c_lidar = np.array([-vm[:3, :3].T @ vm[:3, 3] for vm in lidar_vms])

    # Iterative median-based trimming: gross SfM failures (100s of metres)
    # would poison a mean/std trim, and moderate residuals are expected —
    # they're the LiDAR drift we're measuring, so the threshold must scale
    # with the data.
    keep = np.ones(len(names), bool)
    for _ in range(10):
        s, R, t = umeyama(c_sfm[keep], c_lidar[keep])
        res = np.linalg.norm((s * (R @ c_sfm.T).T + t) - c_lidar, axis=1)
        thresh = max(5 * np.median(res[keep]), 0.05)
        new_keep = res < thresh
        if new_keep.sum() < 10 or (new_keep == keep).all():
            break
        keep = new_keep
    stats = (
        f"sim(3) scale {s:.4f}, {len(names) - keep.sum()} outliers trimmed; "
        f"centre residual (= LiDAR drift) med {np.median(res) * 100:.1f} cm, "
        f"p90 {np.percentile(res, 90) * 100:.1f} cm"
    )
    log(stats)

    # Frames wildly off the fit are SfM failures, not drift — keep them out
    # of the export so the trainer never sees a garbage pose. (Moderate
    # residuals stay: there the SfM pose is the correction we want.)
    export = res < max(20 * np.median(res[keep]), 10.0)
    if (~export).any():
        bad = [n for n, e in zip(names, export) if not e]
        log(f"excluding {len(bad)} SfM-failure frame(s): {bad[:10]}")

    # Rigid per-image world→cam in the LiDAR frame: rotate the SfM pose into
    # the aligned frame, put the camera at the similarity-transformed centre.
    viewmats = []
    for n in names:
        R_i, c_i = sfm[n]
        R_new = R_i @ R.T
        c_w = s * R @ c_i + t
        vm = np.eye(4)
        vm[:3, :3] = R_new
        vm[:3, 3] = -R_new @ c_w
        viewmats.append(vm)
    viewmats = np.array(viewmats)

    exp_names = [n for n, e in zip(names, export) if e]
    return exp_names, viewmats[export], stats


# ── Stage D: loop closure ────────────────────────────────────────────


def build_loop_pairs(names, viewmats, pairs_path: Path):
    """Spatially-close, temporally-distant frame pairs from aligned poses.

    The sequential match graph ties a scan's two visits to the same object
    together only through the long chain of relative poses, so the passes
    drift a few cm apart — invisible on far walls, ~13 px of reprojection at
    2 m. These pairs let the mapper close that loop.
    """
    import numpy as np

    centres = np.array([-vm[:3, :3].T @ vm[:3, 3] for vm in viewmats])
    axes = np.array([vm[2, :3] for vm in viewmats])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    idx = np.array([int(n.split(".")[0]) for n in names])

    pairs = set()
    for i in range(len(names)):
        d = np.linalg.norm(centres - centres[i], axis=1)
        ang = np.degrees(np.arccos(np.clip(axes @ axes[i], -1, 1)))
        cand = np.where(
            (np.abs(idx - idx[i]) > MIN_FRAME_GAP)
            & (d < MAX_DIST_M)
            & (ang < MAX_ANGLE_DEG)
        )[0]
        cand = cand[np.argsort(d[cand])][:NEIGHBOURS]
        for j in cand:
            pairs.add(tuple(sorted((names[i], names[j]))))
    with open(pairs_path, "w") as f:
        for a, b in sorted(pairs):
            f.write(f"{a} {b}\n")
    log(f"loop pairs: {len(pairs)}")
    return len(pairs)


def match_loop_pairs(db: Path, pairs_path: Path):
    import pycolmap

    pairing = pycolmap.ImportedPairingOptions()
    pairing.match_list_path = str(pairs_path)
    pycolmap.match_image_pairs(database_path=db, pairing_options=pairing)
    log("loop-pair matching done")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scan", required=True, help="scan folder (bags + calibration)")
    p.add_argument(
        "--traj",
        required=True,
        help="LiDAR odometry trajectory npz (ts + poses — a cloud's "
        ".traj.npz sidecar) to align the SfM poses to",
    )
    p.add_argument("--output", required=True, help="output sfm_viewmats npz path")
    p.add_argument(
        "--workspace",
        default=None,
        help="scratch dir for images/databases (default: <output>_work); "
        "removed on success unless --keep-workspace",
    )
    p.add_argument("--camera", default="front")
    p.add_argument(
        "--cam-time-offset",
        type=float,
        default=-0.025,
        help="camera↔LiDAR clock skew in seconds (same as process_splat.py)",
    )
    p.add_argument(
        "--no-loop-closure",
        action="store_true",
        help="skip stage D (faster, but revisited objects may stay drifted)",
    )
    p.add_argument("--keep-workspace", action="store_true")
    p.add_argument(
        "--min-registered-frac",
        type=float,
        default=0.4,
        help="fail if SfM registers fewer than this fraction of frames",
    )
    args = p.parse_args()

    import numpy as np

    scan = Path(args.scan)
    output = Path(args.output)
    ws = Path(args.workspace) if args.workspace else output.parent / (
        output.stem + "_work"
    )
    ws.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    bags = sorted(scan.glob("IMAGE_*.bag")) if scan.exists() else []
    image_bag = bags[0] if bags else None
    if image_bag is None:
        print(f"ERROR: no IMAGE_*.bag in {scan}", flush=True)
        return 1
    if not Path(args.traj).exists():
        print(f"ERROR: trajectory not found: {args.traj}", flush=True)
        return 1

    t0 = time.time()
    progress(1, "SfM poses: extracting camera frames from the bag…")
    images_dir = ws / "images"
    ts_by_name = extract_frames(image_bag, args.camera, images_dir, ws / "timestamps.csv")
    n_frames = len(ts_by_name)
    if n_frames < 20:
        print(f"ERROR: only {n_frames} frames in the image bag", flush=True)
        return 1

    # A database left by an interrupted run may be partial (matching half
    # done) — start clean; the extracted JPEGs above are reused as-is.
    db = ws / "database.db"
    if db.exists():
        db.unlink()
    params = fisheye_prior(scan, args.camera)
    extract_and_match(db, images_dir, params)

    progress(38, "SfM poses: global mapping (pass 1)…")
    rec = run_mapping(db, images_dir, ws / "sparse", "pass 1")
    if rec is None:
        print("ERROR: SfM produced no reconstruction", flush=True)
        return 1
    min_reg = max(20, int(args.min_registered_frac * n_frames))
    if rec.num_reg_images() < min_reg:
        print(
            f"ERROR: SfM registered only {rec.num_reg_images()}/{n_frames} "
            "frames — not enough texture/overlap for reliable poses. Use "
            "downscale 2 (odometry poses) for this scan.",
            flush=True,
        )
        return 1

    calib = ppc.load_calibration(scan / "calibration", args.camera)
    T = calib["T"].astype(np.float64)

    progress(55, "SfM poses: aligning to the LiDAR trajectory…")
    sfm, fisheye = poses_from_rec(rec)
    names, viewmats, stats = align_to_lidar(
        sfm, ts_by_name, Path(args.traj), T, args.cam_time_offset
    )

    if not args.no_loop_closure:
        progress(60, "SfM poses: matching loop-closure pairs…")
        pairs_path = ws / "pairs_loop.txt"
        if build_loop_pairs(names, viewmats, pairs_path) > 0:
            db_loop = ws / "database_loop.db"
            shutil.copy2(db, db_loop)
            match_loop_pairs(db_loop, pairs_path)
            progress(68, "SfM poses: global mapping (pass 2, loop-closed)…")
            rec2 = run_mapping(db_loop, images_dir, ws / "sparse_loop", "pass 2")
            # A weaker pass-2 model means the added matches hurt — keep pass 1.
            if rec2 is not None and rec2.num_reg_images() >= rec.num_reg_images():
                progress(88, "SfM poses: final alignment…")
                sfm, fisheye = poses_from_rec(rec2)
                names, viewmats, stats = align_to_lidar(
                    sfm, ts_by_name, Path(args.traj), T, args.cam_time_offset
                )
            else:
                log("pass 2 weaker than pass 1 — keeping pass-1 poses")

    # Write-then-rename so a crash never leaves a half-written npz where the
    # server's existence check would find it. (np.savez appends ".npz" to bare
    # paths, so give it an open handle.)
    tmp = Path(str(output) + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(
            f,
            names=np.array(names),
            bag_ts_ns=np.array([ts_by_name[n] for n in names], dtype=np.int64),
            viewmats=viewmats,
            fisheye_params=fisheye,
            camera=np.array(args.camera),
        )
    tmp.rename(output)
    log(f"wrote {output} ({len(names)}/{n_frames} viewmats; {stats})")

    if not args.keep_workspace:
        shutil.rmtree(ws, ignore_errors=True)
    progress(100, f"SfM poses ready ({len(names)} frames, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
