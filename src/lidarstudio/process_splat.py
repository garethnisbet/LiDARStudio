#!/usr/bin/env python3
"""
Gaussian Splat Generator

Trains a 3D Gaussian Splatting model from a coloured LiDAR point cloud plus the
posed camera frames, and writes a standard 62-field 3DGS PLY.

Camera poses are NOT recovered with COLMAP here — they come for free from the
KISS-ICP trajectory saved by ``process_pointcloud.py`` (``<cloud>.traj.npz``):
each image's world→camera matrix is the LiDAR pose at that timestamp composed
with the fixed lidar→camera extrinsic from ``calibration/calib.json``.  The
coloured cloud seeds the Gaussians, the photos supervise their colour/opacity.

GPU: training uses gsplat's CUDA rasterizer (needs an NVIDIA GPU + nvcc) and the
KISS-ICP trajectory sidecar (``<cloud>.traj.npz``); it errors out if either is
missing rather than emitting a low-value CPU splat.

Usage (called automatically by lidar_server.py):
    python process_splat.py
        --scan       /path/to/20260527195949
        --output     /project/splats/splat_20260527195949.ply
        --pointcloud /project/pointclouds/pointcloud_20260527195949.ply
        [--iterations 7000] [--downscale 1] [--image-rot ccw]

Progress protocol:
    Print "PROGRESS:<percent>:<message>" for the UI progress bar; other lines
    appear in the log view.  Training also prints "PROGRESS <step>/<iters>".
"""

import argparse
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Reduce CUDA fragmentation before torch initialises its caching allocator — the
# tile-intersection step on millions of gaussians allocates large transient
# buffers and OOM'd mid-run without this.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Reuse the bag/calibration/pose helpers from the point-cloud stage.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import process_pointcloud as ppc  # noqa: E402


def progress(pct: int, msg: str):
    print(f"PROGRESS:{pct}:{msg}", flush=True)


# ── CUDA discovery ───────────────────────────────────────────────────


def ensure_cuda_home() -> None:
    """Point CUDA_HOME at a usable nvcc so gsplat can JIT-compile its kernels."""
    home = os.environ.get("CUDA_HOME")
    if home and (Path(home) / "bin" / "nvcc").exists():
        return
    for c in (Path.home() / ".local/micromamba/envs/cuda121", Path("/usr/local/cuda")):
        if (c / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = str(c)
            os.environ["PATH"] = f"{c / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
            return


# ── PLY I/O ──────────────────────────────────────────────────────────


def load_coloured_ply(path: Path):
    """Return (xyz Nx3 float32, rgb Nx3 float[0,1]) from a coloured PLY."""
    import numpy as np

    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, np.float32)
        rgb = (
            np.asarray(pcd.colors, np.float32)
            if pcd.has_colors()
            else np.full((len(pts), 3), 0.5, np.float32)
        )
        return pts, rgb
    except ImportError:
        pass
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"].data
    pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    if all(c in v.dtype.names for c in ("red", "green", "blue")):
        rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32) / 255.0
    else:
        rgb = np.full((len(pts), 3), 0.5, np.float32)
    return pts, rgb


# ── Camera dataset (KISS-ICP trajectory ∘ lidar→camera extrinsic) ─────


def _interp_pose(traj_ts, poses, ts):
    """
    Continuous-time SE(3) pose at time ``ts`` by interpolating the two bracketing
    trajectory samples: SLERP on rotation, LERP on translation.  The LiDAR runs at
    ~10 Hz, so a camera frame lands up to ±50 ms from the nearest sample; over that
    gap a handheld rig rotates/translates enough to smear the projection.  Nearest-
    neighbour snapping bakes that error into the training pose — interpolation
    removes it.  Timestamps outside the trajectory clamp to the nearest endpoint
    (no extrapolation).  Returns (pose 4×4, alpha in [0,1], bracket_gap_ns).
    """
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp

    n = len(traj_ts)
    j = int(np.searchsorted(traj_ts, ts))
    if j <= 0:
        return poses[0], 0.0, 0
    if j >= n:
        return poses[-1], 1.0, 0
    t0, t1 = int(traj_ts[j - 1]), int(traj_ts[j])
    P0, P1 = poses[j - 1], poses[j]
    gap = t1 - t0
    if gap <= 0:
        return P1, 1.0, 0
    a = (int(ts) - t0) / gap
    R = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([P0[:3, :3], P1[:3, :3]])))(
        [a]
    )[0].as_matrix()
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = (1.0 - a) * P0[:3, 3] + a * P1[:3, 3]
    return out, a, gap


def _dynamic_mask_png(img_bgr, K, viewmat, box, canon_pts=None):
    """PNG-encoded loss mask (0 = exclude) for a dynamic object.

    Two regions are excluded, both clipped to the projected world-space
    ``box`` silhouette:

    1. Pixels near saturated-blue paint — the object wherever it has MOVED to
       in this frame. The box gate keeps unrelated blue surfaces (ceiling
       banner) trainable; the colour gate keeps walls behind the object
       trainable (the raw silhouette from a nearby camera covers most of the
       frame).
    2. Pixels where the CANONICAL configuration stands (``canon_pts``, seed
       points inside the box, world coords). Without this, late frames show
       background where the object used to be and actively CARVE the
       canonical reconstruction from the far side — half-built, half-erased
       gaussians render as chaotic smear.
    """
    import cv2
    import numpy as np

    H, W = img_bgr.shape[:2]
    lo, hi = (np.asarray(b, dtype=np.float64) for b in box)
    g = np.linspace(0.0, 1.0, 6)
    pts = np.array(
        [lo + np.array([a, b, c]) * (hi - lo) for a in g for b in g for c in g]
    )
    Xc = (viewmat[:3, :3] @ pts.T + viewmat[:3, 3:4]).T
    front = Xc[:, 2] > 0.05
    sil = np.zeros((H, W), np.uint8)
    if front.sum() >= 3:
        uv = (K @ Xc[front].T).T
        uv = np.clip(uv[:, :2] / uv[:, 2:3], -1e6, 1e6)
        cv2.fillConvexPoly(sil, cv2.convexHull(uv.astype(np.int32)), 255)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue = (
        (hsv[..., 0] >= 100)
        & (hsv[..., 0] <= 130)
        & (hsv[..., 1] > 80)
        & (hsv[..., 2] > 50)
    ).astype(np.uint8) * 255
    # Dilate so the object's non-blue parts (black wrist/joints) adjacent to
    # blue paint are covered too.
    k = max(31, (int(0.04 * W)) | 1)
    excl = cv2.bitwise_and(sil, cv2.dilate(blue, np.ones((k, k), np.uint8)))
    if canon_pts is not None and len(canon_pts):
        Xc = (viewmat[:3, :3] @ canon_pts.T + viewmat[:3, 3:4]).T
        front = Xc[:, 2] > 0.05
        if front.any():
            uv = (K @ Xc[front].T).T
            uv = (uv[:, :2] / uv[:, 2:3]).astype(np.int32)
            inb = (
                (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            )
            canon = np.zeros((H, W), np.uint8)
            canon[uv[inb, 1], uv[inb, 0]] = 255
            kc = max(15, (int(0.02 * W)) | 1)
            excl = cv2.bitwise_or(excl, cv2.dilate(canon, np.ones((kc, kc), np.uint8)))
    keep = cv2.bitwise_not(excl)
    ok, png = cv2.imencode(".png", keep)
    return png if ok else None


def _frame_cache_root(explicit=None, need_bytes=0):
    """Pick a directory on a real disk with room for the frame cache. The default
    temp dir is often small (or tmpfs = RAM, which would defeat the purpose), so
    prefer an explicit dir / $LIDARSTUDIO_FRAME_CACHE, then the biggest-free of a
    few real-disk candidates. Never a Dropbox-synced path (would trigger a sync)."""
    cands = [explicit, os.environ.get("LIDARSTUDIO_FRAME_CACHE")]
    cands += ["/FastDrive", tempfile.gettempdir(), str(Path.home())]
    usable = []
    for c in cands:
        if not c or not os.path.isdir(c) or not os.access(c, os.W_OK):
            continue
        if "dropbox" in c.lower():
            continue
        try:
            free = shutil.disk_usage(c).free
        except OSError:
            continue
        if c in (explicit, os.environ.get("LIDARSTUDIO_FRAME_CACHE")) and free > need_bytes:
            return c  # honour an explicit choice that fits
        usable.append((free, c))
    if not usable:
        return tempfile.gettempdir()
    return max(usable)[1]  # most free space


def build_camera_frames(
    scan: Path,
    traj_path: Path,
    camera: str,
    image_rot: str,
    downscale: int,
    undistort: bool = True,
    undistort_fov: float = 120.0,
    undistort_scale: float = 1.0,
    cam_time_offset: float = 0.0,
    sfm_poses=None,
    drop_blurry: float = 0.0,
    mask_box=None,
    mask_from: int = 0,
    mask_canon_pts=None,
    memmap_frames: bool = False,
    frame_cache_dir=None,
):
    """
    Return a list of {image HxWx3 uint8 RGB, K 3x3, viewmat 4x4 (world→cam)}.
    Images stay uint8 until the training step touches them: a float32 cache of
    900+ full-res frames is tens of GB and OOM-kills modest machines.

    For each camera frame we query the LiDAR pose at its (offset-corrected) exact
    timestamp by SE(3) interpolation, P, and form the world→camera matrix
    T_extrinsic @ inv(P)  — points go world → lidar (inv P) → camera (T).  Images
    are rotated to portrait to match K.  ``cam_time_offset`` (seconds) shifts the
    image clock onto the LiDAR clock before the query (camera_ts + offset), to
    absorb a constant sensor time skew; sweep it if frames look consistently
    mis-registered.

    ``sfm_poses`` (path to an npz with ``bag_ts_ns`` + ``viewmats``, from
    align_sfm.py) replaces the odometry-derived pose of every frame it covers
    with an externally-computed world→cam viewmat (already in the LiDAR world
    frame, extrinsic included); frames it does not cover are DROPPED, since a
    mix of SfM and odometry poses would be inconsistent.

    ``undistort`` (default) remaps each ~180° fisheye photo to a virtual PINHOLE
    of ``undistort_fov`` degrees horizontal, using OpenCV's own Kannala-Brandt
    undistort — the SAME lens model that coloured the cloud. This is essential:
    gsplat's ``camera_model="fisheye"`` does NOT reproduce OpenCV KB (it squashes
    the scene into a central dome), so training in fisheye renders garbage and the
    photometric loss can never converge. Training on undistorted pinhole frames
    (``camera_model="pinhole"``) gives a provably-correct projection; the cost is
    losing the extreme (heavily-distorted) periphery beyond ``undistort_fov``.
    """
    import cv2
    import numpy as np

    calib = ppc.load_calibration(scan / "calibration", camera)
    K0 = calib["K"].astype(np.float64)
    D = calib["D"].astype(np.float64)
    T = calib["T"].astype(np.float64)  # lidar → camera (4×4)

    traj = np.load(traj_path)
    traj_ts = traj["ts"].astype(np.int64)
    poses = traj["poses"].astype(np.float64)  # (N,4,4) sensor→world (levelled)
    order = np.argsort(traj_ts)
    traj_ts, poses = traj_ts[order], poses[order]

    # Locate the image bag in the scan folder.
    img_bags = sorted(scan.glob("IMAGE_*.bag")) or sorted(scan.glob("*.bag"))
    img_bag = next((b for b in img_bags if "IMAGE" in b.name.upper()), None)
    if img_bag is None:
        raise FileNotFoundError(f"no IMAGE_*.bag in {scan}")

    radial = D.reshape(4).astype(np.float64)
    Dk = radial.reshape(4, 1)
    maps = None  # (map1, map2, K_pinhole), built once
    offset_ns = int(round(cam_time_offset * 1e9))
    diag = {"n": 0, "clamped": 0, "snap_err_ns": [], "gap_ns": []}
    sfm_vm = None
    if sfm_poses is not None:
        ext = np.load(sfm_poses)
        sfm_vm = {
            int(t): vm.astype(np.float64)
            for t, vm in zip(ext["bag_ts_ns"], ext["viewmats"])
        }
        print(f"  external SfM poses: {len(sfm_vm)} frames from {sfm_poses}", flush=True)
    skipped = 0
    frames = []
    # Optional on-disk frame cache: instead of holding every uint8 frame in host
    # RAM (~size × count — 56 GB at downscale-1/us-1.5), stream them to a single
    # memmap file and hand out memmap views. The OS page cache keeps hot frames
    # resident, so RSS stays bounded by RAM regardless of resolution × count.
    mm_file = mm_path = mm_shape = None
    mm_count = 0
    for bag_i, (ts, img) in enumerate(ppc.read_images(img_bag, camera, rot=image_rot)):
        if sfm_vm is not None and int(ts) not in sfm_vm:
            skipped += 1
            continue
        h, w = img.shape[:2]
        K = K0.copy()
        if downscale > 1:
            img = cv2.resize(
                img, (w // downscale, h // downscale), interpolation=cv2.INTER_AREA
            )
            K = K / downscale
            K[2, 2] = 1.0
        H, W = img.shape[:2]

        if undistort:
            if maps is None:
                # Virtual pinhole: f = (W/2)/tan(fov/2); centred principal point.
                # ``undistort_scale`` enlarges the output canvas beyond the source
                # size: a wide-FOV tan projection at 1:1 samples the image centre
                # BELOW the fisheye's native px/radian (e.g. 433 vs 563 at 120°),
                # silently blurring the sharpest part of every training photo.
                Wo, Ho = round(W * undistort_scale), round(H * undistort_scale)
                f = (Wo / 2.0) / np.tan(np.radians(undistort_fov) / 2.0)
                Kp = np.array([[f, 0, Wo / 2.0], [0, f, Ho / 2.0], [0, 0, 1.0]])
                m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                    K, Dk, np.eye(3), Kp, (Wo, Ho), cv2.CV_16SC2
                )
                maps = (m1, m2, Kp)
                print(
                    f"  undistort: fisheye → pinhole {undistort_fov:.0f}° "
                    f"(f={f:.1f}px, {Wo}×{Ho})",
                    flush=True,
                )
            img = cv2.remap(
                img,
                maps[0],
                maps[1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            K = maps[2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # uint8; float [0,1] on GPU

        if sfm_vm is not None:
            viewmat = sfm_vm[int(ts)]
        else:
            tq = int(ts) + offset_ns
            P, a, gap = _interp_pose(traj_ts, poses, tq)
            viewmat = T @ np.linalg.inv(P)  # world → camera
            # Diagnostics: how far the old nearest-neighbour snap would have
            # been, and the bracket size we interpolated across.
            diag["n"] += 1
            if gap == 0:
                diag["clamped"] += 1
            else:
                diag["gap_ns"].append(gap)
                diag["snap_err_ns"].append(min(a, 1.0 - a) * gap)

        if memmap_frames:
            if mm_file is None:
                mm_shape = rgb.shape  # (H, W, 3), shared by all frames this run
                root = _frame_cache_root(frame_cache_dir)
                cdir = tempfile.mkdtemp(prefix="lidarstudio_frames_", dir=root)
                atexit.register(shutil.rmtree, cdir, ignore_errors=True)
                mm_path = os.path.join(cdir, "frames.u8")
                mm_file = open(mm_path, "wb", buffering=1 << 20)
                print(
                    f"  frame cache: disk memmap at {cdir} "
                    f"({int(np.prod(mm_shape)) / 1e6:.0f} MB/frame → host RAM stays flat)",
                    flush=True,
                )
            if rgb.shape != mm_shape:
                raise RuntimeError(
                    f"frame shape {rgb.shape} != {mm_shape}; the memmap cache "
                    "requires uniform frame sizes"
                )
            np.ascontiguousarray(rgb).tofile(mm_file)
            frame = {"idx": mm_count, "K": K.astype(np.float64), "viewmat": viewmat}
            mm_count += 1
        else:
            frame = {"image": rgb, "K": K.astype(np.float64), "viewmat": viewmat}
        if mask_box is not None and bag_i >= mask_from and undistort:
            frame["mask_png"] = _dynamic_mask_png(
                img, K, viewmat, mask_box, mask_canon_pts
            )
        if drop_blurry > 0:
            # Laplacian variance as a sharpness score; motion-blurred handheld
            # frames teach the model soft edges everywhere they're sampled.
            frame["sharp"] = cv2.Laplacian(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var()
        if not undistort:  # fisheye path keeps radial coeffs
            frame["radial"] = radial
        frames.append(frame)
    if skipped:
        print(
            f"  dropped {skipped} frame(s) without an SfM pose "
            f"({len(frames)} kept)",
            flush=True,
        )
    if drop_blurry > 0 and frames:
        scores = np.array([f.pop("sharp") for f in frames])
        cut = np.quantile(scores, drop_blurry)
        kept = [f for f, s in zip(frames, scores) if s >= cut]
        print(
            f"  sharpness filter: dropped {len(frames) - len(kept)} blurriest "
            f"frame(s) of {len(frames)} (laplacian-var cut {cut:.1f}, "
            f"median {np.median(scores):.1f})",
            flush=True,
        )
        frames = kept
    if memmap_frames and mm_file is not None:
        # Close the writer and re-open read-only; hand each surviving frame a
        # memmap view (dropped frames keep their slot on disk but are never read).
        mm_file.close()
        mm = np.memmap(
            mm_path, dtype=np.uint8, mode="r", shape=(mm_count, *mm_shape)
        )
        for f in frames:
            f["image"] = mm[f.pop("idx")]
    n_masked = sum(1 for f in frames if f.get("mask_png") is not None)
    if n_masked:
        print(
            f"  dynamic-object mask active on {n_masked}/{len(frames)} frame(s)",
            flush=True,
        )
    if not frames:
        raise RuntimeError("no camera frames loaded from image bag")
    if diag["snap_err_ns"]:
        se = np.array(diag["snap_err_ns"]) / 1e6  # ms
        gp = np.array(diag["gap_ns"]) / 1e6  # ms
        print(
            f"  pose query: continuous-time SE(3) interp, offset "
            f"{cam_time_offset * 1e3:+.1f} ms; {diag['clamped']} frame(s) clamped "
            f"to trajectory ends",
            flush=True,
        )
        print(
            f"    interp removed a median {np.median(se):.1f} ms "
            f"(max {se.max():.1f} ms) nearest-neighbour snap; "
            f"bracket median {np.median(gp):.1f} ms",
            flush=True,
        )
    return frames


# ── Gaussian model + loss (adapted from raven/train_splat.py) ─────────


def rgb_to_sh(rgb):
    return (rgb - 0.5) / 0.28209479177387814


def knn_scale(points, k: int = 4):
    import numpy as np
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    d, _ = tree.query(points, k=k + 1, workers=-1)
    s = d[:, 1:].mean(axis=1)
    # Cap isolated/sparse points: their nearest neighbour is far, so an unbounded
    # knn scale makes one giant Gaussian that renders as a white "bloom" halo.
    # Clamp to a few× the typical spacing so seeds stay surface-sized.
    hi = float(np.percentile(s, 90)) * 3.0
    return np.clip(s, 1e-4, max(hi, 1e-3))


def pca_surfel_frame(
    points, k: int = 12, radius_factor: float = 0.6, flatten: float = 0.25
):
    """Per-point anisotropic disc frame from local kNN PCA (shape from geometry).

    Returns (log_scales Nx3, quats_wxyz Nx4): two in-plane axes sized to the local
    point spacing and a thin normal axis (``flatten``×), oriented by the surface
    tangent plane. Lets trained splats *seed* anisotropy from the (accurate) LiDAR
    geometry instead of trying to learn it from inconsistent fisheye photos.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    tree = cKDTree(points)
    d, idx = tree.query(points, k=k + 1, workers=-1)
    nn = np.clip(d[:, 1], 1e-5, None)
    nbr = points[idx[:, 1:]]
    cen = nbr - nbr.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", cen, cen) / k
    _, evecs = np.linalg.eigh(cov)  # ascending eigenvalues
    R = np.stack([evecs[:, :, 2], evecs[:, :, 1], evecs[:, :, 0]], axis=2)  # [t1,t2,n]
    R[np.linalg.det(R) < 0, :, 2] *= -1.0  # keep proper rotation
    q = Rotation.from_matrix(R).as_quat()  # [x,y,z,w]
    quats = np.column_stack([q[:, 3], q[:, :3]]).astype(np.float32)  # [w,x,y,z]
    r_in = nn * radius_factor
    r_in = np.minimum(r_in, float(np.median(r_in)) * 3.0)
    s_in = np.log(np.clip(r_in, 1e-5, None))
    s_nr = np.log(np.clip(r_in * flatten, 1e-6, None))
    log_scales = np.stack([s_in, s_in, s_nr], axis=1).astype(np.float32)
    return log_scales, quats


def _se3_exp(t):
    """
    4×4 SE(3) transform from a 6-vector tangent [ωx,ωy,ωz, vx,vy,vz] (torch,
    differentiable).  Built with torch.stack (no in-place writes) so autograd can
    backprop through it into a learnable pose delta; matrix_exp handles the
    small-angle limit exactly, so no ω→0 special-casing is needed.
    """
    import torch

    wx, wy, wz, vx, vy, vz = t
    z = torch.zeros((), dtype=t.dtype, device=t.device)
    rows = torch.stack(
        [
            torch.stack([z, -wz, wy, vx]),
            torch.stack([wz, z, -wx, vy]),
            torch.stack([-wy, wx, z, vz]),
            torch.stack([z, z, z, z]),
        ]
    )
    return torch.linalg.matrix_exp(rows)


def build_gaussians(
    points, rgb, sh_degree, device, init_opacity: float = 0.1, anisotropic: bool = False
):
    import numpy as np
    import torch

    n = len(points)
    means = torch.tensor(points, dtype=torch.float32, device=device)
    if anisotropic:  # seed surface-aligned discs from geometry
        log_scales, quats_np = pca_surfel_frame(points)
        scales = torch.tensor(log_scales, dtype=torch.float32, device=device)
        quats = torch.tensor(quats_np, dtype=torch.float32, device=device)
    else:
        scales = torch.tensor(
            np.log(knn_scale(points)), dtype=torch.float32, device=device
        )
        scales = scales[:, None].repeat(1, 3)
        quats = torch.zeros(n, 4, device=device)
        quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((n,), init_opacity, device=device))
    num_sh = (sh_degree + 1) ** 2
    colors = torch.zeros(n, num_sh, 3, device=device)
    colors[:, 0, :] = torch.tensor(rgb_to_sh(rgb), dtype=torch.float32, device=device)
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(colors[:, :1, :]),
            "shN": torch.nn.Parameter(colors[:, 1:, :]),
        }
    ).to(device)


def make_optimizers(params, lr_scale: float, freeze=()):
    import torch

    specs = {
        "means": 1.6e-4 * lr_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    return {
        name: torch.optim.Adam([params[name]], lr=lr, eps=1e-15)
        for name, lr in specs.items()
        if name not in freeze
    }


_SSIM_WINDOW = {}


def windowed_ssim(pred, gt, window_size: int = 11, sigma: float = 1.5):
    import torch
    import torch.nn.functional as F

    key = (window_size, sigma, pred.device)
    win = _SSIM_WINDOW.get(key)
    if win is None:
        coords = (
            torch.arange(window_size, dtype=torch.float32, device=pred.device)
            - window_size // 2
        )
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        win2d = (g[:, None] * g[None, :])[None, None]
        win = win2d.expand(3, 1, window_size, window_size).contiguous()
        _SSIM_WINDOW[key] = win
    a = pred.permute(2, 0, 1)[None]
    b = gt.permute(2, 0, 1)[None]
    pad = window_size // 2
    mu_a = F.conv2d(a, win, padding=pad, groups=3)
    mu_b = F.conv2d(b, win, padding=pad, groups=3)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    va = F.conv2d(a * a, win, padding=pad, groups=3) - mu_a2
    vb = F.conv2d(b * b, win, padding=pad, groups=3) - mu_b2
    cov = F.conv2d(a * b, win, padding=pad, groups=3) - mu_ab
    c1, c2 = 0.01**2, 0.03**2
    smap = ((2 * mu_ab + c1) * (2 * cov + c2)) / ((mu_a2 + mu_b2 + c1) * (va + vb + c2))
    return smap.mean()


def export_ply(params, path: Path, crop_aabb=None):
    """Write a standard 3DGS .ply (means, sh dc+rest, opacity, scale, rot)."""
    import numpy as np
    import torch
    from plyfile import PlyData, PlyElement

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        means = params["means"].cpu().numpy()
        sh0 = params["sh0"].cpu().numpy().reshape(len(means), -1)
        shN = params["shN"].cpu().numpy().reshape(len(means), -1)
        opac = params["opacities"].cpu().numpy().reshape(-1, 1)
        scales = params["scales"].cpu().numpy()
        quats = params["quats"].cpu().numpy()
    if crop_aabb is not None:
        lo, hi = crop_aabb
        keep = np.all((means >= lo) & (means <= hi), axis=1)
        means, sh0, shN = means[keep], sh0[keep], shN[keep]
        opac, scales, quats = opac[keep], scales[keep], quats[keep]
        print(
            f"  crop-to-init: kept {keep.sum():,} / {len(keep):,} gaussians", flush=True
        )
    fields = ["x", "y", "z", "nx", "ny", "nz"]
    fields += [f"f_dc_{i}" for i in range(sh0.shape[1])]
    fields += [f"f_rest_{i}" for i in range(shN.shape[1])]
    fields += ["opacity"]
    fields += [f"scale_{i}" for i in range(scales.shape[1])]
    fields += [f"rot_{i}" for i in range(quats.shape[1])]
    data = np.concatenate(
        [means, np.zeros_like(means), sh0, shN, opac, scales, quats], axis=1
    )
    arr = np.empty(len(means), dtype=[(f, "f4") for f in fields])
    for i, f in enumerate(fields):
        arr[f] = data[:, i]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


# ── Training ─────────────────────────────────────────────────────────


def train_splat(
    frames,
    pts,
    rgb,
    out_path: Path,
    iters: int,
    sh_degree: int,
    ssim_lambda: float,
    max_init: int,
    crop_to_init: bool,
    crop_margin: float,
    densify: bool = True,
    cap_max: int = 3_000_000,
    init_opacity: float = 0.5,
    appearance: bool = True,
    opacity_reg: float = 1e-4,  # holds opacity ~0.13, prevents footprint-bloat OOM
    scale_reg: float = 0.01,
    min_opacity: float = 0.005,
    flat_reg: float = 0.0,
    min_scale: float = 0.001,
    appearance_reg: float = 0.05,
    pose_opt: bool = True,
    pose_opt_lr: float = 1e-2,
    pose_warmup: int = 300,
    pose_reg: float = 1e-3,
    patch_size: int = 0,
    densify_strategy: str = "mcmc",
):
    import cv2
    import numpy as np
    import torch
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy

    device = "cuda"
    # Drop uncoloured (black) points: the camera doesn't see every LiDAR return,
    # and seeding black Gaussians there scatters dark noise through the splat.
    coloured = rgb.sum(axis=1) > 0.02
    if coloured.any():
        n_drop = int((~coloured).sum())
        pts, rgb = pts[coloured], rgb[coloured]
        if n_drop:
            print(f"  dropped {n_drop:,} uncoloured init points", flush=True)
    if len(pts) > max_init:
        idx = np.random.choice(len(pts), max_init, replace=False)
        pts, rgb = pts[idx], rgb[idx]
    print(
        f"  init from cloud: {len(pts):,} points; {len(frames)} posed frames",
        flush=True,
    )

    crop_aabb = None
    if crop_to_init:
        lo, hi = pts.min(0), pts.max(0)
        m = (hi - lo) * crop_margin
        crop_aabb = (lo - m, hi + m)

    # Normalise the scene to ~unit scale for training: the means learning rate and
    # the scale clamp below are tuned for a unit-cube scene, so a metric scene
    # tens-of-metres across would mis-scale them.  We train normalised, then
    # un-normalise means/scales on export.
    centre = pts.mean(0)
    norm = float(np.linalg.norm(pts.std(axis=0))) + 1e-6
    pts_n = (pts - centre) / norm
    S = np.eye(4)
    S[:3, :3] *= norm
    S[:3, 3] = centre  # normalised → world
    for f in frames:  # world→cam ∘ (norm→world)
        f["viewmat"] = f["viewmat"] @ S

    # Full adaptive-density 3DGS (this is what closes the gap to 3DMakerpro).
    #
    # Densification uses MCMC (gsplat's MCMCStrategy), NOT the default densifier:
    # the fisheye renderer needs the unscented transform (with_ut), which doesn't
    # produce the 2-D screen-space means gradient the default densifier clones on.
    # MCMC is gradient-free — it teleports low-opacity Gaussians to high-opacity
    # regions and samples new ones from the opacity distribution — so it works on
    # the fisheye path and grows the cloud where photos demand detail (~1M → ~3M).
    # The earlier NaN that killed MCMC came from an *unbounded* exp(scale) in its
    # scale regulariser; we clamp scales every step so exp() stays finite.
    #
    # Opacity is LEARNED (not frozen): the previous freeze hid surfaces behind hard
    # opaque blobs, the opposite of 3DMakerpro's faint alpha-blended surfels
    # (median opacity ~0.18). To stop the ~17.5 dB photometric inconsistency
    # (per-frame auto-exposure / white-balance / specular) from driving opacity to
    # zero instead, each frame gets a learned affine colour transform (gain+bias)
    # that absorbs exposure drift, leaving the photometric gradient free to shape
    # geometry. The MCMC opacity/scale regularisers then pull toward many small,
    # faint, anisotropic Gaussians — the surfel look we're after.
    params = build_gaussians(
        pts_n, rgb, sh_degree, device, init_opacity=init_opacity, anisotropic=True
    )
    optimizers = make_optimizers(params, lr_scale=1.0)

    strategy = None
    strategy_state = None
    if densify and densify_strategy == "default":
        # Original 3DGS Adaptive Density Control: gradient-driven clone (small
        # Gaussians in under-reconstructed regions) / split (large ones) at
        # grad2d 2e-4, with a periodic opacity reset and no fixed cap. Targets
        # exactly the soft, high-residual regions (e.g. the near-field arm) that
        # MCMC's capped relocation can starve. Pair with --opacity-reg 0 (this
        # strategy resets opacity itself, so a persistent penalty fights it).
        from gsplat.strategy import DefaultStrategy

        strategy = DefaultStrategy(
            prune_opa=min_opacity,
            grow_grad2d=0.0002,
            grow_scale3d=0.01,
            refine_start_iter=max(100, iters // 60),
            refine_stop_iter=int(iters * 0.5),
            reset_every=3000,
            refine_every=100,
            verbose=False,
        )
        strategy.check_sanity(params, optimizers)
        strategy_state = strategy.initialize_state()
    elif densify:
        # noise_lr well below the 5e5 default: the seeds are already on-surface
        # from LiDAR, so we want gentle teleport/growth, not heavy position noise
        # that scrambles the geometry (which flat-lined the fit in testing).
        strategy = MCMCStrategy(
            cap_max=cap_max,
            noise_lr=1e4,
            refine_start_iter=max(100, iters // 50),
            refine_stop_iter=int(iters * 0.8),
            refine_every=100,
            min_opacity=min_opacity,
            verbose=False,
        )
        strategy.check_sanity(params, optimizers)
        strategy_state = strategy.initialize_state()

    # Per-frame appearance compensation: pred_adj = pred * gain + bias, with gain→1
    # / bias→0 init and a light pull-back regulariser so it only soaks up exposure,
    # not scene radiance. Dropped on export (canonical, neutral-exposure radiance).
    app_gain = app_bias = app_opt = None
    if appearance:
        nf = len(frames)
        app_gain = torch.nn.Parameter(torch.ones(nf, 3, device=device))
        app_bias = torch.nn.Parameter(torch.zeros(nf, 3, device=device))
        app_opt = torch.optim.Adam([app_gain, app_bias], lr=1e-3, eps=1e-15)

    # Camera-pose optimization: a learnable per-frame SE(3) delta composed into
    # the view matrix (viewmat' = exp(δ)·viewmat).  The LiDAR-odometry poses are
    # globally consistent but locally off (drift + camera↔LiDAR time-sync), so the
    # photometric loss sees frames that disagree on where a surface projects and
    # blurs Gaussians to average them — the 'solid but blurry' look.  Letting δ
    # absorb that per-frame error frees the geometry to sharpen.  gsplat 1.5
    # backprops through viewmats to δ.  δ affects training views only; the
    # exported world-frame Gaussians need no change.  Warmup lets the Gaussians
    # settle before the poses start moving; a light L2 keeps δ from running away.
    pose_delta = pose_opt_optimizer = None
    if pose_opt:
        pose_delta = torch.nn.Parameter(torch.zeros(len(frames), 6, device=device))
        pose_opt_optimizer = torch.optim.Adam([pose_delta], lr=pose_opt_lr, eps=1e-15)

    means_lr = optimizers["means"].param_groups[0]["lr"]
    scale_clamp = float(np.log(0.1))  # ≤0.1 unit (~0.3-0.5 m world): keeps
    # gaussians surface-sized — fewer tiles
    # each (less VRAM) and less radial blur.
    # Floor the smallest axis in WORLD metres (min_scale / norm in normalised
    # units). The old fixed floor of 3e-3 normalised ≈ 1 cm world silently pinned
    # 95% of gaussians' thin axis — no reg could flatten them, and flat-reg could
    # only "flatten" by inflating s_max (the 18.8→14.3 pancake failure). The
    # needle crash the floor guarded against (aniso ~300:1 blew up gsplat's tile
    # intersection) is handled by the per-gaussian anisotropy cap below instead.
    scale_floor = float(np.log(min_scale / norm))
    # Cap s_max/s_min. 100:1 was too loose: the optimiser drives the median
    # ratio to the ceiling (a field of maximal needles, aniso~100), and at
    # downscale-1 / 6M those needles project to enormous footprints that blow
    # gsplat's packed tile-intersection buffer — OOMing even a 512 px patch
    # mid-run. The flat-surfel target is 3DMakerpro's ~0.11 min/max ratio (≈9:1),
    # so
    # 20:1 leaves generous headroom for genuinely thin structure while keeping
    # per-gaussian footprints (and VRAM) bounded for the whole run.
    max_aniso_log = float(np.log(20.0))  # cap s_max/s_min at 20:1
    sh_increase_every = max(1, round(1000 * iters / 30_000))

    # Undistorted frames render as pinhole; legacy fisheye frames carry "radial".
    pinhole = "radial" not in frames[0]
    print(
        f"  camera model: {'pinhole (undistorted)' if pinhole else 'fisheye (UT)'}",
        flush=True,
    )

    def render(p, f, sh_deg, viewmat, y0=0, x0=0, ch=None, cw=None):
        # Render a crop [y0:y0+ch, x0:x0+cw] of the frame (default = full frame).
        # Rendering only the crop is what makes VRAM independent of the source
        # resolution: shift the principal point into the crop and shrink the
        # raster to the crop size (gaussians outside are culled).
        H0, W0 = f["image"].shape[:2]
        ch = H0 if ch is None else ch
        cw = W0 if cw is None else cw
        K = f["K_t"]
        if x0 or y0:
            K = K.clone()
            K[0, 2] -= x0
            K[1, 2] -= y0
        cam = (
            {"camera_model": "pinhole"}
            if pinhole
            else {
                "camera_model": "fisheye",
                "radial_coeffs": f["radial_t"],
                "with_ut": True,
            }
        )
        return rasterization(
            means=p["means"],
            quats=p["quats"],
            scales=torch.exp(p["scales"]),
            opacities=torch.sigmoid(p["opacities"]),
            colors=torch.cat([p["sh0"], p["shN"]], dim=1),
            viewmats=viewmat[None],
            Ks=K[None],
            width=cw,
            height=ch,
            sh_degree=sh_deg,
            packed=True,
            **cam,
        )  # packed = sparse intersections → far less VRAM

    def frame_viewmat(fi, f, step):
        """View matrix for frame fi, with the learnable pose delta applied once
        warmup has passed (identity/original before that)."""
        if pose_opt and step >= pose_warmup:
            assert pose_delta is not None
            return _se3_exp(pose_delta[fi]) @ f["viewmat_t"]
        return f["viewmat_t"]

    # Images stay uint8 on the host; only the sampled frame is uploaded and
    # converted each step (~10 MB — negligible next to the render). Pre-pinning
    # float32 copies of every frame doubled a tens-of-GB cache and OOM-killed
    # the host at downscale 2.
    def frame_gt(f, y0=0, x0=0, ch=None, cw=None):
        img = f["image"]
        if ch is not None:
            img = np.ascontiguousarray(img[y0 : y0 + ch, x0 : x0 + cw])
        return torch.from_numpy(img).to(device).float().div_(255.0)

    # Effective render-patch cap. Starts at the requested --patch-size but
    # auto-shrinks if a training step hits a CUDA OOM (huge frames + a full 6M
    # cap + needle-anisotropy can blow the packed rasteriser's intersection
    # buffer even with patching + periodic empty_cache). Shrinking in-place keeps
    # a long run alive instead of throwing away thousands of steps.
    MIN_PATCH = 512
    eff_patch = [patch_size]

    def pick_crop(H, W, centre=False):
        """A (y0, x0, ch, cw) render window. When the effective patch cap is set
        and the frame exceeds it, return a random (or centre, for eval) patch;
        otherwise the full frame — so smaller frames are unchanged and patching
        is opt-in."""
        ps = eff_patch[0]
        if ps and (W > ps or H > ps):
            ch = min(ps, H)
            cw = min(ps, W)
            if centre:
                return (H - ch) // 2, (W - cw) // 2, ch, cw
            return (
                int(rng.integers(0, H - ch + 1)),
                int(rng.integers(0, W - cw + 1)),
                ch,
                cw,
            )
        return 0, 0, H, W

    for f in frames:
        f["K_t"] = torch.tensor(f["K"], dtype=torch.float32, device=device)
        f["viewmat_t"] = torch.tensor(f["viewmat"], dtype=torch.float32, device=device)
        if not pinhole:  # gsplat fisheye radial coeffs [C,4]
            f["radial_t"] = torch.tensor(
                f["radial"], dtype=torch.float32, device=device
            )[None]

    rng = np.random.default_rng(0)
    order = rng.permutation(len(frames))
    cur = 0
    def run_step(step, fi, f):
        """One optimiser step for frame ``fi``. Returns the loss tensor, or None
        when the step was skipped (non-finite loss). May raise a CUDA out-of-
        memory RuntimeError, which the caller recovers from by shrinking the
        render patch instead of aborting the whole run."""
        H, W = f["image"].shape[:2]
        y0, x0, ch, cw = pick_crop(H, W)
        sh_deg = min(sh_degree, step // sh_increase_every)
        use_pose = pose_opt and step >= pose_warmup
        viewmat = frame_viewmat(fi, f, step)
        renders, alphas, info = render(params, f, sh_deg, viewmat, y0, x0, ch, cw)
        if strategy is not None:
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        pred = renders[0]
        if appearance:  # soak up this frame's exposure/WB
            assert app_gain is not None and app_bias is not None
            pred = pred * app_gain[fi] + app_bias[fi]
        pred = pred.clamp(0, 1)
        gt = frame_gt(f, y0, x0, ch, cw)
        if f.get("mask_png") is not None:
            # Dynamic-object mask: inside the masked region the GT is replaced
            # by the (detached) render, so both L1 and SSIM gradients vanish
            # there — the moved object teaches nothing, the rest of the frame
            # trains normally. Crop the mask to the same window as the render.
            m = cv2.imdecode(f["mask_png"], cv2.IMREAD_GRAYSCALE)[
                y0 : y0 + ch, x0 : x0 + cw
            ]
            m_t = (
                torch.from_numpy(np.ascontiguousarray(m))
                .to(device)
                .float()
                .div_(255.0)
                .unsqueeze(-1)
            )
            gt = gt * m_t + pred.detach() * (1.0 - m_t)
        l1 = (pred - gt).abs().mean()
        loss = (1 - ssim_lambda) * l1 + ssim_lambda * (1 - windowed_ssim(pred, gt))
        # MCMC regularisers: push toward many small, faint Gaussians. exp(scale) is
        # bounded by the per-step clamp below, so it can't overflow to NaN.
        loss = loss + opacity_reg * torch.sigmoid(params["opacities"]).abs().mean()
        loss = (
            loss
            + scale_reg
            * torch.exp(params["scales"].clamp(max=scale_clamp)).abs().mean()
        )
        if flat_reg:
            # Push gaussians toward flat surface discs (3DMakerpro min/max axis
            # ratio ≈ 0.11 vs our spherical ~0.75): penalise the scale-invariant
            # ratio exp(s_min − s_max) ∈ (0,1], 1 = sphere, →0 = pancake.
            # s_max is DETACHED: without it the optimiser flattens by inflating
            # the disc instead of thinning it (11 cm pancakes, PSNR 18.8→14.3).
            s = params["scales"].clamp(max=scale_clamp)
            loss = (
                loss
                + flat_reg
                * torch.exp(
                    s.min(dim=1).values - s.max(dim=1).values.detach()
                ).mean()
            )
        if appearance:  # keep the affine near identity
            assert app_gain is not None and app_bias is not None
            loss = (
                loss
                + appearance_reg * ((app_gain[fi] - 1) ** 2 + app_bias[fi] ** 2).mean()
            )
        if use_pose:  # light L2 so the pose delta can't run away
            assert pose_delta is not None
            loss = loss + pose_reg * pose_delta[fi].pow(2).sum()
        if not torch.isfinite(loss):  # skip rare bad frames, don't poison params
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            if app_opt is not None:
                app_opt.zero_grad(set_to_none=True)
            if pose_opt_optimizer is not None:
                pose_opt_optimizer.zero_grad(set_to_none=True)
            print(f"  step {step}: non-finite loss, skipped", flush=True)
            return None
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if app_opt is not None:
            app_opt.step()
            app_opt.zero_grad(set_to_none=True)
        if use_pose:
            assert pose_delta is not None and pose_opt_optimizer is not None
            # decay the pose LR toward 0.1× over the run so late steps only
            # micro-adjust; clip keeps a single bad frame from lurching a camera.
            frac = (step - pose_warmup) / max(1, iters - pose_warmup)
            for g in pose_opt_optimizer.param_groups:
                g["lr"] = pose_opt_lr * (0.1**frac)
            torch.nn.utils.clip_grad_norm_([pose_delta], 1.0)
            pose_opt_optimizer.step()
            pose_opt_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():  # bound scales: no overflow, no needles
            params["scales"].clamp_(min=scale_floor, max=scale_clamp)
            smax = params["scales"].max(dim=1, keepdim=True).values
            params["scales"].clamp_(min=smax - max_aniso_log)
        if strategy is not None:  # grow / relocate Gaussians
            if densify_strategy == "default":
                strategy.step_post_backward(
                    params, optimizers, strategy_state, step, info, packed=True
                )
            else:
                strategy.step_post_backward(
                    params, optimizers, strategy_state, step, info, lr=means_lr
                )
        return loss

    def zero_all_grads():
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        if app_opt is not None:
            app_opt.zero_grad(set_to_none=True)
        if pose_opt_optimizer is not None:
            pose_opt_optimizer.zero_grad(set_to_none=True)

    for step in range(iters):
        if cur >= len(order):
            order = rng.permutation(len(frames))
            cur = 0
        fi = int(order[cur])
        f = frames[fi]
        cur += 1
        try:
            loss = run_step(step, fi, f)
        except RuntimeError as exc:
            # Recover from a mid-training CUDA OOM rather than throwing away the
            # thousands of steps already trained: drop the half-built step, free
            # the cache, and permanently shrink the render patch so later steps
            # fit. Only give up once we're already at the smallest patch.
            if "out of memory" not in str(exc).lower():
                raise
            # Drop the traceback BEFORE freeing: exc.__traceback__ pins
            # run_step's frame, which pins every tensor from the step that blew
            # up (render output + the half-built autograd graph). empty_cache()
            # reclaims nothing while those are alive, so the retry starts on a
            # still-full heap and re-OOMs immediately. Clear the frames first.
            exc = exc.with_traceback(None)
            del exc
            zero_all_grads()
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            H, W = f["image"].shape[:2]
            base = eff_patch[0] or max(H, W)
            if base <= MIN_PATCH:
                raise  # already at the floor and still OOM — nothing left to trade
            eff_patch[0] = max(MIN_PATCH, base // 2)
            print(
                f"  step {step}: CUDA out of memory — shrinking render patch "
                f"{base}px → {eff_patch[0]}px and continuing",
                flush=True,
            )
            continue
        if loss is None:  # non-finite step, already zeroed
            continue
        # Periodic defrag: packed rasterisation allocates variable-size
        # intersection buffers each step, so on a full cap (6M) run the CUDA heap
        # fragments over thousands of steps until a large alloc fails mid-training
        # even with headroom (observed: OOM ~step 10k on a 24 GB card). Returning
        # cached-but-free blocks to the driver here lets the next big alloc find a
        # contiguous span. Off the fast path (every 500 steps → negligible cost).
        if step and step % 500 == 0:
            torch.cuda.empty_cache()
        if step % 100 == 0:
            with torch.no_grad():
                op_med = float(torch.sigmoid(params["opacities"]).median())
                sw = torch.exp(params["scales"])
                aniso = float((sw.amax(1) / sw.amin(1).clamp_min(1e-9)).median())
                # World-metre footprint of the largest axis: p50/p99 tell us
                # whether the packed intersection buffer is creeping because
                # Gaussians are physically inflating (scale growth) vs. just
                # covering more of each frame as the splat fills in.
                smax_m = (sw.amax(1) * norm)
                fp50 = float(smax_m.median())
                fp99 = float(torch.quantile(smax_m[:: max(1, smax_m.numel() // 100000)], 0.99))
            mem_dbg = ""
            if torch.cuda.is_available():
                gib = 1024**3
                mem_dbg = (
                    f"  mem alloc {torch.cuda.memory_allocated() / gib:.1f}"
                    f"/resv {torch.cuda.memory_reserved() / gib:.1f}"
                    f"/peak {torch.cuda.max_memory_allocated() / gib:.1f}G"
                    f"  fp50/99 {fp50 * 100:.1f}/{fp99 * 100:.1f}cm"
                )
            pose_dbg = ""
            if pose_opt and pose_delta is not None:
                with torch.no_grad():
                    dn = pose_delta.detach()
                    rot_deg = float(dn[:, :3].norm(dim=1).mean()) * 180.0 / np.pi
                    trans = float(dn[:, 3:].norm(dim=1).mean())
                pose_dbg = f"  poseδ~{rot_deg:.2f}°/{trans:.3f}u"
            print(
                f"  step {step:6d}  loss {loss.item():.4f}  "
                f"gaussians {params['means'].shape[0]:,}  "
                f"opacity~{op_med:.2f}  aniso~{aniso:.1f}{pose_dbg}{mem_dbg}",
                flush=True,
            )
            print(f"PROGRESS {step + 1}/{iters}", flush=True)
            pct = 10 + int(85 * step / max(iters, 1))
            progress(min(pct, 95), f"Training splat {step}/{iters}…")
    print(f"PROGRESS {iters}/{iters}", flush=True)

    # Train-view PSNR readout (cheap sanity / quality gauge). Uses the per-frame
    # appearance transform so PSNR reflects fit, not exposure mismatch.
    with torch.no_grad():
        tot = n = 0
        for fi in range(0, len(frames), max(1, len(frames) // 20)):
            f = frames[fi]
            H, W = f["image"].shape[:2]
            # Same crop policy as training (centre patch) so the eval render can't
            # OOM at full resolution after a memory-frugal patched run.
            y0, x0, ch, cw = pick_crop(H, W, centre=True)
            vm = frame_viewmat(fi, f, iters)  # with the learned pose delta
            out, _, _ = render(params, f, sh_degree, vm, y0, x0, ch, cw)
            pred = out[0]
            if appearance:
                assert app_gain is not None and app_bias is not None
                pred = pred * app_gain[fi] + app_bias[fi]
            mse = (pred.clamp(0, 1) - frame_gt(f, y0, x0, ch, cw)).pow(2).mean()
            tot += float(10 * torch.log10(1.0 / mse))
            n += 1
        print(f"  train-view PSNR (mean over {n} frames): {tot / n:.2f} dB", flush=True)

    # Un-normalise back to world: means scale+shift, log-scales add log(norm).
    with torch.no_grad():
        params["means"].mul_(norm).add_(
            torch.tensor(centre, dtype=torch.float32, device=device)
        )
        params["scales"].add_(float(np.log(norm)))

    export_ply(params, out_path, crop_aabb)
    print(f"Saved: {out_path} ({params['means'].shape[0]:,} gaussians)", flush=True)




# ── Main ─────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pointcloud", default=None)
    p.add_argument("--iterations", type=int, default=7000)
    p.add_argument("--camera", default="front")
    p.add_argument("--image-rot", default="ccw", choices=["cw", "ccw", "180", "none"])
    p.add_argument("--downscale", type=int, default=4)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--ssim-lambda", type=float, default=0.2)
    p.add_argument("--max-init-points", type=int, default=1_000_000)
    p.add_argument(
        "--no-crop-to-init", dest="crop_to_init", action="store_false", default=True
    )
    p.add_argument("--crop-margin", type=float, default=0.05)
    p.add_argument(
        "--no-densify",
        dest="densify",
        action="store_false",
        default=True,
        help="disable MCMC densification (refine-only — the old behaviour)",
    )
    p.add_argument(
        "--densify-strategy",
        choices=["mcmc", "default"],
        default="mcmc",
        help="densification scheme: 'mcmc' = fixed-cap MCMC relocation (our "
        "default); 'default' = original 3DGS Adaptive Density Control (gradient "
        "clone/split at grad2d 2e-4, opacity reset every 3000, no cap). Use "
        "--opacity-reg 0 with 'default'.",
    )
    p.add_argument(
        "--cap-max",
        type=int,
        default=3_000_000,
        help="max gaussians MCMC may grow to (3DMakerpro ref ≈ 3.4M)",
    )
    p.add_argument(
        "--init-opacity",
        type=float,
        default=0.5,
        help="initial (learned) gaussian opacity; reg pulls it lower",
    )
    p.add_argument(
        "--opacity-reg",
        type=float,
        default=1e-4,
        help="weight of the mean-opacity penalty. Too high (0.004) pins most "
        "gaussians at the MCMC min_opacity floor (transparent mush); too low "
        "(2e-5) lets opacity collapse to ~0.03, and the reconstruction loss then "
        "inflates the faint gaussians to the scale ceiling (~30 cm), blowing the "
        "packed rasteriser's intersection buffer → OOM at downscale-1/6M around "
        "step 8500. 1e-4 holds median opacity ~0.13 and footprints ~1 cm: 6M/ds1/"
        "30k trains full-frame in ~9 GB and scored 23.03 dB (dense deskew seed).",
    )
    p.add_argument(
        "--scale-reg",
        type=float,
        default=0.01,
        help="weight of the mean-scale penalty (pulls gaussians small)",
    )
    p.add_argument(
        "--min-opacity",
        type=float,
        default=0.03,
        help="MCMC relocation threshold: gaussians below this opacity get "
        "teleported to useful regions instead of lingering as fog "
        "(0.03 = campaign champion; the old floor was 0.005)",
    )
    p.add_argument(
        "--flat-reg",
        type=float,
        default=0.1,
        help="weight pushing gaussians toward flat surface discs "
        "(penalises min/max axis ratio; 0 = off; 0.1 = campaign champion)",
    )
    p.add_argument(
        "--memmap-frames",
        action="store_true",
        help="stream undistorted frames to an on-disk memmap instead of holding "
        "them all in host RAM — keeps RSS flat regardless of resolution × frame "
        "count (essential for downscale-1 / high undistort-scale, where the RAM "
        "cache can exceed host memory).",
    )
    p.add_argument(
        "--frame-cache-dir",
        default=None,
        help="directory for the --memmap-frames cache (default: auto-pick a real "
        "disk with the most free space; honours $LIDARSTUDIO_FRAME_CACHE).",
    )
    p.add_argument(
        "--patch-size",
        type=int,
        default=0,
        help="train (and eval) on random NxN-pixel crops instead of whole "
        "frames; 0 = full frame. Decouples GPU memory from image resolution — a "
        "1600 patch of a 4800×6400 frame cuts render VRAM ~9× at equivalent "
        "quality, so downscale-1 / high-cap runs fit on smaller cards.",
    )
    p.add_argument(
        "--min-scale",
        type=float,
        default=0.001,
        help="floor for a gaussian's smallest axis, in WORLD metres "
        "(anisotropy is separately capped at 20:1)",
    )
    p.add_argument(
        "--no-appearance",
        dest="appearance",
        action="store_false",
        default=True,
        help="disable per-frame exposure/white-balance compensation",
    )
    p.add_argument(
        "--no-pose-opt",
        dest="pose_opt",
        action="store_false",
        default=True,
        help="disable per-frame camera-pose optimisation (on by default)",
    )
    p.add_argument(
        "--pose-opt-lr",
        type=float,
        default=1e-2,
        help="learning rate for the per-frame SE(3) pose delta",
    )
    p.add_argument(
        "--pose-warmup",
        type=int,
        default=300,
        help="steps to let Gaussians settle before pose deltas update",
    )
    p.add_argument(
        "--pose-reg",
        type=float,
        default=1e-3,
        help="L2 weight keeping pose deltas small",
    )
    p.add_argument(
        "--no-undistort",
        dest="undistort",
        action="store_false",
        default=True,
        help="train in raw fisheye (broken: gsplat fisheye ≠ OpenCV KB) "
        "instead of undistorting photos to pinhole",
    )
    p.add_argument(
        "--undistort-fov",
        type=float,
        default=120.0,
        help="horizontal FOV (deg) of the virtual pinhole when undistorting",
    )
    p.add_argument(
        "--cam-time-offset",
        type=float,
        default=-0.025,
        help="camera↔LiDAR clock offset in SECONDS added to each image "
        "timestamp before the continuous-time pose query; sweep to "
        "calibrate a constant sensor time skew. Default -0.025 is this "
        "rig's calibrated skew (sweep peak); use 0 for other hardware",
    )
    p.add_argument(
        "--undistort-scale",
        type=float,
        default=1.0,
        help="undistorted canvas size as a multiple of the source frame; >1 "
        "preserves the fisheye centre's native px/radian that a 1:1 "
        "wide-FOV pinhole undersamples",
    )
    p.add_argument(
        "--drop-blurry",
        type=float,
        default=0.0,
        help="drop this fraction (0-1) of the blurriest frames "
        "(Laplacian-variance sharpness) before training",
    )
    p.add_argument(
        "--sfm-poses",
        default=None,
        help="npz of externally-computed world→cam viewmats keyed by "
        "bag_ts_ns (from align_sfm.py); overrides the odometry pose of "
        "every covered frame and drops uncovered frames",
    )
    p.add_argument(
        "--mask-box",
        default=None,
        help="x1,y1,z1,x2,y2,z2 world-space box around a dynamic object "
        "(e.g. a robot arm that moved mid-scan); its blue pixels are "
        "excluded from the loss in frames >= --mask-from",
    )
    p.add_argument(
        "--mask-from",
        type=int,
        default=0,
        help="bag frame index from which the --mask-box region is excluded "
        "(frames before it train the object's canonical configuration)",
    )
    p.add_argument(
        "--no-prefer-dense",
        dest="prefer_dense",
        action="store_false",
        default=True,
        help="don't auto-seed from a sibling *_dense.ply if present",
    )
    args = p.parse_args()

    scan = Path(args.scan)
    out = Path(args.output)
    pc = Path(args.pointcloud) if args.pointcloud else None
    out.parent.mkdir(parents=True, exist_ok=True)

    if pc is None or not pc.exists():
        print(
            "ERROR: a coloured --pointcloud is required (run Generate Point "
            "Cloud first).",
            flush=True,
        )
        sys.exit(1)

    traj_path = Path(str(pc) + ".traj.npz")

    # GPU training needs the KISS-ICP trajectory sidecar + CUDA. Fail loudly
    # rather than emit a low-value CPU splat when either is missing.
    if not traj_path.exists():
        print(
            f"ERROR: no trajectory sidecar ({traj_path.name}) — GPU training needs "
            "it. Regenerate the point cloud, re-run the edit (edits carry the "
            "trajectory), or pick a generated cloud as the seed.",
            flush=True,
        )
        sys.exit(1)
    ensure_cuda_home()
    try:
        import torch

        if not torch.cuda.is_available():
            print("ERROR: no CUDA GPU available — splat training needs one.", flush=True)
            sys.exit(1)
    except ImportError:
        print("ERROR: torch/gsplat not installed — splat training needs them.", flush=True)
        sys.exit(1)

    # Prefer a denser sibling cloud for the seed (e.g. *_dense.ply): 3DMakerpro's
    # reference splat is ~3.4M gaussians, and a richer seed → far more detail.
    seed_pc = pc
    if args.prefer_dense and pc.suffix == ".ply" and "_dense" not in pc.stem:
        dense = pc.with_name(pc.stem + "_dense.ply")
        if dense.exists():
            seed_pc = dense
            print(f"  seeding from dense cloud: {dense.name}", flush=True)

    progress(5, "Loading coloured cloud and camera poses…")
    pts, rgb = load_coloured_ply(seed_pc)
    mask_box = None
    mask_canon = None
    if args.mask_box:
        v = [float(x) for x in args.mask_box.split(",")]
        mask_box = (
            [min(a, b) for a, b in zip(v[:3], v[3:])],
            [max(a, b) for a, b in zip(v[:3], v[3:])],
        )
        # Blue seed points inside the box = the canonical configuration's own
        # geometry (the seed cloud is expected to be ghost-filtered via
        # --dynamic-box). Their projection is protected from carving in masked
        # frames; the floor/walls inside the box stay supervised, so only the
        # object itself is exempted. Dilation in the mask covers its adjacent
        # non-blue parts.
        inb = ((pts >= mask_box[0]) & (pts <= mask_box[1])).all(axis=1)
        blue = (rgb[:, 2] > rgb[:, 0] + 0.12) & (rgb[:, 2] > rgb[:, 1] + 0.08)
        canon = pts[inb & blue]
        if len(canon):
            mask_canon = canon[:: max(1, len(canon) // 20000)]
    frames = build_camera_frames(
        scan,
        traj_path,
        args.camera,
        args.image_rot,
        args.downscale,
        undistort=args.undistort,
        undistort_fov=args.undistort_fov,
        undistort_scale=args.undistort_scale,
        cam_time_offset=args.cam_time_offset,
        sfm_poses=Path(args.sfm_poses) if args.sfm_poses else None,
        drop_blurry=args.drop_blurry,
        mask_box=mask_box,
        mask_from=args.mask_from,
        mask_canon_pts=mask_canon,
        memmap_frames=args.memmap_frames,
        frame_cache_dir=args.frame_cache_dir,
    )
    print(
        f"  {len(frames)} posed frames @ {frames[0]['image'].shape[1]}×"
        f"{frames[0]['image'].shape[0]} (downscale {args.downscale})",
        flush=True,
    )
    progress(10, f"Training {args.iterations}-iter splat on GPU…")
    try:
        train_splat(
            frames,
            pts,
            rgb,
            out,
            args.iterations,
            args.sh_degree,
            args.ssim_lambda,
            args.max_init_points,
            args.crop_to_init,
            args.crop_margin,
            densify=args.densify,
            cap_max=args.cap_max,
            init_opacity=args.init_opacity,
            opacity_reg=args.opacity_reg,
            scale_reg=args.scale_reg,
            min_opacity=args.min_opacity,
            flat_reg=args.flat_reg,
            min_scale=args.min_scale,
            appearance=args.appearance,
            pose_opt=args.pose_opt,
            pose_opt_lr=args.pose_opt_lr,
            pose_warmup=args.pose_warmup,
            pose_reg=args.pose_reg,
            patch_size=args.patch_size,
            densify_strategy=args.densify_strategy,
        )
    except RuntimeError as exc:
        # Turn a CUDA OOM traceback into one actionable line (the server surfaces
        # ``ERROR:`` lines to the UI); other RuntimeErrors keep their traceback.
        if "out of memory" not in str(exc).lower():
            raise
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        print(
            f"ERROR: GPU ran out of memory at {frames[0]['image'].shape[1]}×"
            f"{frames[0]['image'].shape[0]}, {args.cap_max // 1_000_000}M splats. "
            "Lower the quality — set downscale to 2, reduce max splats (e.g. 3M), "
            "or lower sharpen× — then retry.",
            flush=True,
        )
        sys.exit(1)
    progress(100, "Complete!")


if __name__ == "__main__":
    main()
