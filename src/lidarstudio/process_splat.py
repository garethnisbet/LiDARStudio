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

GPU: training uses gsplat's CUDA rasterizer (needs an NVIDIA GPU + nvcc).  If
gsplat/CUDA or the trajectory is unavailable, falls back to a CPU "bootstrap"
that turns each point into one isotropic Gaussian (viewable, not optimised).

Usage (called automatically by lidar_server.py):
    python process_splat.py
        --scan       /path/to/20260527195949
        --output     /project/splats/splat_20260527195949.ply
        --pointcloud /project/pointclouds/pointcloud_20260527195949.ply
        [--iterations 7000] [--downscale 4] [--image-rot ccw] [--surfel]

Progress protocol:
    Print "PROGRESS:<percent>:<message>" for the UI progress bar; other lines
    appear in the log view.  Training also prints "PROGRESS <step>/<iters>".
"""

import argparse
import os
import sys
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


def build_camera_frames(
    scan: Path,
    traj_path: Path,
    camera: str,
    image_rot: str,
    downscale: int,
    undistort: bool = True,
    undistort_fov: float = 120.0,
    cam_time_offset: float = 0.0,
):
    """
    Return a list of {image HxWx3 float[0,1], K 3x3, viewmat 4x4 (world→cam)}.

    For each camera frame we query the LiDAR pose at its (offset-corrected) exact
    timestamp by SE(3) interpolation, P, and form the world→camera matrix
    T_extrinsic @ inv(P)  — points go world → lidar (inv P) → camera (T).  Images
    are rotated to portrait to match K.  ``cam_time_offset`` (seconds) shifts the
    image clock onto the LiDAR clock before the query (camera_ts + offset), to
    absorb a constant sensor time skew; sweep it if frames look consistently
    mis-registered.

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
    frames = []
    for ts, img in ppc.read_images(img_bag, camera, rot=image_rot):
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
                f = (W / 2.0) / np.tan(np.radians(undistort_fov) / 2.0)
                Kp = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
                m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                    K, Dk, np.eye(3), Kp, (W, H), cv2.CV_16SC2
                )
                maps = (m1, m2, Kp)
                print(
                    f"  undistort: fisheye → pinhole {undistort_fov:.0f}° "
                    f"(f={f:.1f}px, {W}×{H})",
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
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        tq = int(ts) + offset_ns
        P, a, gap = _interp_pose(traj_ts, poses, tq)
        viewmat = T @ np.linalg.inv(P)  # world → camera
        # Diagnostics: how far the old nearest-neighbour snap would have been, and
        # the bracket size we interpolated across.
        diag["n"] += 1
        if gap == 0:
            diag["clamped"] += 1
        else:
            diag["gap_ns"].append(gap)
            diag["snap_err_ns"].append(min(a, 1.0 - a) * gap)

        frame = {"image": rgb, "K": K.astype(np.float64), "viewmat": viewmat}
        if not undistort:  # fisheye path keeps radial coeffs
            frame["radial"] = radial
        frames.append(frame)
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
    tangent plane. This is the same surface-fitting that ``surfel_splat_from_ply``
    does — factored out so trained splats can *seed* anisotropy from the (accurate)
    LiDAR geometry instead of trying to learn it from inconsistent fisheye photos.
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
    opacity_reg: float = 0.004,
    scale_reg: float = 0.01,
    appearance_reg: float = 0.05,
    pose_opt: bool = True,
    pose_opt_lr: float = 1e-2,
    pose_warmup: int = 300,
    pose_reg: float = 1e-3,
):
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
    if densify:
        # noise_lr well below the 5e5 default: the seeds are already on-surface
        # from LiDAR, so we want gentle teleport/growth, not heavy position noise
        # that scrambles the geometry (which flat-lined the fit in testing).
        strategy = MCMCStrategy(
            cap_max=cap_max,
            noise_lr=1e4,
            refine_start_iter=max(100, iters // 50),
            refine_stop_iter=int(iters * 0.8),
            refine_every=100,
            min_opacity=0.005,
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
    # Floor the smallest axis too: without it the photometric loss collapses one
    # axis → 0, making degenerate "needle" gaussians (aniso blew up to ~300:1 and
    # crashed gsplat's tile intersection). This caps anisotropy at 0.3/3e-3 = 100:1.
    scale_floor = float(np.log(3e-3))
    sh_increase_every = max(1, round(1000 * iters / 30_000))

    # Undistorted frames render as pinhole; legacy fisheye frames carry "radial".
    pinhole = "radial" not in frames[0]
    print(
        f"  camera model: {'pinhole (undistorted)' if pinhole else 'fisheye (UT)'}",
        flush=True,
    )

    def render(p, f, sh_deg, W, H, viewmat):
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
            Ks=f["K_t"][None],
            width=W,
            height=H,
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

    for f in frames:
        f["image_t"] = torch.tensor(f["image"]).pin_memory()
        f["K_t"] = torch.tensor(f["K"], dtype=torch.float32, device=device)
        f["viewmat_t"] = torch.tensor(f["viewmat"], dtype=torch.float32, device=device)
        if not pinhole:  # gsplat fisheye radial coeffs [C,4]
            f["radial_t"] = torch.tensor(
                f["radial"], dtype=torch.float32, device=device
            )[None]

    rng = np.random.default_rng(0)
    order = rng.permutation(len(frames))
    cur = 0
    for step in range(iters):
        if cur >= len(order):
            order = rng.permutation(len(frames))
            cur = 0
        fi = int(order[cur])
        f = frames[fi]
        cur += 1
        H, W = f["image"].shape[:2]
        sh_deg = min(sh_degree, step // sh_increase_every)
        use_pose = pose_opt and step >= pose_warmup
        viewmat = frame_viewmat(fi, f, step)
        renders, alphas, info = render(params, f, sh_deg, W, H, viewmat)
        if strategy is not None:
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        pred = renders[0]
        if appearance:  # soak up this frame's exposure/WB
            assert app_gain is not None and app_bias is not None
            pred = pred * app_gain[fi] + app_bias[fi]
        pred = pred.clamp(0, 1)
        gt = f["image_t"].to(device, non_blocking=True)
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
            continue
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
        if strategy is not None:  # teleport / grow Gaussians (gradient-free)
            strategy.step_post_backward(
                params, optimizers, strategy_state, step, info, lr=means_lr
            )
        if step % 100 == 0:
            with torch.no_grad():
                op_med = float(torch.sigmoid(params["opacities"]).median())
                sw = torch.exp(params["scales"])
                aniso = float((sw.amax(1) / sw.amin(1).clamp_min(1e-9)).median())
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
                f"opacity~{op_med:.2f}  aniso~{aniso:.1f}{pose_dbg}",
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
            vm = frame_viewmat(fi, f, iters)  # with the learned pose delta
            out, _, _ = render(params, f, sh_degree, W, H, vm)
            pred = out[0]
            if appearance:
                assert app_gain is not None and app_bias is not None
                pred = pred * app_gain[fi] + app_bias[fi]
            mse = (pred.clamp(0, 1) - f["image_t"].to(device)).pow(2).mean()
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


# ── CPU fallback: bootstrap one Gaussian per point ───────────────────


def bootstrap_splat_from_ply(input_ply: Path, output_ply: Path, size: float = 0.05):
    """Unoptimised splat: each coloured point → one isotropic Gaussian (no GPU).

    ``size`` is the Gaussian radius (std-dev) in metres for every splat. Smaller
    values give finer splats instead of large blobs (the field stores log-scale)."""
    import numpy as np

    progress(10, f"Bootstrap (no training, size={size} m) from {input_ply.name}…")
    pts, rgb = load_coloured_ply(input_ply)
    n = len(pts)
    dc = rgb_to_sh(rgb).astype(np.float32)
    scale = np.full((n, 3), float(np.log(max(size, 1e-4))), np.float32)
    rot = np.zeros((n, 4), np.float32)
    rot[:, 0] = 1.0
    opac = np.full((n, 1), 2.197, np.float32)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(
            f"property float {p}\n"
            for p in [
                "x",
                "y",
                "z",
                "nx",
                "ny",
                "nz",
                "f_dc_0",
                "f_dc_1",
                "f_dc_2",
                "opacity",
                "scale_0",
                "scale_1",
                "scale_2",
                "rot_0",
                "rot_1",
                "rot_2",
                "rot_3",
            ]
        )
        + "end_header\n"
    )
    data = np.concatenate(
        [pts, np.zeros((n, 3), np.float32), dc, opac, scale, rot], axis=1
    ).astype(np.float32)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ply, "wb") as fh:
        fh.write(header.encode())
        fh.write(data.tobytes())
    print(f"Saved (bootstrap): {output_ply}", flush=True)
    progress(100, "Complete (bootstrap — no GPU training)!")


# ── Surfel splat: one anisotropic surface-aligned disc per point ─────


def surfel_splat_from_ply(
    input_ply: Path,
    output_ply: Path,
    k: int = 12,
    radius_factor: float = 0.6,
    flatten: float = 0.2,
    opacity: float = 0.9,
    sor_std: float = 2.0,
):
    """Turn the coloured cloud into 3DMakerpro-style surfels (no GPU/training).

    Each point becomes a small Gaussian *disc* lying in the local surface: we
    PCA the ``k`` nearest neighbours, orient the disc in the tangent plane and
    squash it along the surface normal.  Disc radius tracks the local point
    spacing so the discs tile the surface without blobbing.  This is what makes
    the splat look photoreal (tiny, anisotropic, surface-aligned) rather than a
    field of fat isotropic balls.

    Two guards keep it from turning into a "snowstorm" of spikes on noisy clouds:
    statistical outlier removal drops floater points, and the disc radius is
    capped so a sparse/isolated point can't spawn one giant edge-on spike.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    progress(10, f"Surfel splat from {input_ply.name}…")
    pts, rgb = load_coloured_ply(input_ply)

    coloured = rgb.sum(1) > 0.02  # drop unseen (black) points
    pts, rgb = pts[coloured], rgb[coloured]
    print(f"  {len(pts):,} coloured points", flush=True)

    progress(30, "Estimating local surface (kNN PCA)…")
    tree = cKDTree(pts)
    d, idx = tree.query(pts, k=k + 1, workers=-1)  # (n,k+1) incl. self

    # Statistical outlier removal: drop points whose mean neighbour distance is
    # far above the global mean — these are the floaters that read as fuzz.
    mdist = d[:, 1:].mean(1)
    keep = mdist < (mdist.mean() + sor_std * mdist.std())
    if not keep.all():
        print(
            f"  outlier removal: dropped {int((~keep).sum()):,}"
            f" / {len(pts):,} floaters",
            flush=True,
        )
        pts, rgb, idx, d = pts[keep], rgb[keep], idx[keep], d[keep]
    n = len(pts)
    # idx still references the pre-filter array, so PCA reads original neighbours.
    nn = np.clip(d[:, 1], 1e-4, None)  # nearest-neighbour dist

    nbr_src = tree.data  # full point set for neighbour lookup
    nbr = nbr_src[idx[:, 1:]]  # (n,k,3)
    cen = nbr - nbr.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", cen, cen) / k  # (n,3,3) covariance
    evals, evecs = np.linalg.eigh(cov)  # ascending; evecs cols
    # Smallest-variance axis = normal; the other two span the tangent plane.
    R = np.stack(
        [evecs[:, :, 2], evecs[:, :, 1], evecs[:, :, 0]], axis=2
    )  # cols [t1,t2,n]
    det = np.linalg.det(R)
    R[det < 0, :, 2] *= -1.0  # keep right-handed (proper rotation)

    quat_xyzw = Rotation.from_matrix(R).as_quat()  # [x,y,z,w]
    quats = np.column_stack([quat_xyzw[:, 3], quat_xyzw[:, :3]]).astype(
        np.float32
    )  # [w,x,y,z]

    # In-plane radius ~ local spacing, but capped at 3× the median so sparse
    # points don't become giant spikes (the dominant snowstorm cause).
    r_in = nn * radius_factor
    cap = float(np.median(r_in)) * 3.0
    r_in = np.minimum(r_in, cap)
    s_in = np.log(np.clip(r_in, 1e-4, None))
    s_nrm = np.log(np.clip(r_in * flatten, 1e-5, None))  # squash along normal
    scale = np.stack([s_in, s_in, s_nrm], axis=1).astype(np.float32)

    dc = rgb_to_sh(rgb).astype(np.float32)
    opac = np.full((n, 1), float(np.log(opacity / (1 - opacity))), np.float32)
    print(
        f"  {n:,} surfels; disc radius median {np.median(r_in) * 1000:.1f} mm "
        f"(cap {cap * 1000:.1f} mm), flatten {flatten:g}, aniso {1 / flatten:.0f}:1",
        flush=True,
    )

    progress(80, f"Writing {n:,} surfels…")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(
            f"property float {p}\n"
            for p in [
                "x",
                "y",
                "z",
                "nx",
                "ny",
                "nz",
                "f_dc_0",
                "f_dc_1",
                "f_dc_2",
                "opacity",
                "scale_0",
                "scale_1",
                "scale_2",
                "rot_0",
                "rot_1",
                "rot_2",
                "rot_3",
            ]
        )
        + "end_header\n"
    )
    normals = R[:, :, 2].astype(np.float32)  # write the real surface normal
    data = np.concatenate([pts, normals, dc, opac, scale, quats], axis=1).astype(
        np.float32
    )
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ply, "wb") as fh:
        fh.write(header.encode())
        fh.write(data.tobytes())
    print(f"Saved (surfel): {output_ply}", flush=True)
    progress(100, "Complete (surfel — no GPU training)!")


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
        default=0.0,
        help="camera↔LiDAR clock offset in SECONDS added to each image "
        "timestamp before the continuous-time pose query; sweep to "
        "calibrate a constant sensor time skew",
    )
    p.add_argument(
        "--no-prefer-dense",
        dest="prefer_dense",
        action="store_false",
        default=True,
        help="don't auto-seed from a sibling *_dense.ply if present",
    )
    p.add_argument(
        "--splat-size",
        type=float,
        default=0.05,
        help="bootstrap Gaussian radius (m) — smaller = finer splats, less blobby",
    )
    p.add_argument(
        "--bootstrap",
        action="store_true",
        help="skip GPU training; one isotropic Gaussian per point (CPU)",
    )
    p.add_argument(
        "--surfel",
        action="store_true",
        help="skip GPU training; one anisotropic surface-aligned disc "
        "per point (CPU). Best match to 3DMakerpro on a dense cloud.",
    )
    p.add_argument(
        "--surfel-flatten",
        type=float,
        default=0.2,
        help="surfel thickness as a fraction of disc radius (smaller = flatter)",
    )
    p.add_argument(
        "--surfel-sor",
        type=float,
        default=2.0,
        help="outlier removal strength (σ); lower = drop more floaters",
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

    # Surfel mode short-circuits everything: a fast CPU anisotropic-disc splat
    # straight from the coloured cloud (no poses/GPU needed).
    if args.surfel:
        surfel_splat_from_ply(
            pc, out, flatten=args.surfel_flatten, sor_std=args.surfel_sor
        )
        return

    # Decide path: real GPU training when we have a trajectory + CUDA, else bootstrap.
    can_train = traj_path.exists() and not args.bootstrap
    if can_train:
        ensure_cuda_home()
        try:
            import torch

            if not torch.cuda.is_available():
                print(
                    "  No CUDA GPU available — falling back to CPU bootstrap.",
                    flush=True,
                )
                can_train = False
        except ImportError:
            print(
                "  torch/gsplat not installed — falling back to CPU bootstrap.",
                flush=True,
            )
            can_train = False

    if not can_train:
        if not traj_path.exists():
            print(
                f"  No trajectory sidecar ({traj_path.name}) — bootstrap only. "
                "Regenerate the point cloud to enable trained splats.",
                flush=True,
            )
        bootstrap_splat_from_ply(pc, out, size=args.splat_size)
        return

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
    frames = build_camera_frames(
        scan,
        traj_path,
        args.camera,
        args.image_rot,
        args.downscale,
        undistort=args.undistort,
        undistort_fov=args.undistort_fov,
        cam_time_offset=args.cam_time_offset,
    )
    print(
        f"  {len(frames)} posed frames @ {frames[0]['image'].shape[1]}×"
        f"{frames[0]['image'].shape[0]} (downscale {args.downscale})",
        flush=True,
    )
    progress(10, f"Training {args.iterations}-iter splat on GPU…")
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
        appearance=args.appearance,
        pose_opt=args.pose_opt,
        pose_opt_lr=args.pose_opt_lr,
        pose_warmup=args.pose_warmup,
        pose_reg=args.pose_reg,
    )
    progress(100, "Complete!")


if __name__ == "__main__":
    main()
