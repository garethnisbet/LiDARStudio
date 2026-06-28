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
from typing import Optional

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
    for c in (Path.home() / ".local/micromamba/envs/cuda121",
              Path("/usr/local/cuda")):
        if (c / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = str(c)
            os.environ["PATH"] = f"{c/'bin'}{os.pathsep}{os.environ.get('PATH','')}"
            return


# ── PLY I/O ──────────────────────────────────────────────────────────

def load_coloured_ply(path: Path):
    """Return (xyz Nx3 float32, rgb Nx3 float[0,1]) from a coloured PLY."""
    import numpy as np
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, np.float32)
        rgb = (np.asarray(pcd.colors, np.float32)
               if pcd.has_colors() else np.full((len(pts), 3), 0.5, np.float32))
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

def build_camera_frames(scan: Path, traj_path: Path, camera: str,
                        image_rot: str, downscale: int):
    """
    Return a list of {image HxWx3 float[0,1], K 3x3, viewmat 4x4 (world→cam)}.

    For each camera frame we take the LiDAR pose nearest its timestamp, P, and
    form the world→camera matrix  T_extrinsic @ inv(P)  — points go world →
    lidar (inv P) → camera (T).  Images are rotated to portrait to match K,
    undistorted to a pinhole model, and downscaled for tractable training.
    """
    import numpy as np
    import cv2

    calib = ppc.load_calibration(scan / "calibration", camera)
    K0 = calib["K"].astype(np.float64)
    D = calib["D"].astype(np.float64)
    T = calib["T"].astype(np.float64)          # lidar → camera (4×4)

    traj = np.load(traj_path)
    traj_ts = traj["ts"].astype(np.int64)
    poses = traj["poses"].astype(np.float64)   # (N,4,4) sensor→world (levelled)
    order = np.argsort(traj_ts)
    traj_ts, poses = traj_ts[order], poses[order]

    # Locate the image bag in the scan folder.
    img_bags = sorted(scan.glob("IMAGE_*.bag")) or sorted(scan.glob("*.bag"))
    img_bag = next((b for b in img_bags if "IMAGE" in b.name.upper()), None)
    if img_bag is None:
        raise FileNotFoundError(f"no IMAGE_*.bag in {scan}")

    # The lens is a ~180° fisheye, so we DON'T undistort to a pinhole (a pinhole
    # can't represent it).  Instead gsplat rasterizes with camera_model="fisheye"
    # and these k1..k4 radial coeffs directly, matching the colouring projection.
    # Distortion coeffs are angle-based and unaffected by downscaling; only K
    # scales with the image.
    radial = D.reshape(4).astype(np.float64)
    frames = []
    for ts, img in ppc.read_images(img_bag, camera, rot=image_rot):
        h, w = img.shape[:2]
        K = K0.copy()
        if downscale > 1:
            img = cv2.resize(img, (w // downscale, h // downscale),
                             interpolation=cv2.INTER_AREA)
            K = K / downscale
            K[2, 2] = 1.0
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        j = int(np.searchsorted(traj_ts, ts))
        j = min(max(j, 0), len(poses) - 1)
        if 0 < j < len(poses) and abs(traj_ts[j - 1] - ts) < abs(traj_ts[j] - ts):
            j -= 1
        viewmat = T @ np.linalg.inv(poses[j])    # world → camera

        frames.append({"image": rgb, "K": K.astype(np.float64),
                       "viewmat": viewmat, "radial": radial})
    if not frames:
        raise RuntimeError("no camera frames loaded from image bag")
    return frames


# ── Gaussian model + loss (adapted from raven/train_splat.py) ─────────

def rgb_to_sh(rgb):
    return (rgb - 0.5) / 0.28209479177387814


def knn_scale(points, k: int = 4):
    from scipy.spatial import cKDTree
    import numpy as np
    tree = cKDTree(points)
    d, _ = tree.query(points, k=k + 1, workers=-1)
    s = d[:, 1:].mean(axis=1)
    # Cap isolated/sparse points: their nearest neighbour is far, so an unbounded
    # knn scale makes one giant Gaussian that renders as a white "bloom" halo.
    # Clamp to a few× the typical spacing so seeds stay surface-sized.
    hi = float(np.percentile(s, 90)) * 3.0
    return np.clip(s, 1e-4, max(hi, 1e-3))


def build_gaussians(points, rgb, sh_degree, device, init_opacity: float = 0.1):
    import numpy as np
    import torch
    n = len(points)
    means = torch.tensor(points, dtype=torch.float32, device=device)
    scales = torch.tensor(np.log(knn_scale(points)), dtype=torch.float32, device=device)
    scales = scales[:, None].repeat(1, 3)
    quats = torch.zeros(n, 4, device=device); quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((n,), init_opacity, device=device))
    num_sh = (sh_degree + 1) ** 2
    colors = torch.zeros(n, num_sh, 3, device=device)
    colors[:, 0, :] = torch.tensor(rgb_to_sh(rgb), dtype=torch.float32, device=device)
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(colors[:, :1, :]),
        "shN": torch.nn.Parameter(colors[:, 1:, :]),
    }).to(device)


def make_optimizers(params, lr_scale: float, freeze=()):
    import torch
    specs = {"means": 1.6e-4 * lr_scale, "scales": 5e-3, "quats": 1e-3,
             "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    return {name: torch.optim.Adam([params[name]], lr=lr, eps=1e-15)
            for name, lr in specs.items() if name not in freeze}


_SSIM_WINDOW = {}


def windowed_ssim(pred, gt, window_size: int = 11, sigma: float = 1.5):
    import torch
    import torch.nn.functional as F
    key = (window_size, sigma, pred.device)
    win = _SSIM_WINDOW.get(key)
    if win is None:
        coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
        win2d = (g[:, None] * g[None, :])[None, None]
        win = win2d.expand(3, 1, window_size, window_size).contiguous()
        _SSIM_WINDOW[key] = win
    a = pred.permute(2, 0, 1)[None]; b = gt.permute(2, 0, 1)[None]
    pad = window_size // 2
    mu_a = F.conv2d(a, win, padding=pad, groups=3)
    mu_b = F.conv2d(b, win, padding=pad, groups=3)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    va = F.conv2d(a * a, win, padding=pad, groups=3) - mu_a2
    vb = F.conv2d(b * b, win, padding=pad, groups=3) - mu_b2
    cov = F.conv2d(a * b, win, padding=pad, groups=3) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
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
        print(f"  crop-to-init: kept {keep.sum():,} / {len(keep):,} gaussians", flush=True)
    fields = ["x", "y", "z", "nx", "ny", "nz"]
    fields += [f"f_dc_{i}" for i in range(sh0.shape[1])]
    fields += [f"f_rest_{i}" for i in range(shN.shape[1])]
    fields += ["opacity"]
    fields += [f"scale_{i}" for i in range(scales.shape[1])]
    fields += [f"rot_{i}" for i in range(quats.shape[1])]
    data = np.concatenate([means, np.zeros_like(means), sh0, shN, opac, scales, quats], axis=1)
    arr = np.empty(len(means), dtype=[(f, "f4") for f in fields])
    for i, f in enumerate(fields):
        arr[f] = data[:, i]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


# ── Training ─────────────────────────────────────────────────────────

def train_splat(frames, pts, rgb, out_path: Path, iters: int,
                sh_degree: int, ssim_lambda: float, max_init: int,
                crop_to_init: bool, crop_margin: float):
    import numpy as np
    import torch
    from gsplat import rasterization

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
    print(f"  init from cloud: {len(pts):,} points; {len(frames)} posed frames", flush=True)

    crop_aabb = None
    if crop_to_init:
        lo, hi = pts.min(0), pts.max(0); m = (hi - lo) * crop_margin
        crop_aabb = (lo - m, hi + m)

    # Normalise the scene to ~unit scale for training: the means learning rate and
    # the scale clamp below are tuned for a unit-cube scene, so a metric scene
    # tens-of-metres across would mis-scale them.  We train normalised, then
    # un-normalise means/scales on export.
    centre = pts.mean(0)
    norm = float(np.linalg.norm(pts.std(axis=0))) + 1e-6
    pts_n = (pts - centre) / norm
    S = np.eye(4); S[:3, :3] *= norm; S[:3, 3] = centre   # normalised → world
    for f in frames:                                      # world→cam ∘ (norm→world)
        f["viewmat"] = f["viewmat"] @ S

    # Refine-only training (no densification).  The coloured cloud is already a
    # dense, geometrically-correct seed (one Gaussian per measured surface point),
    # so — like JMStudio's own splat — we don't synthesise new geometry; we just
    # optimise colour/opacity/scale/rotation so the splat is photo-consistent.
    # This is also the only *stable* path on this data: gsplat's fisheye renderer
    # needs the unscented-transform (with_ut), whose 2-D means gradient the default
    # densifier can't use, and MCMC relocation kept diverging (an unbounded
    # exp(scale) regulariser overflowed to NaN, then crashed torch.multinomial).
    # We start opaque enough to cover surfaces and clamp scales to keep exp finite.
    # Freeze opacity: on this multi-view-inconsistent data (auto-exposure, specular
    # — see the ~17.5 dB photometric ceiling) the photometric gradient otherwise
    # drives opacity → 0, dissolving solid surfaces into translucent haze (worse
    # than the clean init).  Held opaque, training can only sharpen colour/shape:
    # scales+quats flatten the isotropic seeds into surface-aligned discs.
    params = build_gaussians(pts_n, rgb, sh_degree, device, init_opacity=0.8)
    optimizers = make_optimizers(params, lr_scale=1.0, freeze=("opacities",))

    sc = iters / 30_000
    scale_clamp = float(np.log(0.3))         # ≤0.3 in unit space (~1.1 m world)
    sh_increase_every = max(1, round(1000 * sc))

    for f in frames:
        f["image_t"] = torch.tensor(f["image"]).pin_memory()
        f["K_t"] = torch.tensor(f["K"], dtype=torch.float32, device=device)
        f["viewmat_t"] = torch.tensor(f["viewmat"], dtype=torch.float32, device=device)
        # gsplat fisheye radial coeffs: shape [C, 4] (one camera per call).
        f["radial_t"] = torch.tensor(f["radial"], dtype=torch.float32,
                                     device=device)[None]

    rng = np.random.default_rng(0)
    order = rng.permutation(len(frames)); cur = 0
    for step in range(iters):
        if cur >= len(order):
            order = rng.permutation(len(frames)); cur = 0
        f = frames[int(order[cur])]; cur += 1
        H, W = f["image"].shape[:2]
        sh_deg = min(sh_degree, step // sh_increase_every)
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        renders, alphas, info = rasterization(
            means=params["means"], quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors, viewmats=f["viewmat_t"][None], Ks=f["K_t"][None],
            width=W, height=H, sh_degree=sh_deg, packed=False,
            camera_model="fisheye", radial_coeffs=f["radial_t"], with_ut=True)
        pred = renders[0].clamp(0, 1)
        gt = f["image_t"].to(device, non_blocking=True)
        l1 = (pred - gt).abs().mean()
        loss = (1 - ssim_lambda) * l1 + ssim_lambda * (1 - windowed_ssim(pred, gt))
        if not torch.isfinite(loss):         # skip rare bad frames, don't poison params
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            print(f"  step {step}: non-finite loss, skipped", flush=True)
            continue
        loss.backward()
        for opt in optimizers.values():
            opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():                # keep exp(scales) finite/bounded
            params["scales"].clamp_(max=scale_clamp)
        if step % 100 == 0:
            print(f"  step {step:6d}  loss {loss.item():.4f}  "
                  f"gaussians {params['means'].shape[0]:,}", flush=True)
            print(f"PROGRESS {step + 1}/{iters}", flush=True)
            pct = 10 + int(85 * step / max(iters, 1))
            progress(min(pct, 95), f"Training splat {step}/{iters}…")
    print(f"PROGRESS {iters}/{iters}", flush=True)

    # Train-view PSNR readout (cheap sanity / quality gauge).
    with torch.no_grad():
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        tot = n = 0
        for fi in range(0, len(frames), max(1, len(frames) // 20)):
            f = frames[fi]; H, W = f["image"].shape[:2]
            out, _, _ = rasterization(
                means=params["means"], quats=params["quats"],
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]), colors=colors,
                viewmats=f["viewmat_t"][None], Ks=f["K_t"][None],
                width=W, height=H, sh_degree=sh_degree, packed=False,
                camera_model="fisheye", radial_coeffs=f["radial_t"], with_ut=True)
            mse = (out[0].clamp(0, 1) - f["image_t"].to(device)).pow(2).mean()
            tot += float(10 * torch.log10(1.0 / mse)); n += 1
        print(f"  train-view PSNR (mean over {n} frames): {tot / n:.2f} dB", flush=True)

    # Un-normalise back to world: means scale+shift, log-scales add log(norm).
    with torch.no_grad():
        params["means"].mul_(norm).add_(torch.tensor(centre, dtype=torch.float32,
                                                      device=device))
        params["scales"].add_(float(np.log(norm)))

    export_ply(params, out_path, crop_aabb)
    print(f"Saved: {out_path}", flush=True)


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
    rot = np.zeros((n, 4), np.float32); rot[:, 0] = 1.0
    opac = np.full((n, 1), 2.197, np.float32)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in
                        ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
                         "opacity", "scale_0", "scale_1", "scale_2",
                         "rot_0", "rot_1", "rot_2", "rot_3"])
              + "end_header\n")
    data = np.concatenate([pts, np.zeros((n, 3), np.float32), dc, opac, scale, rot],
                          axis=1).astype(np.float32)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ply, "wb") as fh:
        fh.write(header.encode()); fh.write(data.tobytes())
    print(f"Saved (bootstrap): {output_ply}", flush=True)
    progress(100, "Complete (bootstrap — no GPU training)!")


# ── Surfel splat: one anisotropic surface-aligned disc per point ─────

def surfel_splat_from_ply(input_ply: Path, output_ply: Path,
                          k: int = 12, radius_factor: float = 0.6,
                          flatten: float = 0.2, opacity: float = 0.9,
                          sor_std: float = 2.0):
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

    coloured = rgb.sum(1) > 0.02            # drop unseen (black) points
    pts, rgb = pts[coloured], rgb[coloured]
    print(f"  {len(pts):,} coloured points", flush=True)

    progress(30, "Estimating local surface (kNN PCA)…")
    tree = cKDTree(pts)
    d, idx = tree.query(pts, k=k + 1, workers=-1)        # (n,k+1) incl. self

    # Statistical outlier removal: drop points whose mean neighbour distance is
    # far above the global mean — these are the floaters that read as fuzz.
    mdist = d[:, 1:].mean(1)
    keep = mdist < (mdist.mean() + sor_std * mdist.std())
    if not keep.all():
        print(f"  outlier removal: dropped {int((~keep).sum()):,} / {len(pts):,} floaters", flush=True)
        pts, rgb, idx, d = pts[keep], rgb[keep], idx[keep], d[keep]
    n = len(pts)
    # idx still references the pre-filter array, so PCA reads original neighbours.
    nn = np.clip(d[:, 1], 1e-4, None)                    # nearest-neighbour dist

    nbr_src = tree.data                                  # full point set for neighbour lookup
    nbr = nbr_src[idx[:, 1:]]                             # (n,k,3)
    cen = nbr - nbr.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", cen, cen) / k        # (n,3,3) covariance
    evals, evecs = np.linalg.eigh(cov)                   # ascending; evecs cols
    # Smallest-variance axis = normal; the other two span the tangent plane.
    R = np.stack([evecs[:, :, 2], evecs[:, :, 1], evecs[:, :, 0]], axis=2)  # cols [t1,t2,n]
    det = np.linalg.det(R)
    R[det < 0, :, 2] *= -1.0                              # keep right-handed (proper rotation)

    quat_xyzw = Rotation.from_matrix(R).as_quat()        # [x,y,z,w]
    quats = np.column_stack([quat_xyzw[:, 3], quat_xyzw[:, :3]]).astype(np.float32)  # [w,x,y,z]

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
    print(f"  {n:,} surfels; disc radius median {np.median(r_in)*1000:.1f} mm "
          f"(cap {cap*1000:.1f} mm), flatten {flatten:g}, aniso {1/flatten:.0f}:1", flush=True)

    progress(80, f"Writing {n:,} surfels…")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in
                        ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
                         "opacity", "scale_0", "scale_1", "scale_2",
                         "rot_0", "rot_1", "rot_2", "rot_3"])
              + "end_header\n")
    normals = R[:, :, 2].astype(np.float32)              # write the real surface normal
    data = np.concatenate([pts, normals, dc, opac, scale, quats], axis=1).astype(np.float32)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ply, "wb") as fh:
        fh.write(header.encode()); fh.write(data.tobytes())
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
    p.add_argument("--no-crop-to-init", dest="crop_to_init", action="store_false", default=True)
    p.add_argument("--crop-margin", type=float, default=0.05)
    p.add_argument("--splat-size", type=float, default=0.05,
                   help="bootstrap Gaussian radius (m) — smaller = finer splats, less blobby")
    p.add_argument("--bootstrap", action="store_true",
                   help="skip GPU training; one isotropic Gaussian per point (CPU)")
    p.add_argument("--surfel", action="store_true",
                   help="skip GPU training; one anisotropic surface-aligned disc "
                        "per point (CPU). Best match to 3DMakerpro on a dense cloud.")
    p.add_argument("--surfel-flatten", type=float, default=0.2,
                   help="surfel thickness as a fraction of disc radius (smaller = flatter)")
    p.add_argument("--surfel-sor", type=float, default=2.0,
                   help="outlier removal strength (σ); lower = drop more floaters")
    args = p.parse_args()

    scan = Path(args.scan)
    out = Path(args.output)
    pc = Path(args.pointcloud) if args.pointcloud else None
    out.parent.mkdir(parents=True, exist_ok=True)

    if pc is None or not pc.exists():
        print("ERROR: a coloured --pointcloud is required (run Generate Point "
              "Cloud first).", flush=True)
        sys.exit(1)

    traj_path = Path(str(pc) + ".traj.npz")

    # Surfel mode short-circuits everything: a fast CPU anisotropic-disc splat
    # straight from the coloured cloud (no poses/GPU needed).
    if args.surfel:
        surfel_splat_from_ply(pc, out, flatten=args.surfel_flatten,
                              sor_std=args.surfel_sor)
        return

    # Decide path: real GPU training when we have a trajectory + CUDA, else bootstrap.
    can_train = traj_path.exists() and not args.bootstrap
    if can_train:
        ensure_cuda_home()
        try:
            import torch
            if not torch.cuda.is_available():
                print("  No CUDA GPU available — falling back to CPU bootstrap.", flush=True)
                can_train = False
        except ImportError:
            print("  torch/gsplat not installed — falling back to CPU bootstrap.", flush=True)
            can_train = False

    if not can_train:
        if not traj_path.exists():
            print(f"  No trajectory sidecar ({traj_path.name}) — bootstrap only. "
                  "Regenerate the point cloud to enable trained splats.", flush=True)
        bootstrap_splat_from_ply(pc, out, size=args.splat_size)
        return

    progress(5, "Loading coloured cloud and camera poses…")
    pts, rgb = load_coloured_ply(pc)
    frames = build_camera_frames(scan, traj_path, args.camera,
                                 args.image_rot, args.downscale)
    print(f"  {len(frames)} posed frames @ {frames[0]['image'].shape[1]}×"
          f"{frames[0]['image'].shape[0]} (downscale {args.downscale})", flush=True)
    progress(10, f"Training {args.iterations}-iter splat on GPU…")
    train_splat(frames, pts, rgb, out, args.iterations, args.sh_degree,
                args.ssim_lambda, args.max_init_points,
                args.crop_to_init, args.crop_margin)
    progress(100, "Complete!")


if __name__ == "__main__":
    main()
