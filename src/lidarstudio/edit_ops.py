"""Unified headless edit operations for point clouds AND Gaussian splats.

Reuses raven's ``cloud_ops`` (open3d) and ``splat_io`` (format-preserving splat
PLY).  Every op reduces to a *keep-index set* over the points/gaussians, applied
to a cloud via ``cloud_ops.select`` or to a splat via ``splat_io.save_splat`` so
the splat's per-gaussian attributes (scale/rotation/opacity/SH) survive untouched.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

from lidarstudio import cloud_ops, splat_io


def _load(path):
    """Return (xyz Nx3, kind, obj) where obj is a splat record or an o3d cloud."""
    if splat_io.is_splat_ply(path):
        data, _fields = splat_io.load_splat(path)  # (record array, field names)
        return np.asarray(splat_io.xyz(data), float), "splat", data
    pcd = cloud_ops.load(str(path))
    return np.asarray(pcd.points, float), "cloud", pcd


def _region_scoped(pcd, params, fn):
    """Run keep-fn only on points inside the visibility box, leaving the rest
    untouched. ``fn(sub_pcd) -> keep indices into sub_pcd``. ``params["matrix"]``
    is the box's to-local matrix (see cloud_ops.obb_mask); ``region_invert``
    selects the *outside* of the box instead (matching an "outside" view)."""
    mask = cloud_ops.obb_mask(pcd, params["matrix"])
    if params.get("region_invert"):
        mask = ~mask
    in_idx = np.nonzero(mask)[0]
    out_idx = np.nonzero(~mask)[0]
    if len(in_idx) == 0:
        return out_idx.astype(np.int64)  # region empty → keep everything
    sub = cloud_ops.select(pcd, in_idx)
    local_keep = np.asarray(fn(sub), dtype=np.int64)
    return np.sort(np.concatenate([out_idx, in_idx[local_keep]])).astype(np.int64)


def _keep_indices(op, params, pts, pcd):
    op = (op or "").lower()
    n = len(pts)
    scoped = params.get("matrix") is not None
    if op == "decimate":
        f = max(1, int(params.get("factor", 2)))
        if scoped:
            return _region_scoped(pcd, params, lambda s: np.arange(0, len(s.points), f))
        return np.arange(0, n, f, dtype=np.int64)
    if op == "crop":
        invert = bool(params.get("invert", False))
        # Oriented box (translate/rotate/scale) when a matrix is supplied;
        # otherwise fall back to the manual axis-aligned min/max bounds.
        if scoped:
            return cloud_ops.obb_keep(pcd, params["matrix"], invert=invert).astype(
                np.int64
            )
        return cloud_ops.aabb_keep(
            pcd, params["min"], params["max"], invert=invert
        ).astype(np.int64)
    if op in ("denoise_sor", "sor"):
        nb = int(params.get("nb_neighbors", 20))
        std = float(params.get("std_ratio", 2.0))
        if scoped:
            return _region_scoped(
                pcd, params, lambda s: cloud_ops.statistical_outlier_keep(s, nb, std)
            )
        return cloud_ops.statistical_outlier_keep(pcd, nb, std).astype(np.int64)
    if op == "erase":
        return cloud_ops.primitive_erase_keep(pcd, params.get("erasers", [])).astype(
            np.int64
        )
    if op == "drop":
        # Keep all points except the given indices (live eraser commit; indices
        # are in file/vertex order, matching the client's point buffer).
        drop = np.asarray(params.get("drop", []), dtype=np.int64)
        mask = np.ones(n, dtype=bool)
        drop = drop[(drop >= 0) & (drop < n)]
        mask[drop] = False
        return np.nonzero(mask)[0]
    if op in ("denoise_radius", "radius"):
        nb = int(params.get("nb_points", 16))
        rad = float(params.get("radius", 0.05))
        if scoped:
            return _region_scoped(
                pcd, params, lambda s: cloud_ops.radius_outlier_keep(s, nb, rad)
            )
        return cloud_ops.radius_outlier_keep(pcd, nb, rad).astype(np.int64)
    raise ValueError(f"unknown edit op: {op!r}")


def apply_edit(in_path, out_path, op, params):
    """Apply one edit op; write ``out_path``; return a summary dict."""
    import open3d as o3d

    pts, kind, obj = _load(in_path)
    n = len(pts)
    params = params or {}

    # Bake an in-scene transform into the file (save the object "as placed").
    # `matrix` is the object's world matrix as a length-16 column-major array.
    if (op or "").lower() in ("transform", "bake", "bake_transform"):
        M = (
            np.asarray(params["matrix"], float).reshape(4, 4).T
        )  # column-major -> row-major
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if kind == "splat":
            splat_io.save_splat(out_path, splat_io.transform_splat(obj, M))
        else:
            obj.transform(M)  # o3d: transforms points + normals
            cloud_ops.save(obj, str(out_path))
        return {
            "kind": kind,
            "total": int(n),
            "kept": int(n),
            "removed": 0,
            "output": str(out_path),
        }

    # Both kinds need an o3d cloud for the geometric ops (SOR/radius/aabb).
    if kind == "splat":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
    else:
        pcd = obj

    keep = np.asarray(_keep_indices(op, params or {}, pts, pcd), dtype=np.int64)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if kind == "splat":
        splat_io.save_splat(out_path, obj, keep)
    else:
        cloud_ops.save(cloud_ops.select(pcd, keep), str(out_path))

    # Frame-preserving edit → keep the result trainable (bake/transform returned
    # earlier, before this point, so its new frame is correctly excluded).
    if kind != "splat":
        _copy_traj(in_path, out_path)

    return {
        "kind": kind,
        "total": int(n),
        "kept": int(len(keep)),
        "removed": int(n - len(keep)),
        "output": str(out_path),
    }


def splat_to_cloud(in_path, out_path, min_opacity=0.05):
    """Export a trained splat's gaussian centres as a plain coloured point cloud.

    Training refines the seed cloud photometrically (means migrate onto true
    surfaces, noise gaussians are pruned, thin structures are densified), so the
    centres make a cleaner cloud than the LiDAR seed. Colour comes from the SH
    DC term; gaussians whose opacity fell below ``min_opacity`` (transparent →
    unsupported by any photo) are dropped. The splat is exported in the seed
    cloud's world frame, so the seed's trajectory sidecar is carried when it can
    be found, keeping the result trainable/meshable.
    """
    if not splat_io.is_splat_ply(in_path):
        raise ValueError(f"{Path(in_path).name} is not a gaussian-splat PLY")
    data, _fields = splat_io.load_splat(in_path)
    pts = splat_io.xyz(data)
    rgb = np.clip(splat_io.display_colours(data) * 255.0 + 0.5, 0, 255).astype(
        np.uint8
    )
    keep = np.ones(len(pts), bool)
    if min_opacity > 0 and "opacity" in (data.dtype.names or ()):
        alpha = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float64)))
        keep = alpha >= float(min_opacity)

    from lidarstudio import process_pointcloud as ppc

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ppc.write_ply(Path(out_path), pts[keep], rgb[keep])

    traj = _find_traj(in_path) or _seed_traj_for_splat(in_path)
    if traj is not None:
        shutil.copy2(traj, str(out_path) + ".traj.npz")

    return {
        "kind": "cloud",
        "total": int(len(pts)),
        "kept": int(keep.sum()),
        "removed": int(len(pts) - keep.sum()),
        "trainable": traj is not None,
        "output": str(out_path),
    }


def _seed_traj_for_splat(path):
    """Trajectory sidecar of the generated cloud a project splat was trained
    from. Splats carry no sidecar of their own, but they train (and export) in
    the seed cloud's world frame, so the same-timestamp
    ``pointclouds/pointcloud_<ts>.ply.traj.npz`` applies verbatim. Matched by
    the leading timestamp digits, so renamed/edited splats (e.g.
    ``splat_<ts>_aa_lrdecay_edited3``) still resolve."""
    p = Path(path)
    m = re.match(r"splat_(\d+)", p.stem)
    if p.parent.name != "splats" or not m:
        return None
    cand = p.parent.parent / "pointclouds" / f"pointcloud_{m.group(1)}.ply.traj.npz"
    return cand if cand.exists() else None


def is_monochrome(path, sample=200_000):
    """True when a CLOUD carries no photo colour — greyscale (r==g==b for every
    point: the ``--mono`` generator's intensity shading, or its uniform-grey
    fallback) or no colour properties at all. Splats (f_dc_*) are never mono
    here: recolouring a splat seed is a separate, explicit edit."""
    import numpy as np
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"].data
    names = set(v.dtype.names or ())
    if "f_dc_0" in names:
        return False
    if not {"red", "green", "blue"} <= names:
        return True
    idx = np.linspace(0, len(v) - 1, min(sample, len(v))).astype(np.int64)
    return bool(
        np.array_equal(v["red"][idx], v["green"][idx])
        and np.array_equal(v["green"][idx], v["blue"][idx])
    )


def _find_traj(path):
    """Locate the trajectory sidecar for a cloud/splat, resolving chained and
    numbered ``_edited`` / ``_recoloured`` names back to the base cloud (e.g.
    ``pointcloud_x_edited3.ply`` → ``pointcloud_x.ply.traj.npz``)."""
    p = Path(path)
    base = re.sub(r"(_(?:edited|recoloured)\d*)+$", "", p.stem)
    for cand in (Path(str(p) + ".traj.npz"), p.with_name(base + ".ply.traj.npz")):
        if cand.exists():
            return cand
    return None


def _copy_traj(in_path, out_path):
    """Carry the trajectory sidecar onto an edited/recoloured output so the
    result stays trainable: a crop/decimate/denoise/recolour keeps the cloud in
    the same world frame, so the KISS-ICP camera path still applies. Bake /
    transform edits change the frame and must NOT call this."""
    src = _find_traj(in_path)
    if src is None:
        return
    dst = Path(str(out_path) + ".traj.npz")
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


def recolour(
    in_path, out_path, scan_path, camera="front", image_rot="ccw", colour_range=20.0
):
    """Re-project the scan's photos onto a cloud/splat (multi-view, occlusion-aware).

    Uses the saved KISS-ICP trajectory + the scan's fisheye images — the same
    colouring the generator does, so an edited/trimmed cloud can be recoloured.
    """
    import numpy as np

    from lidarstudio import process_pointcloud as ppc

    pts, kind, obj = _load(in_path)
    traj_path = _find_traj(in_path)
    if traj_path is None:
        raise FileNotFoundError(
            "no trajectory sidecar found for this cloud "
            "(recolour needs the generated cloud's .traj.npz)"
        )
    traj = np.load(traj_path)
    poses, ts = traj["poses"], traj["ts"]

    scan = Path(scan_path)
    calib = ppc.load_calibration(scan / "calibration", camera)
    img_bags = sorted(scan.glob("IMAGE_*.bag")) or sorted(scan.glob("*.bag"))
    img_bag = next((b for b in img_bags if "IMAGE" in b.name.upper()), None)
    if img_bag is None:
        raise FileNotFoundError(f"no IMAGE_*.bag in {scan}")

    rgb, n_col = ppc.colour_points_multiview(
        pts,
        poses,
        ts,
        img_bag,
        calib,
        camera=camera,
        image_rot=image_rot,
        max_range=colour_range,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if kind == "splat":
        from splat_io import save_splat

        data = obj.copy()
        dc = ((rgb.astype(np.float32) / 255.0) - 0.5) / 0.28209479177387814
        for i in range(3):
            data[f"f_dc_{i}"] = dc[:, i]
        save_splat(out_path, data)
    else:
        ppc.write_ply(Path(out_path), pts, rgb)
        # Recolour is frame-preserving → keep the recoloured cloud trainable.
        _copy_traj(in_path, out_path)

    return {
        "kind": kind,
        "total": int(len(pts)),
        "coloured": int(n_col),
        "output": str(out_path),
    }
