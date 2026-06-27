"""Unified headless edit operations for point clouds AND Gaussian splats.

Reuses raven's ``cloud_ops`` (open3d) and ``splat_io`` (format-preserving splat
PLY).  Every op reduces to a *keep-index set* over the points/gaussians, applied
to a cloud via ``cloud_ops.select`` or to a splat via ``splat_io.save_splat`` so
the splat's per-gaussian attributes (scale/rotation/opacity/SH) survive untouched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import cloud_ops
import splat_io


def _load(path):
    """Return (xyz Nx3, kind, obj) where obj is a splat record or an o3d cloud."""
    if splat_io.is_splat_ply(path):
        data, _fields = splat_io.load_splat(path)   # (record array, field names)
        return np.asarray(splat_io.xyz(data), float), "splat", data
    pcd = cloud_ops.load(str(path))
    return np.asarray(pcd.points, float), "cloud", pcd


def _keep_indices(op, params, pts, pcd):
    op = (op or "").lower()
    n = len(pts)
    if op == "decimate":
        f = max(1, int(params.get("factor", 2)))
        return np.arange(0, n, f, dtype=np.int64)
    if op == "crop":
        return cloud_ops.aabb_keep(pcd, params["min"], params["max"],
                                   invert=bool(params.get("invert", False))).astype(np.int64)
    if op in ("denoise_sor", "sor"):
        return cloud_ops.statistical_outlier_keep(
            pcd, int(params.get("nb_neighbors", 20)),
            float(params.get("std_ratio", 2.0))).astype(np.int64)
    if op in ("denoise_radius", "radius"):
        return cloud_ops.radius_outlier_keep(
            pcd, int(params.get("nb_points", 16)),
            float(params.get("radius", 0.05))).astype(np.int64)
    raise ValueError(f"unknown edit op: {op!r}")


def apply_edit(in_path, out_path, op, params):
    """Apply one edit op; write ``out_path``; return a summary dict."""
    import open3d as o3d
    pts, kind, obj = _load(in_path)
    n = len(pts)

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

    return {"kind": kind, "total": int(n), "kept": int(len(keep)),
            "removed": int(n - len(keep)), "output": str(out_path)}
