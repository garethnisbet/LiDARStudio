#!/usr/bin/env python3
"""
Surface Mesh Generator (LiDAR → screened Poisson → photo-coloured mesh)

Turns the fused, gravity-levelled world cloud produced by
``process_pointcloud.py`` into a watertight triangle mesh and colours every
vertex by re-projecting it into the camera photos — the "robust deliverable"
alternative to Gaussian splatting.  Because the LiDAR already gives metric
geometry and the trajectory sidecar gives real camera poses, this avoids the
scale/geometry guessing that makes image-only photogrammetry fragile.

Pipeline:
    1. Load the fused cloud + its ``.traj.npz`` trajectory sidecar (same frame).
    2. Estimate normals, then ORIENT them toward the nearest camera (we know
       which way is "outward" from the poses — far more reliable than the
       tangent-plane heuristic Poisson normally leans on).
    3. Screened Poisson reconstruction → triangle mesh.
    4. Trim the low-density Poisson "balloon" and crop to the cloud's bounds.
    5. Colour each vertex via the occlusion-aware multi-view fisheye projector
       reused from process_pointcloud (falls back to the nearest cloud colour
       for vertices no camera saw).
    6. Write a vertex-coloured PLY.

Usage (invoked by the server; runnable standalone):
    python process_mesh.py
        --cloud      /project/pointclouds/pointcloud_<ts>.ply
        --image-bag  /scan/IMAGE_<ts>.bag
        --output     /project/meshes/mesh_<ts>.ply
        [--calibration /scan/calibration]
        [--camera front]           # comma list to colour from several cameras
        [--depth 10]               # Poisson octree depth (9 draft…11 fine)
        [--density-quantile 0.02]  # drop this fraction of lowest-density verts
        [--decimate 0]             # target triangle count (0 = keep all)
        [--colour-range 20.0]

Progress protocol matches the sibling scripts: lines beginning
``PROGRESS:<percent>:<message>`` drive the server's progress bar.

Required packages:
    pip install rosbags open3d numpy opencv-python-headless scipy
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import process_pointcloud as ppc  # noqa: E402


def progress(pct: int, msg: str):
    print(f"PROGRESS:{pct}:{msg}", flush=True)


def _find_traj(cloud: Path):
    """Locate the trajectory sidecar for a cloud, resolving numbered
    ``_edited`` / ``_recoloured`` names back to the base cloud (mirrors
    edit_ops._find_traj so an edited seed still finds its poses)."""
    base = re.sub(r"(_(?:edited|recoloured)\d*)+$", "", cloud.stem)
    for cand in (
        Path(str(cloud) + ".traj.npz"),
        cloud.with_name(base + ".ply.traj.npz"),
    ):
        if cand.exists():
            return cand
    return None


def orient_normals_to_cameras(pcd, cam_centers):
    """Flip each normal so it faces the nearest camera position.

    Poisson is only as good as its normal orientation; a wrong flip yields
    inside-out blobs.  Image-only pipelines must guess orientation from a
    consistent tangent plane — but we recorded where the cameras were, so we
    orient each surface point toward the view that saw it.  O(N·log F) via a
    KD-tree over the (few thousand) camera centres.
    """
    import numpy as np

    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    if len(nrm) == 0:
        return

    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(cam_centers).query(pts, workers=-1)
        nearest_cam = cam_centers[idx]
    except ImportError:
        # Chunked brute force so we never allocate an (N×F) matrix at once.
        nearest_cam = np.empty_like(pts)
        for s in range(0, len(pts), 200_000):
            block = pts[s : s + 200_000]
            d = ((block[:, None, :] - cam_centers[None, :, :]) ** 2).sum(-1)
            nearest_cam[s : s + len(block)] = cam_centers[d.argmin(1)]

    to_cam = nearest_cam - pts
    flip = (nrm * to_cam).sum(1) < 0
    nrm[flip] *= -1
    pcd.normals = _o3d().utility.Vector3dVector(nrm)


def _o3d():
    import open3d as o3d

    return o3d


def fill_uncoloured_from_cloud(rgb, verts, pcd):
    """Vertices no camera saw come out black from the projector; paint them
    with the colour of the nearest coloured cloud point so holes read as
    real geometry rather than black speckle.  Returns the count filled."""
    import numpy as np

    cloud_rgb = np.asarray(pcd.colors)
    if len(cloud_rgb) == 0:
        return 0
    black = rgb.sum(1) == 0
    if not black.any():
        return 0
    cloud_pts = np.asarray(pcd.points)
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(cloud_pts).query(verts[black], workers=-1)
    except ImportError:
        idx = np.empty(int(black.sum()), dtype=np.int64)
        q = verts[black]
        for s in range(0, len(q), 100_000):
            blk = q[s : s + 100_000]
            d = ((blk[:, None, :] - cloud_pts[None, :, :]) ** 2).sum(-1)
            idx[s : s + len(blk)] = d.argmin(1)
    rgb[black] = (cloud_rgb[idx] * 255).astype(np.uint8)
    return int(black.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud", required=True, help="fused world-frame .ply")
    parser.add_argument("--image-bag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration", default=None)
    parser.add_argument(
        "--camera",
        default="front",
        help="camera name(s) to colour from; comma-separated to merge several",
    )
    parser.add_argument("--image-rot", default="ccw", choices=["cw", "ccw", "180", "none"])
    parser.add_argument("--depth", type=int, default=10, help="Poisson octree depth")
    parser.add_argument(
        "--density-quantile",
        type=float,
        default=0.02,
        help="drop this fraction of lowest-density vertices (Poisson balloon)",
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=0,
        help="target triangle count via quadric decimation (0 = keep all)",
    )
    parser.add_argument(
        "--outlier-neighbors",
        type=int,
        default=20,
        help="statistical-outlier removal neighbour count (0 = skip). Strips the "
        "stray returns + surface fuzz that make Poisson wobble.",
    )
    parser.add_argument(
        "--outlier-std",
        type=float,
        default=2.0,
        help="statistical-outlier std-ratio; smaller = more aggressive",
    )
    parser.add_argument(
        "--pre-voxel",
        type=float,
        default=0.0,
        help="light voxel downsample (m) before meshing (0 = off); thins the "
        "cloud's surface shell so Poisson fits a single sheet, not a slab",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=10,
        help="Taubin smoothing iterations on the mesh (0 = off). Volume-preserving "
        "denoise that removes the Poisson 'cauliflower' texture.",
    )
    parser.add_argument("--colour-range", type=float, default=20.0)
    args = parser.parse_args()

    ppc._require("open3d", "numpy", "rosbags", "cv2")
    import numpy as np
    import open3d as o3d

    cloud = Path(args.cloud)
    output = Path(args.output)
    image_bag = Path(args.image_bag)

    # 1. Load cloud + trajectory (both already gravity-levelled → same frame).
    progress(5, f"Loading cloud {cloud.name}…")
    pcd = o3d.io.read_point_cloud(str(cloud))
    n_in = len(pcd.points)
    if n_in == 0:
        print("ERROR: cloud has no points", flush=True)
        sys.exit(1)
    print(f"  cloud: {n_in:,} points", flush=True)

    traj_path = _find_traj(cloud)
    if traj_path is None:
        print(f"ERROR: no trajectory sidecar (.traj.npz) for {cloud.name}", flush=True)
        print("  Re-generate the cloud so poses are available for colouring.", flush=True)
        sys.exit(1)
    traj = np.load(str(traj_path))
    poses, lidar_ts = traj["poses"], traj["ts"]
    cam_centers = poses[:, :3, 3]
    print(f"  trajectory: {len(poses):,} poses from {traj_path.name}", flush=True)

    # 1b. Pre-clean: LiDAR surfaces carry a few cm of thickness plus stray
    #     returns, so Poisson fits a wobbly slab instead of one sheet. Strip
    #     outliers and (optionally) thin the shell before reconstruction.
    if args.outlier_neighbors > 0 and len(pcd.points) > args.outlier_neighbors:
        progress(12, "Removing statistical outliers…")
        pcd, keep = pcd.remove_statistical_outlier(
            nb_neighbors=args.outlier_neighbors, std_ratio=args.outlier_std
        )
        print(f"  outlier removal: kept {len(keep):,}/{n_in:,} points", flush=True)
    if args.pre_voxel > 0:
        pcd = pcd.voxel_down_sample(args.pre_voxel)
        print(f"  pre-voxel {args.pre_voxel} m → {len(pcd.points):,} points", flush=True)

    # 2. Normals, oriented toward the cameras.
    progress(20, "Estimating normals (camera-oriented)…")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    orient_normals_to_cameras(pcd, cam_centers)

    # 3. Screened Poisson.
    progress(40, f"Screened Poisson reconstruction (depth={args.depth})…")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.depth
    )
    densities = np.asarray(densities)
    print(f"  Poisson mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris", flush=True)

    # 4. Trim the low-density balloon, then crop to the cloud's real extent
    #    (Poisson extrapolates a closed hull well past the scanned surface).
    progress(60, "Trimming & cropping…")
    if args.density_quantile > 0:
        thresh = np.quantile(densities, args.density_quantile)
        mesh.remove_vertices_by_mask(densities < thresh)
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if args.decimate > 0 and len(mesh.triangles) > args.decimate:
        mesh = mesh.simplify_quadric_decimation(args.decimate)
        mesh.remove_unreferenced_vertices()
    print(f"  after cleanup: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris", flush=True)

    # 4b. Taubin smoothing — denoise the surface *before* colouring so vertex
    #     colours project onto their final positions (volume-preserving, so the
    #     mesh doesn't shrink the way plain Laplacian would).
    if args.smooth > 0:
        progress(63, f"Taubin smoothing ({args.smooth} iters)…")
        mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth)

    # 5. Colour vertices by re-projecting them into the photos (reuse the
    #    occlusion-aware multi-view fisheye projector from process_pointcloud).
    verts = np.asarray(mesh.vertices)
    rgb = np.zeros((len(verts), 3), np.uint8)
    calib_dir = Path(args.calibration) if args.calibration else None
    cameras = [c.strip() for c in args.camera.split(",") if c.strip()]
    if calib_dir and calib_dir.exists():
        for ci, cam in enumerate(cameras):
            try:
                calib = ppc.load_calibration(calib_dir, cam)
            except Exception as e:
                print(f"  skip camera '{cam}': {e}", flush=True)
                continue
            base = 65 + int(23 * ci / max(len(cameras), 1))
            span = int(23 / max(len(cameras), 1))

            def _cb(i, n, base=base, span=span):
                progress(min(base + int(span * i / max(n, 1)), 89),
                         f"Colouring from '{cam}' photo {i}/{n}…")

            cam_rgb, _ = ppc.colour_points_multiview(
                verts, poses, lidar_ts, image_bag, calib,
                camera=cam, image_rot=args.image_rot,
                max_range=args.colour_range, progress_cb=_cb,
            )
            # Merge: keep whatever the earlier cameras already coloured.
            take = (rgb.sum(1) == 0) & (cam_rgb.sum(1) > 0)
            rgb[take] = cam_rgb[take]
        n_proj = int((rgb.sum(1) > 0).sum())
        print(f"  projected colour on {n_proj:,}/{len(verts):,} verts", flush=True)
        n_fill = fill_uncoloured_from_cloud(rgb, verts, pcd)
        if n_fill:
            print(f"  filled {n_fill:,} unseen verts from nearest cloud colour", flush=True)
    else:
        # No calibration → carry the cloud's own colours onto the mesh.
        print("  no calibration — colouring from nearest cloud point", flush=True)
        fill_uncoloured_from_cloud(rgb, verts, pcd)

    mesh.vertex_colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)

    # 6. Write.
    progress(95, f"Writing {len(mesh.vertices):,} verts to {output.name}…")
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output), mesh)

    progress(100, "Complete!")
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
