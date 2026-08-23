#!/usr/bin/env python3
"""Dense reconstruction (MVS) — SfM sparse model → dense point cloud.

The mesh route for a phone capture. Photogrammetry's normal path to a surface
is SfM for poses, then MULTI-VIEW STEREO for geometry: patch-match computes a
depth map per image by matching image patches along epipolar lines, and fusion
merges the consistent depths into one cloud with per-point normals. That cloud
is what screened Poisson wants — a genuine surface sample, densely covering
everything the cameras saw.

Why this exists: the phone route to a mesh used to go through a trained splat,
harvesting ``edit_ops.splat_to_cloud`` gaussian centres for Poisson. That costs
hours of training to obtain a point cloud, and the centres are a poor surface
sample — they are placed to minimise photometric loss, not to lie on the
surface, they include floaters, and they carry no trustworthy normals (which
is why process_mesh has to orient normals toward the nearest camera). MVS
produces the right thing directly and skips training altogether.

What it cannot do: patch-match needs TEXTURE, and fails on the same surfaces
feature matching fails on. A blank glazed cylinder yields no depth here either.
For a textureless subject on a turntable the answer is silhouette carving
(process_sfm.py --carve-seed), which needs no texture at all — the two methods
read different channels, and the right pipeline for a mixed subject uses both.

Pipeline:
    1. Undistort the images into the workspace (patch-match needs a pinhole).
    2. patch_match_stereo — one depth + normal map per image. THE expensive
       step, and CUDA-only: COLMAP has no CPU patch-match.
    3. stereo_fusion — merge consistent depths into a coloured, normalled PLY,
       cropped to the sparse model's extent so a masked subject capture does
       not fuse the desk it stood on.

Usage (invoked by the server; runnable standalone):
    python process_dense.py
        --sparse  /project/sfm/sfm_viewmats_<ts>_sparse
        --images  /project/scans/<name>/images
        --output  /project/pointclouds/dense_<ts>.ply

Needs a CUDA-enabled pycolmap: a pip wheel is CPU-only and cannot run this at
all. The server locates one via lidar_jobs._sfm_python(); set
$LIDARSTUDIO_SFM_PYTHON to a conda-forge colmap build if the probe picks a
CPU-only interpreter.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def progress(pct: int, msg: str):
    print(f"PROGRESS:{pct}:{msg}", flush=True)


def subject_bounds(rec, margin: float = 0.25):
    """A crop box around the sparse points, or None if there are too few.

    The sparse model of a masked capture is subject-only — it was matched
    under the subject masks — but the DEPTH MAPS are not: patch-match knows
    nothing about masks and reconstructs the desk too. Cropping fusion to the
    sparse extent is what keeps the mesh a mesh of the subject.

    Percentiles rather than min/max, so one stray triangulation cannot inflate
    the box around the whole room.
    """
    import numpy as np

    xyz = np.array([p.xyz for p in rec.points3D.values()
                    if p.track.length() >= 3])
    if len(xyz) < 50:
        return None
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)
    mid, half = (hi + lo) / 2, (hi - lo) / 2 * (1 + margin)
    return mid - half, mid + half


def ply_vertex_count(path: Path) -> int:
    """Vertices declared in a PLY header, without loading the cloud.

    A fused cloud is tens of millions of points; reading it back just to count
    them would cost more than the fusion did.
    """
    with open(path, "rb") as f:
        for _ in range(64):  # headers are short; never scan a binary body
            line = f.readline()
            if not line or line.strip() == b"end_header":
                break
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sparse", required=True,
                   help="COLMAP sparse model dir (<output>_sparse from "
                        "process_sfm.py)")
    p.add_argument("--images", required=True, help="the ingested images/ dir")
    p.add_argument("--output", required=True, help="dense cloud .ply to write")
    p.add_argument("--workspace", default=None,
                   help="scratch dir for undistorted images and depth maps "
                        "(default: <output>_work); removed on success unless "
                        "--keep-workspace")
    p.add_argument("--max-image-size", type=int, default=2000,
                   help="downscale images past this long edge for stereo "
                        "(default 2000; the cost is quadratic in it)")
    p.add_argument("--no-crop", action="store_true",
                   help="fuse everything the depth maps saw, not just the "
                        "sparse model's extent — for a scene rather than an "
                        "object on a desk")
    p.add_argument("--no-geom-consistency", action="store_true",
                   help="skip the geometric-consistency pass: roughly twice "
                        "as fast, and noticeably noisier")
    p.add_argument("--keep-workspace", action="store_true")
    args = p.parse_args()

    import pycolmap

    if not getattr(pycolmap, "has_cuda", False):
        print(
            "ERROR: dense stereo needs a CUDA-enabled pycolmap — COLMAP has no "
            "CPU patch-match, so this cannot run on a pip wheel. Point "
            "$LIDARSTUDIO_SFM_PYTHON at a conda-forge colmap build and restart "
            "the server.",
            flush=True,
        )
        return 1

    sparse, images = Path(args.sparse), Path(args.images)
    output = Path(args.output)
    if not (sparse / "cameras.bin").is_file() and not (
            sparse / "cameras.txt").is_file():
        print(f"ERROR: no COLMAP model in {sparse}", flush=True)
        return 1
    if not images.is_dir():
        print(f"ERROR: images dir not found: {images}", flush=True)
        return 1
    ws = Path(args.workspace) if args.workspace else output.parent / (
        output.stem + "_work"
    )
    # A half-finished previous run leaves depth maps that patch-match would
    # skip as already done, silently mixing two geometries.
    shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rec = pycolmap.Reconstruction(str(sparse))
    log(f"sparse model: {rec.num_reg_images()} images, "
        f"{len(rec.points3D):,} points")

    progress(5, "Dense: undistorting images…")
    pycolmap.undistort_images(
        output_path=str(ws), input_path=str(sparse), image_path=str(images)
    )

    progress(15, f"Dense: patch-match stereo on {rec.num_reg_images()} images "
                 "(GPU, the slow step)…")
    pm = pycolmap.PatchMatchOptions()
    pm.max_image_size = args.max_image_size
    pm.geom_consistency = not args.no_geom_consistency
    pycolmap.patch_match_stereo(workspace_path=str(ws), options=pm)

    progress(80, "Dense: fusing depth maps…")
    fu = pycolmap.StereoFusionOptions()
    fu.max_image_size = args.max_image_size
    if not args.no_crop:
        box = subject_bounds(rec)
        if box is None:
            log("too few sparse points to crop against — fusing everything")
        else:
            fu.bounding_box = box
            log(f"cropping fusion to the sparse extent: "
                f"{box[0].round(3).tolist()} .. {box[1].round(3).tolist()}")
    dense = pycolmap.stereo_fusion(
        output_path=str(output), workspace_path=str(ws),
        input_type="geometric" if not args.no_geom_consistency else "photometric",
        # colmap lowercases this, but the accepted set really is
        # {'bin','ply','txt'} — spell it the way mvs.cc checks it.
        options=fu, output_type="ply",
    )

    del dense  # stereo_fusion writes the PLY itself; the returned model is
    # only a convenience, and is empty for a PLY output on some builds. The
    # header is the authority on what actually landed on disk.
    if not output.is_file():
        print(
            "ERROR: fusion produced no cloud. Either the depth maps are empty "
            "(a textureless subject — try process_sfm.py --carve-seed instead) "
            "or the crop box excluded everything (--no-crop).",
            flush=True,
        )
        return 1
    n = ply_vertex_count(output)
    if n == 0:
        print(
            "ERROR: fusion wrote an EMPTY cloud. The depth maps agreed on "
            "nothing — a textureless subject, or poses the stereo cannot "
            "reconcile. Check the reconstruction before meshing.",
            flush=True,
        )
        return 1
    log(f"dense cloud: {n:,} points → {output} "
        f"({output.stat().st_size / 1e6:.1f} MB)")

    if not args.keep_workspace:
        shutil.rmtree(ws, ignore_errors=True)
    progress(100, f"Dense cloud ready ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
