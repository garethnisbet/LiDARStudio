"""Join the separately solved pieces of a capture into one set of poses.

A capture that falls apart (a blank wall, a blurred pan) is solved as pieces,
each in its own frame and scale (process_sfm writes ``sfm_viewmats_X.npz``
and ``sfm_viewmats_X_part<k>.npz``). Once the user has moved each piece's
cloud or splat into place in the viewer, the same placement applied to that
piece's CAMERAS puts every frame of the capture into one frame, so a single
splat can train on all of it.

The placement is a similarity: rotation, one scale, translation. A camera
cannot be stretched along one axis, so a non-uniform scale is refused.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# Column norms of the placement must agree to this ratio for it to count as a
# uniform scale. The gizmo's scale handles are per axis, and a drag that
# stretched one axis by a percent would otherwise pass silently.
UNIFORM_SCALE_TOL = 1.01


def _sfm_stem(name: str):
    """``pointcloud_X_part2`` / ``splat_X_part2`` → ``X_part2``."""
    for prefix in ("pointcloud_", "splat_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return None


def poses_for_object(path) -> Path | None:
    """The poses file of the piece a cloud or splat came from, or None.

    Generated files live in ``<project>/pointclouds`` or ``<project>/splats``
    as ``pointcloud_<X>.ply`` / ``splat_<X>.ply``, with the poses at
    ``<project>/sfm/sfm_viewmats_<X>.npz``. An edit keeps the file in the same
    frame (crop, denoise, erase), so ``pointcloud_X_part2_edited2.ply`` still
    belongs to piece 2: trailing ``_<word><n>`` tags are stripped until a
    poses file is found.
    """
    p = Path(path)
    stem = _sfm_stem(p.stem)
    if stem is None:
        return None
    sfm = p.parent.parent / "sfm"
    while True:
        cand = sfm / f"sfm_viewmats_{stem}.npz"
        if cand.exists():
            return cand
        shorter = re.sub(r"_[A-Za-z]+\d*$", "", stem)
        # Never strip the piece number itself: "_part2" is a piece, not a tag.
        if shorter == stem or re.search(r"_part\d+$", stem):
            return None
        stem = shorter


def similarity(matrix):
    """``(M, s)`` from a viewer world matrix (16 numbers, column-major): the
    row-major 4x4 and its uniform scale. Raises ValueError for a non-uniform
    scale or a shear, neither of which a camera can follow."""
    M = np.asarray(matrix, float).reshape(4, 4).T
    A = M[:3, :3]
    norms = np.linalg.norm(A, axis=0)
    if norms.min() <= 0 or norms.max() / norms.min() > UNIFORM_SCALE_TOL:
        raise ValueError(
            "a piece is scaled unevenly (x/y/z scales differ by "
            f"{100 * (norms.max() / max(norms.min(), 1e-12) - 1):.1f}%); "
            "cameras can only follow one overall scale. Set the three scale "
            "values equal")
    s = float(np.cbrt(np.prod(norms)))
    R = A / s
    if np.abs(R.T @ R - np.eye(3)).max() > 1e-3 or np.linalg.det(R) <= 0:
        raise ValueError("a piece's placement is sheared or mirrored; "
                         "cameras can only be rotated, scaled and moved")
    return M, s


def move_viewmats(viewmats, M, s):
    """World→camera matrices after the world is moved by the similarity
    ``M`` (scale ``s``): each camera's centre moves with the scene, and its
    orientation turns with the scene's rotation."""
    V = np.asarray(viewmats, float)
    R = M[:3, :3] / s
    t = M[:3, 3]
    Rw = V[:, :3, :3]                                   # world→cam rotation
    C = -np.einsum("nji,nj->ni", Rw, V[:, :3, 3])       # camera centres
    C2 = (s * (R @ C.T)).T + t
    Rw2 = Rw @ R.T
    out = np.tile(np.eye(4), (len(V), 1, 1))
    out[:, :3, :3] = Rw2
    out[:, :3, 3] = -np.einsum("nij,nj->ni", Rw2, C2)
    return out


def join_poses(items, out_path):
    """Write one poses npz for the pieces ``items`` = ``[(npz, matrix)]``.

    Each piece keeps its own solved lens, as ``frame_camera_params`` (one row
    per frame), since each was bundle-adjusted on its own; ``camera_params``
    is the first piece's, for readers that know only one. Returns a summary.
    """
    names, ts, vms, fparams = [], [], [], []
    model = None
    scales = []
    for npz, matrix in items:
        with np.load(npz, allow_pickle=True) as d:
            m = str(d["camera_model"])
            if model is None:
                model = m
                first_params = np.asarray(d["camera_params"], float)
                camera = str(d["camera"]) if "camera" in d.files else "front"
            elif m != model:
                raise ValueError(
                    f"{Path(npz).name} was solved with a {m} lens, the others "
                    f"with {model}; they cannot share one training run")
            M, s = similarity(matrix) if matrix is not None else (np.eye(4), 1.0)
            scales.append(s)
            vm = move_viewmats(d["viewmats"], M, s)
            n = len(vm)
            names += [str(x) for x in d["names"]]
            ts += [int(x) for x in d["bag_ts_ns"]]
            vms.append(vm)
            fparams.append(np.tile(np.asarray(d["camera_params"], float), (n, 1)))
    if len(set(ts)) != len(ts):
        raise ValueError("two of the pieces contain the same frame; join "
                         "pieces of one capture, each only once")
    order = np.argsort(np.array(ts, dtype=np.int64), kind="stable")
    V = np.concatenate(vms)[order]
    P = np.concatenate(fparams)[order]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_path) + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(
            f,
            names=np.array(names)[order],
            bag_ts_ns=np.array(ts, dtype=np.int64)[order],
            viewmats=V,
            fisheye_params=first_params,
            camera_model=np.array(model),
            camera_params=first_params,
            frame_camera_params=P,
            coverage_deg=np.array(float("nan")),
            image_set=np.array(True),
            camera=np.array(camera),
            mask_dir=np.array(""),
            turntable=np.array(False),
            joined_pieces=np.array([str(Path(p).name) for p, _ in items]),
        )
    tmp.replace(out_path)
    return {"poses": str(out_path), "frames": int(len(V)),
            "pieces": len(items), "scales": scales}
