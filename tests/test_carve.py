"""--carve-seed: shape from silhouettes, the channel matching cannot read.

A turntable capture is a tomographic acquisition — each mask is a projection
of the subject, the stack of them against angle is its sinogram — and a
silhouette needs no texture. On the capture this was written for, ~150 frames
of blank glazed cylinder gave zero features and zero sparse points while
carrying a perfectly good outline in every frame.

The subject here is a SPHERE, whose visual hull from a full circle of views is
the sphere itself, so the carve has an exact answer to be checked against.
The camera is a stub: this is geometry, and none of it needs pycolmap.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from lidarstudio import lidar_jobs

sys.path.insert(0, str(Path(lidar_jobs.__file__).parent))
import process_sfm  # noqa: E402
import process_dense as process_sfm_dense  # noqa: E402

W, H, F = 160, 120, 150.0
RADIUS, DIST = 0.5, 3.0


class _StubCam:
    """A pinhole with the one method carve_visual_hull asks a camera for."""

    def img_from_cam(self, p):
        p = np.asarray(p, float)
        return np.stack([F * p[:, 0] / p[:, 2] + W / 2,
                         F * p[:, 1] / p[:, 2] + H / 2], axis=1)


def _scene(tmp_path, n=24, dilate=0, colour=(0, 0, 255), radius=RADIUS):
    """A sphere at the origin on a turntable: exact silhouettes, flat images."""
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    d = np.stack([(uu - W / 2) / F, (vv - H / 2) / F, np.ones_like(uu, float)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    names, vms = [], []
    for i in range(n):
        a = 2 * np.pi * i / n
        C = np.array([DIST * np.cos(a), 0.0, DIST * np.sin(a)])
        z = -C / np.linalg.norm(C)
        x = np.cross([0, 1, 0], z)
        x /= np.linalg.norm(x)
        vm = np.eye(4)
        vm[:3, :3] = np.stack([x, np.cross(z, x), z])
        vm[:3, 3] = -vm[:3, :3] @ C
        # Exact silhouette: the sphere covers a pixel when its centre lies
        # within RADIUS of that pixel's ray.
        c = vm[:3, 3]
        along = d @ c
        m = ((np.linalg.norm(c - along[..., None] * d, axis=-1) < radius)
             & (along > 0)).astype(np.uint8) * 255
        if dilate:
            m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
        name = f"{i:04d}.jpg"
        cv2.imwrite(str(masks / (name + ".png")), m)
        cv2.imwrite(str(images / name), np.full((H, W, 3), colour, np.uint8))
        names.append(name)
        vms.append(vm)

    rng = np.random.default_rng(0)
    seed = rng.normal(size=(300, 3))
    seed = radius * seed / np.linalg.norm(seed, axis=1, keepdims=True)
    return images, masks, names, np.array(vms), _StubCam(), seed


def _carve(tmp_path, grid=48, erode=0, **kw):
    images, masks, names, vms, cam, seed = _scene(tmp_path, **kw)
    return process_sfm.carve_visual_hull(
        masks, images, names, vms, cam, seed, grid=grid, views=24, erode=erode,
    )


def test_a_sphere_carves_to_its_own_radius(tmp_path):
    xyz, _ = _carve(tmp_path)
    r = np.linalg.norm(xyz, axis=1)
    # One voxel is 1.35*2*RADIUS/47 ≈ 0.029, so the surface lands within a
    # voxel of the true radius in both directions.
    assert abs(r.max() - RADIUS) < 0.05, r.max()
    assert abs(np.median(r) - RADIUS) < 0.05, np.median(r)


def test_the_hull_is_a_surface_not_a_solid(tmp_path):
    """Seeding the trainer with the inside of the subject is wasted points."""
    xyz, _ = _carve(tmp_path)
    r = np.linalg.norm(xyz, axis=1)
    assert (r > 0.7 * RADIUS).all(), "interior voxels must not be exported"


def test_the_carve_needs_texture_nowhere(tmp_path):
    """The whole point: the frames are a flat colour with no feature in them,
    and the shape still comes out."""
    xyz, _ = _carve(tmp_path, colour=(200, 200, 200))
    assert len(xyz) > 500


def test_colour_comes_from_the_frames_that_see_the_point(tmp_path):
    _, rgb = _carve(tmp_path, colour=(0, 0, 255))  # BGR red
    assert (rgb[:, 0] > 200).mean() > 0.95, "hull should be painted red"
    assert (rgb[:, 2] < 60).mean() > 0.95


def test_dilated_masks_inflate_the_hull_and_erode_cancels_it(tmp_path):
    """Masks written for MATCHING are dilated to keep the outline keypoints;
    that dilation is wrong for carving, and is what --mask-dilate cancels."""
    plain, _ = _carve(tmp_path / "a", dilate=0)
    fat, _ = _carve(tmp_path / "b", dilate=5)
    fixed, _ = _carve(tmp_path / "c", dilate=5, erode=5)
    r = lambda p: float(np.median(np.linalg.norm(p, axis=1)))  # noqa: E731
    assert r(fat) > r(plain) + 0.02, "dilation must visibly inflate the hull"
    assert abs(r(fixed) - r(plain)) < 0.02, "erode must cancel it"


def test_a_clipped_subject_is_reported(tmp_path, capsys):
    """Outside the image and outside the subject are the same test here, so a
    subject running off the frame edge gets carved away."""
    _carve(tmp_path, radius=2.0)
    assert "touches the frame edge" in capsys.readouterr().out


def test_too_few_views_cannot_intersect(tmp_path):
    images, masks, names, vms, cam, seed = _scene(tmp_path, n=4)
    with pytest.raises(SystemExit, match="at least 8 views"):
        process_sfm.carve_visual_hull(
            masks, images, names, vms, cam, seed, grid=32, views=4
        )


# ── MVS: the direct photos → surface route ───────────────────────────

def test_the_crop_box_brackets_the_sparse_points():
    """Depth maps cover the whole frame even when SfM was masked, so fusion
    has to be cropped or a desk capture meshes the desk."""
    class Pt:
        def __init__(s, x):
            s.xyz = np.array(x, float)
            s.track = type("T", (), {"length": staticmethod(lambda: 5)})()

    rng = np.random.default_rng(0)
    pts = {i: Pt(p) for i, p in enumerate(rng.normal(size=(400, 3)))}
    rec = type("R", (), {"points3D": pts})()
    lo, hi = process_sfm_dense.subject_bounds(rec, margin=0.0)
    inside = ((np.array([p.xyz for p in pts.values()]) >= lo)
              & (np.array([p.xyz for p in pts.values()]) <= hi)).all(1)
    assert inside.mean() > 0.9, "the box must contain almost every sparse point"
    assert (hi > lo).all()


def test_a_thin_reconstruction_is_not_cropped_against():
    """Better to fuse everything than to crop against a box built from noise."""
    class Pt:
        def __init__(s, x):
            s.xyz = np.array(x, float)
            s.track = type("T", (), {"length": staticmethod(lambda: 5)})()

    rec = type("R", (), {"points3D": {i: Pt([i, 0, 0]) for i in range(10)}})()
    assert process_sfm_dense.subject_bounds(rec) is None


def test_a_margin_grows_the_box():
    class Pt:
        def __init__(s, x):
            s.xyz = np.array(x, float)
            s.track = type("T", (), {"length": staticmethod(lambda: 5)})()

    rng = np.random.default_rng(1)
    rec = type("R", (), {"points3D": {i: Pt(p)
                                      for i, p in enumerate(rng.normal(size=(400, 3)))}})()
    tight = process_sfm_dense.subject_bounds(rec, margin=0.0)
    loose = process_sfm_dense.subject_bounds(rec, margin=0.5)
    assert (loose[1] - loose[0] > tight[1] - tight[0]).all()


def test_meshing_falls_back_when_there_is_no_sparse_model(tmp_path):
    """A capture reconstructed before the sparse model was kept must still
    mesh — through the splat route — rather than erroring."""
    import asyncio

    from lidarstudio import lidar_jobs

    proj = tmp_path / "proj"
    (proj / "sfm").mkdir(parents=True)
    npz = proj / "sfm" / "sfm_viewmats_cap.npz"
    npz.write_bytes(b"")
    msgs = []

    class Q:
        async def put(self, m):
            msgs.append(m)

    got = asyncio.run(
        lidar_jobs._dense_cloud(proj, "cap", npz, Q(), {})
    )
    assert got is None
    assert any("No sparse model" in m.get("message", "") for m in msgs)
    assert not any(m.get("event") == "error" for m in msgs), \
        "a missing sparse model is a fallback, not a failure"


def test_the_ply_header_gives_the_point_count_without_loading_it(tmp_path):
    """A fused cloud is tens of millions of points — counting them by reading
    the cloud back would cost more than the fusion did."""
    p = tmp_path / "c.ply"
    p.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1234\n"
        b"property float x\nend_header\n" + b"\x00" * 4936
    )
    assert process_sfm_dense.ply_vertex_count(p) == 1234


def test_a_headerless_file_counts_zero_rather_than_hanging(tmp_path):
    p = tmp_path / "junk.ply"
    p.write_bytes(b"\x00" * 100000)
    assert process_sfm_dense.ply_vertex_count(p) == 0


# ── --matching ordered: frame order as the anti-fold constraint ──────

def test_ordered_pairs_only_link_nearby_frames(tmp_path):
    """The fold is a LONG-RANGE false match. Deleting those edges from the
    graph makes it structurally impossible, not merely unlikely."""
    names = [f"{i:04d}.jpg" for i in range(50)]
    out = tmp_path / "pairs.txt"
    n = process_sfm.build_ordered_pairs(names, 5, out)
    rows = [ln.split() for ln in out.read_text().split("\n") if ln.strip()]
    assert n == len(rows)
    d = [abs(int(b[:4]) - int(a[:4])) for a, b in rows]
    assert max(d) == 5 and min(d) == 1
    assert not any(abs(int(b[:4]) - int(a[:4])) > 5 for a, b in rows), \
        "a pair beyond the window is exactly the edge a fold travels along"


def test_every_frame_reaches_its_neighbour(tmp_path):
    """A gap in the chain splits the capture into two unjoinable models."""
    names = [f"{i:04d}.jpg" for i in range(30)]
    out = tmp_path / "pairs.txt"
    process_sfm.build_ordered_pairs(names, 4, out)
    linked = set()
    for ln in out.read_text().split("\n"):
        if ln.strip():
            a, b = ln.split()
            linked.add(a)
            linked.add(b)
    assert linked == set(names)


def test_consecutive_frames_are_always_paired(tmp_path):
    names = [f"{i:04d}.jpg" for i in range(20)]
    out = tmp_path / "pairs.txt"
    process_sfm.build_ordered_pairs(names, 3, out)
    rows = {tuple(ln.split()) for ln in out.read_text().split("\n") if ln.strip()}
    for i in range(19):
        assert (f"{i:04d}.jpg", f"{i+1:04d}.jpg") in rows


def test_a_window_wider_than_the_capture_is_just_exhaustive(tmp_path):
    names = [f"{i:04d}.jpg" for i in range(10)]
    out = tmp_path / "pairs.txt"
    n = process_sfm.build_ordered_pairs(names, 999, out)
    assert n == 10 * 9 // 2


def test_a_window_that_yields_no_pairs_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no pairs"):
        process_sfm.build_ordered_pairs(["0000.jpg"], 5, tmp_path / "p.txt")


def test_a_turntable_capture_defaults_to_ordered_matching():
    """The GUI exposes no matcher control, so the default is the only lever —
    and on a turntable the fold is the characteristic failure. Measured on the
    House capture: exhaustive 172.8 deg with a 171 deg backward jump, ordered
    410.8 deg clean."""
    assert process_sfm.choose_matching(219, turntable=True) == "ordered"
    assert process_sfm.choose_matching(50, turntable=True) == "ordered"


def test_a_non_turntable_capture_keeps_the_size_based_choice():
    assert process_sfm.choose_matching(50) == "exhaustive"
    assert process_sfm.choose_matching(500) == "sequential"
