"""The splat→mesh route: meshing a phone capture, which has no LiDAR cloud,
no IMAGE bag and no trajectory sidecar.

Geometry comes from the trained splat's gaussian centres; camera centres come
from the SfM npz (normal orientation only); colour is carried from the cloud.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from plyfile import PlyData, PlyElement

from lidarstudio import lidar_jobs
from lidarstudio.splat_io import SH_C0

sys.path.insert(0, str(Path(lidar_jobs.__file__).parent))
import process_mesh  # noqa: E402


def _sphere(n, seed=0):
    """Points on a unit sphere with a little surface thickness, coloured by
    direction — a stand-in for an object-centric capture, and unitless the way
    an SfM frame is."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pts = v * (1.0 + rng.normal(scale=0.004, size=(n, 1)))
    return pts, v * 0.5 + 0.5


def _orbit(n=24, radius=4.0):
    """World→cam viewmats for cameras ringing the origin, plus their centres."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    centres = np.stack(
        [radius * np.cos(ang), radius * np.sin(ang), np.full_like(ang, 1.5)], 1
    )
    vms = []
    for c in centres:
        fwd = -c / np.linalg.norm(c)
        right = np.cross([0, 0, 1.0], fwd)
        right /= np.linalg.norm(right)
        R = np.stack([right, np.cross(fwd, right), fwd])
        vm = np.eye(4)
        vm[:3, :3], vm[:3, 3] = R, -R @ c
        vms.append(vm)
    return np.array(vms), centres


def _write_splat(path, xyz, rgb, opacity):
    """Minimal 3DGS-style splat PLY: means + SH DC colour + opacity logit."""
    fields = [(f, "f4") for f in (
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")]
    arr = np.zeros(len(xyz), dtype=fields)
    arr["x"], arr["y"], arr["z"] = xyz.T
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = ((rgb - 0.5) / SH_C0).T
    arr["opacity"] = np.log(opacity / (1.0 - opacity))
    arr["rot_0"] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def test_cam_centres_from_sfm_inverts_viewmats(tmp_path):
    vms, centres = _orbit()
    npz = tmp_path / "sfm_viewmats_x.npz"
    np.savez(npz, names=np.array(["a"]), viewmats=vms)

    np.testing.assert_allclose(process_mesh.cam_centres_from_sfm(npz), centres,
                               atol=1e-9)


def test_cam_centres_from_sfm_rejects_a_file_without_poses(tmp_path):
    npz = tmp_path / "not_poses.npz"
    np.savez(npz, names=np.array(["a"]))
    with pytest.raises(KeyError):
        process_mesh.cam_centres_from_sfm(npz)


@pytest.fixture
def capture(tmp_path):
    """A project holding one ingested phone capture, its SfM poses and a
    trained splat — the state the mesh job expects to find."""
    ts = "CupPhotos_20260101000000"
    scan = tmp_path / "scans" / ts
    (scan / "images").mkdir(parents=True)
    (scan / "manifest.csv").write_text("name,key_ns\n0000.jpg,0\n")

    vms, _ = _orbit()
    (tmp_path / "sfm").mkdir()
    np.savez(tmp_path / "sfm" / f"sfm_viewmats_{ts}.npz",
             names=np.array(["0000.jpg"]), viewmats=vms)

    pts, rgb = _sphere(20_000)
    _write_splat(tmp_path / "splats" / f"splat_{ts}.ply", pts, rgb,
                 np.full(len(pts), 0.9))
    return tmp_path, scan, ts


def _run_mesh(project, scan, options=None):
    """Drive _job_mesh to completion; return (events, queue-drained)."""
    async def go():
        queue = asyncio.Queue()
        await lidar_jobs._job_mesh(str(project), str(scan), options or {}, queue)
        out = []
        while not queue.empty():
            out.append(queue.get_nowait())
        return out

    return asyncio.run(go())


def test_mesh_job_exports_the_splat_and_reconstructs(capture):
    project, scan, ts = capture
    events = _run_mesh(project, scan, {"depth": 6, "smooth": 5})

    assert not [e for e in events if e["event"] == "error"], events

    # The gaussian centres were exported on the way through, without the user
    # having to run the Library's export button first.
    cloud = project / "pointclouds" / f"splat_{ts}_points.ply"
    assert cloud.exists()
    assert any("UNITLESS" in e.get("message", "") for e in events)

    mesh = o3d.io.read_triangle_mesh(str(project / "meshes" / f"mesh_{ts}.ply"))
    verts = np.asarray(mesh.vertices)
    assert len(verts) > 1000
    # It reconstructed the sphere the gaussians described, at its own scale.
    radius = np.linalg.norm(verts, axis=1)
    assert 0.9 < radius.mean() < 1.1
    # Colour came from the cloud (no bag was projected), and tracks direction.
    colours = np.asarray(mesh.vertex_colors)
    assert (colours.sum(1) > 0).all()
    expected = verts / radius[:, None] * 0.5 + 0.5
    assert np.abs(colours - expected).mean() < 0.05


def _points_cloud(project, ts):
    """Drive the seed-cloud resolver alone — no Poisson, which is seconds of
    CPU and proves nothing about which cloud was chosen."""
    async def go():
        queue = asyncio.Queue()
        cloud = await lidar_jobs._splat_points_cloud(Path(project), ts, queue)
        out = []
        while not queue.empty():
            out.append(queue.get_nowait())
        return cloud, out

    return asyncio.run(go())


def test_splat_export_is_reused_when_fresh_and_rebuilt_when_stale(capture):
    project, _scan, ts = capture
    cloud, events = _points_cloud(project, ts)
    assert cloud == project / "pointclouds" / f"splat_{ts}_points.ply"
    assert cloud.exists()
    assert any("gaussians kept" in e.get("message", "") for e in events)
    stamp = cloud.stat().st_mtime_ns

    _cloud, events = _points_cloud(project, ts)
    assert cloud.stat().st_mtime_ns == stamp  # not re-exported
    assert any("Reusing exported splat cloud" in e.get("message", "")
               for e in events)

    # Re-training the splat invalidates the export: the centres moved.
    splat = project / "splats" / f"splat_{ts}.ply"
    os.utime(splat, (time.time() + 10, time.time() + 10))
    _points_cloud(project, ts)
    assert cloud.stat().st_mtime_ns != stamp


def test_no_splat_and_no_sparse_model_rebuilds_rather_than_dead_ending(capture):
    """The case that used to strand a user: an npz too old to carry a sparse
    model, so stereo cannot run, and no splat to fall back on either. Rebuild
    it — that is the only route to a mesh, and telling them to train a splat
    would be hours of GPU for a side effect."""
    project, scan, ts = capture
    (project / "splats" / f"splat_{ts}.ply").unlink()

    events = _run_mesh(project, scan)
    msgs = [e.get("message", "") for e in events]
    assert any("predates dense stereo" in m for m in msgs), msgs
    assert not any("train one first" in m for m in msgs)


def test_an_existing_splat_is_not_thrown_away_for_a_rebuild(capture):
    """Rebuilding replaces the poses the splat was fitted against and costs
    hours, so a capture that can still mesh the fast way does."""
    project, scan, ts = capture  # fixture has a splat, npz, but no sparse model

    events = _run_mesh(project, scan, {"depth": 6, "smooth": 5})
    msgs = [e.get("message", "") for e in events]
    assert any("mesh will come from the splat's points" in m for m in msgs), msgs
    assert not any("running SfM now" in m for m in msgs)
    assert not [e for e in events if e["event"] == "error"], events


def test_mesh_job_runs_sfm_itself_when_there_are_no_poses(capture):
    """It used to send the user off to train a splat first. The mesh comes
    from dense stereo now, so that would be hours of GPU for a side effect —
    the job runs the reconstruction it actually needs instead."""
    project, scan, ts = capture
    (project / "sfm" / f"sfm_viewmats_{ts}.npz").unlink()

    events = _run_mesh(project, scan)
    msgs = [e.get("message", "") for e in events]
    assert any("running SfM now" in m for m in msgs), msgs
    assert not any("Train a splat" in m for m in msgs)
    # This fixture has no real frames, so the reconstruction fails — the point
    # is that the failure comes from SfM, not from a refusal to attempt it.
    errors = [e["message"] for e in events if e["event"] == "error"]
    assert errors and "Train a splat" not in errors[0]


def test_mesh_job_requires_ingest_first(tmp_path):
    """A raw camera roll is not meshable — it has to go through ingest, which
    is what the splat (and so the mesh) is built from."""
    roll = tmp_path / "CupPhotos"
    roll.mkdir()
    for i in range(6):
        (roll / f"IMG_{i}.jpg").write_bytes(b"")

    events = _run_mesh(tmp_path / "proj", roll)
    errors = [e["message"] for e in events if e["event"] == "error"]
    assert errors and "has not been ingested" in errors[0]


def test_force_sfm_rebuilds_even_when_a_sparse_model_exists(capture):
    """It used to be consulted only when the sparse model was MISSING, so once
    one existed the box did nothing — and a reconstruction that is wrong rather
    than absent is exactly when it gets ticked."""
    project, scan, ts = capture
    (project / "sfm" / f"sfm_viewmats_{ts}_sparse").mkdir()

    events = _run_mesh(project, scan, {"force_sfm": True})
    msgs = [e.get("message", "") for e in events]
    assert any("running SfM now" in m or "rebuilt" in m for m in msgs), msgs
