"""A LiDAR cloud whose odometry tilted off gravity is rebuilt from the SfM
camera poses, and the splat then trains on the pose file in that cloud's
frame (2026-11-22 Clinic: 35° slip → two misaligned copies of the room)."""
import asyncio
import os

import numpy as np
import pytest

from lidarstudio import lidar_jobs
from lidarstudio.process_pointcloud import save_trajectory

TS = "20261122201854"


def _drain(queue):
    out = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


@pytest.fixture
def project(tmp_path):
    scan = tmp_path / "scans" / TS
    scan.mkdir(parents=True)
    (scan / f"LIDAR_{TS}.bag").write_bytes(b"")
    (scan / f"IMAGE_{TS}.bag").write_bytes(b"")
    (tmp_path / "pointclouds").mkdir()
    (tmp_path / "sfm").mkdir()
    return tmp_path, scan


def _fake_cloud_runs(monkeypatch, tilt):
    """process_pointcloud stand-in: odometry run records ``tilt``; a run with
    --sfm-poses records the relevelled pose file, as the real one does."""
    runs = []

    async def fake_run(scan, contents, out, options, queue, job, label, sfm_poses=None):
        runs.append(sfm_poses)
        out.write_bytes(b"ply\n")
        rel = None
        if sfm_poses:
            rel = sfm_poses.with_name(sfm_poses.stem + "_relevelled.npz")
            np.savez(rel, viewmats=np.eye(4)[None])
        save_trajectory(out, [0], [np.eye(4)], odometry_tilt=tilt, sfm_poses=rel)
        return True

    monkeypatch.setattr(lidar_jobs, "_run_lidar_pointcloud", fake_run)
    return runs


def _build(proj, scan, options=None):
    async def go():
        q = asyncio.Queue()
        ok = await lidar_jobs._build_lidar_cloud(
            proj, scan, lidar_jobs._scan_contents(scan), TS,
            proj / "pointclouds" / f"pointcloud_{TS}.ply", options or {}, q, {},
            "Point cloud generation")
        return ok, _drain(q)
    return asyncio.run(go())


def test_level_odometry_builds_once(project, monkeypatch):
    proj, scan = project
    runs = _fake_cloud_runs(monkeypatch, tilt=4.0)
    ok, _ = _build(proj, scan)
    assert ok and runs == [None]


def test_tilted_odometry_rebuilds_from_existing_poses(project, monkeypatch):
    proj, scan = project
    canonical = proj / "sfm" / f"sfm_viewmats_{TS}.npz"
    np.savez(canonical, viewmats=np.eye(4)[None])
    runs = _fake_cloud_runs(monkeypatch, tilt=32.6)

    async def no_sfm(*a, **k):
        raise AssertionError("poses exist: SfM must not re-run")

    monkeypatch.setattr(lidar_jobs, "_run_lidar_sfm", no_sfm)
    ok, events = _build(proj, scan)
    assert ok and runs == [None, canonical]
    assert any("tilted 33°" in e.get("message", "") for e in events)
    meta = lidar_jobs._traj_meta(proj / "pointclouds" / f"pointcloud_{TS}.ply")
    assert meta["sfm_poses"].endswith("_relevelled.npz")


def test_tilted_odometry_solves_poses_first(project, monkeypatch):
    proj, scan = project
    runs = _fake_cloud_runs(monkeypatch, tilt=32.6)
    solved = []

    async def fake_sfm(proj_, scan_, ts, traj, options, queue, job, force, label):
        solved.append(traj)
        out = proj_ / "sfm" / f"sfm_viewmats_{ts}.npz"
        np.savez(out, viewmats=np.eye(4)[None])
        return out, None

    monkeypatch.setattr(lidar_jobs, "_run_lidar_sfm", fake_sfm)
    ok, _ = _build(proj, scan)
    cloud = proj / "pointclouds" / f"pointcloud_{TS}.ply"
    assert ok and solved == [cloud.with_name(cloud.name + ".traj.npz")]
    assert runs == [None, proj / "sfm" / f"sfm_viewmats_{TS}.npz"]


def test_failed_sfm_keeps_the_odometry_cloud_with_a_warning(project, monkeypatch):
    proj, scan = project
    runs = _fake_cloud_runs(monkeypatch, tilt=32.6)

    async def no_colmap(*a, **k):
        return None, "No Python with pycolmap was found"

    monkeypatch.setattr(lidar_jobs, "_run_lidar_sfm", no_colmap)
    ok, events = _build(proj, scan)
    assert ok and runs == [None]
    assert any("WARNING" in e.get("message", "") for e in events)
    assert not [e for e in events if e.get("event") == "error"]


def test_fallback_can_be_turned_off(project, monkeypatch):
    proj, scan = project
    runs = _fake_cloud_runs(monkeypatch, tilt=32.6)
    ok, _ = _build(proj, scan, {"sfm_fallback": False})
    assert ok and runs == [None]


def test_placed_poses_yield_to_a_newer_forced_sfm(project):
    proj, _ = project
    cloud = proj / "pointclouds" / f"pointcloud_{TS}.ply"
    canonical = proj / "sfm" / f"sfm_viewmats_{TS}.npz"
    rel = proj / "sfm" / f"sfm_viewmats_{TS}_relevelled.npz"
    np.savez(canonical, viewmats=np.eye(4)[None])
    np.savez(rel, viewmats=np.eye(4)[None])
    save_trajectory(cloud, [0], [np.eye(4)], odometry_tilt=32.6, sfm_poses=rel)
    os.utime(canonical, (1, 1))
    assert lidar_jobs._placed_poses([cloud]) == rel
    # An edited copy carries the sidecar, and with it the pose file.
    edited = cloud.with_name(cloud.stem + "_edited.ply")
    assert lidar_jobs._placed_poses([edited]) == rel
    os.utime(canonical, (rel.stat().st_mtime + 10,) * 2)
    assert lidar_jobs._placed_poses([cloud]) is None


def test_splat_trains_on_the_poses_the_seed_was_placed_from(project, monkeypatch):
    proj, scan = project
    cloud = proj / "pointclouds" / f"pointcloud_{TS}.ply"
    cloud.write_bytes(b"ply\n")
    canonical = proj / "sfm" / f"sfm_viewmats_{TS}.npz"
    rel = proj / "sfm" / f"sfm_viewmats_{TS}_relevelled.npz"
    np.savez(canonical, viewmats=np.eye(4)[None])
    os.utime(canonical, (1, 1))
    np.savez(rel, viewmats=np.eye(4)[None])
    save_trajectory(cloud, [0], [np.eye(4)], odometry_tilt=32.6, sfm_poses=rel)
    launched = []

    async def fake_exec(*cmd, **kw):
        launched.append(cmd)
        raise RuntimeError("stop: at the training launch")

    monkeypatch.setattr(lidar_jobs.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(lidar_jobs, "_splat_python", lambda *a, **k: "/usr/bin/python3")
    monkeypatch.setattr("lidarstudio.edit_ops.is_monochrome", lambda p: False)

    async def go():
        q = asyncio.Queue()
        try:
            await lidar_jobs._job_splat(str(proj), str(scan), {}, q)
        except RuntimeError:
            pass
        return _drain(q)

    events = asyncio.run(go())
    assert not [e for e in events if e.get("event") == "error"], events
    cmd = [c for c in launched if "process_splat.py" in str(c)][0]
    assert cmd[cmd.index("--sfm-poses") + 1] == str(rel)
