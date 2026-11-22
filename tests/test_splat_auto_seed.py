"""A LiDAR splat on a project with no point cloud builds the coloured seed
itself instead of failing on a missing trajectory sidecar."""
import asyncio

import numpy as np

from lidarstudio import lidar_jobs


def test_splat_builds_missing_seed_cloud_first(tmp_path, monkeypatch):
    ts = "20261122201854"
    scan = tmp_path / "scans" / ts
    scan.mkdir(parents=True)
    (scan / f"LIDAR_{ts}.bag").write_bytes(b"")
    (scan / f"IMAGE_{ts}.bag").write_bytes(b"")
    pc = tmp_path / "pointclouds"
    launched = []

    async def fake_run(scan_, contents, output_file, options, queue, job, label):
        launched.append((output_file, options))
        # What process_pointcloud.py leaves behind: cloud + trajectory sidecar.
        output_file.write_bytes(b"ply\n")
        np.savez(str(output_file) + ".traj.npz", ts=np.zeros(1), poses=np.eye(4)[None])
        return True

    async def fake_exec(*cmd, **kw):
        launched.append(cmd)
        raise RuntimeError("stop: past seed selection")

    monkeypatch.setattr(lidar_jobs, "_run_lidar_pointcloud", fake_run)
    monkeypatch.setattr(lidar_jobs.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(lidar_jobs, "_sfm_python", lambda *a, **k: "/usr/bin/python3")
    monkeypatch.setattr(lidar_jobs, "_splat_python", lambda *a, **k: "/usr/bin/python3")
    monkeypatch.setattr("lidarstudio.edit_ops.is_monochrome", lambda p: False)

    async def go():
        queue = asyncio.Queue()
        try:
            await lidar_jobs._job_splat(str(tmp_path), str(scan), {}, queue)
        except RuntimeError:
            pass
        out = []
        while not queue.empty():
            out.append(queue.get_nowait())
        return out

    events = asyncio.run(go())
    assert not [e for e in events if e.get("event") == "error"], events
    out, opts = launched[0]
    assert out == pc / f"pointcloud_{ts}.ply"
    assert not opts.get("mono"), "the auto seed must be the coloured cloud"
    assert opts.get("voxel_size") == 0.01, "the auto seed must match the GUI's 1 cm"
    sfm = [c for c in launched[1:] if "process_sfm.py" in str(c)]
    assert sfm and str(out) + ".traj.npz" in sfm[0], "SfM must align to the new cloud"
