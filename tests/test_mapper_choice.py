"""Which reconstruction engine runs first.

GLOMAP averages rotations over every pair at once, so on a scene where many
pairs are ambiguous — a repetitive near-planar background, or a subject with
rotational symmetry — a wrong consensus wins outright and the orbit folds.
Incremental grows one image at a time and can reject a view that contradicts
what it has already built. Measured on the CupPhotos capture from one shared
exhaustive match database: GLOMAP 60 deg of solved viewpoint spread at a
degenerate 0.001 px reprojection error, incremental 90 deg at 0.877 px.
"""

import sys
from pathlib import Path

from lidarstudio import lidar_jobs

sys.path.insert(0, str(Path(lidar_jobs.__file__).parent))
import process_sfm  # noqa: E402

def test_incremental_is_the_default_mapper(monkeypatch, tmp_path):
    """Which mapper runs FIRST is the whole change: on the CupPhotos capture
    GLOMAP's global averaging solved 60 deg of viewpoint spread and a
    degenerate 0.001 px error from the same match database that incremental
    turned into 90 deg at 0.877 px."""
    calls = []

    class FakeRec:
        @staticmethod
        def num_reg_images():
            return 3

        @staticmethod
        def num_points3D():
            return 9

        @staticmethod
        def compute_mean_reprojection_error():
            return 0.9

    class FakePycolmap:
        IncrementalPipelineOptions = type(
            "I", (), {"ba_refine_focal_length": False, "ba_refine_extra_params": False}
        )
        GlobalPipelineOptions = type("G", (), {})

        @staticmethod
        def incremental_mapping(**kw):
            calls.append("incremental")
            return {0: FakeRec}

        @staticmethod
        def global_mapping(**kw):
            calls.append("glomap")
            return {0: FakeRec}

    monkeypatch.setitem(sys.modules, "pycolmap", FakePycolmap)

    process_sfm.run_mapping(tmp_path / "db", tmp_path / "im", tmp_path / "sp", "t")
    assert calls == ["incremental"], "incremental must run first, unasked"

    calls.clear()
    process_sfm.run_mapping(
        tmp_path / "db", tmp_path / "im", tmp_path / "sp", "t", mapper="glomap"
    )
    assert calls == ["glomap"], "--mapper glomap must still be honoured"


def test_the_other_mapper_is_the_fallback(monkeypatch, tmp_path):
    """A mapper that returns nothing must not sink the run."""
    calls = []

    class FakeRec:
        @staticmethod
        def num_reg_images():
            return 3

        @staticmethod
        def num_points3D():
            return 9

        @staticmethod
        def compute_mean_reprojection_error():
            return 0.9

    class FakePycolmap:
        IncrementalPipelineOptions = type(
            "I", (), {"ba_refine_focal_length": False, "ba_refine_extra_params": False}
        )
        GlobalPipelineOptions = type("G", (), {})

        @staticmethod
        def incremental_mapping(**kw):
            calls.append("incremental")
            return {}

        @staticmethod
        def global_mapping(**kw):
            calls.append("glomap")
            return {0: FakeRec}

    monkeypatch.setitem(sys.modules, "pycolmap", FakePycolmap)
    rec = process_sfm.run_mapping(tmp_path / "db", tmp_path / "im", tmp_path / "sp", "t")
    assert calls == ["incremental", "glomap"]
    assert rec is FakeRec


def test_a_phone_capture_asks_for_incremental(tmp_path, monkeypatch):
    """The two branches must NOT share a default. Incremental won on an
    ambiguous phone capture; GLOMAP is what the LiDAR champion npz was
    validated with, and is ~2x faster to map. A single default would silently
    change one of them."""
    import asyncio

    import numpy as np

    launched = []

    async def fake_exec(*cmd, **kw):
        launched.append(cmd)
        raise RuntimeError("stop: the command line is already built")

    monkeypatch.setattr(lidar_jobs.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(lidar_jobs, "_sfm_python", lambda *a, **k: "/usr/bin/python3")
    monkeypatch.setattr(lidar_jobs, "_splat_python", lambda *a, **k: "/usr/bin/python3")

    # --- phone capture: images/ + manifest.csv, no stored reconstruction
    scan = tmp_path / "scans" / "photos"
    (scan / "images").mkdir(parents=True)
    (scan / "manifest.csv").write_text("name,key_ns\n0000.jpg,0\n")

    async def run(project, scan_path, options):
        queue = asyncio.Queue()
        try:
            await lidar_jobs._job_splat(str(project), str(scan_path), options, queue)
        except RuntimeError:
            pass

    asyncio.run(run(tmp_path, scan, {}))
    cmd = next(c for c in launched if "process_sfm.py" in str(c))
    assert cmd[cmd.index("--mapper") + 1] == "incremental"

    # --- an explicit option still wins on either branch
    launched.clear()
    asyncio.run(run(tmp_path, scan, {"mapper": "glomap"}))
    cmd = next(c for c in launched if "process_sfm.py" in str(c))
    assert cmd[cmd.index("--mapper") + 1] == "glomap"


def test_a_lidar_scan_asks_for_glomap(tmp_path, monkeypatch):
    """The LiDAR branch is PINNED, not left to process_sfm's default. The
    champion npz was produced under GLOMAP, incremental's win was measured on
    a capture nothing like a room walkthrough, and GLOMAP maps ~2x faster —
    so inheriting the phone default here would be an unmeasured change to the
    one path that is already validated."""
    import asyncio

    import numpy as np

    launched = []

    async def fake_exec(*cmd, **kw):
        launched.append(cmd)
        raise RuntimeError("stop: the command line is already built")

    monkeypatch.setattr(lidar_jobs.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(lidar_jobs, "_sfm_python", lambda *a, **k: "/usr/bin/python3")
    monkeypatch.setattr(lidar_jobs, "_splat_python", lambda *a, **k: "/usr/bin/python3")

    ts = "20260527212832"
    scan = tmp_path / "scans" / ts
    scan.mkdir(parents=True)
    (scan / f"LIDAR_{ts}.bag").write_bytes(b"")
    (scan / f"IMAGE_{ts}.bag").write_bytes(b"")

    # A seed cloud WITH its trajectory sidecar: without one the branch bails
    # out early with "no .traj.npz" instead of reaching the command line.
    pc = tmp_path / "pointclouds"
    pc.mkdir()
    (pc / f"pointcloud_{ts}.ply").write_bytes(b"ply\n")
    np.savez(pc / f"pointcloud_{ts}.ply.traj.npz", ts=np.zeros(1), poses=np.eye(4)[None])

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
    sfm = [c for c in launched if "process_sfm.py" in str(c)]
    assert sfm, f"LiDAR SfM never launched: {[e.get('message') for e in events]}"
    cmd = sfm[0]
    assert cmd[cmd.index("--mapper") + 1] == "glomap"
    assert "--traj" in cmd, "the LiDAR branch must still align to the trajectory"
