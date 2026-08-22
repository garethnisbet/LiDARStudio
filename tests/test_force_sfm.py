"""force_sfm: rebuild a scan's reconstruction instead of reusing the stored one.

SfM runs once per scan and every later run inherits the .npz, which is right
while it is good and a trap when it is not — a folded orbit, or a re-ingest at
a different frame rate, leaves a stale file nothing in the UI would re-run.
"""

import asyncio

import numpy as np
import pytest

from lidarstudio import lidar_jobs


@pytest.fixture
def capture(tmp_path, monkeypatch):
    """An ingested phone capture whose reconstruction already exists, with the
    SfM subprocess stubbed out — these tests are about which branch runs, not
    about COLMAP."""
    ts = "CupPhotos"
    scan = tmp_path / "scans" / ts
    (scan / "images").mkdir(parents=True)
    (scan / "manifest.csv").write_text("name,key_ns\n0000.jpg,0\n")

    (tmp_path / "sfm").mkdir()
    npz = tmp_path / "sfm" / f"sfm_viewmats_{ts}.npz"
    np.savez(npz, names=np.array(["0000.jpg"]), viewmats=np.eye(4)[None])
    (tmp_path / "pointclouds").mkdir()
    (tmp_path / "pointclouds" / f"pointcloud_{ts}.ply").write_bytes(b"ply\n")

    launched = []

    async def fake_exec(*cmd, **kw):
        launched.append(cmd)
        raise RuntimeError("stop here: the SfM decision has already been made")

    monkeypatch.setattr(lidar_jobs.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(lidar_jobs, "_sfm_python", lambda *a, **k: "/usr/bin/python3")
    return tmp_path, scan, npz, launched


def _run(project, scan, options):
    async def go():
        queue = asyncio.Queue()
        try:
            await lidar_jobs._job_splat(str(project), str(scan), options, queue)
        except RuntimeError:
            pass  # the stub fired: a subprocess was about to launch
        out = []
        while not queue.empty():
            out.append(queue.get_nowait())
        return out

    return asyncio.run(go())


def _messages(events):
    return " | ".join(e.get("message", "") for e in events)


def test_without_the_option_the_stored_reconstruction_is_reused(capture):
    project, scan, _npz, launched = capture
    events = _run(project, scan, {})
    assert "Reusing the existing reconstruction" in _messages(events)
    assert not any("process_sfm.py" in str(c) for c in launched)


def test_the_option_rebuilds_it(capture):
    project, scan, _npz, launched = capture
    events = _run(project, scan, {"force_sfm": True})
    assert "Rebuilding the reconstruction from scratch" in _messages(events)
    assert any("process_sfm.py" in str(c) for c in launched)


def test_an_explicit_poses_path_wins_over_the_option(capture):
    """A path typed into the field is an explicit choice; regenerating would
    overwrite the project npz and then ignore what the user asked for."""
    project, scan, npz, launched = capture
    events = _run(project, scan, {"force_sfm": True, "sfm_poses": str(npz)})
    assert "'Force SfM regeneration' ignored" in _messages(events)
    assert not any("process_sfm.py" in str(c) for c in launched)


def test_a_missing_poses_path_still_fails_loudly(capture):
    project, scan, _npz, _launched = capture
    events = _run(project, scan, {"force_sfm": True, "sfm_poses": "/nope/x.npz"})
    assert [e for e in events if e["event"] == "error"]


def _sfm_cmd(launched):
    return next(c for c in launched if "process_sfm.py" in str(c))


def test_subject_masking_reaches_the_sfm_command(capture):
    """The mask lever is only real if it survives the trip from the UI option
    to the process_sfm.py argv."""
    project, scan, _npz, launched = capture
    _run(project, scan, {"force_sfm": True, "mask_subject": True})
    assert "--mask-subject" in _sfm_cmd(launched)


def test_a_named_subject_class_is_passed_through(capture):
    project, scan, _npz, launched = capture
    _run(project, scan,
         {"force_sfm": True, "mask_subject": True, "mask_classes": "cup,bowl"})
    cmd = _sfm_cmd(launched)
    assert cmd[cmd.index("--mask-classes") + 1] == "cup,bowl"


def test_masking_is_off_unless_asked_for(capture):
    """Segmentation costs a model load and a pass over every frame, and on a
    capture with no clear subject it would mask the wrong thing."""
    project, scan, _npz, launched = capture
    _run(project, scan, {"force_sfm": True})
    cmd = _sfm_cmd(launched)
    assert "--mask-subject" not in cmd and "--mask-classes" not in cmd


def test_naming_classes_without_ticking_the_box_does_nothing(capture):
    project, scan, _npz, launched = capture
    _run(project, scan, {"force_sfm": True, "mask_classes": "cup"})
    assert "--mask-classes" not in _sfm_cmd(launched)


def test_turntable_reaches_the_sfm_command_and_implies_masking(capture):
    """--turntable is not a variation on --mask-subject; it replaces it,
    because solving in the subject's frame is the whole mechanism."""
    project, scan, _npz, launched = capture
    _run(project, scan, {"force_sfm": True, "turntable": True})
    cmd = _sfm_cmd(launched)
    assert "--turntable" in cmd
    assert "--mask-subject" not in cmd  # implied, not doubled up


def test_the_masks_a_reconstruction_used_reach_training(capture, tmp_path):
    """Poses solved on the subject alone must be trained on the subject
    alone, or the trainer burns itself fitting a background those cameras
    were never solved against."""
    project, scan, npz, launched = capture
    masks = tmp_path / "sfm" / "sfm_viewmats_CupPhotos_masks"
    masks.mkdir(parents=True)
    np.savez(
        npz,
        names=np.array(["0000.jpg"]),
        viewmats=np.eye(4)[None],
        mask_dir=np.array(str(masks)),
    )
    events = _run(project, scan, {})
    cmd = next(c for c in launched if "process_splat.py" in str(c))
    assert cmd[cmd.index("--frame-masks") + 1] == str(masks)
    assert "Training on the subject only" in _messages(events)


def test_an_unmasked_reconstruction_passes_no_mask_directory(capture):
    project, scan, _npz, launched = capture
    _run(project, scan, {})
    assert "--frame-masks" not in next(
        c for c in launched if "process_splat.py" in str(c)
    )


def test_a_mask_directory_that_no_longer_exists_is_ignored(capture, tmp_path):
    """Masks live beside the npz and can be deleted independently; a stale
    path must not take the whole training run down with it."""
    project, scan, npz, launched = capture
    np.savez(
        npz,
        names=np.array(["0000.jpg"]),
        viewmats=np.eye(4)[None],
        mask_dir=np.array(str(tmp_path / "gone")),
    )
    _run(project, scan, {})
    assert "--frame-masks" not in next(
        c for c in launched if "process_splat.py" in str(c)
    )
