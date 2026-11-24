"""Include/exclude markers set in the photo viewer on an ingested capture.

The viewer works in preview indices; the file beside the manifest stores frame
names; every later stage reads the kept frames through frame_source. A changed
selection has to make the next job re-run SfM, or the old poses would still
train the frames the user cut.
"""

import asyncio
import json

import pytest
from aiohttp import web

from lidarstudio import frame_source, lidar_jobs
from test_photo_preview import _ingested

NAMES = [f"{i:04d}.jpg" for i in range(30)]


@pytest.fixture(autouse=True)
def _clear_index():
    lidar_jobs._photo_index.clear()
    yield
    lidar_jobs._photo_index.clear()


@pytest.fixture
def capture(tmp_path):
    roll = tmp_path / "roll" / "Cup"
    roll.mkdir(parents=True)
    for n in NAMES:
        (roll / n).write_bytes(b"raw")
    project = tmp_path / "project"
    return project, roll, _ingested(project, roll, NAMES)


def test_include_keeps_only_its_sections_and_exclude_cuts_gaps():
    sections = [
        {"start": "0005.jpg", "end": "0014.jpg", "mode": "include"},
        {"start": "0008.jpg", "end": "0009.jpg", "mode": "exclude"},
    ]
    kept = frame_source.apply_selection(NAMES, sections)
    assert kept == [f"{i:04d}.jpg" for i in (5, 6, 7, 10, 11, 12, 13, 14)]


def test_a_section_naming_missing_frames_is_ignored():
    sections = [{"start": "gone.jpg", "end": "0003.jpg", "mode": "include"}]
    assert frame_source.apply_selection(NAMES, sections) == NAMES


def test_every_reader_sees_only_the_kept_frames(capture):
    _project, _roll, ing = capture
    (ing / frame_source.SELECTION_FILE).write_text(json.dumps({"sections": [
        {"start": "0000.jpg", "end": "0019.jpg", "mode": "exclude"}]}))
    assert sorted(frame_source.image_set_keys(ing)) == NAMES[20:]
    assert frame_source.frame_count(ing) == 10


def test_a_corrupt_selection_does_not_empty_the_capture(capture):
    _project, _roll, ing = capture
    (ing / frame_source.SELECTION_FILE).write_text("{not json")
    assert len(frame_source.image_set_keys(ing)) == 30


def test_a_reconstruction_is_stale_once_the_markers_change(capture):
    _project, _roll, ing = capture
    npz = ing.parent.parent / "sfm" / "sfm_viewmats_Cup.npz"
    npz.parent.mkdir(parents=True)
    npz.write_bytes(b"npz")
    # Predates markers, nothing marked: still good.
    assert not lidar_jobs._selection_changed(npz, ing)
    (ing / frame_source.SELECTION_FILE).write_text(json.dumps({"sections": [
        {"start": "0000.jpg", "end": "0004.jpg", "mode": "exclude"}]}))
    assert lidar_jobs._selection_changed(npz, ing)
    # Solved from this selection (as process_sfm stamps it): good again.
    frame_source.selection_stamp(npz).write_text(frame_source.selection_digest(ing))
    assert not lidar_jobs._selection_changed(npz, ing)


def _post(route, body):
    """POST ``body`` to one of the app's routes and return (status, json),
    without needing the pytest-aiohttp plugin."""
    from aiohttp.test_utils import TestClient, TestServer

    async def go():
        app = web.Application()
        lidar_jobs.register_routes(app)
        async with TestClient(TestServer(app)) as client:
            r = await client.post(route, json=body)
            return r.status, await r.json()

    return asyncio.run(go())


def test_the_viewer_round_trips_its_markers(capture):
    project, roll, ing = capture
    body = {"path": str(roll), "project": str(project)}

    _st, data = _post("/api/scan/photos", body)
    assert data["selectable"] and data["sections"] == []

    st, data = _post("/api/scan/selection", {**body, "sections": [
        {"start": 12, "end": 3, "mode": "exclude"}]})
    assert st == 200 and data["kept"] == 20
    stored = json.loads((ing / frame_source.SELECTION_FILE).read_text())
    assert stored["sections"] == [
        {"start": "0003.jpg", "end": "0012.jpg", "mode": "exclude"}]

    _st, data = _post("/api/scan/photos", body)
    assert data["sections"] == [{"start": 3, "end": 12, "mode": "exclude"}]

    st, _ = _post("/api/scan/selection", {**body, "sections": []})
    assert st == 200
    assert not (ing / frame_source.SELECTION_FILE).exists()


def test_raw_photos_cannot_carry_markers(tmp_path):
    roll = tmp_path / "Roll"
    roll.mkdir()
    for n in NAMES:
        (roll / n).write_bytes(b"raw")
    _st, data = _post("/api/scan/photos", {"path": str(roll)})
    assert "selectable" not in data
    st, _ = _post("/api/scan/selection", {
        "path": str(roll), "sections": [{"start": 0, "end": 1, "mode": "exclude"}]})
    assert st == 400
