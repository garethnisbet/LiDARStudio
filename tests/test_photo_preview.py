"""'Load photos' on a phone capture, before and after ingest.

A raw camera roll previews its own photos — that is how you decide whether a
capture is worth an hour of SfM. Once ingested, the frames that matter are the
normalized ones in the project, and for a video they are the only ones that can
be shown at all.
"""

import csv
import json

import pytest

from lidarstudio import lidar_jobs


def _ingested(project, source, names):
    """Write the ingest output for ``source`` the way process_ingest does.

    ``ingest.json`` is what makes the capture folder resolve back to this
    directory, so it has to be written for real: it records the file that was
    actually decoded, which for a video is the .mp4 itself, not its folder.
    """
    out = lidar_jobs._ingest_dir(project, source)
    (out / "images").mkdir(parents=True, exist_ok=True)
    videos = [f for f in source.iterdir() if f.suffix.lower() in lidar_jobs._VIDEO_EXT]
    (out / "ingest.json").write_text(json.dumps({
        "source": str(videos[0] if videos else source),
        "source_kind": "video" if videos else "stills",
        "frames": len(names),
    }))
    for n in names:
        (out / "images" / n).write_bytes(b"jpegbytes")
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "key_ns"])
        w.writeheader()
        for i, n in enumerate(names):
            w.writerow({"name": n, "key_ns": i})
    return out


@pytest.fixture(autouse=True)
def _clear_index():
    lidar_jobs._photo_index.clear()
    yield
    lidar_jobs._photo_index.clear()


@pytest.fixture
def video_capture(tmp_path):
    roll = tmp_path / "roll" / "CupPhotos"
    roll.mkdir(parents=True)
    (roll / "clip.mp4").write_bytes(b"\x00" * 64)
    return tmp_path / "project", roll


def test_a_video_cannot_be_previewed_before_ingest(video_capture):
    project, roll = video_capture
    with pytest.raises(FileNotFoundError, match="run 'ingest'"):
        lidar_jobs._scan_photo_index(roll, project)


def test_after_ingest_the_generated_frames_are_previewed(video_capture):
    project, roll = video_capture
    _ingested(project, roll, [f"{i:04d}.jpg" for i in range(30)])

    src, topic, names = lidar_jobs._scan_photo_index(roll, project)
    assert len(names) == 30 and names[0] == "0000.jpg"
    assert src.name == "images" and "CupPhotos" in topic


def test_without_the_project_there_is_nothing_to_resolve_against(video_capture):
    """The frames live in the project, so the preview cannot find them from the
    capture folder alone — the message must still say what to do."""
    project, roll = video_capture
    _ingested(project, roll, [f"{i:04d}.jpg" for i in range(30)])
    with pytest.raises(FileNotFoundError, match="run 'ingest'"):
        lidar_jobs._scan_photo_index(roll)


def test_re_ingesting_at_another_frame_rate_is_not_served_from_cache(video_capture):
    """The frame list changes under a fixed path; a cache keyed on the path
    alone would keep showing the previous run's frames."""
    project, roll = video_capture
    _ingested(project, roll, [f"{i:04d}.jpg" for i in range(30)])
    assert len(lidar_jobs._scan_photo_index(roll, project)[2]) == 30

    out = lidar_jobs._ingest_dir(project, roll)
    for f in (out / "images").iterdir():
        f.unlink()
    _ingested(project, roll, [f"{i:04d}.jpg" for i in range(75)])

    assert len(lidar_jobs._scan_photo_index(roll, project)[2]) == 75


def test_a_raw_photo_folder_still_previews_its_own_photos(tmp_path):
    roll = tmp_path / "Photos"
    roll.mkdir()
    for i in range(8):
        (roll / f"IMG_{i:03d}.jpg").write_bytes(b"jpegbytes")

    src, _topic, names = lidar_jobs._scan_photo_index(roll, tmp_path / "project")
    assert len(names) == 8 and src == roll


def test_pointing_straight_at_the_ingested_folder_works(video_capture):
    project, roll = video_capture
    out = _ingested(project, roll, [f"{i:04d}.jpg" for i in range(12)])

    _src, _topic, names = lidar_jobs._scan_photo_index(out)
    assert len(names) == 12


def test_frames_missing_from_disk_are_dropped_from_the_index(video_capture):
    """The manifest is written before the sharpness filter runs, so it lists
    frames the filter then deleted."""
    project, roll = video_capture
    out = _ingested(project, roll, [f"{i:04d}.jpg" for i in range(20)])
    (out / "images" / "0005.jpg").unlink()

    _src, _topic, names = lidar_jobs._scan_photo_index(roll, project)
    assert len(names) == 19 and "0005.jpg" not in names
