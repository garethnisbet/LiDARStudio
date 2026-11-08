"""Smart video frame selection: keep frames by how much the view changes."""

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from lidarstudio import process_ingest as pi  # noqa: E402

W, H = 320, 180


def _texture(seed, w=2400, h=H):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 255, (h // 6, w // 6), dtype=np.uint8)
    return cv2.resize(t, (w, h), interpolation=cv2.INTER_NEAREST)


def _sharp(g):
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _seq(xs, tex):
    """Crops of ``tex`` at the given x offsets: a camera panning along it."""
    for x in xs:
        g = np.ascontiguousarray(tex[:, x:x + W])
        yield g, _sharp(g)


def test_a_still_camera_adds_nothing():
    tex = _texture(0)
    kept = pi.select_keyframes(_seq([100] * 60, tex), spacing=0.05)
    assert kept == [0]


def test_a_steady_pan_is_spaced_evenly_and_a_pause_is_skipped():
    tex = _texture(1)
    # 4 px a frame for 60 frames, still for 60, then 4 px a frame for 60 more.
    xs = [4 * i for i in range(60)] + [236] * 60 + [236 + 4 * i for i in range(1, 61)]
    kept = pi.select_keyframes(_seq(xs, tex), spacing=0.05)
    # 5% of a 320 px edge is 16 px, i.e. a kept frame every ~4 frames of pan.
    gaps = np.diff(kept)
    assert 25 <= len(kept) <= 40
    assert not any(60 < k < 119 for k in kept[1:-1]), "the pause should add no frames"
    moving = [g for a, g in zip(kept, gaps) if a < 55 or a > 125]
    assert max(moving) <= 5


def test_smaller_spacing_keeps_more_frames():
    tex = _texture(2)
    xs = [3 * i for i in range(120)]
    fine = pi.select_keyframes(_seq(xs, tex), spacing=0.03)
    coarse = pi.select_keyframes(_seq(xs, tex), spacing=0.10)
    assert len(fine) > 2 * len(coarse)


def test_a_cut_keeps_the_frames_either_side_of_it():
    a, b = _texture(3), _texture(4)
    frames = list(_seq([2 * i for i in range(30)], a))
    frames += list(_seq([2 * i for i in range(30)], b))
    kept = pi.select_keyframes(iter(frames), spacing=0.5)   # huge: only the cut can trigger
    assert 29 in kept and 30 in kept


def test_the_sharpest_frame_of_each_step_is_kept():
    tex = _texture(5)
    xs = [2 * i for i in range(80)]
    frames = list(_seq(xs, tex))
    blurred = set(range(0, 80, 2)) - {0}
    frames = [
        ((g2 := cv2.GaussianBlur(g, (0, 0), 2.0)), _sharp(g2)) if i in blurred else (g, s)
        for i, (g, s) in enumerate(frames)
    ]
    kept = pi.select_keyframes(iter(frames), spacing=0.05)
    assert len(kept) > 5
    assert not (set(kept) & blurred)


def _wall(seed=0):
    """A plain wall as a phone sees it: flat, with faint sensor noise."""
    rng = np.random.default_rng(seed)
    return np.clip(128 + rng.normal(0, 1.5, (H, W)), 0, 255).astype(np.uint8)


def test_a_blank_wall_is_recognised_and_texture_is_not():
    assert pi.is_blank(_wall())
    assert not pi.is_blank(_texture(7)[:, :W])


def test_blank_frames_are_never_kept():
    stats = {}
    frames = [(_wall(k), 0.5) for k in range(40)]
    assert pi.select_keyframes(iter(frames), spacing=0.05, max_gap=8,
                               stats=stats) == []
    assert stats["blank"] == 40


def test_a_blank_stretch_is_bridged_by_the_frames_either_side():
    tex = _texture(8)
    frames = list(_seq([2 * i for i in range(20)], tex))          # 0-19 pan
    frames += [(_wall(k), 0.5) for k in range(20)]                # 20-39 wall
    frames += list(_seq([40 + 2 * i for i in range(20)], tex))    # 40-59 pan
    stats = {}
    kept = pi.select_keyframes(iter(frames), spacing=0.05, max_gap=8, stats=stats)
    assert 19 in kept and 40 in kept
    assert not any(20 <= k < 40 for k in kept)
    assert stats["blank"] == 20


def test_too_little_detail_to_track_falls_back_to_a_steady_rate():
    # Three dark squares on a wall: real corners (not blank), but too few to
    # measure motion from.
    g = _wall()
    for x in (60, 150, 240):
        g[70:110, x:x + 30] = 20
    assert not pi.is_blank(g)
    frames = [(g, _sharp(g))] * 40
    kept = pi.select_keyframes(iter(frames), spacing=0.05, max_gap=8)
    assert kept == [0, 8, 16, 24, 32]


def test_ingest_of_a_video_keeps_only_the_frames_it_selected(tmp_path):
    tex = _texture(6, w=3200, h=360)
    video = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    # Still for 1 s, then pan 8 px a frame for 3 s: the still second adds nothing.
    xs = [0] * 30 + [8 * i for i in range(1, 91)]
    for x in xs:
        vw.write(cv2.cvtColor(np.ascontiguousarray(tex[:, x:x + 640]), cv2.COLOR_GRAY2BGR))
    vw.release()

    out = tmp_path / "scan"
    meta = pi.ingest(video, out, max_edge=2400, fps=4.0, drop_blurry=0.0)
    sel = json.loads((out / "ingest.json").read_text())["selection"]
    assert sel["mode"] == "motion" and sel["kept"] == meta["frames"]
    assert 20 <= meta["frames"] <= 50
    labels = [r.split(",")[2] for r in (out / "manifest.csv").read_text().splitlines()[1:]]
    nums = [int(s.split("#")[1]) for s in labels]
    assert nums[0] == 0 and not any(0 < n < 30 for n in nums)
    assert len(list((out / "images").glob("*.jpg"))) == meta["frames"]


def test_a_blank_video_is_refused_with_a_reason(tmp_path):
    video = tmp_path / "capped.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    for _ in range(30):
        vw.write(np.full((360, 640, 3), 40, np.uint8))
    vw.release()
    with pytest.raises(SystemExit, match="enough detail"):
        pi.ingest(video, tmp_path / "scan", max_edge=2400, fps=4.0, drop_blurry=0.0)
