"""Catching a solve that registers nearly every frame but is still wrong.

On the 11-08 Clinic video incremental mapping placed 1037 of 1052 frames, yet
the camera path teleported up to 12 m and flipped 179 deg between neighbouring
frames: the walk-through had been glued together from misaligned pieces.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lidarstudio import lidar_jobs

sys.path.insert(0, str(Path(lidar_jobs.__file__).parent))
import process_sfm  # noqa: E402


def _walk(n=100, dt_s=0.25, yaw_deg_per_frame=2.0, seed=0):
    """A smooth handheld walk: centres, yaws (deg) and timestamps (ns)."""
    rng = np.random.default_rng(seed)
    centres = np.cumsum(np.c_[np.full(n, 0.05), rng.normal(0, 0.005, (n, 2))],
                        axis=0)
    yaws = np.arange(n) * yaw_deg_per_frame
    ts = (np.arange(n) * dt_s * 1e9).astype(np.int64)
    return centres, yaws, ts


def _viewmats(centres, yaws):
    out = []
    for c, y in zip(centres, yaws):
        R = Rotation.from_euler("y", y, degrees=True).as_matrix()  # world→cam
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = -R @ c
        out.append(m)
    return np.array(out)


def _names(n):
    return [f"{i:04d}.jpg" for i in range(n)]


def test_smooth_walk_has_no_jumps():
    c, y, ts = _walk()
    assert process_sfm.pose_jumps(_names(len(c)), _viewmats(c, y), ts) == []


def test_piece_shifted_sideways_is_a_jump():
    c, y, ts = _walk()
    c[60:] += [0.0, 0.0, 3.0]  # the second half glued on 3 m to the side
    jumps = process_sfm.pose_jumps(_names(len(c)), _viewmats(c, y), ts)
    assert [(a, b) for a, b, _, _ in jumps] == [("0059.jpg", "0060.jpg")]


def test_piece_flipped_round_is_a_jump():
    c, y, ts = _walk()
    y[40:] += 170.0  # same place, facing backwards
    jumps = process_sfm.pose_jumps(_names(len(c)), _viewmats(c, y), ts)
    assert [(a, b) for a, b, _, _ in jumps] == [("0039.jpg", "0040.jpg")]
    assert jumps[0][3] > 160


def test_lone_misplaced_frame_is_reported_once():
    c, y, ts = _walk()
    c[30] += [2.0, 0.0, 0.0]
    jumps = process_sfm.pose_jumps(_names(len(c)), _viewmats(c, y), ts)
    assert [(a, b) for a, b, _, _ in jumps] == [("0030.jpg", "0030.jpg")]


def test_long_move_over_a_long_time_gap_is_not_a_jump():
    # Frames skipped as blank: 4 s of walking happens between two kept frames.
    c, y, ts = _walk()
    c[50:] += [0.8, 0.0, 0.0]  # 16 steps' worth of walking...
    ts[50:] += int(4e9)  # ...over 4 s, i.e. at the usual pace
    assert process_sfm.pose_jumps(_names(len(c)), _viewmats(c, y), ts) == []


def test_quick_pan_is_not_a_flip():
    # A 45 deg turn between frames is fast, but within what a hand can do.
    c, y, ts = _walk()
    y[70:] += 45.0
    assert process_sfm.pose_jumps(_names(len(c)), _viewmats(c, y), ts) == []


def test_order_comes_from_timestamps_not_list_order():
    c, y, ts = _walk()
    perm = np.random.default_rng(1).permutation(len(c))
    names = [_names(len(c))[i] for i in perm]
    vm = _viewmats(c, y)[perm]
    assert process_sfm.pose_jumps(names, vm, ts[perm]) == []


def test_warning_names_the_frames(capsys):
    c, y, ts = _walk()
    c[60:] += [0.0, 0.0, 3.0]
    names = _names(len(c))
    process_sfm.warn_pose_jumps(names, _viewmats(c, y), dict(zip(names, ts)))
    out = capsys.readouterr().out
    assert out.startswith("WARNING:")
    assert "0059.jpg → 0060.jpg" in out
