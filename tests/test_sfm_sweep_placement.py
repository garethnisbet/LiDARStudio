"""Placing LiDAR sweeps from SfM camera poses (Clinic 20261122201854, where
KISS-ICP slipped 35 deg but the photo solve held)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "lidarstudio"))
pytest.importorskip("scipy")
import process_pointcloud as ppc  # noqa: E402


def _walk(n=120, dt_ns=100_000_000):
    """A wandering handheld LiDAR path: sensor→world poses + timestamps."""
    ts = np.arange(n, dtype=np.int64) * dt_ns + 10**18
    poses = []
    for i in range(n):
        P = np.eye(4)
        P[:3, :3] = Rotation.from_euler("zyx", [i * 2.0, 3 * np.sin(i / 9), 0], degrees=True).as_matrix()
        P[:3, 3] = [0.04 * i, 0.5 * np.sin(i / 15), 0.0]
        poses.append(P)
    return ts, np.array(poses)


T_LC = np.eye(4)
T_LC[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
T_LC[:3, 3] = [0.02, -0.05, 0.1]
OFF = -0.025


def _sfm_from(ts, poses, scale):
    """What process_sfm stores: world→cam viewmats at camera times, with the
    world shrunk by 1/scale (an SfM fitted to the wrong-sized trajectory)."""
    cam_ts = ts[::3] - int(OFF * 1e9)  # camera t sees LiDAR at t + OFF
    vms = []
    from process_splat import _interp_pose
    for t in cam_ts:
        P, _, _ = _interp_pose(ts, poses, int(t) + int(OFF * 1e9))
        C = P @ np.linalg.inv(T_LC)  # cam→world
        C[:3, 3] /= scale
        vms.append(np.linalg.inv(C))
    return np.array(vms), cam_ts


def test_scale_recovered_from_odometry():
    ts, poses = _walk()
    vms, cam_ts = _sfm_from(ts, poses, 1.54)
    k = ppc.sfm_scale_from_odometry(vms, cam_ts, T_LC, ts, poses, cam_time_offset=OFF)
    assert k == pytest.approx(1.54, rel=0.02)


def test_sweeps_land_where_the_lidar_was():
    ts, poses = _walk()
    vms, cam_ts = _sfm_from(ts, poses, 1.54)
    placed = ppc.sweep_poses_from_sfm(vms, cam_ts, T_LC, ts, scale=1.54, cam_time_offset=OFF)
    inner = range(5, len(ts) - 5)  # endpoints sit outside camera coverage
    err = [np.linalg.norm(placed[i][:3, 3] - poses[i][:3, 3]) for i in inner]
    assert max(err) < 0.01


def test_sweeps_far_from_any_photo_are_dropped():
    ts, poses = _walk()
    vms, cam_ts = _sfm_from(ts, poses, 1.0)
    late = np.append(ts, ts[-1] + 5_000_000_000)  # a sweep 5 s after the last photo
    placed = ppc.sweep_poses_from_sfm(vms, cam_ts, T_LC, late, cam_time_offset=OFF)
    assert placed[-1] is None


def test_tilt_slip_is_measured():
    ups = np.tile([0.0, 0.0, 1.0], (40, 1))
    poses = [np.eye(4) for _ in range(40)]
    for P in poses[25:]:  # odometry slips 35 deg about x
        P[:3, :3] = Rotation.from_euler("x", 35, degrees=True).as_matrix()
    tilt = ppc.tilt_vs_gravity(poses, ups)
    assert tilt[:25].max() < 1e-6
    assert tilt[30] == pytest.approx(35, abs=0.5)
