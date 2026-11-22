"""Lost-track detection for the LiDAR odometry (Clinic scan 20261122201854)."""
import numpy as np

from lidarstudio.process_pointcloud import odometry_jumps


def _poses(xyz):
    out = []
    for p in xyz:
        m = np.eye(4)
        m[:3, 3] = p
        out.append(m)
    return out


def test_walking_pace_has_no_jumps():
    # 0.05 m per 0.1 s sweep = 0.5 m/s, a brisk handheld walk.
    path = np.cumsum(np.full((200, 3), 0.05 / np.sqrt(3)), axis=0)
    assert len(odometry_jumps(_poses(path))) == 0


def test_lost_track_is_flagged_at_the_sweep_it_happens():
    path = np.zeros((50, 3))
    path[30:] += [2.0, 0, 0]  # a 2 m teleport between sweeps 29 and 30
    assert list(odometry_jumps(_poses(path))) == [30]


def test_too_short_to_judge():
    assert len(odometry_jumps(_poses(np.zeros((1, 3))))) == 0
