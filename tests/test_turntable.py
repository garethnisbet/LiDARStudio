"""Turntable mode: the subject rotated, the camera stayed put.

SfM only ever solves a static scene, so a spinning subject is reconstructible
only in the SUBJECT's frame — which is what masking to it produces. There the
capture collapses to a one-degree-of-freedom model, ``pose_i = Rot(axis,
theta_i) . pose_0``, and the frame ORDER becomes evidence: a turntable turns
one way. That is the one thing rotational symmetry cannot fake, and it catches
the fold that 60-vs-270 deg of coverage on CupPhotos showed plain matching
cannot.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lidarstudio import lidar_jobs

sys.path.insert(0, str(Path(lidar_jobs.__file__).parent))
import process_sfm  # noqa: E402


def _turntable(span_deg=300, n=120, axis=(0.1, 0.98, 0.15),
               point=(0.3, -0.2, 4.0)):
    """Poses for a perfect turntable: one fixed camera, subject spinning about
    a tilted axis that does not pass through the origin."""
    axis = np.array(axis, float)
    axis /= np.linalg.norm(axis)
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("xyz", [10, 20, 5], degrees=True).as_matrix()
    base[:3, 3] = [0.5, 0.4, -3.0]
    out = []
    for t in np.radians(np.linspace(0, span_deg, n)):
        R = Rotation.from_rotvec(axis * t).as_matrix()
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R, np.array(point) - R @ np.array(point)
        out.append(np.linalg.inv(M @ base))
    return np.array(out), axis


def test_the_model_recovers_the_axis_and_the_rotation_exactly():
    vms, axis = _turntable(span_deg=300)
    fit = process_sfm.fit_turntable(vms)
    assert fit["span"] == pytest.approx(300, abs=0.5)
    assert abs(fit["axis"] @ axis) == pytest.approx(1.0, abs=1e-6)
    assert fit["monotonic"] == pytest.approx(1.0)
    assert fit["rms_pos"] < 1e-6 and fit["rms_rot"] < 1e-6


def test_the_axis_need_not_pass_through_the_origin():
    """The turn centre is wherever the subject sat, which is not the frame
    origin — an axis fitted as a pure direction would misplace every pose."""
    vms, _ = _turntable(point=(7.0, -3.0, 11.0))
    fit = process_sfm.fit_turntable(vms)
    assert fit["rms_pos"] < 1e-6


def test_a_noisy_solve_still_fits():
    vms, axis = _turntable()
    rng = np.random.default_rng(0)
    noisy = vms.copy()
    for i in range(len(noisy)):
        d = np.eye(4)
        d[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.004, 3)).as_matrix()
        d[:3, 3] = rng.normal(0, 0.01, 3)
        noisy[i] = d @ noisy[i]
    fit = process_sfm.fit_turntable(noisy)
    assert fit["span"] == pytest.approx(300, abs=5)
    assert abs(fit["axis"] @ axis) > 0.999
    assert fit["monotonic"] > 0.95


def test_a_folded_solve_is_caught_by_the_direction_of_travel():
    """THE point of the feature. A folded reconstruction sends the subject
    back the way it came, which no amount of coverage or reprojection error
    reveals — but a turntable only turns one way."""
    vms, _ = _turntable(span_deg=300, n=120)
    folded = np.concatenate([vms[:60], vms[:60][::-1]])
    fit = process_sfm.fit_turntable(folded)
    assert fit["monotonic"] == pytest.approx(0.5, abs=0.05)


def test_snapping_removes_wobble_the_solve_invented():
    """A real turntable has no radius, elevation or tilt freedom, so noise in
    those is error by definition."""
    vms, axis = _turntable()
    rng = np.random.default_rng(1)
    noisy = np.array(
        [
            np.linalg.inv(
                np.block(
                    [
                        [np.eye(3), rng.normal(0, 0.02, (3, 1))],
                        [np.zeros((1, 3)), np.ones((1, 1))],
                    ]
                )
                @ np.linalg.inv(vm)
            )
            for vm in vms
        ]
    )
    fit = process_sfm.fit_turntable(noisy)
    snapped = process_sfm.apply_turntable(
        noisy, fit["axis"], fit["point"], np.radians(fit["angles"])
    )
    before = np.abs(
        np.array([-vm[:3, :3].T @ vm[:3, 3] for vm in noisy])
        - np.array([-vm[:3, :3].T @ vm[:3, 3] for vm in vms])
    ).mean()
    after = np.abs(
        np.array([-vm[:3, :3].T @ vm[:3, 3] for vm in snapped])
        - np.array([-vm[:3, :3].T @ vm[:3, 3] for vm in vms])
    ).mean()
    assert after < before


def test_a_partial_turn_reports_the_angle_it_actually_covered():
    vms, _ = _turntable(span_deg=140)
    assert process_sfm.fit_turntable(vms)["span"] == pytest.approx(140, abs=0.5)
