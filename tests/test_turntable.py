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


def test_a_walk_around_run_with_turntable_is_told_so(capsys):
    """The failure this guards against is a MISDIAGNOSIS. Poses that are not a
    single-axis rotation at all also look non-monotonic, so without this check
    the fold warning fires and sends the user to fix a turntable they never
    used. Measured on the real walk-around capture: 46% of radius, 13.6 deg."""
    n = 60
    ang = np.radians(np.linspace(0, 300, n))
    centres = np.stack([6 * np.cos(ang), np.linspace(-2, 2, n), 6 * np.sin(ang)], 1)
    vms = []
    for i, c in enumerate(centres):
        f = -c / np.linalg.norm(c)
        r = np.cross([0, 1.0, 0], f)
        r /= np.linalg.norm(r)
        R = np.stack([r, np.cross(f, r), f])
        vm = np.eye(4)
        vm[:3, :3], vm[:3, 3] = R, -R @ c
        vms.append(vm)
    process_sfm.report_turntable(np.array(vms), snap=False)
    out = capsys.readouterr().out
    assert "do not fit a turntable at all" in out
    assert "drop --turntable" in out
    assert "FOLDED" not in out, "must not blame a fold when it is not a turntable"


def test_a_genuine_turntable_draws_no_complaint(capsys):
    vms, _ = _turntable(span_deg=320)
    process_sfm.report_turntable(vms, snap=False)
    out = capsys.readouterr().out
    assert "WARNING" not in out


def test_a_short_but_genuine_turn_warns_only_about_coverage(capsys):
    vms, _ = _turntable(span_deg=140)
    process_sfm.report_turntable(vms, snap=False)
    out = capsys.readouterr().out
    assert "only turned 140 deg" in out
    assert "do not fit a turntable" not in out


def test_a_fold_is_caught_by_net_rotation_not_by_step_direction():
    """Regression: `monotonic` cannot gate the fold warning. Measured on the
    House capture it read 0.87 for BOTH a solve with a 171 deg fold and the
    same capture solved cleanly with --matching ordered — it counts per-frame
    step direction, which on a 219-frame solve is dominated by jitter."""
    import numpy as np

    axis, point = np.array([0.0, 1.0, 0.0]), np.zeros(3)

    def poses(angles_deg):
        base = np.eye(4)
        base[:3, 3] = [0.0, 0.0, -4.0]
        return process_sfm.apply_turntable(
            np.array([base] * len(angles_deg)), axis, point,
            np.radians(np.asarray(angles_deg)),
        )

    n = 200
    clean = np.linspace(0, 360, n)
    # forwards, back, forwards — net progress near zero, like the real fold
    folded = np.concatenate([np.linspace(0, 120, n // 2),
                             np.linspace(120, 5, n // 2)])
    f_clean = process_sfm.fit_turntable(poses(clean))
    f_folded = process_sfm.fit_turntable(poses(folded))
    assert f_clean["directness"] > 0.9, f_clean["directness"]
    assert f_folded["directness"] < 0.3, f_folded["directness"]


def test_uneven_sampling_is_caught_even_when_coverage_is_complete():
    """The House capture turned 411 deg and still lost a roof slope: coverage
    and SAMPLING are different questions, so a full turn is not a clean bill."""
    import numpy as np

    even = np.linspace(0, 360, 216)
    # same span, same frame count, but two thirds of them crammed into 60 deg
    lumpy = np.concatenate([np.linspace(0, 60, 144), np.linspace(60, 360, 72)])
    s_even = process_sfm.angular_sampling(even)
    s_lumpy = process_sfm.angular_sampling(lumpy)
    assert s_even["min"] >= process_sfm.MIN_VIEWS_PER_30DEG, s_even
    assert s_lumpy["min"] < process_sfm.MIN_VIEWS_PER_30DEG, s_lumpy
    # `ratio` is deliberately NOT asserted here: when most of the arc is the
    # sparse part, the median window count equals the minimum and the ratio
    # collapses to 1.0. `min` is the number that carries the meaning.
    assert s_even["min"] > s_lumpy["min"]


def test_sampling_is_not_confused_by_a_short_arc():
    """A capture spanning less than one window is judged by coverage, not by
    sampling — it must not also be accused of turning unevenly."""
    import numpy as np

    s = process_sfm.angular_sampling(np.linspace(0, 12, 40))
    assert s["min"] == 40 and s["ratio"] == 1.0
