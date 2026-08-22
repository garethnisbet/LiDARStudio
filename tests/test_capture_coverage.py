"""capture_coverage: did the capture actually go around its subject?

The failure this guards against looks like success — every frame registers, the
job reports no error, and hours later the splat is solid on one side and mush
everywhere else. Two very different causes produce that, and they need opposite
advice, so the check has to tell them apart: a capture that only ever saw one
side, versus a reconstruction that folded a real orbit onto itself.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from lidarstudio import lidar_jobs

sys.path.insert(0, str(Path(lidar_jobs.__file__).parent))
import process_sfm  # noqa: E402


def _viewmats(centres, targets=None):
    """World→cam matrices for cameras at ``centres`` looking at the origin
    (or at per-camera ``targets``)."""
    out = []
    for i, c in enumerate(centres):
        aim = (targets[i] if targets is not None else np.zeros(3)) - c
        f = aim / np.linalg.norm(aim)
        ref = np.array([0, 0, 1.0])
        if abs(f @ ref) > 0.99:
            ref = np.array([0, 1.0, 0])
        r = np.cross(ref, f)
        r /= np.linalg.norm(r)
        R = np.stack([r, np.cross(f, r), f])
        vm = np.eye(4)
        vm[:3, :3], vm[:3, 3] = R, -R @ c
        out.append(vm)
    return np.array(out)


def orbit(n=60, arc_deg=360.0, radius=4.0, height=1.5, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    a = np.deg2rad(np.linspace(0, arc_deg, n, endpoint=(arc_deg < 359)))
    C = np.stack([radius * np.cos(a), radius * np.sin(a), np.full(len(a), height)], 1)
    return _viewmats(C + rng.normal(scale=jitter, size=(len(a), 3)))


@pytest.mark.parametrize("arc,expected", [(360, 354), (300, 300), (240, 240),
                                          (180, 180), (120, 120), (58, 58)])
def test_arc_is_measured_accurately(arc, expected):
    (measured, _turning), _ = process_sfm.capture_coverage(orbit(arc_deg=arc))
    assert measured == pytest.approx(expected, abs=6)


def test_a_steep_orbit_is_not_mistaken_for_one_sided():
    """A full circle shot looking 45-60 deg down separates its extreme
    viewpoints by only ~90 deg, so any metric based on how far apart two views
    are calls it one-sided. It is not: it saw every side."""
    for height in (4.0, 6.9):          # 45 and 60 degrees above the subject
        (arc, _), _ = process_sfm.capture_coverage(orbit(radius=4.0, height=height))
        assert arc > 340


def test_handheld_wobble_does_not_inflate_a_narrow_arc():
    """Wobble turns a short arc of viewpoints into a fat blob, whose least
    -variance axis is noise. Projecting onto it would smear the cluster over
    the whole circle and report coverage the capture never had."""
    (arc, _), _ = process_sfm.capture_coverage(orbit(arc_deg=58, jitter=0.3))
    assert arc < 90


def test_a_walkthrough_is_not_judged_as_an_orbit():
    """Cameras dollying down a corridor look outward; 'you never went around
    it' is meaningless, so the check must decline to answer."""
    n = 30
    C = np.stack([np.linspace(0, 10, n), np.zeros(n), np.full(n, 1.6)], 1)
    (arc, _), detail = process_sfm.capture_coverage(_viewmats(C, targets=C + [1, 0, 0]))
    assert arc is None and "no common subject" in detail


def test_turning_tracks_the_arc_on_a_genuine_orbit():
    """The fold test compares turning against net spread, so a real orbit must
    not accumulate much more turning than the arc it covers."""
    for arc_deg in (360, 180, 58):
        (arc, turning), _ = process_sfm.capture_coverage(orbit(arc_deg=arc_deg))
        assert turning < 1.5 * arc


def test_a_folded_orbit_is_told_apart_from_a_one_sided_capture(capsys):
    """A solve that collapses opposite views onto each other still racks up
    turning as its poses oscillate — that mismatch is the giveaway, and it must
    produce solve advice, not 'go back and re-shoot'."""
    t = np.linspace(0, 8 * np.pi, 150)
    a = np.deg2rad(25 * np.sin(t))            # wanders inside a 50 deg wedge
    C = np.stack([4 * np.cos(a), 4 * np.sin(a), np.full(len(a), 1.5)], 1)
    (arc, turning), _ = process_sfm.capture_coverage(_viewmats(C))
    assert arc < 90 and turning > 2.5 * arc

    process_sfm.report_coverage(_viewmats(C))
    out = capsys.readouterr().out
    assert "FOLDED" in out and "--matching exhaustive" in out
    assert "re-shoot" not in out.lower()


def test_a_genuinely_one_sided_capture_says_to_re_shoot(capsys):
    process_sfm.report_coverage(orbit(arc_deg=58))
    out = capsys.readouterr().out
    assert "one-sided" in out and "walk a full circle" in out
    assert "FOLDED" not in out


def test_a_full_orbit_warns_about_nothing(capsys):
    process_sfm.report_coverage(orbit(arc_deg=360))
    assert "WARNING" not in capsys.readouterr().out


def test_exhaustive_matching_is_chosen_while_it_is_affordable():
    """Sequential matching never pairs the two ends of an orbit, which is what
    lets a loop fold. Below a few hundred images exhaustive costs little and
    closes it."""
    assert process_sfm.choose_matching(158) == "exhaustive"
    assert process_sfm.choose_matching(299) == "exhaustive"
    assert process_sfm.choose_matching(300) == "sequential"
