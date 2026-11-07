"""Keeping a walk-through in one piece.

The Clinic video (hallway → doorway → room) lost the whole room in two steps:
the blur filter deleted 8 consecutive frames at the doorway, and incremental
mapping then fragmented, with run_mapping keeping only the largest piece.
"""

import sys
from pathlib import Path

from lidarstudio import lidar_jobs

sys.path.insert(0, str(Path(lidar_jobs.__file__).parent))
import process_ingest  # noqa: E402
import process_sfm  # noqa: E402


def test_blur_filter_never_drops_neighbours():
    # A sharp capture with one blurry stretch, like walking through a door.
    scores = [500.0] * 40
    for i in range(20, 28):
        scores[i] = 10.0 + i
    dropped = process_ingest.blurry_drops(scores, 0.2)
    assert dropped, "the blurry stretch should still be thinned"
    assert all(i + 1 not in dropped for i in dropped), "a gap of 2+ frames"
    assert dropped <= set(range(20, 28)), "only blurry frames are dropped"


def test_blur_filter_still_drops_isolated_blurry_frames():
    scores = [500.0] * 30
    for v, i in enumerate((3, 10, 17, 24), start=1):
        scores[i] = float(v)
    assert process_ingest.blurry_drops(scores, 0.1) == {3, 10, 17}


def test_blur_filter_off():
    assert process_ingest.blurry_drops([1.0, 2.0, 3.0], 0) == set()


def test_pick_solve_prefers_rotation_consistency():
    # Clinic, blur filter off: incremental bent (5 deg) and short of frames.
    assert process_sfm.pick_solve(
        [("incremental", 249, 5.0), ("glomap", 291, 0.6)]) == "glomap"
    # CupPhotos: GLOMAP is the one that disagrees with the pairwise evidence.
    assert process_sfm.pick_solve(
        [("incremental", 57, 6.75), ("glomap", 57, 10.13)]) == "incremental"


def test_pick_solve_tie_goes_to_more_frames():
    assert process_sfm.pick_solve(
        [("incremental", 211, 0.54), ("glomap", 273, 0.48)]) == "glomap"
    assert process_sfm.pick_solve(
        [("incremental", 273, 0.50), ("glomap", 211, 0.40)]) == "incremental"


def test_pick_solve_ignores_a_small_consistent_subset():
    # Few frames always look consistent; that is not a reason to keep them.
    assert process_sfm.pick_solve(
        [("incremental", 250, 2.0), ("glomap", 60, 0.3)]) == "incremental"


def _match_db(path, n, links):
    """A minimal COLMAP-style database: ``n`` frames, verified pairs given as
    ``{(i, j): inliers}`` over 0-based frame indices."""
    import sqlite3

    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
    con.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER)")
    # Image ids deliberately NOT in name order: order must come from names.
    ids = {i: 1000 - i for i in range(n)}
    con.executemany("INSERT INTO images VALUES (?, ?)",
                    [(ids[i], f"{i:04d}.jpg") for i in range(n)])
    big = 2147483647
    for (i, j), k in links.items():
        a, b = sorted((ids[i], ids[j]))
        con.execute("INSERT INTO two_view_geometries VALUES (?, ?)", (a * big + b, k))
    con.commit()
    con.close()


def _walkthrough(n=60, door=30, door_inliers=None):
    """Each frame matched to its next 5 with 200 inliers — except across
    ``door``, where only the single (door-1, door) pair links the halves."""
    links = {}
    for i in range(n):
        for j in range(i + 1, min(i + 6, n)):
            if i < door <= j and (i, j) != (door - 1, door):
                continue
            links[(i, j)] = 200
    if door_inliers is not None:
        links[(door - 1, door)] = door_inliers
    return links


def test_bridge_strength_sums_the_pairs_straddling_each_cut(tmp_path):
    db = tmp_path / "d.db"
    _match_db(db, 4, {(0, 1): 10, (0, 2): 5, (1, 3): 7, (2, 3): 3})
    names, s = process_sfm.bridge_strength(db)
    assert names == ["0000.jpg", "0001.jpg", "0002.jpg", "0003.jpg"]
    assert list(s) == [15, 12, 10]


def test_a_thin_doorway_is_reported(tmp_path):
    db = tmp_path / "d.db"
    _match_db(db, 60, _walkthrough(door_inliers=20))
    links = process_sfm.weak_links(*process_sfm.bridge_strength(db))
    assert [(a, b) for a, b, _ in links] == [("0029.jpg", "0030.jpg")]
    assert links[0][2] < process_sfm.WEAK_LINK_FRAC


def test_a_well_linked_capture_is_not(tmp_path):
    db = tmp_path / "d.db"
    _match_db(db, 60, _walkthrough(door=0))  # no door: fully chained
    assert process_sfm.weak_links(*process_sfm.bridge_strength(db)) == []


def test_the_capture_ends_are_never_reported():
    # Ends are thin by construction: one side holds only a frame or two.
    names = [f"{i:04d}.jpg" for i in range(50)]
    strength = [1.0] + [100.0] * 47 + [1.0]
    assert process_sfm.weak_links(names, strength) == []


def test_a_weak_run_is_reported_once():
    names = [f"{i:04d}.jpg" for i in range(50)]
    strength = [100.0] * 49
    strength[20] = strength[21] = strength[22] = 1.0
    assert process_sfm.weak_links(names, strength) == [("0020.jpg", "0023.jpg", 0.01)]
