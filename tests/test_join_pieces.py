"""Joining separately solved pieces of a capture into one set of poses."""

import numpy as np
import pytest

from lidarstudio import pieces


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    i, j = [k for k in range(3) if k != axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def _viewmats(n, rng):
    out = []
    for _ in range(n):
        R = _rot(1, rng.uniform(-90, 90)) @ _rot(0, rng.uniform(-20, 20))
        C = rng.normal(size=3)
        V = np.eye(4)
        V[:3, :3] = R
        V[:3, 3] = -R @ C
        out.append(V)
    return np.array(out)


def _placement(s=2.5, deg=40.0, t=(3.0, -1.0, 0.5)):
    M = np.eye(4)
    M[:3, :3] = s * _rot(2, deg) @ _rot(0, 15)
    M[:3, 3] = t
    return M


def _col_major(M):
    return M.T.reshape(-1).tolist()


def test_a_point_moved_with_its_piece_projects_to_the_same_pixel():
    rng = np.random.default_rng(0)
    V = _viewmats(6, rng)
    M = _placement()
    Mr, s = pieces.similarity(_col_major(M))
    V2 = pieces.move_viewmats(V, Mr, s)
    X = np.array([0.3, 0.2, 4.0, 1.0])
    for a, b in zip(V, V2):
        assert np.allclose(b[:3, :3] @ b[:3, :3].T, np.eye(3), atol=1e-9)
        p, q = a @ X, b @ (M @ X)
        assert np.allclose(p[:2] / p[2], q[:2] / q[2], atol=1e-9)
        # Depth scales with the scene, so the camera stays rigid.
        assert np.isclose(q[2], s * p[2])


def test_uneven_scale_and_mirroring_are_refused():
    M = _placement()
    M[:3, 0] *= 1.2
    with pytest.raises(ValueError, match="unevenly"):
        pieces.similarity(_col_major(M))
    M = _placement()
    M[:3, 0] *= -1
    with pytest.raises(ValueError, match="mirrored"):
        pieces.similarity(_col_major(M))


def _write_piece(path, names, ts, V, f):
    np.savez(path, names=np.array(names), bag_ts_ns=np.array(ts, np.int64),
             viewmats=V, camera_model=np.array("SIMPLE_RADIAL"),
             camera_params=np.array([f, 540.0, 960.0, 0.03]),
             camera=np.array("front"))


def test_join_poses_keeps_each_pieces_lens_and_capture_order(tmp_path):
    rng = np.random.default_rng(1)
    a, b = tmp_path / "sfm_viewmats_X.npz", tmp_path / "sfm_viewmats_X_part1.npz"
    _write_piece(a, ["0010.jpg", "0011.jpg"], [10, 11], _viewmats(2, rng), 1600.0)
    _write_piece(b, ["0000.jpg", "0001.jpg"], [0, 1], _viewmats(2, rng), 1700.0)
    out = tmp_path / "sfm_viewmats_X_joined.npz"
    r = pieces.join_poses([(a, None), (b, _col_major(_placement()))], out)
    assert r["frames"] == 4
    d = np.load(out)
    assert list(d["names"]) == ["0000.jpg", "0001.jpg", "0010.jpg", "0011.jpg"]
    assert list(d["frame_camera_params"][:, 0]) == [1700, 1700, 1600, 1600]
    assert d["camera_params"][0] == 1600      # the first piece's, for old readers


def test_poses_for_object_follows_edits_but_not_into_another_piece(tmp_path):
    (tmp_path / "sfm").mkdir()
    (tmp_path / "pointclouds").mkdir()
    for k in ("X", "X_part2"):
        (tmp_path / "sfm" / f"sfm_viewmats_{k}.npz").write_text("")
    pc = tmp_path / "pointclouds"
    assert pieces.poses_for_object(pc / "pointcloud_X_part2_edited2.ply").name == \
        "sfm_viewmats_X_part2.npz"
    assert pieces.poses_for_object(tmp_path / "splats" / "splat_X.ply").name == \
        "sfm_viewmats_X.npz"
    # Piece 3 has no poses: it must not fall back to the main piece's.
    assert pieces.poses_for_object(pc / "pointcloud_X_part3.ply") is None
    assert pieces.poses_for_object(pc / "other.ply") is None


# ── process_join: refining a hand-made join ──────────────────────────

import sqlite3  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from lidarstudio import lidar_jobs  # noqa: E402

sys.path.insert(0, str(Path(lidar_jobs.__file__).parent))
import process_join  # noqa: E402


def _room(rng, n=4000):
    """Points on the walls and floor of a 10 x 8 x 3 box: overlapping
    structure for ICP to lock onto, like a sparse cloud of a room."""
    pts = []
    for _ in range(n):
        u, v = rng.uniform(0, 1, 2)
        face = rng.integers(5)
        pts.append([(u * 10, v * 8, 0), (u * 10, 0, v * 3), (u * 10, 8, v * 3),
                    (0, u * 8, v * 3), (10, u * 8, v * 3)][face])
    return np.array(pts, float)


def test_umeyama_recovers_a_similarity():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 3))
    M = _placement(s=1.3)
    s, R, t = process_join.umeyama(X, process_join.apply(M, X))
    assert np.allclose(process_join.sim_matrix(s, R, t), M, atol=1e-9)


def test_icp_undoes_a_small_hand_placement_error():
    rng = np.random.default_rng(3)
    room = _room(rng)
    piece = room[room[:, 0] > 4]            # overlaps the rest for x in 4..6
    rest = room[room[:, 0] < 6]
    err = np.eye(4)
    err[:3, :3] = 1.01 * _rot(2, 1.5) @ _rot(0, 0.7)
    err[:3, 3] = (0.08, -0.05, 0.03)
    C, rep = process_join.icp_sim3(process_join.apply(err, piece), rest, 13.0)
    assert rep["accepted"], rep
    back = process_join.apply(C @ err, piece)
    assert np.median(np.linalg.norm(back - piece, axis=1)) < 0.01


def test_icp_refuses_a_piece_that_shares_nothing():
    rng = np.random.default_rng(4)
    room = _room(rng)
    piece, rest = room[room[:, 0] > 6], room[room[:, 0] < 4]
    C, rep = process_join.icp_sim3(piece, rest, 13.0)
    assert not rep["accepted"] and np.allclose(C, np.eye(4))


def test_cross_matches_reads_only_matches_across_pieces(tmp_path):
    db = tmp_path / "database.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
    con.execute("CREATE TABLE two_view_geometries "
                "(pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)")
    con.executemany("INSERT INTO images VALUES (?, ?)",
                    [(1, "a.jpg"), (2, "b.jpg"), (3, "c.jpg")])
    big = 2147483647

    def pair(i, j, m):
        m = np.array(m, np.uint32)
        con.execute("INSERT INTO two_view_geometries VALUES (?, ?, 2, ?)",
                    (i * big + j, len(m), m.tobytes()))
    pair(1, 2, [[5, 7]])                 # both in piece 0: ignored
    pair(1, 3, [[1, 2], [3, 4]])         # piece 0 ↔ piece 1
    pair(2, 3, [[9, 8]])
    con.commit()
    con.close()
    piece_of = {"a.jpg": 1, "b.jpg": 1, "c.jpg": 0}
    m = process_join.cross_matches(db, piece_of)
    # Oriented (lower piece first): c.jpg is piece 0, so its index comes first.
    assert sorted(m[(0, 1)]) == [("c.jpg", 2, "a.jpg", 1),
                                 ("c.jpg", 4, "a.jpg", 3),
                                 ("c.jpg", 8, "b.jpg", 9)]


def test_join_database_is_found_for_any_piece(tmp_path):
    sfm = tmp_path / "sfm"
    sfm.mkdir()
    (sfm / "sfm_viewmats_X_database.db").write_text("")
    for name in ("sfm_viewmats_X.npz", "sfm_viewmats_X_part3.npz"):
        assert lidar_jobs._join_database(sfm / name).name == \
            "sfm_viewmats_X_database.db"
