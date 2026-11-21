"""Solving a capture that falls apart as separate pieces.

The 11-08 Clinic video crosses 18 frames of blank wall that share no matches
with anything either side. Solved whole, the first minute was either dropped
or glued on at the wrong place. Each side is now solved, and trained, alone.
"""

import sys
from pathlib import Path

from lidarstudio import lidar_jobs

sys.path.insert(0, str(Path(lidar_jobs.__file__).parent))
import process_sfm  # noqa: E402


def _names(n):
    return [f"{i:04d}.jpg" for i in range(n)]


def test_unbroken_capture_is_one_piece():
    names = _names(100)
    strength = [1000.0] * 99
    assert process_sfm.capture_pieces(names, strength) == [names]


def test_broken_stretch_splits_and_drops_the_blank_frames():
    names = _names(100)
    strength = [1000.0] * 99
    for c in range(40, 45):  # frames 41-44 match nothing either side
        strength[c] = 0.0
    pieces = process_sfm.capture_pieces(names, strength)
    assert pieces == [names[:41], names[45:]]


def test_weak_but_real_link_is_not_split():
    # 2% of the median is weak (warned about) but not broken.
    names = _names(100)
    strength = [1000.0] * 99
    strength[50] = 20.0
    assert len(process_sfm.capture_pieces(names, strength)) == 1


def test_keep_pieces_picks_the_biggest_and_drops_scraps():
    pieces = [_names(200)[:60], _names(200)[60:65], _names(200)[65:]]
    main, parts = process_sfm.keep_pieces(pieces, 200)
    assert main == pieces[2]
    assert [k for k, _ in parts] == [1]  # numbered 1.. in capture order
    assert parts[0][1] == pieces[0]


def test_part_paths_and_stale_parts_are_cleared(tmp_path):
    out = tmp_path / "sfm_viewmats_X.npz"
    assert process_sfm.part_path(out, 2).name == "sfm_viewmats_X_part2.npz"
    out.write_text("main")
    process_sfm.part_path(out, 1).write_text("old")
    (tmp_path / "sfm_viewmats_X_part1_sparse").mkdir()
    process_sfm.clear_parts(out)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sfm_viewmats_X.npz"]


def test_server_finds_parts_in_capture_order(tmp_path):
    canonical = tmp_path / "sfm_viewmats_X.npz"
    for k in (3, 1, 12):
        (tmp_path / f"sfm_viewmats_X_part{k}.npz").write_text("")
    (tmp_path / "sfm_viewmats_X_part1_sparse").mkdir()
    (tmp_path / "sfm_viewmats_Y_part1.npz").write_text("")
    found = [p.name for p in lidar_jobs._sfm_parts(canonical)]
    assert found == ["sfm_viewmats_X_part1.npz", "sfm_viewmats_X_part3.npz",
                     "sfm_viewmats_X_part12.npz"]


def test_swap_args_replaces_values_and_appends_missing():
    cmd = ["py", "train.py", "--output", "a.ply", "--sfm-poses", "a.npz"]
    out = lidar_jobs._swap_args(cmd, {"--output": "b.ply",
                                      "--sfm-poses": "b.npz",
                                      "--pointcloud": "b_seed.ply"})
    assert out == ["py", "train.py", "--output", "b.ply", "--sfm-poses",
                   "b.npz", "--pointcloud", "b_seed.ply"]
    assert cmd[3] == "a.ply", "the original command is left alone"


def _jump(a, b):
    return (a, b, 50.0, 40.0)


def test_jump_on_a_weak_link_is_cut_at_the_weakest_point():
    names = _names(100)
    strength = [1000.0] * 99
    strength[48] = 30.0    # weak (3%), not broken: stray matches glued it
    cuts = process_sfm.jump_cuts(names, strength, [_jump(names[50], names[51])])
    assert cuts == [48]


def test_jump_with_strong_matches_is_left_alone():
    names = _names(100)
    strength = [1000.0] * 99
    assert process_sfm.jump_cuts(names, strength,
                                 [_jump(names[50], names[51])]) == []


def test_weak_link_far_from_the_jump_is_not_cut():
    names = _names(100)
    strength = [1000.0] * 99
    strength[20] = 10.0
    assert process_sfm.jump_cuts(names, strength,
                                 [_jump(names[50], names[51])]) == []


def test_split_piece_and_unregistered_runs():
    names = _names(100)
    piece = names[10:90]
    assert process_sfm.split_piece(piece, names, [39]) == [names[10:40],
                                                           names[40:90]]
    registered = names[10:30] + names[70:90]   # 30..69 left out
    assert process_sfm.unregistered_runs(piece, registered, 30) == [names[30:70]]
    assert process_sfm.unregistered_runs(piece, registered, 41) == []


def test_largest_parts_keeps_the_biggest_in_capture_order(tmp_path):
    import numpy as np

    parts = []
    for k, n in ((1, 40), (2, 300), (3, 90), (4, 120)):
        p = tmp_path / f"sfm_viewmats_X_part{k}.npz"
        np.savez(p, names=np.array([f"{i}.jpg" for i in range(n)]))
        parts.append(p)
    kept = lidar_jobs._largest_parts(parts, 3)    # main + 2 parts
    assert [p.name for p in kept] == ["sfm_viewmats_X_part2.npz",
                                      "sfm_viewmats_X_part4.npz"]
    assert lidar_jobs._largest_parts(parts, 0) == parts


def test_combine_bakes_each_cloud_at_its_placement(tmp_path):
    import numpy as np
    import open3d as o3d

    from lidarstudio import edit_ops

    paths = []
    for i in range(2):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.zeros((3, 3)) + i)
        pcd.colors = o3d.utility.Vector3dVector(np.full((3, 3), 0.5))
        p = tmp_path / f"piece{i}.ply"
        o3d.io.write_point_cloud(str(p), pcd)
        paths.append(str(p))
    move = np.eye(4)
    move[0, 3] = 10.0                             # second piece shifted 10 in x
    out = tmp_path / "combined.ply"
    r = edit_ops.combine([(paths[0], None),
                          (paths[1], move.T.reshape(-1).tolist())], str(out))
    assert r["count"] == 6
    xs = sorted(np.asarray(o3d.io.read_point_cloud(str(out)).points)[:, 0])
    assert xs == [0, 0, 0, 11, 11, 11]


def test_combine_joins_splats(tmp_path):
    import numpy as np

    from lidarstudio import edit_ops, splat_io

    fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    dt = np.dtype([(f, "<f4") for f in fields])
    paths = []
    for i in range(2):
        d = np.zeros(4, dt)
        d["rot_0"] = 1
        p = tmp_path / f"s{i}.ply"
        splat_io.save_splat(p, d)
        paths.append(str(p))
    move = np.eye(4)
    move[1, 3] = 2.0
    out = tmp_path / "joined.ply"
    r = edit_ops.combine([(paths[0], None),
                          (paths[1], move.T.reshape(-1).tolist())], str(out))
    data, _ = splat_io.load_splat(out)
    assert r["kind"] == "splat" and len(data) == 8
    assert sorted(set(np.round(data["y"], 3))) == [0.0, 2.0]


def test_jump_at_a_piece_start_does_not_recut_the_gap_before_it():
    # The 11-21 Clinic main piece began at 0298, just past a blank gap
    # (0280-0297). Its first frames jumped, and the weakest link in reach was
    # inside that gap, which is not part of the piece: no cut.
    names = _names(100)
    strength = [1000.0] * 99
    for c in range(40, 47):
        strength[c] = 0.0
    piece = names[47:]
    assert process_sfm.jump_cuts(names, strength,
                                 [_jump(names[48], names[49])], piece) == []
    # Without the piece it would cut inside the gap.
    assert process_sfm.jump_cuts(names, strength,
                                 [_jump(names[48], names[49])]) != []


def test_a_project_folder_in_the_scan_field_is_named_as_such(tmp_path):
    (tmp_path / "scans" / "Clinic" / "images").mkdir(parents=True)
    msg = lidar_jobs._not_a_scan(tmp_path)
    assert "PROJECT folder" in msg and str(tmp_path / "scans" / "Clinic") in msg
    assert "PROJECT" not in lidar_jobs._not_a_scan(tmp_path / "scans")
