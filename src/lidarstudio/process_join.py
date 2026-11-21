#!/usr/bin/env python3
"""Refine a hand-made join of a capture's separately solved pieces.

A capture that falls apart is solved as pieces (process_sfm), each in its own
frame. The user moves them into place in the viewer, which is good to a few
centimetres or a degree: enough to see the scene, not enough for a splat,
which wants about a pixel. This script tightens that placement. Each piece
stays RIGID: only where it sits (rotation, scale, shift) is refined, never
the poses inside it.

1. ICP. Each piece's sparse points are pulled onto the pieces already placed
   wherever they overlap, largest piece fixed. Accepted only when enough
   points overlap and the fit clearly improves.
2. Cross-join matching. The placement says which frames of different pieces
   look at the same place, which the original matching could not know: it
   only matched frames near each other in time. Those pairs are matched now.
3. Shared points. A feature matched across a join that each piece had already
   triangulated gives the same scene point in both pieces' frames. A robust
   (RANSAC) similarity fit of those point pairs places the piece where the
   photos agree. Where a join has too few, the placement from steps 1-2 stands.

Why rigid, measured on Clinic (11-21): a free joint bundle adjustment over the
pieces re-solved the errors already INSIDE a piece as well (that main piece
holds a 114 deg flip), wandered 10 deg, never converged, and from two starting
placements 3 deg apart ended 15 deg apart. A refinement that depends on where
it starts is not a refinement.

Writes the joined poses npz (each piece keeps its own lens), its sparse cloud,
and a JSON of each piece's final placement so the viewer can move the loaded
pieces to match.

    process_join.py --spec join.json --output sfm/sfm_viewmats_X_refined.npz \
        --cloud-output pointclouds/pointcloud_X_refined.ply

The spec is ``{"database", "pieces": [{"poses", "matrix", "id"}]}`` with
``matrix`` the viewer's world matrix of that piece (16, column-major) and
``id`` the caller's own label for it, echoed back in the placements.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pieces  # noqa: E402
import process_sfm as ps  # noqa: E402
from process_sfm import log, progress  # noqa: E402

# ICP. The reach is a share of the joined scene's size, coarse to fine: a
# hand placement is off by about a percent of the room, and a reach much
# beyond that lets ICP slide a piece along a wall into whatever is nearest.
ICP_REACH = (0.03, 0.015, 0.0075)
ICP_ITERS = 20
# Accepted only when this share of the piece's points ends up within reach of
# another piece, and the median distance of those falls by this factor.
ICP_MIN_OVERLAP = 0.05
ICP_MIN_POINTS = 50
ICP_MIN_GAIN = 0.8
# ...and only when the correction stays small. A larger one is not a
# refinement of the user's placement but a different placement, which ICP
# has no business choosing. The same limits apply to the shared-point fit.
MAX_TURN_DEG = 10.0
MAX_SCALE = 0.1
MAX_SHIFT = 0.1              # share of the scene's size

# Cross-join pairs: each frame is matched to its nearest frames of OTHER
# pieces that face roughly the same way.
PAIR_NEIGHBOURS = 8
PAIR_MAX_ANGLE_DEG = 60.0
PAIR_RADIUS_STEPS = 50.0     # in median camera steps, as for loop pairs
# Shared-point fit: a pair is an inlier within this share of the scene, and
# a join needs this many inliers to be trusted.
SHARED_INLIER = 0.005
MIN_SHARED_INLIERS = 30


def umeyama(src, dst):
    """Least-squares similarity ``dst ≈ s R src + t``; returns ``(s, R, t)``."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (a ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var) if var > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s


def sim_matrix(s, R, t):
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M


def apply(M, X):
    return X @ M[:3, :3].T + M[:3, 3]


def describe(M, centre, size):
    """``(turn_deg, scale, shift)`` of correction ``M`` for a piece centred at
    ``centre``, the shift as a share of the scene's size."""
    s = float(np.cbrt(np.linalg.det(M[:3, :3])))
    R = M[:3, :3] / s
    turn = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    shift = float(np.linalg.norm(apply(M, centre[None])[0] - centre)) / size
    return turn, s, shift


def too_far(M, centre, size):
    turn, s, shift = describe(M, centre, size)
    if turn > MAX_TURN_DEG or abs(s - 1) > MAX_SCALE or shift > MAX_SHIFT:
        return (f"it would move {turn:.1f} deg, scale {s:.3f}, shift "
                f"{100 * shift:.1f}% of the scene")
    return None


def icp_sim3(src, dst, size):
    """ICP of points ``src`` onto ``dst`` (a similarity, coarse to fine).

    Returns ``(M, report)``: the 4x4 correction to apply to ``src``, identity
    when the fit is refused, and a dict saying what happened.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(dst)
    M = np.eye(4)
    cur = src.copy()
    for frac in ICP_REACH:
        reach = frac * size
        for _ in range(ICP_ITERS):
            d, j = tree.query(cur, distance_upper_bound=reach)
            ok = np.isfinite(d)
            if ok.sum() < ICP_MIN_POINTS:
                break
            s, R, t = umeyama(cur[ok], dst[j[ok]])
            cur = (s * (R @ cur.T)).T + t
            M = sim_matrix(s, R, t) @ M
            if abs(s - 1) < 1e-5 and np.linalg.norm(t) < 1e-5 * size:
                break
    fine = ICP_REACH[-1] * size
    d, _ = tree.query(cur, distance_upper_bound=fine)
    near = np.isfinite(d)
    overlap = float(near.mean()) if len(near) else 0.0
    after = float(np.median(d[near])) if near.sum() >= ICP_MIN_POINTS else None
    d1, _ = tree.query(src, distance_upper_bound=fine)
    n1 = np.isfinite(d1)
    before = float(np.median(d1[n1])) if n1.sum() >= ICP_MIN_POINTS else None
    rep = {"overlap": overlap, "before": before, "after": after}
    why = None
    if after is None or overlap < ICP_MIN_OVERLAP:
        why = f"too little overlap ({100 * overlap:.1f}% of its points)"
    elif before is not None and after > ICP_MIN_GAIN * before:
        why = "no clear improvement"
    else:
        why = too_far(M, src.mean(0), size)
    rep["accepted"] = why is None
    rep["why"] = why
    return (M if why is None else np.eye(4)), rep


def cross_pairs(groups, pairs_path):
    """Pairs of frames from DIFFERENT pieces that, as placed, sit close and
    face the same way. ``groups`` is ``[{name: (centre, axis)}]`` per piece.
    Returns ``(count, radius)``."""
    allc = np.array([c for g in groups for c, _a in g.values()])
    steps = []
    for g in groups:
        cs = np.array([g[n][0] for n in sorted(g)])
        if len(cs) > 1:
            steps.append(np.linalg.norm(np.diff(cs, axis=0), axis=1))
    step = float(np.median(np.concatenate(steps))) if steps else 0.0
    radius = PAIR_RADIUS_STEPS * step if step > 0 else np.inf
    names = [n for g in groups for n in g]
    piece = np.array([k for k, g in enumerate(groups) for _ in g])
    axes = np.array([a for g in groups for _c, a in g.values()])
    pairs = set()
    for i, n in enumerate(names):
        d = np.linalg.norm(allc - allc[i], axis=1)
        ang = np.degrees(np.arccos(np.clip(axes @ axes[i], -1, 1)))
        cand = np.where((piece != piece[i]) & (d < radius)
                        & (ang < PAIR_MAX_ANGLE_DEG))[0]
        for j in cand[np.argsort(d[cand])][:PAIR_NEIGHBOURS]:
            pairs.add(tuple(sorted((n, names[j]))))
    with open(pairs_path, "w") as f:
        for a, b in sorted(pairs):
            f.write(f"{a} {b}\n")
    return len(pairs), radius


def cross_matches(db: Path, piece_of):
    """Verified feature matches between frames of different pieces, as
    ``{(piece_a, piece_b): [(name_a, idx_a, name_b, idx_b), ...]}`` with
    ``piece_a < piece_b``."""
    con = sqlite3.connect(str(db))
    try:
        by_id = dict(con.execute("SELECT image_id, name FROM images"))
        rows = list(con.execute(
            "SELECT pair_id, rows, data FROM two_view_geometries "
            "WHERE rows > 0"))
    finally:
        con.close()
    big = 2147483647  # COLMAP's pair_id = id1 * big + id2, id1 < id2
    out = {}
    for pid, n, blob in rows:
        a, b = by_id[pid // big], by_id[pid % big]
        ka, kb = piece_of.get(a), piece_of.get(b)
        if ka is None or kb is None or ka == kb or blob is None:
            continue
        m = np.frombuffer(blob, dtype=np.uint32).reshape(n, 2)
        if ka > kb:
            ka, kb, a, b = kb, ka, b, a
            m = m[:, ::-1]
        out.setdefault((ka, kb), []).extend(
            (a, int(i), b, int(j)) for i, j in m)
    return out


def point_lookup(rec):
    """``{(image name, feature index): xyz}`` for every feature of ``rec``
    that is part of a triangulated point."""
    out = {}
    for im in rec.images.values():
        if not im.has_pose:
            continue
        for idx, p2 in enumerate(im.points2D):
            if p2.has_point3D():
                out[(im.name, idx)] = rec.points3D[p2.point3D_id].xyz
    return out


def shared_fit(src, dst, size):
    """Robust similarity mapping a piece's copy of the shared points ``src``
    onto the placed pieces' copy ``dst``. Returns ``(M or None, inliers)``."""
    import pycolmap

    if len(src) < MIN_SHARED_INLIERS:
        return None, 0
    opts = pycolmap.RANSACOptions()
    opts.max_error = SHARED_INLIER * size
    sim = pycolmap.estimate_sim3d_robust(src, dst, opts)
    if isinstance(sim, dict):
        sim = sim.get("tgt_from_src")
    if sim is None:
        return None, 0
    M = np.eye(4)
    M[:3, :] = np.asarray(sim.matrix())
    inl = np.linalg.norm(apply(M, src) - dst, axis=1) < SHARED_INLIER * size
    if inl.sum() < MIN_SHARED_INLIERS:
        return None, int(inl.sum())
    # Polish on the inliers with plain least squares.
    s, R, t = umeyama(src[inl], dst[inl])
    return sim_matrix(s, R, t), int(inl.sum())


def main():
    import pycolmap

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cloud-output", default=None)
    p.add_argument("--placements", default=None,
                   help="write each piece's final placement here (JSON)")
    p.add_argument("--workspace", default=None)
    p.add_argument("--no-icp", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    spec = json.loads(Path(args.spec).read_text())
    db_src = Path(spec["database"])
    if not db_src.exists():
        print(f"ERROR: the matches of this capture ({db_src.name}) are gone, "
              "so the join cannot be refined. Re-run the reconstruction "
              "(Force SfM) to rebuild them.", flush=True)
        return 1
    out = Path(args.output)
    ws = Path(args.workspace) if args.workspace else out.parent / f"work_{out.stem}"
    shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True)

    # 1. Each piece's model as solved, and where the user put it. Points and
    # cameras stay in the piece's own frame and are moved by ``placement`` on
    # use, so the result is one matrix per piece.
    progress(5, "Refine join: placing the pieces…")
    recs, placement, npzs = [], [], []
    for item in spec["pieces"]:
        npz = Path(item["poses"])
        sparse = npz.parent / (npz.stem + "_sparse")
        if not sparse.exists():
            print(f"ERROR: {npz.name} has no sparse model beside it "
                  f"({sparse.name}); re-run the reconstruction.", flush=True)
            return 1
        recs.append(pycolmap.Reconstruction(str(sparse)))
        placement.append(pieces.similarity(item["matrix"])[0]
                         if item.get("matrix") is not None else np.eye(4))
        npzs.append(npz)
    # Named as the capture names them ("main piece", "piece 3"), not by
    # their position in this list.
    label = ["piece " + m.group(1) if (m := re.search(r"_part(\d+)$", n.stem))
             else "main piece" for n in npzs]
    own_pts = [np.array([q.xyz for q in r.points3D.values()]).reshape(-1, 3)
               for r in recs]
    allp = np.vstack([apply(M, X) for M, X in zip(placement, own_pts)])
    lo, hi = np.percentile(allp, 2, axis=0), np.percentile(allp, 98, axis=0)
    size = float(np.linalg.norm(hi - lo)) or 1.0
    main_k = max(range(len(recs)), key=lambda k: recs[k].num_reg_images())
    log(f"{len(recs)} pieces, {sum(r.num_reg_images() for r in recs)} frames; "
        f"scene size {size:.3f} units; {label[main_k]} stays fixed")
    order = sorted(range(len(recs)), key=lambda k: -len(own_pts[k]))
    order.remove(main_k)

    # 2. ICP onto everything already placed, biggest pieces first.
    report = {k: [] for k in range(len(recs))}
    if not args.no_icp:
        progress(10, "Refine join: fitting the pieces' points together (ICP)…")
        placed = [main_k]
        for k in order:
            dst = np.vstack([apply(placement[q], own_pts[q]) for q in placed])
            src = apply(placement[k], own_pts[k])
            C, rep = icp_sim3(src, dst, size)
            if rep["accepted"]:
                turn, s, shift = describe(C, src.mean(0), size)
                placement[k] = C @ placement[k]
                msg = (f"ICP moved it {turn:.2f} deg, scale {s:.4f}, "
                       f"{100 * shift:.2f}% of the scene (median gap "
                       f"{rep['before']:.4f} → {rep['after']:.4f})")
            else:
                msg = f"ICP kept it as placed: {rep['why']}"
            log(f"{label[k]}: {msg}")
            report[k].append(msg)
            placed.append(k)

    # 3. Match frames that, as placed, face the same place across the joins.
    progress(25, "Refine join: matching frames across the joins…")
    groups, piece_of = [], {}
    for k, r in enumerate(recs):
        s = float(np.cbrt(np.linalg.det(placement[k][:3, :3])))
        Rp = placement[k][:3, :3] / s
        g = {}
        for im in r.images.values():
            if im.has_pose:
                R = im.cam_from_world().rotation.matrix()
                g[im.name] = (
                    apply(placement[k], np.asarray(im.projection_center())[None])[0],
                    Rp @ R[2, :])
                piece_of[im.name] = k
        groups.append(g)
    db = ws / "database.db"
    shutil.copy2(db_src, db)
    n_pairs, radius = cross_pairs(groups, ws / "pairs_cross.txt")
    log(f"cross-join pairs: {n_pairs} (radius {radius:.3f} units)")
    if n_pairs:
        ps.match_loop_pairs(db, ws / "pairs_cross.txt")

    # 4. Shared points: each piece onto the pieces already placed.
    progress(70, "Refine join: fitting the pieces to their shared points…")
    matches = cross_matches(db, piece_of)
    lookup = [point_lookup(r) for r in recs]
    tied = set()
    placed = [main_k]
    for k in order:
        src, dst = [], []
        for q in placed:
            lo_k, hi_k = min(k, q), max(k, q)
            for a, ia, b, ib in matches.get((lo_k, hi_k), []):
                if lo_k != k:            # orient as (this piece, placed piece)
                    a, ia, b, ib = b, ib, a, ia
                xk = lookup[k].get((a, ia))
                xq = lookup[q].get((b, ib))
                if xk is not None and xq is not None:
                    src.append(xk)
                    dst.append(apply(placement[q], np.asarray(xq)[None])[0])
        src = apply(placement[k], np.array(src).reshape(-1, 3))
        dst = np.array(dst).reshape(-1, 3)
        C, n_in = shared_fit(src, dst, size)
        if C is None:
            msg = (f"shared points: {len(src)} pairs, {n_in} consistent; too "
                   "few to trust, placement kept")
        else:
            why = too_far(C, src.mean(0), size)
            if why:
                msg = (f"shared points: {n_in} of {len(src)} agree, but {why}; "
                       "placement kept. Check this join by eye")
                print(f"WARNING: {label[k]}: the shared points would move it a "
                      f"long way ({why}). Stray matches on repeated texture "
                      "can do that, or the hand placement is far off. The "
                      "placement was kept.", flush=True)
            else:
                turn, s, shift = describe(C, src.mean(0), size)
                placement[k] = C @ placement[k]
                tied.add(k)
                msg = (f"shared points: {n_in} of {len(src)} agree; moved it "
                       f"{turn:.2f} deg, scale {s:.4f}, {100 * shift:.2f}% of "
                       "the scene")
        log(f"{label[k]}: {msg}")
        report[k].append(msg)
        placed.append(k)
    if len(recs) > 1 and not tied:
        print("WARNING: no join had enough shared points to refine it, so "
              "the result is your placement (plus any ICP fit). The joined "
              "splat will train, but the joins are only as good as that "
              "placement.", flush=True)

    # 5. Export: the joined poses (each piece's own lens), the cloud, and the
    # placements for the viewer.
    progress(90, "Refine join: writing poses and cloud…")
    col_major = [M.T.reshape(-1).tolist() for M in placement]
    r = pieces.join_poses(list(zip(npzs, col_major)), out)
    if args.cloud_output:
        xyz, rgb = [], []
        for M, rec in zip(placement, recs):
            pts = list(rec.points3D.values())
            if pts:
                xyz.append(apply(M, np.array([q.xyz for q in pts])))
                rgb.append(np.array([q.color for q in pts]))
        n = ps._write_ply(np.vstack(xyz), np.vstack(rgb), Path(args.cloud_output))
        log(f"joined cloud: {n:,} points → {args.cloud_output}")
    if args.placements:
        Path(args.placements).write_text(json.dumps({
            "pieces": [
                {"id": item.get("id"), "poses": str(npz), "matrix": cm,
                 "tied": k in tied, "report": report[k]}
                for k, (item, npz, cm) in enumerate(
                    zip(spec["pieces"], npzs, col_major))
            ]
        }))
    shutil.rmtree(ws, ignore_errors=True)
    log(f"wrote {out} ({r['frames']} frames)")
    progress(100, f"Join refined ({r['frames']} frames; {len(tied)} of "
                  f"{len(recs) - 1} join(s) tied by shared points; "
                  f"{time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
