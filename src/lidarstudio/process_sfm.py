"""Generate SfM camera poses — for a LiDAR scan, or a phone capture.

Two sources, one reconstruction engine. Given a LiDAR scan folder it refines
the odometry poses by aligning SfM to the trajectory (stages A-D below). Given
an ingested image set (``images/`` + ``manifest.csv`` from process_ingest.py)
it runs the same SIFT/COLMAP machinery and exports the poses in the SfM's OWN
frame — stage A is replaced by the manifest, stage C is skipped, and the result
is unitless and unlevelled because there is no LiDAR to scale or level against.
``--cloud-output`` then writes the sparse points as the splat's seed cloud,
which on the image_set path is the only seed there is.

Automates the July-2026 campaign pipeline that produced the champion
``sfm_viewmats.npz`` (the single biggest splat-quality lever, ~+4 dB):

  A. extract every camera frame from the IMAGE bag to portrait JPEGs,
     keyed by the exact bag timestamp process_splat.py trains on;
  B. pycolmap SIFT + sequential matching (overlap 20, quadratic) +
     incremental mapping (GLOMAP fallback; --mapper) with the factory
     OPENCV_FISHEYE calibration as a shared-camera prior; optionally
     (--mask-subject) restricted to the segmented subject, for an object
     shot on a floor or desk whose flat background folds the orbit, or
     (--mask-region) to a stated image region when the subject is not a
     class COCO knows and the camera did not move;
  C. sim(3)-align (Umeyama, iterative median trimming) the SfM poses to
     the LiDAR odometry trajectory — residuals ARE the LiDAR drift;
  D. loop closure: spatially-close frame pairs from the aligned poses
     (the sequential graph has zero cross-pass matches, so revisited
     objects drift a few cm between passes — the "blurred arm" failure),
     matched into a copy of the database, then a second mapping pass
     and final alignment.

Output: ``--output`` npz with ``names``, ``bag_ts_ns``, ``viewmats``
(world→cam, LiDAR world frame) ready for ``process_splat.py --sfm-poses``,
plus ``fisheye_params`` — the bundle-adjusted OPENCV_FISHEYE intrinsics
(fx,fy,cx,cy,k1..k4, portrait frame) for ``--fisheye-intrinsics``, so the
undistortion uses the same lens model the poses were optimised under.

Needs an interpreter with pycolmap (with GLOMAP bindings), rosbags, scipy
and cv2 — the server locates one via lidar_jobs._sfm_python(). CPU-bound;
a ~900-frame 20 MP scan takes on the order of an hour or two.
"""

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

# Reuse the bag/calibration/pose helpers from the existing pipeline stages.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import frame_source  # noqa: E402

# The bag/calibration helpers are LiDAR-only, and the image_set path never calls
# into them. They import cleanly today (process_pointcloud keeps its heavy deps
# lazy), but guarding here means a photogrammetry-only install — which has no
# rosbags, kiss-icp or open3d — cannot be broken by that changing.
try:
    import process_pointcloud as ppc  # noqa: E402
    from process_splat import _interp_pose  # noqa: E402
except ImportError as _e:  # pragma: no cover - depends on install extras
    ppc = None
    _interp_pose = None
    _LIDAR_IMPORT_ERROR = _e
else:
    _LIDAR_IMPORT_ERROR = None

# Loop-closure pair generation (stage D) — values from the campaign run that
# fixed the cross-pass drift on scan 20260527212832.
# Frame-count thresholds for the image_set path, mirroring process_ingest.py
# (not imported: that module pulls in the ingest deps, and this script runs
# under the pycolmap interpreter). MIN is geometry's hard floor; below REC the
# reconstruction runs but is warned about.
MIN_FRAMES = 4
REC_FRAMES = 20

MIN_FRAME_GAP = 25  # closer pairs are already covered by sequential overlap
MAX_DIST_M = 2.0  # camera centres within 2 m
MAX_ANGLE_DEG = 60.0  # optical axes within 60 deg (else no shared surface)
NEIGHBOURS = 8  # nearest spatial neighbours kept per image

# Subject masking (--mask-subject). COCO classes that a tabletop capture is
# plausibly *of*; the auto pick chooses among these by vote across the capture
# so one odd frame cannot rename the subject.
SUBJECT_CLASSES = (
    "cup", "bowl", "vase", "bottle", "wine glass", "potted plant", "teddy bear",
    "book", "clock", "sports ball", "banana", "apple", "orange", "mouse",
    "keyboard", "cell phone", "remote", "scissors", "toaster",
)
MASK_SCORE = 0.5  # instance-segmentation confidence floor
MASK_VOTE_FRAMES = 12  # frames sampled to choose the subject class
# Share of the sampled frames the winning class must carry to BE the subject.
# A plurality is not enough: a subject COCO cannot name still draws stray
# detections, and the ones it draws are angularly clustered, because what
# fooled the detector is one FACE rather than the object. On the plywood-house
# turntable Mask R-CNN read the round gable window as a 'clock' (0.69-0.79 on
# the two gable faces, nothing on the sides) and won with 3/12. Masking then
# fired on the gable arcs alone: 76% of frames got an all-pass mask, 21% a real
# one, and 7 were blacked out almost entirely. Coverage that is uneven AROUND
# THE ROTATION is worse than no masking at all, since it thins the dense cloud
# on the arcs it skips — the same capture solved unmasked registered 219/219
# frames over 410.8 deg, against 192/199 over 382.6 deg under those masks. A
# subject the detector can really see is visible from every angle and clears
# this easily: the mug turntable voted 12/12.
MASK_VOTE_SHARE = 0.5

# How far the single-axis turntable model may miss the solved poses before the
# capture is called "not a turntable". A jittered synthetic turntable fits to
# 0.5% of radius / 0.3 deg; a walk-around run with --turntable measured 46% and
# 13.6 deg, so anything in between is comfortably separated.
# Frames per 30 deg of turn below which a stretch reconstructs badly. Measured
# on the House capture: sectors with 4-6 frames yielded 608-660 sparse points
# per image and holed geometry, against 1069-1287 and clean surface for
# sectors with 13-19. Eight is the midpoint of that gap.
MIN_VIEWS_PER_30DEG = 8

MAX_TT_POS_RESIDUAL = 0.05  # fraction of the turn radius
MAX_TT_ROT_RESIDUAL = 3.0  # degrees


def progress(pct: int, msg: str):
    print(f"PROGRESS:{pct}:{msg}", flush=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Stage A: extract portrait frames ─────────────────────────────────


def extract_frames(image_bag: Path, camera: str, images_dir: Path, ts_csv: Path):
    """Write every camera frame as a portrait JPEG named ``NNNN.jpg`` and a
    CSV mapping name → bag timestamp.

    Uses the same reader as the point-cloud/splat stages (ppc.read_images),
    so the bag timestamps here are EXACTLY the keys build_camera_frames will
    look up in the output npz — no clock reconciliation needed.

    Frames already on disk are kept (cheap resume after an interrupted run),
    but the CSV is always rewritten from the full bag pass.
    """
    import cv2

    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (bag_ts, img) in enumerate(ppc.read_images(image_bag, camera, rot="ccw")):
        name = f"{i:04d}.jpg"
        dst = images_dir / name
        if not dst.exists():
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append((name, int(bag_ts)))
        if (i + 1) % 100 == 0:
            progress(
                2 + min(10, (i + 1) // 100), f"SfM poses: extracted {i + 1} frames…"
            )
    with open(ts_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "bag_ts_ns"])
        w.writerows(rows)
    log(f"extracted {len(rows)} frames to {images_dir}")
    return dict(rows)


# ── Stage B: SfM (features / matching / mapping) ─────────────────────


def camera_prior(scan: Path, camera: str):
    """The camera model to seed SfM with, as ``(model_name, params or None)``.

    A LiDAR scan ships a factory calibration, so the lens is known exactly and
    the prior is held fixed. A phone is not calibrated: SIMPLE_RADIAL with no
    params lets COLMAP seed the focal length from EXIF and refine it during
    bundle adjustment, which is both the standard photogrammetry setup and the
    only option when the intrinsics are genuinely unknown.
    """
    if (scan / "calibration").is_dir() and ppc is not None:
        calib = ppc.load_calibration(scan / "calibration", camera)
        K, D = calib["K"], calib["D"]
        return "OPENCV_FISHEYE", [
            K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0], D[1], D[2], D[3]
        ]
    return "SIMPLE_RADIAL", None


def choose_matching(n_images: int, turntable: bool = False) -> str:
    """Pick a matching strategy.

    A TURNTABLE capture takes 'ordered' regardless of size. It is ordered video
    by definition, and the fold is its characteristic failure — a symmetric
    subject lets the mapper place a view of the back at the front's angle, and
    no amount of feature evidence can rule that out. Restricting matches to a
    frame window makes that structurally impossible. Measured on a plywood
    model house, same footage: exhaustive returned 172.8 deg of a full turn
    with a 171 deg backward jump, ordered returned 410.8 deg cleanly, with a
    tighter single-axis fit (rms_rot 0.32 vs 2.78 deg) and 5.7x less matching.
    The cost is that the turn's ends never meet, so the arc drifts rather than
    closing — much the cheaper error, and what the single-axis fit regularises.

    Exhaustive matching is affordable below a few hundred images (it is
    quadratic in image count) and strictly stronger than sequential, because it
    also matches the frames that are far apart in TIME but adjacent in SPACE —
    the ones that close an orbit. Sequential matching cannot see those at all,
    so an orbit's two ends never meet and the mapper is free to fold the loop
    back on itself; on a repetitive, near-planar background it usually does.
    Being in capture order is therefore not a reason to prefer sequential: only
    image count is.

    Vocab-tree is the scalable answer above that, but pycolmap ships no
    vocabulary tree — ``vocab_tree_path`` defaults to '.' — so it is never
    chosen automatically; pass --matching vocabtree with --vocab-tree to use a
    downloaded one. Large sets therefore fall back to sequential+quadratic,
    which reaches distant frames through the quadratic overlap and leans on the
    loop-closure pass (stage D) to join the ends.
    """
    if turntable:
        return "ordered"
    return "exhaustive" if n_images < 300 else "sequential"


def _maskrcnn_detector():
    """A ``detect(path) -> [(class name, score, boolean mask), ...]`` backed by
    torchvision's Mask R-CNN with COCO weights.

    Kept behind this factory so build_subject_masks takes the detector as an
    argument: the subject-choosing and fallback logic is then testable without
    torch, and a better segmenter can be swapped in without touching it.
    """
    import cv2

    try:
        import torch
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_V2_Weights,
            maskrcnn_resnet50_fpn_v2,
        )
    except ImportError as e:
        raise SystemExit(
            "--mask-subject needs torchvision (Mask R-CNN): "
            f"pip install torchvision  [{e}]"
        )

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    names = weights.meta["categories"]

    def detect(path):
        bgr = cv2.imread(str(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
        with torch.no_grad():
            out = model([t])[0]
        return [
            (
                names[out["labels"][j]],
                float(out["scores"][j]),
                out["masks"][j, 0].cpu().numpy() > 0.5,
            )
            for j in range(len(out["scores"]))
        ]

    # The vocabulary, so build_subject_masks can reject a --mask-classes name
    # this segmenter can never emit. COCO's list is padded with
    # "__background__" and "N/A" placeholders that are not askable classes.
    detect.labels = frozenset(
        n for n in names if n not in ("__background__", "N/A")
    )
    return detect


# True synonyms, where the user's word shares no stem with COCO's. Edit
# distance cannot reach these ('mug' scores 0.33 against 'cup') and neither
# can the compound rule below.
_CLASS_SYNONYMS = {
    "mug": "cup",
    "tumbler": "cup",
    "mobile": "cell phone",
    "smartphone": "cell phone",
    "jar": "vase",
    "flask": "bottle",
    "controller": "remote",
}


def _near_misses(name, known):
    """Up to three classes the user plausibly meant, best first.

    Three tiers, because fuzzy matching alone is actively misleading here: on
    the real 80-class list difflib answers 'plant' with 'elephant' (0.62) and
    'airplane' (0.62) before 'potted plant' (0.59), and 'phone' with 'spoon'
    and 'horse'. So synonyms and COCO's compound names are resolved first, and
    edit distance is left to do only what it is good at — plain typos.
    """
    import difflib

    q = name.strip().lower()
    hits = []
    syn = _CLASS_SYNONYMS.get(q)
    if syn and syn in known:
        hits.append(syn)
    # COCO compounds its names — 'cell phone', 'potted plant', 'wine glass',
    # 'sports ball', 'teddy bear' — so naming the head noun alone is a
    # systematic miss rather than a slip.
    hits += [c for c in sorted(known) if q in c.lower().split() and c not in hits]
    # Unambiguous typos only: 'bowls'/'bowl' is 0.89, 'scisors'/'scissors'
    # 0.93, while the noise above sits at 0.6.
    hits += [
        c for c in difflib.get_close_matches(name, sorted(known), n=3, cutoff=0.7)
        if c not in hits
    ]
    return hits[:3]


def _reject_unknown_classes(wanted, known):
    """Stop on a --mask-classes name the segmenter can never emit.

    Without this the run does not fail, which is worse than failing: no
    detection ever matches the name, every frame falls through to the all-pass
    mask at the bottom of build_subject_masks, and the reconstruction finishes
    with masking silently doing nothing — re-arming the planar-background fold
    that --mask-subject exists to prevent, and handing the trainer all-pass
    loss masks on the --turntable path.

    ``known`` of None means the detector does not publish a vocabulary (an
    injected test double), and there is nothing to check against.

    The near-miss suggestions are the point: 'mug' for 'cup', 'phone' for
    'cell phone' and 'plant' for 'potted plant' are each one keystroke from
    working, and listing 80 COCO classes would bury that.
    """
    if not known:
        return
    unknown = sorted(w for w in wanted if w not in known)
    if not unknown:
        return
    lines = []
    for u in unknown:
        near = _near_misses(u, known)
        lines.append(
            f"  {u!r}"
            + (f" — did you mean {', '.join(repr(c) for c in near)}?" if near else "")
        )
    raise SystemExit(
        "--mask-classes: the segmenter has no such class:\n"
        + "\n".join(lines)
        + "\n\nIt can only emit its own COCO classes, so an unknown name "
        "matches nothing and leaves every frame unmasked — the run would "
        "finish with masking doing nothing at all. Pass --mask-region instead "
        "when the subject has no COCO class."
    )


def build_subject_masks(images_dir: Path, mask_dir: Path, classes=None,
                        dilate: int = 15, detector=None, required: bool = True):
    """Segment the capture's subject in every frame; write COLMAP masks.

    Why this exists: an object photographed on a floor or a desk is a small
    non-planar island in a large PLANE. Relative pose from a plane is
    two-fold ambiguous, so when the background supplies most of the inliers
    the mapper is free to fold the orbit — and because the false matches are
    homography-consistent, matching HARDER (exhaustive) reinforces the wrong
    solution instead of breaking it. Restricting features to the subject
    removes the degenerate majority rather than out-voting it.

    Masks follow COLMAP's convention: ``<mask_dir>/<image name>.png``, where a
    zero pixel means "ignore this pixel". Returns the number of frames where
    nothing was detected (those get an all-pass mask, so they behave exactly
    as they would without masking rather than dropping out of the capture).

    Needs torchvision (Mask R-CNN, COCO weights) — an optional dependency,
    since every other stage here runs on pycolmap/scipy/cv2 alone.
    """
    import cv2
    import numpy as np

    frames = sorted(images_dir.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames to mask in {images_dir}")
    mask_dir.mkdir(parents=True, exist_ok=True)
    detect = detector or _maskrcnn_detector()

    # Which class IS the subject? Asking per frame would let a glimpsed
    # background object become the subject in the frames where the real one
    # scores lower, so it is decided once, by vote over frames spread across
    # the whole capture.
    if classes:
        wanted = set(classes)
        _reject_unknown_classes(wanted, getattr(detect, "labels", None))
    else:
        votes = {}
        step = max(1, len(frames) // MASK_VOTE_FRAMES)
        for path in frames[::step][:MASK_VOTE_FRAMES]:
            best, best_score = None, MASK_SCORE
            for label, score, _ in detect(path):
                if score > best_score and label in SUBJECT_CLASSES:
                    best, best_score = label, score
            if best:
                votes[best] = votes.get(best, 0) + 1
        sampled = min(len(frames), MASK_VOTE_FRAMES)
        winner = max(votes, key=votes.get) if votes else None
        if not votes or votes[winner] < MASK_VOTE_SHARE * sampled:
            # Why the vote failed, phrased for whoever has to act on it: an
            # outright miss and a class that scraped a plurality need
            # different fixes, and "no recognisable subject" would be a lie
            # about the second.
            why = (
                f"no recognisable subject to mask (looked for: "
                f"{', '.join(SUBJECT_CLASSES)})"
                if not votes else
                f"no subject recognisable across the capture — '{winner}' led "
                f"the vote but carried only {votes[winner]}/{sampled} sampled "
                f"frames, which is one FACE of something rather than a subject"
            )
            if not required:
                # --turntable IMPLIES masking rather than asking for it, and a
                # subject COCO cannot name is not a reason to refuse the run:
                # measured on a plywood house, solving unmasked gave 219/219
                # frames and a 228 deg turn against 162 and 43 deg for the same
                # capture masked to a fixed region. Say so and carry on.
                print(
                    f"WARNING: {why}, so this turntable is "
                    "being solved UNMASKED. That is often fine, and on a "
                    "manual turntable it is often BETTER: the platter "
                    "co-rotates with the subject, so its texture is signal. "
                    "If the solve folds, pass --mask-region around the "
                    "PLATTER — everything that turns — not around the "
                    "subject alone.",
                    flush=True,
                )
                # mask_dir was created before the vote could run; an empty
                # <output>_masks beside the npz reads as "masks were made".
                try:
                    mask_dir.rmdir()
                except OSError:
                    pass
                return None
            raise SystemExit(
                f"--mask-subject found {why}. Pass "
                "--mask-classes with the COCO class name of the subject, "
                "--mask-region with a box around it, or drop --mask-subject."
            )
        wanted = {winner}
        log(f"masking subject class '{winner}' ({votes[winner]}/{sampled} votes)")

    kernel = np.ones((dilate, dilate), np.uint8) if dilate > 0 else None
    kept, missing = [], 0
    for n, path in enumerate(frames):
        if n % 50 == 0:
            progress(6 + int(6 * n / len(frames)),
                     f"SfM poses: masking subject ({n}/{len(frames)})…")
        m = None
        for label, score, mask in detect(path):
            if score > MASK_SCORE and label in wanted:
                m = mask.copy() if m is None else (m | mask)
        if m is None or not m.any():
            # Nothing found: pass the whole frame rather than blinding it.
            # The frame's own size is only needed on this branch, since a
            # detection already carries a full-resolution mask.
            m = np.ones(cv2.imread(str(path)).shape[:2], bool)
            missing += 1
        img = m.astype(np.uint8) * 255
        if kernel is not None:
            # Grow slightly: SIFT keypoints ON the silhouette carry the
            # subject's outline against the background, and a pixel-tight
            # mask throws exactly those away.
            img = cv2.dilate(img, kernel)
        cv2.imwrite(str(mask_dir / (path.name + ".png")), img)
        kept.append((img > 0).mean())

    log(f"subject masks: {len(frames)} frames, keeping "
        f"{100 * float(np.median(kept)):.0f}% of the frame (median)"
        + (f"; {missing} with no detection (left unmasked)" if missing else ""))
    if missing > len(frames) // 2:
        print(
            f"WARNING: no subject found in {missing}/{len(frames)} frames — "
            "masking is doing little here. Check --mask-classes, or drop "
            "--mask-subject.",
            flush=True,
        )
    return missing


def parse_mask_region(spec: str):
    """Parse ``--mask-region`` into a polygon of normalised (x, y) points.

    Four numbers are a box's opposite corners (x0,y0,x1,y1); six or more are a
    polygon, three points minimum. Coordinates are FRACTIONS of the frame
    (0,0 top-left, 1,1 bottom-right) rather than pixels, so one region survives
    a re-ingest at a different --max-edge and can be read off a preview of any
    size.

    Every rejection here is a SystemExit rather than a fallback, because the
    failure this option exists to avoid is a mask that silently keeps
    everything: a region that cannot be parsed must stop the run, not quietly
    become a no-op.
    """
    try:
        vals = [float(v) for v in spec.replace(";", ",").split(",") if v.strip()]
    except ValueError:
        raise SystemExit(f"--mask-region: not a list of numbers: {spec!r}")
    if len(vals) % 2:
        raise SystemExit(
            f"--mask-region takes x,y pairs, so an even count — got {len(vals)}"
        )
    if len(vals) < 4:
        raise SystemExit(
            "--mask-region needs a box (4 numbers: x0,y0,x1,y1) or a polygon "
            f"(6 or more) — got {len(vals)}"
        )
    if not all(0.0 <= v <= 1.0 for v in vals):
        raise SystemExit(
            "--mask-region coordinates are fractions of the frame, so every "
            f"value must be in [0, 1] — got {spec!r}"
        )
    if len(vals) == 4:
        x0, y0, x1, y1 = vals
        if x1 <= x0 or y1 <= y0:
            raise SystemExit(
                "--mask-region box is x0,y0,x1,y1 with x0<x1 and y0<y1 — got "
                f"{spec!r}"
            )
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    pts = list(zip(vals[::2], vals[1::2]))
    # Shoelace area: a polygon whose points are collinear encloses nothing and
    # would mask the whole capture away.
    n = len(pts)
    area = 0.5 * abs(sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    ))
    if area < 1e-6:
        raise SystemExit(f"--mask-region polygon encloses no area: {spec!r}")
    return pts


def build_region_masks(images_dir: Path, mask_dir: Path, region):
    """Write COLMAP masks keeping one FIXED image region in every frame.

    build_subject_masks can only ask for what COCO knows, which is the 19
    tabletop objects in SUBJECT_CLASSES. A machine part, a casting, a rock or
    a printed bracket is not among them and there is no label to pass to
    --mask-classes either: the detector never emits it, every frame falls
    through to an all-pass mask, and the run continues with masking silently
    doing nothing. A stated region is the answer for those subjects, and it
    needs no torchvision at all.

    It is sound precisely when the CAMERA is fixed — a turntable — because the
    subject then occupies the same part of every frame by construction. On a
    walk-around the subject travels through the frame and a fixed region would
    mask it away; that is what build_subject_masks is for.

    It also does something the class mask cannot: the region MAY deliberately
    include the platter, so a patterned cloth or sheet of newspaper laid ON the
    turntable — the standard fix for a symmetric subject folding the solve —
    reaches matching instead of being segmented away with the background.

    ``region`` is a polygon of normalised (x, y) from parse_mask_region. No
    dilation is applied: unlike a pixel-tight instance mask, a stated region is
    already the boundary the user meant.
    """
    import cv2
    import numpy as np

    frames = sorted(images_dir.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames to mask in {images_dir}")
    mask_dir.mkdir(parents=True, exist_ok=True)

    by_size = {}  # rasterise once per distinct frame size, not once per frame
    for n, path in enumerate(frames):
        if n % 50 == 0:
            progress(6 + int(6 * n / len(frames)),
                     f"SfM poses: masking region ({n}/{len(frames)})…")
        h, w = cv2.imread(str(path)).shape[:2]
        img = by_size.get((h, w))
        if img is None:
            img = np.zeros((h, w), np.uint8)
            pts = np.array(
                [(round(x * (w - 1)), round(y * (h - 1))) for x, y in region],
                np.int32,
            )
            cv2.fillPoly(img, [pts], 255)
            by_size[(h, w)] = img
        cv2.imwrite(str(mask_dir / (path.name + ".png")), img)

    frac = 100 * float(np.mean([(m > 0).mean() for m in by_size.values()]))
    log(f"region masks: {len(frames)} frames, keeping {frac:.0f}% of the frame"
        + (f" ({len(by_size)} frame sizes)" if len(by_size) > 1 else ""))
    if frac < 2:
        print(
            f"WARNING: the region keeps only {frac:.1f}% of the frame — too "
            "little for SIFT to find matchable structure in. Widen "
            "--mask-region.",
            flush=True,
        )
    return frac


def auto_match_window(n_frames: int) -> int:
    """Frames either side that --matching ordered may match, chosen from the
    frame count rather than fixed.

    The window counts FRAMES but what matters is the ANGLE it spans, and on a
    turntable angle IS baseline. A fixed window therefore breaks the moment the
    frame rate changes: the same House capture ingested at 15.5 fps (219
    frames) solved cleanly with a window of 20 spanning ~38 deg, while at 30 fps
    (442 frames) that window spanned ~19 deg with only ~0.8 deg between
    neighbours — too little parallax to triangulate against. Measured, same
    footage, window 20 vs 40 at 442 frames:

        20 -> 231/442 registered, 96 deg of turn, rms_rot 8.00 deg
        40 -> 442/442 registered, 412 deg of turn, rms_rot 0.18 deg

    So the window scales with the frame count, targeting roughly 30 deg of
    rotation for a single-turn capture. Clamped at both ends: too narrow cannot
    triangulate, and too wide starts reaching far enough round to let a
    symmetric subject fold, which is what ordered matching exists to prevent.
    """
    return max(12, min(60, round(n_frames / 12)))


def build_ordered_pairs(names, window: int, out_path: Path) -> int:
    """Match pairs within ``window`` frames of each other, and no others.

    The fold that ruins a symmetric subject is a LONG-RANGE false match. A view
    of the back of a two-gabled house really does look like the front, so the
    mapper is free to place frame 150 at frame 30's angle, and no amount of
    feature evidence can rule it out — measured on the House capture as a solve
    that climbed to +79 deg, jumped BACK 136 deg, climbed again, and covered
    173 deg of a full turn while registering all 219 frames.

    Frame ORDER is the evidence matching cannot supply: on an ordered capture
    the subject cannot be at two angles at once, so a pair 120 frames apart is
    not a neighbour however well it matches. Deleting those pairs from the
    graph makes the fold STRUCTURALLY IMPOSSIBLE rather than merely unlikely —
    the mapper never sees the edge it would have folded along.

    Note this is strictly stronger than ``--matching sequential``, which sets
    ``quadratic_overlap`` and therefore deliberately adds pairs at 1, 2, 4, 8,
    16... frames apart. Those long reaches are the fold's route back in.

    The cost is that the two ends of a full turn never meet, so the arc drifts
    instead of closing. On a turntable that is much the cheaper error: drift is
    what the single-axis fit regularises (--turntable-snap), while a fold
    silently merges two different sides of the subject into one.
    """
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, min(i + window + 1, len(names))):
            pairs.append(f"{names[i]} {names[j]}")
    if not pairs:
        raise SystemExit(
            f"--matching ordered: {len(names)} frames with a window of "
            f"{window} yields no pairs"
        )
    out_path.write_text("\n".join(pairs) + "\n")
    return len(pairs)


def extract_and_match(db: Path, images_dir: Path, model: str, params,
                      strategy: str = "sequential", vocab_tree=None,
                      mask_dir=None, window: int = 20):
    import pycolmap

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = model
    if params is not None:
        reader.camera_params = ",".join(str(p) for p in params)
    extra = {}
    if mask_dir is not None:
        reader.mask_path = str(mask_dir)
        # The subject is a fifth of the frame, so the default detector
        # settings yield a fifth of the usual features. Lower the contrast
        # floor and raise the cap to put the budget back onto the subject —
        # smooth glazed ceramic and printed detail are exactly what the
        # default threshold discards.
        ext = pycolmap.FeatureExtractionOptions()
        ext.sift.peak_threshold = 0.004
        ext.sift.max_num_features = 16384
        extra["extraction_options"] = ext

    # Whether this is the slow part depends entirely on how pycolmap was
    # built. A pip wheel is CPU-only; a conda-forge cuda build does the same
    # 150-frame extraction in 3 s instead of 31 s, and matching ~5x faster.
    # Saying "CPU, slow" regardless told half the users the wrong thing.
    gpu = getattr(pycolmap, "has_cuda", False)
    progress(14, "SfM poses: extracting SIFT features"
             + (" (GPU)…" if gpu else " (CPU — no CUDA in this pycolmap build)…"))
    pycolmap.extract_features(
        database_path=db,
        image_path=images_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        **extra,
        reader_options=reader,
    )
    log(f"feature extraction done (camera model {model})")

    if strategy == "ordered":
        names = [p.name for p in sorted(images_dir.glob("*.jpg"))]
        pairs_path = db.parent / "pairs_ordered.txt"
        if window <= 0:
            window = auto_match_window(len(names))
            log(f"match window auto: {window} frames for {len(names)} frames")
        n = build_ordered_pairs(names, window, pairs_path)
        progress(30, f"SfM poses: ordered matching ({n:,} pairs, window "
                     f"{window} frames)…")
        pairing = pycolmap.ImportedPairingOptions()
        pairing.match_list_path = str(pairs_path)
        pycolmap.match_image_pairs(database_path=db, pairing_options=pairing)
    elif strategy == "exhaustive":
        progress(30, "SfM poses: exhaustive matching…")
        pycolmap.match_exhaustive(database_path=db)
    elif strategy == "vocabtree":
        pairing = pycolmap.VocabTreePairingOptions()
        if vocab_tree:
            pairing.vocab_tree_path = Path(vocab_tree)
        progress(30, "SfM poses: vocab-tree matching…")
        pycolmap.match_vocabtree(database_path=db, pairing_options=pairing)
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 20
        pairing.quadratic_overlap = True
        pairing.loop_detection = False
        progress(30, "SfM poses: sequential matching (overlap 20)…")
        pycolmap.match_sequential(database_path=db, pairing_options=pairing)
    log(f"matching done ({strategy})")


def run_mapping(db: Path, images_dir: Path, sparse_dir: Path, label: str,
                refine_intrinsics: bool = False, mapper: str = "incremental"):
    """Map the matched database; returns the best reconstruction (most
    registered images) or None. Whichever mapper runs, the other is the
    fallback if it produces nothing.

    Incremental is the default. Measured on the CupPhotos capture (a mug
    orbited on a plank floor, 165 frames, identical exhaustive match
    database), GLOMAP solved 60 deg of viewpoint spread from a real ~270 deg
    walk-around and reported a degenerate 0.001 px mean reprojection error;
    incremental solved 90 deg from the same matches at a credible 0.877 px.
    Global rotation averaging commits to a consensus over all pairs at once,
    so on a scene where many pairs are ambiguous — a repetitive near-planar
    background, or a subject with rotational symmetry — the wrong consensus
    wins outright. Incremental grows one image at a time and can at least
    reject a view that does not fit what it has already built.

    ``refine_intrinsics`` lets bundle adjustment solve for focal length and
    distortion. It must stay OFF for LiDAR scans — there the factory
    calibration is ground truth and refining it would drift away from the model
    the cloud was coloured with — and ON for phone captures, where the EXIF
    focal length is only a starting guess.
    """
    import pycolmap

    sparse_dir.mkdir(parents=True, exist_ok=True)

    def _incremental():
        log(f"{label}: incremental mapping…")
        iopts = pycolmap.IncrementalPipelineOptions()
        if refine_intrinsics:
            iopts.ba_refine_focal_length = True
            iopts.ba_refine_extra_params = True
        return pycolmap.incremental_mapping(
            database_path=db, image_path=images_dir, output_path=sparse_dir,
            options=iopts,
        )

    def _glomap():
        log(f"{label}: global mapping (GLOMAP)…")
        gopts = pycolmap.GlobalPipelineOptions()
        if refine_intrinsics:
            ba = gopts.mapper.bundle_adjustment
            ba.refine_focal_length = True
            ba.refine_extra_params = True
        return pycolmap.global_mapping(
            database_path=db, image_path=images_dir, output_path=sparse_dir,
            options=gopts,
        )

    first, second = (
        (_glomap, _incremental) if mapper == "glomap" else (_incremental, _glomap)
    )
    recs = None
    try:
        recs = first()
    except Exception as e:
        log(f"{label}: {mapper} mapping failed ({e!r}); falling back")
    if not recs:
        try:
            recs = second()
        except Exception as e:
            log(f"{label}: fallback mapping failed too ({e!r})")
    if not recs:
        return None
    best_id, best = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    for rid, rec in recs.items():
        log(
            f"{label}: model {rid}: {rec.num_reg_images()} registered, "
            f"{rec.num_points3D()} pts, mean reproj "
            f"{rec.compute_mean_reprojection_error():.2f} px"
        )
    return best


# A solve that registers less than this share of the frames has fragmented:
# the mapper built several pieces and run_mapping kept only the largest.
FRAGMENTED_FRAC = 0.95
# A mapper that registers less than this share of its rival's frames is not a
# candidate, however consistent: a small easy subset always scores well.
MIN_COVERAGE_VS_RIVAL = 0.8
# Below this gap in median rotation error the two solves are equally good,
# and the one that registered more frames wins.
ROT_TIE_DEG = 0.25


def two_view_rotation_pairs(db: Path, max_pairs: int = 1500,
                            min_inliers: int = 50, seed: int = 0):
    """Sample matched pairs from the database for :func:`rotation_disagreement`:
    ``[(name_a, name_b, keypoints_a, keypoints_b)]`` holding each pair's
    geometrically verified inlier matches.

    These are the raw evidence BOTH mappers were given, so they can referee
    between them. COLMAP does not store relative poses in this database, so
    the rotation is re-estimated per pair from these points.
    """
    import random
    import sqlite3

    import numpy as np

    con = sqlite3.connect(str(db))
    try:
        names = dict(con.execute("SELECT image_id, name FROM images"))
        kps = {
            i: np.frombuffer(d, np.float32).reshape(r, c)[:, :2]
            for i, r, c, d in con.execute(
                "SELECT image_id, rows, cols, data FROM keypoints")
        }
        rows = list(con.execute(
            "SELECT pair_id, rows, data FROM two_view_geometries "
            "WHERE rows >= ?", (min_inliers,)))
    finally:
        con.close()
    random.Random(seed).shuffle(rows)
    big = 2147483647  # COLMAP's pair_id = id1 * big + id2
    out = []
    for pid, n, data in rows[:max_pairs]:
        a, b = pid // big, pid % big
        m = np.frombuffer(data, np.uint32).reshape(n, 2)
        out.append((names[a], names[b], kps[a][m[:, 0]], kps[b][m[:, 1]]))
    return out


def rotation_disagreement(rec, pairs):
    """Median angle (deg) between each pair's relative rotation in ``rec`` and
    the one its own matches give on their own, over the pairs ``rec`` registered
    both frames of. Returns ``(median_deg, n_pairs)``, or ``(inf, 0)``.

    Reprojection error cannot tell a bent solve from a straight one: on the
    Clinic walk-through incremental mapping scored 0.93 px and GLOMAP 0.80 px,
    yet their camera paths differed by 17-87 deg. Pairwise rotations can. There
    GLOMAP agreed with them to 0.5 deg median, while incremental disagreed by
    5 deg for its main piece and 16 deg for a second piece that had also solved
    a wrong focal length (1024 px against ~730).
    """
    import cv2
    import numpy as np
    from scipy.spatial.transform import Rotation

    cam = next(iter(rec.cameras.values()))
    rot = {}
    for img in rec.images.values():
        rot[img.name] = Rotation.from_matrix(img.cam_from_world().rotation.matrix())
    thresh = 1.0 / float(cam.params[0])  # 1 px, in normalised coordinates
    errs = []
    for na, nb, ka, kb in pairs:
        if na not in rot or nb not in rot:
            continue
        pa = np.asarray(cam.cam_from_img(ka.astype(np.float64)))
        pb = np.asarray(cam.cam_from_img(kb.astype(np.float64)))
        E, mask = cv2.findEssentialMat(pa, pb, np.eye(3), cv2.RANSAC, 0.999, thresh)
        if E is None or E.shape != (3, 3):
            continue
        n_in, R, _, _ = cv2.recoverPose(E, pa, pb, np.eye(3), mask=mask)
        if n_in < 30:
            continue
        solved = rot[nb] * rot[na].inv()
        errs.append(np.degrees((solved.inv() * Rotation.from_matrix(R)).magnitude()))
    if not errs:
        return float("inf"), 0
    return float(np.median(errs)), len(errs)


def pick_solve(candidates):
    """Choose between ``[(label, n_registered, median_rot_err_deg)]``: the most
    rotation-consistent solve among those that registered a fair share of the
    frames; when two are within ROT_TIE_DEG, the one with more frames.
    Returns the chosen label."""
    most = max(n for _, n, _ in candidates)
    fair = [c for c in candidates if c[1] >= MIN_COVERAGE_VS_RIVAL * most]
    best_err = min(e for _, _, e in fair)
    close = [c for c in fair if c[2] <= best_err + ROT_TIE_DEG]
    return max(close, key=lambda c: (c[1], -c[2]))[0]


# A point in an ordered capture whose matches across it fall below this share
# of the capture's median is a weak link. Measured as the summed inliers of
# every verified pair that straddles the point: the Clinic doorway fell to
# 0.03-2.7% of the median at 4 and 10 fps (blurry, near-pure-rotation pan past
# a white door), while four good captures — CupPhotos, two House turntable
# videos, a 334-frame walk-around — never dropped below 7.7%, and that only at
# their very ends.
WEAK_LINK_FRAC = 0.04
# The first and last cuts are thin by construction (one side holds a single
# frame), so a weak link there costs a few frames, not a misaligned half.
WEAK_LINK_EDGE_FRAC = 0.02


def bridge_strength(db: Path):
    """For an ORDERED capture: ``(names, strength)`` where ``strength[c]`` is
    the summed verified inliers of every pair with one frame at or before
    ``names[c]`` and the other after it — how firmly the capture's two halves
    either side of that point are tied together."""
    import sqlite3

    import numpy as np

    con = sqlite3.connect(str(db))
    try:
        by_id = dict(con.execute("SELECT image_id, name FROM images"))
        rows = list(con.execute(
            "SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"))
    finally:
        con.close()
    names = sorted(by_id.values())
    pos = {n: i for i, n in enumerate(names)}
    big = 2147483647  # COLMAP's pair_id = id1 * big + id2
    diff = np.zeros(len(names) + 1)
    for pid, n in rows:
        i, j = sorted((pos[by_id[pid // big]], pos[by_id[pid % big]]))
        diff[i] += n
        diff[j] -= n
    return names, np.cumsum(diff)[: max(len(names) - 1, 0)]


def weak_links(names, strength, frac: float = WEAK_LINK_FRAC,
               edge_frac: float = WEAK_LINK_EDGE_FRAC):
    """Runs of weak cuts as ``[(last name before, first name after,
    weakest share of the median)]``, ignoring the capture's ends."""
    import numpy as np

    s = np.asarray(strength, dtype=float)
    if len(s) < 10:
        return []
    med = float(np.median(s))
    if med <= 0:
        return []
    edge = max(3, int(edge_frac * len(names)))
    weak = [c for c in range(edge, len(s) - edge) if s[c] < frac * med]
    out = []
    for c in weak:
        if out and c == out[-1][1] + 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return [
        (names[a], names[b + 1], float(s[a:b + 1].min() / med)) for a, b in out
    ]


def warn_weak_links(db: Path):
    """Warn about the points where an ordered capture barely holds together.

    Whichever mapper runs, the parts either side of such a point are tied by
    a handful of matches. The solve then either drops one part or places it
    at the wrong angle, and nothing in its own statistics shows it. On Clinic
    GLOMAP registered every frame at a clean 0.8 px reprojection error, yet
    the room sat 45 deg off the only link to the hallway. Returns the links.
    """
    names, strength = bridge_strength(db)
    links = weak_links(names, strength)
    for before, after, share in links:
        print(
            f"WARNING: frames {before} → {after} are barely connected "
            f"({share:.1%} of the capture's typical overlap). The parts of the "
            "scene either side of this point may be MISALIGNED (rotated or "
            "shifted relative to each other) or missing. This is a capture "
            "problem, so no solver can fix it: usually a fast pan, motion "
            "blur or a blank surface filling the frame. Re-shoot that stretch "
            "slowly, stepping through rather than turning on the spot, and "
            "keep textured detail in view.",
            flush=True,
        )
    if names and not links:
        log("capture continuity: no weak links")
    return links


def _try_other_mapper(rec, db, images_dir, ws, refine, args, n_frames):
    """A fragmented pass 1: solve again with the other mapper and keep
    whichever agrees better with the pairwise rotations. Sets ``args.mapper``
    to the winner, so pass 2 uses it too.

    Fragmenting is the signature of incremental mapping drifting: it grows the
    model one frame at a time, a bent path stops accepting new frames, and the
    rest of the capture starts a second model that run_mapping then discards.
    On the Clinic walk-through it kept 249 of 291 frames, while GLOMAP placed
    all 291 and agreed with the pairwise evidence ten times better. Neither
    mapper wins everywhere (on CupPhotos GLOMAP was the one that failed), so
    this measures rather than assumes.
    """
    other = "glomap" if args.mapper == "incremental" else "incremental"
    log(f"pass 1 registered {rec.num_reg_images()}/{n_frames} frames — "
        f"fragmented; trying {other} as well")
    progress(45, f"SfM poses: fragmented — trying {other} mapping…")
    alt = run_mapping(db, images_dir, ws / "sparse_alt", f"pass 1 ({other})",
                      refine, other)
    if alt is None:
        return rec
    progress(52, "SfM poses: comparing the two solves…")
    pairs = two_view_rotation_pairs(db)
    scored = []
    for label, r in ((args.mapper, rec), (other, alt)):
        err, n = rotation_disagreement(r, pairs)
        log(f"{label}: {r.num_reg_images()}/{n_frames} frames, median "
            f"rotation disagreement {err:.2f} deg over {n} pairs")
        scored.append((label, r.num_reg_images(), err))
    chosen = pick_solve(scored)
    log(f"keeping the {chosen} solve")
    if chosen == other:
        args.mapper = other
        return alt
    return rec


def poses_from_rec(rec):
    """name → (R world→cam 3x3, camera centre 3), plus the bundle-adjusted
    camera params and the model name they belong to."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    out = {}
    for img in rec.images.values():
        p = img.cam_from_world()
        q, t = p.rotation.quat, p.translation  # quat is x,y,z,w (scipy order)
        R = Rotation.from_quat(q).as_matrix()
        out[img.name] = (R, -R.T @ np.asarray(t))
    cam = next(iter(rec.cameras.values()))
    model = getattr(cam.model, "name", str(cam.model))
    return out, np.asarray(cam.params, dtype=float), model


# ── Stage C: sim(3) alignment to the LiDAR trajectory ────────────────


def umeyama(src, dst):
    """Least-squares sim(3): dst ≈ s·R·src + t.  Returns s, R, t."""
    import numpy as np

    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    t = mu_d - s * R @ mu_s
    return s, R, t


def align_to_lidar(sfm, ts_by_name, traj_path: Path, T_lidar_cam, cam_time_offset):
    """Fit sim(3) SfM→LiDAR-world and return per-frame rigid world→cam
    viewmats in the LiDAR frame, with SfM-failure frames dropped.

    Returns (names, viewmats, export_mask, stats_str) — names/viewmats are the
    EXPORTED (kept) frames only.
    """
    import numpy as np

    names = sorted(n for n in sfm if n in ts_by_name)
    if len(names) < 10:
        raise RuntimeError(f"only {len(names)} SfM poses have timestamps")

    traj = np.load(traj_path)
    traj_ts = traj["ts"].astype(np.int64)
    poses = traj["poses"].astype(np.float64)
    order = np.argsort(traj_ts)
    traj_ts, poses = traj_ts[order], poses[order]
    offset_ns = int(round(cam_time_offset * 1e9))

    def lidar_vm(ts_ns):
        P, _, _ = _interp_pose(traj_ts, poses, ts_ns + offset_ns)
        return T_lidar_cam @ np.linalg.inv(P)

    c_sfm = np.array([sfm[n][1] for n in names])
    lidar_vms = np.array([lidar_vm(ts_by_name[n]) for n in names])
    c_lidar = np.array([-vm[:3, :3].T @ vm[:3, 3] for vm in lidar_vms])

    # Iterative median-based trimming: gross SfM failures (100s of metres)
    # would poison a mean/std trim, and moderate residuals are expected —
    # they're the LiDAR drift we're measuring, so the threshold must scale
    # with the data.
    keep = np.ones(len(names), bool)
    for _ in range(10):
        s, R, t = umeyama(c_sfm[keep], c_lidar[keep])
        res = np.linalg.norm((s * (R @ c_sfm.T).T + t) - c_lidar, axis=1)
        thresh = max(5 * np.median(res[keep]), 0.05)
        new_keep = res < thresh
        if new_keep.sum() < 10 or (new_keep == keep).all():
            break
        keep = new_keep
    stats = (
        f"sim(3) scale {s:.4f}, {len(names) - keep.sum()} outliers trimmed; "
        f"centre residual (= LiDAR drift) med {np.median(res) * 100:.1f} cm, "
        f"p90 {np.percentile(res, 90) * 100:.1f} cm"
    )
    log(stats)

    # Frames wildly off the fit are SfM failures, not drift — keep them out
    # of the export so the trainer never sees a garbage pose. (Moderate
    # residuals stay: there the SfM pose is the correction we want.)
    export = res < max(20 * np.median(res[keep]), 10.0)
    if (~export).any():
        bad = [n for n, e in zip(names, export) if not e]
        log(f"excluding {len(bad)} SfM-failure frame(s): {bad[:10]}")

    # Rigid per-image world→cam in the LiDAR frame: rotate the SfM pose into
    # the aligned frame, put the camera at the similarity-transformed centre.
    viewmats = []
    for n in names:
        R_i, c_i = sfm[n]
        R_new = R_i @ R.T
        c_w = s * R @ c_i + t
        vm = np.eye(4)
        vm[:3, :3] = R_new
        vm[:3, 3] = -R_new @ c_w
        viewmats.append(vm)
    viewmats = np.array(viewmats)

    exp_names = [n for n, e in zip(names, export) if e]
    return exp_names, viewmats[export], stats


# ── Stage C alternative: no LiDAR to align to ────────────────────────


def _capture_is_ordered(scan: Path) -> bool:
    """Whether an image set's frames are in genuine capture order.

    Video always is. Stills are only if the ingest could sort them by EXIF
    capture time — a folder of photos with the timestamps stripped is an
    unordered pile, and sequential matching would connect the wrong frames.
    """
    import json

    meta = scan / "ingest.json"
    if meta.is_file():
        try:
            if json.loads(meta.read_text()).get("source_kind") == "video":
                return True
        except (OSError, ValueError):
            pass
    try:
        rows = frame_source.read_manifest(scan)
    except (OSError, KeyError):
        return True
    return bool(rows) and all(r.get("exif_ts") for r in rows)


def viewmats_from_sfm(sfm, ts_by_name):
    """World→cam viewmats straight from the reconstruction, no alignment.

    The LiDAR path fits a sim(3) onto the odometry trajectory, which both puts
    the poses in a metric, levelled frame AND exposes gross SfM failures as
    huge residuals. Neither is available here: an image set's only poses are
    these, so the output frame is the SfM's own — unitless, arbitrarily
    oriented — and outliers have to be found without a reference.

    The substitute is a robust spatial test: a camera hundreds of median-
    deviations from the cloud of camera centres did not merely drift, it failed
    to register correctly, and feeding it to the trainer puts a ghost view into
    the model.
    """
    import numpy as np

    names = sorted(n for n in sfm if n in ts_by_name)
    if len(names) < MIN_FRAMES:
        raise RuntimeError(
            f"only {len(names)} SfM poses matched the manifest — the "
            "reconstruction did not register enough frames"
        )
    centres = np.array([sfm[n][1] for n in names])
    med = np.median(centres, axis=0)
    dist = np.linalg.norm(centres - med, axis=1)
    mad = np.median(np.abs(dist - np.median(dist)))
    if mad > 0:
        keep = dist < (np.median(dist) + 20 * max(mad, 1e-9))
    else:
        keep = np.ones(len(names), bool)
    if keep.sum() < 10:  # pathological spread — keep everything, warn instead
        keep = np.ones(len(names), bool)
    if (~keep).any():
        log(f"excluding {int((~keep).sum())} SfM-failure frame(s) far from the "
            "camera cloud")

    viewmats = []
    for n in names:
        R_i, c_i = sfm[n]
        vm = np.eye(4)
        vm[:3, :3] = R_i
        vm[:3, 3] = -R_i @ c_i
        viewmats.append(vm)
    viewmats = np.array(viewmats)

    kept = [n for n, k in zip(names, keep) if k]
    step = np.median(np.linalg.norm(np.diff(centres[keep], axis=0), axis=1))
    stats = (
        f"{len(kept)} poses in the SfM frame (UNITLESS — no LiDAR to scale "
        f"against); median inter-frame step {step:.4f} scene units"
    )
    log(stats)
    return kept, viewmats[keep], stats


# ── Seed cloud from the sparse reconstruction ────────────────────────


def _sparse_points(rec):
    """``(xyz, rgb)`` of the reconstruction's 3D points.

    Points with a long track are kept preferentially — a point seen by two
    images is frequently a mismatch, and seeding the trainer with those puts
    permanent floaters in the splat.
    """
    import numpy as np

    xyz, rgb = [], []
    for pt in rec.points3D.values():
        if pt.track.length() < 3:
            continue
        xyz.append(pt.xyz)
        rgb.append(pt.color)
    if not xyz:  # degenerate reconstruction — take everything rather than fail
        for pt in rec.points3D.values():
            xyz.append(pt.xyz)
            rgb.append(pt.color)
    return (np.asarray(xyz, dtype=np.float64),
            np.asarray(rgb, dtype=np.uint8))


def _write_ply(xyz, rgb, out_path: Path):
    """Coloured binary PLY, written-then-renamed so a crash leaves no
    half-written cloud where an existence check would find one."""
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    data = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    tmp = Path(str(out_path) + ".tmp")
    with open(tmp, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())
    tmp.rename(out_path)
    return n


def export_sparse_cloud(rec, out_path: Path, viewmat_transform=None):
    """Write ``rec.points3D`` as a coloured binary PLY — the splat seed.

    Every COLMAP 3D point carries an RGB colour averaged over the images that
    observed it, which is exactly the (xyz, rgb) form load_coloured_ply wants.
    Typically 50k-500k points: thin next to a LiDAR cloud's millions, but MCMC
    densification grows the model from it.
    """
    xyz, rgb = _sparse_points(rec)
    if viewmat_transform is not None:
        xyz = viewmat_transform(xyz)
    n = _write_ply(xyz, rgb, out_path)
    log(f"seed cloud: {n:,} points → {out_path}")
    return n


# ── Silhouette carving: the shape information matching cannot see ────


def _project(camera, xyz_cam):
    """Pixel coordinates for camera-frame points, and which ones are usable.

    Delegates the lens model to pycolmap so every COLMAP camera (SIMPLE_RADIAL
    for a solved phone lens, OPENCV_FISHEYE for the factory calibration) is
    handled by the same code that solved the poses. Points at or behind the
    principal plane have no projection and are reported unusable rather than
    silently wrapping to a pixel.
    """
    import numpy as np

    front = xyz_cam[:, 2] > 1e-9
    uv = np.full((len(xyz_cam), 2), -1.0)
    if front.any():
        uv[front] = camera.img_from_cam(np.ascontiguousarray(xyz_cam[front]))
    return uv, front


def pycolmap_camera(model, params, width, height):
    """A pycolmap Camera from the npz's ``camera_model``/``camera_params``, so
    the carve projects through the same lens the poses were solved under."""
    import pycolmap

    return pycolmap.Camera(
        model=str(model), width=int(width), height=int(height),
        params=[float(x) for x in params],
    )


def carve_visual_hull(mask_dir: Path, images_dir: Path, names, viewmats,
                      camera, seed_xyz, grid=256, views=72, erode=0):
    """Space-carve the subject from its silhouettes — shape without texture.

    A turntable capture is a tomographic acquisition: every mask is a
    projection of the subject, and the masks stacked against rotation angle
    are its sinogram. The inversion is not filtered backprojection — a
    silhouette is a binary occupancy projection, a max along each ray, not a
    density integral — so the reconstruction is the intersection of the
    back-projected silhouette cones, the visual hull, carved here on a voxel
    grid.

    Why it is worth doing at all: SILHOUETTES NEED NO TEXTURE. On the capture
    this was written for, ~150 frames of blank glazed cylinder yielded zero
    features and zero sparse points, leaving the seed cloud a half-mug — yet
    every one of those frames carries a perfectly good outline. Matching and
    carving read different channels of the same capture, and the channel that
    fails on a featureless subject is not the one used here.

    Limits, in the order they will bite:

    * The hull is an upper bound on shape and CANNOT see concavities. A mug's
      bore carves as solid unless views look down into it; a capture level
      with the rim recovers the outside and fills the inside.
    * It is only as good as its worst silhouette. Masks written for MATCHING
      are dilated (--mask-dilate, 15 px) to keep the silhouette keypoints that
      carry the outline, and that dilation inflates the hull on every ray —
      hence ``erode``, which cancels it.
    * A box from --mask-region is not a silhouette. Carving against one
      returns the box, so this refuses to run on region masks.
    * A subject cut off by the frame edge carves away the part that left the
      frame, since "outside the image" and "outside the subject" are the same
      test here.

    Returns ``(xyz, rgb)`` for the surface voxels, coloured from the frames
    that see them (nearest-surface-per-pixel, so a front surface is not
    painted with the colour of the one behind it).
    """
    import cv2
    import numpy as np

    sel = list(range(0, len(names), max(1, len(names) // max(1, views))))[:views]
    if len(sel) < 8:
        raise SystemExit(
            f"--carve-seed needs at least 8 views to intersect, got {len(sel)}"
        )

    # The volume: the sparse points are subject-only (they were matched under
    # the same masks), so their extent bounds the subject even though they
    # cover only the textured side of it. Percentiles rather than min/max —
    # one stray triangulation must not inflate the grid.
    lo = np.percentile(seed_xyz, 1, axis=0)
    hi = np.percentile(seed_xyz, 99, axis=0)
    mid, half = (hi + lo) / 2, (hi - lo) / 2 * 1.35
    lo, hi = mid - half, mid + half
    step = (hi - lo) / (grid - 1)
    log(f"carving {grid}^3 voxels over a "
        f"{hi[0] - lo[0]:.3f}x{hi[1] - lo[1]:.3f}x{hi[2] - lo[2]:.3f} box "
        f"from {len(sel)} of {len(names)} views")

    occ = np.ones(grid ** 3, bool)
    kernel = np.ones((erode, erode), np.uint8) if erode > 0 else None
    clipped = 0
    for k, i in enumerate(sel):
        if k % 10 == 0:
            progress(93 + int(4 * k / len(sel)),
                     f"SfM poses: carving silhouettes ({k}/{len(sel)})…")
        m = cv2.imread(str(mask_dir / (names[i] + ".png")), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if kernel is not None:
            m = cv2.erode(m, kernel)
        if not m.any():
            continue  # eroded to nothing: this view carries no constraint
        h, w = m.shape
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            clipped += 1
        vm = viewmats[i]
        idx = np.flatnonzero(occ)
        for c0 in range(0, len(idx), 4_000_000):  # bound peak memory
            chunk = idx[c0:c0 + 4_000_000]
            ijk = np.stack(np.unravel_index(chunk, (grid, grid, grid)), axis=1)
            xyz = lo + ijk * step
            cam = xyz @ vm[:3, :3].T + vm[:3, 3]
            uv, front = _project(camera, cam)
            u = np.rint(uv[:, 0]).astype(np.int64)
            v = np.rint(uv[:, 1]).astype(np.int64)
            good = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            inside = np.zeros(len(chunk), bool)
            inside[good] = m[v[good], u[good]] > 0
            occ[chunk[~inside]] = False
        if not occ.any():
            raise SystemExit(
                "--carve-seed: the silhouettes intersect in nothing. The poses "
                "and the masks disagree, which means the solve is wrong rather "
                "than the carve — check the turntable fit first."
            )
    if clipped:
        print(
            f"WARNING: the subject touches the frame edge in {clipped}/"
            f"{len(sel)} carved views. Outside the image reads as outside the "
            "subject, so the part that left the frame is carved away — "
            "re-shoot with the subject fully inside every frame.",
            flush=True,
        )

    vol = occ.reshape(grid, grid, grid)
    interior = np.zeros_like(vol)
    interior[1:-1, 1:-1, 1:-1] = (
        vol[1:-1, 1:-1, 1:-1] & vol[:-2, 1:-1, 1:-1] & vol[2:, 1:-1, 1:-1]
        & vol[1:-1, :-2, 1:-1] & vol[1:-1, 2:, 1:-1]
        & vol[1:-1, 1:-1, :-2] & vol[1:-1, 1:-1, 2:]
    )
    surf = np.flatnonzero(vol & ~interior)
    if not len(surf):
        raise SystemExit("--carve-seed: the carved hull has no surface")
    xyz = lo + np.stack(np.unravel_index(surf, (grid, grid, grid)), axis=1) * step
    log(f"carved hull: {vol.sum():,} voxels, {len(xyz):,} on the surface "
        f"({100 * vol.sum() / vol.size:.1f}% of the box)")

    # Colour: a surface point is painted only by the views where it is the
    # NEAREST hull point along its pixel's ray, so the far wall of the subject
    # is not painted with the near wall's colour.
    acc = np.zeros((len(xyz), 3), np.float64)
    cnt = np.zeros(len(xyz), np.int64)
    for k, i in enumerate(sel):
        img = cv2.imread(str(images_dir / names[i]))
        if img is None:
            continue
        h, w = img.shape[:2]
        cam = xyz @ viewmats[i][:3, :3].T + viewmats[i][:3, 3]
        uv, front = _project(camera, cam)
        u = np.rint(uv[:, 0]).astype(np.int64)
        v = np.rint(uv[:, 1]).astype(np.int64)
        good = np.flatnonzero(front & (u >= 0) & (u < w) & (v >= 0) & (v < h))
        if not len(good):
            continue
        pix = v[good] * w + u[good]
        order = good[np.lexsort((cam[good, 2], pix))]
        first = np.ones(len(order), bool)
        first[1:] = np.diff(v[order] * w + u[order]) != 0
        vis = order[first]
        acc[vis] += img[v[vis], u[vis]][:, ::-1]  # BGR → RGB
        cnt[vis] += 1
    seen = cnt > 0
    rgb = np.full((len(xyz), 3), 128, np.uint8)
    rgb[seen] = np.clip(acc[seen] / cnt[seen, None], 0, 255).astype(np.uint8)
    if (~seen).any():
        log(f"carved hull: {int((~seen).sum()):,} points never faced a camera "
            "— left grey")
    return xyz, rgb


# ── Stage D: loop closure ────────────────────────────────────────────


def capture_coverage(viewmats):
    """How far around its subject the capture actually travelled.

    Returns ``(arc_deg, detail)``, or ``(None, detail)`` when the cameras never
    converge on a subject — a room walkthrough looks outward, and "you never
    went around it" is meaningless there.

    The arc is 360 minus the widest *gap* in the viewing directions, measured
    around the axis the capture turns about. The obvious metric — the widest
    angle between any two views — cannot work: it saturates at a half orbit,
    because once two views are opposite each other, going the rest of the way
    round adds nothing to it. A gap, by contrast, is exactly the part of the
    subject nobody photographed.

    This catches the failure that looks like success — every frame registers,
    yet the subject was only ever seen from one side. Either the camera swept a
    short arc, or (the trap) the SUBJECT was turned instead of the camera,
    which puts every side of it in the video while the reconstruction, solving
    the static scene correctly, gets one side plus contradictory texture for it.
    """
    import numpy as np

    if len(viewmats) < 3:
        return (None, 0.0), "too few poses to judge coverage"
    R = viewmats[:, :3, :3]
    C = -np.einsum("nji,nj->ni", R, viewmats[:, :3, 3])
    fwd = R[:, 2, :]  # optical axis in world coordinates

    # Least-squares point closest to every optical axis: the subject, if there
    # is one. (I - dd^T) projects onto the plane perpendicular to a ray.
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(C, fwd):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    # Near-parallel axes leave A rank-deficient along the shared direction, and
    # its "intersection" is then an artifact. Each (I - dd^T) has eigenvalues
    # (0, 1, 1), so the mean's smallest eigenvalue measures how much the axes
    # actually splay: ~0 for a dolly down a corridor, ~0.5 for a planar orbit.
    if np.linalg.eigvalsh(A / len(C))[0] < 0.02:
        return (None, 0.0), "camera axes are near-parallel — no common subject"
    target = np.linalg.lstsq(A, b, rcond=None)[0]

    v = target - C
    dist = np.linalg.norm(v, axis=1)
    if not np.all(dist > 1e-9):
        return (None, 0.0), "a camera sits on the subject"
    # The rays must pass near that point AND it must lie in front of them,
    # or the capture is a scene rather than an orbit.
    resid = np.linalg.norm(np.cross(v, fwd), axis=1)
    if ((v * fwd).sum(1) > 0).mean() < 0.8 or np.median(resid) > 0.5 * np.median(dist):
        return (None, 0.0), "cameras look outward, not at one subject — a scene capture"

    u = (C - target) / dist[:, None]
    # Turning accumulated along the capture, in frame order. A real orbit
    # advances steadily, so this tracks the arc; a reconstruction that has
    # FOLDED — placed genuinely opposite views on top of each other — still
    # racks up turning as its poses oscillate, while the net spread stays
    # small. The ratio of the two separates "never went round" from "went
    # round, solved wrong", which need opposite fixes.
    turning = float(np.degrees(np.arccos(np.clip((u[:-1] * u[1:]).sum(1), -1.0, 1.0))).sum())

    # Fallback measure: the widest angle between two viewpoints. Basis-free, but
    # it mixes elevation into the answer — a full orbit looking 45 deg down
    # separates its extremes by only 90 — so it is used only when the capture
    # has no orbit structure to measure properly.
    spread = float(np.degrees(np.arccos(np.clip(u @ u.T, -1.0, 1.0))).max())

    # An orbit puts its viewpoints on a CIRCLE, whose pole is the direction they
    # vary in least. Centre them first: an orbit at any elevation rides a small
    # circle whose plane misses the origin, and skipping that biases the pole.
    uc = u - u.mean(0)
    _, sv, vt = np.linalg.svd(uc, full_matrices=False)
    if sv[0] < 1e-9:
        return (0.0, turning), f"every view from one spot, {np.median(dist):.2f} units away"
    # The pole only means something if the viewpoints really do trace a curve.
    # Handheld wobble over a short arc gives a fat blob instead, and its least
    # -variance direction is noise — project onto that and a tight cluster
    # smears over the full circle, reporting coverage it never had.
    if sv[2] > 0.25 * sv[1]:
        return (spread, turning), (
            f"subject ~{np.median(dist):.2f} scene units away, {len(C)} views "
            f"clustered within {spread:.0f} deg of each other; {turning:.0f} deg "
            "of turning accumulated along the path"
        )
    axis = vt[2]
    e1 = u[0] - (u[0] @ axis) * axis
    n1 = np.linalg.norm(e1)
    if n1 < 1e-9:
        return spread, "degenerate viewpoint geometry — falling back to spread"
    e1 /= n1
    e2 = np.cross(axis, e1)
    az = np.sort(np.degrees(np.arctan2(u @ e2, u @ e1)) % 360.0)
    gap = float(np.max(np.diff(np.concatenate([az, [az[0] + 360.0]]))))
    arc = 360.0 - gap
    return (arc, turning), (
        f"subject ~{np.median(dist):.2f} scene units away, {len(C)} views "
        f"covering {arc:.0f} deg around it (widest unseen gap {gap:.0f} deg, "
        f"{turning:.0f} deg of turning along the path)"
    )


def fit_turntable(viewmats):
    """Fit the one-degree-of-freedom model a turntable capture obeys.

    On a turntable the camera never moves; the SUBJECT rotates. Solved in the
    subject's frame (which is what masking to the subject produces), that is
    the same thing as one fixed camera pose swept around a fixed axis:

        pose_i = Rot(axis, theta_i) . pose_0

    so the whole capture is an axis, one reference pose, and one ANGLE per
    frame. That is the payoff of turntable mode. A folded solve — the failure
    that makes a splat look right from one side and like mush from the others
    — places late frames back at the angles of early ones, which shows up here
    as a non-monotonic theta. The frame ORDER is evidence no amount of
    feature matching can supply, and it is the one thing rotational symmetry
    (a printed mug, a patterned bowl) cannot fake.

    Returns a dict with the fitted ``axis``/``point``, per-frame ``angles``
    (unwrapped, degrees), the ``span`` they cover, ``monotonic`` (the fraction
    of steps that keep the dominant direction), and ``rms_pos``/``rms_rot`` —
    how far the model is from the solved poses, position as a fraction of the
    turn radius and rotation in degrees.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    cam_to_obj = np.array([np.linalg.inv(vm) for vm in viewmats])
    # Motion of each frame relative to the first: for a true turntable every
    # one of these is a rotation about the SAME line.
    rel = cam_to_obj @ np.linalg.inv(cam_to_obj[0])
    rots = Rotation.from_matrix(rel[:, :3, :3])
    rotvec = rots.as_rotvec()
    ang = np.linalg.norm(rotvec, axis=1)

    # Common axis: average the per-frame axes, signed so they agree, and
    # weighted by angle — a near-zero rotation has a meaningless direction.
    ref = rotvec[np.argmax(ang)]
    ref = ref / np.linalg.norm(ref)
    signed = np.where((rotvec @ ref)[:, None] < 0, -rotvec, rotvec)
    axis = signed.sum(0)
    axis = axis / np.linalg.norm(axis)

    theta = np.unwrap(np.where(rotvec @ axis < 0, -ang, ang))

    # The axis LINE: the point it passes through is the least-squares fixed
    # point of every relative motion (R p + t = p).
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for m in rel[1:]:
        M = np.eye(3) - m[:3, :3]
        A += M.T @ M
        b += M.T @ m[:3, 3]
    point = np.linalg.lstsq(A, b, rcond=None)[0] if len(rel) > 1 else np.zeros(3)

    model = apply_turntable(viewmats, axis, point, theta)
    centres = np.array([-vm[:3, :3].T @ vm[:3, 3] for vm in viewmats])
    fit_c = np.array([-vm[:3, :3].T @ vm[:3, 3] for vm in model])
    radial = centres - point - np.outer((centres - point) @ axis, axis)
    radius = float(np.median(np.linalg.norm(radial, axis=1)))
    rms_pos = float(np.sqrt(((centres - fit_c) ** 2).sum(1).mean()))
    rot_err = Rotation.from_matrix(
        model[:, :3, :3] @ viewmats[:, :3, :3].transpose(0, 2, 1)
    ).magnitude()

    steps = np.diff(theta)
    forward = float((steps > 0).mean()) if len(steps) else 1.0
    # How much of the turning actually GOT anywhere. A fold walks the subject
    # forwards and back, so its net rotation is a small fraction of the total
    # it travelled; a clean turn spends every degree making progress. Measured
    # on the House capture: 0.06 folded against 1.00 for the same capture
    # solved with --matching ordered.
    #
    # This replaces `monotonic` as the fold TEST. That metric counts the
    # fraction of per-frame steps going the dominant way, which on a 219-frame
    # solve is dominated by jitter on small steps: it read 0.87 for BOTH those
    # solves, missing a 171 deg fold and libelling a clean one. It stays in the
    # log because per-frame jitter is worth seeing, but it cannot gate.
    total = float(np.abs(steps).sum())
    directness = abs(float(theta[-1] - theta[0])) / total if total > 1e-9 else 1.0
    return {
        "axis": axis,
        "point": point,
        "angles": np.degrees(theta),
        "span": float(np.degrees(theta.max() - theta.min())),
        "radius": radius,
        "monotonic": max(forward, 1.0 - forward),
        "directness": directness,
        "rms_pos": rms_pos / radius if radius > 0 else float("inf"),
        "rms_rot": float(np.degrees(rot_err).mean()),
    }


def apply_turntable(viewmats, axis, point, theta):
    """Rebuild the poses from the turntable model: frame i is frame 0 rotated
    ``theta[i]`` about the fitted axis. Every wobble the solve invented in the
    radius, the elevation or the tilt is dropped — on a real turntable those
    degrees of freedom do not exist."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    base = np.linalg.inv(viewmats[0])
    out = []
    for t in theta:
        R = Rotation.from_rotvec(axis * t).as_matrix()
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = point - R @ point
        out.append(np.linalg.inv(M @ base))
    return np.array(out)


def angular_sampling(angles, window: float = 30.0):
    """How EVENLY the capture sampled the turn, as frames per ``window`` deg.

    Coverage and sampling are different questions. A capture can go all the way
    round and still starve one side, because a hand turn is never steady. MVS
    does not need *a* view of a surface, it needs several consistent ones — a
    depth survives only if enough neighbouring views corroborate it — so a thin
    sector reconstructs badly even though nothing about it was occluded.

    Measured on the House capture, which turned 411 deg yet lost most of one
    roof slope: 71 frames in the densest 30 deg against 4 in the thinnest, and
    the starved sectors averaged 608-660 sparse points per image against
    1069-1287 in the well-sampled ones. The two roof slopes are identical
    plywood seen from a camera at a constant 51.8 deg elevation, so nothing
    about the SUBJECT could explain why one reconstructed and the other did
    not — only how long the capture lingered on each.

    The window is slid over the covered arc rather than binned, so the answer
    does not depend on where bin edges happen to fall.
    """
    import numpy as np

    a = np.sort(np.asarray(angles, float))
    if len(a) < 3 or (a[-1] - a[0]) < window:
        return {"min": len(a), "median": float(len(a)), "ratio": 1.0}
    centres = np.arange(a[0] + window / 2, a[-1] - window / 2 + 1e-9, window / 6)
    counts = np.array([int(np.sum(np.abs(a - c) <= window / 2)) for c in centres])
    med = float(np.median(counts))
    return {"min": int(counts.min()), "median": med,
            "ratio": med / max(int(counts.min()), 1)}


def report_turntable(viewmats, snap: bool):
    """Log the turntable fit, warn when the solve does not behave like one,
    and (with ``snap``) replace the poses with the fitted model."""
    import numpy as np

    fit = fit_turntable(viewmats)
    log(
        f"turntable fit: {fit['span']:.0f} deg of subject rotation, "
        f"radius {fit['radius']:.2f} scene units, "
        f"{100 * fit['monotonic']:.0f}% of steps turning the same way, "
        f"{100 * fit['directness']:.0f}% of the turning net; "
        f"model residual {100 * fit['rms_pos']:.1f}% of radius, "
        f"{fit['rms_rot']:.2f} deg"
    )
    if fit["rms_pos"] > MAX_TT_POS_RESIDUAL or fit["rms_rot"] > MAX_TT_ROT_RESIDUAL:
        # Checked FIRST because it subsumes the others: poses that are not a
        # single-axis rotation at all will also look non-monotonic, and
        # blaming a fold would send the user to fix a turntable they never
        # used. A real turntable fits tightly even with a noisy solve —
        # measured 0.5% of radius and 0.3 deg on a jittered synthetic one,
        # against 46% and 13.6 deg for a walk-around run with --turntable.
        print(
            f"WARNING: these poses do not fit a turntable at all — the "
            f"single-axis model misses them by {100 * fit['rms_pos']:.0f}% of "
            f"the turn radius and {fit['rms_rot']:.1f} deg. A turntable "
            "capture is one FIXED camera and a rotating subject, so the "
            "solved cameras must lie on one circle. If you carried the camera "
            "around the subject instead, drop --turntable: that is a "
            "walk-around, and it is judged by coverage instead. If it really "
            "was a turntable, the camera moved during the capture (use a "
            "tripod) or the solve failed outright.",
            flush=True,
        )
    elif fit["directness"] < 0.7:
        print(
            f"WARNING: only {100 * fit['directness']:.0f}% of the rotation "
            "gets anywhere — the subject turns forwards and back, so the "
            "solve has FOLDED, matching late frames onto early ones. "
            "A turntable turns one way, so this is a solve failure, not a "
            "capture failure. It happens when the subject is symmetric or "
            "repetitively patterned and there is nothing else to key on: put "
            "a sheet of newspaper or a patterned cloth ON the turntable so it "
            "rotates WITH the subject and gives it asymmetric detail — and if "
            "you mask, keep that cloth INSIDE the mask (--mask-region around "
            "the platter), or the masking throws away the very texture you "
            "added. To rescue THIS capture without re-shooting, re-run with "
            "--matching ordered: it matches only frames near each other in "
            "time, which makes the fold structurally impossible.",
            flush=True,
        )
    elif fit["span"] < 270:
        print(
            f"WARNING: the subject only turned {fit['span']:.0f} deg, so the "
            "rest of it was never photographed — expect the unseen part to be "
            "missing or soft.",
            flush=True,
        )
    # Reported separately from the fold/residual checks above, not as an
    # alternative to them: a capture can be a perfect single-axis turn and
    # still be sampled too unevenly to reconstruct.
    samp = angular_sampling(fit["angles"])
    log(f"angular sampling: {samp['median']:.0f} frames per 30 deg of turn "
        f"(thinnest stretch {samp['min']})")
    if samp["min"] < MIN_VIEWS_PER_30DEG:
        # Two different faults share the symptom. A capture whose TYPICAL
        # density is also thin was simply too short; one whose typical density
        # is fine was turned unevenly, and only part of the subject suffers.
        even = samp["median"] < MIN_VIEWS_PER_30DEG
        print(
            (f"WARNING: the whole turn is thinly sampled — about "
             f"{samp['median']:.0f} frames per 30 deg, thinnest stretch "
             f"{samp['min']}. "
             if even else
             f"WARNING: the turn was sampled UNEVENLY — the thinnest 30 deg "
             f"of it has only {samp['min']} frames against a typical "
             f"{samp['median']:.0f}. ")
            + "Dense stereo keeps a depth only where enough neighbouring "
            "views agree, so "
            + ("the subject will come out holed all over"
               if even else
               "that stretch of the subject will come out holed while the "
               "rest is clean")
            + " even though it was photographed. "
            + ("Shoot for longer, or lower the frame spacing at ingest."
               if even else
               "Turn more slowly and STEADILY — a motorised turntable removes "
               "this failure entirely."),
            flush=True,
        )
    if snap:
        log("turntable: snapping poses onto the fitted single-axis model")
        return apply_turntable(
            viewmats, fit["axis"], fit["point"], np.radians(fit["angles"])
        ), fit
    return viewmats, fit


def report_coverage(viewmats):
    """Log the capture's coverage and warn when it cannot produce a whole
    object. Returns the arc in degrees, or None for a scene capture."""
    (arc, turning), detail = capture_coverage(viewmats)
    log(f"coverage: {detail}")
    if arc is None:
        return None
    if arc < 90 and turning > 2.5 * max(arc, 1.0):
        # The poses wander far more than they end up spread: the solve has
        # collapsed opposite viewpoints onto each other. Blaming the capture
        # here would send the user to re-shoot a video that was already fine.
        print(
            f"WARNING: the camera path accumulates {turning:.0f} deg of turning "
            f"but the solved viewpoints span only {arc:.0f} deg — the "
            "reconstruction has probably FOLDED the orbit, placing opposite "
            "views on top of each other. That is a solve failure, not a capture "
            "failure. It happens on repetitive, near-planar backgrounds (floor "
            "boards, tiles, plain desks), where features from one part of the "
            "scene match another. Re-run with --matching exhaustive, and give "
            "the scene some irregular non-planar detail around the subject.",
            flush=True,
        )
    elif arc < 90:
        print(
            f"WARNING: this capture only goes {arc:.0f} deg around its subject, "
            "so the far side was never photographed and the result will be "
            "one-sided. If the video does show every side, then the SUBJECT was "
            "turned rather than the camera moved around it — SfM solves the "
            "static scene, so those views become contradictory evidence for one "
            "side instead of coverage of the others. Put the subject down and "
            "walk a full circle around it.",
            flush=True,
        )
    elif arc < 270:
        print(
            f"WARNING: this capture goes {arc:.0f} deg around its subject, not a "
            "full circle — expect the unseen part to be missing or soft.",
            flush=True,
        )
    return arc


def build_loop_pairs(names, viewmats, pairs_path: Path, max_dist=None):
    """Spatially-close, temporally-distant frame pairs from aligned poses.

    The sequential match graph ties a scan's two visits to the same object
    together only through the long chain of relative poses, so the passes
    drift a few cm apart — invisible on far walls, ~13 px of reprojection at
    2 m. These pairs let the mapper close that loop.
    """
    import numpy as np

    centres = np.array([-vm[:3, :3].T @ vm[:3, 3] for vm in viewmats])
    axes = np.array([vm[2, :3] for vm in viewmats])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    idx = np.array([int(n.split(".")[0]) for n in names])

    # "Within 2 metres" only means something in a metric frame. LiDAR-aligned
    # poses have one; a bare SfM reconstruction is unitless, where a fixed 2.0
    # would either match everything or nothing. There, scale the radius by the
    # median distance between consecutive cameras — the capture's own natural
    # unit of movement. The multiplier is calibrated so that on a handheld
    # walkaround (a few cm per frame) the radius lands near the metric 2 m.
    if max_dist is None:
        step = np.median(np.linalg.norm(np.diff(centres, axis=0), axis=1))
        max_dist = 50.0 * step if step > 0 else np.inf
        log(f"loop pairs: unitless frame — radius {max_dist:.3f} "
            f"(50 x median inter-frame step {step:.4f})")

    pairs = set()
    for i in range(len(names)):
        d = np.linalg.norm(centres - centres[i], axis=1)
        ang = np.degrees(np.arccos(np.clip(axes @ axes[i], -1, 1)))
        cand = np.where(
            (np.abs(idx - idx[i]) > MIN_FRAME_GAP)
            & (d < max_dist)
            & (ang < MAX_ANGLE_DEG)
        )[0]
        cand = cand[np.argsort(d[cand])][:NEIGHBOURS]
        for j in cand:
            pairs.add(tuple(sorted((names[i], names[j]))))
    with open(pairs_path, "w") as f:
        for a, b in sorted(pairs):
            f.write(f"{a} {b}\n")
    log(f"loop pairs: {len(pairs)}")
    return len(pairs)


def match_loop_pairs(db: Path, pairs_path: Path):
    import pycolmap

    pairing = pycolmap.ImportedPairingOptions()
    pairing.match_list_path = str(pairs_path)
    pycolmap.match_image_pairs(database_path=db, pairing_options=pairing)
    log("loop-pair matching done")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scan", required=True,
                   help="LiDAR scan folder (bags + calibration), or an ingested "
                        "image set (images/ + manifest.csv)")
    p.add_argument(
        "--traj",
        default=None,
        help="LiDAR odometry trajectory npz (ts + poses — a cloud's "
        ".traj.npz sidecar) to align the SfM poses to. Required for a LiDAR "
        "scan; omitted for an image set, whose poses stay in the SfM's own "
        "(unitless, unlevelled) frame.",
    )
    p.add_argument("--output", required=True, help="output sfm_viewmats npz path")
    p.add_argument(
        "--cloud-output",
        default=None,
        help="write the sparse reconstruction as a coloured PLY here — the "
        "splat seed cloud for an image set, which has no LiDAR cloud",
    )
    p.add_argument(
        "--matching",
        default="auto",
        choices=("auto", "ordered", "sequential", "exhaustive", "vocabtree"),
        help="feature-matching strategy (auto: sequential for video/ordered "
        "captures, exhaustive for small unordered sets). 'ordered' matches "
        "only frames within --match-window of each other, which makes a FOLD "
        "structurally impossible on a symmetric subject — use it when the "
        "turntable fit reports the subject rotating back and forth.",
    )
    p.add_argument(
        "--vocab-tree",
        default=None,
        help="path to a downloaded COLMAP vocabulary tree (--matching vocabtree)",
    )
    p.add_argument(
        "--mapper",
        default="incremental",
        choices=("incremental", "glomap"),
        help="reconstruction engine (default incremental: on an ambiguous "
        "scene GLOMAP's global averaging commits to a wrong consensus, "
        "measured as 60 deg of solved spread vs incremental's 90 deg from "
        "the same matches). The other is the fallback if this one fails.",
    )
    p.add_argument(
        "--turntable",
        action="store_true",
        help="the SUBJECT rotated and the camera stayed put. Solves in the "
        "subject's frame and checks the result against the single-axis model "
        "a turntable obeys. Implies --mask-subject, but note that masking to "
        "the SUBJECT alone often hurts here: the platter co-rotates, so its "
        "texture belongs to the same rigid body. Measured on a plywood house, "
        "unmasked gave 219/219 frames against 162 and a collapsed 43 deg turn "
        "for the same capture masked to a box around the subject. Prefer "
        "--mask-region around the whole platter. Masks are kept beside the "
        "poses so training can exclude the background too.",
    )
    p.add_argument(
        "--turntable-snap",
        action="store_true",
        help="with --turntable, replace the solved poses with the fitted "
        "single-axis model — drops the radius/elevation/tilt wobble the solve "
        "invented, which a real turntable does not have",
    )
    p.add_argument(
        "--mask-subject",
        action="store_true",
        help="segment the subject (Mask R-CNN) and match features only on it "
        "— for an object shot on a floor, desk or table, where the flat "
        "background is what folds the orbit. Needs torchvision.",
    )
    p.add_argument(
        "--mask-classes",
        default=None,
        help="comma-separated COCO class names for --mask-subject "
        "(default: auto-detect the subject by vote across the capture). A "
        "name the segmenter has no class for is rejected up front rather "
        "than matching nothing; use --mask-region for a subject COCO cannot "
        "name.",
    )
    p.add_argument(
        "--mask-region",
        default=None,
        help="mask to a FIXED image region instead of a segmented class: "
        "'x0,y0,x1,y1' for a box, or 'x0,y0,x1,y1,x2,y2,…' for a polygon, as "
        "FRACTIONS of the frame (0,0 top-left). For --turntable subjects COCO "
        "has no name for — a machine part, a casting — where --mask-classes "
        "has nothing to ask for. Needs no torchvision, and may include the "
        "platter so a patterned cloth laid on it survives into matching.",
    )
    p.add_argument(
        "--mask-dilate",
        type=int,
        default=15,
        help="grow each subject mask by this many pixels, keeping the "
        "silhouette keypoints that carry the outline (0 disables)",
    )
    p.add_argument(
        "--match-window", type=int, default=0,
        help="for --matching ordered: how many frames either side may match "
        "(0 = auto, about a twelfth of the frame count). The window counts "
        "FRAMES but what matters is the ANGLE it spans, so a fixed value "
        "breaks when the frame rate changes: the same capture at 442 frames "
        "registered 231 of them with a window of 20 and all 442 with 40.",
    )
    p.add_argument(
        "--carve-seed",
        action="store_true",
        help="seed the cloud from the subject SILHOUETTES as well as the "
        "matched points: a turntable capture is a tomographic acquisition, "
        "and a silhouette needs no texture, so the blank side of a subject "
        "that yields no features still constrains its shape. Needs "
        "--turntable with segmented masks; the hull cannot see concavities.",
    )
    p.add_argument(
        "--carve-grid", type=int, default=256,
        help="voxels per axis for --carve-seed (default 256)",
    )
    p.add_argument(
        "--carve-views", type=int, default=72,
        help="how many frames to carve with, spread over the capture "
        "(default 72 — one every 5 deg of a full turn)",
    )
    p.add_argument(
        "--workspace",
        default=None,
        help="scratch dir for images/databases (default: <output>_work); "
        "removed on success unless --keep-workspace",
    )
    p.add_argument("--camera", default="front")
    p.add_argument(
        "--cam-time-offset",
        type=float,
        default=-0.025,
        help="camera↔LiDAR clock skew in seconds (same as process_splat.py)",
    )
    p.add_argument(
        "--no-loop-closure",
        action="store_true",
        help="skip stage D (faster, but revisited objects may stay drifted)",
    )
    p.add_argument("--keep-workspace", action="store_true")
    p.add_argument(
        "--min-registered-frac",
        type=float,
        default=0.4,
        help="fail if SfM registers fewer than this fraction of frames",
    )
    args = p.parse_args()

    # Parsed up front: a malformed region should cost nothing, not surface an
    # hour into a run. Mutually exclusive with --mask-classes because they are
    # two answers to the same question, and silently letting one win is the
    # class of bug this option exists to remove.
    if args.mask_region and args.mask_classes:
        print(
            "ERROR: --mask-region and --mask-classes both say what the "
            "subject is — pass one. Use --mask-region when COCO has no name "
            "for the subject, --mask-classes when it does.",
            flush=True,
        )
        return 1
    mask_region = parse_mask_region(args.mask_region) if args.mask_region else None
    if args.carve_seed:
        # Refused rather than warned: each of these makes the carve return
        # something confidently wrong — the box, or a hull built on poses the
        # silhouettes were never checked against.
        if mask_region is not None:
            print(
                "ERROR: --carve-seed needs a segmented silhouette, and "
                "--mask-region is a stated box. Carving against a box returns "
                "the box. Use --mask-subject, or a segmenter that outlines "
                "this subject.",
                flush=True,
            )
            return 1
        if not args.turntable:
            print(
                "ERROR: --carve-seed needs --turntable. The hull is only as "
                "good as the poses, and the turntable fit is what checks "
                "them; on a walk-around there is no such check.",
                flush=True,
            )
            return 1
        if not args.cloud_output:
            print("ERROR: --carve-seed has nothing to write without "
                  "--cloud-output", flush=True)
            return 1

    import numpy as np

    scan = Path(args.scan)
    output = Path(args.output)
    ws = Path(args.workspace) if args.workspace else output.parent / (
        output.stem + "_work"
    )
    ws.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Which of the two sources are we reconstructing? An ingested image set
    # carries its own frames and manifest; a LiDAR scan carries bags and a
    # calibration. Everything that differs between them is decided here.
    image_set = frame_source.is_image_set(scan)

    if image_set:
        images_dir = scan / "images"
        ts_by_name = frame_source.image_set_keys(scan)
        # Frames the ingest filter removed are still listed in the manifest
        # (it is written before the filter runs on a resumed set), so trust
        # the directory for what SfM will actually see.
        ts_by_name = {
            n: k for n, k in ts_by_name.items() if (images_dir / n).exists()
        }
        n_frames = len(ts_by_name)
        if n_frames < MIN_FRAMES:
            print(
                f"ERROR: only {n_frames} frames in {images_dir} — need at least "
                f"{MIN_FRAMES}",
                flush=True,
            )
            return 1
        t0 = time.time()
        progress(5, f"SfM poses: {n_frames} ingested frames")
        log(f"image set: {n_frames} frames in {images_dir}")
        if n_frames < REC_FRAMES:
            log(
                f"WARNING: {n_frames} frames is below the {REC_FRAMES} where SfM "
                "becomes reliable — it may register only some of them"
            )
    else:
        bags = sorted(scan.glob("IMAGE_*.bag")) if scan.exists() else []
        image_bag = bags[0] if bags else None
        if image_bag is None:
            print(f"ERROR: no IMAGE_*.bag in {scan}", flush=True)
            return 1
        if not args.traj:
            print(
                "ERROR: --traj is required for a LiDAR scan (SfM poses are "
                "aligned to the odometry trajectory)",
                flush=True,
            )
            return 1
        if not Path(args.traj).exists():
            print(f"ERROR: trajectory not found: {args.traj}", flush=True)
            return 1

        t0 = time.time()
        progress(1, "SfM poses: extracting camera frames from the bag…")
        images_dir = ws / "images"
        ts_by_name = extract_frames(
            image_bag, args.camera, images_dir, ws / "timestamps.csv"
        )
        n_frames = len(ts_by_name)
        if n_frames < 20:
            print(f"ERROR: only {n_frames} frames in the image bag", flush=True)
            return 1

    # A database left by an interrupted run may be partial (matching half
    # done) — start clean; the extracted JPEGs above are reused as-is.
    db = ws / "database.db"
    if db.exists():
        db.unlink()
    model, params = camera_prior(scan, args.camera)
    # An uncalibrated phone lens must be solved for; a factory-calibrated
    # one must not be (see run_mapping).
    refine = params is None
    strategy = args.matching
    if strategy == "auto":
        strategy = choose_matching(n_frames, bool(args.turntable))
        if image_set:
            log(f"capture is {'ordered' if _capture_is_ordered(scan) else 'unordered'}; "
                f"matching {strategy} ({n_frames} frames)")
    mask_dir = None
    if mask_region is not None or args.mask_subject or args.turntable:
        # Masks outlive the workspace: turntable training needs the same
        # per-frame masks the poses were solved from, and --keep-workspace is
        # off by default. Sitting beside the npz, they are found the way the
        # poses are.
        mask_dir = output.parent / (output.stem + "_masks")
        if mask_region is not None:
            if not args.turntable:
                print(
                    "WARNING: --mask-region keeps the SAME image region in "
                    "every frame, which only follows the subject while the "
                    "camera is fixed (--turntable). On a walk-around the "
                    "subject travels through the frame and a fixed region "
                    "will mask it away — use --mask-subject there.",
                    flush=True,
                )
            progress(6, "SfM poses: masking the stated region…")
            build_region_masks(images_dir, mask_dir, mask_region)
        else:
            classes = (
                [c.strip() for c in args.mask_classes.split(",") if c.strip()]
                if args.mask_classes
                else None
            )
            progress(6, "SfM poses: segmenting the subject…")
            # Explicit --mask-subject must fail if it cannot mask; --turntable
            # alone only implies it, so it degrades to unmasked instead.
            if build_subject_masks(images_dir, mask_dir, classes,
                                   args.mask_dilate,
                                   required=bool(args.mask_subject)) is None:
                mask_dir = None
    if strategy == "ordered" and not args.no_loop_closure:
        # Stage D pairs frames that are spatially close in the SOLVED poses,
        # which is exactly the long-range edge 'ordered' removed on purpose —
        # and on a symmetric subject the folded pose is what makes two
        # different sides look spatially close. Re-adding it undoes the fix.
        log("ordered matching: loop closure disabled (it would re-add the "
            "long-range pairs the fold needs)")
        args.no_loop_closure = True
    extract_and_match(db, images_dir, model, params, strategy, args.vocab_tree,
                      mask_dir, args.match_window)
    if image_set and _capture_is_ordered(scan):
        warn_weak_links(db)

    progress(38, "SfM poses: mapping (pass 1)…")
    rec = run_mapping(db, images_dir, ws / "sparse", "pass 1", refine,
                      args.mapper)
    if (rec is not None and image_set and not args.turntable
            and rec.num_reg_images() < FRAGMENTED_FRAC * n_frames):
        rec = _try_other_mapper(rec, db, images_dir, ws, refine, args, n_frames)
    if rec is None:
        print("ERROR: SfM produced no reconstruction", flush=True)
        return 1
    min_reg = max(20, int(args.min_registered_frac * n_frames))
    if rec.num_reg_images() < min_reg:
        hint = (
            "Re-shoot with more overlap: move slowly, keep 70%+ overlap "
            "between consecutive frames, and avoid blank walls and glare."
            if image_set
            else "Use downscale 2 (odometry poses) for this scan."
        )
        print(
            f"ERROR: SfM registered only {rec.num_reg_images()}/{n_frames} "
            f"frames — not enough texture/overlap for reliable poses. {hint}",
            flush=True,
        )
        return 1

    if image_set:
        # No stage C: there is no LiDAR trajectory to align to, so the poses
        # stay in the SfM's own frame — unitless and unlevelled, which the
        # viewer shows as a sideways scene at arbitrary size. That is expected
        # here, and is what the scale/level phase addresses.
        progress(55, "SfM poses: building viewmats (no LiDAR alignment)…")
        sfm, cam_params, cam_model = poses_from_rec(rec)
        names, viewmats, stats = viewmats_from_sfm(sfm, ts_by_name)
    else:
        calib = ppc.load_calibration(scan / "calibration", args.camera)
        T = calib["T"].astype(np.float64)

        progress(55, "SfM poses: aligning to the LiDAR trajectory…")
        sfm, cam_params, cam_model = poses_from_rec(rec)
        names, viewmats, stats = align_to_lidar(
            sfm, ts_by_name, Path(args.traj), T, args.cam_time_offset
        )

    if not args.no_loop_closure:
        progress(60, "SfM poses: matching loop-closure pairs…")
        pairs_path = ws / "pairs_loop.txt"
        # LiDAR-aligned poses are metric, so the campaign's 2 m radius applies
        # literally; an image set's frame is unitless and derives its own.
        loop_radius = None if image_set else MAX_DIST_M
        if build_loop_pairs(names, viewmats, pairs_path, loop_radius) > 0:
            db_loop = ws / "database_loop.db"
            shutil.copy2(db, db_loop)
            match_loop_pairs(db_loop, pairs_path)
            progress(68, "SfM poses: mapping (pass 2, loop-closed)…")
            rec2 = run_mapping(
                db_loop, images_dir, ws / "sparse_loop", "pass 2", refine,
                args.mapper
            )
            # A weaker pass-2 model means the added matches hurt — keep pass 1.
            if rec2 is not None and rec2.num_reg_images() >= rec.num_reg_images():
                progress(88, "SfM poses: final alignment…")
                rec = rec2
                sfm, cam_params, cam_model = poses_from_rec(rec2)
                if image_set:
                    names, viewmats, stats = viewmats_from_sfm(sfm, ts_by_name)
                else:
                    names, viewmats, stats = align_to_lidar(
                        sfm, ts_by_name, Path(args.traj), T, args.cam_time_offset
                    )
            else:
                log("pass 2 weaker than pass 1 — keeping pass-1 poses")

    # Coverage is judged on the FINAL poses (after any loop-closure pass), and
    # only for image sets: a LiDAR scan is a walkthrough, not an orbit.
    # A turntable capture is measured by its own model instead: the
    # walk-around checks would read a static camera as a one-sided capture and
    # tell the user to re-shoot a video that was fine.
    turntable_fit = None
    if args.turntable:
        viewmats, turntable_fit = report_turntable(viewmats, args.turntable_snap)
        coverage = turntable_fit["span"]
    else:
        coverage = report_coverage(viewmats) if image_set else None

    if image_set:
        # The reconstruction outlives the workspace, for the same reason the
        # masks do: dense stereo (process_dense.py) needs the sparse model,
        # and the only other way back to it is re-running the whole solve.
        # Sitting beside the npz, it is found the way the poses are.
        sparse_out = output.parent / (output.stem + "_sparse")
        shutil.rmtree(sparse_out, ignore_errors=True)
        sparse_out.mkdir(parents=True, exist_ok=True)
        rec.write(str(sparse_out))
        log(f"sparse model kept for dense stereo → {sparse_out}")
        if args.turntable_snap:
            print(
                "WARNING: --turntable-snap rewrote the exported poses but not "
                "the sparse model, so a dense reconstruction from it will sit "
                "in the UNSNAPPED frame. Re-run without --turntable-snap "
                "before meshing.",
                flush=True,
            )

    # The seed cloud must come from the SAME reconstruction as the exported
    # poses, or the splat trains geometry against cameras that never saw it.
    if args.cloud_output and args.carve_seed:
        progress(92, "SfM poses: carving the silhouette hull…")
        import cv2

        sparse_xyz, sparse_rgb = _sparse_points(rec)
        h, w = cv2.imread(str(images_dir / names[0])).shape[:2]
        cam = pycolmap_camera(cam_model, cam_params, w, h)
        hull_xyz, hull_rgb = carve_visual_hull(
            mask_dir, images_dir, names, viewmats, cam, sparse_xyz,
            grid=args.carve_grid, views=args.carve_views,
            # Cancel the dilation the MATCHING masks carry: it is there to keep
            # the outline keypoints, and it inflates the hull on every ray.
            erode=args.mask_dilate,
        )
        # Both, not either: the matched points are metrically the better
        # geometry wherever texture exists, and the hull is the only geometry
        # at all where it does not.
        n = _write_ply(
            np.vstack([sparse_xyz, hull_xyz]),
            np.vstack([sparse_rgb, hull_rgb]),
            Path(args.cloud_output),
        )
        log(f"seed cloud: {n:,} points ({len(sparse_xyz):,} matched + "
            f"{len(hull_xyz):,} carved) → {args.cloud_output}")
    elif args.cloud_output:
        progress(92, "SfM poses: exporting the sparse seed cloud…")
        export_sparse_cloud(rec, Path(args.cloud_output))

    # Write-then-rename so a crash never leaves a half-written npz where the
    # server's existence check would find it. (np.savez appends ".npz" to bare
    # paths, so give it an open handle.)
    tmp = Path(str(output) + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(
            f,
            names=np.array(names),
            # Kept under the historic key: it is simply "the join key", and
            # renaming it would break every npz generated before today.
            bag_ts_ns=np.array([ts_by_name[n] for n in names], dtype=np.int64),
            viewmats=viewmats,
            # fisheye_params stays for backwards compatibility (older files
            # have only this key, and process_splat still reads it); the
            # model-tagged pair is what a non-fisheye camera needs.
            fisheye_params=cam_params,
            camera_model=np.array(cam_model),
            camera_params=cam_params,
            # nan = not an orbit (or too few poses) rather than "0 degrees".
            coverage_deg=np.array(float("nan") if coverage is None else coverage),
            image_set=np.array(bool(image_set)),
            camera=np.array(args.camera),
            # Where the per-frame subject masks live, so training can exclude
            # the same background the poses were solved without. Empty when
            # the run was unmasked.
            mask_dir=np.array(str(mask_dir) if mask_dir else ""),
            turntable=np.array(bool(args.turntable)),
        )
    tmp.rename(output)
    log(f"wrote {output} ({len(names)}/{n_frames} viewmats; {stats})")

    if not args.keep_workspace:
        # An image set's frames live in the project, not the workspace — only
        # the databases and sparse models are scratch.
        shutil.rmtree(ws, ignore_errors=True)
    progress(100, f"SfM poses ready ({len(names)} frames, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
