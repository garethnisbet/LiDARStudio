"""Normalize a phone capture (photo folder or video) into a posable image set.

The second scan source alongside the LiDAR bag input. Everything downstream of
this module — SfM, the seed cloud, splat training, the Library, the editing ops
— consumes the *normalized* form this writes, never the user's original folder,
which may live on a read-only mount (a phone over MTP, an SD card).

Output, always inside the project::

    <project>/scans/<name>/images/NNNN.jpg
    <project>/scans/<name>/manifest.csv
    <project>/scans/<name>/ingest.json

``manifest.csv`` is the contract between this stage and every later one. Its
``key_ns`` column is the join key that ties a frame to its pose in the SfM npz
(the role ``bag_ts_ns`` plays on the LiDAR path), and it is *persisted* rather
than recomputed: deriving it from enumeration order would mean a single frame
dropped between the SfM run and the training run silently shifts every
subsequent pose association. Read it with ``csv.DictReader`` and address
columns by name — later phases add per-frame pose columns (qw..tz, reserved
here) and must not force a migration.

Steps: decode (HEIC via pillow-heif, video via cv2) → apply EXIF orientation →
downscale to a max long edge → drop the blurriest frames → write.

Needs cv2 + PIL; pillow-heif only for HEIC input. Emits ``PROGRESS:<pct>:<msg>``
so lidar_jobs._stream_proc renders it in the existing panel.
"""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

STILL_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
HEIC_EXT = {".heic", ".heif"}

# Two thresholds, not one. MIN_FRAMES is geometry's hard floor — below it there
# is nothing to triangulate. REC_FRAMES is the count a capture *should* have:
# SfM needs wide baselines *and* dense overlap, and below ~20 images one missing
# match disconnects the graph. Between the two we ingest and warn, because a
# sparse set of a small object often still reconstructs, and refusing outright
# is worse than saying so.
MIN_FRAMES = 4
REC_FRAMES = 20

# Manifest columns. Address by NAME (csv.DictReader), never by position — the
# pose columns are reserved for ARKit/ARCore ingest and are empty for now.
MANIFEST_COLS = [
    "name", "key_ns", "source_file", "exif_focal_mm", "exif_ts", "sharpness",
    "qw", "qx", "qy", "qz", "tx", "ty", "tz",
]


def progress(pct: int, msg: str):
    print(f"PROGRESS:{pct}:{msg}", flush=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def warn(msg):
    print(f"WARNING: {msg}", flush=True)


# ── Source discovery ─────────────────────────────────────────────────


def find_source(src: Path):
    """Classify a user-supplied folder as ``("video", [file])`` or
    ``("stills", [files...])``.

    Mirrors the detection in ``lidar_jobs._scan_contents`` so what the panel
    reports and what this ingests can never disagree. Photos may sit at the top
    level or one directory down (``images/``, ``DCIM/``, or a lone subfolder) —
    the shape you get from a camera roll export.
    """
    if src.is_file():
        ext = src.suffix.lower()
        if ext in VIDEO_EXT:
            return "video", [src]
        raise SystemExit(f"ERROR: {src.name} is not a video file")

    def stills_in(d: Path):
        return sorted(
            p for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in STILL_EXT
        )

    videos = sorted(
        p for p in src.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXT
    )
    stills = stills_in(src)
    if len(stills) < REC_FRAMES:
        # One level down: images/, DCIM/, or the only subdirectory present.
        # 'thumbnail' is excluded — it is a LiDAR-scan artifact, not a capture.
        subs = [
            d for d in sorted(src.iterdir())
            if d.is_dir() and d.name != "thumbnail"
        ]
        named = [d for d in subs if d.name.lower() in ("images", "dcim", "photos")]
        for d in named or (subs if len(subs) == 1 else []):
            found = stills_in(d)
            if len(found) > len(stills) and len(found) >= MIN_FRAMES:
                stills = found
                break
    if len(stills) >= MIN_FRAMES:
        return "stills", stills
    if videos:
        return "video", videos[:1]
    if stills:
        raise SystemExit(
            f"ERROR: only {len(stills)} image(s) in {src} — need at least "
            f"{MIN_FRAMES} for a reconstruction"
        )
    raise SystemExit(f"ERROR: no photos or video found in {src}")


# ── EXIF ─────────────────────────────────────────────────────────────


def _exif(path: Path):
    """(focal_mm, timestamp_str) from a still's EXIF, or (None, None)."""
    try:
        from PIL import ExifTags, Image
    except ImportError:
        return None, None
    tags = {v: k for k, v in ExifTags.TAGS.items()}
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            if not ex:
                return None, None
            f = ex.get(tags.get("FocalLength"))
            ts = ex.get(tags.get("DateTimeOriginal")) or ex.get(tags.get("DateTime"))
            # EXIF rationals arrive as Fraction-likes on some Pillow versions.
            focal = float(f) if f is not None else None
            return focal, (str(ts) if ts else None)
    except Exception:
        return None, None


def _load_still(path: Path):
    """Decode a still to a BGR ndarray with EXIF orientation APPLIED.

    cv2.imread silently ignores the orientation tag, which would hand SfM a set
    of images whose rotation disagrees with their content — the single most
    common way a phone capture fails to reconstruct. Decoding through Pillow's
    ``exif_transpose`` is the only reliable fix, and it is also what gives HEIC
    support for free.
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    if path.suffix.lower() in HEIC_EXT:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError as exc:
            raise SystemExit(
                "ERROR: HEIC input needs pillow-heif "
                "(pip install 'lidarstudio[photogrammetry]', or convert the "
                "photos to JPEG on the phone: Settings → Camera → Formats → "
                "Most Compatible)"
            ) from exc
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        rgb = np.asarray(im.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ── Frame production ─────────────────────────────────────────────────


# ── Smart frame selection ────────────────────────────────────────────
#
# A fixed frame rate is the wrong control for a hand-held capture. The camera
# stands still at the start, creeps along a wall, then swings through a
# doorway. A rate that is dense enough for the swing buries SfM in duplicates
# of the still parts, and a rate that is sparse enough for those leaves the
# swing with no overlap. What SfM needs is a constant amount of VIEW CHANGE
# between neighbouring frames. So the selector tracks features from the last
# kept frame and keeps a new one once the view has moved far enough.

SELECT_EDGE = 480        # px, long edge of the grey copy the selector tracks on
CANDIDATE_FPS = 15.0     # selection looks at no more frames than this per second
MIN_TRACKS = 25          # fewer corners than this: tracking can't be trusted
LOST_OVERLAP = 0.2       # below this survival the view has jumped (a cut, a whip-pan)
# Parallax trigger, as a fraction of the spacing. Measured on the real captures:
# a pan across the living room reached 5% shift with ~0.003 parallax, while
# the House orbit reached 0.005 parallax in ~10 frames at only 1.5% shift.
# The 4 fps sampling that solved House well was ~7.5 frames.
PARALLAX_RATIO = 0.1
SHARP_TOL = 0.9          # "as sharp as the best" for picking within a step


class _Tracker:
    """Corners from one keyframe, chained frame to frame by Lucas-Kanade."""

    def __init__(self, key_gray):
        import cv2
        import numpy as np

        self.prev = key_gray
        pts = cv2.goodFeaturesToTrack(
            key_gray, maxCorners=400, qualityLevel=0.01, minDistance=8,
        )
        self.key = (pts.reshape(-1, 2) if pts is not None
                    else np.zeros((0, 2), np.float32))
        self.cur = self.key.copy()
        self.n0 = len(self.key)

    @property
    def reliable(self):
        return self.n0 >= MIN_TRACKS

    def step(self, gray):
        """Track into ``gray``; return ``(motion, overlap, parallax)`` against
        the keyframe, each as a fraction of the long edge where it is a length.

        ``motion`` is the median corner displacement, which is how far the view
        has slid. ``overlap`` is the fraction of keyframe corners still held.
        ``parallax`` is the 75th-percentile residual after the best homography,
        i.e. motion no camera rotation could produce. That is the baseline
        triangulation depends on. Orbiting a centred object makes a lot of it
        while hardly shifting the image, which ``motion`` alone would miss.
        """
        import cv2
        import numpy as np

        if len(self.cur) == 0:
            self.prev = gray
            return 0.0, 0.0, 0.0
        p0 = self.cur.reshape(-1, 1, 2).astype(np.float32)
        lk = dict(winSize=(21, 21), maxLevel=3)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None, **lk)
        # Forward-backward check: a corner that doesn't track back to where it
        # started has latched onto something else, and would fake motion.
        pb, stb, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None, **lk)
        fb = np.linalg.norm((pb - p0).reshape(-1, 2), axis=1)
        h, w = gray.shape[:2]
        q = p1.reshape(-1, 2)
        ok = ((st.ravel() == 1) & (stb.ravel() == 1) & (fb < 1.0)
              & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h))
        self.key, self.cur, self.prev = self.key[ok], q[ok], gray
        if not len(self.cur):
            return 0.0, 0.0, 0.0
        edge = float(max(h, w))
        d = np.linalg.norm(self.cur - self.key, axis=1)
        par = 0.0
        if len(self.cur) >= 8:
            H, _ = cv2.findHomography(self.key, self.cur, cv2.RANSAC, 2.0)
            if H is not None:
                fit = cv2.perspectiveTransform(self.key.reshape(-1, 1, 2), H)
                r = np.linalg.norm(fit.reshape(-1, 2) - self.cur, axis=1)
                par = float(np.percentile(r, 75)) / edge
        return float(np.median(d)) / edge, len(self.cur) / max(1, self.n0), par


def select_keyframes(frames, spacing=0.05, min_overlap=0.6, max_gap=None):
    """Choose which frames of a sequence to keep. Returns sorted indices.

    ``frames`` is an iterable of ``(gray, sharpness)`` in capture order, with
    ``gray`` a small greyscale copy. A new frame is kept once the view has
    shifted by ``spacing`` (median feature motion, as a fraction of the long
    edge) since the last kept frame, once the parallax reaches a tenth of
    that (an orbit, see ``_Tracker.step``), or once fewer than ``min_overlap``
    of that frame's features remain in view, whichever comes first. A still
    camera therefore adds nothing, and a fast swing gets frames packed as
    tightly as it needs.

    Within each step the SHARPEST frame is kept, not the one that crossed the
    line. Any frame that has moved at least half the spacing is a candidate, so
    blur is avoided at selection time rather than cut out afterwards, which
    would leave a gap.

    When tracking is lost in one step (a cut, a whip-pan), the frame just
    before the jump is kept as well as the one after it, so whatever overlap
    there is survives for the matcher. On a featureless view (a blank wall)
    the motion can't be measured, so a frame is kept every ``max_gap`` frames
    instead of none at all.
    """
    from collections import deque

    src = iter(frames)
    pending = deque()          # frames to (re)process, as (index, gray, sharp)
    kept = []
    trk = None
    window = []                # (index, gray, sharp, motion) since the keyframe
    last = None                # final (index, motion) seen, for the tail rule
    n = 0

    def pull():
        nonlocal n
        if pending:
            return pending.popleft()
        try:
            gray, sharp = next(src)
        except StopIteration:
            return None
        n += 1
        return n - 1, gray, sharp

    def rekey(idx, gray, rest):
        """Make ``idx`` the keyframe and replay the window after it."""
        nonlocal trk, window
        kept.append(idx)
        trk = _Tracker(gray)
        replay = [(i, g, s) for i, g, s, _m in rest]
        window = []
        pending.extendleft(reversed(replay))

    while True:
        f = pull()
        if f is None:
            break
        i, gray, sharp = f
        if trk is None:
            rekey(i, gray, [])
            continue
        motion, overlap, parallax = trk.step(gray)
        # One number for "how far along the step are we": whichever of slide
        # and parallax is further toward its threshold.
        motion = max(motion, parallax / PARALLAX_RATIO)
        last = (i, motion)
        lost = trk.reliable and overlap < LOST_OVERLAP
        if trk.reliable:
            trigger = lost or motion >= spacing or overlap < min_overlap
        else:
            trigger = max_gap is not None and len(window) + 1 >= max_gap
        if not trigger:
            window.append((i, gray, sharp, motion))
            continue

        if lost:
            # The jump itself can't be bridged by this frame; the one before it
            # holds the most overlap with what comes next. With nothing in the
            # window, the jump happened in a single step: keep this frame.
            if window:
                j = len(window) - 1
                rekey(window[j][0], window[j][1],
                      window[j + 1:] + [(i, gray, sharp, motion)])
            else:
                rekey(i, gray, [])
            continue

        # The LATEST frame that is nearly as sharp as the sharpest. Taking the
        # sharpest outright would land anywhere in the window and shrink the
        # spacing for no gain when nothing is actually blurred.
        cands = [k for k, w in enumerate(window) if w[3] >= 0.5 * spacing]
        top = max([sharp] + [window[k][2] for k in cands])
        if sharp >= SHARP_TOL * top:
            best_k = None
        else:
            best_k = max(k for k in cands if window[k][2] >= SHARP_TOL * top)
        if best_k is None:
            rekey(i, gray, [])
        else:
            rekey(window[best_k][0], window[best_k][1],
                  window[best_k + 1:] + [(i, gray, sharp, motion)])
        last = None

    # The tail: the final frame finishes the capture's coverage. Keep it if
    # the view moved meaningfully since the last kept frame.
    if last is not None and last[0] != kept[-1] and last[1] >= 0.25 * spacing:
        kept.append(last[0])
    return sorted(set(kept))


def _video_candidates(video: Path):
    """Open ``video`` and return ``(cap, native_fps, stride)``: selection looks
    at every ``stride``-th frame, at most CANDIDATE_FPS of them a second."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open video {video}")
    native = cap.get(cv2.CAP_PROP_FPS) or 0.0
    stride = max(1, int(round(native / CANDIDATE_FPS))) if native > 0 else 1
    return cap, native, stride


def plan_video(video: Path, spacing: float, min_overlap: float):
    """Pass 1 of a smart ingest: decide which frame numbers to keep.

    Runs on a small grey copy only, so it holds nothing at full resolution and
    costs a decode of the clip plus a little optical flow.
    """
    import cv2

    cap, native, stride = _video_candidates(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    index = []    # candidate number → frame number in the video

    def grays():
        i = 0
        try:
            while cap.grab():
                if i % stride == 0:
                    ok, img = cap.retrieve()
                    if ok and img is not None:
                        h, w = img.shape[:2]
                        s = SELECT_EDGE / float(max(h, w))
                        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        if s < 1:
                            g = cv2.resize(g, (int(round(w * s)), int(round(h * s))),
                                           interpolation=cv2.INTER_AREA)
                        index.append(i)
                        if len(index) % 50 == 0 and total:
                            progress(2 + int(38 * i / total),
                                     f"Ingest: choosing frames ({i}/{total})…")
                        yield g, float(cv2.Laplacian(g, cv2.CV_64F).var())
                i += 1
        finally:
            cap.release()

    # Featureless stretch: fall back to ~2 frames a second rather than none.
    max_gap = max(1, int(round(CANDIDATE_FPS / 2)))
    picks = select_keyframes(grays(), spacing, min_overlap, max_gap=max_gap)
    frames = [index[k] for k in picks]
    log(f"video {video.name}: {native:.1f} fps native, {len(index)} candidates "
        f"(every {stride}), kept {len(frames)} at {spacing:.0%} spacing "
        f"(≈{len(frames) / max(1e-9, (total or 1) / max(native, 1e-9)):.1f}/s on average)")
    return frames, len(index)


def _iter_video(video: Path, fps: float = 0.0, frames=None):
    """Yield ``(key_ns, bgr, source_label)`` from a video: the given frame
    numbers when ``frames`` is set, else sampled at a fixed ~fps."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open video {video}")
    native = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if frames is not None:
        wanted = set(frames)
        last = max(frames) if frames else -1
    else:
        step = max(1, int(round(native / fps))) if native > 0 and fps > 0 else 1
        log(f"video {video.name}: {native:.1f} fps native, "
            f"taking every {step} frame(s)")
    i = 0
    try:
        while True:
            if frames is not None and i > last:
                break
            ok = cap.grab()
            if not ok:
                break
            if (i in wanted) if frames is not None else (i % step == 0):
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                ok, img = cap.retrieve()
                if ok and img is not None:
                    # PTS is the natural key; some containers report 0 for
                    # every frame, so fall back to the frame index.
                    key = (
                        int(round(pos_ms * 1e6))
                        if pos_ms and pos_ms > 0
                        else i * 1_000_000
                    )
                    yield key, img, f"{video.name}#{i}"
            i += 1
    finally:
        cap.release()


def _iter_stills(files):
    """Yield ``(key_ns, bgr, source_label)`` for stills in capture order.

    Ordered by EXIF capture time when every file has one (a camera roll copied
    out of order is common), else by filename. Sequential matching depends on
    this order being the order the photos were actually taken.
    """
    stamps = [(_exif(p), p) for p in files]
    if all(e[1] for e, _ in stamps):
        stamps.sort(key=lambda s: (s[0][1], s[1].name))
    else:
        stamps.sort(key=lambda s: s[1].name)
    for i, ((focal, ts), p) in enumerate(stamps):
        yield i * 1_000_000, _load_still(p), p.name, focal, ts


def blurry_drops(scores, frac: float) -> set:
    """Indices of the frames the sharpness filter removes: the bottom ``frac``
    by sharpness, except that no two NEIGHBOURING frames are ever both dropped.

    Blur is not spread evenly through a capture. It bunches up where the
    camera moves fastest, and that is usually a doorway or a turn, which is
    exactly where consecutive frames are the only thing linking one part of
    the scene to the next. A plain quantile cut removes the whole stretch. On
    the Clinic walk-through it deleted 8 frames in a row (2 s of video) at the
    doorway into the room, the room frames then had zero matches to the
    hallway, and SfM kept the hallway alone: 203/262 frames, room missing.
    Keeping every other frame of a blurry run keeps the chain intact, while
    still thinning the blur out of training.
    """
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    if frac <= 0 or len(scores) == 0:
        return set()
    cut = float(np.quantile(scores, frac))
    dropped = set()
    # Blurriest first, so a run keeps its sharper half.
    for i in np.argsort(scores, kind="stable"):
        i = int(i)
        if scores[i] >= cut:
            break
        if (i - 1) not in dropped and (i + 1) not in dropped:
            dropped.add(i)
    return dropped


def ingest(src: Path, out_dir: Path, max_edge: int, fps: float, drop_blurry: float,
           select: str = "motion", spacing: float = 0.05, min_overlap: float = 0.6):
    import cv2
    import numpy as np

    kind, files = find_source(src)
    images_dir = out_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)  # a partial previous run would corrupt numbering
    images_dir.mkdir(parents=True, exist_ok=True)

    selection = None
    if kind == "video":
        log(f"source: video {files[0].name}")
        if select == "motion":
            progress(2, "Ingest: choosing frames by how much the view changes…")
            chosen, n_cand = plan_video(files[0], spacing, min_overlap)
            selection = {"mode": "motion", "spacing": spacing,
                         "min_overlap": min_overlap, "candidates": n_cand,
                         "kept": len(chosen)}
            frames_iter = _iter_video(files[0], frames=chosen)
            total = len(chosen)
        else:
            selection = {"mode": "fixed", "fps": fps}
            frames_iter = _iter_video(files[0], fps)
            total = None
        # Video frames carry no per-frame EXIF — pad to the stills tuple shape.
        producer = ((k, img, lbl, None, None) for k, img, lbl in frames_iter)
    else:
        log(f"source: {len(files)} still image(s)")
        producer = _iter_stills(files)
        total = len(files)

    rows, focals, seen_keys = [], [], set()
    for i, (key_ns, img, label, focal, ts) in enumerate(producer):
        if key_ns in seen_keys:  # never let two frames share a join key
            key_ns = (max(seen_keys) + 1) if seen_keys else i
        seen_keys.add(key_ns)

        h, w = img.shape[:2]
        # 48 MP phone stills add nothing to SfM accuracy and dominate both
        # match time and memory; the long edge is what sets feature scale.
        if max(h, w) > max_edge:
            s = max_edge / float(max(h, w))
            img = cv2.resize(
                img,
                (int(round(w * s)), int(round(h * s))),
                interpolation=cv2.INTER_AREA,
            )
        sharp = float(
            cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        )
        name = f"{i:04d}.jpg"
        cv2.imwrite(str(images_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if focal:
            focals.append(round(float(focal), 2))
        rows.append({
            "name": name, "key_ns": int(key_ns), "source_file": label,
            "exif_focal_mm": "" if not focal else f"{float(focal):.2f}",
            "exif_ts": ts or "", "sharpness": f"{sharp:.1f}",
        })
        if (i + 1) % 25 == 0:
            pct = (40 + int(35 * (i + 1) / total) if selection and selection["mode"] == "motion"
                   else 5 + int(70 * (i + 1) / total) if total
                   else min(75, 5 + (i + 1) // 4))
            progress(pct, f"Ingest: decoded {i + 1} frame(s)…")

    if len(rows) < MIN_FRAMES:
        raise SystemExit(
            f"ERROR: only {len(rows)} frame(s) after decoding — need at least "
            f"{MIN_FRAMES}. For video, lower --spacing (or raise --fps); "
            "for stills, capture more."
        )

    # Sharpness filter. Blurry frames do not merely add nothing: a motion-blurred
    # photo teaches soft edges everywhere it is sampled. Dropping them leaves
    # NUMBERING GAPS on purpose — the NNNN index stays a faithful temporal
    # index, which is what build_loop_pairs reads it as.
    warnings = []
    if drop_blurry > 0 and len(rows) > REC_FRAMES:
        scores = np.array([float(r["sharpness"]) for r in rows])
        cut = float(np.quantile(scores, drop_blurry))
        dropped = blurry_drops(scores, drop_blurry)
        keep = [r for i, r in enumerate(rows) if i not in dropped]
        if len(keep) >= REC_FRAMES:
            kept_names = {r["name"] for r in keep}
            for r in rows:
                if r["name"] not in kept_names:
                    (images_dir / r["name"]).unlink(missing_ok=True)
            log(
                f"sharpness filter: dropped {len(rows) - len(keep)} of {len(rows)} "
                f"(laplacian-var cut {cut:.1f}, median {np.median(scores):.1f})"
            )
            rows = keep

    progress(85, "Ingest: writing manifest…")
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in MANIFEST_COLS})

    # Surface the two conditions that quietly wreck a reconstruction, rather
    # than letting them fail an hour into SIFT.
    uniq = sorted(set(focals))
    if len(uniq) > 1:
        warnings.append(
            f"mixed EXIF focal lengths {uniq} — the capture switched lenses "
            "(ultrawide/tele). SfM models one shared camera, so the poses will "
            "be degraded. Re-shoot locked to a single lens for best results."
        )
    if len(rows) < REC_FRAMES:
        warnings.append(
            f"only {len(rows)} frames — below the {REC_FRAMES} where SfM becomes "
            "reliable. It may register only some of them, or fail outright; a "
            "slow orbit giving 150-400 frames reconstructs far better."
        )
    elif len(rows) < 60:
        warnings.append(
            f"only {len(rows)} frames — thin coverage. Expect holes; a slow "
            "video orbit giving 150-400 frames reconstructs far better."
        )
    for m in warnings:
        warn(m)

    meta = {
        "source": str(src.resolve()),
        "source_kind": kind,
        "frames": len(rows),
        "max_edge": max_edge,
        "fps": fps if kind == "video" and select != "motion" else None,
        "selection": selection,
        "exif_focal_mm": uniq,
        "warnings": warnings,
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "ingest.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", required=True, help="photo folder or video file")
    p.add_argument("--output", required=True,
                   help="ingest dir (<project>/scans/<name>)")
    p.add_argument("--max-edge", type=int, default=2400,
                   help="downscale so the long edge is at most this many px")
    p.add_argument("--select", choices=("motion", "fixed"), default="motion",
                   help="video frame choice: 'motion' keeps a frame each time the "
                        "view has changed by --spacing; 'fixed' samples at --fps")
    p.add_argument("--spacing", type=float, default=0.05,
                   help="motion selection: view shift between kept frames, as a "
                        "fraction of the long edge (smaller = more frames)")
    p.add_argument("--min-overlap", type=float, default=0.6,
                   help="motion selection: also keep a frame once fewer than this "
                        "fraction of the last kept frame's features are still in view")
    p.add_argument("--fps", type=float, default=4.0,
                   help="fixed selection: frames per second to sample from a video")
    p.add_argument("--drop-blurry", type=float, default=0.10,
                   help="drop this bottom fraction by Laplacian-variance sharpness")
    args = p.parse_args()

    src, out = Path(args.source), Path(args.output)
    if not src.exists():
        print(f"ERROR: source not found: {src}", flush=True)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    progress(1, "Ingest: scanning the capture…")
    meta = ingest(src, out, args.max_edge, args.fps, args.drop_blurry,
                  args.select, args.spacing, args.min_overlap)
    progress(
        100,
        f"Ingest ready — {meta['frames']} frames in {time.time() - t0:.0f}s",
    )
    log(f"wrote {out}/images ({meta['frames']} frames) + manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
