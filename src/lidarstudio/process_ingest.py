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


def _iter_video(video: Path, fps: float):
    """Yield ``(key_ns, bgr, source_label)`` sampled from a video at ~fps."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open video {video}")
    native = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = max(1, int(round(native / fps))) if native > 0 and fps > 0 else 1
    log(f"video {video.name}: {native:.1f} fps native, "
        f"taking every {step} frame(s)")
    i = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
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


def ingest(src: Path, out_dir: Path, max_edge: int, fps: float, drop_blurry: float):
    import cv2
    import numpy as np

    kind, files = find_source(src)
    images_dir = out_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)  # a partial previous run would corrupt numbering
    images_dir.mkdir(parents=True, exist_ok=True)

    if kind == "video":
        log(f"source: video {files[0].name}")
        # Video frames carry no per-frame EXIF — pad to the stills tuple shape.
        producer = (
            (k, img, lbl, None, None)
            for k, img, lbl in _iter_video(files[0], fps)
        )
        total = None
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
            pct = 5 + int(70 * (i + 1) / total) if total else min(75, 5 + (i + 1) // 4)
            progress(pct, f"Ingest: decoded {i + 1} frame(s)…")

    if len(rows) < MIN_FRAMES:
        raise SystemExit(
            f"ERROR: only {len(rows)} frame(s) after decoding — need at least "
            f"{MIN_FRAMES}. For video, raise --fps; for stills, capture more."
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
        "fps": fps if kind == "video" else None,
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
    p.add_argument("--fps", type=float, default=4.0,
                   help="frames per second to sample from a video")
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
    meta = ingest(src, out, args.max_edge, args.fps, args.drop_blurry)
    progress(
        100,
        f"Ingest ready — {meta['frames']} frames in {time.time() - t0:.0f}s",
    )
    log(f"wrote {out}/images ({meta['frames']} frames) + manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
