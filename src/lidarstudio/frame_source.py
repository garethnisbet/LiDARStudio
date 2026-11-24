"""One iterator over posed camera frames, whatever the scan source is.

The LiDAR path reads frames out of an ``IMAGE_*.bag``; the image_set path reads
them out of a normalized ``images/`` directory written by ``process_ingest.py``.
Both yield ``(key_ns, bgr_image)``, and ``key_ns`` is the join key that ties a
frame to its pose in the SfM npz.

For image sets that key comes from the PERSISTED manifest, never from
enumeration order. This is the whole reason the module exists: if the key were
positional, one frame dropped between the SfM run and the training run would
shift every subsequent frame onto the wrong pose — a failure that produces a
plausible-looking blurry splat rather than an error.

Imported by both process_sfm.py and process_splat.py, which add this directory
to sys.path.
"""

import csv
import hashlib
import json
from pathlib import Path

# The user's include/exclude markers from the photo viewer, beside the
# manifest. Sections name their end frames rather than indexing them, so the
# file means the same thing to every reader whatever else is filtered.
SELECTION_FILE = "selection.json"


def is_image_set(scan: Path) -> bool:
    """True when ``scan`` is an ingested image set (has images/ + manifest)."""
    scan = Path(scan)
    return (scan / "manifest.csv").is_file() and (scan / "images").is_dir()


def read_manifest(scan: Path):
    """The manifest as a list of dicts, in file order.

    Addressed by column NAME so later phases can add columns (per-frame ARKit
    pose is already reserved) without breaking readers written today.
    """
    with open(Path(scan) / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def read_selection(scan: Path) -> list:
    """The marked sections, ``[{"start", "end", "mode"}]`` with ``mode`` either
    'include' or 'exclude' — empty when nothing is marked or the file is bad
    (a corrupt selection must not silently empty the capture)."""
    f = Path(scan) / SELECTION_FILE
    if not f.is_file():
        return []
    try:
        sections = json.loads(f.read_text()).get("sections") or []
    except (OSError, ValueError, AttributeError):
        return []
    return [
        s for s in sections
        if isinstance(s, dict) and s.get("mode") in ("include", "exclude")
        and s.get("start") and s.get("end")
    ]


def apply_selection(names, sections) -> list:
    """``names`` (manifest order) less what the sections drop.

    Include sections, if there are any, say what to keep and everything
    outside them goes; exclude sections then cut from what is left, so an
    exclude inside an include carves a gap. A section whose end frame is not
    in ``names`` (a stale file) is ignored rather than guessed at.
    """
    names = list(names)
    pos = {n: i for i, n in enumerate(names)}
    inc, exc = [], []
    for s in sections:
        a, b = pos.get(s["start"]), pos.get(s["end"])
        if a is None or b is None:
            continue
        (inc if s["mode"] == "include" else exc).append((min(a, b), max(a, b)))

    def inside(i, ranges):
        return any(a <= i <= b for a, b in ranges)

    return [
        n for i, n in enumerate(names)
        if (not inc or inside(i, inc)) and not inside(i, exc)
    ]


def selected_rows(scan: Path) -> list:
    """Manifest rows the user's markers keep, in file order."""
    rows = read_manifest(scan)
    sections = read_selection(scan)
    if not sections:
        return rows
    keep = set(apply_selection([r["name"] for r in rows], sections))
    return [r for r in rows if r["name"] in keep]


def selection_digest(scan: Path) -> str:
    """Identity of the frame set the markers select: '' when nothing is
    marked, so a reconstruction from before markers existed still matches an
    unmarked capture. Hashes the RESULT, not the file, so moving a marker in a
    way that keeps the same frames does not cost a re-run of SfM."""
    if not read_selection(scan):
        return ""
    names = "\n".join(r["name"] for r in selected_rows(scan))
    return hashlib.sha1(names.encode()).hexdigest()


def selection_stamp(npz: Path) -> Path:
    """Sidecar recording which selection a reconstruction was solved from
    (``sfm_viewmats_X.npz`` → ``sfm_viewmats_X.selection``)."""
    npz = Path(npz)
    return npz.with_name(npz.stem + ".selection")


def selection_matches(npz: Path, scan: Path) -> bool:
    """Whether the reconstruction at ``npz`` was solved from the frames the
    markers select now. One with no sidecar predates markers, so it was solved
    from every frame — a match exactly when nothing is marked."""
    f = selection_stamp(npz)
    try:
        done = f.read_text().strip() if f.is_file() else ""
    except OSError:
        done = ""
    return done == selection_digest(scan)


def image_set_keys(scan: Path) -> dict:
    """name → key_ns for an ingested image set, less any frames the user's
    markers exclude."""
    return {r["name"]: int(r["key_ns"]) for r in selected_rows(scan)}


def find_image_bag(scan: Path) -> Path:
    """The scan's camera bag.

    Keeps process_splat.py's original lenient lookup exactly: prefer the
    ``IMAGE_*.bag`` glob, but fall back to any ``*.bag`` whose name contains
    "IMAGE" in any case, so scans that name the file differently still load.
    """
    scan = Path(scan)
    bags = sorted(scan.glob("IMAGE_*.bag")) or sorted(scan.glob("*.bag"))
    bag = next((b for b in bags if "IMAGE" in b.name.upper()), None)
    if bag is None:
        raise FileNotFoundError(f"no IMAGE_*.bag in {scan}")
    return bag


def frame_count(scan: Path, camera: str = "front") -> int:
    if is_image_set(scan):
        return len(selected_rows(scan))
    import process_pointcloud as ppc

    return sum(
        1 for _ in ppc.read_images(find_image_bag(scan), camera, rot="ccw")
    )


def open_frames(scan: Path, camera: str = "front", rot: str = "ccw"):
    """Yield ``(key_ns, bgr_image)`` for every frame of a scan.

    ``camera`` and ``rot`` apply to the bag path only — an ingested image set is
    already normalized (EXIF orientation applied, single implicit camera), so
    rotating it again would undo that work.
    """
    scan = Path(scan)
    if is_image_set(scan):
        import cv2

        images = scan / "images"
        for row in selected_rows(scan):
            path = images / row["name"]
            if not path.exists():
                continue  # filtered out after the manifest was written
            img = cv2.imread(str(path))
            if img is not None:
                yield int(row["key_ns"]), img
        return

    import process_pointcloud as ppc

    for ts, img in ppc.read_images(find_image_bag(scan), camera, rot=rot):
        yield int(ts), img
