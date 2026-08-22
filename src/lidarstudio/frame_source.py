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
from pathlib import Path


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


def image_set_keys(scan: Path) -> dict:
    """name → key_ns for an ingested image set."""
    return {r["name"]: int(r["key_ns"]) for r in read_manifest(scan)}


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
        return len(read_manifest(scan))
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
        for row in read_manifest(scan):
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
