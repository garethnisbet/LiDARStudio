"""--mask-subject: match features on the subject, not on the floor it sits on.

An object on a floor or a desk is a small non-planar island in a large plane,
and a plane gives two-fold ambiguous relative pose. When the background owns
most of the inliers the mapper folds the orbit — measured on the CupPhotos
capture as 60 deg of solved spread from a real ~270 deg walk-around, and
unchanged by exhaustive matching, because the false matches are
homography-consistent and matching harder only reinforces them.

These cover the parts that decide WHAT gets masked; the segmentation model
itself is injected, so none of this needs torch.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from lidarstudio import lidar_jobs

sys.path.insert(0, str(Path(lidar_jobs.__file__).parent))
import process_sfm  # noqa: E402


def _blob(shape=(40, 60), box=(10, 20, 5, 15)):
    """A boolean mask with one rectangular instance in it."""
    m = np.zeros(shape, bool)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = True
    return m


@pytest.fixture
def frames(tmp_path):
    """Five frames on disk; content is irrelevant since detection is faked,
    but the files must exist — the no-detection path reads one for its size."""
    d = tmp_path / "images"
    d.mkdir()
    for i in range(5):
        cv2.imwrite(str(d / f"{i:04d}.jpg"), np.full((40, 60, 3), 200, np.uint8))
    return d


def test_masks_are_named_and_shaped_the_way_colmap_expects(frames, tmp_path):
    out = tmp_path / "masks"
    process_sfm.build_subject_masks(
        frames, out, dilate=0, detector=lambda p: [("cup", 0.99, _blob())]
    )
    # COLMAP looks for "<image name>.png", not "<stem>.png".
    assert sorted(p.name for p in out.glob("*")) == [
        f"{i:04d}.jpg.png" for i in range(5)
    ]
    m = cv2.imread(str(out / "0000.jpg.png"), cv2.IMREAD_GRAYSCALE)
    assert m.shape == (40, 60)
    assert set(np.unique(m)) == {0, 255}  # zero means "ignore this pixel"
    assert m[15, 10] == 255 and m[0, 0] == 0


def test_the_subject_class_is_chosen_by_vote_not_by_one_frame(frames, tmp_path):
    """A background object that outscores the subject in a single frame must
    not become the subject there — otherwise the mask jumps to the wrong thing
    in exactly the frames where the real subject is hardest to see."""
    def detector(path):
        dets = [("cup", 0.9, _blob(box=(10, 20, 5, 15)))]
        if path.name == "0002.jpg":  # one frame where a book scores higher
            dets.append(("book", 0.99, _blob(box=(30, 38, 40, 55))))
        return dets

    out = tmp_path / "masks"
    process_sfm.build_subject_masks(frames, out, dilate=0, detector=detector)
    m = cv2.imread(str(out / "0002.jpg.png"), cv2.IMREAD_GRAYSCALE)
    assert m[15, 10] == 255, "the voted subject (cup) should be kept"
    assert m[34, 45] == 0, "the higher-scoring book is not the subject"


def test_explicit_classes_override_the_vote(frames, tmp_path):
    out = tmp_path / "masks"
    process_sfm.build_subject_masks(
        frames,
        out,
        classes=["book"],
        dilate=0,
        detector=lambda p: [
            ("cup", 0.99, _blob(box=(10, 20, 5, 15))),
            ("book", 0.6, _blob(box=(30, 38, 40, 55))),
        ],
    )
    m = cv2.imread(str(out / "0000.jpg.png"), cv2.IMREAD_GRAYSCALE)
    assert m[34, 45] == 255 and m[15, 10] == 0


def test_a_frame_with_no_detection_is_passed_whole_not_blinded(frames, tmp_path):
    """A frame the segmenter misses must behave as it would without masking.
    An all-zero mask would silently drop it from the capture instead."""
    def detector(path):
        if path.name == "0003.jpg":
            return []
        return [("cup", 0.99, _blob())]

    out = tmp_path / "masks"
    missing = process_sfm.build_subject_masks(
        frames, out, dilate=0, detector=detector
    )
    assert missing == 1
    m = cv2.imread(str(out / "0003.jpg.png"), cv2.IMREAD_GRAYSCALE)
    assert (m == 255).all()


def test_low_confidence_detections_are_ignored(frames, tmp_path):
    out = tmp_path / "masks"
    missing = process_sfm.build_subject_masks(
        frames,
        out,
        classes=["cup"],
        dilate=0,
        detector=lambda p: [("cup", 0.2, _blob())],
    )
    assert missing == 5  # below MASK_SCORE, so every frame falls back to all-pass
    assert (cv2.imread(str(out / "0000.jpg.png"), cv2.IMREAD_GRAYSCALE) == 255).all()


def test_dilation_grows_the_silhouette(frames, tmp_path):
    """Keypoints ON the outline are the ones that separate subject from
    background, so a pixel-tight mask throws away the most useful features."""
    out = tmp_path / "masks"
    process_sfm.build_subject_masks(
        frames, out, dilate=5, detector=lambda p: [("cup", 0.99, _blob())]
    )
    grown = cv2.imread(str(out / "0000.jpg.png"), cv2.IMREAD_GRAYSCALE)
    assert grown[15, 3] == 255, "mask should extend past the instance edge"
    assert (grown > 0).sum() > _blob().sum()


def test_an_unrecognisable_capture_says_so_instead_of_masking_everything(
    frames, tmp_path
):
    with pytest.raises(SystemExit, match="no recognisable subject"):
        process_sfm.build_subject_masks(
            frames, tmp_path / "masks", detector=lambda p: [("dog", 0.99, _blob())]
        )


def test_the_mask_directory_reaches_colmap(monkeypatch, tmp_path):
    """The masks are only worth anything if extract_features is told about
    them — and the unmasked path must stay exactly as it was."""
    seen = {}

    class FakeReader:
        camera_model = ""
        camera_params = ""
        mask_path = None

    class FakePycolmap:
        ImageReaderOptions = FakeReader
        CameraMode = type("M", (), {"SINGLE": "single"})
        SequentialPairingOptions = type(
            "P", (), {"overlap": 0, "quadratic_overlap": False, "loop_detection": False}
        )

        class FeatureExtractionOptions:
            def __init__(self):
                self.sift = type("S", (), {})()

        @staticmethod
        def extract_features(**kw):
            seen.update(kw)

        @staticmethod
        def match_exhaustive(**kw):
            pass

        @staticmethod
        def match_sequential(**kw):
            pass

    monkeypatch.setitem(sys.modules, "pycolmap", FakePycolmap)

    process_sfm.extract_and_match(
        tmp_path / "db", tmp_path / "images", "SIMPLE_RADIAL", None, "exhaustive"
    )
    assert seen["reader_options"].mask_path is None
    assert "extraction_options" not in seen

    process_sfm.extract_and_match(
        tmp_path / "db",
        tmp_path / "images",
        "SIMPLE_RADIAL",
        None,
        "exhaustive",
        mask_dir=tmp_path / "masks",
    )
    assert seen["reader_options"].mask_path == str(tmp_path / "masks")
    # A masked frame yields a fifth of the features, so the detector budget is
    # spent differently: fainter keypoints, higher cap.
    assert seen["extraction_options"].sift.peak_threshold < 0.0066
    assert seen["extraction_options"].sift.max_num_features > 8192


# ── --mask-region: a subject COCO has no name for ────────────────────

def test_a_box_region_parses_to_its_four_corners():
    assert process_sfm.parse_mask_region("0.25,0.1,0.75,0.9") == [
        (0.25, 0.1), (0.75, 0.1), (0.75, 0.9), (0.25, 0.9)
    ]


def test_a_polygon_region_keeps_its_points():
    assert process_sfm.parse_mask_region("0,0,1,0,0.5,1") == [
        (0.0, 0.0), (1.0, 0.0), (0.5, 1.0)
    ]


@pytest.mark.parametrize("spec,match", [
    ("0.1,0.2,0.3", "even count"),
    ("0.1,0.2", "box .4 numbers"),
    ("0.1,0.2,1.5,0.9", r"\[0, 1\]"),
    ("0.8,0.2,0.3,0.9", "x0<x1"),
    ("0.1,0.1,0.5,0.5,0.9,0.9", "encloses no area"),
    ("0.1,0.2,left,0.9", "not a list of numbers"),
])
def test_a_malformed_region_stops_the_run(spec, match):
    """Never a fallback: the failure --mask-region exists to prevent is a mask
    that silently keeps the whole frame, so a bad region must not become one."""
    with pytest.raises(SystemExit, match=match):
        process_sfm.parse_mask_region(spec)


def test_a_region_mask_keeps_exactly_the_stated_box(frames, tmp_path):
    out = tmp_path / "masks"
    # frames are 40 rows x 60 cols; keep the middle half horizontally.
    process_sfm.build_region_masks(
        frames, out, process_sfm.parse_mask_region("0.25,0.0,0.75,1.0")
    )
    m = cv2.imread(str(out / "0000.jpg.png"), cv2.IMREAD_GRAYSCALE)
    assert m.shape == (40, 60)
    assert m[20, 5] == 0 and m[20, 55] == 0, "outside the box must be excluded"
    assert m[20, 30] == 255, "inside the box must be kept"


def test_every_frame_gets_a_region_mask_named_for_it(frames, tmp_path):
    out = tmp_path / "masks"
    process_sfm.build_region_masks(
        frames, out, process_sfm.parse_mask_region("0.1,0.1,0.9,0.9")
    )
    assert sorted(p.name for p in out.glob("*.png")) == [
        f"{i:04d}.jpg.png" for i in range(5)
    ]


def test_a_region_needs_no_segmenter(frames, tmp_path, monkeypatch):
    """The whole point for a non-COCO subject: no torchvision in the path."""
    monkeypatch.setattr(
        process_sfm, "_maskrcnn_detector",
        lambda: pytest.fail("build_region_masks must not load a detector"),
    )
    process_sfm.build_region_masks(
        frames, tmp_path / "masks", process_sfm.parse_mask_region("0.2,0.2,0.8,0.8")
    )


def test_a_too_tight_region_warns(frames, tmp_path, capsys):
    process_sfm.build_region_masks(
        frames, tmp_path / "masks",
        process_sfm.parse_mask_region("0.50,0.50,0.55,0.55"),
    )
    assert "too little for SIFT" in capsys.readouterr().out


# ── --mask-classes: a name the segmenter cannot emit ─────────────────

def _detector_knowing(*labels):
    """A detector that publishes a vocabulary, the way _maskrcnn_detector does."""
    def detect(path):
        return [("cup", 0.99, _blob())]
    detect.labels = frozenset(labels)
    return detect


def test_an_unknown_mask_class_stops_the_run(frames, tmp_path):
    """The silent failure this replaces: no detection matches the name, every
    frame falls through to all-pass, and the run finishes with masking doing
    nothing at all."""
    with pytest.raises(SystemExit, match="no such class"):
        process_sfm.build_subject_masks(
            frames, tmp_path / "masks", classes=["mug"],
            detector=_detector_knowing("cup", "bowl", "cell phone"),
        )


@pytest.mark.parametrize("asked,meant", [
    ("mug", "cup"),            # a synonym: edit distance scores this 0.33
    ("phone", "cell phone"),   # a prefix: difflib finds this one
    ("plant", "potted plant"),
    ("bowls", "bowl"),         # a plain typo
    ("scisors", "scissors"),
    ("glass", "wine glass"),   # the compound rule, with no synonym entry
])
def test_the_rejection_suggests_the_class_that_was_meant(
    frames, tmp_path, asked, meant
):
    with pytest.raises(SystemExit, match=f"did you mean '{meant}'"):
        process_sfm.build_subject_masks(
            frames, tmp_path / "masks", classes=[asked],
            # Includes the distractors the real 80-class list supplies:
            # difflib ranks 'elephant'/'airplane' above 'potted plant' for
            # 'plant', and 'spoon'/'horse' above 'cell phone' for 'phone'.
            detector=_detector_knowing(
                "cup", "bowl", "cell phone", "potted plant", "vase",
                "elephant", "airplane", "spoon", "horse", "scissors", "wine glass",
            ),
        )


def test_a_known_mask_class_is_accepted(frames, tmp_path):
    missing = process_sfm.build_subject_masks(
        frames, tmp_path / "masks", classes=["cup"], dilate=0,
        detector=_detector_knowing("cup", "bowl"),
    )
    assert missing == 0


def test_only_the_unknown_names_are_reported(frames, tmp_path):
    """A partly-right list must point at the wrong entry, not all of them."""
    with pytest.raises(SystemExit) as e:
        process_sfm.build_subject_masks(
            frames, tmp_path / "masks", classes=["cup", "widget"],
            detector=_detector_knowing("cup", "bowl"),
        )
    assert "'widget'" in str(e.value) and "'cup'" not in str(e.value).split("\n\n")[0]


def test_a_detector_without_a_vocabulary_is_not_second_guessed(frames, tmp_path):
    """The injected doubles in these tests publish no labels; validation must
    skip rather than reject everything."""
    missing = process_sfm.build_subject_masks(
        frames, tmp_path / "masks", classes=["anything"], dilate=0,
        detector=lambda p: [("anything", 0.99, _blob())],
    )
    assert missing == 0


def test_turntable_degrades_to_unmasked_rather_than_refusing(frames, tmp_path, capsys):
    """--turntable IMPLIES masking rather than asking for it, so a subject COCO
    cannot name must not block the run. Measured on a plywood house: unmasked
    gave 219/219 frames and a 228 deg turn, against 162 and 43 deg masked."""
    got = process_sfm.build_subject_masks(
        frames, tmp_path / "masks", required=False,
        detector=lambda p: [("dog", 0.99, _blob())],
    )
    assert got is None, "no masks built"
    assert "being solved UNMASKED" in capsys.readouterr().out
    assert not (tmp_path / "masks").exists(), \
        "an empty <output>_masks dir reads as 'masks were made'"


def test_an_explicit_mask_subject_still_refuses(frames, tmp_path):
    with pytest.raises(SystemExit, match="no recognisable subject"):
        process_sfm.build_subject_masks(
            frames, tmp_path / "masks", required=True,
            detector=lambda p: [("dog", 0.99, _blob())],
        )
