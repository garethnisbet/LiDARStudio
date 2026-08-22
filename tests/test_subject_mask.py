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
