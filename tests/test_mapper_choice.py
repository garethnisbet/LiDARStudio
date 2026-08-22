"""Which reconstruction engine runs first.

GLOMAP averages rotations over every pair at once, so on a scene where many
pairs are ambiguous — a repetitive near-planar background, or a subject with
rotational symmetry — a wrong consensus wins outright and the orbit folds.
Incremental grows one image at a time and can reject a view that contradicts
what it has already built. Measured on the CupPhotos capture from one shared
exhaustive match database: GLOMAP 60 deg of solved viewpoint spread at a
degenerate 0.001 px reprojection error, incremental 90 deg at 0.877 px.
"""

import sys
from pathlib import Path

from lidarstudio import lidar_jobs

sys.path.insert(0, str(Path(lidar_jobs.__file__).parent))
import process_sfm  # noqa: E402

def test_incremental_is_the_default_mapper(monkeypatch, tmp_path):
    """Which mapper runs FIRST is the whole change: on the CupPhotos capture
    GLOMAP's global averaging solved 60 deg of viewpoint spread and a
    degenerate 0.001 px error from the same match database that incremental
    turned into 90 deg at 0.877 px."""
    calls = []

    class FakeRec:
        @staticmethod
        def num_reg_images():
            return 3

        @staticmethod
        def num_points3D():
            return 9

        @staticmethod
        def compute_mean_reprojection_error():
            return 0.9

    class FakePycolmap:
        IncrementalPipelineOptions = type(
            "I", (), {"ba_refine_focal_length": False, "ba_refine_extra_params": False}
        )
        GlobalPipelineOptions = type("G", (), {})

        @staticmethod
        def incremental_mapping(**kw):
            calls.append("incremental")
            return {0: FakeRec}

        @staticmethod
        def global_mapping(**kw):
            calls.append("glomap")
            return {0: FakeRec}

    monkeypatch.setitem(sys.modules, "pycolmap", FakePycolmap)

    process_sfm.run_mapping(tmp_path / "db", tmp_path / "im", tmp_path / "sp", "t")
    assert calls == ["incremental"], "incremental must run first, unasked"

    calls.clear()
    process_sfm.run_mapping(
        tmp_path / "db", tmp_path / "im", tmp_path / "sp", "t", mapper="glomap"
    )
    assert calls == ["glomap"], "--mapper glomap must still be honoured"


def test_the_other_mapper_is_the_fallback(monkeypatch, tmp_path):
    """A mapper that returns nothing must not sink the run."""
    calls = []

    class FakeRec:
        @staticmethod
        def num_reg_images():
            return 3

        @staticmethod
        def num_points3D():
            return 9

        @staticmethod
        def compute_mean_reprojection_error():
            return 0.9

    class FakePycolmap:
        IncrementalPipelineOptions = type(
            "I", (), {"ba_refine_focal_length": False, "ba_refine_extra_params": False}
        )
        GlobalPipelineOptions = type("G", (), {})

        @staticmethod
        def incremental_mapping(**kw):
            calls.append("incremental")
            return {}

        @staticmethod
        def global_mapping(**kw):
            calls.append("glomap")
            return {0: FakeRec}

    monkeypatch.setitem(sys.modules, "pycolmap", FakePycolmap)
    rec = process_sfm.run_mapping(tmp_path / "db", tmp_path / "im", tmp_path / "sp", "t")
    assert calls == ["incremental", "glomap"]
    assert rec is FakeRec
