"""adjust_colour: bake the viewer's brightness/contrast into a cloud or splat."""

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from lidarstudio import edit_ops
from lidarstudio.splat_io import SH_C0


def _write_cloud(path, rgb_u8):
    n = len(rgb_u8)
    arr = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                             ("nx", "f4"), ("intensity", "f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"] = np.arange(n)
    arr["nx"] = 1.0
    arr["intensity"] = 7.0
    arr["red"], arr["green"], arr["blue"] = rgb_u8.T
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def _write_splat(path, rgb, rest):
    n = len(rgb)
    arr = np.zeros(n, dtype=[(f, "f4") for f in (
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")])
    dc = (rgb - 0.5) / SH_C0
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = dc.T
    arr["f_rest_0"] = rest
    arr["opacity"] = 2.0
    arr["rot_0"] = 1.0
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def test_cloud_brightness_and_contrast(tmp_path):
    src = tmp_path / "c.ply"
    _write_cloud(src, np.array([[0, 128, 255], [64, 64, 64]], np.uint8))
    out = tmp_path / "c_adj.ply"

    r = edit_ops.adjust_colour(src, out, brightness=0.1, contrast=2.0)

    v = PlyData.read(str(out))["vertex"].data
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(float) / 255
    src_rgb = np.array([[0, 128, 255], [64, 64, 64]]) / 255
    expect = np.clip((src_rgb - 0.5) * 2.0 + 0.6, 0, 1)
    assert np.allclose(rgb, expect, atol=1 / 255)
    assert v.dtype["red"] == np.uint8
    # every other field survives untouched
    assert np.all(v["nx"] == 1.0) and np.all(v["intensity"] == 7.0)
    assert list(v["x"]) == [0, 1]
    assert r == {"kind": "cloud", "total": 2, "brightness": 0.1, "contrast": 2.0,
                 "output": str(out)}


def test_neutral_is_lossless(tmp_path):
    src = tmp_path / "c.ply"
    rgb = np.array([[1, 2, 3], [250, 128, 0]], np.uint8)
    _write_cloud(src, rgb)
    edit_ops.adjust_colour(src, src, 0.0, 1.0)   # in place
    v = PlyData.read(str(src))["vertex"].data
    assert np.array_equal(np.stack([v["red"], v["green"], v["blue"]], 1), rgb)


def test_splat_dc_follows_display_formula(tmp_path):
    src = tmp_path / "s.ply"
    rgb = np.array([[0.2, 0.5, 0.8], [0.4, 0.4, 0.4]])
    _write_splat(src, rgb, rest=np.array([0.3, -0.1]))

    edit_ops.adjust_colour(src, src, brightness=-0.1, contrast=1.5)   # in place

    v = PlyData.read(str(src))["vertex"].data
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)
    shown = 0.5 + SH_C0 * dc
    assert np.allclose(shown, np.clip((rgb - 0.5) * 1.5 + 0.4, 0, 1), atol=1e-5)
    assert np.allclose(v["f_rest_0"], [0.45, -0.15], atol=1e-6)
    assert np.all(v["opacity"] == 2.0)


def test_uncoloured_cloud_is_refused(tmp_path):
    src = tmp_path / "plain.ply"
    arr = np.zeros(3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(src))
    with pytest.raises(ValueError, match="no colours"):
        edit_ops.adjust_colour(src, tmp_path / "o.ply", 0.2, 1.0)
