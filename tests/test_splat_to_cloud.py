"""splat_to_cloud: export a splat's gaussian centres as a coloured point cloud."""

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from lidarstudio import edit_ops
from lidarstudio.splat_io import SH_C0


def _write_splat(path, xyz, rgb, opacity):
    """Minimal 3DGS-style splat PLY: means + SH DC colour + opacity logit."""
    n = len(xyz)
    fields = [(f, "f4") for f in (
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")]
    arr = np.zeros(n, dtype=fields)
    arr["x"], arr["y"], arr["z"] = xyz.T
    dc = (rgb - 0.5) / SH_C0
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = dc.T
    # store opacity as the logit the trainer writes: sigmoid(logit) = opacity
    arr["opacity"] = np.log(opacity / (1.0 - opacity))
    arr["rot_0"] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


@pytest.fixture
def project(tmp_path):
    """A project folder with one splat and its seed cloud's traj sidecar."""
    xyz = np.array([[0, 0, 0], [1, 2, 3], [-1, -2, -3]], float)
    rgb = np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])
    opacity = np.array([0.9, 0.5, 0.01])  # last one is transparent noise
    splat = tmp_path / "splats" / "splat_20260101000000_renamed.ply"
    _write_splat(splat, xyz, rgb, opacity)
    traj = tmp_path / "pointclouds" / "pointcloud_20260101000000.ply.traj.npz"
    traj.parent.mkdir()
    np.savez(traj, poses=np.eye(4)[None], ts=np.array([0.0]))
    return tmp_path, splat


def test_export_prunes_transparent_and_carries_traj(project):
    tmp_path, splat = project
    out = tmp_path / "pointclouds" / "splat_20260101000000_renamed_points.ply"
    r = edit_ops.splat_to_cloud(str(splat), str(out), min_opacity=0.05)

    assert r["total"] == 3 and r["kept"] == 2 and r["removed"] == 1
    assert r["trainable"] is True
    assert (tmp_path / "pointclouds" / (out.name + ".traj.npz")).exists()

    v = PlyData.read(str(out))["vertex"].data
    assert list(v.dtype.names) == ["x", "y", "z", "red", "green", "blue"]
    assert len(v) == 2
    np.testing.assert_allclose(
        np.stack([v["x"], v["y"], v["z"]], 1), [[0, 0, 0], [1, 2, 3]])
    # SH DC round-trips back to the source colours (uint8 quantisation)
    np.testing.assert_allclose(
        np.stack([v["red"], v["green"], v["blue"]], 1) / 255.0,
        [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1]], atol=0.01)


def test_min_opacity_zero_keeps_everything(project):
    tmp_path, splat = project
    out = tmp_path / "pointclouds" / "all_points.ply"
    r = edit_ops.splat_to_cloud(str(splat), str(out), min_opacity=0.0)
    assert r["kept"] == 3 and r["removed"] == 0


def test_rejects_non_splat(project, tmp_path):
    _, splat = project
    out = tmp_path / "pointclouds" / "x_points.ply"
    cloud = tmp_path / "pointclouds" / "plain.ply"
    edit_ops.splat_to_cloud(str(splat), str(cloud), min_opacity=0.0)  # plain cloud
    with pytest.raises(ValueError, match="not a gaussian-splat"):
        edit_ops.splat_to_cloud(str(cloud), str(out))
