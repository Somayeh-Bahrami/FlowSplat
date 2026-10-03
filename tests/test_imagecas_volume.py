"""Runs on: M4 (CPU only). Tests for src/flowsplat/imagecas_volume.py (Step 0.1, ImageCAS vessel)."""

import numpy as np
import pytest

from flowsplat import imagecas_volume as iv


def _two_tubes(shape=(60, 50, 40)):
    """A long tube (big component) and a short one (small component)."""
    m = np.zeros(shape, dtype=np.uint8)
    m[10:50, 20:24, 18:22] = 1  # 40 x 4 x 4
    m[5:10, 40:42, 5:7] = 1  # 5 x 2 x 2, separate
    return m


def test_largest_component_keeps_big_tube():
    comp = iv.largest_component(_two_tubes())
    assert comp.sum() == 40 * 4 * 4
    assert not comp[5:10, 40:42, 5:7].any()


def test_largest_component_empty_raises():
    with pytest.raises(ValueError):
        iv.largest_component(np.zeros((5, 5, 5)))


def test_crop_to_bbox_margin_in_mm():
    comp = iv.largest_component(_two_tubes())
    crop = iv.crop_to_bbox(comp, spacing_mm=[0.5, 0.5, 1.0], margin_mm=2.0)
    # margin 4 voxels on axes 0/1, 2 voxels on axis 2
    assert crop.shape == (40 + 8, 4 + 8, 4 + 4)
    assert crop.sum() == comp.sum()


def test_expand_to_cube_centres():
    v = np.ones((4, 2, 3))
    c = iv.expand_to_cube(v)
    assert c.shape == (4, 4, 4)
    assert c.sum() == v.sum()
    # axis 1: pad (1, 1); axis 2: pad (0, 1)
    assert c[:, 1:3, 0:3].sum() == v.sum()


def test_label_to_r2g_volume_shape_range_and_geometry():
    lab = _two_tubes()
    vol, info = iv.label_to_r2g_volume(lab, spacing_mm=[0.4, 0.4, 0.8], target_size=64, margin_mm=2.0)
    assert vol.shape == (64, 64, 64)
    assert vol.dtype == np.float32
    assert 0.0 <= vol.min() and vol.max() <= 1.0
    assert info["foreground_voxels_out"] > 0
    # longest crop axis: 40 voxels + 2*5 margin at 0.4 mm = 20 mm
    assert info["cube_side_mm"] == pytest.approx(20.0, abs=0.5)
    assert info["voxel_mm"] == pytest.approx(info["cube_side_mm"] / 64, abs=1e-3)


def test_dice():
    a = np.zeros((4, 4), bool)
    a[:2] = True
    b = np.zeros((4, 4), bool)
    b[1:3] = True
    assert iv.dice(a, a) == 1.0
    assert iv.dice(a, b) == pytest.approx(0.5)
    assert iv.dice(np.zeros(3), np.zeros(3)) == 1.0


def test_mips_shapes():
    m = iv.mips(np.zeros((3, 4, 5)))
    assert [x.shape for x in m] == [(4, 5), (3, 5), (3, 4)]
