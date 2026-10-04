"""Runs on: M4 (CPU only). Tests for the numpy parts of src/flowsplat/render_tigre.py (Step 1.2)."""

import numpy as np
import pytest

from flowsplat import data_gen as dg
from flowsplat import render_tigre as rt


def _tube(r_fn, length=40.0):
    s = np.arange(0, length + 1e-9, 0.5)
    xyz = np.stack([s, 0.3 * s, np.zeros_like(s)], axis=1) / np.sqrt(1.09)  # oblique, |d xyz / ds| = 1
    return s, xyz, r_fn(s)


def test_lumen_volume_matches_pi_r2_L():
    s, xyz, r = _tube(lambda s: np.full_like(s, 1.5))
    lum = rt.lumen_voxels(s, xyz, r)
    vol = lum["frac"].sum() * lum["vox_mm"] ** 3
    # cylinder plus two hemispherical caps from nearest-point mapping at the ends
    expected = np.pi * 1.5**2 * 40.0 + 4 / 3 * np.pi * 1.5**3
    assert vol == pytest.approx(expected, rel=0.03)
    assert lum["s_vox"].min() >= 0 and lum["s_vox"].max() <= 40.0 + 1e-6


def test_lumen_cross_section_follows_stenosis():
    s, xyz, r = _tube(lambda s: dg.insert_stenosis(s, np.full_like(s, 1.5), 0.5, 10.0, 20.0))
    lum = rt.lumen_voxels(s, xyz, r)
    v = lum["frac"] * lum["vox_mm"] ** 3

    def area(s0):
        ds = lum["s_vox"] - s0
        m = (ds >= -1.0 - 1e-6) & (ds < 1.0 - 1e-6)  # half-open: s_vox is on a 0.1 mm grid
        return v[m].sum() / 2.0

    # exact mean area of the cosine profile over the 2 mm slab around the throat
    ss = np.linspace(19.0, 21.0, 2001)
    exact = np.mean(np.pi * dg.insert_stenosis(ss, np.full_like(ss, 1.5), 0.5, 10.0, 20.0) ** 2)
    assert area(20.0) == pytest.approx(exact, rel=0.03)
    assert area(8.0) == pytest.approx(np.pi * 1.5**2, rel=0.02)


def test_axis_permutation_puts_SI_first():
    assert rt.axis_permutation("LAS") == [2, 1, 0]
    assert rt.axis_permutation("RSA") == [1, 2, 0]


def test_geometry_covers_volume():
    gd = rt.tigre_geometry_dict([400, 320, 300])
    nv, nu = gd["nDetector"]
    mag = gd["DSD"] / gd["DSO"]
    assert nu * gd["dDetector"][1] >= np.hypot(320, 300) * 0.25 * mag
    assert nv * gd["dDetector"][0] >= 400 * 0.25 * mag
    assert gd["DSO"] == 750.0 and gd["DSD"] == 1200.0


def test_view_subsets():
    vs = rt.view_subsets(133)
    assert len(vs["30"]["train"]) == 30 and len(vs["60"]["train"]) == 60
    assert vs["133"]["test"] == []
    # 4DRGS Sec 2: j_k = floor((k - 1) T / N) + 1 (1-based)
    assert vs["30"]["train"] == [int(np.floor((k - 1) * 133 / 30)) for k in range(1, 31)]
    assert vs["30"]["train"][0] == 0 and vs["30"]["train"][-1] == 128
    assert vs["133"]["train"] == list(range(133))
    assert not set(vs["30"]["train"]) & set(vs["30"]["test"])
