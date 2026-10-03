"""Runs on: M4 (CPU only). Tests for src/flowsplat/data_gen.py (Step 1.1) on synthetic trees."""

import json

import nibabel as nib
import numpy as np
import pytest

from flowsplat import data_gen as dg

SP = 0.5  # mm, isotropic synthetic voxels


def _tube(mask, pts_vox, r_vox):
    """Union of balls along a dense polyline (voxel units)."""
    for p, r in zip(pts_vox, r_vox):
        lo = np.maximum(np.floor(p - r - 1).astype(int), 0)
        hi = np.minimum(np.ceil(p + r + 2).astype(int), mask.shape)
        sub = np.indices(hi - lo).reshape(3, -1).T + lo
        sub = sub[np.linalg.norm(sub - p, axis=1) <= r]
        mask[tuple(sub.T)] = 1


def _poly(a, b, n=200):
    return np.linspace(a, b, n)


def synthetic_label():
    """RAS-oriented volume (identity directions, +x = patient right):
    split_components names the component with the SMALLEST x as LCA.

    LCA: trunk (r = 2 mm, 11 mm) from the ostium (its most rightward point)
    going left (-x), then an anterior branch (+y, LAD, about 65 mm) and a
    posterior branch (-y, LCx, about 70 mm, longer), both further left.
    The LCx tip is made wide (a stump) so the widest-endpoint rule would
    fail. Plus a separate RCA at large x.
    """
    shape = (120, 300, 240)
    m = np.zeros(shape, np.uint8)
    ost = np.array([50.0, 150.0, 220.0])
    bif = np.array([40.0, 150.0, 200.0])
    lad_end = bif + np.array([-5.0, 70.0, -110.0]) * 1.0  # anterior-inferior, 130 vox
    lcx_end = bif + np.array([-10.0, -80.0, -100.0]) * 1.08  # posterior, 139 vox
    trunk = _poly(ost, bif, 60)
    _tube(m, trunk, np.full(len(trunk), 4.0))
    lad = _poly(bif, lad_end, 500)
    _tube(m, lad, np.linspace(3.2, 2.2, len(lad)))
    lcx = _poly(bif, lcx_end, 500)
    _tube(m, lcx, np.linspace(3.0, 2.0, len(lcx)))
    _tube(m, lcx[-5:], np.full(5, 5.0))  # wide distal stump
    rca = _poly(np.array([100.0, 150.0, 200.0]), np.array([105.0, 170.0, 60.0]), 300)
    _tube(m, rca, np.full(len(rca), 3.0))
    affine = np.diag([SP, SP, SP, 1.0])
    return m, affine, lad_end, lcx_end


@pytest.fixture(scope="module")
def lad():
    m, aff, lad_end, lcx_end = synthetic_label()
    s, xyz, r, info, _ = dg.extract_lad(m, aff, (SP, SP, SP))
    return s, xyz, r, info, lad_end, lcx_end


def test_extract_lad_picks_anterior_branch(lad):
    s, xyz, r, info, lad_end, lcx_end = lad
    assert info["axcodes"] == "RAS"
    assert info["n_candidates"] >= 2  # both long branches qualify
    assert not info["lad_is_longest"]  # LCx is longer, LAD still chosen (anterior)
    end_vox = xyz[-1] / SP  # xyz is in the cropped frame: compare direction only
    start_vox = xyz[0] / SP
    assert end_vox[1] - start_vox[1] > 20  # LAD goes to +y (anterior)


def test_extract_lad_length_and_radius(lad):
    s, xyz, r, info, *_ = lad
    assert np.allclose(np.diff(s), dg.DS_MM)
    # trunk 11 mm + LAD branch 65 mm, minus a few mm of skeleton end shortening
    assert 68 < s[-1] < 80
    assert r[0] > r[-1]  # radius decreases distally
    assert 1.7 < r[0] < 2.3 and 0.8 < r[-1] < 1.4
    assert xyz[0][0] > xyz[-1][0]  # starts at the rightward ostium
    assert dg.qc_lad(s, r, xyz, info["axcodes"]) is None


def test_qc_lad_rejects():
    s = np.arange(0, 30, 0.5)
    assert "length" in dg.qc_lad(s, np.ones_like(s))
    s = np.arange(0, 100, 0.5)
    assert "radius" in dg.qc_lad(s, np.full_like(s, 5.0))
    # narrow start, wide later: root is a distal tip
    r = np.concatenate([np.full(40, 0.8), np.full(160, 1.8)])
    assert "proximal" in dg.qc_lad(s, r)
    # tapering vessel that ends posterior of the ostium (RAS: +y = anterior)
    r = np.linspace(1.8, 1.0, len(s))
    xyz = np.stack([np.zeros_like(s), -0.3 * s, -s], axis=1)
    assert "anterior" in dg.qc_lad(s, r, xyz, "RAS")
    xyz[:, 1] *= -1
    assert dg.qc_lad(s, r, xyz, "RAS") is None


def test_insert_stenosis_profile():
    s = np.arange(0, 100, 0.5)
    r_ref = np.full_like(s, 1.5)
    r = dg.insert_stenosis(s, r_ref, 0.5, 10.0, 40.0)
    assert r.min() == pytest.approx(0.75)
    assert s[np.argmin(r)] == pytest.approx(40.0)
    assert np.all(r[np.abs(s - 40) > 5] == 1.5)


def test_sample_case_ranges_and_physics():
    s = np.arange(0, 120, 0.5)
    r_ref = np.linspace(1.8, 1.0, len(s))
    for k in range(50):
        c = dg.sample_case(s, r_ref, np.random.default_rng([0, k]))
        assert 0.10 <= c["v_rest_ms"] <= 0.20
        assert 1.5 <= c["cfr"] <= 4.0
        assert c["v_hyp_ms"] <= dg.V_HYP_MAX_MS + 1e-12
        assert 0.30 <= c["stenosis_degree"] <= 0.70
        assert c["r_min_mm"] == pytest.approx(c["r_ref_at_lesion_mm"] * (1 - c["stenosis_degree"]))
        assert dg.FFR_MIN <= c["ffr"] <= c["ffr_rest"] <= 1
        assert np.all(np.diff(c["tau_rest_s"]) > 0)
        assert c["q_hyp_mls"] == pytest.approx(c["cfr"] * c["q_rest_mls"])


def test_sample_case_deterministic():
    s = np.arange(0, 120, 0.5)
    r_ref = np.full_like(s, 1.5)
    a = dg.sample_case(s, r_ref, np.random.default_rng([3, 11, 2]))
    b = dg.sample_case(s, r_ref, np.random.default_rng([3, 11, 2]))
    assert a["ffr"] == b["ffr"] and a["stenosis_center_mm"] == b["stenosis_center_mm"]


def test_timing_check_matches_plan_numbers():
    # Plan: 30 views over 4.4 s -> one view every 0.147 s; 100 mm at 0.15 m/s -> 0.67 s.
    out = dg.timing_check([0.12, 0.667], [50, 100])
    assert out["per_view_set"]["30"]["view_interval_s"] == pytest.approx(0.1467, abs=1e-4)
    assert out["per_view_set"]["30"]["frac_cases_transit_below_1_view"] == pytest.approx(0.5)


def test_generate_end_to_end(tmp_path):
    m, aff, *_ = synthetic_label()
    src = tmp_path / "imagecas"
    src.mkdir()
    nib.save(nib.Nifti1Image(m, aff), str(src / "7.label.nii.gz"))
    nib.save(nib.Nifti1Image(np.zeros((20, 20, 20), np.uint8), aff), str(src / "8.label.nii.gz"))
    summ = dg.generate(src, tmp_path / "out", n_geom=5, per_geom=3, seed=0)
    assert summ["n_geometries"] == 1 and summ["n_cases"] == 3
    assert summ["n_rejected_files"] == 1 and summ["rejects"][0]["case"] == "8"
    assert summ["check_rest_10_20_cms"]
    z = np.load(tmp_path / "out" / "cases" / "7_0.npz")
    p = json.loads(str(z["params"]))
    assert set(z.files) >= {"s", "xyz_mm", "r_ref_mm", "r_mm", "tau_rest_s", "params"}
    assert 0 < p["ffr"] <= 1
    assert (tmp_path / "out" / "qc" / "7.png").exists()
    assert (tmp_path / "out" / "cases.csv").read_text().count("\n") == 4


def test_resample_removes_staircase_length_bias():
    # straight oblique line as a 26-connected voxel staircase, 0.5 mm voxels
    n = 200
    vox = np.stack([np.arange(n), np.round(np.arange(n) * 0.4), np.round(np.arange(n) * 0.7)], axis=1)
    true_len = np.linalg.norm((vox[-1] - vox[0]) * SP)
    raw_len = np.linalg.norm(np.diff(vox * SP, axis=0), axis=1).sum()
    assert raw_len > 1.05 * true_len  # the bias exists
    s, _, _ = dg.resample_and_smooth(vox * SP, np.ones(n))
    assert s[-1] == pytest.approx(true_len, rel=0.02)
