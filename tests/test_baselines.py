"""Runs on: M4 (CPU only). Tests for Step 4.1 baselines (tdc, kymo, table)."""

import json
import math

import numpy as np
import pytest
import torch

from flowsplat import data_gen as dg
from flowsplat import ffr
from flowsplat import metrics as mt
from flowsplat import model as md
from flowsplat.baselines import kymo, table, tdc

T = np.linspace(0, 4.4, 133)


def _bolus_curves(s, v_mm_s, t0=0.5):
    """Plug-flow gamma-variate curves at arc lengths s (mm)."""
    from flowsplat.transport import gamma_variate

    tt = torch.tensor(T[None, :] - s[:, None] / v_mm_s)
    return gamma_variate(tt, torch.tensor(t0), torch.tensor(3.0), torch.tensor(0.3)).numpy()


def test_half_rise_and_velocity_recover_plug_flow():
    s = np.arange(0, 100.0001, 0.5)
    curves = _bolus_curves(s, 150.0)
    v, n = tdc.velocity_from_arrivals(s, tdc.half_rise_times(curves, T))
    assert v == pytest.approx(150.0, rel=0.02)
    assert n > 150
    # reversed arc length: same speed
    v2, _ = tdc.velocity_from_arrivals(s[::-1].copy(), tdc.half_rise_times(curves, T))
    assert v2 == pytest.approx(150.0, rel=0.02)


def test_q_from_velocity_uses_mean_area():
    s = np.arange(0, 50.0001, 0.5)
    r = np.full_like(s, 1.5)
    assert tdc.q_from_velocity(100.0, s, r) == pytest.approx(100.0 * math.pi * 1.5**2)


def test_project_points_matches_cone_projector():
    gd = {"DSO": 750.0, "DSD": 1200.0, "nDetector": [64, 80], "dDetector": [1.0, 1.0]}
    pj = md.ConeProjector(gd)
    for theta in (0.0, 0.7):
        x = np.array([[5.0, -8.0, 6.0]])
        img = pj.render(torch.tensor(x, dtype=torch.float32), torch.eye(3)[None] * 0.5,
                        torch.tensor([[1.0]]), [theta])[0].numpy()
        u, v, depth = kymo.project_points(x, theta, gd)
        row, col = kymo.pixel_coords(u, v, gd)
        i, j = np.unravel_index(np.argmax(img), img.shape)
        assert abs(i - row[0]) < 1.0 and abs(j - col[0]) < 1.0
        assert depth[0] > 0


def test_arc_length_isocenter_scales_by_magnification():
    gd = {"DSO": 750.0, "DSD": 1200.0}
    u = np.linspace(0, 16, 9)
    assert kymo.arc_length_isocenter(u, np.zeros_like(u), gd)[-1] == pytest.approx(10.0)


def test_fit_plug_velocity_recovers_speed():
    s = np.arange(0, 80.0001, 0.5)
    k = _bolus_curves(s, 120.0) * np.linspace(1.0, 0.6, len(s))[:, None]  # brightness varies along s
    u, loss = kymo.fit_plug_velocity(s, k, T)
    assert u == pytest.approx(120.0, rel=0.03)
    assert loss < 1e-3


def test_diam_err_zero_on_ground_truth_geometry():
    s = np.arange(0, 60.0001, 0.5)
    r = dg.insert_stenosis(s, np.full_like(s, 1.6), 0.5, 10.0, 30.0)
    xyz = np.stack([s, np.zeros_like(s), np.zeros_like(s)], 1)
    case = {"s": s, "xyz_mm": xyz, "r_mm": r,
            "params": json.dumps({"stenosis_center_mm": 30.0, "stenosis_len_mm": 10.0, "r_min_mm": 0.8})}
    assert mt.diam_err_from_geometry(xyz, r, case) == pytest.approx(0.0, abs=1e-9)
    assert mt.diam_err_from_geometry(xyz, r + 0.05, case) == pytest.approx(0.1)


def test_table_oracle_and_missing_artifacts(tmp_path):
    s = np.arange(0, 100.0001, 0.5)
    r_ref = np.linspace(1.8, 1.2, len(s))
    r = dg.insert_stenosis(s, r_ref, 0.5, 10.0, 40.0)
    rn = float(np.interp(40.0, s, r_ref))
    q = 0.15 * math.pi * (rn * 1e-3) ** 2
    p = {"cfr": 2.0, "q_rest_mls": q * 1e6, "q_hyp_mls": 2 * q * 1e6, "r_ref_at_lesion_mm": rn,
         "r_min_mm": rn * 0.5, "stenosis_len_mm": 10.0, "stenosis_center_mm": 40.0}
    p["ffr"] = ffr.ffr_from_lesion(2 * q, rn, rn * 0.5, 10.0)
    (tmp_path / "cases").mkdir()
    (tmp_path / "bolus").mkdir()
    xyz = np.stack([s, np.zeros_like(s), np.zeros_like(s)], 1)
    np.savez(tmp_path / "cases" / "x_0.npz", s=s, xyz_mm=xyz, r_mm=r, params=json.dumps(p))
    np.savez(tmp_path / "bolus" / "x_0.npz", t_s=T)
    # our geometry = GT (k = 1) and a B1 TDC artifact with plug flow at the true speed
    (tmp_path / "step2_1" / "x_0_v60_4drgs").mkdir(parents=True)
    np.savez(tmp_path / "step2_1" / "x_0_v60_4drgs" / "geom_init.npz", s=s, xyz_mm=xyz, r_mm=r)
    v_true = q * 1e9 / tdc.mean_area_mm2(s, r)
    (tmp_path / "step4" / "b1").mkdir(parents=True)
    np.savez(tmp_path / "step4" / "b1" / "x_0_tdc.npz", curves=_bolus_curves(s, v_true))
    out = table.build(["x_0"], tmp_path, tmp_path / "cases", tmp_path / "bolus", None, 1.0)
    m = out["cases"][0]["methods"]
    assert set(m) == {"b1_4drgs_tdc", "b4_oracle"}  # no FlowSplat json, no B2 / B3 artifacts
    assert m["b4_oracle"]["ffr_est"] == p["ffr"]
    assert m["b1_4drgs_tdc"]["q_rel_err"] < 0.03
    assert m["b1_4drgs_tdc"]["diam_err_mm"] == pytest.approx(0.0, abs=1e-9)
    assert out["table"]["b4_oracle"]["ffr_est"]["mae"] == 0.0
