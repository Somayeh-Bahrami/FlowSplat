"""Runs on: M4 (CPU only). Tests for Step 3.1: ffr lesion extraction, metrics, sweeps."""

import json
import math

import numpy as np
import pytest

from flowsplat import data_gen as dg
from flowsplat import ffr
from flowsplat import metrics as mt
from flowsplat import sweeps as sw


def _profile(deg=0.5, length=10.0, center=40.0, taper=(1.8, 1.2)):
    s = np.arange(0, 100.0001, 0.5)
    r_ref = np.linspace(*taper, len(s))
    return s, r_ref, dg.insert_stenosis(s, r_ref, deg, length, center)


def test_lesion_from_profile_recovers_cosine_lesion():
    s, r_ref, r = _profile()
    les = ffr.lesion_from_profile(s, r)
    assert les["s0_mm"] == pytest.approx(40.0, abs=0.5)
    assert les["r_s_mm"] == pytest.approx(np.interp(40.0, s, r_ref) * 0.5, abs=0.02)
    assert les["r_n_mm"] == pytest.approx(np.interp(40.0, s, r_ref), abs=0.03)
    assert les["length_mm"] == pytest.approx(10.0, abs=1.0)
    # reversed arc length gives the same lesion
    les2 = ffr.lesion_from_profile(s[::-1], r[::-1])
    assert les2["s0_mm"] == pytest.approx(les["s0_mm"])


def test_lesion_not_fooled_by_distal_taper():
    """Distal end is thinner than the throat; the relative deficit still finds the lesion."""
    s, _, r = _profile(deg=0.3, center=30.0, taper=(2.0, 0.7))
    assert ffr.lesion_from_profile(s, r)["s0_mm"] == pytest.approx(30.0, abs=1.0)


def test_ffr_from_lesion_matches_worked_example():
    # plan worked example: 3.0 mm normal, 1.5 mm throat, 5 mm, 3 mL/s -> 0.856
    assert ffr.ffr_from_lesion(3e-6, 1.5, 0.75, 5.0) == pytest.approx(0.856, abs=0.001)


def test_ffr_case_exact_on_true_profile_and_q():
    s, r_ref, r = _profile(deg=0.5, length=10.0, center=40.0)
    rn = float(np.interp(40.0, s, r_ref))
    q_rest = 0.15 * math.pi * (rn * 1e-3) ** 2
    params = {"cfr": 2.0, "q_rest_mls": q_rest * 1e6, "r_ref_at_lesion_mm": rn, "r_min_mm": rn * 0.5,
              "stenosis_len_mm": 10.0}
    params["ffr"] = ffr.ffr_from_lesion(2.0 * q_rest, rn, rn * 0.5, 10.0)
    out = mt.ffr_case(s, r, q_rest, params)
    assert out["ffr_true_lesion_est_q"] == pytest.approx(params["ffr"])
    assert out["ffr_est"] == pytest.approx(params["ffr"], abs=0.01)
    assert out["ffr_est_pop_cfr"] < out["ffr_est"]  # CFR_POP 2.75 > 2.0: more flow, lower FFR


def test_ffr_summary_counts():
    rows = [{"ffr_gt": 0.7, "e": 0.75}, {"ffr_gt": 0.9, "e": 0.85}, {"ffr_gt": 0.78, "e": 0.82}]
    sm = mt.ffr_summary(rows, "e")
    assert sm["mae"] == pytest.approx(0.0467, abs=1e-3)
    assert sm["accuracy"] == pytest.approx(2 / 3)
    assert sm["sensitivity"] == pytest.approx(0.5)
    assert sm["specificity"] == pytest.approx(1.0)


def _case_dict():
    s, r_ref, r = _profile()
    xyz = np.stack([0.3 * s, s, 0.2 * s], 1)
    rn = float(np.interp(40.0, s, r_ref))
    p = {"cfr": 2.0, "q_rest_mls": 1.0, "r_ref_at_lesion_mm": rn, "r_min_mm": rn * 0.5, "stenosis_len_mm": 10.0,
         "stenosis_center_mm": 40.0, "ffr": 0.9}
    return {"s": s, "xyz_mm": xyz, "r_mm": r, "r_ref_mm": r_ref, "tau_rest_s": np.zeros_like(s), "params": json.dumps(p)}


def test_with_flow_sets_velocity_and_transit():
    c = sw.with_flow(_case_dict(), 0.4)
    p = json.loads(c["params"])
    a_n = math.pi * (p["r_ref_at_lesion_mm"] * 1e-3) ** 2
    assert p["q_rest_mls"] == pytest.approx(0.4 * a_n * 1e6)
    area = np.pi * (c["r_mm"] * 1e-3) ** 2
    vol = np.sum(0.5 * (area[1:] + area[:-1]) * np.diff(c["s"]) * 1e-3)
    assert c["tau_rest_s"][-1] == pytest.approx(vol / (p["q_rest_mls"] * 1e-6))
    assert p["ffr"] < 1.0


@pytest.mark.parametrize("deg", [0.0, 30.0, 60.0])
def test_with_foreshortening_sets_angle_and_keeps_shape(deg):
    perm = [2, 1, 0]
    c0 = _case_dict()
    c = sw.with_foreshortening(c0, deg, perm)
    assert sw.out_of_plane_deg(c["xyz_mm"], perm) == pytest.approx(deg, abs=1e-6)
    d0 = np.linalg.norm(np.diff(c0["xyz_mm"], axis=0), axis=1)
    d1 = np.linalg.norm(np.diff(c["xyz_mm"], axis=0), axis=1)
    assert np.allclose(d0, d1)  # rigid rotation


def test_resimulate_bolus_peclet_and_shape():
    c = _case_dict()
    t = np.linspace(0, 4.4, 133)
    inlet = np.exp(-((t - 1.0) / 0.3) ** 2)
    b = {"t_s": t, "inlet_noisy": inlet, "c": np.zeros((len(c["s"]), 133)), "angles_deg": np.zeros(133),
         "params": json.dumps({"d_m2s": 1e-6})}
    nb = sw.resimulate_bolus(c, b, d_m2s=1e-5)
    p = json.loads(nb["params"])
    assert p["d_m2s"] == 1e-5 and p["peclet"] > 0
    assert nb["c"].shape == (len(c["s"]), 133)
    assert float(nb["c"].max()) == pytest.approx(1.0, abs=0.05)


def test_settings_complete():
    kinds = {v["kind"] for v in sw.SETTINGS.values()}
    assert kinds == {"views", "noise", "peclet", "flow", "fore", "repeat"}
    assert sw.cases_of("views30") == sw.VAL20 and sw.cases_of("noise_high") == sw.VAL5
