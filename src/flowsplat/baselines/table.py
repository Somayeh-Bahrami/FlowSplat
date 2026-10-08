"""Step 4.1: one results table, all methods, held-out test set.

Runs on: M4 or molab (numpy). Reads the per-case artifacts written by
notebooks/step4_baselines.py (or downloaded from it):
  <work>/step2_1/<case>_v60_4drgs/geom_init.npz   our geometry (radius before k)
  <work>/step2_1/<case>_v60_r2g/geom_init.npz     baseline 3 geometry
  <work>/step4/b1/<case>_tdc.npz                  baseline 1 time-density curves
  <work>/step4/b2/<case>_kymo.npz                 baseline 2 kymograph
  <results>/step2.2_test_v60_4drgs_cal.json       FlowSplat (Step 2.2 frozen)
and the Step 1 case / bolus files (ground truth, frame times).

Plan, Step 4.1 DoD: "One results table (Q error, FFR MAE, FFR accuracy,
diameter error) on the held-out test set for all methods." Ref: plan
(baselines and metrics), OURS (assembly).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from flowsplat import ffr, metrics as mt
from flowsplat.baselines import kymo, tdc

V_FIXED_MS = 0.35  # plan, baseline 3: "fixed 0.35 m/s" (fQFR analog), hyperemic
METHODS = ("flowsplat", "b1_4drgs_tdc", "b2_punch_2d", "b3_r2g_fixed_v", "b4_oracle")


def _geom(path, scale=1.0):
    z = np.load(path)
    return z["s"], z["xyz_mm"], z["r_mm"] * scale


def _ffr_rest(s, r, q_rest_m3s, p, xyz, case):
    try:
        f = mt.ffr_case(s, r, q_rest_m3s, p, xyz, case)
        return {"ffr_est": f["ffr_est"], "ffr_est_pop_cfr": f["ffr_est_pop_cfr"],
                "throat_dist_mm": f.get("throat_dist_mm")}
    except ValueError:
        return {}


def evaluate_case(c, work, case_dir, bolus_dir, flowsplat_row, k):
    """All methods on one test case; methods without their artifact are skipped."""
    work = Path(work)
    case = np.load(Path(case_dir) / f"{c}.npz")
    p = json.loads(str(case["params"]))
    t = np.load(Path(bolus_dir) / f"{c}.npz")["t_s"]
    q_gt = p["q_rest_mls"] * 1e-6
    out = {"case": c, "ffr_gt": p["ffr"], "q_gt_mls": p["q_rest_mls"], "q_hyp_gt_mls": p["q_hyp_mls"]}
    res = {}
    g4 = work / "step2_1" / f"{c}_v60_4drgs" / "geom_init.npz"
    if g4.exists():
        s, xyz, r = _geom(g4, k)
        diam = mt.diam_err_from_geometry(xyz, r, case)
        if flowsplat_row is not None:
            q = flowsplat_row["q_mm3s"] * 1e-9
            res["flowsplat"] = {"q_rel_err": abs(q - q_gt) / q_gt, "diam_err_mm": diam,
                                **_ffr_rest(s, r, q, p, xyz, case)}
        f1 = work / "step4" / "b1" / f"{c}_tdc.npz"
        if f1.exists():
            curves = np.load(f1)["curves"]
            v, n = tdc.velocity_from_arrivals(s, tdc.half_rise_times(curves, t))
            if np.isfinite(v):
                q = tdc.q_from_velocity(v, s, r) * 1e-9
                res["b1_4drgs_tdc"] = {"q_rel_err": abs(q - q_gt) / q_gt, "diam_err_mm": diam, "v_mm_s": v,
                                       **_ffr_rest(s, r, q, p, xyz, case)}
        f2 = work / "step4" / "b2" / f"{c}_kymo.npz"
        if f2.exists():
            z = np.load(f2)
            u, _ = kymo.fit_plug_velocity(z["s2d_mm"], z["kymo"], t)
            q = tdc.q_from_velocity(u, s, r) * 1e-9
            res["b2_punch_2d"] = {"q_rel_err": abs(q - q_gt) / q_gt, "diam_err_mm": diam, "u_mm_s": u,
                                  "out_of_plane_deg": float(z["out_of_plane_deg"]),
                                  **_ffr_rest(s, r, q, p, xyz, case)}
    g3 = work / "step2_1" / f"{c}_v60_r2g" / "geom_init.npz"
    if g3.exists():
        s, xyz, r = _geom(g3)
        row = {"diam_err_mm": mt.diam_err_from_geometry(xyz, r, case)}
        try:
            les = ffr.lesion_from_profile(s, r)
            q_hyp = V_FIXED_MS * math.pi * (les["r_n_mm"] * 1e-3) ** 2
            row.update({"q_hyp_rel_err": abs(q_hyp - p["q_hyp_mls"] * 1e-6) / (p["q_hyp_mls"] * 1e-6),
                        "ffr_est": ffr.ffr_from_lesion(q_hyp, les["r_n_mm"], les["r_s_mm"], les["length_mm"])})
        except ValueError:
            pass
        res["b3_r2g_fixed_v"] = row
    # oracle: ground-truth geometry and Q; exact lesion parameters give the
    # ground truth by construction, the GT profile through our lesion finder
    # gives the extraction floor
    les = ffr.lesion_from_profile(case["s"], case["r_mm"])
    res["b4_oracle"] = {"q_rel_err": 0.0, "diam_err_mm": 0.0, "ffr_est": p["ffr"],
                        "ffr_est_profile": ffr.ffr_from_lesion(p["q_hyp_mls"] * 1e-6, les["r_n_mm"], les["r_s_mm"],
                                                               les["length_mm"])}
    out["methods"] = res
    return out


def summarize(rows):
    """Per method: Q error (rest; hyperemic for baseline 3), diameter error,
    FFR MAE and accuracy / sensitivity / specificity at 0.80."""
    table = {}
    for m in METHODS:
        rr = [dict(r["methods"][m], ffr_gt=r["ffr_gt"]) for r in rows if m in r["methods"]]
        if not rr:
            continue
        e = {"n": len(rr)}
        q = [x["q_rel_err"] for x in rr if x.get("q_rel_err") is not None]
        if q:
            e["q_rel_err"] = mt.q_summary(q)
        qh = [x["q_hyp_rel_err"] for x in rr if x.get("q_hyp_rel_err") is not None]
        if qh:
            e["q_hyp_rel_err"] = mt.q_summary(qh)
        dm = [x["diam_err_mm"] for x in rr if x.get("diam_err_mm") is not None]
        if dm:
            e["diam_err_mm"] = {"mean": float(np.mean(dm)), "median": float(np.median(dm)),
                                "frac_below_0p2": float(np.mean(np.array(dm) < 0.2)), "n_found": len(dm)}
        for k in ("ffr_est", "ffr_est_pop_cfr", "ffr_est_profile"):
            fr = [x for x in rr if x.get(k) is not None]
            if fr:
                e[k] = mt.ffr_summary(fr, k)
                # Added Oct 8 after seeing baseline 3 (Young-Tsai gives FFR < 0
                # for very narrow false throats): the same metric with FFR
                # clipped to [0, 1], reported next to the raw one. Ref: OURS.
                if any(not 0.0 <= x[k] <= 1.0 for x in fr):
                    e[k + "_clipped"] = mt.ffr_summary([dict(x, **{k: min(max(x[k], 0.0), 1.0)}) for x in fr], k)
        td = [x["throat_dist_mm"] for x in rr if x.get("throat_dist_mm") is not None]
        if td:
            e["ffr_lesion_misplaced"] = int(np.sum(np.array(td) > 5.0))
        table[m] = e
    return table


def build(cases, work, case_dir, bolus_dir, flowsplat_json, k):
    """Rows for all cases and the summary table."""
    fs = {}
    if flowsplat_json and Path(flowsplat_json).exists():
        d = json.loads(Path(flowsplat_json).read_text())
        fs = {r["case"]: r for r in d["cases"].values() if r["variant"] == "frozen"}
    rows = [evaluate_case(c, work, case_dir, bolus_dir, fs.get(c), k) for c in cases]
    return {"k": k, "cases": rows, "table": summarize(rows)}
