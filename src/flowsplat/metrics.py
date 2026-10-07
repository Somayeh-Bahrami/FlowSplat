"""Step 3.1: FFR from a reconstruction and summary metrics.

Runs on: M4 (numpy only); also called on molab from notebooks/step2_train.py.

FFR of a reconstruction: lesion parameters from the estimated radius
profile (ffr.lesion_from_profile, no ground truth), hyperemic flow from the
recovered rest flow, Young-Tsai (plan, Approach, FFR). The bolus is imaged
at rest, so hyperemic flow needs a coronary flow reserve (CFR):
  primary:   Q_hyp = CFR_case x Q_rest,est (CFR of the case, as used for the
             ground truth; isolates the reconstruction error)
  secondary: Q_hyp = ffr.CFR_POP x Q_rest,est (one population CFR)
(user decision, Oct 6). Oracle decomposition (plan, Step 3.1 fallback):
estimated lesion with true Q, true lesion with estimated Q.
"""

from __future__ import annotations

import numpy as np

from flowsplat import ffr

FFR_CUT = 0.80  # plan, Evaluation: accuracy / sensitivity / specificity at FFR 0.80


def ffr_case(s_est, r_est, q_rest_est_m3s, params, xyz_est=None, case=None):
    """FFR of one reconstruction and its oracle decomposition.

    s_est, r_est: estimated arc length and radius (mm) along the vessel;
    params: case parameters of data_gen (ground truth). With xyz_est (lumen
    frame, mm) and the case arrays, also the distance from the detected
    throat to the true throat. Ref: plan Step 3.1 ("Young-Tsai on recovered
    A(s), Q", "oracle comparison"); OURS (decomposition, CFR handling).
    """
    les = ffr.lesion_from_profile(s_est, r_est)
    cfr = params["cfr"]
    q_rest_gt = params["q_rest_mls"] * 1e-6
    gt_les = (params["r_ref_at_lesion_mm"], params["r_min_mm"], params["stenosis_len_mm"])
    est_les = (les["r_n_mm"], les["r_s_mm"], les["length_mm"])
    out = {
        "ffr_gt": params["ffr"],
        "ffr_est": ffr.ffr_from_lesion(cfr * q_rest_est_m3s, *est_les),
        "ffr_est_pop_cfr": ffr.ffr_from_lesion(ffr.CFR_POP * q_rest_est_m3s, *est_les),
        "ffr_true_lesion_est_q": ffr.ffr_from_lesion(cfr * q_rest_est_m3s, *gt_les),
        "ffr_est_lesion_true_q": ffr.ffr_from_lesion(cfr * q_rest_gt, *est_les),
        "ffr_gt_pop_cfr": ffr.ffr_from_lesion(ffr.CFR_POP * q_rest_gt, *gt_les),
        "lesion": les,
    }
    if xyz_est is not None and case is not None:
        s_arr = np.asarray(s_est, float)
        i0 = int(np.argmin(np.abs(s_arr - les["s0_mm"])))
        k = int(np.argmin(np.abs(case["s"] - params["stenosis_center_mm"])))
        out["throat_dist_mm"] = float(np.linalg.norm(np.asarray(xyz_est)[i0] - case["xyz_mm"][k]))
    return out


def ffr_summary(rows, key="ffr_est", cut=FFR_CUT):
    """MAE and accuracy / sensitivity / specificity at the cut-off
    (positive = FFR <= cut, the usual ischemia definition, UNVERIFIED
    whether <= or < in the plan). rows: dicts with ffr_gt and `key`."""
    g = np.array([r["ffr_gt"] for r in rows], float)
    e = np.array([r[key] for r in rows], float)
    pos, pred = g <= cut, e <= cut
    return {
        "n": int(len(g)),
        "mae": float(np.abs(e - g).mean()),
        "bias": float((e - g).mean()),
        "accuracy": float((pos == pred).mean()),
        "sensitivity": float((pred & pos).sum() / pos.sum()) if pos.any() else None,
        "specificity": float((~pred & ~pos).sum() / (~pos).sum()) if (~pos).any() else None,
        "n_positive": int(pos.sum()),
    }


def q_summary(q_rel_err, cut=0.10):
    """Mean / median / max relative Q error and the fraction below the cut."""
    q = np.asarray(q_rel_err, float)
    return {"mean": float(q.mean()), "median": float(np.median(q)), "max": float(q.max()),
            "frac_below": float((q < cut).mean()), "n": int(len(q))}
