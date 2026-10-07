"""Young-Tsai stenosis pressure drop and FFR.

Runs on: M4 (numpy only).

Step 1.1 uses it for ground-truth FFR; Step 3.1 will call the same function
on reconstructed (A, Q).
"""

from __future__ import annotations

import numpy as np

MMHG_PA = 133.322  # 1 mmHg in Pa

# Constants of the plan's worked example (plan, Approach, FFR).
MU_PA_S = 3.5e-3  # blood viscosity
RHO_KG_M3 = 1060.0  # blood density
K_T = 1.52  # expansion (turbulence) coefficient
P_A_MMHG = 90.0  # aortic pressure


def young_tsai_dp(q_m3s, a_n_m2, a_s_m2, length_m, mu=MU_PA_S, rho=RHO_KG_M3, k_t=K_T):
    """Pressure drop across one stenosis, in Pa. Returns (total, viscous, expansion).

    dP = (8 pi mu L A_n / A_s^2) V + (rho K_t / 2) (A_n / A_s - 1)^2 V^2,
    V = Q / A_n (mean velocity in the normal segment).

    Ref: Young, Tsai, J Biomech 1973 (citation and equation number
    UNVERIFIED, flagged in the plan). Form and constants as written in the
    plan, Approach, FFR (worked example: 422 Pa + 1305 Pa).
    """
    q = np.asarray(q_m3s, dtype=float)
    a_n = np.asarray(a_n_m2, dtype=float)
    a_s = np.asarray(a_s_m2, dtype=float)
    v = q / a_n
    viscous = 8.0 * np.pi * mu * length_m * a_n / a_s**2 * v
    expansion = 0.5 * rho * k_t * (a_n / a_s - 1.0) ** 2 * v**2
    return viscous + expansion, viscous, expansion


def ffr_from_dp(dp_pa, p_a_mmhg=P_A_MMHG):
    """FFR = (P_a - dP) / P_a.

    Ref: plan, Approach, FFR (definition used there with the Young-Tsai dP).
    """
    p_a = p_a_mmhg * MMHG_PA
    return (p_a - np.asarray(dp_pa, dtype=float)) / p_a


# ---- Step 3.1: lesion parameters from a radius profile (no ground truth) ----
# OURS: reference windows for the local reference radius, chosen so that
# they clear the longest synthetic lesion (half-length 7.5 mm,
# data_gen.STENOSIS_LEN_MM) and stay local on a tapering LAD.
REF_INNER_MM = 8.0
REF_OUTER_MM = 18.0
TRIM_MM = 5.0
# Step 3.1 secondary FFR: one population coronary flow reserve for all cases
# (midpoint of PUNCH's typical range 1.5 to 4.0 used by data_gen.CFR).
CFR_POP = 2.75


def local_reference(s, r, inner=REF_INNER_MM, outer=REF_OUTER_MM):
    """Reference radius at every s: mean of the medians of r in the windows
    [s - outer, s - inner] and [s + inner, s + outer] (only the windows that
    contain samples). An interpolated healthy reference as in quantitative
    coronary angiography. Ref: OURS (QCA-style reference interpolation; the
    QCA source is UNVERIFIED).
    """
    s = np.asarray(s, float)
    r = np.asarray(r, float)
    out = np.full_like(r, np.nan)
    for i, si in enumerate(s):
        meds = []
        for lo, hi in ((si - outer, si - inner), (si + inner, si + outer)):
            m = (s >= lo) & (s <= hi)
            if m.sum() >= 3:
                meds.append(np.median(r[m]))
        if meds:
            out[i] = np.mean(meds)
    return out


def lesion_from_profile(s, r, trim_mm=TRIM_MM):
    """Stenosis parameters from a radius profile R(s), without ground truth.

    Throat s0 = argmax of the relative deficit 1 - R / R_ref (R_ref from
    local_reference) inside the trimmed vessel; r_n = R_ref(s0); r_s = R(s0);
    L = 2 x the full width at half maximum of the deficit around s0 (exact
    for the cosine lesion of data_gen.insert_stenosis, whose deficit FWHM is
    L / 2). Ref: OURS. Inputs in mm; returns mm.
    """
    s = np.asarray(s, float)
    r = np.asarray(r, float)
    if s[0] > s[-1]:
        s, r = s[::-1], r[::-1]
    ref = local_reference(s, r)
    d = 1.0 - r / ref
    ok = np.isfinite(d) & (s >= s[0] + trim_mm) & (s <= s[-1] - trim_mm)
    if not ok.any():
        raise ValueError("profile too short for the lesion search")
    i0 = int(np.argmax(np.where(ok, d, -np.inf)))
    half = 0.5 * d[i0]
    lo = i0
    while lo > 0 and np.isfinite(d[lo - 1]) and d[lo - 1] >= half:
        lo -= 1
    hi = i0
    while hi < len(s) - 1 and np.isfinite(d[hi + 1]) and d[hi + 1] >= half:
        hi += 1
    # FWHM with linear interpolation at both crossings
    def _cross(a, b):
        if a == b or not np.isfinite(d[b]):
            return s[a]
        return s[a] + (half - d[a]) * (s[b] - s[a]) / (d[b] - d[a])
    s_lo = _cross(lo, lo - 1) if lo > 0 else s[lo]
    s_hi = _cross(hi, hi + 1) if hi < len(s) - 1 else s[hi]
    return {"s0_mm": float(s[i0]), "r_n_mm": float(ref[i0]), "r_s_mm": float(r[i0]),
            "length_mm": float(2.0 * (s_hi - s_lo)), "deficit": float(d[i0])}


def ffr_from_lesion(q_hyp_m3s, r_n_mm, r_s_mm, length_mm):
    """FFR with Young-Tsai from lesion parameters in mm and Q in m^3/s.
    Ref: plan, Approach, FFR (as in data_gen ground truth)."""
    a_n = np.pi * (r_n_mm * 1e-3) ** 2
    a_s = np.pi * (r_s_mm * 1e-3) ** 2
    dp, _, _ = young_tsai_dp(q_hyp_m3s, a_n, a_s, length_mm * 1e-3)
    return float(ffr_from_dp(dp))
