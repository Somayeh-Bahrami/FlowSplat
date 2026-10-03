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
