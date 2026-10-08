"""Baseline 1 (4DRGS + post-hoc time-density-curve velocity) and shared
flow helpers.

Runs on: M4 (numpy). The time-density curves come from the 4DRGS 4D model
(baselines/drgs_tdc.py on molab).

Plan, Baselines: "4DRGS (free-time Gaussians) + post-hoc velocity from
time-density curves". The arrival time of the bolus at each centerline
point is the half-rise time of its time-density curve; the mean velocity
is the inverse slope of arrival time over arc length; Q = v x mean lumen
area (frame-count / TDC style flow, as used by QFR-type methods; source
UNVERIFIED). Ref: 4DRGS (model), OURS (TDC post-processing details).
"""

from __future__ import annotations

import numpy as np

TRIM_MM = 5.0  # same end trim as the Step 2 evaluation (OURS)
MIN_PEAK_FRAC = 0.2  # OURS: ignore curves whose peak is below 20 % of the strongest


def half_rise_times(curves, t, frac=0.5, min_peak_frac=MIN_PEAK_FRAC):
    """First time each curve reaches `frac` of its own maximum (linear
    interpolation between frames); NaN for weak curves. curves: (N, T)."""
    c = np.asarray(curves, float)
    t = np.asarray(t, float)
    peak = c.max(axis=1)
    out = np.full(len(c), np.nan)
    gmax = peak.max() if len(peak) else 0.0
    for i in range(len(c)):
        if peak[i] <= 0 or peak[i] < min_peak_frac * gmax:
            continue
        thr = frac * peak[i]
        j = int(np.argmax(c[i] >= thr))
        if j == 0:
            out[i] = t[0]
            continue
        a, b = c[i, j - 1], c[i, j]
        out[i] = t[j - 1] + (thr - a) * (t[j] - t[j - 1]) / (b - a) if b > a else t[j]
    return out


def velocity_from_arrivals(s_mm, t_arr, trim_mm=TRIM_MM):
    """Mean velocity (mm/s) from arrival times: least squares t = t0 + s / v
    on the trimmed vessel (errors are in time). Returns (v, n_used); v is
    positive whatever the direction of s."""
    s = np.asarray(s_mm, float)
    ta = np.asarray(t_arr, float)
    lo, hi = min(s[0], s[-1]) + trim_mm, max(s[0], s[-1]) - trim_mm
    m = np.isfinite(ta) & (s >= lo) & (s <= hi)
    if m.sum() < 5:
        return float("nan"), int(m.sum())
    slope = np.polyfit(s[m], ta[m], 1)[0]
    return (float(1.0 / abs(slope)) if slope != 0 else float("inf")), int(m.sum())


def mean_area_mm2(s_mm, r_mm, trim_mm=TRIM_MM):
    """Mean lumen area pi R^2 on the trimmed vessel (mm^2)."""
    s = np.asarray(s_mm, float)
    r = np.asarray(r_mm, float)
    lo, hi = min(s[0], s[-1]) + trim_mm, max(s[0], s[-1]) - trim_mm
    m = (s >= lo) & (s <= hi)
    return float(np.mean(np.pi * r[m] ** 2))


def q_from_velocity(v_mm_s, s_mm, r_mm, trim_mm=TRIM_MM):
    """Q (mm^3/s) = v x mean area. Ref: OURS (TDC flow convention)."""
    return float(v_mm_s * mean_area_mm2(s_mm, r_mm, trim_mm))
