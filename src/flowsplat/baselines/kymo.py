"""Baseline 2: PUNCH-style 1D fit on a 2D projected kymograph.

Runs on: M4 (numpy / scipy); the fixed-view projections are rendered on
molab (render_tigre.render_case with a fixed angle).

Plan, Baselines: "PUNCH-style 1D advection fit on the 2D projected
kymograph (exposes foreshortening bias)". A conventional angiogram has a
fixed view, so the same bolus is rendered from one gantry angle (frame 0)
for all 133 frames. The kymograph samples the image along the vessel
centerline projected to the detector; arc length is measured in the image
and scaled to the isocenter (standard 2D calibration), so out-of-plane
segments are foreshortened. The 1D model is plug-flow advection,
K(s, t) = C(t - s / u) for the peak-normalised kymograph (PUNCH Eq 2 without
dispersion, Pe argument), fitted for u by least squares. The baseline is
given the ground-truth 3D centerline projected to the image (perfect 2D
centerline extraction, an advantage) and our 3D area for Q = u x mean A,
so only its velocity differs from FlowSplat. Ref: PUNCH (kymograph,
Materials and Methods, Fig 2; Eq 2), OURS (projection, fit details).
"""

from __future__ import annotations

import math

import numpy as np

from flowsplat.model import DEFAULT_CONVENTION

INLET_MM = 3.0  # OURS: inlet curve = mean of the first 3 mm of the kymograph


def project_points(world_mm, theta, gd, convention=None):
    """Detector coordinates (u, v in mm, detector plane) and depth along the
    central ray of world points, with the frame of model.ConeProjector
    (calibrated against TIGRE). Ref: model.ConeProjector.frame."""
    c = dict(DEFAULT_CONVENTION if convention is None else convention)
    th = c["s_theta"] * float(theta) + c["theta0"]
    cos, sin = math.cos(th), math.sin(th)
    dso, dsd = float(gd["DSO"]), float(gd["DSD"])
    S = np.array([dso * cos, dso * sin, 0.0])
    e = np.array([-cos, -sin, 0.0])
    ua = c["s_u"] * np.array([-sin, cos, 0.0])
    va = c["s_v"] * np.array([0.0, 0.0, 1.0])
    d = np.asarray(world_mm, float) - S
    depth = d @ e
    return dsd * (d @ ua) / depth, dsd * (d @ va) / depth, depth


def pixel_coords(u_mm, v_mm, gd):
    """Fractional (row, col) of detector points; pixel centres as in
    model.ConeProjector (u = (j + 0.5 - nu / 2) du)."""
    nv, nu = (int(x) for x in gd["nDetector"])
    dv, du = (float(x) for x in gd["dDetector"])
    return np.asarray(v_mm) / dv + nv / 2 - 0.5, np.asarray(u_mm) / du + nu / 2 - 0.5


def kymograph(proj, rows, cols):
    """(N points, T frames) bilinear samples of every frame along the curve."""
    from scipy.ndimage import map_coordinates

    p = np.asarray(proj, np.float32)
    return np.stack([map_coordinates(p[j], [rows, cols], order=1, mode="nearest") for j in range(len(p))], axis=1)


def arc_length_isocenter(u_mm, v_mm, gd):
    """Image arc length scaled to the isocenter plane (divide by DSD / DSO)."""
    du = np.diff(np.asarray(u_mm, float))
    dv = np.diff(np.asarray(v_mm, float))
    mag = float(gd["DSD"]) / float(gd["DSO"])
    return np.concatenate([[0.0], np.cumsum(np.hypot(du, dv))]) / mag


def fit_plug_velocity(s_mm, kymo, t, trim_mm=5.0, u_range=(5.0, 2000.0), n_grid=200):
    """Least-squares plug-flow velocity (mm/s) of a kymograph with s = 0 at
    the inlet. Rows are normalised to their peak; the inlet curve is the
    mean of the rows within INLET_MM, placed at their mean arc length s_ref;
    prediction K(s, t) = C(t - (s - s_ref) / u).
    Grid search on log u, then golden-section refinement. Returns
    (u, loss). Ref: PUNCH Eq 2 (advection limit), OURS (fit)."""
    s = np.asarray(s_mm, float)
    k = np.asarray(kymo, float)
    t = np.asarray(t, float)
    peak = k.max(axis=1, keepdims=True)
    kn = np.where(peak > 0, k / np.where(peak > 0, peak, 1.0), 0.0)
    inlet = s <= INLET_MM
    cin = kn[inlet].mean(axis=0)
    s_ref = float(s[inlet].mean())  # the inlet curve sits at the mean position of its rows
    rows = (s >= trim_mm) & (s <= s[-1] - trim_mm) & (peak[:, 0] > 0.2 * peak.max())

    def loss(log_u):
        u = math.exp(log_u)
        pred = np.stack([np.interp(t - (si - s_ref) / u, t, cin, left=cin[0], right=cin[-1]) for si in s[rows]])
        return float(np.mean((pred - kn[rows]) ** 2))

    grid = np.linspace(math.log(u_range[0]), math.log(u_range[1]), n_grid)
    vals = [loss(g) for g in grid]
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
    gr = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        c1, c2 = b - gr * (b - a), a + gr * (b - a)
        if loss(c1) < loss(c2):
            b = c2
        else:
            a = c1
    lu = 0.5 * (a + b)
    return float(math.exp(lu)), loss(lu)
