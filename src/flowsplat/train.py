"""Step 2.2: joint optimisation of geometry and flow for one vessel.

Runs on: M4 (CPU torch, tests and the toy prototype) and molab GPU.

Plan, Step 2.2: "Stage A: geometry frozen, fit Q. Stage B: joint fine-tune
with L1 + D-SSIM image loss. First overfit one case, then 20-case
validation."
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from flowsplat import model as md
from flowsplat.transport import Transport

# 4DRGS Sec 3.2 Eq 6 and Sec 4.1: L = (1 - lambda) L1 + lambda L_ssim, lambda = 0.2.
LAMBDA_SSIM = 0.2
# Learning rates of 4DRGS Sec 4.1 (position 1e-4, rotation 1e-3, scale 5e-3,
# in R2-Gaussian's [-1, 1]^3 scene units). Positions here are in mm; with
# volumes of about 100 mm the scene scale is about 50 mm per unit, so 1e-4
# becomes 5e-3 mm. Conversion OURS.
LR_POSITION_MM = 5e-3
LR_ROTATION = 1e-3
LR_SCALE = 5e-3
# OURS: transport learning rate, iteration counts, alignment weight, Q start.
LR_TRANSPORT = 3e-2
ITERS_DIRECTION = 150
ITERS_A = 600
ITERS_B = 1500
LAMBDA_ALIGN = 0.1
# OURS: views rendered per Stage B iteration (random subset, as 3DGS samples
# one view per step); all views at once would need about 60 GB on the GPU.
VIEWS_PER_ITER = 8
V0_MM_S = 150.0  # start velocity: middle of the resting range 10 to 20 cm/s (PUNCH)


# ============================ loss ============================
def _gauss_window(size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    g = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(g**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(x, y, data_range):
    """Mean SSIM of image stacks (V, H, W), 11 x 11 Gaussian window, sigma 1.5.

    Ref: Wang et al. 2004 (standard SSIM; constants K1 0.01, K2 0.03), as
    used by 3DGS / R2-Gaussian / 4DRGS for the D-SSIM loss. Implementation OURS.
    """
    w = _gauss_window(device=x.device, dtype=x.dtype)
    x, y = x[:, None], y[:, None]
    mu_x = F.conv2d(x, w, padding=5)
    mu_y = F.conv2d(y, w, padding=5)
    sxx = F.conv2d(x * x, w, padding=5) - mu_x**2
    syy = F.conv2d(y * y, w, padding=5) - mu_y**2
    sxy = F.conv2d(x * y, w, padding=5) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return m.mean()


def image_loss(pred, gt, data_range):
    """(1 - lambda) L1 + lambda (1 - SSIM), lambda = 0.2. Ref: 4DRGS Sec 3.2 Eq 6."""
    l1 = (pred - gt).abs().mean()
    return (1 - LAMBDA_SSIM) * l1 + LAMBDA_SSIM * (1 - ssim(pred, gt, data_range)), l1


# ============================ case model ============================
class CaseModel:
    """Tube Gaussians (TIGRE world frame, ordered from the inlet) + transport."""

    def __init__(self, s, xyz_world, radius, q_mm3s=None, device="cpu", voxel_mm=0.25):
        self.tube = md.TubeGaussians(s, xyz_world, radius, device=device, voxel_mm=voxel_mm)
        area = float(np.mean(np.pi * np.asarray(radius) ** 2))
        q = V0_MM_S * area if q_mm3s is None else q_mm3s
        self.transport = Transport(q).to(device)

    def render(self, projector, thetas, times):
        tg = self.tube
        rho = self.transport.density(tg.position, tg.area(), tg.sigma_a(), times)
        return projector.render(tg.position, tg.covariance(), rho, thetas)

    def estimates(self):
        tg = self.tube
        return {
            "q_mm3s": float(self.transport.q().detach()),
            "t0_s": float(self.transport.t0.detach()),
            "alpha": float(self.transport.log_alpha.exp().detach()),
            "beta_s": float(self.transport.log_beta.exp().detach()),
            "k_per_mm": float(self.transport.log_k.exp().detach()),
            "radius_mm": tg.radius().detach().cpu().numpy(),
            "position_world_mm": tg.position.detach().cpu().numpy(),
        }


def _init_inlet(model, proj, times):
    """Start values from the data (OURS): t0 = time when the total projected
    contrast first exceeds 5 % of its maximum minus 0.1 s; k scaled so the
    brightest rendered frame matches the brightest data frame in total."""
    tot = proj.sum(dim=(1, 2))
    first = int(torch.nonzero(tot > 0.05 * tot.max())[0]) if (tot > 0.05 * tot.max()).any() else 0
    with torch.no_grad():
        model.transport.t0.fill_(max(float(times[first]) - 0.1, 0.0))


def _fit_k(model, projector, proj, thetas, times, cache=None):
    with torch.no_grad():
        if cache is not None:
            tg = model.tube
            rho = model.transport.density(tg.position, tg.area(), tg.sigma_a(), times)
            pred = torch.einsum("vn,vnp->vp", rho, cache).reshape(proj.shape)
        else:
            pred = model.render(projector, thetas, times)
        ratio = float(proj.sum(dim=(1, 2)).max() / pred.sum(dim=(1, 2)).max().clamp_min(1e-12))
        model.transport.log_k.add_(math.log(max(ratio, 1e-6)))


def _run(model, projector, proj, thetas, times, params, iters, align=0.0, log_every=50, tag="", cache=None,
         views_per_iter=None, seed=0):
    """Adam on `params`. With `cache` (footprints of frozen geometry, Stage A)
    each image is a matrix product of densities and cached footprints."""
    opt = torch.optim.Adam(params)
    dr = float(proj.max())
    hist = []
    tg = model.tube
    gen = torch.Generator().manual_seed(seed)
    n_v = len(thetas)
    for it in range(iters):
        opt.zero_grad()
        if cache is not None:
            rho = model.transport.density(tg.position.detach(), tg.area().detach(), tg.sigma_a().detach(), times)
            pred = torch.einsum("vn,vnp->vp", rho, cache).reshape(proj.shape)
            target = proj
        elif views_per_iter and views_per_iter < n_v:
            sel = torch.randperm(n_v, generator=gen)[:views_per_iter].tolist()
            pred = model.render(projector, [thetas[i] for i in sel], times[sel])
            target = proj[sel]
        else:
            pred = model.render(projector, thetas, times)
            target = proj
        loss, l1 = image_loss(pred, target, dr)
        if align > 0:
            loss = loss + align * model.tube.alignment_loss()
        loss.backward()
        opt.step()
        if it % log_every == 0 or it == iters - 1:
            hist.append({"stage": tag, "it": it, "loss": float(loss.detach()), "l1": float(l1.detach()),
                         "q": float(model.transport.q().detach())})
    return hist


def fit_case(proj, thetas, times, s, xyz_world, radius, projector, device="cpu",
             iters_dir=ITERS_DIRECTION, iters_a=ITERS_A, iters_b=ITERS_B, q_init=None, log=print):
    """Fit one case. proj: (V, nv, nu) training projections (torch), thetas
    (rad) and times (s) of these views; (s, xyz_world, radius): initial tube
    (Step 2.1 FDK skeleton). The inlet end is unknown, so a short Stage A is
    run for both orderings and the one with the lower loss is kept (OURS).

    Returns (model, info dict with history, chosen direction, timings).
    """
    t0 = time.time()
    proj = proj.to(device)
    times = torch.as_tensor(times, dtype=torch.float32, device=device)
    best = None
    for direction in (+1, -1):
        sl = slice(None) if direction > 0 else slice(None, None, -1)
        m = CaseModel(s, xyz_world[sl], radius[sl], q_mm3s=q_init, device=device)
        _init_inlet(m, proj, times)
        with torch.no_grad():
            cache = projector.footprints(m.tube.position, m.tube.covariance(), thetas)
        m.cache = cache
        _fit_k(m, projector, proj, thetas, times, cache)
        tr = list(m.transport.parameters())
        h = _run(m, projector, proj, thetas, times, [{"params": tr, "lr": LR_TRANSPORT}], iters_dir,
                 tag=f"dir{direction:+d}", cache=cache)
        log(f"direction {direction:+d}: loss {h[-1]['loss']:.5f}, Q {h[-1]['q']:.1f} mm3/s")
        if best is None or h[-1]["loss"] < best[1][-1]["loss"]:
            best = (m, h, direction)
    m, hist, direction = best
    # Stage A: geometry frozen, fit Q and C_in (plan).
    hist += _run(m, projector, proj, thetas, times, [{"params": list(m.transport.parameters()), "lr": LR_TRANSPORT}],
                 iters_a, tag="A", cache=m.cache)
    m.cache = None  # free the footprints before Stage B
    q_after_a = float(m.transport.q().detach())
    log(f"Stage A: Q {q_after_a:.1f} mm3/s, loss {hist[-1]['loss']:.5f}")
    # Stage B: joint fine-tune (plan), tube-alignment regulariser (plan, Step 2.1).
    tg = m.tube
    groups = [
        {"params": list(m.transport.parameters()), "lr": LR_TRANSPORT / 3},
        {"params": [tg.position], "lr": LR_POSITION_MM},
        {"params": [tg.raw_sigma_r], "lr": LR_SCALE},
        {"params": [tg.quat], "lr": LR_ROTATION},
    ]
    hist += _run(m, projector, proj, thetas, times, groups, iters_b, align=LAMBDA_ALIGN, tag="B",
                 views_per_iter=VIEWS_PER_ITER)
    log(f"Stage B: Q {float(m.transport.q().detach()):.1f} mm3/s, loss {hist[-1]['loss']:.5f}")
    return m, {"history": hist, "direction": direction, "q_after_stage_a": q_after_a,
               "minutes": round((time.time() - t0) / 60, 2)}


# ============================ evaluation ============================
def evaluate(est, case, lumen, axis_perm):
    """DoD metrics of Step 2.2 against the Step 1.1 ground truth.

    Q: relative error vs Q at rest (the flow during the imaged bolus).
    Stenosis diameter: estimated radius R = sqrt(2) sigma (plan) of the
    Gaussians whose nearest GT centerline point lies in the lesion window
    |s - s0| <= L/2; throat = minimum; error = 2 |R_min - r_min,GT|.
    Ref: plan Step 2.2 DoD; matching rule OURS (as in Step 2.1).
    """
    import json

    from scipy.spatial import cKDTree

    p = json.loads(str(case["params"]))
    q_gt = p["q_rest_mls"] * 1000.0
    xyz_l = md.world_to_lumen(est["position_world_mm"], lumen, axis_perm)
    d, j = cKDTree(case["xyz_mm"]).query(xyz_l)
    s_gt = case["s"][j]
    lesion = (np.abs(s_gt - p["stenosis_center_mm"]) <= p["stenosis_len_mm"] / 2) & (d < 2.0)
    r = est["radius_mm"]
    out = {
        "q_est_mls": est["q_mm3s"] / 1000.0,
        "q_gt_mls": p["q_rest_mls"],
        "q_rel_err": abs(est["q_mm3s"] - q_gt) / q_gt,
        "lesion_found": bool(lesion.any()),
        "r_min_gt_mm": p["r_min_mm"],
        "stenosis_degree_gt": p["stenosis_degree"],
        "centerline_dist_mm": float(d.mean()),
        "radius_mae_mm": float(np.abs(r - case["r_mm"][j])[d < 2.0].mean()) if (d < 2.0).any() else float("nan"),
    }
    if lesion.any():
        r_min = float(r[lesion].min())
        out.update({"r_min_est_mm": r_min, "diam_err_mm": 2 * abs(r_min - p["r_min_mm"])})
    else:
        out.update({"r_min_est_mm": None, "diam_err_mm": None})
    return out
