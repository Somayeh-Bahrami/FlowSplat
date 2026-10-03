"""Step 1.2 (part 1): contrast bolus along each LAD, 1D advection-dispersion.

Runs on: M4 (CPU only, numpy).

The simulator is deliberately richer than the plug-flow model FlowSplat
fits (plan, Dataset: "to avoid an inverse crime"): it keeps the area term
and the dispersion of PUNCH Eq. 2, a noisy inlet, and a fine grid.

Usage:
  python -m flowsplat.bolus_sim --cases data/cases_v1 --out data/bolus_v1 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ---- Acquisition timing (plan + 4DRGS) ----
# 4DRGS, Sec 4.1: "133 mask-fill X-ray image pairs, evenly distributed across
# a rotational range of 198 degrees".
N_FRAMES = 133
ARC_DEG = 198.0
# Plan, Dataset: "198 deg over about 4.4 s" (4DRGS gives no duration: UNVERIFIED).
SWEEP_S = 4.4
# ---- Bolus parameters ----
# PUNCH, Sec "Materials and Methods" (Pe paragraph): D ~ 1e-6 m^2/s; synthetic
# data: D "sampled over an order-of-magnitude physiological range" (range not
# given). OURS: log-uniform over one decade centred on 1e-6.
D_RANGE = (10**-6.5, 10**-5.5)
# OURS: gamma-variate inlet (plan: "gamma-variate inlet C_in(t)"), timed so
# wash-in and wash-out both fall inside the 4.4 s sweep.
T0_RANGE = (0.2, 1.0)  # s, arrival at the ostium after the sweep starts
ALPHA_RANGE = (2.5, 4.5)
BETA_RANGE = (0.25, 0.5)  # s
# OURS: multiplicative inlet noise (mixing / injection irregularity), 5 % rms,
# smoothed over 0.1 s.
INLET_NOISE = 0.05
INLET_NOISE_TAU_S = 0.1
# Numerics (OURS): finite-volume cell size and CFL number. With the superbee
# limiter at 0.1 mm the numerical 10-90 % front width after 50 mm is 0.29 mm
# (van Leer: 1.17 mm), a quarter of the smallest physical one (1.2 mm at
# D = 10^-6.5); measured Oct 3, see tests/test_bolus_sim.py.
DS_SIM_M = 0.1e-3
CFL = 0.4


def frame_times(n=N_FRAMES, sweep_s=SWEEP_S):
    """Acquisition time of each frame, s. Ref: 4DRGS Sec 4.1 (even spacing), plan (4.4 s)."""
    return np.linspace(0.0, sweep_s, n)


def frame_angles_deg(n=N_FRAMES, arc=ARC_DEG):
    """Gantry angle of each frame. Ref: 4DRGS Sec 4.1 (133 frames evenly over 198 deg)."""
    return np.linspace(0.0, arc, n)


def gamma_variate(t, t0, alpha, beta):
    """Gamma-variate bolus, peak normalised to 1.

    c(t) = ((t - t0) / (alpha beta))^alpha exp(alpha - (t - t0) / beta) for t > t0.
    Peak at t0 + alpha beta.

    Ref: plan, Approach ("gamma-variate inlet curve C_in(t)"). Classic
    indicator-dilution form (Thompson et al. 1964; Madsen 1992): citation
    and normalisation UNVERIFIED.
    """
    x = np.clip(np.asarray(t, dtype=float) - t0, 0.0, None) / (alpha * beta)
    with np.errstate(divide="ignore"):
        out = np.where(x > 0, np.exp(alpha * np.log(np.where(x > 0, x, 1.0)) + alpha * (1.0 - x)), 0.0)
    return out


def smooth_noise(t, rms, tau_s, rng):
    """Band-limited multiplicative noise factor 1 + n(t), n with the given rms. Ref: OURS."""
    from scipy.ndimage import gaussian_filter1d

    dt = float(t[1] - t[0])
    n = gaussian_filter1d(rng.standard_normal(len(t)), tau_s / dt, mode="wrap")
    n *= rms / (n.std() + 1e-12)
    return 1.0 + n


def _limiter(r):
    """Superbee flux limiter, max(0, min(2r, 1), min(r, 2)).

    Ref: Roe 1985 (superbee), standard TVD limiter; citation details
    UNVERIFIED. Chosen over van Leer for 4x less numerical diffusion.
    """
    return np.maximum(0.0, np.maximum(np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)))


def simulate(s_mm, r_mm, q_m3s, d_m2s, inlet_fn, t_out, ds_m=DS_SIM_M, cfl=CFL):
    """Solve PUNCH Eq. 2 in conservative form on a fine 1D grid.

    d(A c)/dt + d(Q c)/ds = d/ds (A D dc/ds),   Q = u A constant (steady flow),

    which equals PUNCH Eq. 2 (u dc/ds = (1/A) d(AD)/ds dc/ds + D d2c/ds2 on
    the right after dividing by A, with D constant along the vessel).
    Advection: upwind MUSCL reconstruction with the superbee limiter;
    dispersion: central differences; time: SSP-RK2 (Heun). Inlet: Dirichlet
    c = inlet_fn(t) in a ghost cell; outlet: zero gradient.

    Ref: PUNCH, Materials and Methods, Eq. 2 (model); numerics OURS (standard
    TVD finite-volume scheme, LeVeque 2002, chapter UNVERIFIED).
    Returns c at the input stations s_mm (shape len(s_mm) x len(t_out)).
    """
    s_mm = np.asarray(s_mm, dtype=float)
    length_m = s_mm[-1] * 1e-3
    n = max(int(np.ceil(length_m / ds_m)), 10)
    ds = length_m / n
    xc = (np.arange(n) + 0.5) * ds  # cell centres, m
    xf = np.arange(n + 1) * ds  # faces
    a_c = np.pi * (np.interp(xc * 1e3, s_mm, r_mm) * 1e-3) ** 2
    a_f = np.pi * (np.interp(xf * 1e3, s_mm, r_mm) * 1e-3) ** 2
    q = float(q_m3s)
    d = float(d_m2s)
    u_max = q / a_f.min()
    dt = cfl * min(ds / u_max, ds**2 / (2.0 * d + 1e-30))
    t_out = np.asarray(t_out, dtype=float)
    t_end = float(t_out[-1])
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps

    def rhs(c, t):
        cin = float(inlet_fn(t))
        # padded: two ghost cells upstream (inlet), one downstream (outlet)
        cp = np.concatenate([[cin, cin], c, [c[-1]]])
        dl = cp[1:-1] - cp[:-2]  # backward differences at cells -1..n-1
        dr = cp[2:] - cp[1:-1]  # forward differences
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(dr) > 1e-14, dl / dr, 0.0)
        phi = _limiter(ratio)
        c_face = cp[1:-1] + 0.5 * phi * dr  # left state at faces 0..n (upwind, Q > 0)
        c_face[0] = cin
        f_adv = q * c_face  # length n + 1
        grad = np.diff(np.concatenate([[cin], c, [c[-1]]])) / ds  # at faces 0..n
        f_dif = -a_f * d * grad
        f = f_adv + f_dif
        return -(f[1:] - f[:-1]) / (ds * a_c)

    c = np.zeros(n)
    out = np.zeros((n, len(t_out)))
    k_out = 0
    t = 0.0
    while k_out < len(t_out) and t_out[k_out] <= 0.0:
        out[:, k_out] = c
        k_out += 1
    for step in range(n_steps):
        k1 = rhs(c, t)
        c1 = c + dt * k1
        k2 = rhs(c1, t + dt)
        c = 0.5 * (c + c1 + dt * k2)
        t = (step + 1) * dt
        while k_out < len(t_out) and t_out[k_out] <= t + 1e-12:
            out[:, k_out] = c
            k_out += 1
    # back to the centerline stations
    return np.stack([np.interp(s_mm * 1e-3, xc, out[:, j]) for j in range(len(t_out))], axis=1)


def sample_bolus(rng):
    """Draw inlet and dispersion parameters for one case. Ref: see constants above."""
    return {
        "t0_s": float(rng.uniform(*T0_RANGE)),
        "alpha": float(rng.uniform(*ALPHA_RANGE)),
        "beta_s": float(rng.uniform(*BETA_RANGE)),
        "d_m2s": float(10 ** rng.uniform(np.log10(D_RANGE[0]), np.log10(D_RANGE[1]))),
    }


def simulate_case(case_npz, rng, t_frames=None):
    """Bolus for one Step 1.1 case at rest flow. Returns (c_frames, params, inlet_clean, inlet_noisy)."""
    z = np.load(case_npz)
    p = json.loads(str(z["params"]))
    s, r = z["s"], z["r_mm"]
    t_frames = frame_times() if t_frames is None else t_frames
    b = sample_bolus(rng)
    t_fine = np.linspace(0.0, t_frames[-1], 4401)
    clean = gamma_variate(t_fine, b["t0_s"], b["alpha"], b["beta_s"])
    noisy = clean * smooth_noise(t_fine, INLET_NOISE, INLET_NOISE_TAU_S, rng)
    inlet = lambda t: np.interp(t, t_fine, noisy)  # noqa: E731
    q = p["q_rest_mls"] * 1e-6
    c = simulate(s, r, q, b["d_m2s"], inlet, t_frames)
    u_mean = q / np.mean(np.pi * (r * 1e-3) ** 2)
    b["peclet"] = float(u_mean * s[-1] * 1e-3 / b["d_m2s"])  # PUNCH Pe = u L / D
    return c.astype(np.float32), b, np.interp(t_frames, t_fine, clean), np.interp(t_frames, t_fine, noisy)


def make_splits(geom_ids, seed=0, frac=(0.7, 0.15, 0.15)):
    """Patient-level train / val / test split of geometry ids (all cases of a
    patient stay in one split). Ref: OURS (same principle as the owner's
    earlier make_splits: never split one patient across sets)."""
    ids = sorted(geom_ids, key=int)
    rng = np.random.default_rng(seed)
    perm = [ids[i] for i in rng.permutation(len(ids))]
    n_tr = int(round(frac[0] * len(ids)))
    n_va = int(round(frac[1] * len(ids)))
    return {"train": sorted(perm[:n_tr], key=int), "val": sorted(perm[n_tr : n_tr + n_va], key=int),
            "test": sorted(perm[n_tr + n_va :], key=int)}


def kymograph_png(path, s, t, c, title):
    """Concentration c(s, t) image for the visual spot check. Ref: OURS (PUNCH uses kymographs, Fig. 1)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(c, aspect="auto", origin="lower", extent=[t[0], t[-1], s[0], s[-1]], cmap="magma", vmin=0, vmax=1)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("s (mm)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def run_all(cases_dir, out_dir, seed=0, n_spot=5):
    cases_dir, out_dir = Path(cases_dir), Path(out_dir)
    (out_dir / "bolus").mkdir(parents=True, exist_ok=True)
    (out_dir / "kymo").mkdir(parents=True, exist_ok=True)
    files = sorted((cases_dir / "cases").glob("*.npz"), key=lambda p: tuple(int(x) for x in p.stem.split("_")))
    t = frame_times()
    rows, t0 = [], time.time()
    for i, f in enumerate(files):
        g, k = (int(x) for x in f.stem.split("_"))
        rng = np.random.default_rng([seed, g, k, 12])  # 12: stream tag for Step 1.2
        c, b, clean, noisy = simulate_case(f, rng, t)
        z = np.load(f)
        np.savez(out_dir / "bolus" / f"{f.stem}.npz", s=z["s"], c=c.astype(np.float16), t_s=t,
                 angles_deg=frame_angles_deg(), inlet_clean=clean, inlet_noisy=noisy, params=json.dumps(b))
        rows.append({"case": f.stem, **b, "c_max": float(c.max()), "frac_frames_with_contrast": float((c.max(0) > 0.05).mean())})
        if i < n_spot or f.stem in ():
            kymograph_png(out_dir / "kymo" / f"{f.stem}.png", z["s"], t, c, f"case {f.stem}, Q_rest {json.loads(str(z['params']))['q_rest_mls']:.2f} mL/s")
        print(f"[{i + 1}/{len(files)}] {f.stem}: Pe {b['peclet']:.0f}, c_max {c.max():.2f}", flush=True)
    geoms = sorted({r["case"].split("_")[0] for r in rows}, key=int)
    splits = make_splits(geoms, seed)
    splits["cases"] = {k: [r["case"] for r in rows if r["case"].split("_")[0] in v] for k, v in splits.items()}
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    summary = {"seed": seed, "n_cases": len(rows), "seconds": round(time.time() - t0, 1),
               "frames": N_FRAMES, "arc_deg": ARC_DEG, "sweep_s": SWEEP_S,
               "peclet": {"min": min(r["peclet"] for r in rows), "max": max(r["peclet"] for r in rows)},
               "c_max_min": min(r["c_max"] for r in rows),
               "split_sizes": {k: len(v) for k, v in splits["cases"].items()}, "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def make_pack(cases_dir, bolus_dir, zip_path):
    """Zip what the molab renderer needs: Step 1.1 cases + geometries (for the
    NIfTI axis codes), Step 1.2 bolus files, splits. Ref: OURS (tooling)."""
    import zipfile

    cases_dir, bolus_dir = Path(cases_dir), Path(bolus_dir)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for sub, root in (("cases", cases_dir), ("geom", cases_dir), ("bolus", bolus_dir)):
            for f in sorted((root / sub).glob("*.npz")):
                z.write(f, f"{sub}/{f.name}")
        z.write(bolus_dir / "splits.json", "splits.json")
        z.write(cases_dir / "summary.json", "cases_summary.json")
    return zip_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", default="data/cases_v1")
    ap.add_argument("--out", default="data/bolus_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pack", default=None, help="also write this zip for the molab renderer")
    a = ap.parse_args(argv)
    s = run_all(a.cases, a.out, a.seed)
    if a.pack:
        print("pack:", make_pack(a.cases, a.out, a.pack))
    print(json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
