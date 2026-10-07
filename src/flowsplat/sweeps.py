"""Step 3.1: robustness sweeps (plan: "views (30/60/133), noise, Peclet,
flow speed, foreshortening angle").

Runs on: M4 (settings, case / bolus edits, tests) and molab (rendering with
TIGRE, called from notebooks/step3_sweeps.py).

Each setting changes one factor of the Step 1 data and keeps the rest; the
middle level of every factor is the Step 2.2 run 8 baseline (60 views,
default noise, sampled D and rest velocity, natural orientation). Scope
(user decision, Oct 6): the views sweep on the 20 validation cases with k
recalibrated per view count on the 15 training cases; the other factors on
the 5 standard validation cases; 3 repeat 4DRGS runs on the 5 cases
(Gate 1). Levels are OURS (the plan names the factors, not the levels).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from flowsplat import bolus_sim as bs
from flowsplat import ffr
from flowsplat import render_tigre as rt

VAL5 = ["121_0", "122_0", "130_0", "131_0", "135_0"]
VAL20 = ["121_0", "121_1", "121_2", "121_3", "122_0", "122_1", "122_2", "122_3", "130_0", "130_1",
         "130_2", "131_0", "131_1", "131_2", "135_0", "135_1", "135_2", "667_0", "667_1", "667_2"]

# name -> one changed factor. kind: views (no new data), noise (re-render),
# peclet / flow (new bolus + re-render), fore (rotated geometry + re-render),
# repeat (4DRGS re-run, same data).
SETTINGS = {
    # views: plan 30 / 60 / 133; 60 = run 8
    "views30": {"kind": "views", "views": 30, "cases": "val20"},
    "views133": {"kind": "views", "views": 133, "cases": "val20"},
    # noise: default Poisson I0 1e5, Gaussian sigma 10 (render_tigre, Step 1.2)
    "noise_low": {"kind": "noise", "poisson": 1e6, "gauss": (0.0, 1.0), "cases": "val5"},
    "noise_high": {"kind": "noise", "poisson": 1e4, "gauss": (0.0, 30.0), "cases": "val5"},
    # Peclet: default D sampled in 10^-6.5 .. 10^-5.5 m^2/s (Pe about 1e4 to 1e5
    # with L = vessel length); 10x and 100x more dispersion
    "pe_d1e-5": {"kind": "peclet", "d_m2s": 1e-5, "cases": "val5"},
    "pe_d1e-4": {"kind": "peclet", "d_m2s": 1e-4, "cases": "val5"},
    # flow speed: default rest velocity sampled 0.10 to 0.20 m/s (PUNCH);
    # slow, fast rest and hyperemic-like (plan: up to about 0.4 m/s)
    "flow_v0.10": {"kind": "flow", "v_ms": 0.10, "cases": "val5"},
    "flow_v0.25": {"kind": "flow", "v_ms": 0.25, "cases": "val5"},
    "flow_v0.40": {"kind": "flow", "v_ms": 0.40, "cases": "val5"},
    # foreshortening: out-of-plane angle of the vessel chord at the reference
    # view (frame 0, the view a 2D method would use)
    "fore_0": {"kind": "fore", "deg": 0.0, "cases": "val5"},
    "fore_30": {"kind": "fore", "deg": 30.0, "cases": "val5"},
    "fore_60": {"kind": "fore", "deg": 60.0, "cases": "val5"},
    # 4DRGS run-to-run spread (Gate 1): same data, 3 re-runs
    "repeat1": {"kind": "repeat", "cases": "val5"},
    "repeat2": {"kind": "repeat", "cases": "val5"},
    "repeat3": {"kind": "repeat", "cases": "val5"},
}


def cases_of(setting):
    return VAL20 if SETTINGS[setting]["cases"] == "val20" else VAL5


def needs_new_data(setting):
    return SETTINGS[setting]["kind"] in ("noise", "peclet", "flow", "fore")


# ---------------------------------------------------------------- flow
def with_flow(case, v_ms):
    """Case arrays with the rest velocity set to v_ms (Q = v A_n, as
    data_gen.sample_case defines v_rest), transit times and the ground-truth
    FFR recomputed with the same CFR (FFR may fall below data_gen.FFR_MIN
    here; flow sweep results are judged on Q). Ref: data_gen.sample_case; OURS."""
    out = {k: case[k] for k in case.files} if hasattr(case, "files") else dict(case)
    p = json.loads(str(out["params"]))
    a_n = math.pi * (p["r_ref_at_lesion_mm"] * 1e-3) ** 2
    a_s = math.pi * (p["r_min_mm"] * 1e-3) ** 2
    q = v_ms * a_n
    s, r = out["s"], out["r_mm"]
    area = np.pi * (r * 1e-3) ** 2
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (area[1:] + area[:-1]) * np.diff(s) * 1e-3)]) / q
    dp, _, _ = ffr.young_tsai_dp(p["cfr"] * q, a_n, a_s, p["stenosis_len_mm"] * 1e-3)
    p.update({"v_rest_ms": v_ms, "q_rest_mls": q * 1e6, "q_hyp_mls": p["cfr"] * q * 1e6,
              "v_hyp_ms": p["cfr"] * v_ms, "v_in_rest_ms": float(q / area[0]),
              "transit_rest_s": float(tau[-1]), "ffr": float(ffr.ffr_from_dp(dp)),
              "sweep": {"flow_v_ms": v_ms}})
    out["tau_rest_s"] = tau
    out["params"] = json.dumps(p)
    return out


# ---------------------------------------------------------------- foreshortening
def world_perm_matrix(axis_perm):
    """Lumen -> TIGRE world direction matrix (same as model.lumen_to_world)."""
    m = np.zeros((3, 3))
    m[0, axis_perm[2]] = m[1, axis_perm[1]] = m[2, axis_perm[0]] = 1.0
    return m


def rotation_to(a, b):
    """Rotation matrix taking unit vector a to unit vector b (Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1.0 / (1.0 + c))


def out_of_plane_deg(xyz_lumen, axis_perm):
    """Angle between the vessel chord (first to last point) and the detector
    plane at frame 0, whose beam runs along world -x (model.ConeProjector,
    theta = 0, calibrated against TIGRE)."""
    d = world_perm_matrix(axis_perm) @ (xyz_lumen[-1] - xyz_lumen[0])
    d = d / np.linalg.norm(d)
    return float(np.degrees(np.arcsin(abs(d[0]))))


def with_foreshortening(case, deg, axis_perm):
    """Rotate the vessel about its centroid so that its chord makes `deg`
    with the frame-0 detector plane, keeping the in-plane direction.
    Ref: OURS (plan names the factor; 30 deg gives the 13 % 2D velocity bias
    of the plan's motivation)."""
    out = {k: case[k] for k in case.files} if hasattr(case, "files") else dict(case)
    xyz = np.asarray(out["xyz_mm"], float)
    m = world_perm_matrix(axis_perm)
    d = m @ (xyz[-1] - xyz[0])
    d = d / np.linalg.norm(d)
    inplane = np.array([0.0, d[1], d[2]])
    if np.linalg.norm(inplane) < 1e-9:
        inplane = np.array([0.0, 1.0, 0.0])
    inplane /= np.linalg.norm(inplane)
    sign = 1.0 if d[0] >= 0 else -1.0
    target = math.cos(math.radians(deg)) * inplane + sign * math.sin(math.radians(deg)) * np.array([1.0, 0.0, 0.0])
    r_world = rotation_to(d, target)
    r_lumen = m.T @ r_world @ m
    c0 = xyz.mean(axis=0)
    out["xyz_mm"] = (xyz - c0) @ r_lumen.T + c0
    p = json.loads(str(out["params"]))
    p["sweep"] = {"fore_deg": deg, "fore_deg_before": out_of_plane_deg(xyz, axis_perm)}
    out["params"] = json.dumps(p)
    return out


# ---------------------------------------------------------------- bolus
def resimulate_bolus(case, bolus, d_m2s=None):
    """New bolus for an edited case with the same inlet curve (the stored
    noisy inlet at the frame times) and gamma-variate parameters; D from the
    baseline unless given. Ref: bolus_sim.simulate_case (PUNCH Eq 2); OURS."""
    b = {k: bolus[k] for k in bolus.files} if hasattr(bolus, "files") else dict(bolus)
    bp = json.loads(str(b["params"]))
    p = json.loads(str(case["params"]))
    t = np.asarray(b["t_s"], float)
    inlet = np.asarray(b["inlet_noisy"], float)
    fn = lambda tt: np.interp(tt, t, inlet)  # noqa: E731
    d = bp["d_m2s"] if d_m2s is None else float(d_m2s)
    q = p["q_rest_mls"] * 1e-6
    s, r = case["s"], case["r_mm"]
    c = bs.simulate(s, r, q, d, fn, t)
    u_mean = q / np.mean(np.pi * (r * 1e-3) ** 2)
    bp.update({"d_m2s": d, "peclet": float(u_mean * s[-1] * 1e-3 / d), "resimulated": True})
    b["c"] = c.astype(np.float32)
    b["params"] = json.dumps(bp)
    return b


# ---------------------------------------------------------------- driver (molab)
def make_setting_data(setting, base_data, out_dir, render=True, log=print):
    """Write a pack-like folder for one setting: cases/, bolus/, geom/
    (links), splits.json, and render/ (TIGRE, molab). Returns out_dir.
    Cases already written are skipped (resumable)."""
    spec = SETTINGS[setting]
    base = Path(base_data)
    out = Path(out_dir)
    for sub in ("cases", "bolus", "render"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if not (out / "geom").exists():
        (out / "geom").symlink_to((base / "geom").resolve(), target_is_directory=True)
    (out / "splits.json").write_text((base / "splits.json").read_text())
    for c in cases_of(setting):
        g = c.split("_")[0]
        f_case, f_bol, f_ren = out / "cases" / f"{c}.npz", out / "bolus" / f"{c}.npz", out / "render" / f"{c}.npz"
        if f_ren.exists() or (not render and f_bol.exists()):
            continue
        axcodes = json.loads(str(np.load(base / "geom" / f"{g}.npz")["info"]))["axcodes"]
        case = np.load(base / "cases" / f"{c}.npz")
        bolus = np.load(base / "bolus" / f"{c}.npz")
        if spec["kind"] == "flow":
            case = with_flow(case, spec["v_ms"])
            bolus = resimulate_bolus(case, bolus)
        elif spec["kind"] == "peclet":
            case = {k: case[k] for k in case.files}
            bolus = resimulate_bolus(case, bolus, d_m2s=spec["d_m2s"])
        elif spec["kind"] == "fore":
            case = with_foreshortening(case, spec["deg"], rt.axis_permutation(axcodes))
            bolus = {k: bolus[k] for k in bolus.files}
        else:
            case = {k: case[k] for k in case.files}
            bolus = {k: bolus[k] for k in bolus.files}
        np.savez(f_case, **case)
        np.savez(f_bol, **bolus)
        if render:
            kw = {}
            if spec["kind"] == "noise":
                kw = {"noise_poisson": spec["poisson"], "noise_gauss": spec["gauss"]}
            seed = int(g) * 10 + int(c.split("_")[1])
            proj, meta = rt.render_case(f_case, f_bol, axcodes, seed=seed, **kw)
            np.savez(f_ren, proj=proj, angles_deg=bolus["angles_deg"], t_s=bolus["t_s"], meta=rt.meta_json(meta))
            log(setting, c, "rendered")
    return out


# ---------------------------------------------------------------- collection (M4 or molab)
K60 = 1.0860335028183803  # Gate 1 k, 60 views (results/step2.1_geom_init_v60_4drgs_train15.json)
BASE_RESULT = "step2.2_val20_v60_4drgs_cal_k15.json"  # run 8 = middle level of every factor


def result_name(setting):
    """Step 2.2 results file written for a setting (notebooks/step2_train.py naming)."""
    spec = SETTINGS[setting]
    if spec["kind"] == "views":
        return f"step2.2_val20_v{spec['views']}_4drgs_cal.json"
    return f"step2.2_sweep_v60_4drgs_cal_{setting}.json"


def summarize(result_json, cases=None, variant="frozen"):
    """Q, diameter and FFR summaries of one Step 2.2 results file, optionally
    restricted to a case list. Ref: OURS (reporting)."""
    from flowsplat import metrics as mt

    d = json.loads(Path(result_json).read_text())
    rows = [r for r in d["cases"].values() if r["variant"] == variant and (cases is None or r["case"] in cases)]
    dm = [r["diam_err_mm"] for r in rows if r.get("diam_err_mm") is not None]
    out = {"n": len(rows), "q": mt.q_summary([r["q_rel_err"] for r in rows]),
           "q_signed_mean": float(np.mean([r["q_est_mls"] / r["q_gt_mls"] - 1 for r in rows])),
           "diam_mm": {"mean": float(np.mean(dm)) if dm else None, "median": float(np.median(dm)) if dm else None,
                       "frac_below_0p2": float(np.mean(np.array(dm) < 0.2)) if dm else None, "n_found": len(dm)}}
    for k in ("ffr_est", "ffr_est_pop_cfr"):
        rr = [r for r in rows if k in r]
        if rr:
            out[k] = mt.ffr_summary(rr, k)
    return out


def collect(results_dir, settings=None):
    """One table of all finished settings plus the run 8 baseline (on the
    same case set). Missing settings are listed, not guessed."""
    res = Path(results_dir)
    table, missing = {}, []
    base = res / BASE_RESULT
    if base.exists():
        table["base_v60_val20"] = summarize(base)
        table["base_v60_val5"] = summarize(base, VAL5)
        # run 8 predates FFR inside the results; FFR of run 8 computed on the
        # M4 from its estimates (results/step3.1_ffr_val20_v60.json)
        f8 = res / "step3.1_ffr_val20_v60.json"
        if f8.exists():
            from flowsplat import metrics as mt

            rows = json.loads(f8.read_text())["cases"]
            for name, cs in (("base_v60_val20", None), ("base_v60_val5", VAL5)):
                rr = [r for r in rows if cs is None or r["case"] in cs]
                for k in ("ffr_est", "ffr_est_pop_cfr"):
                    table[name][k] = mt.ffr_summary(rr, k)
    for s in settings or SETTINGS:
        f = res / result_name(s)
        if f.exists():
            table[s] = {"spec": SETTINGS[s], **summarize(f)}
        else:
            missing.append(s)
    return {"table": table, "missing": missing}
