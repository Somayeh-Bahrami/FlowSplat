# Runs on: molab GPU (RTX Pro 6000 Blackwell). Not runnable on the M4 at full resolution.
#
# Step 2.2: transport layer and joint optimization.
# DoD: on 20 validation cases, Q relative error below 10 % and stenosis
# diameter error below 0.2 mm (30 or 60 views).
#
#   a) TIGRE importable? Else build (step0_setup.py build-only); needed for
#      the FDK geometry init and to re-render missing cases
#   b) data: step1_pack.zip -> pack_v1; render_v1 (missing cases rendered)
#   c) per case: FDK geometry init (Step 2.1, final method), then
#      flowsplat.train.fit_case: direction test, Stage A (geometry frozen, fit
#      Q and C_in), Stage B (joint, L1 + D-SSIM, alignment regulariser);
#      evaluation against the Step 1.1 ground truth
#   d) results/step2.2_<mode>_v<views>.json, per-case figures
#
# Run from the repo root:  python notebooks/step2_train.py
# FLOWSPLAT_MODE = overfit (one case, default) | val20 (20 validation cases)
# FLOWSPLAT_VIEWS = 60 (default) | 30;  FLOWSPLAT_CASES = comma list (overrides)
# FLOWSPLAT_VARIANTS = comma list of base, frozen, pose, pose_arr (default: all;
#   the Step 2.2 fallback variants, see train.VARIANTS)

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import subprocess
    import sys
    import time
    import zipfile
    from pathlib import Path

    import marimo as mo
    import numpy as np

    return Path, json, mo, np, os, subprocess, sys, time, zipfile


@app.cell
def _(Path, mo, os, sys):
    WORK = Path(os.environ.get("FLOWSPLAT_WORK", str(Path.home() / "flowsplat_work")))
    try:
        FLOWSPLAT_DIR = Path(mo.notebook_dir()).resolve().parent
    except Exception:
        FLOWSPLAT_DIR = Path.cwd().resolve()
    if not (FLOWSPLAT_DIR / "src" / "flowsplat").exists():
        FLOWSPLAT_DIR = Path.cwd().resolve()
    sys.path.insert(0, str(FLOWSPLAT_DIR / "src"))
    from flowsplat import model as md
    from flowsplat import render_tigre as rt
    from flowsplat import train as tr

    PACK = Path(os.environ.get("FLOWSPLAT_PACK", str(WORK / "data" / "step1_pack.zip")))
    DATA = WORK / "data" / "pack_v1"
    RENDER = WORK / "data" / "render_v1"
    OUT = WORK / "step2_2"
    MODE = os.environ.get("FLOWSPLAT_MODE", "overfit")
    # Fallback variants of the plan (train.VARIANTS); overfit runs all four.
    VARIANTS = tuple(os.environ.get("FLOWSPLAT_VARIANTS", ",".join(tr.VARIANTS)).split(","))
    N_VIEWS = int(os.environ.get("FLOWSPLAT_VIEWS", "60"))
    DOD_Q, DOD_DIAM_MM = 0.10, 0.2  # plan, Step 2.2 DoD
    print("mode:", MODE, "| views:", N_VIEWS, "| variants:", VARIANTS)
    return DATA, DOD_DIAM_MM, DOD_Q, FLOWSPLAT_DIR, MODE, N_VIEWS, OUT, PACK, RENDER, VARIANTS, md, rt, tr


@app.cell
def _(FLOWSPLAT_DIR, os, subprocess, sys):
    # a) TIGRE (FDK init, re-rendering)
    def _ok():
        return subprocess.run([sys.executable, "-c", "import tigre"], cwd="/", capture_output=True).returncode == 0

    if not _ok():
        print("Building TIGRE (step0_setup.py build-only).", flush=True)
        subprocess.run([sys.executable, "notebooks/step0_setup.py"], cwd=str(FLOWSPLAT_DIR),
                       env=dict(os.environ, FLOWSPLAT_BUILD_ONLY="1", FLOWSPLAT_RUN_4DRGS="0"))
    if not _ok():
        raise RuntimeError("TIGRE not importable.")
    tigre_ready = True
    return (tigre_ready,)


@app.cell
def _(DATA, MODE, PACK, RENDER, json, np, os, rt, tigre_ready, zipfile):
    # b) data and the case list
    _ = tigre_ready
    if not (DATA / "splits.json").exists():
        if not PACK.exists():
            raise FileNotFoundError(f"Upload step1_pack.zip to {PACK}")
        DATA.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(PACK) as _z:
            _z.extractall(DATA)
    _val = json.loads((DATA / "splits.json").read_text())["cases"]["val"]
    # 20 validation cases: ordered by stenosis index, then patient, so all 6
    # validation patients are included (OURS, fixed rule).
    _ordered = sorted(_val, key=lambda c: (int(c.split("_")[1]), int(c.split("_")[0])))
    if os.environ.get("FLOWSPLAT_CASES"):
        CASES = os.environ["FLOWSPLAT_CASES"].split(",")
    elif MODE == "overfit":
        CASES = ["121_0"]
    else:
        CASES = _ordered[:20]
    RENDER.mkdir(parents=True, exist_ok=True)
    for _c in CASES:
        if not (RENDER / f"{_c}.npz").exists():
            _g = _c.split("_")[0]
            _ax = json.loads(str(np.load(DATA / "geom" / f"{_g}.npz")["info"]))["axcodes"]
            print("rendering", _c, flush=True)
            _p, _m = rt.render_case(DATA / "cases" / f"{_c}.npz", DATA / "bolus" / f"{_c}.npz", _ax,
                                    seed=int(_g) * 10 + int(_c.split("_")[1]))
            _b = np.load(DATA / "bolus" / f"{_c}.npz")
            np.savez(RENDER / f"{_c}.npz", proj=_p, angles_deg=_b["angles_deg"], t_s=_b["t_s"], meta=rt.meta_json(_m))
    print(len(CASES), "cases:", CASES)
    return (CASES,)


@app.cell
def _(CASES, DATA, N_VIEWS, OUT, RENDER, VARIANTS, json, md, np, rt, time, tr):
    # c) fit every case with every requested variant (same FDK init per case)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for _c in CASES:
        _z = np.load(RENDER / f"{_c}.npz")
        _meta = json.loads(str(_z["meta"]))
        _case = np.load(DATA / "cases" / f"{_c}.npz")
        _idx = rt.view_subsets(len(_z["angles_deg"]), (N_VIEWS,))[str(N_VIEWS)]["train"]
        # Step 2.1 final init: FDK of the same training views, half-max radius
        _vol, _ = md.fdk_volume_lumen_frame(RENDER / f"{_c}.npz", n_views=N_VIEWS)
        _s, _xyz_l, _r = md.build_from_volume(_vol, _meta["lumen"]["vox_mm"], _meta["lumen"]["origin_mm"])
        _world, _ = md.lumen_to_world(_xyz_l, _meta["lumen"], _meta["axis_perm"])
        _proj = torch.tensor(_z["proj"][_idx].astype(np.float32), device=device)
        _pj = md.ConeProjector(_meta["geometry"], device=device)
        _thetas = list(np.deg2rad(_z["angles_deg"][_idx]))
        for _v in VARIANTS:
            _t0 = time.time()
            torch.manual_seed(0)
            _m, _info = tr.fit_case(_proj, _thetas, _z["t_s"][_idx], _s, _world, _r, _pj, device=device,
                                    variant=_v, log=lambda *a: print(_c, _v, *a, flush=True))
            _est = _m.estimates()
            _ev = tr.evaluate(_est, _case, _meta["lumen"], _meta["axis_perm"])
            _ev["q_rel_err_stage_a"] = abs(_info["q_after_stage_a"] / 1000 - _ev["q_gt_mls"]) / _ev["q_gt_mls"]
            _d = OUT / f"{_c}_v{N_VIEWS}_{_v}"
            _d.mkdir(parents=True, exist_ok=True)
            np.savez(_d / "estimate.npz", radius_mm=_est["radius_mm"], position_world_mm=_est["position_world_mm"],
                     init_s=_s, init_xyz_lumen=_xyz_l, init_r=_r)
            (_d / "history.json").write_text(json.dumps(_info["history"]))
            results[f"{_c}|{_v}"] = {**_ev, "case": _c, "variant": _v, "direction": _info["direction"],
                                     "minutes": round((time.time() - _t0) / 60, 2),
                                     **{k: v for k, v in _est.items() if k not in ("radius_mm", "position_world_mm")}}
            print(_c, _v, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in results[f"{_c}|{_v}"].items()}), flush=True)
    return (results,)


@app.cell
def _(DATA, DOD_DIAM_MM, DOD_Q, FLOWSPLAT_DIR, MODE, N_VIEWS, OUT, RENDER, VARIANTS, json, md, np, results, time):
    # d) summary per variant and figures per case (all variants in one plot)
    import matplotlib as _mpl

    _mpl.use("Agg")
    import matplotlib.pyplot as _plt
    from scipy.spatial import cKDTree as _kd

    _res = FLOWSPLAT_DIR / "results"
    _res.mkdir(exist_ok=True)
    per_variant = {}
    for _v in VARIANTS:
        _rows = [r for r in results.values() if r["variant"] == _v]
        _q = np.array([r["q_rel_err"] for r in _rows])
        _qa = np.array([r["q_rel_err_stage_a"] for r in _rows])
        _dm = np.array([r["diam_err_mm"] for r in _rows if r["diam_err_mm"] is not None])
        per_variant[_v] = {
            "n_cases": len(_rows),
            "q_rel_err": {"mean": float(_q.mean()), "median": float(np.median(_q)), "max": float(_q.max()),
                          "frac_below_10pct": float((_q < DOD_Q).mean())},
            "q_rel_err_stage_a_only": {"mean": float(_qa.mean()), "median": float(np.median(_qa))},
            "diam_err_mm": {"mean": float(_dm.mean()) if len(_dm) else None,
                            "median": float(np.median(_dm)) if len(_dm) else None,
                            "frac_below_0p2": float((_dm < DOD_DIAM_MM).mean()) if len(_dm) else None,
                            "n_lesion_found": int(len(_dm))},
            "radius_mae_mm_mean": float(np.nanmean([r["radius_mae_mm"] for r in _rows])),
            "dod_met": bool(_q.mean() < DOD_Q and len(_dm) == len(_rows) and _dm.mean() < DOD_DIAM_MM),
        }
    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": MODE, "views": N_VIEWS, "variants": list(VARIANTS),
        "dod_rule": "mean Q relative error < 10 % and mean stenosis diameter error < 0.2 mm over the cases (per-case fractions reported too)",
        "per_variant": per_variant,
        "cases": results,
    }
    (_res / f"step2.2_{MODE}_v{N_VIEWS}.json").write_text(json.dumps(summary, indent=2, default=str))
    for _c in sorted({r["case"] for r in results.values()}):
        _meta = json.loads(str(np.load(RENDER / f"{_c}.npz")["meta"]))
        _case = np.load(DATA / "cases" / f"{_c}.npz")
        _p = json.loads(str(_case["params"]))
        _tree = _kd(_case["xyz_mm"])
        _fig, _ax = _plt.subplots(figsize=(10, 4))
        _ax.plot(_case["s"], _case["r_mm"], color="k", lw=1.5, label="ground truth")
        _first = True
        for _v in VARIANTS:
            _e = np.load(OUT / f"{_c}_v{N_VIEWS}_{_v}" / "estimate.npz")
            if _first:
                _ax.plot(_case["s"][_tree.query(_e["init_xyz_lumen"])[1]], _e["init_r"], ".", ms=2, color="0.6", label="init (FDK)")
                _first = False
            _xyz = md.world_to_lumen(_e["position_world_mm"], _meta["lumen"], _meta["axis_perm"])
            _r = results[f"{_c}|{_v}"]
            _ax.plot(_case["s"][_tree.query(_xyz)[1]], _e["radius_mm"], ".", ms=2,
                     label=f"{_v}: Q {_r['q_rel_err']:.1%}, diam {_r['diam_err_mm'] if _r['diam_err_mm'] is None else round(_r['diam_err_mm'], 3)} mm")
        _ax.axvspan(_p["stenosis_center_mm"] - _p["stenosis_len_mm"] / 2, _p["stenosis_center_mm"] + _p["stenosis_len_mm"] / 2, color="0.92")
        _ax.set_xlabel("GT s (mm)")
        _ax.set_ylabel("R (mm)")
        _ax.set_title(_c)
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _fig.savefig(_res / f"step2.2_{_c}_v{N_VIEWS}.png", dpi=80)
        _plt.close(_fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))
    return


if __name__ == "__main__":
    app.run()
