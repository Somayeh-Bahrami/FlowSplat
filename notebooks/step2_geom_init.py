# Runs on: molab GPU (RTX Pro 6000 Blackwell). Not runnable on the M4 (needs CUDA).
#
# Step 2.1: geometry initialisation and tube-aligned Gaussians.
# DoD: lumen radius error below 0.2 mm on 5 validation cases.
#
#   a) R2-Gaussian extensions + TIGRE importable? Else build (step0_setup.py
#      build-only)
#   b) data: pack_v1 (cases, geometries, bolus) and render_v1; the 5 cases
#      are rendered here if missing (fresh session)
#   c) per case: write the views as an R2-Gaussian dataset
#      (flowsplat.model.write_r2g_case), run R2-Gaussian's own
#      initialize_pcd.py and train.py at defaults (static model on the
#      dynamic views = time-averaged volume, plan Step 2.1)
#   d) centerline + half-maximum radius from the static volume
#      (flowsplat.model.build_from_volume), tube Gaussians, radius error vs
#      ground truth; results/step2.1_geom_init.json and figures
#
# Run from the repo root:  python notebooks/step2_geom_init.py
# FLOWSPLAT_VIEWS (default 60), FLOWSPLAT_CASES (default: first case of 5
# validation patients).

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

    R2G_DIR = WORK / "external" / "r2_gaussian"
    PACK = Path(os.environ.get("FLOWSPLAT_PACK", str(WORK / "data" / "step1_pack.zip")))
    DATA = WORK / "data" / "pack_v1"
    RENDER = WORK / "data" / "render_v1"
    OUT = WORK / "step2_1"
    N_VIEWS = int(os.environ.get("FLOWSPLAT_VIEWS", "60"))
    DOD_MM = 0.2  # plan, Step 2.1 DoD
    print("views:", N_VIEWS, "| out:", OUT)
    return DATA, DOD_MM, FLOWSPLAT_DIR, N_VIEWS, OUT, PACK, R2G_DIR, RENDER, md, rt


@app.cell
def _(FLOWSPLAT_DIR, R2G_DIR, os, subprocess, sys):
    # a) extensions
    def _ok():
        return subprocess.run(
            [sys.executable, "-c", "import torch, simple_knn._C, xray_gaussian_rasterization_voxelization, tigre"],
            cwd="/", capture_output=True).returncode == 0

    if not _ok() or not (R2G_DIR / "train.py").exists():
        print("Building extensions (step0_setup.py build-only).", flush=True)
        subprocess.run([sys.executable, "notebooks/step0_setup.py"], cwd=str(FLOWSPLAT_DIR),
                       env=dict(os.environ, FLOWSPLAT_BUILD_ONLY="1", FLOWSPLAT_RUN_4DRGS="0"))
    if not _ok():
        raise RuntimeError("R2-Gaussian extensions not importable.")
    ext_ready = True
    return (ext_ready,)


@app.cell
def _(DATA, PACK, RENDER, ext_ready, json, np, os, rt, zipfile):
    # b) data
    _ = ext_ready
    if not (DATA / "splits.json").exists():
        if not PACK.exists():
            raise FileNotFoundError(f"Upload step1_pack.zip to {PACK}")
        DATA.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(PACK) as _z:
            _z.extractall(DATA)
    _splits = json.loads((DATA / "splits.json").read_text())
    _env_cases = os.environ.get("FLOWSPLAT_CASES")
    CASES = _env_cases.split(",") if _env_cases else [f"{g}_0" for g in _splits["val"][:5]]
    RENDER.mkdir(parents=True, exist_ok=True)
    AXCODES = {}
    for _c in CASES:
        _g = _c.split("_")[0]
        AXCODES[_c] = json.loads(str(np.load(DATA / "geom" / f"{_g}.npz")["info"]))["axcodes"]
        _f = RENDER / f"{_c}.npz"
        if not _f.exists():  # fresh session: render this case again (same seed as Step 1.2)
            print("rendering", _c, flush=True)
            _p, _m = rt.render_case(DATA / "cases" / f"{_c}.npz", DATA / "bolus" / f"{_c}.npz", AXCODES[_c],
                                    seed=int(_g) * 10 + int(_c.split("_")[1]))
            _b = np.load(DATA / "bolus" / f"{_c}.npz")
            np.savez(_f, proj=_p, angles_deg=_b["angles_deg"], t_s=_b["t_s"], meta=rt.meta_json(_m))
    print("cases:", CASES)
    return AXCODES, CASES


@app.cell
def _(CASES, DATA, N_VIEWS, OUT, R2G_DIR, RENDER, md, np, subprocess, sys, time):
    # c) static R2-Gaussian per case (official scripts, defaults)
    runs = {}
    for _c in CASES:
        _d = OUT / f"{_c}_v{N_VIEWS}" / "data"
        _m = OUT / f"{_c}_v{N_VIEWS}" / "model"
        _info = md.write_r2g_case(RENDER / f"{_c}.npz", DATA / "cases" / f"{_c}.npz", DATA / "bolus" / f"{_c}.npz",
                                  _d, n_views=N_VIEWS)
        _vg = np.load(_d / "vol_gt.npy")
        _n_fg = int((_vg > 0.05).sum())
        _cmd = [sys.executable, "initialize_pcd.py", "--data", str(_d)]
        _dev = []
        if _n_fg < 75000:  # same rule as the Step 0.1 ImageCAS run
            _cmd += ["--n_points", str(max(5000, _n_fg // 2))]
            _dev.append(f"initialize_pcd.py --n_points {max(5000, _n_fg // 2)} ({_n_fg} GT voxels > 0.05)")
        for _old in _d.glob("init_*.npy"):
            _old.unlink()
        subprocess.run(_cmd, cwd=str(R2G_DIR), check=True)
        _t0 = time.time()
        subprocess.run([sys.executable, "train.py", "-s", str(_d), "-m", str(_m)], cwd=str(R2G_DIR), check=True)
        runs[_c] = {**_info, "model": str(_m), "train_minutes": round((time.time() - _t0) / 60, 1), "deviations": _dev}
        print(_c, runs[_c]["train_minutes"], "min", flush=True)
    return (runs,)


@app.cell
def _(CASES, DATA, DOD_MM, FLOWSPLAT_DIR, N_VIEWS, OUT, json, md, np, runs, time):
    # d) geometry from the static volume, radius error, tube Gaussians
    import matplotlib as _mpl

    _mpl.use("Agg")
    import matplotlib.pyplot as _plt
    import yaml as _yaml

    _res = FLOWSPLAT_DIR / "results"
    _res.mkdir(exist_ok=True)
    rows = []
    for _c in CASES:
        _r = runs[_c]
        _vol = np.load(f"{_r['model']}/point_cloud/iteration_30000/vol_pred.npy")
        _lv = md.r2g_volume_to_lumen_frame(_vol, _r["axis_perm"])
        _lum = _r["lumen"]
        _case = np.load(DATA / "cases" / f"{_c}.npz")
        s_e, xyz_e, r_e = md.build_from_volume(_lv, _lum["vox_mm"], _lum["origin_mm"])
        s_e, xyz_e, r_e = md.orient_by_reference(s_e, xyz_e, r_e, _case["xyz_mm"][0])
        _err = md.radius_error(xyz_e, r_e, _case["s"], _case["xyz_mm"], _case["r_mm"])
        _tg = md.TubeGaussians(s_e, xyz_e, r_e)
        _ev = {}
        for _f in sorted((OUT / f"{_c}_v{N_VIEWS}" / "model" / "eval" / "iter_030000").glob("eval*.yml")):
            _ev[_f.stem] = {k: v for k, v in _yaml.safe_load(_f.read_text()).items() if isinstance(v, (int, float))}
        _p = json.loads(str(_case["params"]))
        np.savez(OUT / f"{_c}_v{N_VIEWS}" / "geom_init.npz", s=s_e, xyz_mm=xyz_e, r_mm=r_e)
        row = {"case": _c, "views": N_VIEWS, **_err, "alignment_loss_init": float(_tg.alignment_loss().item()),
               "n_gaussians": int(len(s_e)), "length_est_mm": float(s_e[-1]), "length_gt_mm": float(_case["s"][-1]),
               "stenosis_degree_gt": _p["stenosis_degree"], "r_min_gt_mm": _p["r_min_mm"],
               "r2g_eval": _ev, "train_minutes": _r["train_minutes"], "deviations": _r["deviations"],
               "pass": _err["mae_mm"] < DOD_MM}
        rows.append(row)
        _fig, _ax = _plt.subplots(1, 2, figsize=(14, 4))
        _ax[0].imshow(_lv.max(axis=2).T, origin="lower", cmap="gray")
        _ax[0].set_title(f"{_c}: static R2-Gaussian, MIP")
        _ax[1].plot(_case["s"], _case["r_mm"], label="GT")
        _ax[1].plot(s_e, r_e, label=f"estimate (MAE {_err['mae_mm']:.3f} mm)")
        _ax[1].set_xlabel("s (mm)")
        _ax[1].set_ylabel("R (mm)")
        _ax[1].legend()
        _fig.tight_layout()
        _fig.savefig(_res / f"step2.1_{_c}_v{N_VIEWS}.png", dpi=80)
        _plt.close(_fig)
        print(json.dumps({k: v for k, v in row.items() if k != "r2g_eval"}), flush=True)
    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "views": N_VIEWS,
        "dod_mm": DOD_MM,
        "mae_mm_per_case": {r["case"]: round(r["mae_mm"], 4) for r in rows},
        "mae_mm_mean": float(np.mean([r["mae_mm"] for r in rows])),
        "all_pass": all(r["pass"] for r in rows),
        "rows": rows,
    }
    (_res / f"step2.1_geom_init_v{N_VIEWS}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return


if __name__ == "__main__":
    app.run()
