# Runs on: molab GPU (RTX Pro 6000 Blackwell). Not runnable on the M4 (needs CUDA).
#
# Step 2.1: geometry initialisation and tube-aligned Gaussians.
# DoD: lumen radius error below 0.2 mm on 5 validation cases.
#
#   a) R2-Gaussian extensions + TIGRE importable? Else build (step0_setup.py
#      build-only)
#   b) data: pack_v1 (cases, geometries, bolus) and render_v1; the 5 cases
#      are rendered here if missing (fresh session)
#   c) per case (FLOWSPLAT_INIT=fdk, default since Oct 4: plan FALLBACK, TIGRE
#      FDK volume of the same views; FLOWSPLAT_INIT=r2g: the original route):
#      write the views as an R2-Gaussian dataset
#      (flowsplat.model.write_r2g_case), run R2-Gaussian's own
#      initialize_pcd.py and train.py at defaults (static model on the
#      dynamic views = time-averaged volume, plan Step 2.1)
#      FLOWSPLAT_INIT=4drgs (Step 2.1 v2, plan deviation approved Oct 5):
#      write the case as a 4DRGS dataset (flowsplat.model.write_4drgs_case),
#      run the official 4DRGS train.py (README fast setting, 10k / ADC 5k, as
#      reproduced in Step 0.1), take its time-averaged voxel query volume
#      (test.py VQR, written before its mesh / ICP evaluation, which needs a
#      reference mesh we do not have) and its held-out view PSNR (test.py
#      --render_2d)
#   d) centerline + half-maximum radius from the static volume
#      (flowsplat.model.build_from_volume), tube Gaussians, radius error vs
#      ground truth; results/step2.1_geom_init.json and figures.
#      v2 pass rule (fixed before the run): radius MAE < 0.2 mm AND vessel
#      coverage >= 90 % AND lesion coverage >= 90 % (a missed lesion fails).
#
# Run from the repo root:  python notebooks/step2_geom_init.py
# FLOWSPLAT_VIEWS (default 60), FLOWSPLAT_CASES (default: first case of 5
# validation patients), FLOWSPLAT_4DRGS_ITERS (default 10000).
# FLOWSPLAT_SPLIT=train (Oct 6, option 1 approved by the user): default cases
#   = first case of the first 5 TRAINING patients; result files get the
#   suffix _train5 and the summary holds radius_scale_calib
#   (md.radius_scale), applied unchanged in Step 2.2 (FLOWSPLAT_R_SCALE).
# FLOWSPLAT_N_CASES (default 5): number of patients (first case each); Gate 1
#   (Oct 6): 15 training patients, k from the median (radius_scale_calib_median).

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
    DRGS_DIR = WORK / "external" / "4DRGS"
    PACK = Path(os.environ.get("FLOWSPLAT_PACK", str(WORK / "data" / "step1_pack.zip")))
    DATA = WORK / "data" / "pack_v1"
    RENDER = WORK / "data" / "render_v1"
    OUT = WORK / "step2_1"
    N_VIEWS = int(os.environ.get("FLOWSPLAT_VIEWS", "60"))
    # Plan Step 2.1: static R2-Gaussian ("r2g"). Its DoD failed on Oct 4
    # (radius MAE 0.20 to 0.42 mm), so the plan's fallback "fdk" (FDK volume
    # skeleton) is the default.
    # Oct 5 (Step 2.1 v2, approved plan deviation): "4drgs".
    INIT = os.environ.get("FLOWSPLAT_INIT", "fdk")
    DOD_MM = 0.2  # plan, Step 2.1 DoD
    COV_MIN = 0.9  # v2 pass rule: vessel and lesion coverage (OURS, fixed Oct 5 before the run)
    DRGS_ITERS = int(os.environ.get("FLOWSPLAT_4DRGS_ITERS", "10000"))
    SPLIT = os.environ.get("FLOWSPLAT_SPLIT", "val")
    N_CASES = int(os.environ.get("FLOWSPLAT_N_CASES", "5"))
    RTAG = f"_train{N_CASES}" if SPLIT == "train" else ""
    print("views:", N_VIEWS, "| init:", INIT, "| split:", SPLIT, "| out:", OUT)
    return (COV_MIN, DATA, DOD_MM, DRGS_DIR, DRGS_ITERS, FLOWSPLAT_DIR, INIT, N_VIEWS, OUT, PACK, R2G_DIR,
            N_CASES, RENDER, RTAG, SPLIT, md, rt)


@app.cell
def _(DRGS_DIR, FLOWSPLAT_DIR, INIT, R2G_DIR, os, subprocess, sys):
    # a) extensions (4DRGS route: also tiny-cuda-nn, the 4DRGS rasterizers,
    #    LEAP and the Python packages its train.py / test.py import)
    _mods = "import torch, simple_knn._C, xray_gaussian_rasterization_voxelization, tigre"
    if INIT == "4drgs":
        _mods += (", tinycudann, diff_Xray_gaussian_rasterization_voxelization,"
                  " diff_Xray_gaussian_rasterization_voxelization_legacy, leapctype, pyhocon, SimpleITK, mcubes, pymeshlab, open3d")

    def _ok():
        return subprocess.run([sys.executable, "-c", _mods], cwd="/", capture_output=True).returncode == 0

    _repo = DRGS_DIR if INIT == "4drgs" else R2G_DIR
    if not _ok() or not (_repo / "train.py").exists():
        print("Building extensions (step0_setup.py build-only).", flush=True)
        _flags = ({"FLOWSPLAT_4DRGS_BUILD_ONLY": "1", "FLOWSPLAT_RUN_4DRGS": "1"} if INIT == "4drgs"
                  else {"FLOWSPLAT_BUILD_ONLY": "1", "FLOWSPLAT_RUN_4DRGS": "0"})
        subprocess.run([sys.executable, "notebooks/step0_setup.py"], cwd=str(FLOWSPLAT_DIR),
                       env=dict(os.environ, **_flags))
    if not _ok():
        _err = subprocess.run([sys.executable, "-c", _mods], cwd="/", capture_output=True, text=True).stderr
        raise RuntimeError(f"Extensions not importable:\n{_err[-2000:]}")
    ext_ready = True
    return (ext_ready,)


@app.cell
def _(DATA, N_CASES, PACK, RENDER, SPLIT, ext_ready, json, np, os, rt, zipfile):
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
    CASES = _env_cases.split(",") if _env_cases else [f"{g}_0" for g in _splits[SPLIT][:N_CASES]]
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
def _(CASES, DATA, DRGS_DIR, DRGS_ITERS, INIT, N_VIEWS, OUT, R2G_DIR, RENDER, json, md, np, subprocess, sys, time):
    # c) init volume per case, saved in the lumen frame:
    #    INIT=r2g: static R2-Gaussian (official scripts, defaults), plan Step 2.1
    #    INIT=fdk: TIGRE FDK (Hann) of the same views, plan Step 2.1 FALLBACK
    #    INIT=4drgs: official 4DRGS, time-averaged VQR volume (Step 2.1 v2)
    from flowsplat import setup_check as _sc

    runs = {}
    for _c in CASES:
        _base = OUT / f"{_c}_v{N_VIEWS}_{INIT}"
        _base.mkdir(parents=True, exist_ok=True)
        _meta = json.loads(str(np.load(RENDER / f"{_c}.npz")["meta"]))
        _lum = _meta["lumen"]
        _dev = []
        _t0 = time.time()
        _ev_dir = None
        _drgs = None
        if INIT == "fdk":
            _lv, _ = md.fdk_volume_lumen_frame(RENDER / f"{_c}.npz", n_views=N_VIEWS)
        elif INIT == "4drgs":
            import SimpleITK as _sitk

            _d, _m = _base / "data", _base / "model"
            md.write_4drgs_case(RENDER / f"{_c}.npz", _d)
            # number of FDK-masked init points (see md.fourdrgs_m1)
            _fv, _ = md.fdk_volume_lumen_frame(RENDER / f"{_c}.npz", n_views=N_VIEWS)
            _n_above = int((_fv * _meta["mu_per_mm"] > md.FOURDRGS_THRES_FDK).sum())
            _m1 = md.fourdrgs_m1(_n_above)
            if _m1 != md.FOURDRGS_M1:
                _dev.append(f"4DRGS --M1 {_m1} instead of {md.FOURDRGS_M1} ({_n_above} FDK voxels above "
                            f"{md.FOURDRGS_THRES_FDK})")
            _common = [f"-m={_m}", f"-s={_d}", f"--Nviews={N_VIEWS}", f"--iteration={DRGS_ITERS}"]
            _vq = _m / "train" / "VQR_output" / "vol_pred.nii.gz"
            if _vq.exists():
                _vq.unlink()
            _tr = subprocess.run([sys.executable, "train.py", *_common, f"--ADC_until_iter={DRGS_ITERS // 2}",
                                  "--fdk_toolbox=tigre", f"--M1={_m1}", "--save_iterations", str(DRGS_ITERS)],
                                 cwd=str(DRGS_DIR))
            if not _vq.exists():
                raise RuntimeError(f"4DRGS train.py gave no VQR volume (exit {_tr.returncode})")
            if _tr.returncode != 0:  # expected: mesh / ICP evaluation needs mesh_ref.obj
                _dev.append("4DRGS train.py stopped in its mesh / ICP evaluation (no reference mesh); "
                            "VQR volume written before it is used")
            _lv = md.fourdrgs_volume_to_lumen_frame(_sitk.GetArrayFromImage(_sitk.ReadImage(str(_vq))),
                                                    _meta["axis_perm"])
            _te = subprocess.run([sys.executable, "test.py", *_common, "--render_2d"], cwd=str(DRGS_DIR),
                                 capture_output=True, text=True)
            _drgs = {"m1": _m1, "n_fdk_above": _n_above, "train_exit": _tr.returncode,
                     **_sc.parse_4drgs_log(_te.stdout)}
            if _te.returncode != 0:
                _drgs["test_error"] = _te.stderr[-1500:]
        else:
            _d, _m = _base / "data", _base / "model"
            _info = md.write_r2g_case(RENDER / f"{_c}.npz", DATA / "cases" / f"{_c}.npz", DATA / "bolus" / f"{_c}.npz",
                                      _d, n_views=N_VIEWS)
            _n_fg = int((np.load(_d / "vol_gt.npy") > 0.05).sum())
            _cmd = [sys.executable, "initialize_pcd.py", "--data", str(_d)]
            if _n_fg < 75000:  # same rule as the Step 0.1 ImageCAS run
                _cmd += ["--n_points", str(max(5000, _n_fg // 2))]
                _dev.append(f"initialize_pcd.py --n_points {max(5000, _n_fg // 2)} ({_n_fg} GT voxels > 0.05)")
            for _old in _d.glob("init_*.npy"):
                _old.unlink()
            subprocess.run(_cmd, cwd=str(R2G_DIR), check=True)
            subprocess.run([sys.executable, "train.py", "-s", str(_d), "-m", str(_m)], cwd=str(R2G_DIR), check=True)
            _lv = md.r2g_volume_to_lumen_frame(np.load(_m / "point_cloud" / "iteration_30000" / "vol_pred.npy"),
                                               _info["axis_perm"])
            _ev_dir = str(_m / "eval" / "iter_030000")
        np.save(_base / "vol_lumen.npy", _lv.astype(np.float32))
        runs[_c] = {"base": str(_base), "lumen": _lum, "minutes": round((time.time() - _t0) / 60, 1),
                    "deviations": _dev, "eval_dir": _ev_dir, "drgs": _drgs}
        print(_c, INIT, runs[_c]["minutes"], "min", flush=True)
    return (runs,)


@app.cell
def _(COV_MIN, CASES, DATA, DOD_MM, FLOWSPLAT_DIR, INIT, N_VIEWS, Path, RTAG, SPLIT, json, md, np, runs, time):
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
        _lv = np.load(f"{_r['base']}/vol_lumen.npy")
        _lum = _r["lumen"]
        _case = np.load(DATA / "cases" / f"{_c}.npz")
        s_e, xyz_e, r_e = md.build_from_volume(_lv, _lum["vox_mm"], _lum["origin_mm"])
        s_e, xyz_e, r_e = md.orient_by_reference(s_e, xyz_e, r_e, _case["xyz_mm"][0])
        _err = md.radius_error(xyz_e, r_e, _case["s"], _case["xyz_mm"], _case["r_mm"])
        _p = json.loads(str(_case["params"]))
        _cov = md.coverage_metrics(xyz_e, r_e, _case["s"], _case["xyz_mm"], _p["stenosis_center_mm"],
                                   _p["stenosis_len_mm"], _p["r_min_mm"])
        _tg = md.TubeGaussians(s_e, xyz_e, r_e)
        _ev = {}
        if _r["eval_dir"]:
            for _f in sorted(Path(_r["eval_dir"]).glob("eval*.yml")):
                _ev[_f.stem] = {k: v for k, v in _yaml.safe_load(_f.read_text()).items() if isinstance(v, (int, float))}
        np.savez(f"{_r['base']}/geom_init.npz", s=s_e, xyz_mm=xyz_e, r_mm=r_e)
        row = {"case": _c, "views": N_VIEWS, "init": INIT, **_err, **_cov, "alignment_loss_init": float(_tg.alignment_loss().item()),
               "n_gaussians": int(len(s_e)), "length_est_mm": float(s_e[-1]), "length_gt_mm": float(_case["s"][-1]),
               "stenosis_degree_gt": _p["stenosis_degree"], "r_min_gt_mm": _p["r_min_mm"],
               "r2g_eval": _ev, "drgs": _r["drgs"], "minutes": _r["minutes"], "deviations": _r["deviations"],
               "pass": _err["mae_mm"] < DOD_MM,
               "pass_v2": bool(_err["mae_mm"] < DOD_MM and _cov["gt_coverage"] >= COV_MIN
                               and _cov["lesion_coverage"] >= COV_MIN)}
        rows.append(row)
        _fig, _ax = _plt.subplots(1, 2, figsize=(14, 4))
        _ax[0].imshow(_lv.max(axis=2).T, origin="lower", cmap="gray")
        _ax[0].set_title(f"{_c}: init volume ({INIT}), MIP")
        _ax[1].plot(_case["s"], _case["r_mm"], label="GT")
        _ax[1].plot(md.matched_gt_s(xyz_e, _case["s"], _case["xyz_mm"]), r_e, ".", ms=2,
                    label=f"estimate at matched GT s (MAE {_err['mae_mm']:.3f} mm, coverage {_cov['gt_coverage']:.0%})")
        _ax[1].axvspan(_p["stenosis_center_mm"] - _p["stenosis_len_mm"] / 2,
                       _p["stenosis_center_mm"] + _p["stenosis_len_mm"] / 2, color="0.9")
        _ax[1].set_xlabel("GT s (mm)")
        _ax[1].set_ylabel("R (mm)")
        _ax[1].legend()
        _fig.tight_layout()
        _fig.savefig(_res / f"step2.1_{_c}_v{N_VIEWS}_{INIT}{RTAG}.png", dpi=80)
        _plt.close(_fig)
        print(json.dumps({k: v for k, v in row.items() if k != "r2g_eval"}), flush=True)
    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "views": N_VIEWS,
        "init": INIT,
        "split": SPLIT,
        "dod_mm": DOD_MM,
        "area_ratio_per_case": {r["case"]: round(r["area_ratio"], 4) for r in rows},
        "radius_scale_calib": md.radius_scale([r["area_ratio"] for r in rows]),
        "radius_scale_calib_median": md.radius_scale([r["area_ratio"] for r in rows], stat="median"),
        "mae_mm_per_case": {r["case"]: round(r["mae_mm"], 4) for r in rows},
        "mae_mm_mean": float(np.mean([r["mae_mm"] for r in rows])),
        "all_pass": all(r["pass"] for r in rows),
        "cov_min": COV_MIN,
        "all_pass_v2": all(r["pass_v2"] for r in rows),
        "pass_v2_per_case": {r["case"]: r["pass_v2"] for r in rows},
        "heldout_psnr_per_case": {r["case"]: (r["drgs"] or {}).get("psnr_eval") for r in rows},
        "gt_coverage_per_case": {r["case"]: round(r["gt_coverage"], 3) for r in rows},
        "lesion_coverage_per_case": {r["case"]: round(r["lesion_coverage"], 3) for r in rows},
        "throat_err_mm_per_case": {r["case"]: round(r["throat_err_mm"], 3) for r in rows},
        "rows": rows,
    }
    (_res / f"step2.1_geom_init_v{N_VIEWS}_{INIT}{RTAG}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return


if __name__ == "__main__":
    app.run()
