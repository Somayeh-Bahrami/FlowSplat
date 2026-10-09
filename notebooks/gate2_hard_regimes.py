# Runs on: molab GPU (RTX Pro 6000 Blackwell). Not runnable on the M4 (TIGRE, 4DRGS).
#
# Gate 2 (held Oct 9, reports/gate2.md): hard-regime comparison of FlowSplat
# and baseline 1 (4DRGS + post-hoc TDC) on the 30 held-out test cases. Both
# use the same 4DRGS reconstruction and radius calibration; only the flow
# estimate differs (as in Step 4.1).
#
# Regimes (FLOWSPLAT_REGIME, one per run; finished parts are skipped):
#   views30     30 views, k = 1.126 (Step 3.1, 15 training cases, median)
#   flow040     rest velocity 0.40 m/s (sweeps setting flow_v0.40), k = 1.086
#   noise_high  I0 1e4, Gaussian sigma 30 (sweeps setting noise_high), k = 1.086
# Per regime: data for the test cases (sweeps.make_setting_data, flow040 and
# noise_high), 4DRGS + Step 2.2 frozen (step2_geom_init.py, step2_train.py),
# B1 time-density curves (baselines/drgs_tdc.py), table
# (baselines/table.build + paired_q) -> results/gate2_<regime>.json.
#
# Run from the repo root:  python notebooks/gate2_hard_regimes.py

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import subprocess
    import sys
    import zipfile
    from pathlib import Path

    import marimo as mo
    import numpy as np

    return Path, json, mo, np, os, subprocess, sys, zipfile


@app.cell
def _(Path, json, mo, os, sys):
    WORK = Path(os.environ.get("FLOWSPLAT_WORK", str(Path.home() / "flowsplat_work")))
    try:
        FLOWSPLAT_DIR = Path(mo.notebook_dir()).resolve().parent
    except Exception:
        FLOWSPLAT_DIR = Path.cwd().resolve()
    if not (FLOWSPLAT_DIR / "src" / "flowsplat").exists():
        FLOWSPLAT_DIR = Path.cwd().resolve()
    sys.path.insert(0, str(FLOWSPLAT_DIR / "src"))
    from flowsplat import model as md
    from flowsplat import sweeps as sw
    from flowsplat.baselines import table

    PACK = Path(os.environ.get("FLOWSPLAT_PACK", str(WORK / "data" / "step1_pack.zip")))
    DATA = WORK / "data" / "pack_v1"
    RES = FLOWSPLAT_DIR / "results"
    DRGS_DIR = WORK / "external" / "4DRGS"
    _k30 = json.loads((RES / "step2.1_geom_init_v30_4drgs_train15.json").read_text())["radius_scale_calib_median"]
    REGIMES = {  # Gate 2 decision 3 (reports/gate2.md)
        "views30": {"views": 30, "k": _k30, "setting": None, "tag": ""},
        "flow040": {"views": 60, "k": sw.K60, "setting": "flow_v0.40", "tag": "_g2_flow040"},
        "noise_high": {"views": 60, "k": sw.K60, "setting": "noise_high", "tag": "_g2_noise_high"},
    }
    REGIME = os.environ.get("FLOWSPLAT_REGIME", "views30")
    R = REGIMES[REGIME]
    G2 = WORK / "gate2" / REGIME
    print("regime:", REGIME, R)
    return DATA, DRGS_DIR, FLOWSPLAT_DIR, G2, PACK, R, REGIME, RES, md, sw, table


@app.cell
def _(FLOWSPLAT_DIR, os, subprocess, sys):
    # extensions (as the other notebooks)
    _mods = ("import torch, tigre.utilities, simple_knn._C, tinycudann, diff_Xray_gaussian_rasterization_voxelization,"
             " leapctype, pyhocon, SimpleITK, mcubes, pymeshlab, open3d")

    def _ok():
        return subprocess.run([sys.executable, "-c", _mods], cwd="/", capture_output=True).returncode == 0

    if not _ok():
        print("Building extensions (step0_setup.py, 4DRGS build only).", flush=True)
        _env = {k: v for k, v in os.environ.items() if not k.startswith("FLOWSPLAT_") or k == "FLOWSPLAT_WORK"}
        subprocess.run([sys.executable, "notebooks/step0_setup.py"], cwd=str(FLOWSPLAT_DIR),
                       env={**_env, "FLOWSPLAT_4DRGS_BUILD_ONLY": "1", "FLOWSPLAT_RUN_4DRGS": "1"})
    if not _ok():
        _err = subprocess.run([sys.executable, "-c", _mods], cwd="/", capture_output=True, text=True).stderr
        raise RuntimeError(f"Extensions not importable:\n{_err[-2000:]}")
    ext_ready = True
    return (ext_ready,)


@app.cell
def _(DATA, G2, PACK, R, ext_ready, json, sw, zipfile):
    # data: Step 1 pack; for flow040 / noise_high the edited test cases (rendered)
    _ = ext_ready
    if not (DATA / "splits.json").exists():
        if not PACK.exists():
            raise FileNotFoundError(f"Upload step1_pack.zip to {PACK}")
        DATA.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(PACK) as _z:
            _z.extractall(DATA)
    CASES = json.loads((DATA / "splits.json").read_text())["cases"]["test"]
    if R["setting"]:
        DDIR = sw.make_setting_data(R["setting"], DATA, G2 / "data", cases=CASES)
        RDIR = DDIR / "render"
    else:
        DDIR, RDIR = DATA, DATA.parent / "render_v1"
    print(len(CASES), "test cases; data:", DDIR)
    return CASES, DDIR, RDIR


@app.cell
def _(CASES, DDIR, FLOWSPLAT_DIR, G2, R, RDIR, RES, os, subprocess, sys):
    # FlowSplat (4DRGS + Step 2.2 frozen) on the test cases of this regime
    def run_nb(name, env, log_path):
        _env = {k: v for k, v in os.environ.items()
                if not k.startswith("FLOWSPLAT_") or k in ("FLOWSPLAT_WORK", "FLOWSPLAT_PACK")}
        _env.update({k: str(v) for k, v in env.items()})
        _p = subprocess.Popen([sys.executable, f"notebooks/{name}"], cwd=str(FLOWSPLAT_DIR), env=_env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with open(log_path, "a") as _f:
            for _line in _p.stdout:
                _f.write(_line)
                print(_line, end="", flush=True)
        _p.wait()
        if _p.returncode != 0:
            raise RuntimeError(f"{name} failed (exit {_p.returncode}), see {log_path}")

    G2.mkdir(parents=True, exist_ok=True)
    _w = G2.parent.parent
    GEOM_TAG = f"v{R['views']}_4drgs{R['tag']}"
    _env = {"FLOWSPLAT_INIT": "4drgs", "FLOWSPLAT_VIEWS": R["views"], "FLOWSPLAT_RUN_TAG": R["tag"]}
    if R["setting"]:
        _env.update({"FLOWSPLAT_DATA_DIR": str(DDIR), "FLOWSPLAT_RENDER_DIR": str(RDIR)})
    _miss = [c for c in CASES if not (_w / "step2_1" / f"{c}_{GEOM_TAG}" / "vol_lumen.npy").exists()]
    _log = RES / f"gate2_{G2.name}.log"
    if _miss:
        run_nb("step2_geom_init.py", {**_env, "FLOWSPLAT_CASES": ",".join(_miss)}, _log)
    FS_JSON = RES / f"step2.2_test_v{R['views']}_4drgs_cal{R['tag']}.json"
    if not FS_JSON.exists():
        run_nb("step2_train.py", {**_env, "FLOWSPLAT_MODE": "test", "FLOWSPLAT_VARIANTS": "frozen",
                                  "FLOWSPLAT_R_SCALE": repr(R["k"])}, _log)
    return FS_JSON, GEOM_TAG


@app.cell
def _(CASES, DRGS_DIR, FLOWSPLAT_DIR, G2, GEOM_TAG, R, RDIR, json, md, np, os, subprocess, sys):
    # Baseline 1: time-density curves of the same 4DRGS models
    B1 = G2 / "b1"
    B1.mkdir(parents=True, exist_ok=True)
    _w = G2.parent.parent
    for _c in CASES:
        _out = B1 / f"{_c}_tdc.npz"
        _base = _w / "step2_1" / f"{_c}_{GEOM_TAG}"
        if _out.exists() or not (_base / "geom_init.npz").exists():
            continue
        _meta = json.loads(str(np.load(RDIR / f"{_c}.npz")["meta"]))
        _world, _ = md.lumen_to_world(np.load(_base / "geom_init.npz")["xyz_mm"], _meta["lumen"], _meta["axis_perm"])
        _pts = B1 / f"{_c}_points_world.npy"
        np.save(_pts, _world.astype(np.float32))
        _env = dict(os.environ, PYTHONPATH=str(FLOWSPLAT_DIR / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        _r = subprocess.run([sys.executable, "-m", "flowsplat.baselines.drgs_tdc", "--model", str(_base / "model"),
                             "--data", str(_base / "data"), "--points", str(_pts), "--out", str(_out),
                             "--nviews", str(R["views"])], cwd=str(DRGS_DIR), env=_env, capture_output=True, text=True)
        print(_c, "b1", "ok" if _r.returncode == 0 else "FAILED " + _r.stderr[-800:], flush=True)
    b1_ready = True
    return B1, b1_ready


@app.cell
def _(B1, CASES, DDIR, FS_JSON, G2, GEOM_TAG, R, REGIME, RES, b1_ready, json, table):
    # table: FlowSplat vs B1 (same geometry), paired comparison
    _ = b1_ready
    _t = table.build(CASES, G2.parent.parent, DDIR / "cases", DDIR / "bolus", FS_JSON, R["k"], GEOM_TAG, B1)
    _t["paired_q"] = table.paired_q(_t["cases"])
    _t["regime"] = {"name": REGIME, **R}
    (RES / f"gate2_{REGIME}.json").write_text(json.dumps(_t, indent=2, default=float))
    for _m in ("flowsplat", "b1_4drgs_tdc"):
        _e = _t["table"].get(_m, {})
        print(_m, "Q", _e.get("q_rel_err", {}).get("mean"), "FFR MAE", _e.get("ffr_est", {}).get("mae"),
              "acc", _e.get("ffr_est", {}).get("accuracy"))
    print("paired:", _t["paired_q"])
    return


if __name__ == "__main__":
    app.run()
