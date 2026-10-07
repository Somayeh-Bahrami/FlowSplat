# Runs on: molab GPU (RTX Pro 6000 Blackwell). Not runnable on the M4 (TIGRE, 4DRGS).
#
# Step 3.1: robustness sweeps (plan: views 30 / 60 / 133, noise, Peclet,
# flow speed, foreshortening angle) with the Gate 1 method (4DRGS init,
# radius x k, variant frozen), plus 3 repeat 4DRGS runs (Gate 1).
# Settings and levels: flowsplat.sweeps.SETTINGS. Middle level of every
# factor = Step 2.2 run 8 (60 views).
#
# Per setting (skipped when its results file exists, so the notebook can be
# run in chunks):
#   views:  k recalibrated on the 15 training cases at that view count
#           (median rule), 4DRGS on the 20 validation cases, Step 2.2 frozen
#   noise / peclet / flow / fore: new data for the 5 validation cases
#           (sweeps.make_setting_data: edited case / bolus, TIGRE render),
#           4DRGS, Step 2.2 frozen with the 60-view k
#   repeat: 4DRGS re-run on the 5 validation cases, Step 2.2 frozen
# Finally results/step3.1_sweeps.json (sweeps.collect).
#
# Run from the repo root:  python notebooks/step3_sweeps.py
# FLOWSPLAT_SWEEPS = comma list of setting names, or groups: views, noise,
#   peclet, flow, fore, repeat, all (default: all)

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

    return Path, json, mo, os, subprocess, sys, time, zipfile


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
    from flowsplat import sweeps as sw

    PACK = Path(os.environ.get("FLOWSPLAT_PACK", str(WORK / "data" / "step1_pack.zip")))
    DATA = WORK / "data" / "pack_v1"
    SWEEP_DIR = WORK / "data" / "sweeps"
    RES = FLOWSPLAT_DIR / "results"
    _req = os.environ.get("FLOWSPLAT_SWEEPS", "all").split(",")
    _groups = {k: [s for s, v in sw.SETTINGS.items() if v["kind"] == k]
               for k in ("views", "noise", "peclet", "flow", "fore", "repeat")}
    SELECTED = []
    for _r in _req:
        _add = list(sw.SETTINGS) if _r == "all" else _groups.get(_r, [_r])
        SELECTED += [s for s in _add if s not in SELECTED]
    _unknown = [s for s in SELECTED if s not in sw.SETTINGS]
    if _unknown:
        raise ValueError(f"unknown settings {_unknown}; known: {list(sw.SETTINGS)}")
    print("settings:", SELECTED)
    return DATA, FLOWSPLAT_DIR, PACK, RES, SELECTED, SWEEP_DIR, sw


@app.cell
def _(FLOWSPLAT_DIR, os, subprocess, sys):
    # extensions first: the sweep renders (TIGRE) run in this process before
    # any Step 2 notebook could build them (fresh session, Oct 7:
    # "No module named 'tigre.utilities'")
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
def _(DATA, PACK, ext_ready, zipfile):
    # data (same pack as Step 2)
    _ = ext_ready
    if not (DATA / "splits.json").exists():
        if not PACK.exists():
            raise FileNotFoundError(f"Upload step1_pack.zip to {PACK}")
        DATA.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(PACK) as _z:
            _z.extractall(DATA)
    data_ready = True
    return (data_ready,)


@app.cell
def _(FLOWSPLAT_DIR, os, subprocess, sys):
    def run_nb(name, env, log_path):
        """Run one notebook as a script, stream and save its output; raise on failure."""
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

    return (run_nb,)


@app.cell
def _(DATA, RES, SELECTED, SWEEP_DIR, data_ready, json, run_nb, sw, time):
    # all selected settings, one after the other
    _ = data_ready
    done = {}
    for _s in SELECTED:
        _spec = sw.SETTINGS[_s]
        _out = RES / sw.result_name(_s)
        if _out.exists():
            print(f"== {_s}: done ({_out.name}), skipped", flush=True)
            done[_s] = "skipped"
            continue
        _t0 = time.time()
        _log = RES / f"step3.1_{_s}.log"
        print(f"== {_s}: {_spec}", flush=True)
        _base = {"FLOWSPLAT_INIT": "4drgs"}
        if _spec["kind"] == "views":
            _v = _spec["views"]
            _cal = RES / f"step2.1_geom_init_v{_v}_4drgs_train15.json"
            if not _cal.exists():
                run_nb("step2_geom_init.py", {**_base, "FLOWSPLAT_VIEWS": _v, "FLOWSPLAT_SPLIT": "train",
                                              "FLOWSPLAT_N_CASES": 15}, _log)
            _k = json.loads(_cal.read_text())["radius_scale_calib_median"]
            print(f"k (median, {_v} views) = {_k:.4f}", flush=True)
            run_nb("step2_geom_init.py", {**_base, "FLOWSPLAT_VIEWS": _v, "FLOWSPLAT_CASES": ",".join(sw.VAL20)}, _log)
            run_nb("step2_train.py", {**_base, "FLOWSPLAT_VIEWS": _v, "FLOWSPLAT_MODE": "val20",
                                      "FLOWSPLAT_VARIANTS": "frozen", "FLOWSPLAT_R_SCALE": repr(_k)}, _log)
        else:
            _env = {**_base, "FLOWSPLAT_VIEWS": 60, "FLOWSPLAT_RUN_TAG": f"_{_s}",
                    "FLOWSPLAT_CASES": ",".join(sw.cases_of(_s))}
            if sw.needs_new_data(_s):
                _d = sw.make_setting_data(_s, DATA, SWEEP_DIR / _s)
                _env.update({"FLOWSPLAT_DATA_DIR": str(_d), "FLOWSPLAT_RENDER_DIR": str(_d / "render")})
            run_nb("step2_geom_init.py", _env, _log)
            run_nb("step2_train.py", {**_env, "FLOWSPLAT_MODE": "sweep", "FLOWSPLAT_VARIANTS": "frozen",
                                      "FLOWSPLAT_R_SCALE": repr(sw.K60)}, _log)
        done[_s] = round((time.time() - _t0) / 60, 1)
        print(f"== {_s}: finished in {done[_s]} min", flush=True)
    return (done,)


@app.cell
def _(RES, done, json, sw):
    # one table of everything finished so far
    _ = done
    sweep_table = sw.collect(RES)
    (RES / "step3.1_sweeps.json").write_text(json.dumps(sweep_table, indent=2, default=str))
    for _k, _v in sweep_table["table"].items():
        _f = _v.get("ffr_est", {})
        print(f"{_k:18s} n={_v['n']:2d} Q {_v['q']['mean']:.3f} (signed {_v['q_signed_mean']:+.3f}) "
              f"diam {_v['diam_mm']['mean']} FFR MAE {_f.get('mae')} acc {_f.get('accuracy')}")
    print("missing:", sweep_table["missing"])
    return


if __name__ == "__main__":
    app.run()
