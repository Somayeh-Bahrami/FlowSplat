# Runs on: molab GPU (RTX Pro 6000 Blackwell). Not runnable on the M4 (TIGRE, 4DRGS, R2-Gaussian).
#
# Step 4.1: baselines and the main table on the held-out test split (30 cases).
# DoD: one results table (Q error, FFR MAE, FFR accuracy, diameter error) on
# the held-out test set for all methods.
#
# Parts (FLOWSPLAT_PARTS, comma list; default flowsplat,b1,b2,eval). Every
# part skips cases whose output already exists, so the notebook runs in chunks.
#   flowsplat  4DRGS init (60 views) + Step 2.2 frozen with k = sweeps.K60
#              (Gate 1 method; k from the training split)
#   b1         4DRGS + post-hoc TDC velocity: time-density curves of the same
#              4DRGS models at our centerline (flowsplat.baselines.drgs_tdc,
#              run inside the 4DRGS repo, source untouched)
#   b2         PUNCH-style 2D: fixed-view angiogram (gantry angle 0, all 133
#              frames, render_tigre fixed_angle_deg), kymograph along the
#              projected ground-truth centerline (flowsplat.baselines.kymo)
#   b3         static R2-Gaussian geometry at defaults (step2_geom_init.py
#              INIT=r2g, about 14 min per case); FLOWSPLAT_B3_MAX = new cases
#              per run (default all); FLOWSPLAT_B3_CASES = explicit case list
#              (a fresh session does not know which cases an earlier one did)
#   eval       flowsplat.baselines.table -> results/step4.1_test_table.json
# The oracle (b4) needs no GPU and is part of eval.
#
# Run from the repo root:  python notebooks/step4_baselines.py

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
    from flowsplat import sweeps as sw
    from flowsplat.baselines import kymo, table

    PACK = Path(os.environ.get("FLOWSPLAT_PACK", str(WORK / "data" / "step1_pack.zip")))
    DATA = WORK / "data" / "pack_v1"
    RENDER = WORK / "data" / "render_v1"
    RES = FLOWSPLAT_DIR / "results"
    DRGS_DIR = WORK / "external" / "4DRGS"
    B4 = WORK / "step4"
    PARTS = os.environ.get("FLOWSPLAT_PARTS", "flowsplat,b1,b2,eval").split(",")
    B3_MAX = int(os.environ.get("FLOWSPLAT_B3_MAX", "1000"))
    B3_CASES = [c for c in os.environ.get("FLOWSPLAT_B3_CASES", "").split(",") if c]
    K = sw.K60  # Gate 1 k (60 views, 15 training cases, median rule)
    print("parts:", PARTS, "| k:", K)
    return B3_CASES, B3_MAX, B4, DATA, DRGS_DIR, FLOWSPLAT_DIR, K, PACK, PARTS, RENDER, RES, kymo, md, rt, sw, table


@app.cell
def _(FLOWSPLAT_DIR, os, subprocess, sys):
    # extensions (TIGRE, R2-Gaussian, 4DRGS), as in step3_sweeps.py
    _mods = ("import torch, tigre.utilities, simple_knn._C, xray_gaussian_rasterization_voxelization, tinycudann,"
             " diff_Xray_gaussian_rasterization_voxelization, leapctype, pyhocon, SimpleITK, mcubes, pymeshlab, open3d")

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
def _(DATA, PACK, RENDER, ext_ready, json, np, rt, zipfile):
    # data, test cases, renders (re-rendered with the Step 1.2 seed if missing)
    _ = ext_ready
    if not (DATA / "splits.json").exists():
        if not PACK.exists():
            raise FileNotFoundError(f"Upload step1_pack.zip to {PACK}")
        DATA.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(PACK) as _z:
            _z.extractall(DATA)
    CASES = json.loads((DATA / "splits.json").read_text())["cases"]["test"]
    AX = {}
    RENDER.mkdir(parents=True, exist_ok=True)
    for _c in CASES:
        _g = _c.split("_")[0]
        AX[_c] = json.loads(str(np.load(DATA / "geom" / f"{_g}.npz")["info"]))["axcodes"]
        if not (RENDER / f"{_c}.npz").exists():
            print("rendering", _c, flush=True)
            _p, _m = rt.render_case(DATA / "cases" / f"{_c}.npz", DATA / "bolus" / f"{_c}.npz", AX[_c],
                                    seed=int(_g) * 10 + int(_c.split("_")[1]))
            _b = np.load(DATA / "bolus" / f"{_c}.npz")
            np.savez(RENDER / f"{_c}.npz", proj=_p, angles_deg=_b["angles_deg"], t_s=_b["t_s"], meta=rt.meta_json(_m))
    print(len(CASES), "test cases:", CASES)
    return AX, CASES


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
def _(B4, CASES, K, PARTS, RES, run_nb):
    # FlowSplat on the test split (Gate 1 method)
    flowsplat_done = False
    if "flowsplat" in PARTS:
        _w = B4.parent  # WORK
        _miss = [c for c in CASES if not (_w / "step2_1" / f"{c}_v60_4drgs" / "vol_lumen.npy").exists()]
        print("flowsplat: 4DRGS missing for", len(_miss), "cases", flush=True)
        _log = RES / "step4.1_flowsplat.log"
        if _miss:
            run_nb("step2_geom_init.py", {"FLOWSPLAT_INIT": "4drgs", "FLOWSPLAT_VIEWS": 60,
                                          "FLOWSPLAT_CASES": ",".join(_miss)}, _log)
        if not (RES / "step2.2_test_v60_4drgs_cal.json").exists():
            run_nb("step2_train.py", {"FLOWSPLAT_INIT": "4drgs", "FLOWSPLAT_VIEWS": 60, "FLOWSPLAT_MODE": "test",
                                      "FLOWSPLAT_VARIANTS": "frozen", "FLOWSPLAT_R_SCALE": repr(K)}, _log)
        flowsplat_done = True
    return (flowsplat_done,)


@app.cell
def _(AX, B4, CASES, DATA, DRGS_DIR, FLOWSPLAT_DIR, PARTS, RENDER, flowsplat_done, json, md, np, os, subprocess, sys):
    # Baseline 1: time-density curves of the 4DRGS models at our centerline
    _ = flowsplat_done
    b1_done = False
    if "b1" in PARTS:
        _w = B4.parent
        (B4 / "b1").mkdir(parents=True, exist_ok=True)
        for _c in CASES:
            _out = B4 / "b1" / f"{_c}_tdc.npz"
            _base = _w / "step2_1" / f"{_c}_v60_4drgs"
            if _out.exists():
                continue
            if not (_base / "geom_init.npz").exists():
                print("b1: no 4DRGS model for", _c, "(run part flowsplat first)", flush=True)
                continue
            _meta = json.loads(str(np.load(RENDER / f"{_c}.npz")["meta"]))
            _world, _ = md.lumen_to_world(np.load(_base / "geom_init.npz")["xyz_mm"], _meta["lumen"], _meta["axis_perm"])
            _pts = B4 / "b1" / f"{_c}_points_world.npy"
            np.save(_pts, _world.astype(np.float32))
            _env = dict(os.environ, PYTHONPATH=str(FLOWSPLAT_DIR / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
            _r = subprocess.run([sys.executable, "-m", "flowsplat.baselines.drgs_tdc", "--model", str(_base / "model"),
                                 "--data", str(_base / "data"), "--points", str(_pts), "--out", str(_out)],
                                cwd=str(DRGS_DIR), env=_env, capture_output=True, text=True)
            print(_c, "b1", "ok" if _r.returncode == 0 else "FAILED", _r.stdout.strip().splitlines()[-1:] if _r.stdout else "", flush=True)
            if _r.returncode != 0:
                print(_r.stderr[-2000:], flush=True)
        b1_done = True
    _ = AX, DATA
    return (b1_done,)


@app.cell
def _(AX, B4, CASES, DATA, PARTS, RENDER, b1_done, json, kymo, md, np, rt, sw):
    # Baseline 2: fixed-view angiogram and kymograph along the projected GT centerline
    _ = b1_done
    b2_done = False
    if "b2" in PARTS:
        (B4 / "b2").mkdir(parents=True, exist_ok=True)
        for _c in CASES:
            _out = B4 / "b2" / f"{_c}_kymo.npz"
            if _out.exists():
                continue
            _g = _c.split("_")[0]
            _proj, _m = rt.render_case(DATA / "cases" / f"{_c}.npz", DATA / "bolus" / f"{_c}.npz", AX[_c],
                                       seed=int(_g) * 10 + int(_c.split("_")[1]), fixed_angle_deg=0.0)
            _case = np.load(DATA / "cases" / f"{_c}.npz")
            _world, _ = md.lumen_to_world(_case["xyz_mm"], _m["lumen"], _m["axis_perm"])
            _u, _v, _ = kymo.project_points(_world, 0.0, _m["geometry"])
            _rows, _cols = kymo.pixel_coords(_u, _v, _m["geometry"])
            _k = kymo.kymograph(_proj.astype(np.float32), _rows, _cols)
            np.savez(_out, s2d_mm=kymo.arc_length_isocenter(_u, _v, _m["geometry"]), kymo=_k.astype(np.float32),
                     rows=_rows, cols=_cols, s3d_mm=_case["s"],
                     out_of_plane_deg=sw.out_of_plane_deg(_case["xyz_mm"], _m["axis_perm"]))
            print(_c, "b2 kymograph", _k.shape, flush=True)
        b2_done = True
    _ = RENDER, json
    return (b2_done,)


@app.cell
def _(B3_CASES, B3_MAX, B4, CASES, PARTS, RES, b2_done, run_nb):
    # Baseline 3: static R2-Gaussian geometry (defaults), FLOWSPLAT_B3_MAX new cases per run
    _ = b2_done
    b3_done = False
    if "b3" in PARTS:
        _w = B4.parent
        _miss = [c for c in (B3_CASES or CASES) if not (_w / "step2_1" / f"{c}_v60_r2g" / "geom_init.npz").exists()]
        _todo = _miss[:B3_MAX]
        print("b3: R2-Gaussian missing for", len(_miss), "cases; this run:", _todo, flush=True)
        if _todo:
            run_nb("step2_geom_init.py", {"FLOWSPLAT_INIT": "r2g", "FLOWSPLAT_VIEWS": 60,
                                          "FLOWSPLAT_CASES": ",".join(_todo)}, RES / "step4.1_b3.log")
        b3_done = True
    return (b3_done,)


@app.cell
def _(B4, CASES, DATA, K, PARTS, RES, b3_done, json, table):
    # main table (all methods, whatever is available so far)
    _ = b3_done
    if "eval" in PARTS:
        _t = table.build(CASES, B4.parent, DATA / "cases", DATA / "bolus", RES / "step2.2_test_v60_4drgs_cal.json", K)
        (RES / "step4.1_test_table.json").write_text(json.dumps(_t, indent=2, default=float))
        for _m, _e in _t["table"].items():
            _q = _e.get("q_rel_err", _e.get("q_hyp_rel_err", {}))
            _f = _e.get("ffr_est", {})
            print(f"{_m:16s} n={_e['n']:2d} Q {_q.get('mean')} diam {_e.get('diam_err_mm', {}).get('mean')} "
                  f"FFR MAE {_f.get('mae')} acc {_f.get('accuracy')}")
    return


if __name__ == "__main__":
    app.run()
