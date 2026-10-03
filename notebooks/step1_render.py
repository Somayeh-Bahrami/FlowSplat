# Runs on: molab GPU (RTX Pro 6000 Blackwell). Not runnable on the M4 (needs TIGRE + CUDA).
#
# Step 1.2, rendering: rotational DSA of all Step 1.1 / 1.2 cases with TIGRE.
#
#   a) TIGRE importable? If not (fresh session), build it with
#      notebooks/step0_setup.py in build-only mode
#   b) unpack the data pack made on the M4 (step1_pack.zip: cases, geometries,
#      bolus, splits) into ~/flowsplat_work/data/pack_v1
#   c) render every case: 133 frames over 198 deg, each frame at its own
#      angle and time (flowsplat.render_tigre.render_case), save
#      ~/flowsplat_work/data/render_v1/<case>.npz (resumes: skips done cases)
#   d) spot check: 5 cases, 6 frames each at increasing time + kymograph,
#      results/step1.2_spot_<case>.png; summary results/step1.2_render_summary.json
#
# Input: FLOWSPLAT_PACK = path of step1_pack.zip on molab (default
# ~/flowsplat_work/data/step1_pack.zip). Run from the repo root:
#   python notebooks/step1_render.py
# FLOWSPLAT_RENDER_LIMIT=N renders only the first N cases (trial).

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
    from flowsplat import render_tigre as rt

    PACK = Path(os.environ.get("FLOWSPLAT_PACK", str(WORK / "data" / "step1_pack.zip")))
    DATA = WORK / "data" / "pack_v1"
    OUT = WORK / "data" / "render_v1"
    LIMIT = int(os.environ.get("FLOWSPLAT_RENDER_LIMIT", "0")) or None
    print("FlowSplat:", FLOWSPLAT_DIR, "| pack:", PACK, "| out:", OUT, "| limit:", LIMIT)
    return DATA, FLOWSPLAT_DIR, LIMIT, OUT, PACK, WORK, rt


@app.cell
def _(FLOWSPLAT_DIR, os, subprocess, sys):
    # a) TIGRE importable? Else build (fresh molab session).
    def _ok():
        return subprocess.run([sys.executable, "-c", "import tigre"], cwd="/", capture_output=True).returncode == 0

    if not _ok():
        print("TIGRE not importable; building with step0_setup.py (build only).", flush=True)
        subprocess.run([sys.executable, "notebooks/step0_setup.py"], cwd=str(FLOWSPLAT_DIR),
                       env=dict(os.environ, FLOWSPLAT_BUILD_ONLY="1", FLOWSPLAT_RUN_4DRGS="0"))
    if not _ok():
        raise RuntimeError("TIGRE still not importable.")
    tigre_ready = True
    return (tigre_ready,)


@app.cell
def _(DATA, PACK, json, tigre_ready, zipfile):
    # b) Unpack the M4 data pack.
    _ = tigre_ready
    if not (DATA / "splits.json").exists():
        if not PACK.exists():
            raise FileNotFoundError(f"Data pack not found: {PACK}. Upload step1_pack.zip (made on the M4).")
        DATA.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(PACK) as _z:
            _z.extractall(DATA)
    splits = json.loads((DATA / "splits.json").read_text())
    CASES = sorted(p.stem for p in (DATA / "cases").glob("*.npz"))
    print(len(CASES), "cases; split sizes:", {k: len(v) for k, v in splits["cases"].items()})
    return CASES, splits


@app.cell
def _(CASES, DATA, LIMIT, OUT, json, rt, time):
    # c) Render all cases (resumable).
    import numpy as _np

    OUT.mkdir(parents=True, exist_ok=True)
    _todo = CASES[:LIMIT] if LIMIT else CASES
    render_rows = []
    _t0 = time.time()
    for _i, _c in enumerate(_todo):
        _f = OUT / f"{_c}.npz"
        _geom = _c.split("_")[0]
        _ax = json.loads(str(_np.load(DATA / "geom" / f"{_geom}.npz")["info"]))["axcodes"]
        if not _f.exists():
            _t = time.time()
            _p, _meta = rt.render_case(DATA / "cases" / f"{_c}.npz", DATA / "bolus" / f"{_c}.npz", _ax,
                                       seed=int(_geom) * 10 + int(_c.split("_")[1]))
            _b = _np.load(DATA / "bolus" / f"{_c}.npz")
            _np.savez(_f, proj=_p, angles_deg=_b["angles_deg"], t_s=_b["t_s"], meta=rt.meta_json(_meta))
            _sec = round(time.time() - _t, 1)
        else:
            _sec = None
        _z = _np.load(_f)
        render_rows.append({"case": _c, "shape": list(_z["proj"].shape), "max": float(_z["proj"].max()),
                            "mb": round(_f.stat().st_size / 1e6, 1), "seconds": _sec})
        print(f"[{_i + 1}/{len(_todo)}] {_c}: {render_rows[-1]}", flush=True)
    render_minutes = round((time.time() - _t0) / 60, 1)
    return render_minutes, render_rows


@app.cell
def _(DATA, FLOWSPLAT_DIR, OUT, json, render_minutes, render_rows, splits, time):
    # d) Spot check (5 cases: first case of 5 different val/test patients) and summary.
    import matplotlib as _mpl

    _mpl.use("Agg")
    import matplotlib.pyplot as _plt
    import numpy as _np

    _done = {r["case"] for r in render_rows}
    _pool = [c for c in splits["cases"]["val"] + splits["cases"]["test"] if c.endswith("_0") and c in _done]
    SPOT = (_pool + sorted(_done))[:5]
    _res = FLOWSPLAT_DIR / "results"
    _res.mkdir(exist_ok=True)
    for _c in SPOT:
        _z = _np.load(OUT / f"{_c}.npz")
        _b = _np.load(DATA / "bolus" / f"{_c}.npz")
        _p = _z["proj"].astype(_np.float32)
        _frames = _np.linspace(0, len(_p) - 1, 6).astype(int)
        _fig, _ax = _plt.subplots(1, 7, figsize=(28, 4.5))
        _vmax = float(_np.percentile(_p, 99.9))
        for _k, _j in enumerate(_frames):
            _ax[_k].imshow(_p[_j], cmap="gray_r", vmin=0, vmax=_vmax)
            _ax[_k].set_title(f"frame {_j}: t {_z['t_s'][_j]:.2f} s, {_z['angles_deg'][_j]:.0f} deg")
            _ax[_k].axis("off")
        _ax[6].imshow(_b["c"].astype(_np.float32), aspect="auto", origin="lower", cmap="magma",
                      extent=[_b["t_s"][0], _b["t_s"][-1], _b["s"][0], _b["s"][-1]], vmin=0, vmax=1)
        _ax[6].set_xlabel("t (s)")
        _ax[6].set_ylabel("s (mm)")
        _ax[6].set_title("bolus c(s, t)")
        _fig.suptitle(f"case {_c}")
        _fig.tight_layout()
        _fig.savefig(_res / f"step1.2_spot_{_c}.png", dpi=70)
        _plt.close(_fig)
    _summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rendered": len(render_rows),
        "render_minutes_this_run": render_minutes,
        "total_gb": round(sum(r["mb"] for r in render_rows) / 1e3, 2),
        "detector_shapes": sorted({tuple(r["shape"][1:]) for r in render_rows}),
        "proj_max": {"min": min(r["max"] for r in render_rows), "max": max(r["max"] for r in render_rows)},
        "spot_cases": SPOT,
        "rows": render_rows,
    }
    (_res / "step1.2_render_summary.json").write_text(json.dumps(_summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in _summary.items() if k != "rows"}, indent=2, default=str))
    return


if __name__ == "__main__":
    app.run()
