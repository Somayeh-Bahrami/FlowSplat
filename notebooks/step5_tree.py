# Runs on: molab GPU (RTX Pro 6000 Blackwell). Not runnable on the M4 (TIGRE, 4DRGS).
#
# Step 5.1: Y-bifurcation (LAD + diagonal), per-branch flow with continuity
# and Murray prior, ablation.
# DoD: flow split error below 10 % on the bifurcation test cases (split error
# = mean over the two daughters of |f_k,est - f_k,gt| / f_k,gt, f_k = Q_k / Q_p;
# user decision Oct 8); ablation table.
#
#   a) extensions (TIGRE, 4DRGS), as in the other notebooks
#   b) data: data/step5_tree_pack.zip (made on the M4 by
#      `python -m flowsplat.tree`), renders of the selected cases
#      (tree.render_tree_case; seed = 10 x geometry id + case index + 5000)
#   c) per case (skipped when its result exists): 4DRGS (model.run_fourdrgs,
#      60 views, 10k iterations), tree from the volume (tree.tree_from_volume,
#      half-max radii x k = sweeps.K60, Gate 1 method), inlet hypothesis
#      chosen with continuity, then the 4 variants of tree.TREE_VARIANTS with
#      frozen geometry; evaluation against the ground truth (tree.evaluate_tree)
#   d) results/step5.1_tree_<set>.json: per-case rows and the ablation table
#
# Run from the repo root:  python notebooks/step5_tree.py
# FLOWSPLAT_TREE_SET = dev (default; 16 geometries, pipeline check) | test
# (24 geometries, DoD); FLOWSPLAT_CASES = comma list (overrides);
# FLOWSPLAT_TREE_RERUN=1 refits cases that have a result (cached 4DRGS volumes are reused).

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
    from flowsplat import tree

    PACK = Path(os.environ.get("FLOWSPLAT_TREE_PACK", str(WORK / "data" / "step5_tree_pack.zip")))
    TDATA = WORK / "data" / "trees_v1"
    TREN = WORK / "data" / "render_trees"
    OUT = WORK / "step5"
    RES = FLOWSPLAT_DIR / "results"
    DRGS_DIR = WORK / "external" / "4DRGS"
    SET = os.environ.get("FLOWSPLAT_TREE_SET", "dev")
    N_VIEWS = 60
    K = sw.K60  # Gate 1 radius factor (single-vessel training cases)
    print("set:", SET, "| k:", K)
    return DRGS_DIR, FLOWSPLAT_DIR, K, N_VIEWS, OUT, PACK, RES, SET, TDATA, TREN, md, rt, tree


@app.cell
def _(FLOWSPLAT_DIR, os, subprocess, sys):
    # a) extensions
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
def _(PACK, SET, TDATA, TREN, ext_ready, json, np, os, rt, tree, zipfile):
    # b) data and renders
    _ = ext_ready
    if not (TDATA / "splits_tree.json").exists():
        if not PACK.exists():
            raise FileNotFoundError(f"Upload step5_tree_pack.zip to {PACK}")
        TDATA.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(PACK) as _z:
            _z.extractall(TDATA)
    _sp = json.loads((TDATA / "splits_tree.json").read_text())
    CASES = os.environ["FLOWSPLAT_CASES"].split(",") if os.environ.get("FLOWSPLAT_CASES") else _sp[f"{SET}_runs"]
    TREN.mkdir(parents=True, exist_ok=True)
    for _c in CASES:
        _f = TREN / f"{_c}.npz"
        if _f.exists():
            continue
        _, _p = tree.load_tree_case(TDATA / "cases" / f"{_c}.npz")
        _g, _k = _c.split("_")
        _proj, _m = tree.render_tree_case(TDATA / "cases" / f"{_c}.npz", TDATA / "bolus" / f"{_c}.npz", _p["axcodes"],
                                          seed=int(_g) * 10 + int(_k) + 5000)
        _b = np.load(TDATA / "bolus" / f"{_c}.npz")
        np.savez(_f, proj=_proj, angles_deg=_b["angles_deg"], t_s=_b["t_s"], meta=rt.meta_json(_m))
        print("rendered", _c, flush=True)
    print(len(CASES), "cases:", CASES)
    return (CASES,)


@app.cell
def _(CASES, DRGS_DIR, K, N_VIEWS, OUT, TDATA, TREN, json, md, np, os, rt, time, tree):
    # c) 4DRGS, tree extraction, fits of all variants, evaluation
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for _c in CASES:
        _d = OUT / _c
        _d.mkdir(parents=True, exist_ok=True)
        if (_d / "result.json").exists() and os.environ.get("FLOWSPLAT_TREE_RERUN") != "1":
            rows.append(json.loads((_d / "result.json").read_text()))
            print(_c, "done, loaded", flush=True)
            continue
        _t0 = time.time()
        try:
            _z = np.load(TREN / f"{_c}.npz")
            _meta = json.loads(str(_z["meta"]))
            _lum = _meta["lumen"]
            if (_d / "vol_lumen.npy").exists():
                _vol = np.load(_d / "vol_lumen.npy")
                _dinfo = json.loads((_d / "drgs.json").read_text())
            else:
                _vol, _dinfo = md.run_fourdrgs(TREN / f"{_c}.npz", _d / "drgs", DRGS_DIR, n_views=N_VIEWS)
                np.save(_d / "vol_lumen.npy", _vol.astype(np.float32))
                (_d / "drgs.json").write_text(json.dumps(_dinfo))
            _segs, _tinfo = tree.tree_from_volume(_vol, _lum["vox_mm"], _lum["origin_mm"])
            _radii = [r * K for r in tree.tree_radii(_vol, _lum["vox_mm"], _lum["origin_mm"], _segs)]
            _hyps_l = tree.orientations(_segs, _radii)
            _hyps_w = [[(s, md.lumen_to_world(x, _lum, _meta["axis_perm"])[0], r) for s, x, r in h] for h in _hyps_l]
            _idx = rt.view_subsets(len(_z["angles_deg"]), (N_VIEWS,))[str(N_VIEWS)]["train"]
            _proj = torch.tensor(_z["proj"][_idx].astype(np.float32), device=device)
            _pj = md.ConeProjector(_meta["geometry"], device=device)
            _th = list(np.deg2rad(_z["angles_deg"][_idx]))
            _tt = _z["t_s"][_idx]
            _infos = {}
            _dir = None
            for _v in ("cont",) + tuple(v for v in tree.TREE_VARIANTS if v != "cont"):
                torch.manual_seed(0)
                _m, _inf = tree.fit_tree(_proj, _th, _tt, _hyps_w, _pj, variant=_v, device=device, direction=_dir,
                                         log=lambda *a, c=_c: print(c, *a, flush=True))
                _dir = _inf["direction"]
                _infos[_v] = {k: _inf[k] for k in ("direction", "dir_loss", "q_mm3s")}
                del _m
                torch.cuda.empty_cache() if device == "cuda" else None
            _est = _hyps_l[_dir]
            _row = {"case": _c, **tree.evaluate_tree(_infos, _est, TDATA / "cases" / f"{_c}.npz"),
                    "fits": _infos, "tree_info": _tinfo, "drgs": _dinfo, "minutes": round((time.time() - _t0) / 60, 1)}
            np.savez(_d / "estimate.npz", **{f"b{k}_s": _est[k][0] for k in range(3)},
                     **{f"b{k}_xyz_lumen": _est[k][1] for k in range(3)}, **{f"b{k}_r": _est[k][2] for k in range(3)})
        except Exception as _e:  # recorded, next case
            _row = {"case": _c, "error": f"{type(_e).__name__}: {_e}"}
        (_d / "result.json").write_text(json.dumps(_row, default=float))
        rows.append(_row)
        print(_c, json.dumps({k: v for k, v in _row.items() if k not in ("fits", "drgs")}, default=float)[:600], flush=True)
    return (rows,)


@app.cell
def _(RES, SET, json, rows, time, tree):
    # d) ablation table
    _ok = [r for r in rows if "error" not in r]
    table = tree.summarize_tree(_ok)
    _out = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "set": SET,
            "n_cases": len(rows), "n_ok": len(_ok), "errors": {r["case"]: r["error"] for r in rows if "error" in r},
            "inlet_correct": sum(r["inlet_correct"] for r in _ok), "table": table, "cases": rows}
    RES.mkdir(parents=True, exist_ok=True)  # missing on molab in the test run of Oct 9 (summary rebuilt on the M4)
    (RES / f"step5.1_tree_{SET}.json").write_text(json.dumps(_out, indent=2, default=float))
    for _v, _e in table.items():
        print(f"{_v:12s} n={_e['n']} split err {_e['split_err_mean']:.3f} (median {_e['split_err_median']:.3f}, "
              f"<10 %: {_e['frac_below_10pct']:.2f}) | Q daughters {_e['q_daughter_rel_err_mean']:.3f} "
              f"| Q parent {_e['q_parent_rel_err_mean']:.3f}")
    print("inlet correct:", _out["inlet_correct"], "of", len(_ok), "| errors:", _out["errors"])
    return


if __name__ == "__main__":
    app.run()
