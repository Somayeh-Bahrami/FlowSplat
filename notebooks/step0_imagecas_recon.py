# Runs on: molab GPU (RTX Pro 6000 Blackwell). Not runnable on the M4 (needs CUDA).
#
# Step 0.1, DoD item: "Static reconstruction of one ImageCAS vessel with
# R2-Gaussian renders correctly."
#
#   a) make sure the R2-Gaussian extensions and TIGRE are importable; if not
#      (fresh molab session), build them with notebooks/step0_setup.py in
#      build-only mode
#   b) load ONE ImageCAS label (*.label.nii.gz), keep its largest connected
#      component, turn it into a 256^3 volume in [0, 1]
#      (flowsplat.imagecas_volume, same steps as R2-Gaussian process_raw_data.py)
#   c) X-ray projections with R2-Gaussian's own generate_data.py and
#      scanner/cone_beam.yml (50 train views, 100 test views), unchanged
#   d) initialisation with R2-Gaussian's initialize_pcd.py, training with
#      train.py at default settings (30k iterations), unchanged
#   e) metrics (R2-Gaussian PSNR / SSIM + Dice of the vessel), MIP figure,
#      results/step0.1_imagecas.json and results/step0.1_imagecas.png
#
# Input: IMAGECAS_LABEL = path on molab to one ImageCAS label file, e.g.
# /home/marimo/imagecas/1.label.nii.gz (upload it to molab first). If not
# set, the first *.label.nii.gz under ~/flowsplat_work/imagecas is used.
#
# Run from the repo root:  python notebooks/step0_imagecas_recon.py
# No source file of R2-Gaussian or TIGRE is edited.

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import glob
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    import marimo as mo

    return Path, glob, json, mo, os, subprocess, sys, time


@app.cell
def _(Path, mo, os, sys):
    # Paths. WORK and the R2-Gaussian clone are the ones of step0_setup.py.
    WORK = Path(os.environ.get("FLOWSPLAT_WORK", str(Path.home() / "flowsplat_work")))
    R2G_DIR = WORK / "external" / "r2_gaussian"
    try:
        FLOWSPLAT_DIR = Path(mo.notebook_dir()).resolve().parent
    except Exception:
        FLOWSPLAT_DIR = Path.cwd().resolve()
    if not (FLOWSPLAT_DIR / "src" / "flowsplat").exists():
        FLOWSPLAT_DIR = Path.cwd().resolve()
    sys.path.insert(0, str(FLOWSPLAT_DIR / "src"))
    from flowsplat import imagecas_volume as iv

    # Same data setting as the R2-Gaussian example reproduced in Step 0.1
    # (README Sec 3.2 data: cone_ntrain_50_angle_360; generate_data.py
    # README: --n_train 50 --n_test 100).
    N_TRAIN, N_TEST = 50, 100
    TARGET_SIZE = 256  # R2-Gaussian synthetic README: "resize to 256x256x256"
    # Pass rule for "renders correctly". Ref: OURS (no published value for an
    # ImageCAS vessel). Dice of pred > 0.5 vs GT > 0.5, density scale [0, 1].
    DICE_THRESH, DICE_PASS = 0.5, 0.7
    print("FlowSplat:", FLOWSPLAT_DIR, "| WORK:", WORK)
    return (
        DICE_PASS,
        DICE_THRESH,
        FLOWSPLAT_DIR,
        N_TEST,
        N_TRAIN,
        R2G_DIR,
        TARGET_SIZE,
        WORK,
        iv,
    )


@app.cell
def _(FLOWSPLAT_DIR, R2G_DIR, os, subprocess, sys):
    # a) Extensions importable? If not, build them (fresh molab session).
    def _importable():
        _r = subprocess.run(
            [sys.executable, "-c", "import torch, simple_knn._C, xray_gaussian_rasterization_voxelization, tigre"],
            cwd="/", capture_output=True, text=True,
        )
        return _r.returncode == 0, (_r.stderr.strip().splitlines() or [""])[-1]

    _ok, _err = _importable()
    built_now = False
    if not _ok or not (R2G_DIR / "train.py").exists():
        print("Extensions not importable (", _err, "). Building with step0_setup.py (build only).", flush=True)
        _env = dict(os.environ, FLOWSPLAT_BUILD_ONLY="1", FLOWSPLAT_RUN_4DRGS="0")
        _p = subprocess.run([sys.executable, "notebooks/step0_setup.py"], cwd=str(FLOWSPLAT_DIR), env=_env)
        _ok, _err = _importable()
        built_now = True
    if not _ok:
        raise RuntimeError(f"R2-Gaussian extensions still not importable: {_err}")
    print("Extensions importable. Built in this run:", built_now)
    return (built_now,)


@app.cell
def _(Path, TARGET_SIZE, WORK, built_now, glob, iv, json, os):
    # b) One ImageCAS label -> R2-Gaussian GT volume.
    _ = built_now
    import numpy as _np
    import SimpleITK as _sitk

    _label_path = os.environ.get("IMAGECAS_LABEL")
    if not _label_path:
        _found = sorted(glob.glob(str(WORK / "imagecas" / "**" / "*.label.nii.gz"), recursive=True))
        if not _found:
            raise FileNotFoundError(
                "No ImageCAS label. Upload one *.label.nii.gz to molab and set IMAGECAS_LABEL, "
                f"or put it under {WORK / 'imagecas'}."
            )
        _label_path = _found[0]
    LABEL_PATH = Path(_label_path)
    CASE_ID = LABEL_PATH.name.split(".")[0]

    _img = _sitk.ReadImage(str(LABEL_PATH))
    # SimpleITK array order is (z, y, x); spacing is (x, y, z).
    _label = _np.transpose(_sitk.GetArrayFromImage(_img), (2, 1, 0))
    _spacing = list(_img.GetSpacing())
    vol_gt, vol_info = iv.label_to_r2g_volume(_label, _spacing, target_size=TARGET_SIZE, margin_mm=5.0)
    vol_info.update({"label_path": str(LABEL_PATH), "case_id": CASE_ID, "label_values": sorted(_np.unique(_label).tolist())[:10]})

    VOL_NAME = f"imagecas_{CASE_ID}"
    VOL_PATH = WORK / "imagecas" / "vol" / f"{VOL_NAME}.npy"
    VOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _np.save(VOL_PATH, vol_gt)
    print(json.dumps(vol_info, indent=2))
    return CASE_ID, VOL_NAME, VOL_PATH, vol_gt, vol_info


@app.cell
def _(N_TEST, N_TRAIN, R2G_DIR, VOL_NAME, VOL_PATH, WORK, subprocess, sys):
    # c) Projections with R2-Gaussian's own generator and scanner config.
    # Ref: R2-Gaussian data_generator/synthetic_dataset/README.md, "(Optional)
    # Single case example" command; scanner/cone_beam.yml (commit f2579bf).
    # generate_data.py imports plotly, which R2-Gaussian's requirements.txt
    # does not list (the example data used so far came pre-generated).
    if subprocess.run([sys.executable, "-c", "import plotly"], capture_output=True).returncode != 0:
        _pip = [sys.executable, "-m", "pip", "install", "plotly"]
        if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode != 0:
            _pip = ["uv", "pip", "install", "--python", sys.executable, "plotly"]
        subprocess.run(_pip, check=True)
    _scanner = R2G_DIR / "data_generator" / "synthetic_dataset" / "scanner" / "cone_beam.yml"
    DATA_ROOT = WORK / "data_imagecas" / f"cone_ntrain_{N_TRAIN}_angle_360"
    CASE_DIR = DATA_ROOT / f"{VOL_NAME}_cone"  # generate_data.py names it <vol>_<mode>
    subprocess.run(
        [
            sys.executable, "data_generator/synthetic_dataset/generate_data.py",
            "--vol", str(VOL_PATH), "--scanner", str(_scanner), "--output", str(DATA_ROOT),
            "--n_train", str(N_TRAIN), "--n_test", str(N_TEST),
        ],
        cwd=str(R2G_DIR), check=True,
    )
    if not (CASE_DIR / "meta_data.json").exists():
        raise RuntimeError(f"generate_data.py did not write {CASE_DIR / 'meta_data.json'}")
    print("Data:", CASE_DIR)
    return (CASE_DIR,)


@app.cell
def _(CASE_DIR, R2G_DIR, WORK, json, subprocess, sys, time, vol_gt):
    # d) Initialisation and training, R2-Gaussian defaults.
    import numpy as _np

    deviations = []
    # initialize_pcd.py samples n_points (default 50000) voxels of the FDK
    # volume above density_thresh (0.05) and asserts that enough exist. A
    # coronary tree fills only a few percent of the cube, so n_points is
    # lowered when the GT has too few voxels. CLI flag only.
    _n_fg = int((vol_gt > 0.05).sum())
    _init_cmd = [sys.executable, "initialize_pcd.py", "--data", str(CASE_DIR)]
    if _n_fg < 75000:
        _n_points = max(5000, _n_fg // 2)
        _init_cmd += ["--n_points", str(_n_points)]
        deviations.append(
            {
                "what": f"initialize_pcd.py --n_points {_n_points}",
                "instead_of": "default n_points = 50000",
                "reason": f"Only {_n_fg} GT voxels above 0.05 (sparse vessel); the default would fail the "
                "'Valid voxels less than target number' assertion. CLI flag only.",
            }
        )
    # initialize_pcd.py refuses to overwrite its output; remove ours from a previous run.
    for _old in CASE_DIR.glob("init_*.npy"):
        _old.unlink()
    subprocess.run(_init_cmd, cwd=str(R2G_DIR), check=True)

    MODEL_DIR = WORK / "output_imagecas" / CASE_DIR.name
    _t0 = time.time()
    # Same command as the reproduced example (README Sec 3.2) plus -m.
    subprocess.run([sys.executable, "train.py", "-s", str(CASE_DIR), "-m", str(MODEL_DIR)], cwd=str(R2G_DIR), check=True)
    train_minutes = round((time.time() - _t0) / 60, 1)
    print("Trained in", train_minutes, "min. Deviations:", json.dumps(deviations, indent=2))
    return MODEL_DIR, deviations, train_minutes


@app.cell
def _(
    CASE_ID,
    DICE_PASS,
    DICE_THRESH,
    FLOWSPLAT_DIR,
    MODEL_DIR,
    N_TRAIN,
    deviations,
    iv,
    json,
    time,
    train_minutes,
    vol_info,
):
    # e) Metrics, figure, results.
    import matplotlib as _mpl

    _mpl.use("Agg")
    import matplotlib.pyplot as _plt
    import numpy as _np
    import yaml as _yaml

    _it = MODEL_DIR / "point_cloud" / "iteration_30000"
    _pred = _np.load(_it / "vol_pred.npy")
    _gt = _np.load(_it / "vol_gt.npy")
    _metrics = {}
    for _f in sorted((MODEL_DIR / "eval" / "iter_030000").glob("eval*.yml")):
        _d = _yaml.safe_load(_f.read_text())
        _metrics[_f.stem] = {k: v for k, v in _d.items() if isinstance(v, (int, float))}
    _dice = iv.dice(_pred > DICE_THRESH, _gt > DICE_THRESH)
    _finite = bool(_np.isfinite(_pred).all())

    # Figure: MIPs of GT, prediction and |difference| along x, y, z.
    _fig, _ax = _plt.subplots(3, 3, figsize=(10, 10))
    for _i, (_g, _p) in enumerate(zip(iv.mips(_gt), iv.mips(_pred))):
        for _j, (_im, _t) in enumerate([(_g, "GT"), (_p, "R2-Gaussian"), (_np.abs(_g - _p), "|diff|")]):
            _ax[_i, _j].imshow(_im.T, origin="lower", cmap="gray", vmin=0, vmax=1)
            _ax[_i, _j].set_title(f"{_t}, MIP axis {'xyz'[_i]}")
            _ax[_i, _j].axis("off")
    _fig.suptitle(f"ImageCAS case {CASE_ID}, {N_TRAIN} views, Dice {_dice:.3f}")
    _fig.tight_layout()
    _png = FLOWSPLAT_DIR / "results" / "step0.1_imagecas.png"
    _png.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_png, dpi=120)

    imagecas_result = {
        "step": "0.1",
        "item": "ImageCAS vessel static reconstruction",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "volume": vol_info,
        "views_train": N_TRAIN,
        "train_minutes": train_minutes,
        "r2g_metrics": _metrics,
        "dice": round(_dice, 4),
        "dice_threshold": DICE_THRESH,
        "pred_finite": _finite,
        "pass_rule": f"training finished, prediction finite, Dice >= {DICE_PASS} (Ref: OURS)",
        "pass": _finite and _dice >= DICE_PASS,
        "deviations": deviations,
        "figure": str(_png),
    }
    _out = FLOWSPLAT_DIR / "results" / "step0.1_imagecas.json"
    _out.write_text(json.dumps(imagecas_result, indent=2, default=str))
    print(json.dumps(imagecas_result, indent=2, default=str))
    return (imagecas_result,)


if __name__ == "__main__":
    app.run()
