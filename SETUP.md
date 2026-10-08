# Setup

How to get the GPU environment for FlowSplat running. Verified on molab on
Oct 3, 2026 (repo `11a482a`): R2-Gaussian builds and reproduces its example
case (PSNR 3D 35.146 vs 35.09 dB), TIGRE forward projection passes. ImageCAS case 11 vessel tree reconstructed with Dice 0.977 (Oct 3). Full
record: `results/step0.1.json`, report: `reports/step0.1.md`.

## Where things run

| What | Runs on |
|---|---|
| `src/flowsplat/`, `tests/` (pytest, CPU only) | Mac M4 |
| `notebooks/*.py` (marimo, CUDA builds and training) | molab GPU, RTX PRO 6000 Blackwell (sm_120) |

## M4 (local)

No install needed: `tests/conftest.py` puts `src/` on the path.

```bash
pip install pytest
```

```bash
pytest -q
```

## molab (GPU)

### Environment found on molab

| Item | Value |
|---|---|
| Python | 3.13.11, uv venv `/tmp/uv-venv` on top of the image's system site-packages `/usr/local/lib/python3.13` |
| torch | 2.11.0+cu130 (system site-packages), sm_120 in its arch list |
| nvcc | none matching: the image has nvcc 13.3 only (pip prefix in the system site-packages) |
| GCC | 14.2.0 |
| Driver | 595.71.05 |

A fresh molab session starts with an empty `/tmp/uv-venv` and
`~/flowsplat_work`. Build time about 3.5 min (R2-Gaussian + TIGRE),
R2-Gaussian example training 13 min.

### Run Step 0.1

Three cells in the molab bootstrap notebook:

1. **Pull:** clone or pull `main` of this repo (token from a password field).
2. **Cleanup (only after a failed Option B run):** delete
   `~/flowsplat_work/cuda_option.json` and any `torch-*+cu128` left in the
   venv. A full session restart does the same and is cleaner.
3. **Launcher:** `python notebooks/step0_setup.py` from the repo root, with
   `GH_TOKEN` in the environment; output streamed and saved to
   `results/step0.1.log`. `FLOWSPLAT_RUN_4DRGS=0` skips 4DRGS.

### What the notebook does to build on molab (Option A, the one that works)

All are build flags or environment settings. No source file of
R2-Gaussian, TIGRE or 4DRGS is edited (checked with `git diff` at the end).

1. Keep the image's torch 2.11.0+cu130.
2. Install the CUDA toolkit pinned to torch's CUDA version into the venv:
   `cuda-toolkit[nvcc,nvvm,crt,cccl,cudart]==13.0.*`. Any nvcc whose release
   differs from torch's CUDA (the image's 13.3) is rejected.
3. Build ONE `CUDA_HOME` at `~/flowsplat_work/cuda-13.0` as a symlink tree:
   venv `nvidia/cu13` first (nvcc 13.0, crt, nvvm, cccl), then the system
   `nvidia/cu13` (cudart headers and libraries, which pip does not reinstall
   because torch's copy already satisfies the pin). Add `libcudart.so` and
   `lib64` links. Use this `CUDA_HOME` for every build.
4. Add the CUDA 13 CCCL folder `<CUDA_HOME>/include/cccl` (cub, thrust) via
   `CPATH` and `NVCC_APPEND_FLAGS=-I...`.
5. Install ninja.
6. Install every CUDA extension with `pip install --no-build-isolation`
   (setup.py imports torch).
7. If a compile fails, retry once with
   `NVCC_APPEND_FLAGS="-include cfloat -include cstdint"` (GCC 14 no longer
   pulls these headers in). Needed for xray-gaussian-rasterization-voxelization.
8. After each build, check `import torch, <module>` in a fresh interpreter.
   If it fails, reinstall with `-e --config-settings editable_mode=compat`,
   then as a regular install. simple-knn needs `editable_compat`.

### Why the earlier attempts failed

| Attempt | Failure | Cause |
|---|---|---|
| Plain README install | `nvcc not found` | molab has no CUDA toolkit matching torch |
| pip build of simple-knn | `No module named 'torch'` | pip's isolated build env has no torch |
| First Option A | `cub/cub.cuh: No such file or directory`; nvcc 13.3 used | CUDA 13 moved CCCL to `include/cccl`; unpinned nvcc picked up |
| Second Option A | `cuda_runtime.h: No such file or directory` | nvcc in the venv prefix, cudart in the system prefix; `CUDA_HOME` pointed at the venv prefix only. Fixed by the merged `CUDA_HOME` |
| Option B (torch cu128 + CUDA 12.8 in the venv) | first `expected CUDA 12.8, got 13.0`, then `nvcc not found` | torch cu130 sits outside the venv (fixed by installing cu128 into the venv with `--ignore-installed`; torch cu128 then loaded). But PyPI has no CUDA 12.8 nvcc: `nvidia-cuda-nvcc-cu12` 12.8.93 ships only ptxas, crt headers and libnvvm. Option B cannot work with pip alone and is not used |
| R2-Gaussian `train.py` | `No module named 'simple_knn'` | `pip install -e` built the extension but did not put it on the import path. Fixed by the import check and `editable_compat` |

Option B code is still in the notebook. It starts only if an Option A
compile fails, and then stops with the reason above and the plan fallback.
The choice is kept in `~/flowsplat_work/cuda_option.json`; delete it (or set
`FLOWSPLAT_CUDA_OPTION=A`) to go back to Option A.

### Run the ImageCAS vessel reconstruction (Step 0.1 DoD)

1. Upload one ImageCAS label file (`<id>.label.nii.gz`, a few MB) to molab,
   e.g. to `~/flowsplat_work/imagecas/`.
2. Launcher cell: `python notebooks/step0_imagecas_recon.py` from the repo
   root, with `IMAGECAS_LABEL=<path>` in the environment (or leave it unset
   to use the first label under `~/flowsplat_work/imagecas/`).
3. In a fresh session it first rebuilds the extensions with
   `FLOWSPLAT_BUILD_ONLY=1 python notebooks/step0_setup.py` (about 4 min),
   then generates projections, trains (about 13 min) and writes
   `results/step0.1_imagecas.json` and `results/step0.1_imagecas.png`.

### Steps 1.1 to 3.1 (data, geometry init, flow fit, sweeps)

| Step | Where | Command / notebook | Output |
|---|---|---|---|
| 1.1 cases | M4 | `PYTHONPATH=src python3 -m flowsplat.data_gen --imagecas_dir <labels> --out data/cases_v1 --n_geom 40 --per_geom 5 --seed 0` | `data/cases_v1/` (2.4 min) |
| 1.2 bolus + pack | M4 | `PYTHONPATH=src python3 -m flowsplat.bolus_sim --cases data/cases_v1 --out data/bolus_v1 --seed 0 --pack data/step1_pack.zip` | `data/bolus_v1/`, `data/step1_pack.zip` (9 MB, 13 min) |
| 1.2 render | molab | upload the pack, `python notebooks/step1_render.py` (`FLOWSPLAT_RENDER_LIMIT=N` for a trial) | `~/flowsplat_work/data/render_v1/` (200 x 133 frames, 2.86 GB, 25 min) |
| 2.1 geometry init (final, Gate 1) | molab | `FLOWSPLAT_INIT=4drgs python notebooks/step2_geom_init.py` (`fdk`, `r2g`: earlier routes); calibration: `FLOWSPLAT_SPLIT=train FLOWSPLAT_N_CASES=15` | `results/step2.1_geom_init_v<views>_4drgs*.json`, `~/flowsplat_work/step2_1/<case>_v<views>_4drgs*/` |
| 2.2 flow fit (final, Gate 1) | molab | `FLOWSPLAT_INIT=4drgs FLOWSPLAT_MODE=val20 FLOWSPLAT_VARIANTS=frozen FLOWSPLAT_R_SCALE=<k> python notebooks/step2_train.py` | `results/step2.2_val20_v<views>_4drgs_cal*.json`, `~/flowsplat_work/step2_2/*/estimate.npz` |
| 3.1 sweeps | molab | `FLOWSPLAT_SWEEPS=<settings or groups> python notebooks/step3_sweeps.py` (builds what is missing, skips finished settings) | `results/step3.1_sweeps.json`, `results/step2.2_sweep_*.json` |
| 3.1 FFR (run 8) | M4 | `flowsplat.metrics.ffr_case` on the run 8 estimates (data/step2.2_run8_k15) | `results/step3.1_ffr_val20_v60.json` |
| 0.1 4DRGS only | molab | `FLOWSPLAT_ONLY_4DRGS=1 python notebooks/step0_setup.py`; dependencies only: `FLOWSPLAT_4DRGS_BUILD_ONLY=1` | `results/step0.1_4drgs.json` (nothing saved in build-only mode) |

Flags shared by the Step 2 notebooks (Step 3.1): `FLOWSPLAT_DATA_DIR`,
`FLOWSPLAT_RENDER_DIR` (sweep data folders) and `FLOWSPLAT_RUN_TAG`
(suffix of work folders and result files).

molab storage does not survive a session reset. After rendering, download
`render_v1` in 8 parts of about 360 MB (5 patients each) with
`mo.download` and unzip into `data/render_v1/` on the M4 (gitignored);
re-upload what a later notebook needs.

### 4DRGS build notes (resolved Oct 4)

The 4DRGS rasterizers fail on the first plain build with
`namespace "std" has no member "uintptr_t"` (GCC 14, missing `<cstdint>`);
the setup notebook retries with `NVCC_APPEND_FLAGS="-include cfloat -include
cstdint"` (flag only), which works. These first-attempt errors therefore
appear in every fresh-session log and are expected. LEAP is built with
Ninja; 4DRGS's final mesh / ICP evaluation cannot run on molab (no
libOpenGL, no reference mesh) and is not used.
