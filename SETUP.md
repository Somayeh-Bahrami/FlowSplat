# Setup

How to get the GPU environment for FlowSplat running. Verified on molab on
Oct 3, 2026 (repo `11a482a`): R2-Gaussian builds and reproduces its example
case (PSNR 3D 35.146 vs 35.09 dB), TIGRE forward projection passes. Full
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

### Known open issue

4DRGS: `diff-Xray-gaussian-rasterization-voxelization-sortfree` fails with
`namespace "std" has no member "uintptr_t"` (missing `<cstdint>`). The
4DRGS cell does not yet use the `NVCC_APPEND_FLAGS` retry of step 7.
tiny-cuda-nn builds fine.
