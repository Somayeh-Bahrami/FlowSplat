# Step 0.1: GPU environment and code base

Dates: Sep 30 to Oct 4, 2026. Status: **done, DoD met** (Oct 3, 2026).
ImageCAS vessel run: Oct 3, 17:57 UTC, record `results/step0.1_imagecas.json`. Results from the molab run of Oct 3, 2026,
05:23 UTC, repo at `11a482a`, exit code 0. Raw record:
`results/step0.1.json`.

## Scope of this pass

Decided by the project owner on Oct 1:

| Plan bullet | This pass |
|---|---|
| Run R2-Gaussian on a CUDA GPU, reproduce one static reconstruction | Done: `0_chest_cone`, PSNR 3D within 0.06 dB of published |
| Check whether 4DRGS code is public; if yes, run it on one case | Public: yes. Run: not done yet (submodule build failed on missing <cstdint>; retry with the build flag added Oct 4, run with FLOWSPLAT_ONLY_4DRGS=1 pending) |
| Confirm TIGRE on the same GPU environment | Done: forward projection check passes |
| OpenReview profile | Done (owner, Oct 3) |
| Data-request emails to 4DRGS and PUNCH authors | Done (owner, Oct 3) |
| CADICA download | Done (owner, Oct 3) |

## Where things run

| Piece | Runs on |
|---|---|
| `notebooks/step0_setup.py` | molab GPU |
| `notebooks/step0_imagecas_recon.py` | molab GPU |
| `src/flowsplat/setup_check.py`, `src/flowsplat/imagecas_volume.py`, `tests/` | M4 (CPU), 25 tests pass |

The notebook uses the repo it already sits in (molab bootstrap case) and
skips the clone. Only when it is not inside the repo does it clone, with the
token from `GH_TOKEN` if set, else from the password field. Run as a script
(`python notebooks/step0_setup.py`) it starts without the button.

## Pinned commits

| Repo | URL | Commit | Note |
|---|---|---|---|
| R2-Gaussian | https://github.com/Ruyi-Zha/r2_gaussian | `f2579bfddd9aac009cb797c8503bef8119bbd022` | `main`, committed 2025-10-10 |
| TIGRE | https://github.com/CERN/TIGRE | `6b0951a8a88aa88a7db5b2a95181a33a95aa8b28` | `master`, committed 2026-09-29, one commit after tag v3.1.3 line |
| 4DRGS | https://github.com/ShanghaiTech-IMPACT/4DRGS | `38006edbb105b363ae2ed6b075e1766ec70a91dc` | committed 2026-08-24 |
| tiny-cuda-nn (4DRGS dependency) | https://github.com/NVlabs/tiny-cuda-nn | `3daa6e5b8b8e0fa17971a1a20acfbb3754494b38` | default branch head on Oct 1, 2026 |

Commit hashes confirmed by the notebook at run time: yes, all four (HEAD equals the pin; `pins` in the JSON).

## Code availability (checked Oct 1, 2026)

| Code | Public | Where | Example data |
|---|---|---|---|
| R2-Gaussian | Yes | URL above | Google Drive link in README Sec 2; `synthetic_dataset/cone_ntrain_50_angle_360.zip` |
| TIGRE | Yes | URL above | None needed |
| 4DRGS | Yes | URL above (linked from the R2-Gaussian README) | `case2` from the paper, Google Drive link in README "Data-Preparation" |
| TOGS | Yes | https://github.com/hustvl/TOGS | Not checked (not needed for Step 0.1) |
| PUNCH | No code link in the PDF | | |

Note: the 4DRGS repo and its BibTeX give the venue as IPMI 2025 (Oral).
REFERENCES.md said MICCAI 2025; corrected.

## Environment

| Item | Value |
|---|---|
| GPU name | NVIDIA RTX PRO 6000 Blackwell Server Edition (95.0 GB) |
| Compute capability | 12.0 (sm_120, in torch's arch list) |
| torch | 2.11.0+cu130 |
| CUDA (torch build) | 13.0, cuDNN 9.19 |
| nvcc | 13.0 (V13.0.88), from pip `cuda-toolkit[nvcc,nvvm,crt,cccl,cudart]==13.0.*` into the venv. The image only has nvcc 13.3 (system site-packages), rejected because torch is cu130 |
| CUDA_HOME | `~/flowsplat_work/cuda-13.0`: symlink tree of the venv `nvidia/cu13` (nvcc, crt, nvvm, cccl) and the system `nvidia/cu13` (cudart) |
| CCCL include | `<CUDA_HOME>/include/cccl` (via `CPATH` and `NVCC_APPEND_FLAGS`) |
| ninja | 1.13.2 |
| Python env | uv venv `/tmp/uv-venv` on top of the image's system site-packages (`/usr/local/lib/python3.13`); torch is loaded from the system copy |
| Python | 3.13.11 |
| Driver | 595.71.05 |
| GCC | 14.2.0 (Debian) |

Values from the molab run of Oct 3, 2026 (repo at `11a482a`). CUDA option used: **A** (image torch cu130 + CUDA 13.0 toolkit). Option B was not used.

## Debug log

| Date | Problem | Action | Solved |
|---|---|---|---|
| Oct 2 | `nvcc not found`: the molab image has torch with pip CUDA libraries but no CUDA toolkit | Notebook now installs the matching compiler from NVIDIA's pip wheels, uses `<site-packages>/nvidia/cu13` as `CUDA_HOME`, and adds the linker symlinks (`libcudart.so`, `lib64`) inside that pip prefix | Yes (second run: nvcc 13.0 found, sm_120 supported) |
| Oct 2 | simple-knn build: `ModuleNotFoundError: No module named 'torch'` inside pip's isolated build env (setup.py imports torch) | Every CUDA extension install (simple-knn, xray rasterizer/voxelizer, TIGRE, tiny-cuda-nn, 4DRGS submodules) now uses `pip install --no-build-isolation`, so it builds against the installed torch 2.11.0+cu130. Build flag only | Yes (third run: torch import fixed) |
| Oct 2 | simple-knn: `cub/cub.cuh: No such file or directory`. Also: nvcc 13.3 used (pip prefix in /usr/local site-packages) while torch is cu130 (earlier run: nvcc 13.0 from /tmp/uv-venv); ninja missing | (1) ONE `CUDA_HOME` for every build, nvcc release must equal torch's CUDA (any other nvcc is rejected, pip toolkit pinned to `==<torch CUDA>.*`). (2) ninja installed. (3) Option A: CCCL dir (`include/cccl` in CUDA 13) found and added via `CPATH` and `NVCC_APPEND_FLAGS=-I...`. (4) Option B, automatic if any A build fails: torch + torchvision cu128 and CUDA 12.8 pip toolkit (nvcc, runtime, cccl), state in `WORK/cuda_option.json`, script restarts itself | Option A: No. First compiler error: `<command-line>: fatal error: cuda_runtime.h: No such file or directory` (run of Oct 3, repo `eab0468`). Option B: see next rows |
| Oct 2 | Option B: `Option B expected torch with CUDA 12.8, got 13.0`. torch 2.11.0+cu130 is in the system site-packages (`/usr/local/lib/python3.13`), outside the uv venv (`/tmp/uv-venv`); pip could not uninstall it and reported cu128 torch as already satisfied | torch + torchvision cu128 installed INTO the venv with `--ignore-installed` (deps too), venv site-packages put first on `sys.path` and `PYTHONPATH`, then the process restarts. Checks: `torch.version.cuda == 12.8`, `sm_120` in the arch list; `sys.path` and `torch.__file__` printed and saved. CUDA 12.8 toolkit merged from venv wheels only | Shadowing: Yes (torch 2.11.0+cu128 loaded from `/tmp/uv-venv/...`, sm_120 in arch list). Build: No, see next row |
| Oct 3 | (a) Option A root cause: nvcc 13.0 was installed into the venv's `nvidia/cu13`, but `nvidia-cuda-runtime` 13.0.96 was already satisfied in the system `nvidia/cu13`, so `cuda_runtime.h` was not under `CUDA_HOME`. A tooling problem, not a CUDA 13 API error. (b) Option B: `nvcc not found`. File list of `nvidia-cuda-nvcc-cu12` 12.8.93 checked: only `ptxas`, crt headers, libnvvm; no nvcc binary on PyPI for CUDA 12.8 | (a) `CUDA_HOME` is now one symlink tree `WORK/cuda-13.0`: venv `nvidia/cu13` first (nvcc 13.0, crt, nvvm, cccl), then system `nvidia/cu13` (cudart headers and libs). (b) Option B now stops with this reason and the plan fallback | Yes (Oct 3 rerun: Option A builds pass) |
| Oct 3 | Builds pass under Option A, but R2-Gaussian `train.py` fails: `ModuleNotFoundError: No module named 'simple_knn'` after `pip install -e` (README Sec 1). Likely cause: simple-knn's setup.py builds `simple_knn._C` without listing a package, so the setuptools editable finder does not map it (UNVERIFIED) | After each build, `import torch, <module>` is checked in a fresh interpreter. If it fails: reinstall with `-e --config-settings editable_mode=compat`, then a regular install; the mode used is logged as a deviation. Import failures no longer trigger the Option A to B switch | Yes (Oct 3: simple-knn installed as `editable_compat`, R2-Gaussian trained) |
| Oct 3 | 4DRGS: `diff-Xray-gaussian-rasterization-voxelization-sortfree` fails: `rasterizer_impl.h(24): error: namespace "std" has no member "uintptr_t"` (same missing `<cstdint>` as the R2-Gaussian rasterizer) | Not fixed. The 4DRGS cell has no `NVCC_APPEND_FLAGS="-include cfloat -include cstdint"` retry; the R2-Gaussian cell does and it worked there | No (open) |

## R2-Gaussian reproduction (its own example case)

Case `0_chest_cone`, 50 views, default settings, 30k iterations (README
Sec 3.2 command).

| Metric | Ours | Expected | Source of expected |
|---|---|---|---|
| PSNR 3D (dB) | **35.146** | 35.09 | R2-Gaussian repo, `assets/results.md`, table "Synthetic dataset", row Chest, column "R2-Gaussian (50-view)" |
| SSIM 3D | 0.935 | no per-case value published | For orientation only: 15-case average 0.952 (paper Table 1, synthetic, 50-view, "Ours (iter=30k)") |
| PSNR 2D (dB) | 45.261 (eval yml); stdout at iter 30000 says 49.553. Which split each number is: UNVERIFIED | not published | |
| SSIM 2D | 0.983 (eval yml); stdout says 0.985 | not published | |
| Training time (min) | 13.0 | 5 to 15 on an RTX 3090 | README Sec 3.2 |

Pass rule: abs(ours minus expected) <= 1 dB on PSNR 3D. Difference: +0.056 dB. **Pass: yes.**

## TIGRE check

Uniform 64^3 cube (mu = 1), default cone geometry, 4 angles. Central detector
value must equal the cube side length `sVoxel` within 2%.

| Item | Value |
|---|---|
| TIGRE version | not reported (`tigre.__version__` is missing); commit `6b0951a` |
| Projection shape | [4, 64, 64] |
| Central value vs expected | 256.002 vs 256.0 (relative error 1e-5) |
| Forward time | 127.3 ms |
| Pass | **yes** |

## 4DRGS on one case

Case `case2`, 30 views, fast setting from the README (10k iterations, ADC
until 5k), 2D rendering metrics on held-out views.

| Metric | Ours | Expected | Source of expected |
|---|---|---|---|
| Eval PSNR (dB) | not run | 36.349 | 4DRGS README, table under "Sort-free X-ray rasterization", case2, 30 views, "10k / 5k", Sort-free (authors' RTX 4060 Ti) |
| Eval SSIM | not run | 0.9021 | same row |
| Ran without error | No. tiny-cuda-nn built; `diff-Xray-gaussian-rasterization-voxelization-sortfree` failed to compile (`std::uintptr_t`, `uint32_t` undefined, missing `<cstdint>`) | | |

## ImageCAS vessel (DoD item)

Notebook `notebooks/step0_imagecas_recon.py` (molab), helper
`src/flowsplat/imagecas_volume.py` (M4 tests: `tests/test_imagecas_volume.py`).

Pipeline: one ImageCAS label, largest connected component (one vessel tree),
cropped with 5 mm margin, turned into a 256^3 volume in [0, 1] with the
steps of R2-Gaussian `process_raw_data.py`. Projections, initialisation and
training with R2-Gaussian's own `generate_data.py` (`cone_beam.yml`, 50
train / 100 test views), `initialize_pcd.py` and `train.py`, all unchanged
(only `--n_points` lowered if the vessel has too few voxels).

Pass rule (Ref: OURS, no published value exists): training finishes,
prediction finite, Dice(pred > 0.5, GT > 0.5) >= 0.7. PSNR / SSIM from
R2-Gaussian are reported too.

| Item | Value |
|---|---|
| ImageCAS case | 11 (`11.label.nii.gz`, 512 x 512 x 275, spacing 0.33 x 0.33 x 0.5 mm); largest component 37,886 voxels |
| Cube side / voxel size | 94.05 mm / 0.367 mm (256^3) |
| PSNR 3D / SSIM 3D | 45.17 dB / 0.998 (inflated by the empty background; see Dice) |
| PSNR 2D / SSIM 2D, test views (100) | 47.38 dB / 0.874 |
| PSNR 2D / SSIM 2D, train views (50) | 44.90 dB / 0.958 |
| Dice (pred > 0.5 vs GT > 0.5) | **0.977** |
| Training time | 5.5 min (30k iterations) |
| Deviation | `initialize_pcd.py --n_points 26163` instead of 50000 (52,326 GT voxels above 0.05). CLI flag only |
| Figure | `results/step0.1_imagecas.png`: MIPs along x, y, z; prediction matches GT, differences only at the vessel wall |
| Pass | **yes** (Dice 0.977 >= 0.7, prediction finite) |

## Changes to the original code

Source edits to R2-Gaussian, TIGRE, 4DRGS: **none**. `git diff` after the
run (`source_untouched`): r2_gaussian true, tigre true, 4drgs true.

Deviations from the READMEs that are not source edits (final list in
`deviations` of the JSON):

| Deviation | Instead of | Reason |
|---|---|---|
| Python 3.13.11, torch 2.11.0+cu130 (CUDA 13.0) of the molab image | R2-Gaussian README: Python 3.9, torch 2.1.2+cu118; pinned `requirements.txt` | The Blackwell GPU is not supported by those old builds. Same packages, unpinned |
| TIGRE at commit `6b0951a` (v3.1.3 line) | R2-Gaussian README: TIGRE v2.3 zip | Rule: pin the official repo to a commit; v3.1.x reads the GPU architectures from nvcc at build time |
| CUDA compiler from pip (`cuda-toolkit` extras, 13.0), `CUDA_HOME` = symlink tree of venv and system `nvidia/cu13` | A system CUDA toolkit | molab has no matching `nvcc`; the cudart headers stay in the system prefix. Same CUDA version as torch. No repo file touched |
| `pip install --no-build-isolation` for every CUDA extension (simple-knn, xray rasterizer/voxelizer, TIGRE, tiny-cuda-nn, 4DRGS submodules) | Plain `pip install` / `pip install -e` | Isolated build env has no torch, so setup.py fails to import it. Build flag only; no source change |
| One `CUDA_HOME` with nvcc pinned to torch's CUDA version; ninja installed | Whatever nvcc is on PATH | A 13.3 nvcc was picked up against torch cu130. Tooling only |
| Option A: CCCL include dir (`<CUDA_HOME>/include/cccl`) via `CPATH` and `NVCC_APPEND_FLAGS=-I...` | Plain build | CUDA 13 moved cub/thrust to `include/cccl`. Build flag only. Used: **yes** |
| Option B (only if an A build fails): torch + torchvision cu128 installed into the venv with `--ignore-installed` (shadows the system torch), CUDA 12.8 pip toolkit, merged into one `CUDA_HOME` by symlinks | Image torch 2.11.0+cu130 in the system site-packages | Planned fallback for CUDA 13 incompatibilities. Used: **no**. Not usable with pip anyway: PyPI has no CUDA 12.8 nvcc binary |
| Install mode `editable_compat` or regular, only if `import` fails after `pip install -e` | `pip install -e` (R2-Gaussian README Sec 1) | Extension compiled but was not importable. Install mode only. Used: **yes**, simple-knn as `editable_compat` |
| `NVCC_APPEND_FLAGS="-include cfloat -include cstdint"`, only if the plain build fails | Plain `pip install -e` | Build flag only, for newer GCC. Used: **yes**, xray-gaussian-rasterization-voxelization (`std::uintptr_t` / `uint32_t` undefined without it) |
| 4DRGS: `--fdk_toolbox=tigre`; test with `--render_2d` only | Default `leap`; README test also has `--render_fixview --VQR` | TIGRE is already built; CLI flags only. Whether the README numbers used LEAP or TIGRE is UNVERIFIED |

## DoD

Plan DoD: "Static reconstruction of one ImageCAS vessel with R²-Gaussian
renders correctly; OpenReview profile active; data-request emails sent."

| DoD part | Met |
|---|---|
| R2-Gaussian builds and reproduces its example case on molab | **Yes** (PSNR 3D 35.146 vs 35.09 dB, +0.056 dB; TIGRE check passes) |
| Static reconstruction of one ImageCAS vessel renders correctly | **Yes** (case 11, Dice 0.977, PSNR 3D 45.17 dB, figure) |
| OpenReview profile active | **Yes** (owner, Oct 3) |
| Data-request emails sent | **Yes** (owner, Oct 3) |

**DoD met: yes.**

**Fallback used: no.** The rasterizer builds and runs on molab, so the
fallback below is not triggered (Oct 4 setup gate condition "CUDA rasterizer
does not run" is false). Plan fallback, for reference: "If the CUDA
rasterizer will not build: another cloud GPU. If nothing works by Oct 4:
plain PyTorch Gaussian splatter (slower, no custom CUDA). Do not change the
project."

## Still open

Not part of the DoD:

1. **4DRGS on case2 (plan bullet).** Its rasterizer submodule failed on the
   missing `<cstdint>`. The 4DRGS cell now retries with the same build flag
   as the R2-Gaussian rasterizer (Oct 4); run `FLOWSPLAT_ONLY_4DRGS=1 python
   notebooks/step0_setup.py` (skips the chest example, writes
   `results/step0.1_4drgs.json`). Needed by Step 4.1 (baseline 1, Oct 27).
2. Small: PSNR 2D differs between stdout (49.55) and the eval yml (45.26);
   which split each is: UNVERIFIED. TIGRE version string not recorded.

## Not verified

- Which 2D split the stored PSNR 2D / SSIM 2D come from (see above).
- simple-knn setup.py contents (reason the plain `-e` install was not
  importable): UNVERIFIED, gitlab.inria.fr blocked automated reading.
- Whether 4DRGS reproduces its README numbers on this GPU (not run).
