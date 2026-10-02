# Step 0.1: GPU environment and code base

Dates: Sep 30 to Oct 4, 2026. Status: **in progress** (notebook written on
Oct 1; molab run not done yet). Result fields are `TODO` until the molab
output (`results/step0.1.json`) is pasted back.

## Scope of this pass

Decided by the project owner on Oct 1:

| Plan bullet | This pass |
|---|---|
| Run R2-Gaussian on a CUDA GPU, reproduce one static reconstruction | Notebook written, run pending |
| Check whether 4DRGS code is public; if yes, run it on one case | Public: yes. Run is in the notebook, pending |
| Confirm TIGRE on the same GPU environment | In the notebook, pending |
| OpenReview profile | Skipped for now |
| Data-request emails to 4DRGS and PUNCH authors | Skipped for now |
| CADICA download | Skipped for now |

## Where things run

| Piece | Runs on |
|---|---|
| `notebooks/step0_setup.py` | molab GPU |
| `src/flowsplat/setup_check.py`, `tests/test_setup_check.py` | M4 (CPU), 9 tests pass |

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

Commit hashes confirmed by the notebook at run time: TODO.

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
| nvcc | Not in the molab image. Installed from pip (`cuda-toolkit[nvcc,nvvm,crt,cccl,cudart]`, same version as torch's). Version: 13.0 (second molab run, Oct 2) |
| Python | 3.13.11 |
| Driver | 595.71.05 |
| GCC | 14.2.0 (Debian) |

Values from the first molab run on Oct 2, 2026 (repo at `d517294`).

## Debug log

| Date | Problem | Action | Solved |
|---|---|---|---|
| Oct 2 | `nvcc not found`: the molab image has torch with pip CUDA libraries but no CUDA toolkit | Notebook now installs the matching compiler from NVIDIA's pip wheels, uses `<site-packages>/nvidia/cu13` as `CUDA_HOME`, and adds the linker symlinks (`libcudart.so`, `lib64`) inside that pip prefix | Yes (second run: nvcc 13.0 found, sm_120 supported) |
| Oct 2 | simple-knn build: `ModuleNotFoundError: No module named 'torch'` inside pip's isolated build env (setup.py imports torch) | Every CUDA extension install (simple-knn, xray rasterizer/voxelizer, TIGRE, tiny-cuda-nn, 4DRGS submodules) now uses `pip install --no-build-isolation`, so it builds against the installed torch 2.11.0+cu130. Build flag only | TODO (rerun pending). Time box ends Oct 4 |

## R2-Gaussian reproduction (its own example case)

Case `0_chest_cone`, 50 views, default settings, 30k iterations (README
Sec 3.2 command).

| Metric | Ours | Expected | Source of expected |
|---|---|---|---|
| PSNR 3D (dB) | TODO | 35.09 | R2-Gaussian repo, `assets/results.md`, table "Synthetic dataset", row Chest, column "R2-Gaussian (50-view)" |
| SSIM 3D | TODO | no per-case value published | For orientation only: 15-case average 0.952 (paper Table 1, synthetic, 50-view, "Ours (iter=30k)") |
| PSNR 2D (dB) | TODO | not published | |
| SSIM 2D | TODO | not published | |
| Training time (min) | TODO | 5 to 15 on an RTX 3090 | README Sec 3.2 |

Pass rule: abs(ours minus expected) <= 1 dB on PSNR 3D. Pass: TODO.

## TIGRE check

Uniform 64^3 cube (mu = 1), default cone geometry, 4 angles. Central detector
value must equal the cube side length `sVoxel` within 2%.

| Item | Value |
|---|---|
| TIGRE version | TODO |
| Projection shape | TODO |
| Central value vs expected | TODO |
| Pass | TODO |

## 4DRGS on one case

Case `case2`, 30 views, fast setting from the README (10k iterations, ADC
until 5k), 2D rendering metrics on held-out views.

| Metric | Ours | Expected | Source of expected |
|---|---|---|---|
| Eval PSNR (dB) | TODO | 36.349 | 4DRGS README, table under "Sort-free X-ray rasterization", case2, 30 views, "10k / 5k", Sort-free (authors' RTX 4060 Ti) |
| Eval SSIM | TODO | 0.9021 | same row |
| Ran without error | TODO | | |

## Changes to the original code

Source edits to R2-Gaussian, TIGRE, 4DRGS: **none planned**. The notebook
checks `git diff` of every repo after the run and stores the result
(`source_untouched`): TODO.

Deviations from the READMEs that are not source edits (planned; the final
list comes from `deviations` in the JSON):

| Deviation | Instead of | Reason |
|---|---|---|
| Python, torch and CUDA versions of the molab image (TODO exact) | R2-Gaussian README: Python 3.9, torch 2.1.2+cu118; pinned `requirements.txt` | The Blackwell GPU is not supported by those old builds. Same packages, unpinned |
| TIGRE at commit `6b0951a` (v3.1.3 line) | R2-Gaussian README: TIGRE v2.3 zip | Rule: pin the official repo to a commit; v3.1.x reads the GPU architectures from nvcc at build time |
| CUDA compiler from pip (`cuda-toolkit` extras), `CUDA_HOME` inside site-packages, linker symlinks added there | A system CUDA toolkit | molab has no `nvcc`. Same CUDA version as torch. No repo file touched |
| `pip install --no-build-isolation` for every CUDA extension (simple-knn, xray rasterizer/voxelizer, TIGRE, tiny-cuda-nn, 4DRGS submodules) | Plain `pip install` / `pip install -e` | Isolated build env has no torch, so setup.py fails to import it. Build flag only; no source change |
| `NVCC_APPEND_FLAGS="-include cfloat -include cstdint"`, only if the plain build fails | Plain `pip install -e` | Build flag only, for newer GCC. Used: TODO |
| 4DRGS: `--fdk_toolbox=tigre`; test with `--render_2d` only | Default `leap`; README test also has `--render_fixview --VQR` | TIGRE is already built; CLI flags only. Whether the README numbers used LEAP or TIGRE is UNVERIFIED |

## DoD

Plan DoD: "Static reconstruction of one ImageCAS vessel with R²-Gaussian
renders correctly; OpenReview profile active; data-request emails sent."

| DoD part | Met |
|---|---|
| R2-Gaussian builds and reproduces its example case on molab | TODO |
| Static reconstruction of one ImageCAS vessel renders correctly | No (not part of this pass) |
| OpenReview profile active | No (skipped for now) |
| Data-request emails sent | No (skipped for now) |

**DoD met: no (in progress).**

**Fallback used: no.** Plan fallback, to apply only if the build fails: "If
the CUDA rasterizer will not build: another cloud GPU. If nothing works by
Oct 4: plain PyTorch Gaussian splatter (slower, no custom CUDA). Do not
change the project."

## Not verified yet

- The notebook has not been executed. On the M4 it was only parsed and its
  cell inputs and outputs were checked.
- Whether R2-Gaussian and 4DRGS run unchanged with a current numpy / torch.
- Whether the pip CUDA compiler builds the extensions (layout checked from
  the wheel file lists only).
- Whether every Python dependency has a wheel for Python 3.13 (open3d is the
  one to watch). Failures are listed in `build.pip_failed`.
- Size of the R2-Gaussian dataset zip (all 15 cases are in one file; only
  `0_chest_cone` is extracted).
