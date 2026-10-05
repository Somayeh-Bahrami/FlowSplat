# Negative Results Log

If something fails, write why here. Diagnosed failures become ablations or
limitations in the paper (plan, Focus Rules).

| Date | Step | What failed | Diagnosis | Action (fallback used) |
|---|---|---|---|---|
| Oct 2 | 0.1 | Option B (torch cu128 + CUDA 12.8 pip toolkit) | PyPI has no CUDA 12.8 nvcc: nvidia-cuda-nvcc-cu12 12.8.93 ships only ptxas, crt headers and libnvvm (file list checked) | Not used; Option A (image torch cu130 + pip CUDA 13.0, merged CUDA_HOME) works. reports/step0.1.md |
| Oct 2 | 0.1 | First Option A builds: cub/cub.cuh and cuda_runtime.h not found | CUDA 13 moved CCCL to include/cccl; nvcc in the venv prefix, cudart in the system prefix | CCCL include flag; one CUDA_HOME as a symlink tree of both prefixes (tooling, no source change) |
| Oct 3 | 1.1 | LAD root = widest skeleton endpoint (reused convention) | Picked distal stumps in 5 of the first 20 ImageCAS cases (QC sheet) | Root = most rightward LCA endpoint; QC adds proximal-radius and anterior checks; 15 of 55 labels rejected |
| Oct 3 | 1.1 | Young-Tsai FFR at fixed hyperemic flow | dP > P_a (FFR < 0) for severe lesions at 0.4 m/s, because a real lesion would reduce the flow | Flow redrawn until FFR >= 0.3 (6 % of cases); limitation of fixed-flow Young-Tsai for the paper |
| Oct 4 | 2.1 | Static R2-Gaussian on the dynamic views (plan route) | Radius MAE 0.20 to 0.42 mm (1 of 5 cases below 0.2 mm); 3D PSNR falls from 34.3 dB (FDK init) to 32.3 dB while fitting views taken at different times | Plan fallback: FDK volume skeleton (run 2: MAE 0.05 to 0.11 mm). Candidate ablation: static R2-Gaussian vs FDK init |
| Oct 4 | 2.1 | FDK skeleton with one threshold | Case 131_0 lost its dim proximal third, stenosis included (lesion coverage 0 %), invisible to the radius MAE | Hysteresis threshold; coverage, lesion coverage and throat error added to the evaluation |
| Oct 4 | 2.1 | Hysteresis threshold (0.2 seed, 0.1 grow) for the FDK skeleton (run 3) | Centerline followed FDK streak artifacts (121_0 MAE 0.05 -> 0.32 mm, coverage 100 -> 73 %); dim proximal 131_0 still lost | Reverted to the single 0.2 threshold; 131_0 coverage gap kept as a known limitation of the FDK init |
| Oct 4 | 0.1 | 4DRGS training on case2 (attempt 1) | Builds now pass (cstdint flag), but train.py imports LEAP unconditionally: No module named 'leapctype' | LEAP built from source as in the 4DRGS README (pinned 0c8846f); rerun pending |
| Oct 4 | 0.1 | 4DRGS attempts 3 to 5 | No `make` (CMake), missing user site folder (LEAP manual_install.py), missing libOpenGL for pymeshlab (mesh/ICP evaluation after training) | Ninja generator; create the folder; libopengl0 unpacked without root + save at 10k; training itself works (10k iterations, 1 min 51 s) |
| Oct 4 | 0.1 | 4DRGS attempt 6: libOpenGL for pymeshlab (apt-get download libopengl0) | No package lists and no root on molab: package not found; mesh / ICP evaluation still fails | 2D evaluation of the saved 10k model passes (36.26 vs 36.35 dB); mesh metrics deferred to Step 4.1 if needed |
| Oct 5 | 2.2 | Stage B on all geometry (plan route), overfit 121_0 | Image loss falls (0.0074 -> 0.0061) while the radius shrinks by about 0.3 mm everywhere: Gaussian cross-section with free amplitude prefers narrower tubes; stenosis diameter error 0.28 mm (DoD 0.2), Q 4.4 % | Plan fallback: keep geometry frozen longer, add arrival-time loss (variants frozen / pose / pose_arr) |
