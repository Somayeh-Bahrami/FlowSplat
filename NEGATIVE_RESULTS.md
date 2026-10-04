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
