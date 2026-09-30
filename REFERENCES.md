# REFERENCES

Traceability table required by rule 4. One row per function, equation or
design choice. Rows marked `(planned)` point to the module named in the plan's
Repo field; replace with `file.py:function` once the code exists.
Section and equation numbers below were checked against the PDFs in
docs/papers on Sep 29, 2026, unless marked UNVERIFIED.

## Sources

| Key | Paper | Code |
|---|---|---|
| 4DRGS | Liu, Zha, Zhao, Li, Cui. 4DRGS: 4D Radiative Gaussian Splatting for Efficient 3D Vessel Reconstruction from Sparse-View Dynamic DSA Images. MICCAI 2025. arXiv:2412.12919v2 | No code link in the PDF. Availability UNVERIFIED (check in Step 0.1) |
| R2G | Zha et al. R2-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction. NeurIPS 2024. arXiv:2405.20693 | No link found in extracted PDF text. Repo URL UNVERIFIED |
| PUNCH | Thakur et al. PUNCH: Physics-informed Uncertainty-aware Network for Coronary Hemodynamics. arXiv:2601.17192, 2026 | No code link in the PDF |
| TOGS | Zhang et al. TOGS: Gaussian Splatting with Temporal Opacity Offset for Real-Time 4D DSA Rendering. IEEE JBHI 2025. arXiv:2403.19586 | https://github.com/hustvl/TOGS (stated in abstract) |
| TIGRE | Biguri et al. TIGRE: a MATLAB-GPU toolbox for CBCT image reconstruction. Biomed. Phys. Eng. Express, 2016 | PDF not in docs/papers; sections UNVERIFIED |
| ImageCAS | Zeng et al. ImageCAS: A large-scale dataset and benchmark for coronary artery segmentation. CMIG, 2023 | PDF not in docs/papers; sections UNVERIFIED |
| YoungTsai | Young, Tsai. Flow characteristics in models of arterial stenoses. J Biomech, 1973 | Citation UNVERIFIED (flagged in plan) |
| Murray | Murray. The physiological principle of minimum work. PNAS, 1926 | Citation UNVERIFIED (flagged in plan) |

## Traceability table

| Code location | Paper | Section / equation | What we took | What we changed |
|---|---|---|---|---|
| **4DRGS** | | | | |
| render_tigre.py (planned) | 4DRGS | Sec 2 "DSA Imaging" | Log-subtracted DSA: I_j = ln(I^m_j) - ln(I^f_j) | Synthetic: we render the contrast-only volume directly, so no mask/fill pair is simulated |
| render_tigre.py, data split (planned) | 4DRGS | Sec 2 "Sparse-View DSA Reconstruction" | Timestamps t_j = j/T; uniform view subset j_k = floor((k-1) T/N) + 1 | Timestamps in seconds (needed for Q), not normalized order |
| render_tigre.py (planned) | 4DRGS | Sec 4.1 "Dataset" | Protocol: 133 frames over 198 deg, SOD 750 mm, SDD 1200 mm | Views subsampled to 30 / 60 / 133 (4DRGS uses 30/40/50/60) per plan |
| model.py (planned) | 4DRGS | Sec 3.1, Eq 1 and Eq 2 | Kernel G_i(x,t) = rho(p_i,t) exp(-0.5 (x-p_i)^T Sigma_i^-1 (x-p_i)); mu(x,t) = sum_i G_i | rho(p_i,t) replaced by mu_I c_i(t) from transport (see OURS rows) |
| model.py (planned) | 4DRGS | Sec 3.1, Eq 3; Sec 4.1 "Implementation Details" | Bounded scaling s_i = (s_max - s_min) sigmoid(s~_i) + s_min; s_min = 0.1, s_max = 10 times voxel spacing | None planned |
| (replaced, baseline 1 only) | 4DRGS | Sec 3.1, Eq 4 | DNAF: hash encodings h3D, h4D + 2-layer MLP width 64 | Removed in FlowSplat; kept only inside baseline 1 |
| model.py (planned) | 4DRGS | Sec 3.1 "Model Initialization" | FDK init: M = 30k kernels, threshold delta = 0.016, scales = NN distance, identity rotation | UNVERIFIED whether 4DRGS or R2G values fit our synthetic intensities; decide in Step 2.1 |
| train.py (planned) | 4DRGS | Sec 3.2, Eq 6 | L = (1 - lambda_ssim) L1 + lambda_ssim L_ssim, lambda_ssim = 0.2, temporal perturbation tau ~ N(0, w^2) | Temporal perturbation to be kept or dropped in Step 2.2 (UNVERIFIED that it helps with a physical c(t)) |
| train.py (planned) | 4DRGS | Sec 3.2, Eq 7; Sec 4.1 | Accumulated attenuation pruning A_i = mean over iters of rho(p_i,t), prune if A_i < eps = 1e-6 | rho replaced by mu_I c_i(t) |
| train.py (planned) | 4DRGS | Sec 4.1 "Implementation Details" | Adam, 30k iters (fast: 10k); adaptive control 600 to 15k every 200, grad thr 1e-4; lr pos 1e-4, rot 1e-3, scale 5e-3, decay to 0.1x | Q and C_in learning rates are ours (none in 4DRGS) |
| metrics.py (planned) | 4DRGS | Sec 3.3; Sec 4.1 "Evaluation Metrics" | Time-averaged voxelization V = (1/T) sum_j V(t_j); marching cubes thr 0.025 ref / 0.008 recon; ICP; CD, HD (mm); PSNR/SSIM on held-out views | With synthetic GT we compare against GT mesh directly (no ICP needed) |
| **R2-Gaussian** | | | | |
| model.py (planned) | R2G | Sec 4.1, Eq 3 and Eq 4 | Radiative Gaussian: central density, position, covariance Sigma = R S S^T R^T; density = sum of kernels | Central density becomes time dependent via transport |
| rasterizer (reuse, planned) | R2G | Sec 3.1 Eq 1; Sec 4.2.1 Eq 5 to Eq 7 | Log-space line integral; X-ray rasterization by summing 2D Gaussians with rectified density rho^_i = (2 pi abs(Sigma~_i)/abs(Sigma^_i))^(1/2) rho_i | None: reuse CUDA code as is |
| voxelizer (reuse, planned) | R2G | Sec 4.2.2 | CUDA voxelizer, 8x8x8 tiles, 99% confidence culling | None |
| model.py (planned) | R2G | Sec 4.1 "Initialization" | FDK init, M = 50k, tau = 0.05, density scale k = 0.15 | See 4DRGS init row |
| train.py (planned) | R2G | Sec 4.2.3, Eq 8; Sec 5.1 | L1 + lambda_ssim D-SSIM + lambda_tv 3D TV; lambda_ssim = 0.25, lambda_tv = 0.05, TV volume D = 32; densify halves densities | UNVERIFIED whether TV is kept (4DRGS Eq 6 has no TV) |
| **PUNCH** | | | | |
| bolus_sim.py (planned) | PUNCH | Materials and Methods, Eq 1 and Eq 2 | 1D mass conservation of blood and contrast in a tube of area A(s,t) | Steady flow: Q = uA constant along a branch |
| bolus_sim.py (planned) | PUNCH | Materials and Methods, Eq 3 | 1D advection-diffusion dI/dt + u dI/ds - D d2I/ds2 = 0 as the simulator (richer than the fitted model) | u(s) = Q/A(s) from our geometry instead of random profiles |
| bolus_sim.py (planned) | PUNCH | Materials and Methods, "Synthetic Data Generation" | Method-of-lines, finite-volume advection + finite-difference diffusion, explicit Runge-Kutta; finite-duration inlet pulse; D over one order of magnitude; noise types (shot, blur, banding, dropout, temporal downsampling) | Noise applied to projections, not to a kymograph |
| data_gen.py (planned) | PUNCH | Materials and Methods, "Synthetic Data Generation" | Rest velocity 10 to 20 cm/s; 25% atypical 20 to 40 cm/s; CFR 1.5 to 4.0 (low-CFR 1.0 to 2.0); vessel 6 to 10 cm; sim time 3 to 6 s | Plan uses hyperemia up to about 0.4 m/s and full LAD about 100 mm |
| transport.py (planned) | PUNCH | Materials and Methods, text after Eq 2 | Pe = uL/D about 4000 (u about 0.04 m/s, L about 0.1 m, D about 1e-6 m^2/s); diffusion only matters in fronts of length about d/Pe^(1/2) | Used to justify dropping D in the fitted model (plug flow). Note: at plan rest velocity 0.15 m/s the same formula gives larger Pe, so the argument holds more strongly |
| baselines/ (planned) | PUNCH | Materials and Methods, Fig 2 | Kymograph along 2D centerline, slope ds/dt approximates u | Baseline 2: fit on 2D projected kymograph to expose foreshortening |
| paper/ | PUNCH | Discussion, "Limitations" | States bifurcations, overlap, multi-branch as future work | Motivates contribution 3 |
| **TOGS** | | | | |
| paper/, baselines/ (planned) | TOGS | Sec IV.B, Eq 6; Sec IV.C, Eq 7 | Opacity offset table (length 5, linear interp); random pruning | Related work only; 4DRGS reports TOGS as weaker baseline |
| paper/ | TOGS | Sec II.B | Notes time-density curves enable flow quantification but does not do it | Motivation for FlowSplat |
| **TIGRE** | | | | |
| render_tigre.py (planned) | TIGRE | UNVERIFIED | Cone-beam forward projector; FDK for init volume | None planned |
| **ImageCAS** | | | | |
| data_gen.py (planned) | ImageCAS | UNVERIFIED | Expert coronary masks for LAD centerline + radius (existing centerline.py) | Synthetic stenoses 30 to 70% inserted |
| **Reduced-order physics** | | | | |
| ffr.py (planned) | YoungTsai | UNVERIFIED | Delta P = (8 pi mu L A_n/A_s^2) V + (rho K_t/2)(A_n/A_s - 1)^2 V^2; K_t = 1.52 | None. Plan worked example re-checked numerically: 422 Pa + 1305 Pa, 12.96 mmHg, FFR 0.856 |
| tree.py (planned) | Murray | UNVERIFIED | r_p^3 = r_1^3 + r_2^3 | Used as soft prior, not hard constraint |
| **OURS** | | | | |
| transport.py (planned) | Ref: OURS | Plan, Approach | c_i(t) = C_in(t - tau_i), tau_i = int_0^{s_i} A(s)/Q ds | Contribution 1: plug-flow limit of PUNCH Eq 3 (D to 0) placed inside Gaussians instead of a free time network |
| model.py (planned) | Ref: OURS | Plan, Approach | A = 2 pi sigma^2, R = sqrt(2) sigma from matching projected mass mu_0 2 pi sigma^2 = mu_0 pi R^2 | Contribution 2: lumen area read from Gaussian scale |
| transport.py (planned) | UNVERIFIED | Plan, Approach | Gamma-variate inlet curve C_in(t) | Source for gamma-variate bolus model not yet identified |
| tree.py (planned) | Ref: OURS | Plan, Bifurcation | Hard Q_p = Q_1 + Q_2 (mass conservation) in the Gaussian model | Contribution 3: per-branch Q with continuity |
