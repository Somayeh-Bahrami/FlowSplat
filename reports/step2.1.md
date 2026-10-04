# Step 2.1: Geometry initialisation and tube-aligned Gaussians

Dates: Oct 12 to Oct 14, 2026 (started Oct 3). Status: **in progress**:
plan route failed the DoD (run 1); fallback (FDK skeleton) meets the DoD as
written (run 2) but missed one lesion; hysteresis fix failed (run 3) and is
reverted; run 4 (run-2 extraction with the 4DRGS view subsets) pending.

Plan DoD: "Lumen radius error below 0.2 mm on 5 validation cases."

## Where things run

| Piece | Runs on |
|---|---|
| `src/flowsplat/model.py` | M4 tests; molab |
| `notebooks/step2_geom_init.py` (R2-Gaussian training) | molab GPU |
| `tests/test_model.py` | M4 (CPU); whole suite 60 tests pass |

## What was done

1. **Static R2-Gaussian on the dynamic views** (plan: "time-averaged
   views"). Each validation case is written as an R2-Gaussian dataset with
   R2-Gaussian's own conventions (vertical flip, x/y/z transpose, radians,
   physical units rescaled by R2-Gaussian), projections divided by mu so the
   reconstruction is in concentration units, and trained with the official
   `initialize_pcd.py` and `train.py` at defaults (60 views by default).
   `vol_gt` = lumen concentration averaged over the training frames, so
   R2-Gaussian's own PSNR / SSIM also checks that the frames line up.
2. **Centerline** from the static volume: threshold (0.2 x p99.9), largest
   component, skeleton (owner's code), longest geodesic path.
3. **Radius** by the half-maximum of the cross-section profile (16 rays,
   median), scale-free, so the uneven time-averaged brightness along the
   vessel does not bias it.
4. **Tube-aligned Gaussians** (`TubeGaussians`): one Gaussian per 0.5 mm at
   arc length s_i, major axis along the tangent, sigma_r = R / sqrt(2)
   (plan), 4DRGS bounded scaling (s_min = 0.1, s_max = 10 voxel spacings),
   alignment regularizer mean(1 - (axis . tangent)^2), torch parameters for
   Step 2.2.

## Bugs found by the tests and fixed

| Problem | Fix |
|---|---|
| Centerline flipped for evaluation without flipping the radius array (MAE 0.29 mm on the synthetic tube) | `orient_by_reference` flips both (test added) |
| 2 mm Gaussian smoothing of R(s) flattened the stenosis | Replaced by a 5-sample median |
| Scales used exp (unbounded), not the 4DRGS bounded activation listed in the base papers | Bounded scaling implemented (test: sigma capped at 2.5 mm) |

Synthetic check (curved stenosed tube, partial volume, 3x brightness change
along s, 0.3 mm blur): radius MAE 0.035 mm, bias -0.035 mm, centerline
distance 0.15 mm.

## Results (molab)

### Run 1: static R2-Gaussian (plan route), Oct 4, repo `43a1eb5`, 60 views

| Case | Radius MAE (mm) | Bias (mm) | p90 (mm) | Centerline distance (mm) | Length est / GT (mm) | Pass |
|---|---|---|---|---|---|---|
| 121_0 | 0.230 | -0.163 | 0.62 | 0.73 | 157.0 / 150.0 | no |
| 122_0 | 0.199 | -0.110 | 0.41 | 0.98 | 175.0 / 162.5 | yes |
| 130_0 | 0.211 | -0.051 | 0.48 | 1.43 | 173.5 / 163.5 | no |
| 131_0 | 0.425 | -0.310 | 0.83 | 1.93 | 116.0 / 104.0 | no |
| 135_0 | 0.324 | -0.307 | 0.74 | 0.77 | 122.5 / 117.5 | no |

Mean MAE 0.278 mm; 1 of 5 below 0.2 mm. **DoD failed.** R2-Gaussian 13 to
26 min per case.

Diagnosis: for 135_0, R2-Gaussian's own 3D PSNR against the time-averaged
ground truth is 34.3 dB at iteration 1 (its FDK initialisation) and drops
to 32.3 dB at 30k while the 2D PSNR only reaches 28 dB. A static model
cannot explain views taken at different times; fitting them pulls the
volume away from the vessel (centerline 0.7 to 1.9 mm off the axis vs.
0.15 mm on the synthetic tube; radius biased low). A frame error would
show a poor PSNR from iteration 1, which is not the case. Figures and the
output zip of run 1 were recovered on Oct 4: `results/step2.1_geom_init_v60_r2g_run1.json`,
`results/step2.1_<case>_v60_r2g_run1.png`.

### Fallback applied (rule 2)

Plan, Step 2.1 Fallback: "If Gaussians do not align with the vessel:
initialize directly from the ground-truth-free FDK volume skeleton."

`FLOWSPLAT_INIT=fdk` (now the default): TIGRE FDK with a Hann filter (as
the 4DRGS FDK baseline, Sec 4.1) of the same 60 views, then the same
centerline and half-maximum radius.

### Run 2: FDK fallback, Oct 4, repo `866ab52`, 60 views

`results/step2.1_geom_init_v60_fdk_run2.json`, figures
`results/step2.1_<case>_v60_fdk_run2.png` (their right panel plots the
estimate against its own arc length, which hides gaps; fixed for run 3).
Coverage columns computed afterwards on the M4 from the saved
`geom_init.npz` (2 mm tolerance).

| Case | Radius MAE (mm) | Bias (mm) | GT coverage | Lesion coverage | Throat r est / GT (mm) | Pass (MAE) |
|---|---|---|---|---|---|---|
| 121_0 | 0.054 | -0.045 | 100 % | 100 % | 0.75 / 0.81 | yes |
| 122_0 | 0.106 | -0.061 | 90 % | 100 % | 0.65 / 0.89 | yes |
| 130_0 | 0.085 | -0.039 | 97 % | 100 % | 0.63 / 0.79 | yes |
| 131_0 | 0.052 | -0.016 | **67 %** | **0 %** | (missing: nearest point 5.3 mm away) | yes |
| 135_0 | 0.067 | -0.048 | 100 % | 100 % | 0.69 / 0.63 | yes |

Mean radius MAE 0.073 mm; all 5 below 0.2 mm, so the DoD as written is
met. But the MAE only scores points that were found: in 131_0 the FDK
volume's dim proximal third (with the stenosis) was cut off by the single
0.2 x p99.9 threshold, so the centerline starts after the lesion. That
would give Step 2.2 a wrong inlet.

Attempted fix (part of the fallback's skeleton extraction, threshold
choice OURS): hysteresis thresholding (seed 0.2, grow to 0.1 x p99.9).
Synthetic test: a proximal segment at 15 % brightness is kept. Failed on
real data (run 3, below). The notebook now also reports GT coverage, lesion
coverage and throat radius error.

### Run 3: FDK + hysteresis + 4DRGS view subsets, Oct 4, repo `b6c19ec`

`results/step2.1_geom_init_v60_fdk_run3.json`, figures `..._fdk_run3.png`.

| Case | Radius MAE (mm) | GT coverage | Lesion coverage | Throat error (mm) |
|---|---|---|---|---|
| 121_0 | 0.320 | 73 % | 100 % | -0.06 |
| 122_0 | 0.147 | 77 % | 82 % | -0.38 |
| 130_0 | 0.184 | 92 % | 100 % | -0.19 |
| 131_0 | 0.229 | 67 % | 0 % | +1.07 (lesion missing) |
| 135_0 | 0.172 | 87 % | 85 % | +0.08 |

Worse than run 2 (mean MAE 0.21 mm, 3 of 5 above 0.2 mm). The 121_0 figure
shows why: with the low 0.1 threshold the centerline runs into FDK streak
artifacts (estimates of 0.4 to 0.8 mm radius at s = 40 to 70 mm), and the
dim proximal part is still not recovered in 131_0. Hysteresis is reverted
(single threshold 0.2 as in run 2; kept only as an option). Run 4 = run 2
extraction with the 4DRGS view subsets: TODO.

## DoD

| DoD part | Met |
|---|---|
| Lumen radius error < 0.2 mm on 5 validation cases | Run 2 (FDK fallback): yes, MAE 0.05 to 0.11 mm; coverage gap in 131_0; run 3 (hysteresis) failed; run 4 pending |

**DoD met: as written yes (run 2); final confirmation with run 4. Fallback used: yes (FDK volume skeleton). Known limitation: 131_0 misses its proximal third with the stenosis.**

Plan fallback (if Gaussians do not align with the vessel): "initialize
directly from the ground-truth-free FDK volume skeleton."
