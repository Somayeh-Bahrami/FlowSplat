# Step 2.1: Geometry initialisation and tube-aligned Gaussians

Dates: Oct 12 to Oct 14, 2026 (done early, Oct 4 to 6). Status: **closed.**

**Final method (Gate 1, Oct 6): 4DRGS init (Step 2.1 v2) + radius x k,
k = 1.086 from 15 training cases (median rule).** On the 20 validation
cases: lesion coverage 100 %, vessel coverage >= 97 %, radius MAE 0.056 to
0.197 mm before calibration, 0.071 mm after (Step 2.2 run 8). History:
run 1 static R2-Gaussian (failed), runs 2 to 4 FDK fallback (DoD met, but
131_0 lesion missing), v2 4DRGS (approved plan deviation, Oct 5), radius
calibration (Oct 6). Sections below are in chronological order.

Plan DoD: "Lumen radius error below 0.2 mm on 5 validation cases."

## Where things run

| Piece | Runs on |
|---|---|
| `src/flowsplat/model.py` | M4 tests; molab |
| `notebooks/step2_geom_init.py` (R2-Gaussian, FDK or 4DRGS init) | molab GPU |
| `tests/test_model.py` | M4 (CPU); whole suite 83 tests pass (Oct 6) |

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
(single threshold 0.2 as in run 2; kept only as an option).

### Run 4 (final): FDK fallback, single threshold, 4DRGS view subsets, Oct 4, repo `c83190b`

`results/step2.1_geom_init_v60_fdk_run4.json`, figures `..._fdk_run4.png`;
initial geometries `geom_init.npz` per case (molab
`~/flowsplat_work/step2_1/<case>_v60_fdk/`, copy on the M4 in
`data/open_items2/step2_1/`).

| Case | Radius MAE (mm) | Bias (mm) | GT coverage | Lesion coverage | Throat error (mm) | Length est / GT (mm) | Pass |
|---|---|---|---|---|---|---|---|
| 121_0 | 0.050 | -0.040 | 100 % | 100 % | -0.04 | 156.0 / 150.0 | yes |
| 122_0 | 0.117 | -0.054 | 88 % | 100 % | -0.48 | 169.5 / 162.5 | yes |
| 130_0 | 0.103 | -0.057 | 97 % | 100 % | -0.17 | 174.5 / 163.5 | yes |
| 131_0 | 0.045 | -0.040 | 67 % | 0 % | (lesion missing) | 73.5 / 104.0 | yes |
| 135_0 | 0.066 | -0.044 | 100 % | 100 % | +0.03 | 121.0 / 117.5 | yes |

Mean radius MAE 0.076 mm; all 5 cases below 0.2 mm. Run 4 reproduces run 2
(the view-subset change has no material effect).

## DoD

| DoD part | Met |
|---|---|
| Lumen radius error < 0.2 mm on 5 validation cases | **Yes** (run 4: 0.045 to 0.117 mm, mean 0.076 mm) |

**DoD met: yes. Fallback used: yes**: plan, Step 2.1: "If Gaussians do not
align with the vessel: initialize directly from the ground-truth-free FDK
volume skeleton." The plan route (static R2-Gaussian, run 1) gave 0.20 to
0.42 mm.

## Known limitation and open items

- **131_0:** the dim proximal third of the vessel (filled early, washed out
  before most views) is lost in the FDK volume, so the initial centerline
  starts after the stenosis (coverage 67 %, lesion 0 %). Step 2.2 starts
  from an incomplete vessel there; this is reported, not fixed (the
  hysteresis attempt failed, run 3).
- The radius MAE scores only points that were found; coverage and lesion
  coverage are now reported next to it.
- Throat radius at initialisation is off by up to 0.48 mm (122_0); Step 2.2
  refines geometry against the projections (its DoD: stenosis diameter
  error < 0.2 mm).
- Estimated centerlines are 3.5 to 11 mm longer than GT (end caps and FDK
  tails), excluded by the 5 mm end trim.

## Step 2.1 v2: 4DRGS initialisation (plan deviation, approved Oct 5)

**Why.** Step 2.2 on 20 validation cases met the Q part of its DoD but not
the stenosis diameter part (mean 0.36 mm). The diameter errors are the same
for the variants frozen, pose and pose_arr, which keep the initial radius,
so the error comes from the Step 2.1 FDK init. The FDK radius MAE (0.045 to
0.117 mm) only scores points that were found: 131_0 misses its lesion
(lesion coverage 0 %) and the 122_0 throat is 0.41 vs 0.89 mm. External
feedback (Oct 5) proposed 4DRGS as the initializer. The user approved
re-doing Step 2.1 with 4DRGS, then a 5-case Step 2.2, before concluding
Step 2.2. Static R2-Gaussian is not re-tried: it already failed (run 1).

**Protocol, fixed before any 4DRGS result.**
- Same 5 cases (121_0, 122_0, 130_0, 131_0, 135_0), same 60 views.
- Official 4DRGS at commit 38006ed, source untouched, README fast setting
  (10k iterations, ADC until 5k, as reproduced in Step 0.1), TIGRE FDK
  init (`--fdk_toolbox=tigre`).
- Init volume: the 4DRGS time-averaged voxel query (VQR) volume; then the
  same centerline and half-max radius code as the FDK route.
- Pass (per case): radius MAE < 0.2 mm AND vessel coverage >= 90 % AND
  lesion coverage >= 90 %. A missed lesion is a failure.
- Also reported: throat radius error, 4DRGS held-out view PSNR / SSIM.
- Then Step 2.2 on the same 5 cases with frozen and pose_arr, compared with
  the FDK init (variants5 run 3: pose_arr Q 4.4 %, diameter 0.63 mm mean;
  frozen Q 3.8 %, diameter 0.36 mm on 4 cases with a lesion).
- Decision after the 5 cases: run the 20 validation cases with the better
  init. If 4DRGS does not improve lesion coverage on 131_0, that case is
  reported as an observability limitation (lesion opacified over part of
  the sweep only).
- Caveat for the paper: 4DRGS is also baseline 1 (Step 4.1). Using it as
  the initializer must be stated in the method section.

### Results: run 5, 4DRGS init, Oct 5 (molab, repo `dc10e28`), 60 views

4DRGS training: 1 min 31 s to 1 min 42 s per case (10k iterations), default
M1 = 30000 on every case (84521 to 105133 FDK voxels above 0.016). As
expected, train.py exits in its mesh / ICP evaluation (no reference mesh);
the VQR volume is written before that and is used.

| Case | Radius MAE (mm) | Bias (mm) | Vessel cov. | Lesion cov. | Throat err. (mm) | Held-out PSNR (dB) | Pass v2 |
|---|---|---|---|---|---|---|---|
| 121_0 | 0.089 | -0.089 | 99.3 % | 100 % | -0.048 | 38.00 | yes |
| 122_0 | 0.151 | -0.151 | 100 % | 100 % | -0.063 | 34.81 | yes |
| 130_0 | 0.125 | -0.125 | 99.7 % | 100 % | -0.080 | 32.57 | yes |
| 131_0 | 0.096 | -0.094 | 98.6 % | 100 % | +0.040 | 34.51 | yes |
| 135_0 | 0.075 | -0.075 | 99.6 % | 100 % | +0.015 | 37.95 | yes |

Compared with the FDK init (run 4): lesion coverage of 131_0 goes from 0 %
to 100 % (vessel 67 % -> 98.6 %), the 122_0 throat error from -0.48 mm to
-0.06 mm, and estimated lengths now match GT within 2.5 mm (FDK: 3.5 to
11 mm too long). The radius MAE is slightly higher than FDK (mean 0.107 vs
0.076 mm) and is almost entirely a uniform negative bias (MAE = |bias|):
the half-max radius on the 4DRGS volume is 0.075 to 0.151 mm too small, so
the lumen area is 9 to 19 % too small (FDK: about 5 %).

**Step 2.1 v2: pass on all 5 cases under the stricter rule (and the plan
DoD).** The area bias matters for Q, see reports/step2.2.md run 5.

### Final state after Gate 1 (Oct 6)

The Step 2.1 method used from now on is the 4DRGS init (v2) with the
radius multiplied by k = 1.086 (median rule, 15 training cases; see
reports/gate1.md and reports/step2.2.md run 8). On the 20 validation cases
the uncalibrated v2 geometry passes the stricter rule on every case (lesion
coverage 100 %, vessel coverage >= 97 %, radius MAE 0.056 to 0.197 mm,
results/step2.1_geom_init_v60_4drgs_val20.json); after calibration the
radius MAE in Step 2.2 is 0.071 mm. 2 of 15 training cases (110_0, 129_0)
fail the v2 rule (radius MAE 0.33 and 0.44 mm, lesion found).
