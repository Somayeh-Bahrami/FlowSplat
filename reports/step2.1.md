# Step 2.1: Geometry initialisation and tube-aligned Gaussians

Dates: Oct 12 to Oct 14, 2026 (started Oct 3). Status: **in progress**:
plan route failed the DoD (run 1); fallback (FDK skeleton) implemented, run
pending.

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
output zip of run 1 did not reach the M4 (only the printed summary).

### Fallback applied (rule 2)

Plan, Step 2.1 Fallback: "If Gaussians do not align with the vessel:
initialize directly from the ground-truth-free FDK volume skeleton."

`FLOWSPLAT_INIT=fdk` (now the default): TIGRE FDK with a Hann filter (as
the 4DRGS FDK baseline, Sec 4.1) of the same 60 views, then the same
centerline and half-maximum radius. Run 2: TODO.

## DoD

| DoD part | Met |
|---|---|
| Lumen radius error < 0.2 mm on 5 validation cases | TODO |

**DoD met: no (in progress). Fallback used: yes (FDK volume skeleton), run pending.**

Plan fallback (if Gaussians do not align with the vessel): "initialize
directly from the ground-truth-free FDK volume skeleton."
