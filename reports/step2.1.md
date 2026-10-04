# Step 2.1: Geometry initialisation and tube-aligned Gaussians

Dates: Oct 12 to Oct 14, 2026 (started Oct 3). Status: **in progress**:
code and tests done on the M4; molab run pending.

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

TODO.

## DoD

| DoD part | Met |
|---|---|
| Lumen radius error < 0.2 mm on 5 validation cases | TODO |

**DoD met: no (in progress). Fallback used: no.**

Plan fallback (if Gaussians do not align with the vessel): "initialize
directly from the ground-truth-free FDK volume skeleton."
