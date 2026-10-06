# Step 2.2: Transport layer and joint optimization

Dates: Oct 14 to Oct 19, 2026 (run Oct 4 to 5). Status: **finished, DoD
partly met**: Q met, stenosis diameter not met; plan fallback applied
(all clauses); outcome goes to Gate 1 (Oct 20).

Plan DoD: "On 20 validation cases: Q relative error below 10 % and stenosis
diameter error below 0.2 mm (30 or 60 views)." Interpreted as the mean over
the 20 cases; per-case fractions are reported too (interpretation is ours,
to confirm).

## Where things run

| Piece | Runs on |
|---|---|
| `src/flowsplat/transport.py`, `train.py`, `model.py:ConeProjector` | M4 (tests, prototype) and molab |
| `notebooks/step2_train.py` | molab GPU |
| `tests/test_transport_train.py` | M4; whole suite 76 tests pass |

## What was done

1. **Transport layer** (`transport.py`): tau_i = int_0^{s_i} A/Q ds
   (trapezoid between Gaussian centres), A = 2 pi sigma_r^2, c_i(t) =
   C_in(t - tau_i), gamma-variate C_in (t0, alpha, beta) with amplitude k.
   Peak densities normalised so the Gaussian chain is a lumen of area A with
   attenuation k c.
2. **Projector** (`model.py:ConeProjector`): exact line integral of every
   Gaussian along every cone-beam ray (TIGRE geometry), in PyTorch.
   Calibrated on the M4 against the TIGRE renders of Step 1.2 (ground-truth
   tube, true concentration, 32 conventions): only the default convention
   matches (relative error 0.26 vs 0.65 or more), see
   `results/step2.2_projector_convention.json`. Remaining mismatch: Gaussian
   versus uniform-disk cross-section and the clipped noise floor.
3. **Training** (`train.py:fit_case`): inlet end chosen by a short Stage A
   in both directions; Stage A (geometry frozen, Q and C_in, footprints
   cached); Stage B (joint, L1 + 0.2 D-SSIM as 4DRGS Eq 6, 4DRGS learning
   rates converted to mm, alignment regulariser, 8 random views per step).

## Prototype results on the M4 (case 121_0, detector binned 2 x 2)

| Run | Q est / GT (mL/s) | Q error | Note |
|---|---|---|---|
| Stage A, ground-truth geometry, 60 views | 1.216 / 1.218 | **0.2 %** | Transport layer recovers Q from the richer bolus data |
| Stage A, Step 2.1 FDK geometry, 60 views | 1.143 / 1.218 | 6.2 % | Follows the init radius bias (about -0.04 mm, -5 % area) |
| Stage A, FDK geometry, 30 views | 1.145 / 1.218 | 6.0 % | |
| Stage A, FDK geometry, 131_0 (inlet third missing), 60 views | 2.725 / 2.626 | 3.8 % | |
| Dry run: Stage A + 60 Stage B iterations, FDK geometry | 1.042 / 1.218 | 14.5 % | Loss decreases but Q and radius get worse; stenosis diameter error 0.12 mm |

The inlet end was identified correctly in every run. The dry run suggests
a geometry / flow ambiguity in Stage B (narrower Gaussians with a larger
amplitude k fit the flat-topped lumen profile better, which shrinks A and
therefore Q). The full molab overfit run decides; the plan's fallback for
this case is "keep geometry frozen longer, add arrival-time loss".

## Results (molab)

### Overfit run 1: 121_0, 60 views, plan route (Stage A then full Stage B), Oct 5, repo `9083c78`

`results/step2.2_overfit_v60_run1_base.json`, figure `results/step2.2_121_0_v60_run1_base.png`.

| Metric | Stage A (geometry = FDK init) | After Stage B | DoD |
|---|---|---|---|
| Q relative error | 6.1 % | **4.4 %** | < 10 % |
| Stenosis diameter error | about 0.07 mm (Step 2.1 init throat) | **0.28 mm** (throat 0.67 vs 0.81 mm) | < 0.2 mm |
| Radius MAE | 0.05 mm (Step 2.1) | 0.38 mm | |
| Image loss | 0.0074 | 0.0061 | |

Stage B lowers the image loss while shrinking the radius by about 0.3 mm
along the whole vessel (figure). The objective itself prefers thinner
tubes: a Gaussian cross-section with a free amplitude k fits the
flat-topped lumen profile better when it is narrower, so the area
A = 2 pi sigma^2 is biased low (a geometry / flow ambiguity). The stenosis
diameter fails the DoD on the overfit case. 1.8 min per case on the GPU.

### Fallback applied (rule 2)

Plan, Step 2.2 Fallback: "Q does not converge: grid search over Q, then
fine-tune. Geometry/flow ambiguity: keep geometry frozen longer, add
arrival-time loss. 30 views insufficient: use 60 and report error vs views
as a result."

Q converges; the second clause applies. Realised as variants
(`train.VARIANTS`): `frozen` (geometry frozen for the whole budget),
`pose` (Stage B moves positions and rotations, radius frozen), `pose_arr`
(pose + arrival-time loss: arrival at each Gaussian measured from the data
as the half-peak time of its normalised time-intensity curve, compared with
t0 + tau_i + the inlet half-rise time). Compared on the first case of each
of the 5 validation patients before the 20-case run.

### Run 2: variants on 5 validation cases, Oct 5, repo `4ffd92b`

`results/step2.2_variants5_v60_run2_bug_moving_arc.json`.

| Variant | Mean Q error | Stage A Q error | Mean diameter error | Radius MAE |
|---|---|---|---|---|
| base | 123 % | 3.9 % | 0.82 mm | 0.47 mm |
| frozen | **3.8 %** (5 of 5 below 10 %) | 3.8 % | 0.36 mm (2 of 4 below 0.2; 131_0 lesion missing) | 0.07 mm |
| pose | 143 % | 3.9 % | 0.65 mm | 0.09 mm |
| pose_arr | 129 % | 4.3 % | 0.71 mm | 0.09 mm |

Stage A is good on all 5 cases (0.4 to 6.1 %). Every variant that moves
positions in Stage B ruined Q. Diagnosis (bug in our discretisation): tau
and the density weights used distances between the moving Gaussian
centres; in Stage B the chain zig-zagged at no image cost (overlapping
Gaussians), its length grew 174 -> 449 mm (130_0, pose), and Q grew by the
same factor at a constant transit time. Fix (plan-consistent: Gaussians sit
at fixed arc lengths s_i): tau and the weights use the s_i assigned at
initialisation (regression test added).

M4 check after the fix (121_0, detector binned 2 x 2, 300 Stage B
iterations): pose: Q error 4.5 % (Stage A 6.1 %), Q stable during Stage B
(1155 -> 1162 mm^3/s), chain 164 mm, diameter error 0.071 mm, radius MAE
0.068 mm. base: Q error 28 %, radius MAE 0.26 mm: with the radius free the
Gaussian-profile bias remains (separate from the bug).

### Run 3: variants on the same 5 cases after the fix, Oct 5, repo `3156009`

`results/step2.2_variants5_v60_run3_fixed.json`, figures `..._run3_fixed.png`.

| Variant | Mean Q error (max) | Q below 10 % | Mean / median diameter error | Diameter below 0.2 mm | Radius MAE |
|---|---|---|---|---|---|
| **pose_arr** | **4.4 %** (7.3 %) | **5 / 5** | 0.63 / 0.34 mm | 2 / 5 | 0.09 mm |
| pose | 4.5 % (11.0 %) | 4 / 5 | 0.63 / 0.34 mm | 2 / 5 | 0.09 mm |
| frozen | 3.8 % (6.1 %) | 5 / 5 | 0.36 / 0.21 mm (4 cases) | 2 / 4 | 0.07 mm |
| base | 24 % (52 %) | 2 / 5 | 0.88 / 0.75 mm | 0 / 5 | 0.46 mm |

Per case (pose_arr): Q error 3.4 / 7.3 / 4.4 / 3.8 / 3.0 % and diameter
error 0.071 / 0.953 / 0.342 / 1.716 / 0.064 mm for 121_0 / 122_0 / 130_0 /
131_0 / 135_0. Q is stable in Stage B after the fix. The diameter is set by
the FDK init: 122_0 (init throat 0.48 mm too small) and 131_0 (lesion
missing from the init; the lesion window matches Gaussians at the end of
the chain) fail.

Pre-specified before the 20-case run: primary variant **pose_arr** (both
parts of the fallback: geometry radius kept frozen, positions refined,
arrival-time loss); frozen and pose reported as ablations; base reported as
the failed plan route.

### Run 4 (DoD run): 20 validation cases, 60 views, Oct 5, repo `75fada9`

`results/step2.2_val20_v60.json`, figures `results/step2.2_val20_figs/`
(each shows ground truth, FDK init and the four variants). Cases: the 20
first validation cases ordered by stenosis index then patient (all 6
validation patients). 0.6 to 2.1 min per case and variant on the GPU.

| Variant | Mean Q error | Median | Max | Q below 10 % | Mean diameter error | Median | Diameter below 0.2 mm | Lesion found | Radius MAE |
|---|---|---|---|---|---|---|---|---|---|
| **pose_arr (primary, pre-specified)** | **5.2 %** | 3.6 % | 30.9 % | 17 / 20 | **0.36 mm** | 0.12 mm | 10 / 19 | 19 / 20 | 0.09 mm |
| pose | 4.8 % | 4.2 % | 11.0 % | 17 / 20 | 0.47 mm | 0.21 mm | 10 / 20 | 20 / 20 | 0.09 mm |
| frozen | 4.1 % | 3.5 % | 10.5 % | 19 / 20 | 0.26 mm | 0.09 mm | 10 / 16 | 16 / 20 | 0.07 mm |
| base (plan route) | 36.7 % | 43.5 % | 64.0 % | 3 / 20 | 0.98 mm | 1.03 mm | 0 / 19 | 19 / 20 | 0.52 mm |

Per case, Q error / stenosis diameter error (mm):

| Case | pose_arr | pose | frozen | base |
|---|---|---|---|---|
| 121_0 | 3.4 % / 0.07 | 5.1 % / 0.07 | 6.1 % / 0.07 | 21.5 % / 1.41 |
| 122_0 | 7.3 % / 0.95 | 1.2 % / 0.95 | 0.3 % / 0.95 | 5.4 % / 0.48 |
| 130_0 | 4.4 % / 0.34 | 0.7 % / 0.34 | 1.1 % / 0.34 | 2.9 % / 0.72 |
| 131_0 | 3.8 % / 1.72 | 11.0 % / 1.72 | 5.9 % / - | 52.3 % / 0.75 |
| 135_0 | 3.0 % / 0.06 | 4.4 % / 0.06 | 5.7 % / 0.06 | 39.6 % / 1.03 |
| 667_0 | 0.2 % / - | 9.5 % / 2.60 | 2.9 % / - | 46.6 % / - |
| 121_1 | 3.3 % / 0.07 | 6.2 % / 0.07 | 6.6 % / 0.07 | 14.1 % / 0.37 |
| 122_1 | 0.4 % / 0.10 | 4.4 % / 0.10 | 3.2 % / 0.10 | 25.3 % / 1.25 |
| 130_1 | 5.8 % / 0.06 | 6.2 % / 0.06 | 6.8 % / 0.06 | 36.4 % / 0.47 |
| 131_1 | 30.9 % / 0.76 | 10.5 % / 0.76 | 6.6 % / 0.76 | 64.0 % / 1.55 |
| 135_1 | 2.3 % / 0.30 | 2.4 % / 0.30 | 2.1 % / 0.30 | 46.6 % / 0.55 |
| 667_1 | 10.7 % / 0.02 | 2.7 % / 0.02 | 10.5 % / 0.02 | 52.2 % / 1.31 |
| 121_2 | 3.9 % / 0.07 | 3.5 % / 0.07 | 4.5 % / 0.07 | 23.8 % / 0.81 |
| 122_2 | 1.2 % / 0.46 | 6.4 % / 0.46 | 1.1 % / - | 60.8 % / 1.06 |
| 130_2 | 10.9 % / 0.50 | 0.2 % / 0.50 | 3.7 % / - | 53.9 % / 0.59 |
| 131_2 | 0.8 % / 0.58 | 11.0 % / 0.58 | 7.8 % / 0.58 | 46.3 % / 1.41 |
| 135_2 | 4.7 % / 0.03 | 3.2 % / 0.03 | 3.3 % / 0.03 | 52.1 % / 1.66 |
| 667_2 | 4.7 % / 0.06 | 3.3 % / 0.06 | 2.6 % / 0.06 | 40.8 % / 0.46 |
| 121_3 | 1.4 % / 0.53 | 0.5 % / 0.53 | 0.3 % / 0.53 | 0.4 % / 1.51 |
| 122_3 | 1.8 % / 0.12 | 4.1 % / 0.12 | 1.7 % / 0.12 | 48.1 % / 1.29 |

Observations:

- **Q is recovered**: mean 5.2 % (primary), 4.1 % with the geometry fully
  frozen. Stage A alone gives 4.1 % on average; Stage B with pose updates
  does not improve Q on average.
- **The stenosis diameter is set by the Step 2.1 FDK init**: the diameter
  errors of pose_arr, pose and frozen are identical case by case (the radius
  is frozen in all three; they differ only in which Gaussians fall inside
  the lesion window). Cases with a good init throat pass (0.02 to 0.12 mm);
  cases with a poor init fail (122_0 0.95 mm, 131_0 1.72 mm with the lesion
  missing from the init, 130_2 0.50 mm, 121_3 0.53 mm).
- **Freeing the radius (base) makes it worse**: Q 37 %, diameter 0.98 mm,
  radius MAE 0.52 mm: the Gaussian cross-section with free amplitude is
  biased thin against the flat-topped lumen.
- **Failures in Q**: 131_1 (30.9 %; patient 131 has its dim proximal third
  missing from the init, as in Step 2.1), 667_1 and 130_2 (10.7 and 10.9 %).

## DoD

| DoD part | Result (pose_arr, 20 val cases, 60 views) | Met |
|---|---|---|
| Q relative error < 10 % | mean 5.2 % (median 3.6 %, 17 / 20 below 10 %) | **Yes** |
| Stenosis diameter error < 0.2 mm | mean 0.36 mm (median 0.12 mm, 10 / 19 below 0.2 mm) | **No** |

DoD rule as interpreted: mean over the 20 cases (interpretation ours, to
confirm). With the median, the diameter would pass (0.12 mm) and Q would
pass; per case, 10 of 19 pass the diameter.

**DoD met: no (Q yes, stenosis diameter no). Fallback used: yes.** Plan,
Step 2.2 Fallback: "Q does not converge: grid search over Q, then
fine-tune. Geometry/flow ambiguity: keep geometry frozen longer, add
arrival-time loss. 30 views insufficient: use 60 and report error vs views
as a result." Q converges (first clause not needed); the second clause is
applied (pose_arr); 60 views are used (third clause). All applicable
clauses of the fallback are used, and the diameter part still fails.
Rule 2: no further method change before the Oct 20 gate review.

## For the Gate 1 review (Oct 20)

- Gate 1 condition (plan): "Q error < 10 % on single vessel?" Result: mean
  5.2 % (primary), 4.1 % (frozen); 17 to 19 of 20 cases below 10 %.
- Not met: stenosis diameter (mean 0.36 mm vs 0.2 mm), limited by the FDK
  init at the throat and by the thin bias of free Gaussian radii. Relevant
  for Step 3.1 (FFR depends on the throat area to the fourth power through
  Young-Tsai).
- Known weak patient: 131 (init misses the inlet third).
- 30-view results not run yet (DoD allows 30 or 60).
