# Step 2.2: Transport layer and joint optimization

Dates: Oct 14 to Oct 19, 2026 (started Oct 4). Status: **in progress**:
transport layer, projector and training code done and tested on the M4;
molab runs pending (overfit one case, then 20 validation cases).

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

Pre-specified before the 20-case run: primary variant **pose_arr** (both
parts of the fallback: geometry radius kept frozen, positions refined,
arrival-time loss); frozen and pose reported as ablations; base reported as
the failed plan route.

## DoD

| DoD part | Met |
|---|---|
| 20 val cases: Q relative error < 10 % | TODO |
| 20 val cases: stenosis diameter error < 0.2 mm | TODO |

**DoD met: no (in progress). Fallback used: yes (keep geometry frozen longer, add arrival-time loss), variant comparison pending.**
