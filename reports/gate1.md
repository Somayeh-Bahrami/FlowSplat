# Gate 1 review (held Oct 6, 2026, scheduled Oct 20)

Called by the user ("Gate review") after Step 2.2 run 7.

## Condition and pre-committed decision (plan)

"Q error < 10 % on single vessel?" Yes: continue CVPR. No: retarget CVPR 2027
workshops or MICCAI 2027, same plan.

## Evidence (20 validation cases, 60 views; reports/step2.2.md)

| Route | Q rel. error (mean) | Stenosis diameter error (mean) | DoD 2.2 |
|---|---|---|---|
| FDK init, pose_arr (run 4) | 5.2 % | 0.36 mm | Q only |
| 4DRGS init + k = 1.114, pose_arr (run 7) | 10.5 % | 0.146 mm | diameter only |
| 4DRGS init + k = 1.114, frozen (run 7) | 8.3 % | 0.146 mm | both |

## Decisions (user, Oct 6)

1. **Gate 1: pass. Continue CVPR 2027.**
2. **Method for all later steps (3.1, 4.1, 5.1, 6.1): 4DRGS init (Step 2.1
   v2) + calibrated radius + variant frozen** (geometry fixed after the
   init, Stage A fits Q and the inlet curve). Chosen on the validation
   split; the test split stays untouched.
   Paper consequence: geometry from 4DRGS, Q from the transport-constrained
   Gaussians on that geometry (contributions 1 and 2 hold; "joint" means a
   single Gaussian model for geometry and flow, not a joint refinement).
   Baseline 1 (4DRGS + post-hoc TDC) then shares our geometry exactly.
3. **Recalibrate k** on the first case of the first 15 training patients
   (109_0, 110_0, 112_0, 114_0, 115_0, 120_0, 127_0, 128_0, 129_0, 571_0,
   572_0, 573_0, 592_0, 593_0, 594_0), k = 1 / sqrt(median area ratio)
   (md.radius_scale, stat="median"). Rule fixed before those runs. Then one
   re-run of the 20 validation cases with frozen and the new k.
4. **Step 3.1 scope addition:** 3 repeat 4DRGS runs on the 5 validation
   cases (121_0, 122_0, 130_0, 131_0, 135_0) to report run-to-run spread
   (131_0 diameter error 0.43 mm in run 6, 0.18 mm in run 7).

## Open for Gate 2 (Nov 2)

- Does the calibration transfer to real data (no ground truth there)?
- 30-view results not run yet (Step 3.1 sweeps).

## Outcome of decision 3 (run 8, Oct 6)

k = 1.0860 (median of 15 training cases). Step 2.2 on the 20 validation
cases with frozen: Q 5.3 % (19 / 20 below 10 %), stenosis diameter
0.104 mm (16 / 20 below 0.2 mm). Step 2.2 DoD met. Step 2.2 closed.
