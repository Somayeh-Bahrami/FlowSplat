# Step 3.1: Sweeps and FFR

Dates: Oct 20 to Oct 26, 2026 (plan; started Oct 6). Status: **in progress**
(FFR done on the validation set; sweeps queued on molab).

Plan DoD: "Sweep tables saved; FFR MAE and accuracy at 0.80 on validation set."
Method for all runs: Gate 1 method (reports/gate1.md): 4DRGS init, radius x k
(median rule on 15 training cases), Step 2.2 variant frozen.

## Where things run

| Piece | Runs on |
|---|---|
| `src/flowsplat/ffr.py`, `metrics.py` | M4 (and inside notebooks/step2_train.py on molab) |
| `src/flowsplat/sweeps.py` | M4 (edits, tests, collection); molab (TIGRE rendering) |
| `notebooks/step3_sweeps.py` | molab GPU |
| `tests/test_sweeps_ffr.py` | M4; whole suite 96 tests pass |

## Decisions (user, Oct 6)

- **FFR flow.** The bolus is imaged at rest, the ground-truth FFR is at
  hyperemia (Q_hyp = CFR x Q_rest, CFR 1.5 to 4.0 per case). Primary:
  Q_hyp = case CFR x recovered Q_rest (isolates the reconstruction error;
  an assumption to state in the paper). Secondary: one population CFR 2.75.
- **Sweep scope.** Views 30 / 60 / 133 on the 20 validation cases (k
  recalibrated per view count on the 15 training cases); noise, Peclet,
  flow speed and foreshortening at 2 or 3 levels on the 5 standard
  validation cases; 3 repeat 4DRGS runs on the 5 cases (Gate 1).

## FFR (done, M4)

**Lesion parameters without ground truth** (`ffr.lesion_from_profile`, OURS):
throat at the largest relative deficit against an interpolated reference
radius, lesion length from the deficit width. Extraction floor on the
ground-truth radius of all 200 cases with the true hyperemic flow: FFR MAE
0.011, accuracy 98.5 % at 0.80, 8 of 200 lesions misplaced
(results/step3.1_ffr_extraction_floor.json).

**Validation set (20 cases, 60 views, run 8 estimates)**
(results/step3.1_ffr_val20_v60.json):

| FFR from | MAE | Bias | Accuracy at 0.80 | Sens. | Spec. |
|---|---|---|---|---|---|
| **FlowSplat, case CFR (primary)** | **0.030** | +0.002 | **90 %** | 3 / 4 | 15 / 16 |
| FlowSplat, population CFR 2.75 | 0.079 | -0.066 | 85 % | 4 / 4 | 13 / 16 |
| Oracle: true lesion + estimated Q | 0.012 | -0.005 | 100 % | | |
| Oracle: estimated lesion + true Q | 0.034 | +0.004 | 95 % | | |
| Ground truth geometry and Q, population CFR | 0.071 | -0.067 | 85 % | | |

- The FFR error comes from the lesion geometry, not from Q: with the true
  lesion the error is 0.012; with the true Q it stays 0.034.
- Largest errors: 131_0 (0.699 -> 0.803, throat radius +0.10 mm), 135_0
  (0.605 -> 0.715, throat +0.06 mm), 130_0 (0.577 -> 0.510): throat radius
  errors of 0.05 to 0.1 mm move FFR by 0.05 to 0.1 for severe lesions
  (Young-Tsai: dP ~ A_s^-2).
- Only 4 of 20 validation cases have FFR <= 0.80, so sensitivity is coarse.
- With a population CFR, the CFR uncertainty dominates (MAE 0.071 even with
  perfect geometry and Q).

## Sweeps

Settings and levels: flowsplat.sweeps.SETTINGS; table: results/step3.1_sweeps.json
(sweeps.collect). Run 8 (60 views) is the middle level.

### Views (20 validation cases)

| Views | k (median, 15 train) | Q err mean (median) | Q < 10 % | Q signed | Diam. err mean | Diam. < 0.2 mm | FFR MAE / acc. (case CFR) | FFR MAE / acc. (pop. CFR) |
|---|---|---|---|---|---|---|---|---|
| 30 | 1.126 | 7.8 % (6.9 %) | 12 / 20 | +5.8 % | 0.171 mm | 15 / 20 | 0.063 / 85 % | 0.113 / 75 % |
| 60 (run 8) | 1.086 | 5.3 % (4.3 %) | 19 / 20 | +3.0 % | 0.104 mm | 16 / 20 | 0.030 / 90 % | 0.079 / 85 % |
| 133 | | pending | | | | | | |

30 views (Oct 7, molab, repo `fbead1b`, results/step2.2_val20_v30_4drgs_cal.json):
- Step 2.2 DoD (means) also met at 30 views: Q 7.8 %, diameter 0.171 mm.
- The 4DRGS radius bias is larger with fewer views (training area ratio
  median 0.79 vs 0.85 at 60 views, k 1.126 vs 1.086); the per-view-count
  calibration absorbs it.
- 4DRGS geometry at 30 views: lesion coverage 100 % on all 20, 3 cases fail
  the v2 radius rule (130_1, 130_2, 131_1, MAE 0.22 to 0.27 mm); training: 10
  of 15 pass.
- FFR doubles in error (0.063) and the same lesion-geometry pattern holds:
  true lesion + estimated Q 0.019, estimated lesion + true Q 0.073. Largest
  FFR errors: 130_0 (0.577 -> 0.334), 130_1 (0.873 -> 0.636), 131_1
  (0.962 -> 0.782), 135_0 (0.605 -> 0.794).
