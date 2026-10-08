# Step 3.1: Sweeps and FFR

Dates: Oct 20 to Oct 26, 2026 (plan; done Oct 6 to 7). Status: **closed, DoD
met** (sweep tables saved; FFR MAE 0.030 and accuracy 90 % at 0.80 on the
20 validation cases, 60 views). Fallback: not needed.

Plan DoD: "Sweep tables saved; FFR MAE and accuracy at 0.80 on validation set."
Method for all runs: Gate 1 method (reports/gate1.md): 4DRGS init, radius x k
(median rule on 15 training cases), Step 2.2 variant frozen.

## Where things run

| Piece | Runs on |
|---|---|
| `src/flowsplat/ffr.py`, `metrics.py` | M4 (and inside notebooks/step2_train.py on molab) |
| `src/flowsplat/sweeps.py` | M4 (edits, tests, collection); molab (TIGRE rendering) |
| `notebooks/step3_sweeps.py` | molab GPU |
| `tests/test_sweeps_ffr.py` | M4; whole suite 96 tests pass (Oct 7) |

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
| 133 | 1.070 | 3.8 % (3.7 %) | 20 / 20 | +1.8 % | 0.112 mm | 19 / 20 | 0.040 / 85 % | 0.099 / 80 % |

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

133 views (Oct 7, molab, repo `57197b8`, results/step2.2_val20_v133_4drgs_cal.json):
- Q 3.8 %, all 20 below 10 %; diameter 0.112 mm, 19 / 20 below 0.2 mm.
- More views, less radius bias: training area ratio median 0.87 (k 1.070);
  validation area mean 0.893; all 20 validation cases pass the v2 rule.
- FFR MAE 0.040 is above the 60-view 0.030 because of one case: 131_1
  (throat 0.85 mm diameter error, FFR 0.962 -> 0.727). Without 131_1:
  0.057 / 0.029 / 0.029 at 30 / 60 / 133 views. 131_1 fails at every view
  count (patient 131 is the weak patient since Step 2.1).
- The 4DRGS held-out PSNR is reported as 0 at 133 views: all frames are
  training views, so 4DRGS test.py has no held-out view (not an error).

**Views trend:** Q error 7.8 / 5.3 / 3.8 %, diameter 0.171 / 0.104 / 0.112 mm,
FFR MAE 0.063 / 0.030 / 0.040 (0.057 / 0.029 / 0.029 without 131_1) at
30 / 60 / 133 views. The Step 2.2 DoD holds at every view count.

### Noise and Peclet (5 validation cases, 60 views, k = 1.086)

Oct 7, molab, repo `2e02924`. Middle level = run 8 on the same 5 cases.

| Setting | 4DRGS v2 pass | Area ratio (mean) | Q err mean | Q signed | Diam. err mean | FFR MAE / acc. (case CFR) |
|---|---|---|---|---|---|---|
| noise low (I0 1e6, sigma 1) | 5 / 5 | 0.905 | 6.1 % | +6.1 % | 0.111 mm | 0.083 / 100 % |
| **default (run 8, I0 1e5, sigma 10)** | 5 / 5 | 0.877 (20 cases) | 4.4 % | +1.7 % | 0.083 mm | 0.060 / 80 % |
| noise high (I0 1e4, sigma 30) | 1 / 5 | 0.669 | **20.9 %** | -18.3 % | 0.157 mm | 0.025 / 100 % |
| D = 1e-5 m^2/s (Pe 10x lower) | 5 / 5 | 0.871 | 3.6 % | +1.8 % | 0.115 mm | 0.070 / 80 % |
| D = 1e-4 m^2/s (Pe 100x lower) | 5 / 5 | 0.872 | 4.5 % | +4.1 % | 0.104 mm | 0.073 / 100 % |

(Pe = u L / D; for 121_0 about 21000 at the default D, so about 2400 and 240
at the two sweep levels; PUNCH's reported value of about 4000 lies between.)

- **Peclet: robust.** Q error 3.6 % and 4.5 % even at 100x more dispersion,
  so the plug-flow model holds well beyond PUNCH's regime. The fitted
  model ignores dispersion; the half-rise arrival used by the fit is
  little affected by symmetric front spreading.
- **Noise low: unchanged** within case-to-case variation.
- **Noise high: Q fails (20.9 %, 1 of 5 below 10 %).** Same mechanism as
  in Step 2.2: the 4DRGS half-max radius gets much thinner on noisy data
  (area ratio 0.67 instead of 0.88), and k calibrated at the default noise
  level does not correct it, so Q is underestimated (-18 %; 130_0 -47 %).
  The calibration is noise-level specific.
- FFR at high noise looks good (0.025) only because two errors cancel: the
  too-small throat lowers FFR, the too-small Q raises it (estimated lesion
  with true Q: 0.239). Not a real robustness.
- 4DRGS held-out PSNR follows the noise: 37 to 44 dB (low), 32 to 38 dB
  (default), 26 to 32 dB (high).
- Per-case FFR on 5 cases is dominated by the severe lesions 131_0 and
  135_0 (throat errors of 0.1 mm); FFR MAE differences between these
  settings are within that variation.

### Flow speed and foreshortening (5 validation cases, 60 views, k = 1.086)

Oct 7, molab, repo `8e64d89`.

| Setting | Q err mean (median) | Q < 10 % | Diam. err mean | FFR MAE / acc. (case CFR) | FFR lesion misplaced |
|---|---|---|---|---|---|
| rest velocity 0.10 m/s | 3.3 % (1.8 %) | 5 / 5 | 0.145 mm | 0.048 / 100 % | 0 |
| default (run 8, 0.10 to 0.20 m/s) | 4.4 % | 5 / 5 | 0.083 mm | 0.060 / 80 % | 0 |
| rest velocity 0.25 m/s | 4.4 % (3.3 %) | 5 / 5 | 0.058 mm | (not valid, see below) | 0 |
| rest velocity 0.40 m/s | 6.3 % (5.4 %) | 4 / 5 | 0.139 mm | (not valid) | 0 |
| out-of-plane 0 deg | 4.5 % | 5 / 5 | 0.118 mm | 0.081 / 80 % | 0 |
| out-of-plane 30 deg | 1.7 % | 5 / 5 | 0.053 mm | 0.040 / 100 % | 0 |
| out-of-plane 60 deg | 4.9 % | 4 / 5 | 0.148 mm | 0.531 / 80 % (0.073 without 135_0) | 1 |

- **Flow speed: robust.** Q error 3.3 to 6.3 % from slow rest to
  hyperemia-like velocity (0.40 m/s: transit about 0.3 s, about 4 view
  intervals at 60 views). Slightly larger at 0.40 m/s, consistent with
  fewer views per transit (plan risk "temporal sampling").
- **FFR is not evaluated for the 0.25 and 0.40 m/s levels**: the
  ground-truth FFR is unphysical there (Young-Tsai at CFR x 0.25 to 0.40
  m/s gives FFR <= 0 for the severe lesions, e.g. 135_0 -1.40), as stated
  in sweeps.with_flow (FFR_MIN of data_gen not enforced in the sweep).
- **Foreshortening: Q robust** (1.7 to 4.9 %), as expected for a 3D method
  (the reconstruction sees the vessel from 198 deg of views; the angle only
  matters for 2D methods, Step 4.1 baseline 2).
- **fore_60, 135_0:** the GT-free lesion finder picked a false narrowing
  24.7 mm from the true throat (estimated R 0.40 mm), FFR -1.76. Q and the
  diameter in the true lesion window are fine. The new column "FFR lesion
  misplaced" (throat found > 5 mm from the true throat) counts these: 1 in
  fore_60, 1 each at 30 and 133 views (122_2 16.8 mm, 121_3 14.9 mm), 0
  elsewhere, 0 in run 8.

### 4DRGS repeats (Gate 1 addition; 5 validation cases, 60 views)

Oct 7, molab, repo `9889280`. Four independent 4DRGS runs on the same data
(run 8 + repeat 1 to 3), each followed by the same Step 2.2 fit.

| Run | Q err mean | Diam. err mean | FFR MAE / acc. | 4DRGS v2 pass |
|---|---|---|---|---|
| run 8 | 4.4 % | 0.083 mm | 0.060 / 80 % | 5 / 5 |
| repeat 1 | 3.3 % | 0.098 mm | 0.068 / 80 % | 5 / 5 |
| repeat 2 | 4.0 % | 0.100 mm | 0.063 / 80 % | 5 / 5 |
| repeat 3 | 4.5 % | 0.094 mm | 0.065 / 80 % | 5 / 5 |

Per-case standard deviation over the 4 runs (results/step3.1_sweeps.json,
repeat_spread): Q 0.1 to 1.4 points (mean 0.8), diameter 0.001 to 0.052 mm
(mean 0.022), FFR 0.001 to 0.053 (mean 0.021; largest on the severe lesion
130_0, FFR 0.51 to 0.63 against 0.577). The 4DRGS randomness is small
compared with the case-to-case errors; the run 6 vs run 7 change on 131_0
(0.43 vs 0.18 mm) was the extreme of this spread together with the
different k.

## Summary of all sweeps (frozen, Gate 1 method)

| Factor | Levels | Q err mean | Diam. err mean (mm) | FFR MAE (case CFR) |
|---|---|---|---|---|
| Views (20 val) | 30 / **60** / 133 | 7.8 / **5.3** / 3.8 % | 0.171 / **0.104** / 0.112 | 0.063 / **0.030** / 0.040 |
| Noise (5 val) | low / **default** / high | 6.1 / **4.4** / 20.9 % | 0.111 / **0.083** / 0.157 | 0.083 / **0.060** / (0.025, errors cancel) |
| Dispersion (5 val) | **default** / D 1e-5 / D 1e-4 | **4.4** / 3.6 / 4.5 % | **0.083** / 0.115 / 0.104 | **0.060** / 0.070 / 0.073 |
| Rest velocity (5 val) | 0.10 / **default** / 0.25 / 0.40 m/s | 3.3 / **4.4** / 4.4 / 6.3 % | 0.145 / **0.083** / 0.058 / 0.139 | 0.048 / **0.060** / n.v. / n.v. |
| Out-of-plane angle (5 val) | 0 / 30 / 60 deg | 4.5 / 1.7 / 4.9 % | 0.118 / 0.053 / 0.148 | 0.081 / 0.040 / 0.531 (0.073 w/o 135_0) |
| 4DRGS repeats (5 val) | 4 runs | 3.3 to 4.5 % | 0.083 to 0.100 | 0.060 to 0.068 |

Bold: run 8 (middle level). n.v.: ground-truth FFR not valid at that flow.

**Answer to the week question ("Are results robust?").** Q is robust to the
number of views, dispersion (Pe down to about 240), flow speed up to
0.40 m/s, out-of-plane angle and 4DRGS randomness (all 1.7 to 7.8 %). It
is not robust to high noise (20.9 %), because the radius calibration is
noise-level specific. FFR follows the lesion geometry: about 0.03 at 60
views and above, larger at 30 views, and it fails when the GT-free lesion
finder picks a false narrowing (3 of about 230 runs).

## DoD

| DoD part | Result | Met |
|---|---|---|
| Sweep tables saved | results/step3.1_sweeps.json; tables above (views, noise, Peclet, flow speed, foreshortening, repeats) | **Yes** |
| FFR MAE and accuracy at 0.80 on the validation set | MAE 0.030, accuracy 90 % (sens. 3 / 4, spec. 15 / 16), 20 val cases, 60 views, case CFR; population CFR: 0.079 / 85 % | **Yes** |

**DoD met: yes. Fallback used: no** (plan fallback "If FFR error is large:
decompose with the oracle; center the paper on Q and present FFR as an
application": the oracle decomposition is reported anyway; FFR error is
not large at 60 views).

## Known limitations

- Radius calibration k depends on the view count (recalibrated) and on the
  noise level (not recalibrated: high noise fails).
- FFR needs the CFR (case CFR assumed; population CFR adds about 0.05 MAE).
- Only 4 of 20 validation cases have FFR <= 0.80.
- GT-free lesion finder: 3 false picks in about 230 runs.
- Flow-sweep ground-truth FFR is unphysical at 0.25 and 0.40 m/s (FFR_MIN
  of data_gen not enforced there).

