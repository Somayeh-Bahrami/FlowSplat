# Step 4.1: Baselines and main table

Dates: Oct 27 to Nov 1, 2026 (plan; done Oct 7 to 8). Status: **closed, DoD
met** (one table, all methods, 30 held-out test cases). Fallback: not
triggered (FlowSplat beats baseline 1 on average), but the margin is small
and not significant; see "For Gate 2".

Plan DoD: "One results table (Q error, FFR MAE, FFR accuracy, diameter
error) on the held-out test set for all methods."
Plan fallback: "If FlowSplat does not beat baseline 1 on average: show the
advantage in hard regimes (few views, fast flow, strong foreshortening)."

## Protocol (fixed Oct 7, before any test result)

Test split: all 30 cases of the 6 test patients (116, 132, 133, 612, 639,
641); 60 views; never used before. 4 of 30 have FFR <= 0.80.

| Method | Geometry | Flow | FFR |
|---|---|---|---|
| FlowSplat (Gate 1 method) | 4DRGS x k, k = 1.086 (training split) | transport fit, variant frozen | lesion finder + case CFR (primary), population CFR (secondary) |
| B1: 4DRGS + TDC | same 4DRGS models and geometry | half-rise arrival times of the 4DRGS time-density curves at our centerline; v from t = t0 + s / v; Q = v x mean area | same |
| B2: PUNCH-style 2D | same geometry (for area and lesion) | fixed-view angiogram (gantry angle 0), kymograph along the projected GT centerline, isocenter scaling, plug-flow least squares; Q = u x mean area | same |
| B3: R2-Gaussian + 0.35 m/s | static R2-Gaussian at defaults, no calibration | none at rest; Q_hyp = 0.35 m/s x A_n | lesion finder on the R2-Gaussian profile |
| B4: Oracle | ground truth | ground truth | exact by construction; plus through the lesion finder (extraction floor) |

- B1 and B2 share FlowSplat's geometry on purpose: their difference to
  FlowSplat is the flow estimate only. B2 is given the true centerline
  (an advantage).
- Diameter error: same rule for every method (metrics.diam_err_from_geometry).
- Comparison "beat baseline 1 on average": mean Q error and FFR MAE.

## Where things run

| Piece | Runs on |
|---|---|
| `src/flowsplat/baselines/tdc.py`, `kymo.py`, `table.py` | M4 (tests, evaluation) and molab |
| `src/flowsplat/baselines/drgs_tdc.py` | molab, inside the 4DRGS repo |
| `notebooks/step4_baselines.py` | molab GPU |
| `tests/test_baselines.py` | M4; whole suite 105 tests pass |

## Results

### Run 1: FlowSplat, B1, B2, oracle (Oct 8, molab, repo `0103514`)

results/step4.1_test_table_run1.json; per-case artifacts in data/step4.1_run1
(gitignored). 4DRGS on the 30 test cases: lesion coverage 100 % on all, 27 of
30 pass the Step 2.1 v2 radius rule.

| Method | Q err mean (median) | Q < 10 % | Diam. err mean | FFR MAE / acc. (case CFR) | Sens. / spec. | FFR MAE / acc. (pop. CFR) |
|---|---|---|---|---|---|---|
| **FlowSplat** | **5.5 % (5.5 %)** | 27 / 30 | 0.091 mm | **0.031 / 96.7 %** | 4/4, 25/26 | 0.084 / 90 % |
| B1: 4DRGS + TDC | 6.4 % (6.7 %) | 24 / 30 | 0.091 mm (same geometry) | 0.033 / 96.7 % | 4/4, 25/26 | 0.094 / 90 % |
| B2: PUNCH-style 2D | 24.3 % (23.2 %) | 1 / 30 | 0.091 mm (same geometry) | 0.039 / 90 % | 1/4, 26/26 | 0.046 / 93 % |
| B3: R2-Gaussian + 0.35 m/s | pending | | | | | |
| B4: oracle | 0 | 30 / 30 | 0 | 0 (0.009 through the lesion finder) | | |

- **FlowSplat vs B1:** lower mean Q error (5.5 vs 6.4 %) and FFR MAE (0.031 vs
  0.033), better on 17 of 30 cases; the paired difference is not
  significant (Wilcoxon p = 0.52). With the same 4DRGS geometry and
  calibration, the TDC velocity of the 4DRGS 4D model is almost as good
  as the transport fit on this data.
- **B2 (2D):** Q error 24 %, the velocity is underestimated; the error grows
  with the out-of-plane angle of the vessel at the fixed view (about 17 to
  20 % at 11 deg, 33 to 46 % at 35 deg), which is the foreshortening bias
  the plan expects (plus isocenter calibration and tortuosity). Its FFR is
  less hurt because a too small flow raises FFR: sensitivity 1 / 4.
  With the population CFR the two biases partly cancel (MAE 0.046).
- FlowSplat signed Q bias on test +2.6 %.
- GT-free lesion finder: 6 of 30 test cases misplaced (throat found 16 to
  78 mm away), all mild lesions (FFR 0.94 to 0.99), so their FFR stays
  close (largest: 133_1, 0.992 -> 0.927).

### B3 chunk 1: R2-Gaussian + 0.35 m/s, 10 of 30 test cases (Oct 8, repo `7a40c8d`)

Cases 116_0 to 132_4 (fresh session). Merged table on the M4:
results/step4.1_test_table.json (all methods, B3 on 10 cases so far).

- Static R2-Gaussian geometry on the dynamic views is poor, as in Step 2.1
  run 1: radius MAE 0.24 to 0.55 mm, stenosis diameter error 0.65 mm
  (mean, 8 of 10 lesions in the window; 1 / 8 below 0.2 mm).
- Hyperemic flow from 0.35 m/s x A_n: error 35 % (mean).
- FFR: the GT-free lesion finder picks false narrow throats on this
  geometry, where Young-Tsai diverges (FFR down to -33); raw MAE 7.5 is not
  meaningful. With FFR clipped to [0, 1] (added Oct 8 after seeing these
  values, reported next to the raw metric): MAE 0.61; accuracy at 0.80:
  0 / 10 (every case called significant; none is).
- Chunks 2 (133_*, 612_*) and 3 (639_*, 641_*) pending; notebook now takes
  FLOWSPLAT_B3_CASES so a fresh session does not repeat chunk 1.

### B3 chunk 2: 133_* and 612_* (Oct 8, repo `729943c`)

- R2-Gaussian radius MAE: 133_* 0.42 to 1.39 mm (patient 133 fails badly),
  612_* 0.19 to 0.27 mm; 12 to 23 min per case.
- B3 on 20 of 30 cases: hyperemic Q error 41.6 % (mean), diameter error
  0.57 mm (13 of 20 lesions in the window, 1 / 13 below 0.2 mm), FFR raw MAE
  10.1 (clipped 0.71), accuracy 15 % (all 20 called significant).

### B3 chunk 3: 639_* and 641_* (Oct 8, repo `637d489`)

R2-Gaussian radius MAE 0.13 to 0.50 mm. B3 complete on all 30 test cases.

## Main table (held-out test split, 30 cases, 60 views)

results/step4.1_test_table.json (built on the M4 from the per-case
artifacts in data/step4.1_merged; run 1 table identical for FlowSplat, B1,
B2).

| Method | Q error mean (median) | Q < 10 % | Diam. err mean (lesion found) | FFR MAE | FFR acc. at 0.80 | Sens. / spec. | FFR MAE, pop. CFR |
|---|---|---|---|---|---|---|---|
| **FlowSplat** | **5.5 % (5.5 %)** | **27 / 30** | **0.091 mm (30 / 30)** | **0.031** | **96.7 %** | 4/4, 25/26 | 0.084 |
| B1: 4DRGS + TDC | 6.4 % (6.7 %) | 24 / 30 | 0.091 mm (same geometry) | 0.033 | 96.7 % | 4/4, 25/26 | 0.094 |
| B2: PUNCH-style 2D | 24.3 % (23.2 %) | 1 / 30 | 0.091 mm (same geometry) | 0.039 | 90.0 % | 1/4, 26/26 | 0.046 |
| B3: R2-Gaussian + 0.35 m/s | 43.1 % (45.9 %) (hyperemic) | 3 / 30 | 0.519 mm (22 / 30) | 6.94 raw, 0.591 clipped | 26.7 % | 4/4, 4/26 | n.a. |
| B4: oracle | 0 | 30 / 30 | 0 | 0 (0.009 via lesion finder) | 100 % | 4/4, 26/26 | n.a. |

4 of 30 test cases have FFR <= 0.80.

## DoD

| DoD part | Result | Met |
|---|---|---|
| One results table (Q error, FFR MAE, FFR accuracy, diameter error) on the held-out test set for all methods | Table above, 5 methods x 30 cases | **Yes** |

**DoD met: yes. Fallback used: no.** Plan fallback "If FlowSplat does not
beat baseline 1 on average: show the advantage in hard regimes (few views,
fast flow, strong foreshortening)" is not triggered: FlowSplat has the
lower mean Q error (5.5 vs 6.4 %) and FFR MAE (0.031 vs 0.033).

## For Gate 2 (Nov 2): "Beat baselines on average?"

- vs B1 (4DRGS + TDC): yes on average, small margin, better on 17 / 30
  cases, paired Wilcoxon p = 0.52 (not significant). Both use the same
  4DRGS geometry and radius calibration; the difference is the flow
  estimate only.
- vs B2 (2D): clearly (Q 5.5 vs 24.3 %; FFR sensitivity 4/4 vs 1/4); the 2D
  error grows with the out-of-plane angle (foreshortening).
- vs B3 (static geometry + fixed velocity): clearly (geometry 0.09 vs
  0.52 mm, FFR accuracy 96.7 vs 26.7 %).
- The hard-regime comparison against B1 (30 views, fast flow, noise) has
  not been run; it is the plan's fallback content and a Gate 2 decision.

## Known limitations

- B1 is close to FlowSplat on clean synthetic data at 60 views.
- Clipped FFR metric for B3 added after seeing its values (labelled).
- 6 of 30 lesions misplaced by the GT-free lesion finder (mild lesions).
- B2 is given the true centerline and our 3D area (favourable to B2).

