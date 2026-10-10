# Gate 2 review (held Oct 9, 2026, scheduled Nov 2)

Called by the user (approved order: Gate review, hard-regime comparison,
then Step 6.1).

## Conditions and pre-committed decisions (plan)

| Condition | Evidence | Pre-committed decision | Triggered |
|---|---|---|---|
| Bifurcation works? | 20 of 24 test Ys fitted; split error 3.8 % (continuity) / 6.3 % (continuity + Murray); 4 short-parent extraction failures (reports/step5.1.md) | No: single vessel only | No |
| Beat baselines on average? | Test Q error FlowSplat 5.5 % vs B1 6.4 % (p = 0.52), B2 24.3 %, B3 43.1 %; FFR MAE 0.031 / 0.033 / 0.039 / 0.59 (reports/step4.1.md) | No: present hard-regime advantage | No (but small margin over B1) |

## Decisions (user, Oct 9)

1. **Keep the bifurcation** in the paper; short-parent failures as a limitation.
2. **Bifurcation variant reported as FlowSplat: continuity only** (test 3.8 %).
   Decided here, after the pre-registered primary (continuity + Murray,
   6.3 %) was found worse on dev and test; the paper states this and shows
   all four variants in the ablation (Murray prior = negative result).
3. **Hard-regime comparison FlowSplat vs B1** (same 4DRGS geometry, only the
   flow estimate differs) on all 30 held-out test cases in 3 regimes:
   30 views (k = 1.126, recalibrated in Step 3.1), fast flow (rest velocity
   0.40 m/s, k = 1.086), high noise (I0 1e4, Gaussian sigma 30, k = 1.086).
   Comparison: mean Q error and paired Wilcoxon test per regime; FFR
   reported where the ground truth is valid (not at 0.40 m/s).
4. **Venue decided after the comparison:** CVPR if FlowSplat clearly beats B1
   in the hard regimes, else MICCAI 2027 / CVPR workshop (same draft).

## Hard-regime comparison (Oct 9 to 10, molab, repo `38d95a3`)

30 held-out test cases per regime; FlowSplat and B1 share the 4DRGS
reconstruction and k, so only the flow estimate differs.
results/gate2_views30.json, gate2_flow040.json, gate2_noise_high.json.

| Regime | FlowSplat Q err. mean (median) | B1 Q err. mean (median) | FlowSplat better (of 30) | Wilcoxon p | Q < 10 % FS / B1 |
|---|---|---|---|---|---|
| 60 views, default (Step 4.1) | 5.5 % (5.5 %) | 6.4 % (6.7 %) | 17 | 0.52 | 27 / 24 |
| 30 views | 9.0 % (9.6 %) | 8.3 % (6.9 %) | 13 | 0.23 | 16 / 21 |
| **Fast flow 0.40 m/s** | **7.0 % (6.1 %)** | **10.7 % (9.5 %)** | **19** | **0.023** | **25 / 16** |
| High noise | 14.1 % (9.8 %) | 13.4 % (8.7 %) | 13 | 0.69 | 16 / 18 |

- **Fast flow:** the only regime with a clear, significant advantage
  (p = 0.023; 0.07 after Bonferroni for 3 regimes). With a transit of a few
  view intervals, the arrival-time regression of B1 is undersampled,
  while the transport fit uses every view jointly.
- **30 views and high noise:** no advantage (B1 slightly better, not
  significant). Both share the geometry error (high noise: area ratio
  0.73, 17 of 30 lesions misplaced by the lesion finder; 30 views: area
  ratio 0.81).
- FFR (case CFR): 30 views 0.031 vs 0.029, high noise 0.272 vs 0.249 (both
  fail with the geometry); at 0.40 m/s the ground-truth FFR is not valid
  (Step 3.1), so not compared.

**Answer to decision 4 (pre-committed rule: CVPR if FlowSplat clearly beats
B1 in the hard regimes):** not met. FlowSplat beats B1 clearly in one of
three hard regimes (fast flow) and matches it in the other two. By the rule
fixed at this gate the venue is MICCAI 2027 / CVPR workshop (same draft);
confirmation by the user pending.

