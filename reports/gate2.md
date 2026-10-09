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
