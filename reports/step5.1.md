# Step 5.1: Y-bifurcation with continuity and Murray prior

Dates: Oct 27 to Nov 1, 2026 (plan; done Oct 8 to 9). Status: **closed, DoD
met on the fitted test cases** (primary variant cont_murray: mean split error
6.3 % on 20 of 24 test cases; 4 cases without a usable tree count as
failures). Fallback: not used; whether "the bifurcation works" is the Gate 2
question.

Plan DoD: "Flow split error below 10% on bifurcation test cases; ablation
table." Plan fallback: "If it does not work by Nov 2: submit with single
vessel only; bifurcation becomes future work. The paper remains valid."

## Protocol (fixed Oct 8, before any result)

- **Metric (user decision, Oct 8).** f_k = Q_k / Q_p for the two daughters;
  split error = mean over k of |f_k,est - f_k,gt| / f_k,gt per case; DoD:
  mean over the test cases < 10 %. Also reported: relative error of each
  daughter flow and of the parent flow.
- **Data.** ImageCAS LCA skeleton: LAD path as in Step 1.1; diagonal = first
  side tree after the left main bifurcation with length >= 30 mm, entry
  radius >= 0.6 mm, within the proximal 70 % of the LAD, not thicker than
  1.1 x the LAD daughter. One stenosis on the LAD (>= L/2 + 5 mm from the
  junction). Q_p from the rest velocity (PUNCH range) x mean parent area;
  Murray-consistent split Q_k ~ r_k^3 (plan, Dataset). Bolus per branch
  (PUNCH Eq 2), daughters fed by the parent's outlet. Same rendering
  protocol and noise as Step 1.2.
- **Splits.** 40 of 95 ImageCAS patients have a usable Y (scan Oct 8).
  dev = Step 1 train / val patients (16 geometries); test = Step 1 test
  patients with a usable Y (116, 132) + 22 ImageCAS patients never used
  before (24 geometries). Runs use case _0 of each geometry.
- **Method (Gate 1 method, unchanged).** 4DRGS (60 views, 10k iterations),
  GT-free tree from the volume (longest path + longest side branch),
  half-max radius x k = 1.086 (single-vessel training cases), frozen
  geometry, Stage A fit (2100 iterations). Inlet hypothesis chosen with
  continuity, then all variants on it.
- **Ablation (plan).** indep (Q_p, Q_1, Q_2 free), cont (Q_p = Q_1 + Q_2),
  murray (soft prior only), cont_murray (both). With frozen geometry the
  Murray prior acts on the flows: LAMBDA_MURRAY (log(Q_1/Q_2) - 3
  log(R_1/R_2))^2. Primary variant: cont_murray (plan's full model).
- Caveat: the ground-truth split is Murray-consistent by construction (plan),
  so the Murray prior is favoured by the data; the ablation shows how much.

## Where things run

| Piece | Runs on |
|---|---|
| `src/flowsplat/tree.py` (data, bolus, extraction, fit, metrics) | M4 (data, tests) and molab |
| `python -m flowsplat.tree` | M4: data/trees_v1, data/step5_tree_pack.zip |
| `notebooks/step5_tree.py` | molab GPU |
| `tests/test_tree.py` | M4 (toy Y fit recovers the split); whole suite 114 tests pass |

## Data (M4, Oct 8, repo `2662c6e`)

`python -m flowsplat.tree --imagecas_dir <ImageCAS> --out data/trees_v1
--per_geom 2 --pack data/step5_tree_pack.zip` (about 15 min, 12.9 MB pack).
40 geometries, 80 cases: dev 16 geometries (32 cases), test 24 geometries
(48 cases); runs on case _0: 16 dev, 24 test
(results/step5.1_splits_tree.json, results/step5.1_data_summary.json, QC
figures results/step5.1_qc/).
- Diagonal flow fraction 0.14 to 0.51 (median 0.31).
- Stenosis in the parent in 30 of 80 cases, else in the LAD daughter; FFR
  <= 0.80 in 28 of 80.
- Rest transit: parent 0.11 to 0.91 s, LAD daughter 0.42 to 2.61 s,
  diagonal 0.41 to 2.06 s (each branch spans several 60-view intervals of
  0.074 s).

## Results

### Dev run 1 (pipeline check; Oct 9, molab, repo `50fc354`)

results/step5.1_tree_dev_run1.json. 16 dev cases: 13 fitted, 3 errors.
- **Bug 1, wrong inlet (4 of 13: 115, 121, 130, 572).** The inlet was
  chosen among the two ends of the longest skeleton path. In all four the
  diagonal is longer than the parent (e.g. 121: parent 25 mm, diagonal
  43 mm), so the longest path runs LAD tip -> diagonal tip and the parent is
  the side branch; the true inlet was never a candidate. Then the parent and
  a daughter are swapped and the flows are meaningless (split error 1.1 to
  1.4 with continuity). Fix: all three free ends are inlet hypotheses
  (tree.orientations), still chosen by the fit loss, no ground truth.
- **Bug 2, no side branch (3 of 16: 594, 609, 656).** Diagonals of 28 to
  33 mm came out shorter than the 20 mm minimum in the 4DRGS skeleton (ends
  trimmed); on the ground-truth lumen volume all three are found. Fix:
  minimum 10 mm (the longest side branch is taken anyway; the minimum only
  guards against skeleton spurs).
- **Where the inlet was right (9 cases)**, the method works: with continuity
  the split error is 0.2 to 4.1 % (mean 1.8 %), parent flow error 1 to 9 %.
  The Murray prior did not help there (cont_murray 1.4 to 14.6 %): it uses
  the estimated daughter radii, which are less reliable near the junction.
- Nothing was tuned on these results; both changes are bug fixes of the
  GT-free tree extraction. Dev is run again before the test set.

### Dev run 2 (after the fixes; Oct 9, repo `313b8eb`)

results/step5.1_tree_dev_run2.json. 15 of 16 fitted, inlet correct in 15 of
15 (three hypotheses); 609_0 still has no side branch in the 4DRGS volume
(diagonal 28 mm).

| Variant | Split err. mean (median) | < 10 % | Daughter Q err. | Parent Q err. |
|---|---|---|---|---|
| indep | 7.5 % (4.2 %) | 12 / 15 | 7.1 % | 5.6 % |
| cont | 2.3 % (1.7 %) | 15 / 15 | 5.5 % | 4.5 % |
| murray | 10.5 % (10.9 %) | 7 / 15 | 7.3 % | 6.8 % |
| cont_murray (primary) | 4.9 % (4.6 %) | 14 / 15 | 6.5 % | 4.3 % |

Pipeline works; the test run follows unchanged. Rule fixed now for the
test set (before running it): a case without a usable tree (error) counts
as a failure in the "< 10 %" share, and the mean split error is reported
over the fitted cases together with the number of failures.

### Test run (DoD; Oct 9, molab, repo `313b8eb`)

results/step5.1_tree_test.json (built on the M4 from the 24 per-case
result.json files: the molab summary write failed because results/ was
missing in that checkout; the notebook now creates it). 24 test cases: 20
fitted, inlet correct in 20 of 20; 4 without a usable tree (669_0, 733_0,
741_0, 799_0).

**Ablation table (20 fitted test cases):**

| Variant | Continuity | Murray prior | Split err. mean (median) | < 10 % (of 20) | < 10 % (of 24, failures count) | Daughter Q err. | Parent Q err. |
|---|---|---|---|---|---|---|---|
| indep | no | no | 17.2 % (12.8 %) | 9 | 9 | 7.6 % | 20.9 % |
| murray | no | yes | 18.2 % (13.7 %) | 8 | 8 | 9.8 % | 22.3 % |
| cont | yes | no | **3.8 % (2.1 %)** | **17** | **17** | 6.2 % | 5.4 % |
| **cont_murray (primary)** | yes | yes | **6.3 % (4.6 %)** | 15 | 15 | 8.6 % | 5.8 % |

- **DoD (pre-registered rule):** mean split error of the primary variant
  6.3 % < 10 % on the fitted cases; 4 of 24 cases failed tree extraction.
- **Continuity is the effective constraint:** it cuts the split error from
  17.2 to 3.8 % and the parent flow error from 20.9 to 5.4 %. Without it,
  the parent flow is fitted almost independently of the daughters.
- **The Murray prior does not help:** +2.5 points with continuity, +1.0
  without, although the ground-truth split is Murray-consistent by
  construction. The prior uses the estimated daughter radii next to the
  junction, where the 4DRGS radius is least reliable (cube of the radius
  ratio amplifies their error). Same pattern on dev (2.3 vs 4.9 %).
- **Extraction failures (4 / 24 test, 1 / 16 dev):** all five have a short
  parent (18 to 24 mm; median diagonal of the fitted cases 41 mm). With a
  short parent the longest skeleton path is LAD -> diagonal and the parent
  must be found as the side branch; in the 4DRGS volume it is not (on the
  ground-truth lumen volume it is, 22 to 27 mm). Likely cause: the parent is
  the earliest filled and washed-out segment, dim in the time-averaged
  volume (as 131_0 in Step 2.1); not verified (4DRGS volumes not
  downloaded). Not fixed after seeing the test set.

## DoD

| DoD part | Result | Met |
|---|---|---|
| Flow split error below 10 % on bifurcation test cases | Primary (cont_murray) 6.3 % mean on 20 / 24 fitted (15 of 24 cases below 10 %); cont 3.8 % (17 of 24) | **Yes** (fitted cases; 4 extraction failures) |
| Ablation table | Table above (continuity x Murray) | **Yes** |

**DoD met: yes, with 4 of 24 extraction failures. Fallback used: no.**

## For Gate 2 (Nov 2): "Bifurcation works?"

- Yes for 20 of 24 test Ys (split error 6.3 % primary, 3.8 % with
  continuity only); the GT-free tree extraction fails on short parents
  (< 25 mm).
- The full model (continuity + Murray) is not the best variant;
  continuity alone is. Which variant the paper reports as FlowSplat is a
  Gate 2 decision (the pre-registered primary is cont_murray).

