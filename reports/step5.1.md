# Step 5.1: Y-bifurcation with continuity and Murray prior

Dates: Oct 27 to Nov 1, 2026 (plan; started Oct 8). Status: **in progress**
(code and data on the M4 done; molab runs pending).

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

Pending (molab).
