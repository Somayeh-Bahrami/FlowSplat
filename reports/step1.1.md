# Step 1.1: Geometry, stenoses, flow and FFR ground truth

Dates: Oct 5 to Oct 8, 2026 (started Oct 3). Status: **in progress**. Code
and tests done on the M4; the run on real ImageCAS labels is pending (the
labels are not on the M4 yet).

Plan DoD: "About 200 cases, each with geometry, Q and FFR saved; ranges
checked (rest 10 to 20 cm/s)."

## Where things run

| Piece | Runs on |
|---|---|
| `src/flowsplat/centerline.py`, `data_gen.py`, `ffr.py` | M4 (CPU) |
| `tests/test_data_gen.py`, `tests/test_ffr.py` | M4 (CPU), 37 tests pass in total (whole suite) |

Command (M4):

```bash
python -m flowsplat.data_gen --imagecas_dir <dir with *.label.nii.gz> --out data/cases_v1 --n_geom 40 --per_geom 5 --seed 0
```

## What was done

1. **Reused extraction.** The owner's earlier code (repo
   cs6999-coronary-centerline, commit `f17a5f0`) copied verbatim into
   `src/flowsplat/centerline.py`: skeleton + radius in mm + DFS order
   (`centerline.py`), LCA/RCA split by the NIfTI L-R axis and isotropic
   nearest-neighbour resampling (`build_dataset_v3.py`).
2. **LAD choice (OURS).** ImageCAS has one binary label for all coronaries
   (Sec 3, Fig. 1 caption). Root = widest skeleton endpoint; LAD = the most
   anterior leaf among leaves at least 0.7 x the longest geodesic from the
   root. One QC figure per geometry (`qc/<id>.png`) for the spot check.
3. **Path smoothing (OURS).** Positions smoothed (1 mm) before arc length is
   measured, radius median + Gaussian (2 mm), uniform 0.5 mm sampling. QC
   gate: length 60 to 250 mm, radius 0.3 to 3 mm.
4. **Stenosis.** Diameter reduction U(30, 70) % (plan); cosine profile,
   length U(5, 15) mm, centre in the proximal 60 % (OURS).
5. **Flow.** Rest velocity U(10, 20) cm/s in the normal segment at the
   lesion; hyperemic = rest x CFR, CFR U(1.5, 4.0) (PUNCH, Materials and
   Methods), truncated so V_hyp <= 0.40 m/s (plan).
6. **FFR.** Young-Tsai at hyperemic flow, P_a 90 mmHg (`ffr.py`). Tests
   reproduce the plan's worked example exactly (422 + 1305 Pa, 12.96 mmHg,
   FFR 0.856; 0.798 and 0.888 for the two error cases).
7. **Timing check.** Transit time tau = int A/Q ds at rest and the number of
   view intervals per transit for 30 / 60 / 133 views over 4.4 s.

Per case saved: `s`, `xyz_mm`, `r_ref_mm`, `r_mm` (with stenosis),
`tau_rest_s`, and all parameters (stenosis, Q_rest, Q_hyp, CFR, dP terms,
FFR, FFR at rest). Plus `cases.csv` and `summary.json` (ranges, DoD check,
timing check, rejects).

## Bugs found by the tests and fixed

| Problem | Fix |
|---|---|
| Arc length about 8 % too long (voxel staircase measured before smoothing) | Smooth first, then measure (test on an oblique staircase line) |
| Throat radius taken as the vessel minimum: on a tapering LAD the distal end can be narrower than the throat | Throat radius = r_ref(s0) x (1 - degree), exact |
| Young-Tsai at fixed hyperemic flow gives dP > P_a (FFR < 0) for severe lesions, e.g. 70 % at 0.4 m/s: dP about 98 mmHg | Flow redrawn until FFR >= 0.3 (OURS; floor UNVERIFIED). Share of redrawn cases is reported |

## Numbers so far (synthetic LAD profile, not ImageCAS)

200 draws on a 120 mm LAD, radius 1.8 to 0.9 mm:

| Quantity | Value |
|---|---|
| FFR (hyperemic) | min 0.31, median 0.90, max 0.99; 32 % below 0.80 |
| Flow redrawn (FFR floor) | 7 % of cases |
| Median dP terms | viscous 4.3 mmHg, expansion 5.4 mmHg |
| Transit at rest | 0.40 to 1.36 s (median 0.69) |
| Views per transit, 30 / 60 / 133 views | min 2.7 / 5.5 / 12.1 |

The plan's concern (a vessel crossed within one view interval) does not
occur at rest for a full LAD: at least 2.7 view intervals with 30 views.

## DoD

| DoD part | Met |
|---|---|
| About 200 cases with geometry, Q, FFR saved | No (ImageCAS run pending) |
| Ranges checked (rest 10 to 20 cm/s) | Check implemented (`check_rest_10_20_cms`); real run pending |

**DoD met: no (in progress). Fallback used: no.**

Plan fallback (only if ImageCAS extraction causes problems): "procedural
vessels with realistic ranges; ImageCAS moves to supplementary."

## Open

- ImageCAS label files on the M4 (about 40 good LCAs needed; the reject
  rate of the LCA split and QC is not known yet).
- Visual spot check of the LAD choice on the QC figures.
- Design choices marked OURS above (lesion length and position, FFR floor,
  LAD rule) are open to the owner's review.
