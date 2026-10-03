# Step 1.1: Geometry, stenoses, flow and FFR ground truth

Dates: Oct 5 to Oct 8, 2026 (done early, Oct 3). Status: **done, DoD met.**

Plan DoD: "About 200 cases, each with geometry, Q and FFR saved; ranges
checked (rest 10 to 20 cm/s)."

## Where things run

| Piece | Runs on |
|---|---|
| `src/flowsplat/centerline.py`, `data_gen.py`, `ffr.py` | M4 (CPU) |
| `tests/test_data_gen.py`, `tests/test_ffr.py` | M4 (CPU); whole suite 37 tests pass |

Command used (M4, 2.4 min):

```bash
PYTHONPATH=src python3 -m flowsplat.data_gen --imagecas_dir ~/Desktop/GT_Courses/3rd_Semster/CS6999/ImageCAS --out data/cases_v1 --n_geom 40 --per_geom 5 --seed 0
```

Outputs: `data/cases_v1/` (gitignored: `geom/`, `cases/` with one `.npz`
per case, `qc/` with one figure per geometry). Copied to the repo:
`results/step1.1_summary.json`, `results/step1.1_cases.csv`,
`results/step1.1_qc_sheet.png` (all 40 LAD choices).

## What was done

1. **Reused extraction.** The owner's earlier code (repo
   cs6999-coronary-centerline, commit `f17a5f0`) copied verbatim into
   `src/flowsplat/centerline.py`: skeleton + radius in mm + DFS order
   (`centerline.py`), LCA/RCA split by the NIfTI L-R axis and isotropic
   nearest-neighbour resampling at the finest spacing (`build_dataset_v3.py`).
2. **LAD choice (OURS).** ImageCAS has one binary label for all coronaries
   (Sec 3, Fig. 1 caption). Ostium = most rightward LCA endpoint (the left
   main leaves the aorta toward the patient's left). LAD = most anterior
   leaf among leaves at least 0.7 x the longest geodesic from the ostium.
3. **Path smoothing (OURS).** Positions smoothed (1 mm) before arc length is
   measured, radius median + Gaussian (2 mm), uniform 0.5 mm sampling.
4. **QC gate (OURS).** Length 60 to 250 mm; radius 0.3 to 3 mm; median
   radius of the first 10 mm >= 0.8 x the 95th percentile (root really at
   the ostium); distal end >= 10 mm anterior of the ostium (not the LCx);
   visual exclusion list.
5. **Stenosis.** Diameter reduction U(30, 70) % (plan); cosine profile,
   length U(5, 15) mm, centre in the proximal 60 % (OURS).
6. **Flow.** Rest velocity U(10, 20) cm/s in the normal segment at the
   lesion; hyperemic = rest x CFR, CFR U(1.5, 4.0) (PUNCH, Materials and
   Methods), truncated so V_hyp <= 0.40 m/s (plan).
7. **FFR.** Young-Tsai at hyperemic flow, P_a 90 mmHg (`ffr.py`). Tests
   reproduce the plan's worked example (422 + 1305 Pa, 12.96 mmHg, FFR
   0.856; 0.798 and 0.888 for the two error cases). Flow redrawn when
   FFR < 0.3 (OURS).
8. **Timing check** (plan, Dataset), below.

Per case saved: `s`, `xyz_mm`, `r_ref_mm`, `r_mm` (with stenosis),
`tau_rest_s`, and all parameters (stenosis, Q_rest, Q_hyp, CFR, dP terms,
FFR, FFR at rest, number of flow draws). Seeds: `default_rng([0, case, k])`.

## Numbers (ImageCAS run, Oct 3)

Geometries: 40 LADs from 55 label files tried (of 95 on the M4); 15
rejected:

| Reason | Files |
|---|---|
| LCA/RCA merged in one component (reused split rule) | 117, 579, 648 |
| Radius above 3 mm | 113, 575, 642 |
| Length above 250 mm | 111 |
| Proximal radius too small (root not at the ostium) | 134, 574, 576, 577, 578, 597, 607 |
| Visual QC | 637 |

| Quantity | Range (mean) |
|---|---|
| LAD length (ostium to distal end) | 80.5 to 165.0 mm (130.9) |
| Reference radius at the lesion | 1.24 to 2.55 mm (1.70) |
| Stenosis degree / length | 30.2 to 69.8 % (49.9) / 5.0 to 15.0 mm (10.2) |
| **V_rest** | **10.1 to 19.9 cm/s (15.2)**, check passed |
| CFR / V_hyp | 1.51 to 3.75 (2.10) / 16.7 to 40.0 cm/s (31.1) |
| Q_rest / Q_hyp | 0.57 to 3.40 mL/s (1.40) / 1.19 to 6.94 mL/s (2.87) |
| dP at hyperemia | 0.7 to 63.0 mmHg (17.4) |
| **FFR** | **0.30 to 0.99 (0.81)**; 35.5 % below 0.80 |
| FFR at rest flow | 0.70 to 1.00 (0.93) |
| Flow redrawn (FFR floor) | 6 % of cases |

### Timing check (rest flow, 198 deg over 4.4 s)

| Views | View interval | View intervals per transit (min / median / max) | Transit < 1 view |
|---|---|---|---|
| 30 | 0.147 s | 1.9 / 5.4 / 13.7 | 0 % |
| 60 | 0.073 s | 3.8 / 10.7 / 27.4 | 0 % |
| 133 | 0.033 s | 8.5 / 23.8 / 60.8 | 0 % |

Transit time of the full LAD at rest: 0.28 to 2.01 s (median 0.79); all
within the 4.4 s sweep. The plan's worry (a vessel crossed within one view
interval) does not occur at rest; the fastest case still spans about 2 view
intervals at 30 views. Hyperemic flow (about 2x faster) would halve these
numbers; Step 1.2 renders at rest flow as the plan says.

## Bugs found and fixed during the step

| Problem | Fix |
|---|---|
| Arc length about 8 % too long (voxel staircase measured before smoothing) | Smooth first, then measure (unit test) |
| Throat radius taken as the vessel minimum (a tapering distal end can be narrower) | Throat = r_ref(s0) x (1 - degree), exact |
| Young-Tsai at fixed hyperemic flow gave dP > P_a (FFR < 0) for severe lesions | Flow redrawn until FFR >= 0.3 |
| Widest-endpoint root (reused convention) picked distal stumps in 5 of the first 20 cases (QC sheet) | Root = most rightward endpoint; proximal-radius and anterior checks added to the QC gate |

## DoD

| DoD part | Met |
|---|---|
| About 200 cases with geometry, Q, FFR saved | **Yes**: 200 cases (40 LADs x 5) |
| Ranges checked (rest 10 to 20 cm/s) | **Yes**: V_rest 10.1 to 19.9 cm/s (`check_rest_10_20_cms: true`) |
| Timing check run before the full set (plan bullet) | **Yes** (table above) |

**DoD met: yes. Fallback used: no** (ImageCAS extraction worked; the
procedural-vessel fallback was not needed).

## Open, for the owner's review

- Design choices marked OURS: LAD rule and QC thresholds, lesion length
  (5 to 15 mm) and position (proximal 60 %), FFR floor 0.3, cosine shape.
- Visual LAD check was done by me on the QC sheet; 637 excluded. A second
  look at `results/step1.1_qc_sheet.png` is welcome.
- ImageCAS labels give a distal LAD radius of about 1.2 to 1.4 mm (the
  thin distal end is not in the label); the lesion reference radius is 1.24
  to 2.55 mm.
- Case split (train / val / test, by patient) is a Step 1.2 item.
