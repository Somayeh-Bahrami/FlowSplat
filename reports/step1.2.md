# Step 1.2: Bolus simulation and TIGRE rendering

Dates: Oct 8 to Oct 11, 2026 (started Oct 3). Status: **in progress**: bolus
simulation and splits done on the M4; rendering on molab pending.

Plan DoD: "All cases rendered; 5 cases spot-checked visually (bolus visibly
advancing along the vessel); fixed split lists saved."

## Where things run

| Piece | Runs on |
|---|---|
| `src/flowsplat/bolus_sim.py` (bolus, splits, data pack) | M4 (CPU) |
| `src/flowsplat/render_tigre.py` (`lumen_voxels`, geometry, subsets) | M4 tests; molab |
| `src/flowsplat/render_tigre.py:render_case`, `notebooks/step1_render.py` | molab GPU |
| `tests/test_bolus_sim.py`, `tests/test_render_tigre.py` | M4 (CPU); whole suite 49 tests pass |

## What was done

1. **Bolus model.** PUNCH Eq. 2 (area term + dispersion) in conservative
   form, d(Ac)/dt + d(Qc)/ds = d/ds(A D dc/ds), at the rest flow of each
   case. Richer than the plug flow FlowSplat fits (plan: avoid an inverse
   crime): dispersion, area-dependent speed, noisy inlet.
2. **Numerics (OURS).** Finite volume, 0.1 mm cells, MUSCL with the superbee
   limiter, SSP-RK2, CFL 0.4. Tests: arrival time through a 60 % stenosis
   equals the plug-flow tau = int A/Q ds within 10 ms; mass conserved within
   2 %; front width follows 2 D t within 20 %; numerical front width 0.29 mm
   after 50 mm (van Leer at the same grid: 1.17 mm; physical width at the
   lowest D: 1.2 mm).
3. **Inlet and dispersion.** Gamma-variate inlet (plan), t0 U(0.2, 1.0) s,
   alpha U(2.5, 4.5), beta U(0.25, 0.5) s, with 5 % smooth multiplicative
   noise (OURS). D log-uniform 10^-6.5 to 10^-5.5 m^2/s, one decade around
   PUNCH's 1e-6.
4. **Acquisition.** 133 frames evenly over 198 deg (4DRGS Sec 4.1) and
   4.4 s (plan; duration UNVERIFIED); each frame has its own time.
5. **Splits.** Patient-level 70/15/15, seed 0: 28 / 6 / 6 patients =
   140 / 30 / 30 cases (`results/splits_v1.json`).
6. **Renderer (molab).** Lumen voxels from centerline + stenosed radius
   (0.25 mm, linear partial volume; tested: tube volume and stenosis
   cross-section within 3 %). Volume per frame = mu x fraction x c(s, t_j),
   mu = 0.1 /mm (OURS). TIGRE cone beam, DSO 750 / DSD 1200 mm (4DRGS),
   detector pixel 0.64 mm (OURS, 0.4 mm at the isocentre), rotation about
   the S-I axis, R2-Gaussian noise (Poisson 1e5, Gaussian std 10).
   View subsets 30 / 60 / 133, evenly spaced; the rest are test views.

## Numbers (bolus, M4)

Command (M4, 12.7 min for 200 cases):

```bash
PYTHONPATH=src python3 -m flowsplat.bolus_sim --cases data/cases_v1 --out data/bolus_v1 --seed 0
```

| Quantity | min / median / max |
|---|---|
| D (m^2/s) | 3.2e-7 / 8.8e-7 / 3.1e-6 |
| Peclet number u L / D | 3,155 / 23,488 / 153,152 (PUNCH: about 4000 with u = 0.04 m/s; ours uses rest velocities of 0.10 to 0.20 m/s) |
| Peak concentration (inlet noise included) | 0.88 / 1.02 / 1.17 |
| Share of frames with contrast (c > 0.05 anywhere) | 58 % / 78 % / 91 % |
| t0 / alpha / beta | 0.21 to 1.00 s / 2.50 to 4.50 / 0.25 to 0.50 s |

Checked on all 200 cases: the bolus front reaches the distal end at half peak or more in every case, and at the last frame the inlet concentration has fallen below half its peak in 199 of 200 cases (median 5 % of peak; worst case 53 %), so wash-in and wash-out are both inside the 4.4 s sweep. Kymographs
of the first 5 cases (`data/bolus_v1/kymo/`) show the bolus advancing with a
slope set by Q and a visible kink at the stenosis.

Splits (`results/splits_v1.json`, patient level, seed 0):

| Split | Patients | Cases |
|---|---|---|
| train | 28 | 140 |
| val | 121, 122, 130, 131, 135, 667 | 30 |
| test | 116, 132, 133, 612, 639, 641 | 30 |

Data pack for molab: `data/step1_pack.zip` (cases, geometries, bolus,
splits).

## Rendering (molab)

TODO.

## DoD

| DoD part | Met |
|---|---|
| All cases rendered | TODO (molab) |
| 5 cases spot-checked visually | TODO (molab figures; kymographs on the M4 already show the bolus advancing) |
| Fixed split lists saved | **Yes** (`results/splits_v1.json`) |

**DoD met: no (in progress). Fallback used: no.**
