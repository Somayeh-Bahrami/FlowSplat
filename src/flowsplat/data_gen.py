"""Step 1.1: LAD geometry, synthetic stenoses, flow and FFR ground truth.

Runs on: M4 (CPU only).

Pipeline per ImageCAS label:
  1. left coronary tree (LCA) via the reused split_components (centerline.py)
  2. isotropic resample, skeleton + radius in mm (reused extract_centerline)
  3. LAD = root-to-leaf skeleton path picked by `lad_path_indices` (OURS)
  4. resample the path at DS_MM, smooth position and radius (OURS)
Per case (several per geometry):
  5. one cosine stenosis, 30 to 70 % diameter reduction (plan)
  6. rest velocity 10 to 20 cm/s, hyperemic = rest x CFR (PUNCH)
  7. ground-truth FFR with Young-Tsai at hyperemic flow (ffr.py)
Also: the plan's timing check (transit time vs. view interval).

Usage:
  python -m flowsplat.data_gen --imagecas_dir <dir with *.label.nii.gz> \
      --out data/cases_v1 --n_geom 40 --per_geom 5 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from flowsplat import centerline as cl
from flowsplat import ffr

# ---- Ranges and settings, each with its source ----
# Plan, Step 1.1: "Insert synthetic stenoses (30 to 70% diameter)".
STENOSIS_DIAM = (0.30, 0.70)
# OURS: lesion length range. The plan's worked example uses L = 5 mm; 15 mm
# keeps lesions focal. Young-Tsai's own model lengths: UNVERIFIED.
STENOSIS_LEN_MM = (5.0, 15.0)
# PUNCH, Materials and Methods (synthetic data): "Mean resting velocities
# were sampled between 10 and 20 cm/s for typical cases".
V_REST_MS = (0.10, 0.20)
# PUNCH, same paragraph: hyperemia = rest x CFR, CFR from 1.5 to 4.0 for
# typical cases. Upper end truncated so that V_hyp <= 0.40 m/s (plan,
# Dataset: "hyperemia up to about 0.4 m/s").
CFR = (1.5, 4.0)
V_HYP_MAX_MS = 0.40
# OURS: lowest accepted ground-truth FFR. Young-Tsai at a FIXED hyperemic
# flow can exceed P_a for severe lesions (e.g. 70 % at 0.4 m/s: dP about
# 98 mmHg > 90 mmHg), which is unphysical because a real lesion reduces the
# flow. Such flow draws are redrawn. 0.3 as a plausible lower end of
# measured FFR: UNVERIFIED.
FFR_MIN = 0.3
MAX_FLOW_DRAWS = 200
# OURS: path sampling and smoothing.
DS_MM = 0.5
SMOOTH_POS_MM = 1.0
SMOOTH_RAD_MM = 2.0
# OURS: QC gate for an LAD geometry.
LAD_LEN_MM = (60.0, 250.0)
LAD_RADIUS_MM = (0.3, 3.0)
PROX_MM, PROX_FRAC = 10.0, 0.8
MIN_ANTERIOR_MM = 10.0
# OURS: LAD candidates are leaves at least this fraction of the longest
# root-to-leaf geodesic.
LAD_MIN_LEN_FRAC = 0.7
# Plan, Dataset, timing check: 198 deg over about 4.4 s (4DRGS protocol, 133
# frames); view subsets 30 / 60 / 133.
SWEEP_S = 4.4
VIEW_SETS = (30, 60, 133)


# =============================== geometry ===============================
def _axis_score_fn(axcodes, toward):
    """Function coords -> larger is further toward `toward` ('A', 'R', ...),
    from NIfTI axis codes. Ref: OURS."""
    pair = {"A": "P", "P": "A", "R": "L", "L": "R", "S": "I", "I": "S"}[toward]
    ax = next((i for i, c in enumerate(axcodes) if c in (toward, pair)), None)
    if ax is None:
        return None
    sign = 1.0 if axcodes[ax] == toward else -1.0
    return lambda c: sign * np.asarray(c, dtype=float)[..., ax]


def lad_path_indices(coords, radii, iso_mm, axcodes, min_len_frac=LAD_MIN_LEN_FRAC):
    """Pick the LAD as one root-to-leaf path of the LCA skeleton.

    Root (ostium): the endpoint furthest toward the patient's RIGHT. The left
    main leaves the aorta toward the patient's left, so the ostium is the LCA
    tip nearest the aorta. (The widest-endpoint convention of the reused
    _traversal_order picked a distal stump in 5 of the first 20 ImageCAS
    cases on the QC sheet, Oct 3, so it is not used here.) Leaves: endpoints
    whose geodesic distance from the root is at
    least `min_len_frac` of the longest. LAD: the candidate whose leaf is the
    most anterior (the LAD runs in the anterior interventricular groove,
    the LCx posteriorly in the AV groove).

    Ref: OURS. ImageCAS does not label LAD / LCx separately (Zeng et al.
    2023, Sec 3, Fig. 1 caption: "subclasses of coronary arteries are not
    further individually labeled"). Anatomy of the LAD and LCx course:
    general anatomy, no paper reference (UNVERIFIED). Checked visually with
    the QC figures.

    Returns (path indices root -> leaf, info dict).
    """
    adj = cl._build_adjacency(coords)
    rows, cols, w = [], [], []
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            rows.append(i)
            cols.append(j)
            w.append(float(np.linalg.norm(coords[i] - coords[j])) * iso_mm)
    n = len(coords)
    graph = csr_matrix((w, (rows, cols)), shape=(n, n))
    degree = np.array([len(a) for a in adj])
    endpoints = np.where(degree <= 1)[0]
    if len(endpoints) < 2:
        raise ValueError("skeleton has fewer than 2 endpoints")
    right = _axis_score_fn(axcodes, "R")
    if right is None:
        raise ValueError(f"no L/R axis in NIfTI axis codes {axcodes}")
    root = int(endpoints[np.argmax(right(coords[endpoints]))])
    dist, pred = dijkstra(graph, indices=root, return_predecessors=True)
    leaves = [int(e) for e in endpoints if e != root and np.isfinite(dist[e])]
    dmax = max(dist[e] for e in leaves)
    cands = [e for e in leaves if dist[e] >= min_len_frac * dmax]
    score = _axis_score_fn(axcodes, "A")
    if score is None:
        raise ValueError(f"no A/P axis in NIfTI axis codes {axcodes}")
    leaf = max(cands, key=lambda e: score(coords[e]))
    path = [leaf]
    while path[-1] != root:
        p = int(pred[path[-1]])
        if p < 0:
            raise ValueError("leaf not connected to root")
        path.append(p)
    path = np.array(path[::-1])
    info = {
        "root_radius_mm": float(radii[root]),
        "n_endpoints": int(len(endpoints)),
        "n_candidates": int(len(cands)),
        "longest_geodesic_mm": float(dmax),
        "lad_geodesic_mm": float(dist[leaf]),
        "lad_is_longest": bool(dist[leaf] >= dmax - 1e-6),
    }
    return path, info


def resample_and_smooth(points_mm, radii_mm, ds=DS_MM, sigma_pos_mm=SMOOTH_POS_MM, sigma_rad_mm=SMOOTH_RAD_MM):
    """Smooth a skeleton path, then resample it at uniform arc length.

    Skeleton points sit on voxel centres: the staircase makes the raw
    polyline up to about 8 % longer than the vessel (measured on a straight
    oblique synthetic tube), so positions are smoothed BEFORE arc length is
    measured. EDT radii jump by voxel steps: median filter (5 samples)
    against single-voxel spikes, then Gaussian.

    Ref: OURS (standard 1D Gaussian smoothing and polyline resampling).
    Returns (s, xyz, r) with s in mm from the root.
    """
    pts = np.asarray(points_mm, dtype=float)
    step = float(np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    pts = gaussian_filter1d(pts, sigma_pos_mm / step, axis=0, mode="nearest")
    rad = gaussian_filter1d(median_filter(np.asarray(radii_mm, dtype=float), size=5, mode="nearest"),
                            sigma_rad_mm / step, mode="nearest")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_raw = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, s_raw[-1] + 1e-9, ds)
    xyz = np.stack([np.interp(s, s_raw, pts[:, k]) for k in range(3)], axis=1)
    r = np.interp(s, s_raw, rad)
    return s, xyz, r


def extract_lad(label, affine, spacing, iso_mm=None, margin_mm=2.0):
    """ImageCAS label -> LAD (s, xyz_mm, r_mm), info, skeleton_mm; or raise ValueError.

    xyz is in mm in the cropped isotropic array frame (array axes, not world
    axes); only lengths and radii are used downstream.

    Ref: LCA split and isotropic resampling reused from build_dataset_v3.py
    (split_components, to_isotropic; iso = finest spacing, as in its
    resolve_iso 'auto'); skeleton + radius reused from centerline.py;
    LAD choice and smoothing OURS.
    """
    import nibabel as nib

    spacing = np.asarray(spacing, dtype=float)
    comps, why = cl.split_components(np.asarray(label) > 0.5, affine, spacing)
    if why:
        raise ValueError(f"LCA split: {why}")
    lca = next((c for c in comps if c["name"] == "LCA"), None)
    if lca is None:
        raise ValueError("no LCA component")
    m = lca["mask"]
    idx = np.argwhere(m)
    pad = np.ceil(margin_mm / spacing).astype(int)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, m.shape)
    crop = m[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    iso = float(spacing.min()) if iso_mm is None else float(iso_mm)
    iso_mask, _ = cl.to_isotropic(crop, spacing, iso)
    sk = cl.extract_centerline(iso_mask, spacing=(iso, iso, iso))
    coords = sk[:, :3].astype(np.int64)
    radii = sk[:, 3]
    axcodes = nib.aff2axcodes(affine)
    path, info = lad_path_indices(coords, radii, iso, axcodes)
    s, xyz, r = resample_and_smooth(coords[path] * iso, radii[path])
    info.update({"iso_mm": iso, "lca_voxels": int(lca["n_vox"]), "axcodes": "".join(axcodes)})
    return s, xyz, r, info, coords * iso  # whole LCA skeleton, for the QC figure


def qc_lad(s, r, xyz=None, axcodes=None):
    """QC gate for one LAD. Returns None if ok, else the reason.

    Besides length and radius ranges: (a) the first PROX_MM must be wide
    (median >= PROX_FRAC x 95th percentile of r), else the root is a distal
    tip, not the ostium; (b) the distal end must lie at least
    MIN_ANTERIOR_MM anterior of the ostium, else the LCx was picked. Both
    added after the QC-sheet review of Oct 3 (cases 574, 577, 578 failed by
    eye).

    Ref: OURS.
    """
    length = float(s[-1])
    if not (LAD_LEN_MM[0] <= length <= LAD_LEN_MM[1]):
        return f"length {length:.1f} mm outside {LAD_LEN_MM}"
    if not np.all(np.isfinite(r)):
        return "non-finite radius"
    if r.min() < LAD_RADIUS_MM[0] or r.max() > LAD_RADIUS_MM[1]:
        return f"radius {r.min():.2f} to {r.max():.2f} mm outside {LAD_RADIUS_MM}"
    prox = float(np.median(r[s <= PROX_MM]))
    if prox < PROX_FRAC * float(np.percentile(r, 95)):
        return f"proximal radius {prox:.2f} mm < {PROX_FRAC} x p95 ({np.percentile(r, 95):.2f}): root not at ostium?"
    if xyz is not None and axcodes is not None:
        ant = _axis_score_fn(axcodes, "A")
        if ant is not None:
            gain = float(ant(xyz[-1]) - ant(xyz[0]))
            if gain < MIN_ANTERIOR_MM:
                return f"distal end only {gain:.1f} mm anterior of the ostium: LCx picked?"
    return None


# =============================== stenosis ===============================
def insert_stenosis(s, r_ref, degree, length_mm, center_mm):
    """Cosine-shaped stenosis on top of the reference radius.

    r(s) = r_ref(s) * (1 - degree * (1 + cos(2 pi (s - s0) / L)) / 2) for
    |s - s0| <= L / 2, else r_ref(s). Minimum radius r_ref(s0) * (1 - degree),
    i.e. `degree` is the diameter reduction.

    Ref: OURS. The cosine (smooth, axisymmetric) profile is a common
    idealised stenosis shape; whether Young-Tsai used it: UNVERIFIED.
    """
    x = (s - center_mm) / length_mm
    bump = np.where(np.abs(x) <= 0.5, 0.5 * (1.0 + np.cos(2.0 * np.pi * x)), 0.0)
    return r_ref * (1.0 - degree * bump)


def sample_case(s, r_ref, rng):
    """Draw one stenosis and one flow for an LAD geometry.

    Lesion centre: uniform in the proximal 60 % of the vessel, at least
    L/2 + 5 mm from both ends and 10 mm from the ostium (OURS: proximal and
    mid LAD lesions; keeps an upstream reference segment).
    """
    length = float(s[-1])
    deg = rng.uniform(*STENOSIS_DIAM)
    L = rng.uniform(*STENOSIS_LEN_MM)
    lo = max(10.0, L / 2 + 5.0)
    hi = min(0.6 * length, length - L / 2 - 5.0)
    if hi <= lo:
        raise ValueError("vessel too short for a stenosis")
    s0 = rng.uniform(lo, hi)
    r = insert_stenosis(s, r_ref, deg, L, s0)

    r_n = float(np.interp(s0, s, r_ref))  # reference (normal) radius at the lesion
    r_s = r_n * (1.0 - deg)  # exact throat radius of the inserted profile (not grid-sampled)
    a_n, a_s = np.pi * (r_n * 1e-3) ** 2, np.pi * (r_s * 1e-3) ** 2
    for n_draws in range(1, MAX_FLOW_DRAWS + 1):
        v_rest = rng.uniform(*V_REST_MS)
        cfr = rng.uniform(CFR[0], min(CFR[1], V_HYP_MAX_MS / v_rest))
        q_rest = v_rest * a_n
        q_hyp = cfr * q_rest
        dp_hyp, dp_visc, dp_exp = ffr.young_tsai_dp(q_hyp, a_n, a_s, L * 1e-3)
        if ffr.ffr_from_dp(dp_hyp) >= FFR_MIN:
            break
    else:
        raise ValueError(f"no flow with FFR >= {FFR_MIN} in {MAX_FLOW_DRAWS} draws")
    dp_rest, _, _ = ffr.young_tsai_dp(q_rest, a_n, a_s, L * 1e-3)
    # Transit time at rest, tau(s) = int_0^s A(s') / Q ds' (plan, Approach).
    area = np.pi * (r * 1e-3) ** 2
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (area[1:] + area[:-1]) * np.diff(s) * 1e-3)]) / q_rest
    return {
        "r": r,
        "tau_rest_s": tau,
        "stenosis_degree": deg,
        "stenosis_len_mm": L,
        "stenosis_center_mm": s0,
        "r_ref_at_lesion_mm": r_n,
        "r_min_mm": r_s,
        "v_rest_ms": v_rest,
        "v_in_rest_ms": float(q_rest / (np.pi * (r[0] * 1e-3) ** 2)),
        "cfr": cfr,
        "v_hyp_ms": cfr * v_rest,
        "q_rest_mls": q_rest * 1e6,
        "q_hyp_mls": q_hyp * 1e6,
        "dp_hyp_mmhg": float(dp_hyp / ffr.MMHG_PA),
        "dp_hyp_viscous_mmhg": float(dp_visc / ffr.MMHG_PA),
        "dp_hyp_expansion_mmhg": float(dp_exp / ffr.MMHG_PA),
        "ffr": float(ffr.ffr_from_dp(dp_hyp)),
        "ffr_rest": float(ffr.ffr_from_dp(dp_rest)),
        "transit_rest_s": float(tau[-1]),
        "n_flow_draws": n_draws,
    }


# =============================== timing check ===============================
def timing_check(transit_s, length_mm, view_sets=VIEW_SETS, sweep_s=SWEEP_S):
    """Plan, Dataset, timing check: frames the bolus needs to cross the vessel.

    View interval dt = sweep_s / n_views. Reports transit time stats and the
    number of view intervals per transit for each view set.

    Ref: plan, Dataset ("198 deg over about 4.4 s with 30 views means one
    view every 0.147 s ..."). Sweep duration as stated there (4DRGS
    acquisition timing: UNVERIFIED).
    """
    t = np.asarray(transit_s, dtype=float)
    out = {
        "sweep_s": sweep_s,
        "transit_s": {"min": float(t.min()), "median": float(np.median(t)), "max": float(t.max())},
        "length_mm": {"min": float(np.min(length_mm)), "median": float(np.median(length_mm)), "max": float(np.max(length_mm))},
        "per_view_set": {},
    }
    for n in view_sets:
        dt = sweep_s / n
        k = t / dt
        out["per_view_set"][str(n)] = {
            "view_interval_s": round(dt, 4),
            "views_per_transit": {"min": float(k.min()), "median": float(np.median(k)), "max": float(k.max())},
            "frac_cases_transit_below_1_view": float((k < 1).mean()),
            "frac_cases_transit_within_sweep": float((t <= sweep_s).mean()),
        }
    return out


# =============================== driver ===============================
def _qc_figure(path_png, s, xyz, r, case_id, skel_mm=None, axcodes="???"):
    """Whole LCA skeleton (gray) with the chosen LAD (coloured by radius), in
    three projections with anatomical axis labels; radius profile. Used for
    the visual spot check of the LAD choice. Ref: OURS."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
    for k, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
        if skel_mm is not None:
            ax[k].scatter(skel_mm[:, a], skel_mm[:, b], c="0.8", s=1)
        sc = ax[k].scatter(xyz[:, a], xyz[:, b], c=r, s=4, cmap="viridis")
        ax[k].plot(xyz[0, a], xyz[0, b], "r*", ms=12)
        ax[k].set_aspect("equal")
        ax[k].set_xlabel(f"axis {a}: + = {axcodes[a]}")
        ax[k].set_ylabel(f"axis {b}: + = {axcodes[b]}")
        ax[k].set_title(f"case {case_id} (star = ostium)")
    fig.colorbar(sc, ax=ax[2], label="r (mm)")
    ax[3].plot(s, r)
    ax[3].set_xlabel("s (mm)")
    ax[3].set_ylabel("r (mm)")
    fig.tight_layout()
    fig.savefig(path_png, dpi=90)
    plt.close(fig)


# Visual QC exclusions (QC sheet review, Oct 3, 2026). Ref: OURS.
VISUAL_EXCLUDE = {
    "637": "path dips posterior, then ends far left; shorter branches reach further anterior (LAD unclear)",
}


def generate(imagecas_dir, out_dir, n_geom=40, per_geom=5, seed=0, max_try=None, exclude=None):
    """Extract up to n_geom LADs and draw per_geom cases each. Writes:
    out_dir/geom/<id>.npz, out_dir/cases/<id>_<k>.npz, out_dir/qc/<id>.png,
    out_dir/cases.csv, out_dir/summary.json. Returns the summary dict.
    """
    import nibabel as nib

    imagecas_dir, out_dir = Path(imagecas_dir), Path(out_dir)
    for sub in ("geom", "cases", "qc"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    files = sorted(imagecas_dir.glob("*.label.nii.gz"), key=lambda p: int(p.name.split(".")[0]))
    if max_try:
        files = files[:max_try]
    rejects, rows, geoms = [], [], []
    t0 = time.time()
    for f in files:
        if len(geoms) >= n_geom:
            break
        cid = f.name.split(".")[0]
        exclude = VISUAL_EXCLUDE if exclude is None else exclude
        if cid in exclude:
            rejects.append({"case": cid, "reason": f"visual QC: {exclude[cid]}"})
            print(f"[reject] {cid}: visual QC", flush=True)
            continue
        try:
            nii = nib.load(str(f))
            s, xyz, r, info, skel = extract_lad(np.asanyarray(nii.dataobj), nii.affine, nii.header.get_zooms()[:3])
            why = qc_lad(s, r, xyz, info["axcodes"])
            if why:
                raise ValueError(why)
        except Exception as e:  # recorded, the next file is tried
            rejects.append({"case": cid, "reason": f"{type(e).__name__}: {e}"})
            print(f"[reject] {cid}: {e}", flush=True)
            continue
        np.savez(out_dir / "geom" / f"{cid}.npz", s=s, xyz_mm=xyz, r_ref_mm=r, info=json.dumps(info))
        _qc_figure(out_dir / "qc" / f"{cid}.png", s, xyz, r, cid, skel, info["axcodes"])
        geoms.append({"case": cid, "length_mm": float(s[-1]), **info})
        for k in range(per_geom):
            rng = np.random.default_rng([seed, int(cid), k])
            for _ in range(20):  # a new stenosis if no flow gives FFR >= FFR_MIN
                try:
                    c = sample_case(s, r, rng)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError(f"{cid}_{k}: no valid stenosis/flow draw")
            name = f"{cid}_{k}"
            np.savez(
                out_dir / "cases" / f"{name}.npz",
                s=s, xyz_mm=xyz, r_ref_mm=r, r_mm=c["r"], tau_rest_s=c["tau_rest_s"],
                params=json.dumps({k2: v for k2, v in c.items() if k2 not in ("r", "tau_rest_s")}),
            )
            rows.append({"case": name, "geom": cid, "length_mm": round(float(s[-1]), 2),
                         **{k2: round(v, 5) if isinstance(v, float) else v for k2, v in c.items() if k2 not in ("r", "tau_rest_s")}})
        print(f"[ok] {cid}: LAD {s[-1]:.1f} mm, r {r.min():.2f} to {r.max():.2f} mm", flush=True)

    if rows:
        with open(out_dir / "cases.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    summary = summarize(rows, geoms, rejects, seed, time.time() - t0)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def summarize(rows, geoms, rejects, seed, seconds):
    """Range checks for the DoD and the timing check. Ref: plan, Step 1.1 DoD."""
    def rng_of(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())} if len(v) else None

    v_rest = np.array([r["v_rest_ms"] for r in rows], dtype=float)
    ffr_v = np.array([r["ffr"] for r in rows], dtype=float)
    return {
        "seed": seed,
        "n_cases": len(rows),
        "n_geometries": len(geoms),
        "n_rejected_files": len(rejects),
        "rejects": rejects,
        "seconds": round(seconds, 1),
        "ranges": {k: rng_of(k) for k in ("v_rest_ms", "v_hyp_ms", "cfr", "q_rest_mls", "q_hyp_mls",
                                         "stenosis_degree", "stenosis_len_mm", "r_ref_at_lesion_mm",
                                         "dp_hyp_mmhg", "ffr", "ffr_rest", "length_mm")} if rows else {},
        "check_rest_10_20_cms": bool(len(v_rest) and v_rest.min() >= V_REST_MS[0] and v_rest.max() <= V_REST_MS[1]),
        "frac_ffr_below_0p80": float((ffr_v < 0.80).mean()) if len(ffr_v) else None,
        "frac_cases_flow_redrawn": float(np.mean([r["n_flow_draws"] > 1 for r in rows])) if rows else None,
        "timing_check": timing_check([r["transit_rest_s"] for r in rows], [r["length_mm"] for r in rows]) if rows else None,
        "geometries": geoms,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--imagecas_dir", required=True)
    ap.add_argument("--out", default="data/cases_v1")
    ap.add_argument("--n_geom", type=int, default=40)
    ap.add_argument("--per_geom", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_try", type=int, default=None, help="only look at the first N label files")
    a = ap.parse_args(argv)
    s = generate(a.imagecas_dir, a.out, a.n_geom, a.per_geom, a.seed, a.max_try)
    print(json.dumps({k: v for k, v in s.items() if k not in ("geometries", "rejects")}, indent=2))


if __name__ == "__main__":
    main()
