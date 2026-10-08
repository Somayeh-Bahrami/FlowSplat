"""Step 5.1: Y-bifurcation (LAD + diagonal) with continuity and Murray prior.

Runs on: M4 (data generation, bolus, tree extraction, tests) and molab GPU
(rendering, 4DRGS, fitting; same code).

Plan, Step 5.1: "Y-bifurcation cases from ImageCAS (LAD + diagonal).
Per-branch Q with hard constraint Q_p = Q_1 + Q_2 and soft Murray prior.
Ablation: with vs without continuity / Murray." Plan, Dataset:
"Y-bifurcations (LAD + diagonal) from ImageCAS with a Murray-consistent
flow split." DoD: "Flow split error below 10% on bifurcation test cases;
ablation table."

Branch order everywhere: 0 = parent (ostium -> junction), 1 = LAD after the
junction, 2 = diagonal; daughters start at the junction (s = 0).
Flow split error (user decision, Oct 8): f_k = Q_k / Q_parent for the two
daughters; error = mean_k |f_k,est - f_k,gt| / f_k,gt.
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from flowsplat import bolus_sim as bs
from flowsplat import centerline as cl
from flowsplat import data_gen as dg
from flowsplat import ffr

# OURS: branch selection on the LCA skeleton (lengths in mm, radii from the EDT).
LM_BRANCH_MIN_MM = 40.0  # a side tree this long near the ostium is the LCx (or ramus)
LM_SEARCH_MM = 40.0  # the LM bifurcation lies within this distance of the ostium
DIAG_MIN_LEN_MM = 30.0  # diagonal: at least this long
DIAG_MIN_R_MM = 0.6  # diagonal: entry radius at least this
DIAG_MIN_GAP_MM = 5.0  # diagonal: at least this far beyond the LM bifurcation
DIAG_MAX_FRAC = 0.7  # diagonal: attaches within the proximal 70 % of the LAD
MIN_DAUGHTER_MM = 25.0  # each daughter at least this long after smoothing
MURRAY_WINDOW_MM = (2.0, 8.0)  # daughter radius for the split: median over this window
MURRAY_EXP = 3.0  # Murray's law r^3 (plan; Murray 1926, UNVERIFIED)
DIAG_MAX_REL_R = 1.1  # OURS: a diagonal is not thicker than the LAD after it (Oct 8, case 592)
TOUCH_SPAN_MM = 2.0  # contacts of one side tree with the path span at most this
JUNCTION_CLEAR_MM = 5.0  # stenosis at least this far (plus L/2) from the junction


# ============================ skeleton graph ============================
def _graph(coords, iso_mm):
    adj = cl._build_adjacency(coords)
    rows, cols, w = [], [], []
    for i, nb in enumerate(adj):
        for j in nb:
            rows.append(i)
            cols.append(j)
            w.append(float(np.linalg.norm(coords[i] - coords[j])) * iso_mm)
    n = len(coords)
    return csr_matrix((w, (rows, cols)), shape=(n, n)), adj


def side_branches(coords, radii, iso_mm, path):
    """Side trees hanging off a path of a skeleton graph.

    The path nodes are removed; every remaining connected component whose
    contacts with the path lie within TOUCH_SPAN_MM of path arc length is a
    side tree (in a 26-connected skeleton a branch usually touches 2 or 3
    consecutive path voxels; requiring exactly one contact dropped most
    diagonals, Oct 8); attach = the middle contact. For each: the attach node
    (index into `path`), the arc length of the attach node along the path,
    the longest geodesic path inside the side tree starting next to the
    attach node, its length and its entry radius. Ref: OURS.
    """
    g, adj = _graph(coords, iso_mm)
    on_path = np.zeros(len(coords), bool)
    on_path[path] = True
    keep = np.where(~on_path)[0]
    sub = g[keep][:, keep]
    n_comp, lab = connected_components(sub, directed=False)
    seg = np.linalg.norm(np.diff(coords[path], axis=0), axis=1) * iso_mm
    s_path = np.concatenate([[0.0], np.cumsum(seg)])
    pos_in_path = {int(p): k for k, p in enumerate(path)}
    out = []
    for c in range(n_comp):
        members = keep[lab == c]
        touch = sorted({pos_in_path[j] for m in members for j in adj[m] if on_path[j]})
        if not touch or s_path[touch[-1]] - s_path[touch[0]] > TOUCH_SPAN_MM:
            continue  # loops back to the path: not a side tree
        k_att = touch[len(touch) // 2]
        entry = [m for m in members if any(on_path[j] for j in adj[m])]
        loc = {int(m): i for i, m in enumerate(members)}
        gm = g[members][:, members]
        d, pred = dijkstra(gm, indices=loc[int(entry[0])], return_predecessors=True)
        far = int(np.argmax(np.where(np.isfinite(d), d, -1)))
        p = [far]
        while p[-1] != loc[int(entry[0])]:
            p.append(int(pred[p[-1]]))
        branch = members[np.array(p[::-1])]
        out.append({"k_attach": k_att, "s_attach_mm": float(s_path[k_att]), "nodes": branch,
                    "length_mm": float(d[far] + np.linalg.norm(coords[branch[0]] - coords[path[k_att]]) * iso_mm),
                    "entry_r_mm": float(np.median(radii[branch[: max(3, len(branch) // 10)]]))})
    return sorted(out, key=lambda b: b["s_attach_mm"]), s_path


def pick_diagonal(branches, lad_len_mm):
    """LM bifurcation = first side tree within LM_SEARCH_MM of the ostium
    that is at least LM_BRANCH_MIN_MM long (LCx / ramus; 0 if none). Diagonal
    = first later side tree with length, entry radius and position rules
    (constants above). Returns (diagonal dict or None, s_lm). Ref: OURS
    (coronary anatomy: diagonals leave the LAD after the left main
    bifurcation; UNVERIFIED general anatomy, checked on the QC figures)."""
    s_lm = 0.0
    for b in branches:
        if b["s_attach_mm"] <= LM_SEARCH_MM and b["length_mm"] >= LM_BRANCH_MIN_MM:
            s_lm = b["s_attach_mm"]
            break
    for b in branches:
        if (b["s_attach_mm"] >= s_lm + DIAG_MIN_GAP_MM and b["s_attach_mm"] <= DIAG_MAX_FRAC * lad_len_mm
                and b["length_mm"] >= DIAG_MIN_LEN_MM and b["entry_r_mm"] >= DIAG_MIN_R_MM):
            return b, s_lm
    return None, s_lm


def extract_bifurcation(label, affine, spacing, iso_mm=None, margin_mm=2.0):
    """ImageCAS label -> three smoothed branches (parent, LAD daughter,
    diagonal) as lists of (s, xyz_mm, r_mm), info, skeleton; or ValueError.
    Front end identical to data_gen.extract_lad (LCA split, isotropic
    resampling, owner's skeleton, LAD path). Ref: data_gen.extract_lad; OURS."""
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
    iso = float(spacing.min()) if iso_mm is None else float(iso_mm)
    iso_mask, _ = cl.to_isotropic(m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], spacing, iso)
    sk = cl.extract_centerline(iso_mask, spacing=(iso, iso, iso))
    coords = sk[:, :3].astype(np.int64)
    radii = sk[:, 3]
    axcodes = nib.aff2axcodes(affine)
    path, info = dg.lad_path_indices(coords, radii, iso, axcodes)
    branches, s_path = side_branches(coords, radii, iso, path)
    diag, s_lm = pick_diagonal(branches, float(s_path[-1]))
    if diag is None:
        raise ValueError(f"no diagonal ({len(branches)} side trees, LM bifurcation at {s_lm:.1f} mm)")
    k = diag["k_attach"]
    pts = coords * iso
    parts = [
        (pts[path[: k + 1]], radii[path[: k + 1]]),
        (pts[path[k:]], radii[path[k:]]),
        (np.concatenate([pts[path[k : k + 1]], pts[diag["nodes"]]]), np.concatenate([radii[path[k : k + 1]], radii[diag["nodes"]]])),
    ]
    out = []
    for p, r in parts:
        if len(p) < 6:
            raise ValueError("branch too short to smooth")
        out.append(dg.resample_and_smooth(p, r))
    # daughters start exactly at the parent's end point (junction)
    junction = out[0][1][-1]
    for b in (1, 2):
        s, xyz, r = out[b]
        xyz = xyz + (junction - xyz[0]) * np.exp(-s / 2.0)[:, None]  # blend the first ~5 mm onto the junction
        out[b] = (s, xyz, r)
    info.update({"iso_mm": iso, "axcodes": "".join(axcodes), "s_lm_mm": s_lm, "diag_attach_mm": diag["s_attach_mm"],
                 "diag_len_mm": diag["length_mm"], "diag_entry_r_mm": diag["entry_r_mm"], "n_side_trees": len(branches)})
    return out, info, pts


def qc_bifurcation(branches):
    """None if usable, else the reason. Ref: OURS (same ranges as data_gen)."""
    (s0, _, r0), (s1, _, r1), (s2, _, r2) = branches
    lad_len = s0[-1] + s1[-1]
    if not dg.LAD_LEN_MM[0] <= lad_len <= dg.LAD_LEN_MM[1]:
        return f"LAD length {lad_len:.1f} mm out of range"
    if s0[-1] < 15.0:
        return f"parent only {s0[-1]:.1f} mm"
    if min(s1[-1], s2[-1]) < MIN_DAUGHTER_MM:
        return f"daughter shorter than {MIN_DAUGHTER_MM} mm"
    for r in (r0, r1, r2):
        if r.min() < dg.LAD_RADIUS_MM[0] or r.max() > dg.LAD_RADIUS_MM[1]:
            return "radius out of range"
    if daughter_radius(s2, r2) > DIAG_MAX_REL_R * daughter_radius(s1, r1):
        return "diagonal thicker than the LAD daughter (likely not a diagonal)"
    return None


# ============================ cases ============================
def murray_split(r1_mm, r2_mm, exp=MURRAY_EXP):
    """Daughter flow fractions from Murray's law Q ~ r^3. Ref: Murray 1926 (UNVERIFIED); plan."""
    a, b = r1_mm**exp, r2_mm**exp
    return a / (a + b), b / (a + b)


def daughter_radius(s, r, window=MURRAY_WINDOW_MM):
    """Median radius of a daughter just after the junction (OURS window)."""
    m = (s >= window[0]) & (s <= window[1])
    return float(np.median(r[m])) if m.any() else float(np.median(r[: max(1, len(r) // 5)]))


def tau_branches(s_list, r_list, q_list):
    """Plug-flow transit times per branch (s), daughters continue from the
    parent's end. Ref: plan, Approach (tau = int A / Q ds)."""
    out = []
    for b, (s, r) in enumerate(zip(s_list, r_list)):
        area = np.pi * (np.asarray(r) * 1e-3) ** 2
        t = np.concatenate([[0.0], np.cumsum(0.5 * (area[1:] + area[:-1]) * np.diff(s) * 1e-3)]) / q_list[b]
        out.append(t if b == 0 else t + out[0][-1])
    return out


def sample_tree_case(branches, rng):
    """One stenosis on the LAD (parent or LAD daughter), rest velocity, Murray
    split, ground-truth FFR (Young-Tsai at the flow of the stenosed branch x
    CFR, redrawn below data_gen.FFR_MIN). Rest velocity is defined in the
    parent: Q_p = v x mean parent reference area (OURS; data_gen defines it
    at the lesion for a single vessel). Ref: data_gen.sample_case; OURS."""
    (s0, x0, r0), (s1, x1, r1), (s2, x2, r2) = branches
    sj = float(s0[-1])
    s_lad = np.concatenate([s0, sj + s1[1:]])
    r_lad = np.concatenate([r0, r1[1:]])
    deg = rng.uniform(*dg.STENOSIS_DIAM)
    L = rng.uniform(*dg.STENOSIS_LEN_MM)
    lo = max(10.0, L / 2 + 5.0)
    hi = min(0.6 * s_lad[-1], s_lad[-1] - L / 2 - 5.0)
    allowed = [x for x in np.arange(lo, hi, 0.5) if abs(x - sj) >= L / 2 + JUNCTION_CLEAR_MM]
    if not allowed:
        raise ValueError("no stenosis position away from the junction")
    c0 = float(rng.choice(allowed))
    r_lad_st = dg.insert_stenosis(s_lad, r_lad, deg, L, c0)
    r0s, r1s = r_lad_st[: len(s0)], np.concatenate([r_lad_st[len(s0) - 1 : len(s0)], r_lad_st[len(s0):]])
    branch = 0 if c0 < sj else 1
    r_n = float(np.interp(c0, s_lad, r_lad))
    r_s = r_n * (1 - deg)
    a_n, a_s = np.pi * (r_n * 1e-3) ** 2, np.pi * (r_s * 1e-3) ** 2
    f1, f2 = murray_split(daughter_radius(s1, r1), daughter_radius(s2, r2))
    area_p = float(np.mean(np.pi * (r0 * 1e-3) ** 2))
    for n_draws in range(1, dg.MAX_FLOW_DRAWS + 1):
        v = rng.uniform(*dg.V_REST_MS)
        q_p = v * area_p
        q_b = q_p if branch == 0 else f1 * q_p
        cfr = rng.uniform(dg.CFR[0], min(dg.CFR[1], dg.V_HYP_MAX_MS / (q_b / a_n)))
        dp, _, _ = ffr.young_tsai_dp(cfr * q_b, a_n, a_s, L * 1e-3)
        if ffr.ffr_from_dp(dp) >= dg.FFR_MIN:
            break
    else:
        raise ValueError("no flow with FFR >= FFR_MIN")
    q = [q_p, f1 * q_p, f2 * q_p]
    taus = tau_branches([s0, s1, s2], [r0s, r1s, r2], q)
    params = {
        "q_mls": [x * 1e6 for x in q], "frac_gt": [f1, f2], "v_rest_ms": v, "cfr": cfr,
        "stenosis_branch": branch, "stenosis_center_lad_mm": c0, "stenosis_degree": deg, "stenosis_len_mm": L,
        "r_ref_at_lesion_mm": r_n, "r_min_mm": r_s, "ffr": float(ffr.ffr_from_dp(dp)),
        "junction_s_mm": sj, "daughter_r_mm": [daughter_radius(s1, r1), daughter_radius(s2, r2)],
        "transit_s": [float(t[-1]) for t in taus], "n_flow_draws": n_draws,
    }
    return [(s0, x0, r0s), (s1, x1, r1s), (s2, x2, r2)], params


def save_tree_case(path, branches, params, r_ref=None):
    """npz with b<k>_s, b<k>_xyz_mm, b<k>_r_mm (k = 0, 1, 2) and params."""
    d = {"params": json.dumps(params)}
    for k, (s, xyz, r) in enumerate(branches):
        d[f"b{k}_s"], d[f"b{k}_xyz_mm"], d[f"b{k}_r_mm"] = s, xyz, r
        if r_ref is not None:
            d[f"b{k}_r_ref_mm"] = r_ref[k]
    np.savez(path, **d)


def load_tree_case(path):
    z = np.load(path)
    br = [(z[f"b{k}_s"], z[f"b{k}_xyz_mm"], z[f"b{k}_r_mm"]) for k in range(3)]
    return br, json.loads(str(z["params"]))


def simulate_tree_bolus(branches, q_m3s, rng, t_frames=None):
    """Bolus in all three branches: PUNCH Eq 2 in the parent with a
    gamma-variate inlet (bolus_sim.sample_bolus), then in each daughter with
    the parent's outlet curve as inlet (concentration continuous at the
    junction, complete mixing). Returns (list of c (N_b, T), bolus params).
    Ref: bolus_sim.simulate (PUNCH Eq 2); junction coupling OURS."""
    t_frames = bs.frame_times() if t_frames is None else t_frames
    b = bs.sample_bolus(rng)
    t_fine = np.linspace(0.0, t_frames[-1], 4401)
    clean = bs.gamma_variate(t_fine, b["t0_s"], b["alpha"], b["beta_s"])
    noisy = clean * bs.smooth_noise(t_fine, bs.INLET_NOISE, bs.INLET_NOISE_TAU_S, rng)
    s0, _, r0 = branches[0]
    c0_fine = bs.simulate(s0, r0, q_m3s[0], b["d_m2s"], lambda t: np.interp(t, t_fine, noisy), t_fine)
    outlet = c0_fine[-1]
    cs = [np.stack([np.interp(t_frames, t_fine, row) for row in c0_fine])]
    for k in (1, 2):
        s, _, r = branches[k]
        cs.append(bs.simulate(s, r, q_m3s[k], b["d_m2s"], lambda t, o=outlet: np.interp(t, t_fine, o), t_frames))
    b["inlet_noisy"] = np.interp(t_frames, t_fine, noisy).tolist()
    return [c.astype(np.float32) for c in cs], b


# ============================ rendering (molab) ============================
def tree_lumen_voxels(branches, vox_mm=None, pad_mm=None):
    """Voxels of the union of the three tubes on one grid. Where tubes
    overlap (junction) the voxel keeps the branch with the larger partial
    volume. Returns dict like render_tigre.lumen_voxels plus `branch`.
    Ref: render_tigre.lumen_voxels; union rule OURS."""
    from flowsplat import render_tigre as rt

    vox_mm = rt.VOX_MM if vox_mm is None else vox_mm
    pad_mm = rt.PAD_MM if pad_mm is None else pad_mm
    allx = np.concatenate([b[1] for b in branches])
    rmax = max(float(np.max(b[2])) for b in branches)
    origin = allx.min(0) - rmax - pad_mm
    shape = np.ceil((allx.max(0) + rmax + pad_mm - origin) / vox_mm).astype(int) + 1
    best = {}
    for k, (s, xyz, r) in enumerate(branches):
        lv = rt.lumen_voxels(s, xyz, r, vox_mm=vox_mm, origin_mm=origin, shape=shape)
        lin = np.ravel_multi_index(lv["idx"].T, shape)
        for li, f, sv in zip(lin, lv["frac"], lv["s_vox"]):
            if li not in best or f > best[li][0]:
                best[li] = (f, k, sv)
    lin = np.array(sorted(best))
    vals = np.array([best[li] for li in lin])
    return {"idx": np.stack(np.unravel_index(lin, shape), 1).astype(np.int32), "frac": vals[:, 0].astype(np.float32),
            "branch": vals[:, 1].astype(np.int32), "s_vox": vals[:, 2].astype(np.float32),
            "shape": shape.tolist(), "origin_mm": origin.tolist(), "vox_mm": vox_mm}


def render_tree_case(case_npz, bolus_npz, axcodes, seed=0, mu=None):
    """Rotational DSA of a bifurcation case (same protocol, noise and seeding
    as render_tigre.render_case). bolus_npz holds c0, c1, c2 (stations x
    frames), s0..s2, angles_deg, t_s. Runs on: molab (TIGRE).
    Ref: render_tigre.render_case."""
    import tigre
    from tigre.utilities import CTnoise

    from flowsplat import render_tigre as rt

    mu = rt.MU_I_PER_MM if mu is None else mu
    br, _ = load_tree_case(case_npz)
    b = np.load(bolus_npz)
    lum = tree_lumen_voxels(br)
    perm = rt.axis_permutation(axcodes)
    idx = lum["idx"][:, perm]
    shape = [lum["shape"][p] for p in perm]
    gd = rt.tigre_geometry_dict(shape)
    geo = rt.tigre_geo(gd)
    angles = np.deg2rad(b["angles_deg"]).astype(np.float32)
    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, np.float32)
    projs = np.zeros((len(angles), gd["nDetector"][0], gd["nDetector"][1]), np.float32)
    for j in range(len(angles)):
        cj = np.zeros(len(idx), np.float32)
        for k in range(3):
            m = lum["branch"] == k
            cj[m] = np.interp(lum["s_vox"][m], b[f"s{k}"], b[f"c{k}"][:, j])
        vol[idx[:, 0], idx[:, 1], idx[:, 2]] = mu * lum["frac"] * cj
        projs[j] = tigre.Ax(vol, geo, angles[j : j + 1])[0]
    np.random.seed(int(rng.integers(2**31)))
    projs = CTnoise.add(projs, Poisson=rt.NOISE_POISSON, Gaussian=np.array(rt.NOISE_GAUSS))
    projs[projs < 0] = 0.0
    meta = {"geometry": gd, "axis_perm": perm, "axcodes": axcodes, "mu_per_mm": mu,
            "lumen": {k: lum[k] for k in ("shape", "origin_mm", "vox_mm")}, "n_lumen_voxels": int(len(idx)),
            "noise": {"poisson": rt.NOISE_POISSON, "gaussian": list(rt.NOISE_GAUSS)},
            "view_subsets": rt.view_subsets(len(angles)), "tree": True}
    return projs.astype(np.float16), meta


# ============================ tree from a volume (no ground truth) ============================
SIDE_MIN_MM = 20.0  # OURS: shortest skeleton side branch taken as the second daughter


def tree_from_volume(vol, vox_mm, origin_mm):
    """Static volume -> main path split at the attach point of the longest
    side branch: (seg_a, seg_b, side) as (s, xyz) each, all three starting at
    the junction except seg_a, which ends there. Which end of the main path
    is the inlet is not known (fit_tree tests both). Ref:
    model.centerline_from_volume (threshold, skeleton, longest path); side
    branch rule OURS."""
    from scipy.ndimage import label as cc_label

    from flowsplat import model as md

    v = np.asarray(vol, float)
    ref = np.percentile(v[v > 0], 99.9)
    lab, n = cc_label(v > md.THRESH_FRAC * ref, structure=np.ones((3, 3, 3)))
    if n == 0:
        raise ValueError("no foreground")
    mask = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    sk = cl.extract_centerline(mask, spacing=(vox_mm,) * 3)
    coords = sk[:, :3].astype(np.int64)
    path = md.longest_path(coords, vox_mm)
    sides, s_path = side_branches(coords, np.ones(len(coords)), vox_mm, path)
    sides = [b for b in sides if b["length_mm"] >= SIDE_MIN_MM]
    if not sides:
        raise ValueError("no side branch in the volume")
    side = max(sides, key=lambda b: b["length_mm"])
    k = side["k_attach"]
    pts = np.asarray(origin_mm) + coords * vox_mm
    segs = [pts[path[: k + 1]], pts[path[k:]], np.concatenate([pts[path[k : k + 1]], pts[side["nodes"]]])]
    out = []
    for p in segs:
        s, xyz, _ = dg.resample_and_smooth(p, np.ones(len(p)))
        out.append((s, xyz))
    return out, {"side_len_mm": side["length_mm"], "n_side": len(sides), "attach_mm": float(s_path[k])}


def tree_radii(vol, vox_mm, origin_mm, segs, median_n=5):
    """Half-maximum radius per branch (model.half_max_radius), gaps
    interpolated, median filtered, as model.build_from_volume. Ref: OURS."""
    from scipy.ndimage import median_filter

    from flowsplat import model as md

    out = []
    for s, xyz in segs:
        r = md.half_max_radius(vol, vox_mm, origin_mm, xyz)
        good = np.isfinite(r)
        r = np.interp(s, s[good], r[good]) if good.any() else np.full_like(s, 1.0)
        out.append(median_filter(r, size=median_n, mode="nearest"))
    return out


def orientations(segs, radii):
    """The two inlet hypotheses as (parent, daughter_lad, daughter_side), each
    (s, xyz, r) with the parent ordered inlet -> junction and daughters from
    the junction. Ref: OURS."""
    (sa, xa), (sb, xb), (sc, xc) = segs
    ra, rb, rc = radii
    h1 = [(sa, xa, ra), (sb, xb, rb), (sc, xc, rc)]
    h2 = [(sb, xb[::-1].copy(), rb[::-1].copy()), (sa, xa[::-1].copy(), ra[::-1].copy()), (sc, xc, rc)]
    return [h1, h2]


# ============================ transport on a tree ============================
TREE_VARIANTS = ("indep", "cont", "murray", "cont_murray")  # plan ablation: continuity / Murray on / off
LAMBDA_MURRAY = 0.1  # OURS: a 10 % split deviation (log ratio 0.15) costs about 0.002, like the image loss


class TreeTransport:
    """Plug flow on a Y: one inlet curve, flows Q_p, Q_1, Q_2.

    tau_i = int_0^{s_i} A / Q_p ds on the parent; on daughter k,
    tau_i = tau_J + int_0^{s_i} A / Q_k ds (tau_J = parent transit).
    continuity=True: Q_p = Q_1 + Q_2 (hard, plan contribution 3); else Q_p is
    a free parameter. Murray prior (soft): LAMBDA_MURRAY (log(Q_1 / Q_2) - 3
    log(R_1 / R_2))^2 with the daughter radii of the frozen geometry; with
    the geometry fixed (Gate 1 method) the plan's radius prior r_p^3 = r_1^3
    + r_2^3 can only act through the flows (Q ~ r^3, Murray 1926,
    UNVERIFIED). Ref: plan, Approach and Step 5.1; transport.Transport; OURS.
    """

    def __init__(self, q_p_mm3s, f1=0.5, continuity=True, device="cpu"):
        import torch

        from flowsplat.transport import Transport

        self.continuity = continuity
        self.inlet = Transport(q_p_mm3s).to(device)  # t0, alpha, beta, k (its log_q is Q_p when free)
        self.log_q1 = torch.nn.Parameter(torch.tensor(math.log(f1 * q_p_mm3s), device=device))
        self.log_q2 = torch.nn.Parameter(torch.tensor(math.log((1 - f1) * q_p_mm3s), device=device))

    def parameters(self):
        return list(self.inlet.parameters()) + [self.log_q1, self.log_q2]

    def q(self):
        q1, q2 = self.log_q1.exp(), self.log_q2.exp()
        qp = q1 + q2 if self.continuity else self.inlet.q()
        return qp, q1, q2

    def tau(self, area, s, branch):
        import torch

        qp, q1, q2 = self.q()
        out = torch.zeros_like(area)
        tau_j = None
        for k, q in ((0, qp), (1, q1), (2, q2)):
            m = branch == k
            a, sk = area[m], s[m]
            ds = (sk[1:] - sk[:-1]).abs()
            t = torch.cat([a.new_zeros(1), torch.cumsum(0.5 * (a[1:] + a[:-1]) * ds, 0)]) / q
            if k == 0:
                tau_j = t[-1]
                out = out.masked_scatter(m, t)
            else:
                out = out.masked_scatter(m, t + tau_j)
        return out

    def density(self, area, sigma_a, s, branch, times):
        """k c_i(t) ds_i / (sqrt(2 pi) sigma_a,i), per branch (Transport.density)."""
        import torch

        tau = self.tau(area, s, branch)
        c = self.inlet.inlet(times[:, None] - tau[None, :])
        ds_node = torch.zeros_like(area)
        for k in range(3):
            m = branch == k
            sk = s[m]
            d = (sk[1:] - sk[:-1]).abs()
            ds_node = ds_node.masked_scatter(m, torch.cat([d[:1], 0.5 * (d[1:] + d[:-1]), d[-1:]]))
        return self.inlet.log_k.exp() * c * (ds_node / (math.sqrt(2 * math.pi) * sigma_a))[None, :]

    def murray_loss(self, r1, r2):
        _, q1, q2 = self.q()
        return LAMBDA_MURRAY * (q1.log() - q2.log() - MURRAY_EXP * math.log(r1 / r2)) ** 2


class TreeModel:
    """All tube Gaussians of the Y in one TubeGaussians (concatenated
    branches) + TreeTransport. Geometry is frozen (Gate 1 method)."""

    def __init__(self, hyp_world, continuity=True, f1=0.5, device="cpu", q_p=None):
        import torch

        from flowsplat import model as md
        from flowsplat import train as tr

        s = np.concatenate([b[0] for b in hyp_world])
        xyz = np.concatenate([b[1] for b in hyp_world])
        r = np.concatenate([b[2] for b in hyp_world])
        self.branch = torch.tensor(np.concatenate([np.full(len(b[0]), k) for k, b in enumerate(hyp_world)]), device=device)
        self.tube = md.TubeGaussians(s, xyz, r, device=device)
        self.r_daughter = [daughter_radius(hyp_world[k][0], hyp_world[k][2]) for k in (1, 2)]
        qp = tr.V0_MM_S * float(np.mean(np.pi * hyp_world[0][2] ** 2)) if q_p is None else q_p
        self.transport = TreeTransport(qp, f1=f1, continuity=continuity, device=device)

    def density(self, times):
        tg = self.tube
        return self.transport.density(tg.area().detach(), tg.sigma_a().detach(), tg.s, self.branch, times)


def fit_tree(proj, thetas, times, hyps_world, projector, variant="cont", iters_dir=150, iters=2100,
             device="cpu", log=print, direction=None):
    """Fit one bifurcation with frozen geometry (Gate 1 method: Stage A only,
    cached footprints, L1 + D-SSIM). The inlet hypothesis is chosen by the
    lower loss of a short fit with continuity (as train.fit_case), unless
    `direction` is given. Returns (model, info). Ref: train.fit_case; OURS."""
    import torch

    from flowsplat import train as tr

    proj = proj.to(device)
    times = torch.as_tensor(times, dtype=torch.float32, device=device)
    dr = float(proj.max())

    def run(m, cache, n_it, murray):
        params = m.transport.parameters()
        opt = torch.optim.Adam([{"params": params, "lr": tr.LR_TRANSPORT}])
        hist = []
        for it in range(n_it):
            opt.zero_grad()
            pred = torch.einsum("vn,vnp->vp", m.density(times), cache).reshape(proj.shape)
            loss, l1 = tr.image_loss(pred, proj, dr)
            if murray:
                loss = loss + m.transport.murray_loss(*m.r_daughter)
            loss.backward()
            opt.step()
            if it % 100 == 0 or it == n_it - 1:
                qp, q1, q2 = (float(x.detach()) for x in m.transport.q())
                hist.append({"it": it, "loss": float(loss.detach()), "qp": qp, "q1": q1, "q2": q2})
        return hist

    def init(h, continuity):
        m = TreeModel(h, continuity=continuity, device=device)
        tr._init_inlet(type("M", (), {"transport": m.transport.inlet})(), proj, times)  # same t0 rule as Step 2.2
        with torch.no_grad():
            cache = projector.footprints(m.tube.position, m.tube.covariance(), thetas)
            pred = torch.einsum("vn,vnp->vp", m.density(times), cache).reshape(proj.shape)
            ratio = float(proj.sum(dim=(1, 2)).max() / pred.sum(dim=(1, 2)).max().clamp_min(1e-12))
            m.transport.inlet.log_k.add_(math.log(max(ratio, 1e-6)))
        return m, cache

    dir_loss = []
    if direction is None:
        for d, h in enumerate(hyps_world):
            m, cache = init(h, True)
            hh = run(m, cache, iters_dir, False)
            dir_loss.append(hh[-1]["loss"])
            log(f"inlet hypothesis {d}: loss {hh[-1]['loss']:.5f}")
            del cache
        direction = int(np.argmin(dir_loss))
    murray = variant in ("murray", "cont_murray")
    continuity = variant in ("cont", "cont_murray")
    m, cache = init(hyps_world[direction], continuity)
    hist = run(m, cache, iters, murray)
    qp, q1, q2 = (float(x.detach()) for x in m.transport.q())
    log(f"{variant}: Qp {qp:.1f} Q1 {q1:.1f} Q2 {q2:.1f} mm3/s, loss {hist[-1]['loss']:.5f}")
    return m, {"direction": direction, "dir_loss": dir_loss, "variant": variant, "history": hist,
               "q_mm3s": [qp, q1, q2]}


# ============================ evaluation ============================
def match_daughters(est_daughters_xyz, gt_daughters_xyz):
    """Which estimated daughter is which ground-truth daughter: the
    assignment with the smaller mean nearest-point distance. Returns
    (perm, distances) with perm[k] = estimated index matched to GT daughter k."""
    from scipy.spatial import cKDTree

    def dist(a, b):
        return float(cKDTree(b).query(a)[0].mean())

    same = dist(est_daughters_xyz[0], gt_daughters_xyz[0]) + dist(est_daughters_xyz[1], gt_daughters_xyz[1])
    swap = dist(est_daughters_xyz[1], gt_daughters_xyz[0]) + dist(est_daughters_xyz[0], gt_daughters_xyz[1])
    return ([0, 1], same / 2) if same <= swap else ([1, 0], swap / 2)


def split_metrics(q_est, q_gt, perm):
    """DoD metric (user decision, Oct 8): f_k = Q_k / Q_p; error = mean over
    the two daughters of |f_k,est - f_k,gt| / f_k,gt. Also the relative error
    of each daughter flow and of the parent flow. q_*: (Q_p, Q_1, Q_2) with
    daughters of q_est in estimate order; perm maps GT daughter k to it."""
    qp_e = q_est[0]
    qd_e = [q_est[1 + perm[0]], q_est[1 + perm[1]]]
    f_e = [q / qp_e for q in qd_e]
    f_g = [q_gt[1] / q_gt[0], q_gt[2] / q_gt[0]]
    rel_f = [abs(a - b) / b for a, b in zip(f_e, f_g)]
    rel_q = [abs(a - b) / b for a, b in zip(qd_e, q_gt[1:])]
    return {"split_err": float(np.mean(rel_f)), "frac_err": rel_f, "frac_est": f_e, "frac_gt": f_g,
            "q_daughter_rel_err": rel_q, "q_daughter_rel_err_mean": float(np.mean(rel_q)),
            "q_parent_rel_err": abs(qp_e - q_gt[0]) / q_gt[0]}


# ============================ dataset (M4) ============================
def _qc_tree_png(path, branches, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for k, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        for kb, (s, xyz, r) in enumerate(branches):
            ax[k].plot(xyz[:, a], xyz[:, b], ("k", "b", "r")[kb], lw=1.5)
        ax[k].set_aspect("equal")
        ax[k].set_title(f"axes {a}-{b}")
    fig.suptitle(f"{title}: parent black, LAD daughter blue, diagonal red")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def generate_trees(imagecas_dir, out_dir, splits_json, per_geom=2, seed=0):
    """Bifurcation dataset. Every ImageCAS label (except data_gen
    VISUAL_EXCLUDE) is tried; qualifying geometries are assigned to
    `dev` if their patient is in the Step 1 train / val split, else to
    `test` (Step 1 test patients and patients never used before).
    Writes out_dir/{cases,bolus,qc}/, splits_tree.json, summary.json.
    Ref: data_gen.generate; OURS (split rule fixed Oct 8, before results)."""
    import nibabel as nib
    from pathlib import Path

    out = Path(out_dir)
    for sub in ("cases", "bolus", "qc"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    sp = json.loads(Path(splits_json).read_text())
    dev_pat = set(sp["train"]) | set(sp["val"])
    files = sorted(Path(imagecas_dir).glob("*.label.nii.gz"), key=lambda p: int(p.name.split(".")[0]))
    rejects, geoms, cases = {}, [], {"dev": [], "test": []}
    t_frames = bs.frame_times()
    for f in files:
        cid = f.name.split(".")[0]
        if cid in dg.VISUAL_EXCLUDE:
            rejects[cid] = "visual exclude (Step 1.1)"
            continue
        try:
            nii = nib.load(str(f))
            br, info, _ = extract_bifurcation(np.asanyarray(nii.dataobj), nii.affine, nii.header.get_zooms()[:3])
            why = qc_bifurcation(br)
            if why:
                raise ValueError(why)
        except Exception as e:  # recorded, next file
            rejects[cid] = f"{type(e).__name__}: {e}"
            continue
        split = "dev" if cid in dev_pat else "test"
        _qc_tree_png(out / "qc" / f"{cid}.png", br, f"{cid} ({split})")
        geoms.append({"geom": cid, "split": split, **info, "len_mm": [float(b[0][-1]) for b in br]})
        for k in range(per_geom):
            rng = np.random.default_rng([seed, int(cid), k, 5])
            for _ in range(20):
                try:
                    brs, prm = sample_tree_case(br, rng)
                    break
                except ValueError:
                    continue
            else:
                rejects[f"{cid}_{k}"] = "no valid stenosis / flow"
                continue
            name = f"{cid}_{k}"
            q = [x * 1e-6 for x in prm["q_mls"]]
            cs, bp = simulate_tree_bolus(brs, q, rng, t_frames)
            save_tree_case(out / "cases" / f"{name}.npz", brs, {**prm, "axcodes": info["axcodes"], "geom": cid},
                           r_ref=[b[2] for b in br])
            np.savez(out / "bolus" / f"{name}.npz", **{f"s{j}": brs[j][0] for j in range(3)},
                     **{f"c{j}": cs[j] for j in range(3)}, t_s=t_frames, angles_deg=bs.frame_angles_deg(),
                     params=json.dumps(bp))
            cases[split].append(name)
        print(f"[ok] {cid} ({split})", flush=True)
    splits = {"dev": cases["dev"], "test": cases["test"],
              "test_runs": [c for c in cases["test"] if c.endswith("_0")],
              "dev_runs": [c for c in cases["dev"] if c.endswith("_0")],
              "rule": "dev = Step 1 train/val patients; test = Step 1 test patients + ImageCAS patients never used; runs use case _0"}
    (out / "splits_tree.json").write_text(json.dumps(splits, indent=1))
    (out / "summary.json").write_text(json.dumps({"geoms": geoms, "rejects": rejects,
                                                  "n_dev": len(cases["dev"]), "n_test": len(cases["test"])}, indent=1))
    return splits


def make_tree_pack(out_dir, zip_path):
    """Zip cases, bolus and splits for molab (as bolus_sim.make_pack)."""
    import zipfile
    from pathlib import Path

    out = Path(out_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("cases", "bolus"):
            for f in sorted((out / sub).glob("*.npz")):
                z.write(f, f"{sub}/{f.name}")
        z.write(out / "splits_tree.json", "splits_tree.json")
    return zip_path


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Step 5.1 bifurcation dataset (M4)")
    ap.add_argument("--imagecas_dir", required=True)
    ap.add_argument("--out", default="data/trees_v1")
    ap.add_argument("--splits", default="results/splits_v1.json")
    ap.add_argument("--per_geom", type=int, default=2)
    ap.add_argument("--pack", default="data/step5_tree_pack.zip")
    a = ap.parse_args(argv)
    sp = generate_trees(a.imagecas_dir, a.out, a.splits, per_geom=a.per_geom)
    make_tree_pack(a.out, a.pack)
    print({k: (len(v) if isinstance(v, list) else v) for k, v in sp.items()})


if __name__ == "__main__":
    main()


def evaluate_tree(info_by_variant, est_lumen, case_npz):
    """Per-case metrics for all variants: daughter matching against the
    ground truth (geometry only), inlet correctness, split metrics.
    est_lumen: chosen hypothesis as [(s, xyz_lumen, r)] x 3. Ref: OURS."""
    from scipy.spatial import cKDTree

    br, p = load_tree_case(case_npz)
    perm, dist = match_daughters([est_lumen[1][1], est_lumen[2][1]], [br[1][1], br[2][1]])
    inlet_ok = bool(np.linalg.norm(est_lumen[0][1][0] - br[0][1][0]) < np.linalg.norm(est_lumen[0][1][-1] - br[0][1][0]))
    gt_all = np.concatenate([b[1] for b in br])
    cover = float((cKDTree(np.concatenate([b[1] for b in est_lumen])).query(gt_all)[0] < 2.0).mean())
    q_gt = [x * 1000.0 for x in p["q_mls"]]  # mm^3/s
    out = {"inlet_correct": inlet_ok, "daughter_perm": perm, "daughter_match_dist_mm": dist, "gt_coverage": cover,
           "frac_gt": p["frac_gt"], "q_gt_mm3s": q_gt}
    for v, info in info_by_variant.items():
        out[v] = split_metrics(info["q_mm3s"], q_gt, perm)
    return out


def summarize_tree(rows, variants=TREE_VARIANTS, cut=0.10):
    """Ablation table: per variant mean / median split error, share below
    10 %, daughter and parent flow errors. Ref: OURS."""
    tab = {}
    for v in variants:
        rr = [r[v] for r in rows if v in r]
        if not rr:
            continue
        se = np.array([x["split_err"] for x in rr])
        tab[v] = {"n": len(rr), "split_err_mean": float(se.mean()), "split_err_median": float(np.median(se)),
                  "frac_below_10pct": float((se < cut).mean()),
                  "q_daughter_rel_err_mean": float(np.mean([x["q_daughter_rel_err_mean"] for x in rr])),
                  "q_parent_rel_err_mean": float(np.mean([x["q_parent_rel_err"] for x in rr])),
                  "dod_met": bool(se.mean() < cut)}
    return tab
