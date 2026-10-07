"""Step 1.2 (part 2): rotational DSA of each case with TIGRE.

Runs on: molab GPU for `render_case` (needs TIGRE + CUDA). `lumen_voxels`,
`tigre_geometry_dict` and `view_subsets` are numpy only and also run on the
M4 (tests).

Each of the 133 frames is projected at its own gantry angle AND its own time
(4DRGS acquisition; plan, Dataset): the volume of frame j is the lumen
filled with the bolus concentration c(s, t_j) from bolus_sim.py.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

# 4DRGS, Sec 4.1: "source-to-object and source-to-detector distances are
# 750mm and 1200mm"; detector pixel about 0.32 mm (Table 1).
DSO_MM = 750.0
DSD_MM = 1200.0
# OURS: detector pixel 0.64 mm (2 x 2 binning of the 4DRGS pixel), i.e.
# 0.4 mm at the isocentre (4DRGS reference volumes: 0.38 to 0.49 mm voxels,
# Table 1). Keeps 200 cases x 133 frames at about 5 GB in float16.
DET_PX_MM = 0.64
# OURS: voxel size of the rendered lumen volume (resolves a 30 to 70 %
# stenosis of a 2.5 to 5 mm vessel with partial volume).
VOX_MM = 0.25
PAD_MM = 3.0
# OURS (value UNVERIFIED): attenuation of fully opacified blood, 1/mm at
# c = 1. A 3 mm lumen at peak contrast then absorbs about 26 %.
MU_I_PER_MM = 0.1
# R2-Gaussian scanner/cone_beam.yml noise (follows TIGRE demo d03):
# Poisson lambda 1e5, Gaussian mean 0, std 10.
NOISE_POISSON = 1e5
NOISE_GAUSS = (0.0, 10.0)
# Plan: subsets of 30 / 60 / 133 views.
VIEW_SETS = (30, 60, 133)


def lumen_voxels(s_mm, xyz_mm, r_mm, vox_mm=VOX_MM, pad_mm=PAD_MM, dense_mm=0.1):
    """Voxels of the tube {x : |x - axis(s*)| < r(s*)} with linear partial volume.

    s* is the nearest point of the (densified) centerline. Fraction per voxel:
    clip((r - d) / vox + 0.5, 0, 1). Returns a dict with integer voxel indices
    (N, 3) in the grid of shape `shape` starting at `origin_mm`, `frac` and
    the arc length `s_vox` of the nearest centerline point.

    Ref: OURS (tube from centerline + radius; the lumen model A = pi r^2 of
    the plan). The fallback of the plan (concentration on the centerline,
    splatted at render time) is the same idea.
    """
    s_mm, xyz_mm, r_mm = (np.asarray(a, dtype=float) for a in (s_mm, xyz_mm, r_mm))
    sd = np.arange(0.0, s_mm[-1] + 1e-9, dense_mm)
    xd = np.stack([np.interp(sd, s_mm, xyz_mm[:, k]) for k in range(3)], axis=1)
    rd = np.interp(sd, s_mm, r_mm)
    origin = xd.min(0) - rd.max() - pad_mm
    shape = np.ceil((xd.max(0) + rd.max() + pad_mm - origin) / vox_mm).astype(int) + 1
    # candidate voxels: local boxes around every 4th dense sample
    cand = []
    for x, r in zip(xd[::4], rd[::4]):
        lo = np.floor((x - r - 2 * vox_mm - origin) / vox_mm).astype(int)
        hi = np.ceil((x + r + 2 * vox_mm - origin) / vox_mm).astype(int) + 1
        g = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
        cand.append(np.ravel_multi_index(g.T, shape))
    lin = np.unique(np.concatenate(cand))
    idx = np.stack(np.unravel_index(lin, shape), axis=1)
    centers = origin + idx * vox_mm
    d, j = cKDTree(xd).query(centers)
    frac = np.clip((rd[j] - d) / vox_mm + 0.5, 0.0, 1.0)
    keep = frac > 0
    return {
        "idx": idx[keep].astype(np.int32),
        "frac": frac[keep].astype(np.float32),
        "s_vox": sd[j[keep]].astype(np.float32),
        "shape": shape.tolist(),
        "origin_mm": origin.tolist(),
        "vox_mm": vox_mm,
    }


def axis_permutation(axcodes):
    """Array-axis order for TIGRE ([z, y, x], rotation about z): put the
    superior-inferior axis first so the gantry rotates about the patient's
    long axis like a C-arm; the other two follow in reverse order (as
    R2-Gaussian's transpose (2, 1, 0) for an x, y, z volume). Ref: TIGRE
    volume order as used in R2-Gaussian generate_data.py; C-arm axis OURS."""
    s_ax = next(i for i, c in enumerate(axcodes) if c in ("S", "I"))
    rest = [i for i in (2, 1, 0) if i != s_ax]
    return [s_ax] + rest


def tigre_geometry_dict(shape_zyx, vox_mm=VOX_MM, det_px_mm=DET_PX_MM, dso=DSO_MM, dsd=DSD_MM):
    """Cone-beam geometry for a volume of shape [nz, ny, nx], centred at the
    isocentre. Detector sized to cover the volume at every gantry angle
    (in-plane diagonal and height at the near-side magnification).

    Ref: 4DRGS Sec 4.1 (DSO, DSD); TIGRE geometry fields as in R2-Gaussian
    ct_utils.get_geometry_tigre / cone_beam.yml. Detector size OURS.
    """
    nz, ny, nx = shape_zyx
    sz, sy, sx = nz * vox_mm, ny * vox_mm, nx * vox_mm
    r_xy = 0.5 * np.hypot(sx, sy)
    mag = dsd / (dso - r_xy)
    nu = int(np.ceil(2 * r_xy * mag / det_px_mm / 2) * 2) + 8
    nv = int(np.ceil(sz * mag / det_px_mm / 2) * 2) + 8
    return {
        "mode": "cone",
        "DSD": dsd,
        "DSO": dso,
        "nDetector": [nv, nu],
        "dDetector": [det_px_mm, det_px_mm],
        "sDetector": [nv * det_px_mm, nu * det_px_mm],
        "nVoxel": [int(nz), int(ny), int(nx)],
        "dVoxel": [vox_mm] * 3,
        "sVoxel": [sz, sy, sx],
        "offOrigin": [0.0, 0.0, 0.0],
        "offDetector": [0.0, 0.0],
        "accuracy": 0.5,
    }


def view_subsets(n_frames=133, sets=VIEW_SETS):
    """Uniform frame subset per view set; the rest are test views.

    4DRGS Sec 2 ("Sparse-View DSA Reconstruction"): j_k = floor((k - 1) T / N)
    + 1, k = 1..N (1-based), i.e. j = floor(k T / N), k = 0..N-1 here.
    Ref: 4DRGS Sec 2; view counts 30 / 60 / 133 from the plan. (Until Oct 4
    this used rounded even spacing including the last frame; changed to the
    4DRGS rule before Step 2.1 run 3.)
    """
    out = {}
    for n in sets:
        tr = np.unique((np.arange(n) * n_frames) // n).astype(int)
        out[str(n)] = {"train": tr.tolist(), "test": sorted(set(range(n_frames)) - set(tr.tolist()))}
    return out


def tigre_geo(gd):
    """TIGRE geometry object from a `tigre_geometry_dict` (molab). Ref: TIGRE
    geometry fields, as in R2-Gaussian ct_utils.get_geometry_tigre."""
    import tigre

    geo = tigre.geometry(mode="cone", nVoxel=np.array(gd["nVoxel"]), default=True)
    for k in ("DSD", "DSO", "nDetector", "dDetector", "sDetector", "nVoxel", "dVoxel", "sVoxel", "offOrigin", "offDetector"):
        setattr(geo, k, np.array(gd[k], dtype=float) if isinstance(gd[k], list) else float(gd[k]))
    geo.nDetector = geo.nDetector.astype(int)
    geo.nVoxel = geo.nVoxel.astype(int)
    geo.accuracy = gd["accuracy"]
    return geo


def render_case(case_npz, bolus_npz, axcodes, seed=0, mu=MU_I_PER_MM, noise=True,
                noise_poisson=NOISE_POISSON, noise_gauss=NOISE_GAUSS):
    """Project all frames of one case (Step 1.1 case file with the stenosed
    radius r_mm, Step 1.2 bolus file). Returns (proj float16 [T, nv, nu], meta).

    Runs on: molab GPU. Ref: TIGRE tigre.Ax and CTnoise.add, used as in
    R2-Gaussian data_generator/synthetic_dataset/generate_data.py.
    """
    import tigre
    from tigre.utilities import CTnoise

    g = np.load(case_npz)
    b = np.load(bolus_npz)
    case_s, c = b["s"], b["c"].astype(np.float32)
    lum = lumen_voxels(g["s"], g["xyz_mm"], g["r_mm"])
    perm = axis_permutation(axcodes)
    idx = lum["idx"][:, perm]
    shape = [lum["shape"][p] for p in perm]
    gd = tigre_geometry_dict(shape)
    geo = tigre_geo(gd)
    angles = np.deg2rad(b["angles_deg"]).astype(np.float32)
    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, dtype=np.float32)
    projs = np.zeros((len(angles), gd["nDetector"][0], gd["nDetector"][1]), dtype=np.float32)
    for j in range(len(angles)):
        cj = np.interp(lum["s_vox"], case_s, c[:, j]).astype(np.float32)
        vol[idx[:, 0], idx[:, 1], idx[:, 2]] = mu * lum["frac"] * cj
        projs[j] = tigre.Ax(vol, geo, angles[j : j + 1])[0]
    if noise:
        np.random.seed(int(rng.integers(2**31)))  # CTnoise uses numpy's global RNG
        projs = CTnoise.add(projs, Poisson=noise_poisson, Gaussian=np.array(noise_gauss))
        projs[projs < 0] = 0.0
    meta = {"geometry": gd, "axis_perm": perm, "axcodes": axcodes, "mu_per_mm": mu,
            "lumen": {k: lum[k] for k in ("shape", "origin_mm", "vox_mm")}, "n_lumen_voxels": int(len(idx)),
            "noise": {"poisson": noise_poisson, "gaussian": list(noise_gauss)} if noise else None,
            "view_subsets": view_subsets(len(angles))}
    return projs.astype(np.float16), meta


def meta_json(meta):
    return json.dumps(meta)
