"""Step 2.1: geometry initialisation and tube-aligned Gaussians.

Runs on: M4 (numpy / scipy / CPU torch, tests) and molab GPU (same code).

Input: the volume of a static R2-Gaussian reconstruction from the dynamic
views (plan: "Static R2-Gaussian on time-averaged views (as in 4DRGS
voxelization)"). A static model fitted to views taken at different times
converges to a time-averaged attenuation, so the brightness along the vessel
varies with how long each part was filled; the radius estimator below is
therefore scale-free (half maximum of the local cross-section).

Output: centerline (arc length s, position) and lumen radius R(s), and a
chain of tube-aligned Gaussians mapped to s with sigma = R / sqrt(2)
(plan: A = 2 pi sigma^2, R = sqrt(2) sigma).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.ndimage import map_coordinates
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from flowsplat import centerline as cl
from flowsplat import data_gen as dg

# OURS: foreground threshold for the skeleton mask, as a fraction of the
# 99.9th percentile of the positive voxels. Hysteresis (grow down to 0.1)
# was tried on Oct 4 (run 3) to recover the dim proximal segment of case
# 131_0; it let the centerline run into FDK streaks (radius MAE 0.05 -> 0.32
# mm on 121_0) and did not recover 131_0, so LOW_FRAC = THRESH_FRAC (single
# threshold, as in run 2). Only used to find the centerline.
THRESH_FRAC = 0.2
LOW_FRAC = 0.2
# OURS: rays per cross-section and maximum search radius for the half maximum.
N_RAYS = 16
R_MAX_MM = 4.0
DR_MM = 0.02


# ============================ centerline ============================
def longest_path(coords, iso_mm):
    """Longest geodesic path of a 26-connected skeleton (two Dijkstra passes).

    Ref: OURS (standard double-sweep for the diameter of a tree). The
    rendered lumen is one LAD without branches, so its skeleton is one path
    plus short spurs.
    """
    adj = cl._build_adjacency(coords)
    rows, cols, w = [], [], []
    for i, nb in enumerate(adj):
        for j in nb:
            rows.append(i)
            cols.append(j)
            w.append(float(np.linalg.norm(coords[i] - coords[j])) * iso_mm)
    g = csr_matrix((w, (rows, cols)), shape=(len(coords), len(coords)))
    d0 = dijkstra(g, indices=0)
    a = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    da, pred = dijkstra(g, indices=a, return_predecessors=True)
    b = int(np.argmax(np.where(np.isfinite(da), da, -1)))
    path = [b]
    while path[-1] != a:
        path.append(int(pred[path[-1]]))
    return np.array(path[::-1])


def centerline_from_volume(vol, vox_mm, origin_mm, thresh_frac=THRESH_FRAC, low_frac=LOW_FRAC):
    """Hysteresis threshold -> largest component -> skeleton -> longest path -> smooth.

    Returns (s, xyz_mm) in the frame of `origin_mm`. Orientation (which end
    is the ostium) is not known from a static volume; `orient_by_reference`
    is used for evaluation only.

    Ref: skeleton reused from centerline.py (owner's code); smoothing and
    resampling reused from data_gen.resample_and_smooth; hysteresis
    thresholding as in Canny, IEEE TPAMI 1986 (applied to a volume; section
    UNVERIFIED); thresholds OURS.
    """
    v = np.asarray(vol, dtype=float)
    pos = v[v > 0]
    if pos.size == 0:
        raise ValueError("empty volume")
    ref = np.percentile(pos, 99.9)
    lab, n = cc_label(v > low_frac * ref, structure=np.ones((3, 3, 3)))
    if n == 0:
        raise ValueError("no foreground")
    seeded = np.unique(lab[v > thresh_frac * ref])
    seeded = seeded[seeded > 0]
    if len(seeded) == 0:
        raise ValueError("no voxel above the high threshold")
    sizes = np.bincount(lab.ravel())
    mask = lab == seeded[np.argmax(sizes[seeded])]
    sk = cl.extract_centerline(mask, spacing=(vox_mm,) * 3)
    coords = sk[:, :3].astype(np.int64)
    path = longest_path(coords, vox_mm)
    pts = np.asarray(origin_mm) + coords[path] * vox_mm
    s, xyz, _ = dg.resample_and_smooth(pts, np.ones(len(pts)))
    return s, xyz


def tangents(xyz):
    """Unit tangents by central differences. Ref: OURS."""
    t = np.gradient(np.asarray(xyz, dtype=float), axis=0)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


# ============================ radius ============================
def half_max_radius(vol, vox_mm, origin_mm, xyz, n_rays=N_RAYS, r_max=R_MAX_MM, dr=DR_MM):
    """Lumen radius per centerline point: distance at which the cross-section
    profile falls to half of its central peak, median over `n_rays` rays in
    the plane normal to the tangent.

    For a uniform disk blurred by a symmetric kernel narrower than the disk,
    the half-maximum contour lies at the disk edge, and the estimate is
    independent of the intensity scale (time-averaged filling).

    Ref: OURS (half-maximum edge criterion, standard full-width-at-half-
    maximum idea). Sampling: trilinear (scipy map_coordinates, order 1).
    Returns R in mm (nan where no crossing was found).
    """
    v = np.asarray(vol, dtype=np.float32)
    origin = np.asarray(origin_mm, dtype=float)
    t = tangents(xyz)
    helper = np.where(np.abs(t[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = np.cross(t, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(t, e1)
    radii_axis = np.arange(0.0, r_max + 1e-9, dr)
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dirs = np.cos(ang)[None, :, None] * e1[:, None, :] + np.sin(ang)[None, :, None] * e2[:, None, :]  # P, A, 3
    pts = xyz[:, None, None, :] + dirs[:, :, None, :] * radii_axis[None, None, :, None]  # P, A, R, 3
    idx = (pts - origin) / vox_mm
    prof = map_coordinates(v, idx.reshape(-1, 3).T, order=1, mode="constant", cval=0.0).reshape(pts.shape[:3])
    peak = prof[:, :, radii_axis <= 0.5].max(axis=2).max(axis=1)  # common peak per point
    half = 0.5 * peak[:, None]
    out = np.full(prof.shape[:2], np.nan)
    for p in range(prof.shape[0]):
        for a in range(prof.shape[1]):
            row = prof[p, a]
            k0 = int(np.argmax(row))
            below = np.where(row[k0:] < half[p, 0])[0]
            if len(below) == 0:
                continue
            k = k0 + below[0]
            # linear interpolation between k-1 and k
            y0, y1 = row[k - 1], row[k]
            out[p, a] = radii_axis[k - 1] + (y0 - half[p, 0]) / (y0 - y1 + 1e-12) * dr
    return np.nanmedian(out, axis=1)


def orient_by_reference(s, xyz, r, ref_xyz_start):
    """Flip the centerline (positions AND radii) so it starts at the end
    nearest `ref_xyz_start`. EVALUATION ONLY (uses the ground-truth ostium).
    Ref: OURS."""
    if np.linalg.norm(xyz[-1] - ref_xyz_start) < np.linalg.norm(xyz[0] - ref_xyz_start):
        xyz, r = xyz[::-1].copy(), np.asarray(r)[::-1].copy()
    return s, xyz, r


def radius_error(xyz_est, r_est, s_gt, xyz_gt, r_gt, trim_mm=5.0):
    """Mean absolute radius error against ground truth, matched by nearest
    ground-truth centerline point; points within `trim_mm` of either GT end
    are excluded (end caps). Ref: OURS (DoD metric of Step 2.1)."""
    from scipy.spatial import cKDTree

    d, j = cKDTree(xyz_gt).query(xyz_est)
    keep = (s_gt[j] > trim_mm) & (s_gt[j] < s_gt[-1] - trim_mm) & np.isfinite(r_est)
    err = np.abs(r_est[keep] - r_gt[j[keep]])
    return {
        "mae_mm": float(err.mean()),
        # mean estimated / GT lumen area on the same matched points (Q scales
        # with A, Step 2.2 run 5); input of radius_scale
        "area_ratio": float((r_est[keep] ** 2).mean() / (r_gt[j[keep]] ** 2).mean()),
        "p90_mm": float(np.percentile(err, 90)),
        "bias_mm": float((r_est[keep] - r_gt[j[keep]]).mean()),
        "centerline_dist_mm": float(d[keep].mean()),
        "coverage": float(keep.mean()),
        "n": int(keep.sum()),
    }


def radius_scale(area_ratios, stat="mean"):
    """One multiplicative radius correction from calibration cases:
    k = 1 / sqrt(mean area ratio), so the mean lumen area (which sets Q,
    plan: tau = int A / Q ds) is unbiased on the calibration set. Fitted on
    training-split cases only and applied unchanged to validation / test.
    Multiplicative rather than an offset because the bias grows with the
    radius (Step 2.1 v2: throat nearly exact, healthy segments 0.08 to
    0.15 mm low). Ref: OURS (plan deviation approved Oct 6, option 1).
    stat="median" (Gate 1, Oct 6): robust to one outlier case (training case
    110_0, area ratio 0.64, pulled the 5-case mean k to 1.114 while the
    validation cases needed about 1.07).
    """
    a = np.asarray(area_ratios, float)
    m = np.median(a) if stat == "median" else a.mean()
    return float(1.0 / np.sqrt(m))


def coverage_metrics(xyz_est, r_est, s_gt, xyz_gt, lesion_center_mm, lesion_len_mm, r_min_gt_mm, tol_mm=2.0):
    """How much of the GT vessel the estimate covers, and the throat radius.

    gt_coverage: share of GT centerline points within tol_mm of the estimate;
    lesion_coverage: the same inside the lesion window; throat: estimated
    radius at the estimate point nearest the GT throat, its error and that
    point's distance. Added after run 2 (Oct 4): the radius MAE alone did not
    see that case 131_0 missed its proximal third with the stenosis.
    Ref: OURS (evaluation only).
    """
    from scipy.spatial import cKDTree

    d, j = cKDTree(xyz_est).query(xyz_gt)
    lesion = np.abs(s_gt - lesion_center_mm) <= lesion_len_mm / 2
    k = int(np.argmin(np.abs(s_gt - lesion_center_mm)))
    return {
        "gt_coverage": float((d < tol_mm).mean()),
        "lesion_coverage": float((d[lesion] < tol_mm).mean()),
        "throat_r_est_mm": float(r_est[j[k]]),
        "throat_err_mm": float(r_est[j[k]] - r_min_gt_mm),
        "throat_dist_mm": float(d[k]),
    }


def matched_gt_s(xyz_est, s_gt, xyz_gt):
    """GT arc length of the nearest GT point for each estimate point (for plots). Ref: OURS."""
    from scipy.spatial import cKDTree

    return s_gt[cKDTree(xyz_gt).query(xyz_est)[1]]


# ============================ tube-aligned Gaussians ============================
def _torch():
    import torch

    return torch


class TubeGaussians:
    """Chain of anisotropic Gaussians along the centerline, mapped to arc length.

    Per Gaussian i: centre p_i (on the centerline at s_i), unit quaternion
    q_i (rotation; major axis = local x after rotation), cross-section scale
    sigma_r,i = R_i / sqrt(2) (plan: A = 2 pi sigma^2), axial scale
    sigma_a,i (OURS: the sample spacing, so neighbours overlap), density mu_i.
    Scales use the 4DRGS bounded activation s = (s_max - s_min) sigmoid(s~)
    + s_min with s_min = 0.1 and s_max = 10 voxel spacings (4DRGS Sec 3.1
    Eq. 3; bounds Sec 4.1 Implementation Details), not exp.
    Parameters are torch tensors with requires_grad, so Step 2.2 can optimise
    them; `alignment_loss` is the plan's tube-alignment regularizer.

    Ref: plan, Approach (Gaussians at arc length s_i; A = 2 pi sigma^2,
    R = sqrt(2) sigma) and Step 2.1 ("tube-alignment regularizer (major axis
    along tangent)"). Parameterisation (quaternion, log-scales) follows the
    3D Gaussian splatting convention used by R2-Gaussian (gaussian_model.py:
    rotation as quaternion, scaling through exp); details OURS.
    """

    def __init__(self, s, xyz, radius, density=1.0, device="cpu", voxel_mm=0.25):
        torch = _torch()
        self.s_min, self.s_max = 0.1 * voxel_mm, 10.0 * voxel_mm
        s = np.asarray(s, dtype=np.float32)
        xyz = np.asarray(xyz, dtype=np.float32)
        r = np.asarray(radius, dtype=np.float32)
        spacing = float(np.median(np.diff(s))) if len(s) > 1 else 1.0
        self.s = torch.tensor(s, device=device)
        self._sqrt2 = float(np.sqrt(2.0))
        self.position = torch.tensor(xyz, device=device, requires_grad=True)
        self.raw_sigma_r = torch.tensor(self._inv_bound(r / np.sqrt(2.0)), device=device, requires_grad=True)
        self.raw_sigma_a = torch.tensor(self._inv_bound(np.full(len(s), spacing)), device=device, requires_grad=True)
        self.quat = torch.tensor(_quat_from_x_to(tangents(xyz)).astype(np.float32), device=device, requires_grad=True)
        self.density = torch.tensor(np.full(len(s), density, np.float32), device=device, requires_grad=True)

    def _inv_bound(self, sigma):
        """Raw value whose bounded activation gives `sigma` (clipped inside the bounds)."""
        x = np.clip((np.asarray(sigma, dtype=np.float64) - self.s_min) / (self.s_max - self.s_min), 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x)).astype(np.float32)

    def _bound(self, raw):
        """4DRGS Eq. 3: (s_max - s_min) sigmoid(raw) + s_min."""
        return (self.s_max - self.s_min) * raw.sigmoid() + self.s_min

    def parameters(self):
        return [self.position, self.raw_sigma_r, self.raw_sigma_a, self.quat, self.density]

    def sigma_r(self):
        return self._bound(self.raw_sigma_r)

    def sigma_a(self):
        return self._bound(self.raw_sigma_a)

    def radius(self):
        """R = sqrt(2) sigma_r (plan, Approach)."""
        return self.sigma_r() * self._sqrt2

    def area(self):
        """A = 2 pi sigma_r^2 (plan, Approach)."""
        torch = _torch()
        return 2.0 * torch.pi * self.sigma_r() ** 2

    def major_axis(self):
        """Rotated x axis of each Gaussian (unit)."""
        torch = _torch()
        q = self.quat / self.quat.norm(dim=1, keepdim=True)
        w, x, y, z = q.unbind(1)
        return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=1)

    def tangent(self):
        """Tangent from neighbouring centres (central differences, one-sided at the ends)."""
        torch = _torch()
        p = self.position
        t = torch.cat([p[1:2] - p[0:1], (p[2:] - p[:-2]) / 2, p[-1:] - p[-2:-1]], dim=0)
        return t / t.norm(dim=1, keepdim=True)

    def alignment_loss(self):
        """mean(1 - (major axis . tangent)^2): zero when every Gaussian points
        along the vessel, sign-free. Ref: plan Step 2.1 regularizer; form OURS."""
        return (1.0 - (self.major_axis() * self.tangent()).sum(1) ** 2).mean()

    def covariance(self):
        """Sigma_i = R diag(sigma_a^2, sigma_r^2, sigma_r^2) R^T. Ref: 3DGS convention."""
        torch = _torch()
        q = self.quat / self.quat.norm(dim=1, keepdim=True)
        w, x, y, z = q.unbind(1)
        rot = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
        ], 1)
        var = torch.stack([self.sigma_a() ** 2, self.sigma_r() ** 2, self.sigma_r() ** 2], 1)
        return rot @ torch.diag_embed(var) @ rot.transpose(1, 2)


def _quat_from_x_to(t):
    """Unit quaternions (w, x, y, z) rotating the x axis onto each row of t. Ref: standard."""
    x = np.array([1.0, 0.0, 0.0])
    t = np.asarray(t, dtype=float)
    c = np.cross(x, t)
    w = 1.0 + t @ x
    q = np.concatenate([w[:, None], c], axis=1)
    bad = w < 1e-8  # t = -x: rotate 180 deg about z
    q[bad] = np.array([0.0, 0.0, 0.0, 1.0])
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def build_from_volume(vol, vox_mm, origin_mm, median_n=5):
    """Static volume -> (s, xyz, R). Radius: gaps interpolated, then a short
    median filter (5 samples = 2.5 mm) against outliers. A Gaussian smoothing
    of 2 mm was tried first and flattened the stenosis (synthetic test: MAE
    0.27 mm instead of 0.04 mm), so it is not used. Ref: OURS."""
    from scipy.ndimage import median_filter

    s, xyz = centerline_from_volume(vol, vox_mm, origin_mm)
    r = half_max_radius(vol, vox_mm, origin_mm, xyz)
    good = np.isfinite(r)
    r = np.interp(s, s[good], r[good])
    r = median_filter(r, size=median_n, mode="nearest")
    return s, xyz, r


# ============================ R2-Gaussian I/O ============================
def write_r2g_case(render_npz, case_npz, bolus_npz, out_dir, n_views=60, n_test=20, mu=None):
    """Write one rendered case as an R2-Gaussian dataset folder (meta_data.json,
    proj_train / proj_test .npy, vol_gt.npy), without changing R2-Gaussian.

    Conventions copied from R2-Gaussian data_generator/synthetic_dataset/
    generate_data.py (commit f2579bf): projections flipped vertically
    ([::-1, :]) relative to tigre.Ax, angles in radians, volume stored as
    [x, y, z] = transpose of the TIGRE [z, y, x] array, scanner in physical
    units (R2-Gaussian rescales to [-1, 1]^3 itself, dataset_readers.py).
    Projections are divided by mu so R2-Gaussian reconstructs concentration
    (densities near 1, the range its defaults assume) (OURS).
    vol_gt = lumen fraction x concentration averaged over the training
    frames: the volume a static model of these views can best reach (used
    by R2-Gaussian's own PSNR / SSIM report).

    Ref: plan Step 2.1 ("Static R2-Gaussian on time-averaged views").
    Returns a dict with the paths and the frame mapping.
    """
    import json
    from pathlib import Path

    from flowsplat import render_tigre as rt

    out = Path(out_dir)
    (out / "proj_train").mkdir(parents=True, exist_ok=True)
    (out / "proj_test").mkdir(parents=True, exist_ok=True)
    z = np.load(render_npz)
    meta = json.loads(str(z["meta"]))
    mu = meta["mu_per_mm"] if mu is None else mu
    gd = meta["geometry"]
    proj = z["proj"].astype(np.float32) / mu
    ang = np.deg2rad(z["angles_deg"]).astype(float)
    vs = rt.view_subsets(len(ang), (n_views,))[str(n_views)]
    test = vs["test"][:: max(1, len(vs["test"]) // n_test)][:n_test] if vs["test"] else []
    files = {}
    for split, idxs in (("proj_train", vs["train"]), ("proj_test", test)):
        files[split] = []
        for k, j in enumerate(idxs):
            name = f"{split}/{split}_{k:04d}.npy"
            np.save(out / name, proj[j][::-1, :].copy())
            files[split].append({"file_path": name, "angle": float(ang[j]), "frame": int(j)})
    # time-averaged GT in the TIGRE array, then R2-Gaussian's [x, y, z]
    c = np.load(case_npz)
    b = np.load(bolus_npz)
    lum = rt.lumen_voxels(c["s"], c["xyz_mm"], c["r_mm"])
    perm = meta["axis_perm"]
    cbar = b["c"].astype(np.float32)[:, vs["train"]].mean(axis=1)
    vt = np.zeros([lum["shape"][p] for p in perm], np.float32)
    idx = lum["idx"][:, perm]
    vt[idx[:, 0], idx[:, 1], idx[:, 2]] = lum["frac"] * np.interp(lum["s_vox"], b["s"], cbar)
    np.save(out / "vol_gt.npy", vt.transpose(2, 1, 0).copy())
    nz, ny, nx = gd["nVoxel"]
    scanner = {
        "mode": "cone", "filter": None, "DSD": gd["DSD"], "DSO": gd["DSO"],
        "nDetector": gd["nDetector"], "sDetector": gd["sDetector"],
        "nVoxel": [nx, ny, nz], "sVoxel": gd["sVoxel"][::-1],
        "offOrigin": [0.0, 0.0, 0.0], "offDetector": [0.0, 0.0], "accuracy": gd["accuracy"],
        "totalAngle": float(np.rad2deg(ang[-1] - ang[0])), "startAngle": float(np.rad2deg(ang[0])),
    }
    (out / "meta_data.json").write_text(json.dumps(
        {"scanner": scanner, "vol": "vol_gt.npy", "bbox": [[-1, -1, -1], [1, 1, 1]], **files}, indent=2))
    return {"dir": str(out), "axis_perm": perm, "lumen": {k: lum[k] for k in ("shape", "origin_mm", "vox_mm")},
            "n_train": len(vs["train"]), "n_test": len(test), "n_lumen_voxels": int(len(idx))}


def r2g_volume_to_lumen_frame(vol_r2g, axis_perm):
    """R2-Gaussian volume [x, y, z] -> TIGRE [z, y, x] -> lumen-grid axes
    (inverse of render_tigre's axis permutation). Ref: OURS (inverse of
    write_r2g_case)."""
    vt = np.asarray(vol_r2g).transpose(2, 1, 0)
    return vt.transpose(np.argsort(axis_perm))


def fdk_volume_lumen_frame(render_npz, n_views=60, filt="hann"):
    """Step 2.1 FALLBACK init volume: TIGRE FDK of the training views,
    returned in the lumen-grid frame and in concentration units (/ mu).

    Ref: plan Step 2.1 fallback ("initialize directly from the
    ground-truth-free FDK volume skeleton"); FDK with Hann filter as the
    4DRGS FDK baseline (4DRGS Sec 4.1, "FDK with hann filtering", via
    TIGRE). Runs on: molab (TIGRE + CUDA).
    """
    import json

    import tigre.algorithms as algs

    from flowsplat import render_tigre as rt

    z = np.load(render_npz)
    meta = json.loads(str(z["meta"]))
    tr = rt.view_subsets(len(z["angles_deg"]), (n_views,))[str(n_views)]["train"]
    proj = (z["proj"].astype(np.float32)[tr] / meta["mu_per_mm"]).copy()
    ang = np.deg2rad(z["angles_deg"][tr]).astype(np.float32)
    vol_t = algs.fdk(proj, rt.tigre_geo(meta["geometry"]), ang, filter=filt)
    return np.asarray(vol_t).transpose(np.argsort(meta["axis_perm"])), meta


# ============================ 4DRGS I/O (Step 2.1 v2) ============================
# Plan deviation approved by the user on Oct 5 (Step 2.1 v2): 4DRGS (static
# Gaussian geometry + time-dependent attenuation) replaces the FDK skeleton
# as the init volume. Defaults of 4DRGS ModelParams (commit 38006ed):
FOURDRGS_THRES_FDK = 0.016  # thres_percent_fdk (absolute FDK value)
FOURDRGS_M1 = 30_000  # FDK-masked points sampled without replacement


def fourdrgs_transforms(gd, angles_rad):
    """transforms.json of a 4DRGS case for one TIGRE geometry dict.

    Keys and axis order as read by 4DRGS scene/dataset_readers.py
    readCameras and ct/tigre_ct.py (commit 38006ed): volume_* in [x, y, z]
    (TIGRE nVoxel / sVoxel / dVoxel reversed), proj_* as [width, height],
    PrimaryAngle in radians (passed to tigre.algorithms.fdk unchanged),
    SecondaryAngle ignored, origin and detector offsets fixed to 0.
    Every frame is listed; 4DRGS picks the training views itself with
    arange(0, N, N / Nviews), the same frames as render_tigre.view_subsets.
    Ref: 4DRGS code (data format), OURS (conversion).
    """
    if any(abs(v) > 0 for v in list(gd["offOrigin"]) + list(gd["offDetector"])):
        raise ValueError("4DRGS assumes zero origin and detector offsets")
    nv, nu = gd["nDetector"]
    dv, du = gd["dDetector"]
    return {
        "volume_resolution": [int(n) for n in gd["nVoxel"][::-1]],
        "volume_phy": [float(x) for x in gd["sVoxel"][::-1]],
        "volume_spacing": [float(x) for x in gd["dVoxel"][::-1]],
        "sad": float(gd["DSO"]),
        "sid": float(gd["DSD"]),
        "proj_resolution": [int(nu), int(nv)],
        "proj_phy": [float(nu * du), float(nv * dv)],
        "proj_spacing": [float(du), float(dv)],
        "N_views": int(len(angles_rad)),
        "frames": [{"file": f"frame_{j:03d}", "PrimaryAngle": float(a)} for j, a in enumerate(angles_rad)],
    }


def fourdrgs_runs(proj):
    """(mask_run, fill_run) with log(mask) - log(fill) = proj, the DSA that
    4DRGS readCameras forms. Images flipped vertically ([::-1, :]) relative
    to tigre.Ax, because 4DRGS tigre_ct.fdk flips them back before FDK.
    proj is the physical log ratio (mu x path length), as in real DSA, so the
    4DRGS absolute FDK threshold keeps its meaning. Ref: 4DRGS code, OURS.
    """
    img = np.asarray(proj, np.float64)[:, ::-1, :]
    return np.ones(img.shape, np.float32), np.exp(-img).astype(np.float32)


def fourdrgs_m1(n_above, m1=FOURDRGS_M1):
    """Number of FDK-masked init points. n_above: voxels of our TIGRE FDK of
    the same views above the 4DRGS threshold (its own FDK clips negative
    projections first, so its count differs slightly). 4DRGS default when
    n_above >= 2 m1 (margin), else half of them, as for R2-Gaussian in
    Step 2.1 run 1: np.random.choice without replacement fails if fewer
    voxels than m1 pass. Ref: OURS."""
    return m1 if n_above >= 2 * m1 else max(1, n_above // 2)


def write_4drgs_case(render_npz, out_dir):
    """Write one rendered case as a 4DRGS dataset folder (transforms.json,
    mask_run.nii.gz, fill_run.nii.gz) without changing 4DRGS. Needs
    SimpleITK (installed on molab with 4DRGS). Returns the transforms dict.
    Ref: 4DRGS code (format), OURS (conversion)."""
    import json
    from pathlib import Path

    import SimpleITK as sitk

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(render_npz)
    meta = json.loads(str(z["meta"]))
    tf = fourdrgs_transforms(meta["geometry"], np.deg2rad(z["angles_deg"]).astype(float))
    mask, fill = fourdrgs_runs(z["proj"])
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(out / "mask_run.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(fill), str(out / "fill_run.nii.gz"))
    (out / "transforms.json").write_text(json.dumps(tf, indent=1))
    return tf


def fourdrgs_volume_to_lumen_frame(vol_zyx, axis_perm):
    """4DRGS vol_pred.nii.gz array (SimpleITK order [z, y, x] = TIGRE order,
    test.py voxel_query_reconstruction writes vol.transpose(2, 1, 0)) ->
    lumen-grid axes. Ref: 4DRGS test.py (VQR output), OURS."""
    return np.asarray(vol_zyx).transpose(np.argsort(axis_perm))


# ============================ cone-beam Gaussian projector ============================
# Geometry convention of the detector relative to TIGRE (sign of the angle,
# a start offset, signs of the u and v detector axes). Calibrated on Oct 4
# against the TIGRE renders of Step 1.2 (ground-truth tube with the true
# concentration, 32 conventions, case 121_0): this default is the only one
# that matches (relative error 0.26 vs 0.65 or more for all others at
# 60 and 105 deg); results/step2.2_projector_convention.json.
DEFAULT_CONVENTION = {"s_theta": 1.0, "theta0": 0.0, "s_u": 1.0, "s_v": 1.0}


class ConeProjector:
    """Exact line integrals of 3D Gaussians along the rays of a flat-panel
    cone-beam scanner with the TIGRE geometry fields (DSO, DSD, nDetector
    [v, u], dDetector [v, u]; volume centred at the isocentre).

    For a ray from source S with unit direction r and a Gaussian with peak
    density rho, centre m and covariance Sigma (x0 = m - S, P = Sigma^-1):
        int rho exp(-1/2 (S + l r - m)^T P (S + l r - m)) dl
          = rho sqrt(2 pi / a) exp(-1/2 (c - b^2 / a)),
        a = r^T P r, b = r^T P x0, c = x0^T P x0.
    This is the X-ray rendering of R2-Gaussian (each Gaussian contributes its
    exact line integral, "rectified" splatting) evaluated per ray in PyTorch;
    exact for every ray, no local affine approximation.

    Ref: R2-Gaussian Sec 3.1 Eq 1 and Sec 4.2.1 (line integral of radiative
    Gaussians); plan, Approach (y = sum_i mu_I c_i(t) G_i^2D(p)). Closed form:
    standard Gaussian integral. Implementation in PyTorch instead of the CUDA
    rasterizer: OURS (about 300 tube Gaussians per case; runs on the M4).
    """

    def __init__(self, gd, convention=None, device="cpu", dtype=None):
        import torch

        self.torch = torch
        self.dtype = torch.float32 if dtype is None else dtype
        self.device = device
        self.dso, self.dsd = float(gd["DSO"]), float(gd["DSD"])
        self.nv, self.nu = (int(x) for x in gd["nDetector"])
        dv, du = (float(x) for x in gd["dDetector"])
        self.conv = dict(DEFAULT_CONVENTION if convention is None else convention)
        self.u = (torch.arange(self.nu, dtype=self.dtype, device=device) + 0.5 - self.nu / 2) * du
        self.v = (torch.arange(self.nv, dtype=self.dtype, device=device) + 0.5 - self.nv / 2) * dv

    def frame(self, theta):
        """Source position S and unit axes (e: source to detector, u, v)."""
        torch = self.torch
        c = self.conv
        th = c["s_theta"] * float(theta) + c["theta0"]
        cos, sin = math.cos(th), math.sin(th)
        s = torch.tensor([self.dso * cos, self.dso * sin, 0.0], dtype=self.dtype, device=self.device)
        e = torch.tensor([-cos, -sin, 0.0], dtype=self.dtype, device=self.device)
        u = c["s_u"] * torch.tensor([-sin, cos, 0.0], dtype=self.dtype, device=self.device)
        v = c["s_v"] * torch.tensor([0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
        return s, e, u, v

    def rays(self, theta):
        """Unit ray directions for every pixel, shape (nv * nu, 3), and S."""
        s, e, u, v = self.frame(theta)
        pts = (s + self.dsd * e)[None, None, :] + self.v[:, None, None] * v + self.u[None, :, None] * u
        r = pts.reshape(-1, 3) - s
        return r / r.norm(dim=1, keepdim=True), s

    def footprint(self, means, cov, theta, prec=None):
        """Line integral of each unit-density Gaussian along every ray of one
        view, shape (N, nv * nu)."""
        torch = self.torch
        prec = torch.linalg.inv(cov) if prec is None else prec  # (N, 3, 3)
        r, s = self.rays(theta)  # (P, 3)
        x0 = means - s  # (N, 3)
        px0 = torch.einsum("nij,nj->ni", prec, x0)
        c = (x0 * px0).sum(-1)  # (N,)
        pr = torch.einsum("nij,pj->npi", prec, r)  # (N, P, 3)
        a = (pr * r[None]).sum(-1)  # (N, P)
        b = px0 @ r.T  # (N, P)
        return torch.sqrt(2 * math.pi / a) * torch.exp(-0.5 * (c[:, None] - b * b / a))

    def render(self, means, cov, rho, thetas):
        """Projections (V, nv, nu) of N Gaussians with peak densities rho (V, N)."""
        torch = self.torch
        prec = torch.linalg.inv(cov)
        out = [(rho[j][:, None] * self.footprint(means, cov, th, prec)).sum(0) for j, th in enumerate(thetas)]
        return torch.stack(out).reshape(len(thetas), self.nv, self.nu)

    def footprints(self, means, cov, thetas):
        """Footprints of all views (V, N, nv * nu) for frozen geometry (Stage A)."""
        torch = self.torch
        prec = torch.linalg.inv(cov)
        return torch.stack([self.footprint(means, cov, th, prec) for th in thetas])


def bin_detector(proj, gd, f):
    """f x f detector binning (mean) of projections (V, nv, nu) and the
    matching geometry (fewer, larger pixels). Used to run on the M4. Ref: OURS."""
    import torch

    v, nv, nu = proj.shape
    nv2, nu2 = nv // f, nu // f
    p = proj[:, : nv2 * f, : nu2 * f].reshape(v, nv2, f, nu2, f).mean(dim=(2, 4))
    g = dict(gd)
    g["nDetector"] = [nv2, nu2]
    g["dDetector"] = [gd["dDetector"][0] * f, gd["dDetector"][1] * f]
    g["sDetector"] = [nv2 * g["dDetector"][0], nu2 * g["dDetector"][1]]
    return p, g


def lumen_to_world(xyz_mm, lumen, axis_perm):
    """Lumen-grid frame (mm, Step 1.2 renderer) -> TIGRE world frame (mm,
    volume centred at the isocentre, x = last array axis). Returns
    (world (N, 3), M) with M the axis permutation matrix. Ref: TIGRE voxel
    centres at (i + 0.5 - n/2) d; array order [z, y, x] (render_tigre)."""
    xyz = np.asarray(xyz_mm, dtype=float)
    vox = float(lumen["vox_mm"])
    shape_t = [lumen["shape"][p] for p in axis_perm]  # [nz, ny, nx]
    n_xyz = np.array([shape_t[2], shape_t[1], shape_t[0]], dtype=float)
    m = np.zeros((3, 3))
    m[0, axis_perm[2]] = m[1, axis_perm[1]] = m[2, axis_perm[0]] = 1.0
    idx = (xyz - np.asarray(lumen["origin_mm"])) / vox
    return (idx @ m.T + 0.5 - n_xyz / 2) * vox, m


def world_to_lumen(world_mm, lumen, axis_perm):
    """Inverse of lumen_to_world. Ref: OURS."""
    vox = float(lumen["vox_mm"])
    shape_t = [lumen["shape"][p] for p in axis_perm]
    n_xyz = np.array([shape_t[2], shape_t[1], shape_t[0]], dtype=float)
    _, m = lumen_to_world(np.zeros((1, 3)), lumen, axis_perm)
    idx = (np.asarray(world_mm, dtype=float) / vox - 0.5 + n_xyz / 2) @ m
    return np.asarray(lumen["origin_mm"]) + idx * vox
