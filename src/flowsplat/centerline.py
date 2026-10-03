"""Centerline + radius extraction from ImageCAS labels.

Runs on: M4 (CPU only).

Reused from the project owner's earlier code (plan: "reuse centerline.py"):
repo cs6999-coronary-centerline, commit f17a5f07e65f2e8ba817614e5197a6ebc2bd563b
(2026-09-18):
  - src/coronarycl/centerline.py: label constants, _build_adjacency,
    _traversal_order, _classify_topology, extract_centerline (verbatim).
  - build_dataset_v3.py: side_labels, split_components, to_isotropic
    (verbatim; constants MIN_COMPONENT_FRAC, _CONN26 copied with them).
Ref for the method: classical skeletonization of the expert mask, radius from
the Euclidean distance transform in mm (ImageCAS masks: Zeng et al. 2023,
Sec 3; skeletonize: scikit-image). Changes: none in the copied functions;
I/O helpers (extract_case, extract_all) are not copied.
"""

import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass, convolve, distance_transform_edt, zoom
from scipy.ndimage import label as cc_label
from skimage.morphology import skeletonize

# From build_dataset_v3.py (same commit).
MIN_COMPONENT_FRAC = 0.05
_CONN26 = np.ones((3, 3, 3), int)

# Branch/topology label values (5th column of the output array).
LABEL_ENDPOINT = 0     # 1 skeleton neighbor  -- tip of a vessel branch
LABEL_REGULAR = 1      # 2 skeleton neighbors -- ordinary point along a branch
LABEL_BIFURCATION = 2  # 3+ skeleton neighbors -- branch point (vessel splits)

_NEIGHBOR_KERNEL = np.ones((3, 3, 3))
_NEIGHBOR_KERNEL[1, 1, 1] = 0  # don't count the voxel itself
# 26-connected offsets for the path-traversal below (separate from
# _NEIGHBOR_KERNEL, which is only used for the topology neighbor-count).
_NEIGHBOR_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


def _build_adjacency(coords: np.ndarray) -> list:
    """26-connected adjacency list over skeleton voxels, index-aligned
    with `coords` (adjacency[i] = indices j such that coords[i] and
    coords[j] are 26-connected neighbors)."""
    index_of = {tuple(c): i for i, c in enumerate(coords.astype(np.int64))}
    adjacency = [[] for _ in range(len(coords))]
    for i, c in enumerate(coords.astype(np.int64)):
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            j = index_of.get((c[0] + dx, c[1] + dy, c[2] + dz))
            if j is not None:
                adjacency[i].append(j)
    return adjacency


def _traversal_order(coords: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Reorder skeleton points via DFS traversal over 26-connectivity,
    replacing np.argwhere()'s raster-scan order.

    Rationale: the 1D-UNet denoiser (Conv1d, strided pool/upsample) assumes
    array-adjacent points are spatially/topologically adjacent. Raster
    order violates this. Measured on a synthetic single-bifurcation test
    case: raster order has only 53.4% of consecutive pairs 26-connected
    (mean graph-geodesic hop distance 9.7, up to 37 hops between
    "consecutive" indices). DFS-preorder from a radius-max endpoint raises
    this to 98.6%, with residual jumps only at true bifurcation points
    (irreducible for any 1D serialization of a branching tree).

    Handles disconnected skeleton components (segmentation/skeletonization
    artifacts) by traversing each separately, largest first. Root of each
    component is the largest-radius endpoint (proxy for the proximal/
    ostium end, since coronary radius decreases distally); falls back to
    the largest-radius point overall if the component has no endpoint
    (e.g. a closed-loop artifact).

    Returns:
        order: (N,) int64 permutation such that coords[order],
        radii[order], branch_labels[order] are path-ordered.
    """
    n = len(coords)
    adjacency = _build_adjacency(coords)
    degree = np.array([len(nbrs) for nbrs in adjacency])

    unvisited = set(range(n))
    components = []
    while unvisited:
        start = next(iter(unvisited))
        stack, seen = [start], {start}
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in adjacency[node]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        components.append(comp)
        unvisited -= seen
    components.sort(key=len, reverse=True)

    order = []
    for comp in components:
        endpoints = [i for i in comp if degree[i] <= 1]
        candidates = endpoints if endpoints else comp
        root = max(candidates, key=lambda i: radii[i])

        stack, seen = [root], {root}
        comp_order = []
        while stack:
            node = stack.pop()
            comp_order.append(node)
            for nb in sorted(adjacency[node]):  # sorted -> deterministic
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        order.extend(comp_order)

    order = np.array(order, dtype=np.int64)
    assert len(order) == n and len(set(order.tolist())) == n, \
        "traversal order lost or duplicated points -- bug"
    return order


def _classify_topology(skeleton: np.ndarray) -> np.ndarray:
    """For each skeleton voxel, count its 26-connected skeleton neighbors
    and classify it as an endpoint, regular point, or bifurcation.

    Returns an array of labels aligned with np.argwhere(skeleton) order.
    """
    neighbor_count = convolve(skeleton.astype(np.uint8), _NEIGHBOR_KERNEL,
                              mode="constant", cval=0)
    counts_at_skeleton = neighbor_count[skeleton]

    labels = np.full(counts_at_skeleton.shape, LABEL_REGULAR, dtype=np.int64)
    labels[counts_at_skeleton <= 1] = LABEL_ENDPOINT
    labels[counts_at_skeleton >= 3] = LABEL_BIFURCATION
    return labels


def extract_centerline(volume: np.ndarray, spacing=None) -> np.ndarray:
    """Skeletonize `volume` and return (N, 5): x, y, z (voxel index),
    radius, topology label -- path-ordered by `_traversal_order`.

    `spacing` is the (sx, sy, sz) voxel size in mm. When given, the
    distance transform is computed with physical sampling, so the radius
    column is in **mm**. When omitted the radius is in **voxels**, which
    is only correct for isotropic data -- see the note below.

    NOTE (bug fixed 2026-09): earlier revisions called
    distance_transform_edt(volume) with no `sampling`, producing radii in
    voxel units (recognisable as exact square roots of integers) that were
    later scaled by mean(spacing) in preprocessing.voxel_to_mm. That
    isotropic approximation is wrong for anisotropic ImageCAS data
    (typically 0.377 x 0.377 x 0.5 mm). Pass `spacing` to get true mm and
    do NOT rescale afterwards.
    """
    skeleton = skeletonize(volume)
    dist = distance_transform_edt(volume, sampling=spacing)
    coords = np.argwhere(skeleton)
    radii = dist[skeleton]
    branch_labels = _classify_topology(skeleton)

    order = _traversal_order(coords, radii)
    coords, radii, branch_labels = (
        coords[order], radii[order], branch_labels[order])

    return np.concatenate(
        [coords, radii[:, None], branch_labels[:, None]], axis=1
    )


def side_labels(affine, centroids, spacing):
    """Name exactly two components by anatomical side from the NIfTI axis codes.
    Returns (names, separation_mm), or (None, nan) if the affine has no L/R axis.
    Never called with one component -- side is not identifiable from one centroid.
    """
    try:
        codes = nib.aff2axcodes(affine)
    except Exception:
        return None, float("nan")
    ax = next((i for i, c in enumerate(codes) if c in ("L", "R")), None)
    if ax is None or len(centroids) != 2:
        return None, float("nan")
    vals = [c[ax] for c in centroids]
    sep_mm = abs(vals[0] - vals[1]) * float(spacing[ax])
    order = np.argsort(vals)
    # codes[ax]=="L": increasing index -> more Left, so LARGEST value is LCA.
    # codes[ax]=="R": increasing index -> more Right, so SMALLEST value is LCA.
    ranked = order[::-1] if codes[ax] == "L" else order
    names = [None] * len(vals)
    for rank, ci in enumerate(ranked):
        names[ci] = "LCA" if rank == 0 else "RCA"
    return names, sep_mm


def split_components(mask, affine, spacing, single_component="skip",
                     extra_components="skip", min_lr_sep_mm=10.0):
    """One sample per coronary system.  Returns (components, skip_reason).

    2 significant components -> LCA/RCA by affine L-R axis, if separated
    1 component              -> merged tree; side not identifiable -> skip
    >2 components            -> keeping the 2 largest discards anatomy -> skip
    """
    lab, n = cc_label(mask, structure=_CONN26)
    if n == 0:
        return [], "no components"
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    sig = [i for i in range(1, n + 1) if sizes[i] >= MIN_COMPONENT_FRAC * mask.sum()]
    sig = sorted(sig, key=lambda i: -sizes[i])
    if not sig:
        return [], "no significant components"

    if len(sig) == 1:
        if single_component != "keep":
            return [], "1 significant component (LCA/RCA merged); side not identifiable"
        i = sig[0]
        return [dict(name="MERGED", mask=(lab == i), n_vox=int(sizes[i]),
                     lr_sep_mm=float("nan"))], None

    if len(sig) > 2:
        frac3 = sizes[sig[2]] / mask.sum()
        if extra_components != "largest2":
            return [], (f"{len(sig)} significant components "
                        f"(3rd = {frac3:.1%} of mask) -- would discard anatomy")
        sig = sig[:2]

    keep = sig[:2]
    centroids = [np.array(center_of_mass(lab == i)) for i in keep]
    names, sep_mm = side_labels(affine, centroids, spacing)
    if names is None:
        return [], "no L/R axis in NIfTI affine; cannot name LCA/RCA"
    if not np.isfinite(sep_mm) or sep_mm < min_lr_sep_mm:
        return [], f"L-R centroid separation {sep_mm:.1f} mm < {min_lr_sep_mm} mm (ambiguous side)"
    return [dict(name=names[j], mask=(lab == i), n_vox=int(sizes[i]),
                 lr_sep_mm=float(sep_mm)) for j, i in enumerate(keep)], None


def to_isotropic(mask_crop, spacing, iso):
    """Nearest-neighbour resample to an isotropic grid.

    Returns (mask, downsampled_axes).  When iso <= min(spacing) every zoom
    factor is >= 1 and this is pure upsampling, so no thin vessel can be lost.
    That precondition is NOT automatic -- 508/1000 ImageCAS cases have in-plane
    spacing below 0.35 mm -- so the violated axes are returned and the caller
    gates on the resulting topology (P1-h).
    """
    factors = np.array(spacing, float) / float(iso)
    down = [int(a) for a in np.where(factors < 1.0)[0]]
    out = zoom(mask_crop.astype(np.uint8), factors, order=0,
               grid_mode=True, mode="nearest")
    return out.astype(bool), down
