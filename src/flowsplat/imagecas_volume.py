"""Turn one ImageCAS coronary label into an R2-Gaussian ground-truth volume.

Runs on: M4 and molab GPU (numpy + scipy only; file reading is done by the
caller with SimpleITK).

Used by notebooks/step0_imagecas_recon.py for the Step 0.1 DoD item "static
reconstruction of one ImageCAS vessel with R2-Gaussian renders correctly".
The output follows the R2-Gaussian synthetic-data convention: a cube volume,
values in [0, 1], `target_size`^3 voxels, later placed in the unit cube of
`cone_beam.yml` (sVoxel = 2.0) by R2-Gaussian's own generate_data.py.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Largest 26-connected component of a binary mask.

    ImageCAS labels hold the left and right coronary trees in one binary
    mask; the largest component is used as "one vessel" here (usually the
    left tree: LM, LAD, LCx). Isolating the LAD itself needs the centerline
    code of Step 1.1.

    Ref: OURS. Reason: the DoD asks for one vessel; a connected component is
    the simplest choice that needs no centerline. ImageCAS label format
    (binary coronary mask): Zeng et al., CMIG 2023, section UNVERIFIED.
    """
    mask = np.asarray(mask) > 0
    labels, n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        raise ValueError("Empty label: no foreground voxels.")
    sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, n + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def crop_to_bbox(mask: np.ndarray, spacing_mm, margin_mm: float = 5.0) -> np.ndarray:
    """Crop to the foreground bounding box plus a margin in millimetres.

    Ref: OURS. Reason: a coronary tree fills a small part of a CTA field of
    view; cropping keeps the vessel at about 0.4 mm per voxel in a 256^3 cube
    instead of about 1 mm for the full field.
    """
    idx = np.argwhere(mask)
    margin = np.ceil(margin_mm / np.asarray(spacing_mm, dtype=float)).astype(int)
    lo = np.maximum(idx.min(axis=0) - margin, 0)
    hi = np.minimum(idx.max(axis=0) + margin + 1, mask.shape)
    return mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]


def resample_isotropic(vol: np.ndarray, spacing_mm, new_spacing_mm: float):
    """Resample to isotropic spacing. Returns (volume, actual spacing per axis).

    Ref: R2-Gaussian code, data_generator/synthetic_dataset/process_raw_data.py,
    function `resample` (ndimage.zoom, mode="nearest") at commit f2579bf.
    Changed: target spacing is a parameter (theirs is fixed at 1).
    """
    spacing = np.asarray(spacing_mm, dtype=float)
    new_shape = np.round(np.asarray(vol.shape) * spacing / new_spacing_mm)
    factor = new_shape / np.asarray(vol.shape)
    out = ndimage.zoom(vol, factor, mode="nearest")
    return out, spacing / factor


def expand_to_cube(vol: np.ndarray) -> np.ndarray:
    """Zero-pad to a cube, centred.

    Ref: R2-Gaussian code, process_raw_data.py, function `expand_to_cube`
    (commit f2579bf). Taken as is.
    """
    n = max(vol.shape)
    pad = [((n - s) // 2, n - s - (n - s) // 2) for s in vol.shape]
    return np.pad(vol, pad, mode="constant", constant_values=0)


def resize_cube(vol: np.ndarray, target_size: int) -> np.ndarray:
    """Resize a cube to target_size^3.

    Ref: R2-Gaussian code, process_raw_data.py, function `resize`
    (ndimage.zoom, default spline order, mode="nearest"), commit f2579bf.
    Taken as is.
    """
    zoom = [target_size / s for s in vol.shape]
    if any(z != 1.0 for z in zoom):
        vol = ndimage.zoom(vol, zoom, mode="nearest")
    return vol


def label_to_r2g_volume(label: np.ndarray, spacing_mm, target_size: int = 256, margin_mm: float = 5.0):
    """ImageCAS label -> R2-Gaussian GT volume and a record of the geometry.

    Steps: largest component, crop with margin, isotropic resampling at the
    finest input spacing, centred cube padding, resize to target_size^3,
    clip to [0, 1]. The vessel lumen gets density 1 and the background 0,
    i.e. an ideal contrast-filled, background-subtracted (DSA-like) volume.

    Ref: preprocessing order from R2-Gaussian process_raw_data.py (`process_raw`:
    normalise to [0, 1], `reshape_vol` with mode "expand" = resample +
    expand_to_cube, then `resize`, clip). Vessel-only density: OURS (the
    project renders contrast in the lumen only; plan, Data and evaluation).
    """
    spacing = np.asarray(spacing_mm, dtype=float)
    comp = largest_component(label)
    crop = crop_to_bbox(comp, spacing, margin_mm).astype(np.float32)
    iso_mm = float(spacing.min())
    vol, iso_spacing = resample_isotropic(crop, spacing, iso_mm)
    vol = expand_to_cube(vol.clip(0.0, 1.0))
    cube_mm = float(vol.shape[0] * iso_spacing.mean())
    vol = resize_cube(vol, target_size).clip(0.0, 1.0).astype(np.float32)
    info = {
        "input_shape": list(label.shape),
        "input_spacing_mm": spacing.tolist(),
        "component_voxels": int(comp.sum()),
        "crop_shape": list(crop.shape),
        "cube_side_mm": round(cube_mm, 2),
        "voxel_mm": round(cube_mm / target_size, 4),
        "target_size": target_size,
        "foreground_voxels_out": int((vol > 0.5).sum()),
    }
    return vol, info


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap of two binary masks (1.0 when both are empty).

    Ref: standard Dice coefficient (Dice, Ecology 1945), used here as an
    extra check of the vessel shape; R2-Gaussian itself reports PSNR/SSIM.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(a, b).sum() / s)


def mips(vol: np.ndarray) -> list:
    """Maximum intensity projections along x, y, z (for the figure).

    Ref: OURS (visual check only).
    """
    return [np.asarray(vol).max(axis=i) for i in range(3)]
