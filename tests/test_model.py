"""Runs on: M4 (CPU only). Tests for src/flowsplat/model.py (Step 2.1)."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from flowsplat import data_gen as dg
from flowsplat import model as md
from flowsplat import render_tigre as rt


def _synthetic_volume(blur_mm=0.3, brightness=(0.3, 1.0)):
    """Curved stenosed tube, partial volume, brightness varying along s
    (time-averaged filling), blurred like a reconstruction."""
    s = np.arange(0, 60.0001, 0.5)
    xyz = np.stack([s, 8 * np.sin(s / 15.0), 0.2 * s], axis=1)
    r = dg.insert_stenosis(s, np.linspace(1.8, 1.2, len(s)), 0.5, 10.0, 30.0)
    lum = rt.lumen_voxels(s, xyz, r)
    vol = np.zeros(lum["shape"], np.float32)
    b = np.interp(lum["s_vox"], [0, s[-1]], brightness)
    vol[tuple(lum["idx"].T)] = lum["frac"] * b
    vol = gaussian_filter(vol, blur_mm / lum["vox_mm"])
    return vol, lum, s, xyz, r


@pytest.fixture(scope="module")
def recon():
    vol, lum, s, xyz, r = _synthetic_volume()
    s_e, xyz_e, r_e = md.build_from_volume(vol, lum["vox_mm"], lum["origin_mm"])
    s_e, xyz_e, r_e = md.orient_by_reference(s_e, xyz_e, r_e, xyz[0])
    return s_e, xyz_e, r_e, s, xyz, r


def test_centerline_follows_truth(recon):
    s_e, xyz_e, r_e, s, xyz, r = recon
    err = md.radius_error(xyz_e, r_e, s, xyz, r)
    assert err["centerline_dist_mm"] < 0.2
    assert err["coverage"] > 0.7
    assert abs(s_e[-1] - s[-1]) < 4.0  # end caps shorten the skeleton by about one radius each


def test_radius_error_below_dod_despite_brightness_change(recon):
    s_e, xyz_e, r_e, s, xyz, r = recon
    err = md.radius_error(xyz_e, r_e, s, xyz, r)
    assert err["mae_mm"] < 0.1  # DoD is 0.2 mm
    assert abs(err["bias_mm"]) < 0.1


def test_half_max_radius_scale_free():
    vol, lum, s, xyz, r = _synthetic_volume()
    pts = xyz[20:100:10]
    r1 = md.half_max_radius(vol, lum["vox_mm"], lum["origin_mm"], pts)
    r2 = md.half_max_radius(7.3 * vol, lum["vox_mm"], lum["origin_mm"], pts)
    assert np.allclose(r1, r2, atol=1e-6)


def test_tube_gaussians_init_and_alignment():
    torch = pytest.importorskip("torch")
    s = np.arange(0, 20.0001, 0.5)
    xyz = np.stack([s, np.sin(s / 5), np.zeros_like(s)], axis=1)
    r = np.full_like(s, 1.5)
    tg = md.TubeGaussians(s, xyz, r)
    assert torch.allclose(tg.radius(), torch.full((len(s),), 1.5), atol=1e-5)
    assert torch.allclose(tg.area(), torch.full((len(s),), np.pi * 1.5**2), rtol=1e-5)  # 2 pi sigma^2 = pi R^2
    assert tg.alignment_loss().item() < 1e-3  # init: major axis along the tangent
    assert tg.s_min == pytest.approx(0.025) and tg.s_max == pytest.approx(2.5)  # 4DRGS bounds, 0.25 mm voxels
    with torch.no_grad():
        tg.raw_sigma_r.fill_(50.0)
    assert tg.sigma_r().max().item() <= 2.5 + 1e-6  # bounded, no needle kernels
    tg = md.TubeGaussians(s, xyz, r)
    with torch.no_grad():
        tg.quat.copy_(torch.tensor([[0.7071, 0.0, 0.0, 0.7071]]).repeat(len(s), 1))  # x rotated onto y
    loss = tg.alignment_loss()
    assert loss.item() > 0.5
    loss.backward()
    assert tg.quat.grad is not None and torch.isfinite(tg.quat.grad).all()
    cov = tg.covariance()
    assert cov.shape == (len(s), 3, 3)
    assert torch.allclose(cov, cov.transpose(1, 2), atol=1e-6)


def test_orient_flips_radius_with_positions():
    s = np.arange(5.0)
    xyz = np.stack([s, 0 * s, 0 * s], 1)
    r = np.array([2.0, 1.9, 1.8, 1.7, 1.6])
    _, x2, r2 = md.orient_by_reference(s, xyz, r, np.array([4.0, 0, 0]))
    assert x2[0, 0] == 4.0 and r2[0] == 1.6


def test_write_r2g_case_roundtrip(tmp_path):
    """Fake render of a small tube: files, flips and the frame mapping back."""
    import json

    from flowsplat import render_tigre as rt

    s = np.arange(0, 30.0001, 0.5)
    xyz = np.stack([s, 0.2 * s, 0.5 * s], 1)
    r = np.full_like(s, 1.5)
    np.savez(tmp_path / "case.npz", s=s, xyz_mm=xyz, r_mm=r)
    t = np.linspace(0, 4.4, 133)
    c = np.tile(np.linspace(0, 1, 133), (len(s), 1)).astype(np.float16)
    np.savez(tmp_path / "bolus.npz", s=s, c=c, t_s=t, angles_deg=np.linspace(0, 198, 133))
    lum = rt.lumen_voxels(s, xyz, r)
    perm = rt.axis_permutation("LAS")
    shape = [lum["shape"][p] for p in perm]
    gd = rt.tigre_geometry_dict(shape)
    proj = np.random.default_rng(0).random((133, *gd["nDetector"])).astype(np.float16)
    meta = {"geometry": gd, "axis_perm": perm, "mu_per_mm": 0.1}
    np.savez(tmp_path / "render.npz", proj=proj, angles_deg=np.linspace(0, 198, 133), t_s=t, meta=json.dumps(meta))
    info = md.write_r2g_case(tmp_path / "render.npz", tmp_path / "case.npz", tmp_path / "bolus.npz", tmp_path / "r2g", n_views=30)
    m = json.loads((tmp_path / "r2g" / "meta_data.json").read_text())
    assert len(m["proj_train"]) == 30 and info["n_test"] == 20
    assert m["scanner"]["nVoxel"] == shape[::-1] and m["scanner"]["DSO"] == 750.0
    p0 = np.load(tmp_path / "r2g" / m["proj_train"][0]["file_path"])
    assert np.allclose(p0, proj[0].astype(np.float32)[::-1] / 0.1)  # flipped, divided by mu
    assert m["proj_train"][-1]["frame"] == 128  # 4DRGS subset rule, 30 of 133
    assert m["proj_train"][-1]["angle"] == pytest.approx(np.deg2rad(198.0 * 128 / 132))
    vg = np.load(tmp_path / "r2g" / "vol_gt.npy")
    assert list(vg.shape) == shape[::-1]
    back = md.r2g_volume_to_lumen_frame(vg, perm)
    assert list(back.shape) == lum["shape"]
    # mean concentration of a linear ramp over the 30 training frames (4DRGS subset)
    expected = np.mean([np.linspace(0, 1, 133)[j] for j in rt.view_subsets(133, (30,))["30"]["train"]])
    v = back[tuple(lum["idx"].T)] / lum["frac"]
    assert np.allclose(v, expected, atol=0.005)


def test_dim_segment_hysteresis_option():
    """Hysteresis (low_frac 0.1) keeps a proximal segment at 15 % brightness;
    the default single threshold (0.2) loses it. The default stays single
    because hysteresis followed FDK streaks on real data (run 3, Oct 4)."""
    s = np.arange(0, 60.0001, 0.5)
    xyz = np.stack([s, 8 * np.sin(s / 15.0), 0.2 * s], axis=1)
    r = dg.insert_stenosis(s, np.linspace(1.8, 1.2, len(s)), 0.5, 8.0, 15.0)
    lum = rt.lumen_voxels(s, xyz, r)
    vol = np.zeros(lum["shape"], np.float32)
    b = np.where(lum["s_vox"] < 25.0, 0.15, 1.0)  # dim proximal 25 mm, stenosis inside
    vol[tuple(lum["idx"].T)] = lum["frac"] * b
    vol = gaussian_filter(vol, 0.3 / lum["vox_mm"])
    s_e, xyz_e = md.centerline_from_volume(vol, lum["vox_mm"], lum["origin_mm"], low_frac=0.1)
    assert s_e[-1] > 54.0  # whole vessel (minus end caps), not only the bright 35 mm
    s_e1, _ = md.centerline_from_volume(vol, lum["vox_mm"], lum["origin_mm"])
    assert s_e1[-1] < 40.0  # default single threshold loses the dim part
