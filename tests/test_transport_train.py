"""Runs on: M4 (CPU only). Tests for transport.py, the projector in model.py and train.py (Step 2.2)."""

import math

import numpy as np
import pytest
import torch

from flowsplat import model as md
from flowsplat import train as tr
from flowsplat.transport import Transport, gamma_variate


def _gd(nv=48, nu=64, d=1.0):
    return {"DSO": 750.0, "DSD": 1200.0, "nDetector": [nv, nu], "dDetector": [d, d]}


# ---------------- transport ----------------
def test_gamma_variate_matches_bolus_sim():
    from flowsplat import bolus_sim as bs

    t = np.linspace(0, 4.4, 200)
    a = gamma_variate(torch.tensor(t), torch.tensor(0.4), torch.tensor(3.2), torch.tensor(0.33)).numpy()
    assert np.allclose(a, bs.gamma_variate(t, 0.4, 3.2, 0.33), atol=1e-6)


def test_tau_equals_volume_over_q():
    tr_ = Transport(q_mm3s=500.0)
    pos = torch.stack([torch.zeros(41), torch.linspace(0, 20, 41), torch.zeros(41)], 1)
    area = torch.full((41,), 7.0)
    tau = tr_.tau(pos, area)
    assert tau[0] == 0
    assert float(tau[-1].detach()) == pytest.approx(20 * 7.0 / 500.0, rel=1e-5)
    assert torch.all(tau[1:] > tau[:-1])


def test_density_gives_linear_attenuation_k_c_A():
    """A chain of Gaussians with the density normalisation reproduces a tube of
    area A filled with attenuation k c (mass matching of the plan)."""
    s = np.arange(0, 20.0001, 0.5)
    xyz = np.stack([np.zeros_like(s), s, np.zeros_like(s)], 1)
    tg = md.TubeGaussians(s, xyz, np.full_like(s, 1.5))
    trn = Transport(1e9, t0=-10.0, k=0.1)  # huge Q: c = C_in(t) everywhere
    t = torch.tensor([float(-10 + 3.0 * 0.35)])  # inlet peak, c = 1
    rho = trn.density(tg.position, tg.area(), tg.sigma_a(), t)[0]
    # line density along the axis = sum_i rho_i * (2 pi sigma_r^2) * sqrt(2 pi) sigma_a / ds  ->  k * c * A
    lin = (rho * tg.area() * math.sqrt(2 * math.pi) * tg.sigma_a() / 0.5)[5:-5]
    assert torch.allclose(lin, torch.full_like(lin, 0.1 * math.pi * 1.5**2), rtol=1e-4)


# ---------------- projector ----------------
def test_projector_isotropic_peak_and_mass():
    pj = md.ConeProjector(_gd(nv=64, nu=64, d=0.5))
    sig, rho = 3.0, 0.2
    cov = torch.eye(3)[None] * sig**2
    img = pj.render(torch.zeros(1, 3), cov, torch.tensor([[rho]]), [0.0])[0]
    assert float(img.max()) == pytest.approx(rho * math.sqrt(2 * math.pi) * sig, rel=2e-3)
    mag = 1200.0 / 750.0
    mass = float(img.sum()) * 0.5 * 0.5 / mag**2
    assert mass == pytest.approx(rho * (2 * math.pi) ** 1.5 * sig**3, rel=0.01)


def test_projector_position_and_anisotropy():
    pj = md.ConeProjector(_gd(nv=47, nu=63, d=1.0))  # odd sizes: a pixel centre on the central ray
    # Gaussian elongated along x (the central ray direction at theta = 0)
    cov = torch.diag(torch.tensor([16.0, 1.0, 1.0]))[None]
    img = pj.render(torch.zeros(1, 3), cov, torch.tensor([[1.0]]), [0.0])[0]
    assert float(img.max()) == pytest.approx(math.sqrt(2 * math.pi) * 4.0, rel=5e-3)
    # off-axis point y = 10 mm lands at u = 10 * DSD / DSO = 16 mm (default convention)
    img = pj.render(torch.tensor([[0.0, 10.0, 0.0]]), torch.eye(3)[None], torch.tensor([[1.0]]), [0.0])[0]
    col = int(img.sum(0).argmax())
    assert (col + 0.5 - 31.5) * 1.0 == pytest.approx(16.0, abs=0.6)


def test_lumen_world_roundtrip():
    lumen = {"vox_mm": 0.25, "shape": [120, 200, 160], "origin_mm": [3.0, -2.0, 7.5]}
    xyz = np.random.default_rng(0).uniform(0, 30, (20, 3)) + np.array(lumen["origin_mm"])
    w, m = md.lumen_to_world(xyz, lumen, [2, 1, 0])
    assert np.allclose(md.world_to_lumen(w, lumen, [2, 1, 0]), xyz)
    assert np.allclose(np.abs(np.linalg.det(m)), 1.0)


# ---------------- toy end-to-end: Stage A recovers Q ----------------
def _toy(q_true=300.0):
    s = np.arange(0, 30.0001, 0.5)
    xyz = np.stack([4 * np.sin(s / 10), s - 15, 0.2 * s], 1)
    r = np.full_like(s, 1.5)
    truth = tr.CaseModel(s, xyz, r, q_mm3s=q_true)
    with torch.no_grad():
        truth.transport.t0.fill_(0.6)
        truth.transport.log_k.fill_(math.log(0.1))
    idx = list(range(0, 133, 5))  # 27 views
    thetas = list(np.deg2rad(np.linspace(0, 198, 133))[idx])
    times = torch.tensor(np.linspace(0, 4.4, 133)[idx], dtype=torch.float32)
    pj = md.ConeProjector(_gd(nv=40, nu=56, d=1.0))
    with torch.no_grad():
        proj = truth.render(pj, thetas, times)
    return s, xyz, r, proj, thetas, times, pj


def test_toy_stage_a_recovers_q_and_direction():
    torch.manual_seed(0)
    s, xyz, r, proj, thetas, times, pj = _toy(q_true=300.0)
    m, info = tr.fit_case(proj, thetas, times, s, xyz, r, pj, iters_dir=60, iters_a=250, iters_b=0,
                          q_init=480.0, log=lambda *a: None)
    assert info["direction"] == +1
    assert float(m.transport.q()) == pytest.approx(300.0, rel=0.05)


def test_ssim_identical_is_one_and_loss_zero():
    x = torch.rand(3, 20, 24)
    assert float(tr.ssim(x, x, 1.0)) == pytest.approx(1.0, abs=1e-5)
    loss, l1 = tr.image_loss(x, x, 1.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_toy_stage_b_refines_radius():
    """Start the tube 0.15 mm too wide; Stage B (geometry + transport) must
    bring the radius back towards the truth (prototype of the joint fit)."""
    torch.manual_seed(0)
    s, xyz, r, proj, thetas, times, pj = _toy(q_true=300.0)
    r0 = r + 0.15
    m, info = tr.fit_case(proj, thetas, times, s, xyz, r0, pj, iters_dir=40, iters_a=150, iters_b=300,
                          q_init=300.0, log=lambda *a: None)
    rad = m.tube.radius().detach().numpy()
    err0 = np.abs(r0 - r)[5:-5].mean()
    err1 = np.abs(rad - r)[5:-5].mean()
    assert err1 < 0.5 * err0
    assert float(m.transport.q().detach()) == pytest.approx(300.0, rel=0.08)


def test_half_rise_x():
    from flowsplat.transport import half_rise_x

    for a in (2.5, 3.5, 4.5):
        al = torch.tensor(a, requires_grad=True)
        x = half_rise_x(al)
        val = gamma_variate(torch.tensor(0.0) + 1.0 * x, torch.tensor(0.0), al, torch.tensor(1.0) / al)
        assert float(val.detach()) == pytest.approx(0.5, abs=1e-6)
        x.backward()
        assert torch.isfinite(al.grad)
    # gradient check against finite differences
    al = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    half_rise_x(al).backward()
    h = 1e-5
    fd = (half_rise_x(torch.tensor(3.0 + h, dtype=torch.float64)) - half_rise_x(torch.tensor(3.0 - h, dtype=torch.float64))) / (2 * h)
    assert float(al.grad) == pytest.approx(float(fd), rel=1e-4)


def test_measured_arrival_matches_model_on_toy():
    s, xyz, r, proj, thetas, times, pj = _toy(q_true=300.0)
    truth = tr.CaseModel(s, xyz, r, q_mm3s=300.0)
    with torch.no_grad():
        truth.transport.t0.fill_(0.6)
        truth.transport.log_k.fill_(math.log(0.1))
        cache = pj.footprints(truth.tube.position, truth.tube.covariance(), thetas)
        arr, valid = tr.measure_arrival(proj, cache, truth.transport.density_full(truth.tube.position, truth.tube.sigma_a()), times)
        model_arr = truth.transport.arrival(truth.tube.position, truth.tube.area())
    assert valid.float().mean() > 0.8
    dt = 4.4 / 132 * 5  # view spacing of the toy (every 5th frame)
    assert float((arr - model_arr)[valid].abs().median()) < 0.5 * dt


def test_variants_keep_radius_frozen():
    torch.manual_seed(0)
    s, xyz, r, proj, thetas, times, pj = _toy(q_true=300.0)
    for v in ("frozen", "pose", "pose_arr"):
        m, info = tr.fit_case(proj, thetas, times, s, xyz, r + 0.1, pj, iters_dir=20, iters_a=40, iters_b=20,
                              q_init=330.0, log=lambda *a: None, variant=v)
        assert np.allclose(m.tube.radius().detach().numpy(), r + 0.1, atol=1e-4), v
        assert info["variant"] == v
