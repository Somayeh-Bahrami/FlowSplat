"""Runs on: M4 (CPU only). Tests for Step 5.1 (tree.py)."""

import math

import numpy as np
import pytest
import torch

from flowsplat import model as md
from flowsplat import tree


def _y_branches(r=(1.6, 1.4, 1.0)):
    """Parent along y, LAD daughter continuing, diagonal at 40 deg (lumen mm)."""
    s0 = np.arange(0, 30.0001, 0.5)
    p0 = np.stack([np.zeros_like(s0), s0, np.zeros_like(s0)], 1)
    s1 = np.arange(0, 40.0001, 0.5)
    p1 = p0[-1] + np.stack([np.zeros_like(s1), s1, np.zeros_like(s1)], 1)
    s2 = np.arange(0, 30.0001, 0.5)
    d = np.array([math.sin(math.radians(40)), math.cos(math.radians(40)), 0.0])
    p2 = p0[-1] + s2[:, None] * d
    return [(s0, p0, np.full_like(s0, r[0])), (s1, p1, np.full_like(s1, r[1])), (s2, p2, np.full_like(s2, r[2]))]


def test_side_branches_finds_attach_and_length():
    main = np.stack([np.zeros(101, int), np.arange(101), np.zeros(101, int)], 1)  # 100 voxels along y
    side = np.stack([np.arange(1, 41), 50 + np.arange(1, 41), np.zeros(40, int)], 1)  # diagonal from (0, 50)
    coords = np.concatenate([main, side])
    brs, s_path = tree.side_branches(coords, np.ones(len(coords)), 0.5, np.arange(101))
    assert len(brs) == 1
    assert brs[0]["s_attach_mm"] == pytest.approx(25.0, abs=1.0)
    assert brs[0]["length_mm"] == pytest.approx(40 * math.sqrt(2) * 0.5, rel=0.05)


def test_murray_split_and_tau_continuity():
    f1, f2 = tree.murray_split(1.4, 1.0)
    assert f1 + f2 == pytest.approx(1.0)
    assert f1 / f2 == pytest.approx(1.4**3)
    br = _y_branches()
    q = [2.0e-6, f1 * 2.0e-6, f2 * 2.0e-6]
    taus = tree.tau_branches([b[0] for b in br], [b[2] for b in br], q)
    assert taus[1][0] == pytest.approx(taus[0][-1]) and taus[2][0] == pytest.approx(taus[0][-1])
    assert taus[0][-1] == pytest.approx(30e-3 * math.pi * 1.6e-3**2 / 2.0e-6, rel=1e-6)


def test_tree_transport_matches_ground_truth_tau():
    br = _y_branches()
    tt = tree.TreeTransport(2000.0, f1=0.7, continuity=True)
    qp, q1, q2 = (float(x) for x in tt.q())
    assert qp == pytest.approx(q1 + q2)
    s = torch.tensor(np.concatenate([b[0] for b in br]), dtype=torch.float32)
    area = torch.tensor(np.concatenate([np.pi * b[2] ** 2 for b in br]), dtype=torch.float32)
    branch = torch.tensor(np.concatenate([np.full(len(b[0]), k) for k, b in enumerate(br)]))
    tau = tt.tau(area, s, branch).detach().numpy()
    gt = tree.tau_branches([b[0] for b in br], [b[2] for b in br], [qp * 1e-9, q1 * 1e-9, q2 * 1e-9])
    assert np.allclose(tau, np.concatenate(gt), rtol=1e-4)
    free = tree.TreeTransport(2000.0, f1=0.7, continuity=False)
    assert float(free.q()[0]) == pytest.approx(2000.0)


def test_split_metrics_and_matching():
    m = tree.split_metrics([100.0, 66.0, 33.0], [100.0, 60.0, 40.0], [0, 1])
    assert m["frac_est"] == pytest.approx([0.66, 0.33])
    assert m["split_err"] == pytest.approx(np.mean([0.1, 0.175]))
    br = _y_branches()
    perm, d = tree.match_daughters([br[2][1], br[1][1]], [br[1][1], br[2][1]])
    assert perm == [1, 0] and d == pytest.approx(0.0)


def test_tree_lumen_union_labels_branches():
    br = _y_branches()
    lum = tree.tree_lumen_voxels(br)
    assert set(np.unique(lum["branch"]).tolist()) == {0, 1, 2}
    vol_mm3 = float(lum["frac"].sum()) * lum["vox_mm"] ** 3
    expect = sum(math.pi * b[2][0] ** 2 * b[0][-1] for b in br)
    assert vol_mm3 == pytest.approx(expect, rel=0.08)  # overlap at the junction and caps


def test_tree_bolus_daughters_follow_parent():
    br = _y_branches()
    rng = np.random.default_rng(0)
    q = [2.0e-6, 1.3e-6, 0.7e-6]
    cs, b = tree.simulate_tree_bolus(br, q, rng)
    t = np.linspace(0, 4.4, 133)
    arrive = [t[np.argmax(c[-1] > 0.5 * c[-1].max())] for c in cs]
    assert arrive[1] > arrive[0] and arrive[2] > arrive[0]
    assert cs[0].shape == (len(br[0][0]), 133)


def test_toy_fit_recovers_split():
    """Render a Y with known flows through the cone projector, fit with
    continuity; the split error must be small and the inlet found."""
    torch.manual_seed(0)
    br = _y_branches()
    gd = {"DSO": 750.0, "DSD": 1200.0, "nDetector": [48, 64], "dDetector": [1.5, 1.5]}
    pj = md.ConeProjector(gd)
    world = [(s, x - np.array([0.0, 35.0, 0.0]), r) for s, x, r in br]
    truth = tree.TreeModel(world, continuity=True, f1=0.68, q_p=2500.0)
    with torch.no_grad():
        truth.transport.inlet.t0.fill_(0.6)
        truth.transport.inlet.log_k.fill_(math.log(0.1))
    idx = list(range(0, 133, 5))
    thetas = list(np.deg2rad(np.linspace(0, 198, 133))[idx])
    times = torch.tensor(np.linspace(0, 4.4, 133)[idx], dtype=torch.float32)
    with torch.no_grad():
        cache = pj.footprints(truth.tube.position, truth.tube.covariance(), thetas)
        proj = torch.einsum("vn,vnp->vp", truth.density(times), cache).reshape(len(idx), 48, 64)
    s_r = [b[2] for b in world]
    hyps = tree.orientations([(world[0][0], world[0][1]), (world[1][0], world[1][1]), (world[2][0], world[2][1])], s_r)
    m, info = tree.fit_tree(proj, thetas, times, hyps, pj, variant="cont", iters_dir=60, iters=400, log=lambda *a: None)
    assert info["direction"] == 0
    met = tree.split_metrics(info["q_mm3s"], [2500.0, 0.68 * 2500.0, 0.32 * 2500.0], [0, 1])
    assert met["split_err"] < 0.05
    assert met["q_parent_rel_err"] < 0.05
