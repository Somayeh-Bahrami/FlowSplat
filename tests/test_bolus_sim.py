"""Runs on: M4 (CPU only). Tests for src/flowsplat/bolus_sim.py (Step 1.2) against physics."""

import numpy as np
import pytest

from flowsplat import bolus_sim as bs
from flowsplat import data_gen as dg


def _arrival(t, c, level=0.5):
    """First time c crosses `level` (linear interpolation)."""
    k = int(np.argmax(c >= level))
    if k == 0:
        return t[0]
    return t[k - 1] + (level - c[k - 1]) * (t[k] - t[k - 1]) / (c[k] - c[k - 1])


def test_gamma_variate_peak():
    t = np.linspace(0, 5, 50001)
    c = bs.gamma_variate(t, 0.5, 3.0, 0.4)
    assert c.max() == pytest.approx(1.0, abs=1e-6)
    assert t[np.argmax(c)] == pytest.approx(0.5 + 3.0 * 0.4, abs=1e-3)
    assert np.all(c[t <= 0.5] == 0)


def test_step_front_arrives_at_plug_flow_time_through_stenosis():
    s = np.arange(0, 60.001, 0.5)
    r = dg.insert_stenosis(s, np.full_like(s, 1.5), 0.6, 10.0, 25.0)
    q = 0.15 * np.pi * (1.5e-3) ** 2  # 15 cm/s in the normal segment
    t = np.linspace(0, 0.6, 601)
    c = bs.simulate(s, r, q, 1e-9, lambda tt: 1.0 if tt > 0 else 0.0, t)
    area = np.pi * (r * 1e-3) ** 2
    tau = np.concatenate([[0], np.cumsum(0.5 * (area[1:] + area[:-1]) * np.diff(s) * 1e-3)]) / q
    for k in (40, 60, 80, 110):  # 20, 30 (in the lesion), 40, 55 mm
        assert _arrival(t, c[k]) == pytest.approx(tau[k], abs=0.01)


def test_numerical_diffusion_small():
    # D = 0: a sharp front after 50 mm should stay sharp (10-90 % width in space)
    s = np.arange(0, 80.001, 0.5)
    r = np.full_like(s, 1.5)
    q = 0.15 * np.pi * (1.5e-3) ** 2
    t = np.array([0.0, 50e-3 / 0.15])
    c = bs.simulate(s, r, q, 0.0, lambda tt: 1.0, t)[:, -1]
    width = s[np.argmax(c < 0.1)] - s[np.argmax(c < 0.9)]
    assert width <= 1.0  # mm at 0.5 mm stations; van Leer at 0.2 mm gave 2 mm, first-order upwind about 7 mm


def test_dispersion_matches_2Dt():
    # long pulse front: erf profile, 10-90 % width = 2 * 1.2816 * sqrt(2 D t)
    s = np.arange(0, 100.001, 0.5)
    r = np.full_like(s, 1.5)
    u = 0.05
    q = u * np.pi * (1.5e-3) ** 2
    d = 2e-6
    t_end = 50e-3 / u  # front at 50 mm, 1 s
    c = bs.simulate(s, r, q, d, lambda tt: 1.0, np.array([0.0, t_end]))[:, -1]
    width_mm = s[np.argmax(c < 0.1)] - s[np.argmax(c < 0.9)]
    expected_mm = 2 * 1.2816 * np.sqrt(2 * d * t_end) * 1e3
    assert width_mm == pytest.approx(expected_mm, rel=0.2)


def test_mass_conservation():
    s = np.arange(0, 60.001, 0.25)
    r = dg.insert_stenosis(s, np.linspace(1.8, 1.2, len(s)), 0.5, 8.0, 20.0)
    q = 1.0e-6
    t = np.linspace(0, 0.5, 51)  # front still inside the vessel at 0.5 s
    inlet = lambda tt: bs.gamma_variate(tt, 0.05, 3.0, 0.05)  # noqa: E731
    c = bs.simulate(s, r, q, 1e-6, inlet, t)[:, -1]
    area = np.pi * (r * 1e-3) ** 2
    stored = np.trapezoid(area * c, s * 1e-3)
    tt = np.linspace(0, 0.5, 50001)
    injected = q * np.trapezoid(inlet(tt), tt)
    assert stored == pytest.approx(injected, rel=0.02)


def test_frames_and_splits():
    assert len(bs.frame_times()) == 133 and bs.frame_times()[-1] == pytest.approx(4.4)
    assert bs.frame_angles_deg()[-1] == pytest.approx(198.0)
    ids = [str(i) for i in range(100, 140)]
    sp = bs.make_splits(ids, seed=0)
    assert [len(sp[k]) for k in ("train", "val", "test")] == [28, 6, 6]
    assert set(sp["train"]) | set(sp["val"]) | set(sp["test"]) == set(ids)
    assert not (set(sp["train"]) & set(sp["test"]))
    assert bs.make_splits(ids, seed=0) == sp
