"""Runs on: M4 (CPU only). Tests for src/flowsplat/ffr.py against the plan's worked example."""

import numpy as np
import pytest

from flowsplat import ffr


def _example(d_s_mm=1.5, v_scale=1.0):
    a_n = np.pi * (1.5e-3) ** 2  # 3.0 mm normal diameter
    a_s = np.pi * (d_s_mm / 2 * 1e-3) ** 2
    q = 3e-6 * v_scale  # 3 mL/s
    return ffr.young_tsai_dp(q, a_n, a_s, 5e-3)


def test_worked_example_terms():
    total, visc, expn = _example()
    assert visc == pytest.approx(422, abs=1)
    assert expn == pytest.approx(1305, abs=2)
    assert total / ffr.MMHG_PA == pytest.approx(12.96, abs=0.02)
    assert ffr.ffr_from_dp(total) == pytest.approx(0.856, abs=0.001)


def test_worked_example_sensitivities():
    # 0.1 mm diameter error -> FFR 0.798 (crosses 0.80)
    total, _, _ = _example(d_s_mm=1.4)
    assert total / ffr.MMHG_PA == pytest.approx(18.2, abs=0.1)
    assert ffr.ffr_from_dp(total) == pytest.approx(0.798, abs=0.001)
    # 30 deg foreshortening: V measured as 0.866 V -> FFR 0.888
    total, _, _ = _example(v_scale=0.866)
    assert total / ffr.MMHG_PA == pytest.approx(10.1, abs=0.1)
    assert ffr.ffr_from_dp(total) == pytest.approx(0.888, abs=0.001)


def test_no_stenosis_no_expansion_loss():
    a = np.pi * (1.5e-3) ** 2
    _, _, expn = ffr.young_tsai_dp(3e-6, a, a, 5e-3)
    assert expn == pytest.approx(0.0)
