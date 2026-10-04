"""Step 2.2: plug-flow transport layer.

Runs on: M4 (CPU torch, tests and prototype) and molab GPU (same code).

Each tube Gaussian i sits at arc length s_i. Its contrast concentration is
the inlet curve delayed by the plug-flow transit time (plan, Approach):

    c_i(t) = C_in(t - tau_i),   tau_i = int_0^{s_i} A(s) / Q ds,   A = 2 pi sigma^2

Unknowns here: one flow Q per vessel and the gamma-variate inlet C_in
(arrival t0, shape alpha, scale beta, amplitude k = mu_I x peak
concentration). Units: mm, s, Q in mm^3/s (1 mL/s = 1000 mm^3/s).
"""

from __future__ import annotations

import math

import torch


def gamma_variate(t, t0, alpha, beta):
    """Gamma-variate with peak 1 at t0 + alpha beta; 0 before t0.

    Same form as bolus_sim.gamma_variate (data generation), differentiable.
    Ref: plan, Approach ("gamma-variate inlet curve C_in(t)"); classic
    indicator-dilution form (Thompson 1964 / Madsen 1992), UNVERIFIED.
    """
    x = (t - t0) / (alpha * beta)
    pos = x > 0
    xs = torch.where(pos, x, torch.ones_like(x))
    val = torch.exp(alpha * torch.log(xs) + alpha * (1.0 - xs))
    return torch.where(pos, val, torch.zeros_like(val))


class Transport(torch.nn.Module):
    """Q, inlet curve and the plug-flow concentration of every Gaussian.

    Parameters are unconstrained (log for positive quantities):
    log_q (mm^3/s), t0 (s), log_alpha, log_beta (s), log_k (1/mm).

    Ref: plan, Approach (c_i(t) = C_in(t - tau_i), tau_i = int A / Q ds,
    A = 2 pi sigma^2) and Step 2.2 ("parameters Q and gamma-variate C_in;
    compute tau(s) and time-dependent concentration per Gaussian").
    Discretisation (trapezoid between neighbouring Gaussian centres) OURS.
    """

    def __init__(self, q_mm3s, t0=0.5, alpha=3.0, beta=0.35, k=0.1):
        super().__init__()
        self.log_q = torch.nn.Parameter(torch.tensor(math.log(q_mm3s)))
        self.t0 = torch.nn.Parameter(torch.tensor(float(t0)))
        self.log_alpha = torch.nn.Parameter(torch.tensor(math.log(alpha)))
        self.log_beta = torch.nn.Parameter(torch.tensor(math.log(beta)))
        self.log_k = torch.nn.Parameter(torch.tensor(math.log(k)))

    def q(self):
        return self.log_q.exp()

    def inlet(self, t):
        return gamma_variate(t, self.t0, self.log_alpha.exp(), self.log_beta.exp())

    @staticmethod
    def segment_lengths(positions):
        """|p_{i+1} - p_i| for a chain of centres (N-1,)."""
        return (positions[1:] - positions[:-1]).norm(dim=1)

    def tau(self, positions, area):
        """Plug-flow arrival time of every Gaussian, tau_0 = 0 at the inlet.

        tau_i = sum_{k<i} 0.5 (A_k + A_{k+1}) |p_{k+1} - p_k| / Q.
        """
        ds = self.segment_lengths(positions)
        vol = 0.5 * (area[1:] + area[:-1]) * ds
        return torch.cat([vol.new_zeros(1), torch.cumsum(vol, 0)]) / self.q()

    def concentration(self, positions, area, t):
        """c_i(t_j) = C_in(t_j - tau_i), shape (T, N)."""
        tau = self.tau(positions, area)
        return self.inlet(t[:, None] - tau[None, :])

    def density(self, positions, area, sigma_axial, t):
        """Peak density of each Gaussian at each time, shape (T, N), in 1/mm.

        rho_i(t) = k c_i(t) ds_i / (sqrt(2 pi) sigma_a,i): with this factor a
        chain of Gaussians with spacing ds_i and axial scale sigma_a,i has a
        linear attenuation of k c A along the vessel, i.e. a lumen of area
        A = 2 pi sigma_r^2 filled with attenuation k c (plan's mass matching
        mu_0 2 pi sigma^2 = mu_0 pi R^2). Ref: OURS (normalisation).
        """
        ds = self.segment_lengths(positions)
        ds_node = torch.cat([ds[:1], 0.5 * (ds[1:] + ds[:-1]), ds[-1:]])
        c = self.concentration(positions, area, t)
        return self.log_k.exp() * c * (ds_node / (math.sqrt(2 * math.pi) * sigma_axial))[None, :]
