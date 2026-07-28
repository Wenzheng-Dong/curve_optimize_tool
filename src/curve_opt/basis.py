"""Sine basis: coefficients <-> waveforms, the gate row, analytic bounds.

Home of every closed-form expression in the project. Depends on numpy only
(no JAX, no scipy) so that the analytic formulas stay independently checkable
against the differentiable implementations.

Conventions
-----------
* Basis functions ``sin(n pi t / T)``, ``n = 1..M``; endpoints vanish
  identically for any coefficient vector -- an identity, never a penalty.
* Even harmonics contribute exactly zero to the gate angle; only odd harmonics
  move the gate. This split is the skeleton of the optimization structure.
* Gate row (planar): ``g_n = (T / (n pi)) * (1 - (-1)**n)``, so that the gate
  condition reads ``g . a = theta`` -- linear in the coefficients.
* Analytic bound: ``(int Omega^2 dt) * L >= theta_gate**2``, evaluated *per
  winding branch* (theta and theta + 2 pi are the same gate but different
  constraint surfaces; see ``_plan.md`` §8.3).

Planned API (Step 03)
---------------------
``omega(coeffs, t, T)``, ``gate_row(M, T)``, ``gate_angle(a, T)``,
``energy(a, T)``, ``energy_bound(theta)``, ``naive_single_mode(theta, T)``.
"""

from __future__ import annotations

__all__: list[str] = []
