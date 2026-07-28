"""Individual cost terms, scale-invariant, plus the naive normalization.

Each cost is its own function. The total objective is only a combinator over
them. That is what makes any weight combination decomposable after the fact:
:mod:`curve_opt.recorder` stores every term separately, and
:mod:`curve_opt.plotting` re-derives them from the stored coefficients
(``_plan.md`` §4.3 item 4, §5.1 ``history.npz``).

Scale-invariant forms only (``_plan.md`` §2.4): ``int |Omega| dt``,
``T * Omega_max``, ``closure / L``, ``area / L^2``, ``(int Omega^2 dt) * L``.

Reference point is the naive single mode ``Omega_x = (pi^2 / 2T) sin(pi t / T)``:
ansatz-independent, analytic (``energy * L = pi^4 / 8 = 12.18``,
``int |Omega| dt = pi``), and therefore fair by construction. Each winding
branch gets its own naive reference at its own theta (``_plan.md`` §8.3).

Smoothness contract
-------------------
* ``peak = max |Omega|`` is **not** differentiable in the coefficients (the
  argmax switches, and the absolute value kinks). It is reported here, but it
  enters the objective only through the epigraph form built in
  :mod:`curve_opt.optimize` (``_plan.md`` §3).
* ``int |Omega| dt`` kinks wherever the signed Omega_x crosses zero, which is
  the normal case. Report-only quantity; a smoothed variant
  ``int sqrt(Omega^2 + delta^2) dt`` is available if it ever must be optimized.
* Robustness floor: clamp ``C_hat < 1e-3`` so a machine-zero does not spuriously
  beat an already-robust solution (``_plan.md`` §2.4).

Planned API (Step 05)
---------------------
``energy_invariant``, ``total_curvature``, ``peak``, ``closure_norm``,
``area_norm``, ``normalize_to_naive``.
"""

from __future__ import annotations

__all__: list[str] = []
