"""Planar (L1) geometry chain: Omega_x => theta => T-vector => r => closure/area.

Valid only when ``Omega_y == 0``, where the control Hamiltonian commutes with
itself at all times and no propagator is needed: the gate angle is an integral,
the space curve is a pure quadrature. This is the main workhorse layer -- under
pure Z noise the optimum is planar (``_plan.md`` §7.3).

Numerical contract
------------------
* Quadrature is **midpoint**, ``O(dt^2)``. First-order right-endpoint rules
  create *grid artifacts*: the optimizer exploits the discretization error and
  reports a 1e-9 residual where the true residual is 1.28e-2 (``_plan.md``
  §6.2). Refining the grid does not rescue a low-order rule.
* First-order Z cancellation <=> ``r(T) = 0`` (closure).
* Second-order <=> ``int_0^T r x r_dot dt = 0`` (projected area).
* The area criterion's normalization, sign and tolerance are pinned down in
  Step 04 -- in the function docstring, with a test -- because cct's registry
  and this project may normalize differently (``_plan.md`` §7.2).

Planned API (Step 04)
---------------------
``tangent(a, T, N)``, ``curve(a, T, N)``, ``closure(a, T, N)``,
``area(a, T, N)``, all returning scale-invariant forms (closure/L, area/L^2).
"""

from __future__ import annotations

__all__: list[str] = []
