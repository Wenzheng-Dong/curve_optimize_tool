"""General (L2) layer: JAX SU(2) propagator and the 3D constraints.

Used when ``Omega_y != 0``, i.e. for 3D ansatz inputs and for the future
curve-family members proposed by the user. Same public constraint API as the
planar layer (``gate`` / ``closure`` / ``area``) so the optimizer never has to
know which layer it is on; in the planar limit the two implementations must
agree to <= 1e-12 (regression test, Step 11).

Numerical contract -- the multiplication order is a *physics* bug if wrong
-------------------------------------------------------------------------
``U(t_k) = U_k U_{k-1} ... U_1``, newest on the left::

    Uc = jax.lax.associative_scan(lambda A, B: B @ A, Us)   # correct
    Uc = jax.lax.associative_scan(jnp.matmul, Us)           # reversed order

The wrong version is invisible on planar test cases (everything commutes) and
was caught only on a non-commuting case, where it deviated by 3.7e-2 from an
explicit numpy product and from qutip ``sesolve`` (``_plan.md`` §6.3).
Therefore: **any new propagator code must pass a non-commuting case plus an
independent second implementation (numpy / qutip) before use.** A planar case
does not count as that gate.

Time stepping uses the same midpoint rule as :mod:`curve_opt.geometry`.

Planned API (Step 11-12)
------------------------
``propagator(a, b, T, N)``, ``gate_su2(...)``, ``closure3d(...)``,
``area3d(...)``, plus the Gauss-Newton constraint-Hessian hooks.
"""

from __future__ import annotations

__all__: list[str] = []
