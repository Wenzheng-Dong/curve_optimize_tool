"""Solver wrapper: trust-constr with the mandatory configuration of §3.1.

Does no I/O. It hands every iterate's coefficients to :mod:`curve_opt.recorder`
through a callback; the recorder owns the filesystem.

Problem form (epigraph, ``_plan.md`` §3)
---------------------------------------
    min over (a, b, s)   (int Omega^2 dt) * L  +  lambda * T * s
    s.t.  gate = target,  closure = 0,  area = 0,
          -s <= Omega(t_k) <= s   for all grid points k

The epigraph variable ``s`` keeps the objective convex quadratic (its term is
linear) so the analytic Hessian still applies; in the planar layer ``Omega(t_k)``
is *linear* in the coefficients, so the peak constraints are linear
inequalities, which trust-constr supports natively.

Mandatory configuration -- each item was worth 6x..25x in practice
-----------------------------------------------------------------
* float64 process-wide (enabled in :mod:`curve_opt.__init__`).
* ``scipy.optimize.minimize(method='trust-constr')``.
* Gate (planar): eliminated by affine projection ``a = a0 + P z``, or given as a
  ``LinearConstraint``. Never a large-weight soft penalty -- no finite weight
  holds the gate (LESSONS §4).
* Constraint Jacobian: ``jax.jacrev`` over ``associative_scan`` (fastest of the
  variants measured; the others ran at 0.58..0.76x).
* Objective Hessian: the analytic constant ``T^2 I``.
* Constraint Hessian: **must be supplied together with it**; zeroing it gives
  Gauss-Newton. Supplying only the objective Hessian buys nothing; both
  together gave 5.9..25.3x. Keep a ``default`` switch for the control
  comparison -- zeroing is an approximation, not an identity, and has to be
  re-verified on every new gate or constraint set.
* ``maxiter``: >= 1500 planar, >= 6000 in 3D. Whether the run hit the cap must
  be checked and recorded; a run that hit it cannot enter a quantitative claim.
* Early stop: a callback stops after K consecutive relative improvements below
  tol and sets ``stop_reason='early_stop'``.

Two kinds of run
----------------
* **claim-run** -- feeds a number into a results table or an external figure:
  requires ``stop_reason='converged'`` plus a fine-grid re-evaluation (4..25x).
* **survey-run** -- ansatz / weight / gate comparison sweeps: early stop is
  allowed, but ``stop_reason`` must be recorded and annotated on the figure, and
  comparisons are only fair under the *same budget rule*.

Planned API (Step 08)
---------------------
``solve(problem, recorder=None, gn_hessian=True, early_stop=...) -> RunResult``.
"""

from __future__ import annotations

__all__: list[str] = []
