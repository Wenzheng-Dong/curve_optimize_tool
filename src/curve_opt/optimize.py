"""Solver wrapper: trust-constr with the mandatory configuration of §3.1.

Does no I/O. It hands every iterate's coefficients to :mod:`curve_opt.recorder`
through a callback; the recorder owns the filesystem. A caller that wants a run
on disk constructs the :class:`curve_opt.recorder.Recorder` itself and passes it
in -- this module never builds a path.

Problem form (epigraph, ``_plan.md`` §3)
---------------------------------------
    min over (a, b, s)   (int Omega^2 dt) * L  +  lambda * T * s
    s.t.  gate = target,  closure = 0,  area = 0,
          -s <= Omega(t_k) <= s   for all grid points k

The epigraph variable ``s`` keeps the objective convex quadratic (its term is
linear) so the analytic Hessian still applies; in the planar layer ``Omega(t_k)``
is *linear* in the coefficients, so the peak constraints are linear
inequalities, which trust-constr supports natively. ``max|Omega|`` is never
handed to ``minimize`` directly: it is not differentiable in the coefficients,
and ``lambda = 0`` skips the whole epigraph block rather than adding a
zero-weight non-smooth term.

:attr:`Problem.extra_constraints` optionally adds the ``_plan.md`` §2.3 row-7
constraints -- a bandwidth cap ``(int Omega_dot^2 dt) T^3 <= B`` and the C1
switch-on/off equalities ``Omega_dot(0) = 0`` / ``Omega_dot(T) = 0``. Both are
analytic in the coefficients (a diagonal quadratic form and a free linear row),
so both come with exact Jacobians *and* exact Hessians. Default empty: a
``Problem`` built without them reproduces Steps 08/12 exactly.

Gate elimination
----------------
Two routes, both exact enough and both available (``_plan.md`` §3.1):

* ``gate="projection"`` (default) -- affine elimination ``a = a0 + P z`` with the
  columns of ``P`` an *orthonormal* basis of ``ker g``. The gate then holds
  identically, to floating point, and because ``P^T P = I`` the objective Hessian
  in the reduced variables is still exactly ``T^2 I``, so the analytic Hessian of
  §3.1 carries over unchanged.
* ``gate="linear_constraint"`` -- ``g . a = theta`` as a ``LinearConstraint``.
  This is what step00d used; it leaves a residual around 1e-11.

Mandatory configuration -- each item was worth 6x..25x in practice
-----------------------------------------------------------------
* float64 process-wide (enabled in :mod:`curve_opt.__init__`).
* ``scipy.optimize.minimize(method='trust-constr')``.
* Never a large-weight soft penalty for the gate -- no finite weight holds it
  (LESSONS §4); it is eliminated or a hard linear equality.
* Constraint Jacobian: ``jax.jacrev`` over the ``associative_scan``-style
  cumulative chain of :mod:`curve_opt.geometry` (fastest of the variants
  measured; the others ran at 0.58..0.76x), jitted.
* Objective Hessian: the analytic constant ``T^2 I``.
* Constraint Hessian: **must be supplied together with it**; zeroing it gives
  Gauss-Newton. :data:`HESSIAN_MODES` keeps all three configurations so the claim
  can be re-measured on every new gate or constraint set:
  ``gauss_newton`` (analytic objective + zeroed constraint),
  ``objective_only`` (analytic objective, constraint left to scipy -- the control
  for "supplying only the objective Hessian buys nothing"), and
  ``default`` (both left to scipy).
* ``maxiter``: >= 1500 planar, >= 6000 in 3D. Whether the run hit the cap is
  checked and recorded; a run that hit it is marked ``maxiter`` and is void for
  quantitative claims.
* Early stop: :class:`EarlyStop` stops after ``patience`` consecutive relative
  improvements below ``tol`` and sets ``stop_reason='early_stop'``.

★ The optimization grid sets the fine-grid residual
---------------------------------------------------
A solution drives the residuals to zero *on the grid it was optimized on*; on a
finer grid what remains is that grid's ``O(dt^2)`` error. Measured: optimizing at
``N = 20000`` leaves 8.4e-9 at ``N = 500000`` (step00d), while optimizing at
``N = 4000`` leaves 2.0e-7. So the acceptance target "fine-grid residual <= 1e-8"
is a statement about ``N_grid``, not only about convergence:
:data:`N_CLAIM` = 20000 is the claim-run default, and :data:`N_SURVEY` = 4000 is
for comparison sweeps where residual level is not the deliverable.

Two kinds of run
----------------
* **claim-run** -- feeds a number into a results table or an external figure:
  requires ``stop_reason='converged'`` plus a fine-grid re-evaluation
  (:func:`reevaluate`, 4..25x).
* **survey-run** -- ansatz / weight / gate comparison sweeps: early stop is
  allowed, but ``stop_reason`` must be recorded and annotated on the figure, and
  comparisons are only fair under the *same budget rule*.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import BFGS, LinearConstraint, NonlinearConstraint, minimize
from scipy.sparse import csr_matrix

from curve_opt import basis, geometry, metrics, propagate

__all__ = [
    "EarlyStop",
    "HESSIAN_MODES",
    "MAXITER_FLOOR",
    "N_CLAIM",
    "N_CLAIM_3D",
    "N_SURVEY",
    "Problem",
    "SolveResult",
    "extra_constraint_residuals",
    "manifest_for",
    "reevaluate",
    "snap_winding_branch",
    "solve",
]

#: Optimization grid for a claim-run: what step00d used, and what makes the
#: fine-grid residual land at 8e-9 rather than 2e-7. See the module docstring.
N_CLAIM = 20_000

#: Optimization grid for survey runs, where the deliverable is the trajectory.
N_SURVEY = 4_000

#: Optimization grid for the general layer. step00e used 4000, and the constraint
#: Jacobian there costs 37..180 ms against 0.07 ms for the objective, so the grid
#: cannot simply be raised to N_CLAIM: at N = 20000 one 3D solve would run for
#: tens of minutes. Fine-grid re-evaluation still happens at 1e5 / 5e5.
N_CLAIM_3D = 4_000

#: §3.1: >= 1500 planar, >= 6000 in the general layer.
MAXITER_FLOOR = {"planar": 1500, "general": 6000}

HESSIAN_MODES = ("gauss_newton", "objective_only", "default")

#: Grids for the mandatory re-evaluation of a claim (``_plan.md`` §6.4 item 2).
FINE_GRIDS = (100_000, 500_000)


def snap_winding_branch(
    theta_measured: float,
    theta_target: float,
    *,
    max_distance: float = 0.5,
) -> tuple[float, int]:
    """Snap a measured total turning to the exact branch value ``theta_target + 2 pi k``.

    ★ ``_plan.md`` §5.1 (v4.2): the right-hand side of the gate constraint must be
    the *exact* branch value. The optimizer exists partly to correct the ansatz's
    gate error, so baking a measured 2.2e-4 error into the constraint surface
    would define the error as the target. Identifying ``k`` needs only ~1e-3
    accuracy, so a quadrature estimate of the ansatz's total turning is plenty;
    that estimate is kept separately as ``theta_before``.

    Raises :class:`ValueError` when the measured turning is not near any branch of
    *theta_target*: such an ansatz does not implement this gate on any branch (an
    identity-gate 3D family against an X(pi) target, say), and silently snapping
    it would hide that the two are unrelated. Pass a larger *max_distance* only
    with a reason.
    """
    two_pi = 2.0 * np.pi
    k = int(np.round((float(theta_measured) - float(theta_target)) / two_pi))
    exact = float(theta_target) + two_pi * k
    distance = abs(float(theta_measured) - exact)
    if distance > max_distance:
        raise ValueError(
            f"measured total turning {theta_measured:.6f} is {distance:.4f} rad from the "
            f"nearest branch of theta_target={theta_target:.6f} (k={k} -> {exact:.6f}). "
            "This ansatz does not implement the target gate on any winding branch; "
            "choose a target matching it, or pass max_distance explicitly."
        )
    return exact, k


class Problem(NamedTuple):
    """One optimization problem, fully specified. Pure data, no solver state."""

    M: int
    T: float
    theta: float
    """Exact winding-branch value: the right-hand side of the gate constraint."""
    coeffs0: np.ndarray
    """Starting coefficients for ``Omega_x``, normally the projected ansatz."""
    coeffs0_b: np.ndarray | None = None
    """Starting coefficients for ``Omega_y``. Present => the general (L2) layer.

    ``None`` selects the planar layer, where ``Omega_y`` is identically zero and no
    propagator is needed. The two layers share this one entry point on purpose
    (``_plan.md`` §4.3 item 3): the solver does not know which one it is on.
    """
    N_grid: int = N_CLAIM
    lam: float = 0.0
    """Weight of ``T * s`` in the objective. ``0`` disables the epigraph block."""
    n_peak: int = 400
    """Number of grid points carrying the epigraph peak inequalities."""
    gate: str = "projection"
    hessian_mode: str = "gauss_newton"
    maxiter: int = 1500
    fixed_budget_survey: bool = False
    """Allow a sub-floor maxiter only for an explicitly labelled survey run."""
    gtol: float = 1e-12
    xtol: float = 1e-14
    checkpoint_every: int = 1
    target_gate: dict | None = None
    ansatz: dict | None = None
    theta_before: float | None = None
    """The ansatz's *measured* total turning, recorded alongside the exact branch."""
    extra_constraints: tuple = ()
    """Declarative specs for the ``_plan.md`` §2.3 row-7 constraints, e.g.

    * ``{"kind": "bandwidth", "bound": B}`` -- ``(int Omega_dot^2 dt) T^3 <= B``,
      one convex quadratic inequality with an exact constant Hessian;
    * ``{"kind": "c1", "ends": ("start", "end")}`` -- ``Omega_dot = 0`` at those
      ends, one *free linear equality* per end (per field component in the
      general layer).

    Declarative on purpose: the specs go into the manifest verbatim, so a record
    stays self-describing (``_plan.md`` §5.1). Empty tuple = the baseline
    constraint set of Steps 08/12, bit-for-bit.
    """

    @property
    def uses_epigraph(self) -> bool:
        return self.lam != 0.0

    @property
    def layer(self) -> str:
        return "planar" if self.coeffs0_b is None else "general"

    @property
    def n_coeffs(self) -> int:
        return self.M if self.coeffs0_b is None else 2 * self.M


class EarlyStop(NamedTuple):
    """Stop after *patience* consecutive relative improvements below *tol*.

    ``_plan.md`` §3.1: survey runs measure differences, not convergence. The
    threshold is a *comparison rule*, so every member of a comparison group has to
    share one, and the resulting ``stop_reason='early_stop'`` must appear on any
    figure drawn from the run.
    """

    tol: float = 1e-8
    patience: int = 5


class SolveResult(NamedTuple):
    """Outcome of one solve. ``stop_reason`` decides whether it may be claimed."""

    coeffs: np.ndarray
    stop_reason: str
    status: int
    nit: int
    wall_clock_s: float
    n_checkpoints: int
    terms: metrics.CostTerms
    gate_residual: float
    equality_residuals: np.ndarray
    epigraph_s: float | None
    fine_grid: dict
    scipy_message: str
    fixed_budget_survey: bool = False

    @property
    def void_for_claims(self) -> bool:
        return self.stop_reason != "converged" or self.fixed_budget_survey


# --------------------------------------------------------------------------
# gate elimination
# --------------------------------------------------------------------------


def _gate_projection(M: int, T: float, theta: float):
    """Return ``(a0, P)`` with ``g . (a0 + P z) = theta`` for every ``z``.

    ``a0`` is the minimum-norm particular solution and the columns of ``P`` are an
    orthonormal basis of ``ker g`` taken from the SVD. Orthonormality is what
    keeps the reduced objective Hessian equal to ``T^2 I`` exactly.
    """
    g = basis.gate_row(M, T)
    a0 = theta * g / (g @ g)
    _, _, vh = np.linalg.svd(g[None, :])
    P = vh[1:].T  # columns span ker g, orthonormal by construction
    return a0, P


# --------------------------------------------------------------------------
# assembling the scipy problem
# --------------------------------------------------------------------------


def _objective(problem: Problem, a0, P):
    """Analytic value / gradient / Hessian in the solver variables.

    ``f = (T^2 / 2) |a0 + P z|^2 + lambda T s``, so the gradient is
    ``T^2 P^T a`` (plus ``lambda T`` on ``s``) and the Hessian is the constant
    ``blockdiag(T^2 P^T P, 0)`` -- ``T^2 I`` in the reduced block. In the general
    layer ``P`` is the identity on 2M variables and ``a`` is the concatenation
    ``[a, b]``, so the same expression covers the fluence of both components
    (``int Omega^2 dt = int (Omega_x^2 + Omega_y^2) dt``, one closed form).
    """
    T, lam = problem.T, problem.lam
    nz = P.shape[1]
    n = nz + (1 if problem.uses_epigraph else 0)

    H = np.zeros((n, n))
    H[:nz, :nz] = T**2 * (P.T @ P)

    def fun(x):
        a = a0 + P @ x[:nz]
        value = 0.5 * T**2 * float(a @ a)
        if problem.uses_epigraph:
            value += lam * T * float(x[nz])
        return value

    def jac(x):
        a = a0 + P @ x[:nz]
        out = np.zeros(n)
        out[:nz] = T**2 * (P.T @ a)
        if problem.uses_epigraph:
            out[nz] = lam * T
        return out

    return fun, jac, (lambda x: H)


def _nonlinear_constraint(problem: Problem, a0, P):
    """The hard equality block, with its Jacobian by ``jax.jacrev``.

    Planar layer: closure (2) + area (1); the gate is eliminated or linear.
    General layer: gate (3) + closure (3) + area (3) = the nine equations of
    ``_plan.md`` §2.3, all nonlinear, evaluated through the SU(2) propagator. The
    gate cannot be eliminated affinely there -- it is not a linear functional of
    the coefficients once ``Omega_y != 0``.
    """
    T, N = problem.T, problem.N_grid
    nz = P.shape[1]
    n = nz + (1 if problem.uses_epigraph else 0)
    a0_j, P_j = jnp.asarray(a0), jnp.asarray(P)

    if problem.layer == "general":
        M = problem.M
        U_target = propagate.target_x(problem.theta)

        def residual(x):
            return propagate.equality_residuals(x[:M], x[M : 2 * M], T, N, U_target)

    else:

        def residual(x):
            a = a0_j + P_j @ x[:nz]
            return geometry.equality_residuals(a, T, N)

    fun = jax.jit(residual)
    jac = jax.jit(jax.jacrev(residual))

    if problem.hessian_mode == "gauss_newton":
        zeros = csr_matrix((n, n))

        def hess(x, v):  # Gauss-Newton: drop the constraint curvature
            return zeros

    else:
        # ``objective_only`` and ``default`` differ only in the *objective* Hessian;
        # both leave the constraint Hessian to scipy's quasi-Newton update. Lumping
        # objective_only in with gauss_newton here (as this function did until
        # Step 12) makes the two modes literally identical, which is how Step 08
        # came to report that the objective Hessian alone carries the speedup.
        hess = BFGS()

    return NonlinearConstraint(
        lambda x: np.asarray(fun(jnp.asarray(x))),
        0.0,
        0.0,
        jac=lambda x: np.asarray(jac(jnp.asarray(x))),
        hess=hess,
    )


def _epigraph_constraints(problem: Problem, a0, P):
    """``-s <= Omega(t_k) <= s`` as two blocks of *linear* inequalities.

    With ``a = a0 + P z`` and ``Omega(t_k) = S[k] a`` the rows read
    ``S P z - s <= -S a0`` and ``-S P z - s <= S a0``.

    The peak grid (``n_peak``) is deliberately coarser than the quadrature grid:
    at ``N_grid = 20000`` this block would carry 40000 inequality rows. Omega is
    band-limited (M harmonics), so a few hundred points resolve its maximum to
    O(dt^2); the shortfall between the constrained grid maximum and the reported
    peak is measured in the smoke test and recorded in the manifest.
    """
    nz = P.shape[1]
    M, T = problem.M, problem.T
    t_peak = np.linspace(0.0, T, problem.n_peak + 2)[1:-1]
    S = basis.design_matrix(t_peak, T, M)

    if problem.layer == "general":
        # ★ With Omega_y != 0 the peak is |Omega| = hypot(Omega_x, Omega_y), so the
        # epigraph is a second-order cone rather than a pair of half-spaces:
        # Omega_x(t_k)^2 + Omega_y(t_k)^2 - s^2 <= 0. Still smooth, still convex,
        # but no longer a LinearConstraint -- section 3's "peak constraints are all
        # linear inequalities" is a statement about the planar layer only.
        S_j = jnp.asarray(S)

        def cone(x):
            om_x = S_j @ x[:M]
            om_y = S_j @ x[M : 2 * M]
            return om_x**2 + om_y**2 - x[2 * M] ** 2

        fun = jax.jit(cone)
        jac = jax.jit(jax.jacrev(cone))
        return NonlinearConstraint(
            lambda x: np.asarray(fun(jnp.asarray(x))),
            -np.inf,
            0.0,
            jac=lambda x: np.asarray(jac(jnp.asarray(x))),
            hess=BFGS(),
        )

    SP, Sa0 = S @ P, S @ a0
    ones = np.ones((problem.n_peak, 1))
    A = np.vstack([np.hstack([SP, -ones]), np.hstack([-SP, -ones])])
    ub = np.concatenate([-Sa0, Sa0])
    return LinearConstraint(A, -np.inf, ub)


EXTRA_CONSTRAINT_KINDS = ("bandwidth", "c1")


def _c1_rows(problem: Problem, ends) -> np.ndarray:
    """C1 conditions as rows in *coefficient* space (length :attr:`Problem.n_coeffs`).

    One row per end in the planar layer; two per end in the general layer, since
    ``Omega_dot(0) = 0`` there means both components switch on smoothly.
    """
    M, T = problem.M, problem.T
    rows = []
    for end in ends:
        r = basis.c1_row(M, T, end)
        if problem.layer == "general":
            rows.append(np.concatenate([r, np.zeros(M)]))
            rows.append(np.concatenate([np.zeros(M), r]))
        else:
            rows.append(r)
    return np.asarray(rows)


def _extra_constraint_objects(problem: Problem, a0, P, n_x: int) -> list:
    """Turn :attr:`Problem.extra_constraints` into scipy constraint objects.

    Everything here is analytic in the coefficients -- a linear row for C1, a
    diagonal quadratic form for the bandwidth -- so both get exact Jacobians and
    exact Hessians regardless of ``hessian_mode``. That flag governs the
    *nonlinear equality* block, which is where §3.1's 5.9..25.3x was measured;
    approximating a constant Hessian that costs nothing would only blur the
    comparison with the baseline runs.
    """
    nz = P.shape[1]
    out = []
    for spec in problem.extra_constraints:
        kind = spec.get("kind")
        if kind not in EXTRA_CONSTRAINT_KINDS:
            raise ValueError(f"unknown extra constraint {kind!r}, expected one of "
                             f"{EXTRA_CONSTRAINT_KINDS}")
        if kind == "c1":
            rows = _c1_rows(problem, spec.get("ends", ("start",)))
            A = np.zeros((rows.shape[0], n_x))
            A[:, :nz] = rows @ P
            rhs = -(rows @ a0)
            out.append(LinearConstraint(A, rhs, rhs))
        else:
            bound = float(spec["bound"])
            if bound <= 0.0:
                raise ValueError(f"bandwidth bound must be positive, got {bound}")
            M_block = problem.M
            # ★ Normalized by the bound: the raw form has values ~1e4 and Jacobian
            # entries ~1e4 against an equality block of order 1, and trust-constr
            # weighs constraint violations against each other unscaled. Dividing by
            # the bound makes the constraint read ``bandwidth / B - 1 <= 0``, an
            # O(1) quantity, without changing the feasible set.
            H_full = basis.bandwidth_hessian(problem.n_coeffs, problem.T, M=M_block) / bound
            H_red = np.zeros((n_x, n_x))
            H_red[:nz, :nz] = P.T @ H_full @ P

            def value(x, _b=bound, _M=M_block):
                a = a0 + P @ np.asarray(x[:nz], dtype=float)
                return np.array([basis.bandwidth_invariant(a, problem.T, M=_M) / _b - 1.0])

            def jacobian(x, _b=bound, _M=M_block):
                a = a0 + P @ np.asarray(x[:nz], dtype=float)
                row = np.zeros((1, n_x))
                row[0, :nz] = P.T @ basis.bandwidth_gradient(a, problem.T, M=_M) / _b
                return row

            def hessian(x, v, _H=H_red):
                return csr_matrix(float(v[0]) * _H)

            out.append(
                NonlinearConstraint(value, -np.inf, 0.0, jac=jacobian, hess=hessian)
            )
    return out


# --------------------------------------------------------------------------
# evaluation helpers
# --------------------------------------------------------------------------


def reevaluate(coeffs, T: float, grids=FINE_GRIDS, coeffs_b=None, theta=None) -> dict:
    """Re-evaluate the residuals on finer grids -- mandatory for a claim.

    ``_plan.md`` §6.4 item 2: a residual claim must survive 4..25x refinement. The
    energy invariant is analytic and grid-free, so it is reported once; closure
    and area are re-integrated per grid. With *coeffs_b* given the general layer's
    nine residuals are re-evaluated through the propagator instead.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    general = coeffs_b is not None
    flat = np.concatenate([coeffs, np.asarray(coeffs_b, dtype=float)]) if general else coeffs
    out = {"energy": basis.energy_invariant(flat, T), "layer": "general" if general else "planar",
           "grids": {}}
    for N in grids:
        if general:
            U_target = propagate.target_x(np.pi if theta is None else theta)
            residuals = np.asarray(
                propagate.equality_residuals(coeffs, coeffs_b, T, N, U_target)
            )
            out["grids"][str(N)] = {
                "closure": propagate.closure_invariant(coeffs, coeffs_b, T, N),
                "area": propagate.area_invariant(coeffs, coeffs_b, T, N),
                "gate": float(np.max(np.abs(residuals[:3]))),
                "max_equality_residual": float(np.max(np.abs(residuals))),
            }
        else:
            out["grids"][str(N)] = {
                "closure": geometry.closure_invariant(coeffs, T, N),
                "area": geometry.area_invariant(coeffs, T, N),
                "max_equality_residual": float(
                    np.max(np.abs(np.asarray(geometry.equality_residuals(coeffs, T, N))))
                ),
            }
    return out


def extra_constraint_residuals(problem: Problem, coeffs, coeffs_b=None) -> dict:
    """Violation of each §2.3 row-7 constraint at a point, as a plain dict.

    Signed: ``c1`` entries are equality residuals (zero when satisfied), the
    ``bandwidth`` entry is ``value - bound`` (``<= 0`` when satisfied). Analytic,
    so it needs no grid and is the right thing to check a converged point with.
    """
    flat = np.asarray(coeffs, dtype=float)
    if coeffs_b is not None:
        flat = np.concatenate([flat, np.asarray(coeffs_b, dtype=float)])
    out = {}
    for i, spec in enumerate(problem.extra_constraints):
        if spec.get("kind") == "c1":
            rows = _c1_rows(problem, spec.get("ends", ("start",)))
            out[f"c1_{i}"] = float(np.max(np.abs(rows @ flat)))
        else:
            value = basis.bandwidth_invariant(flat, problem.T, M=problem.M)
            out[f"bandwidth_{i}"] = float(value - float(spec["bound"]))
            out[f"bandwidth_{i}_value"] = value
    return out


def manifest_for(problem: Problem, *, early_stop: EarlyStop | None = None, **extra) -> dict:
    """Build the manifest dict for *problem*. Pure data -- the recorder writes it."""
    man = {
        "ansatz": dict(problem.ansatz or {}),
        "M": problem.M,
        "T": problem.T,
        "target_gate": dict(problem.target_gate or {"name": "unspecified"}),
        "N_grid": problem.N_grid,
        "layer": problem.layer,
        "winding_branch": float(problem.theta),
        "extra_constraints": [dict(spec) for spec in problem.extra_constraints],
        "theta_before": (None if problem.theta_before is None else float(problem.theta_before)),
        "objective": {
            "terms": ["energy"] + (["peak"] if problem.uses_epigraph else []),
            "lambda": float(problem.lam),
            "epigraph": bool(problem.uses_epigraph),
            "peak_grid_N": (problem.n_peak if problem.uses_epigraph else None),
        },
        "solver": {
            "method": "trust-constr",
            "maxiter": problem.maxiter,
            "fixed_budget_survey": bool(problem.fixed_budget_survey),
            "hessian_mode": problem.hessian_mode,
            "gate_handling": problem.gate,
            "gtol": problem.gtol,
            "xtol": problem.xtol,
            "checkpoint_every": problem.checkpoint_every,
            "early_stop": (None if early_stop is None else early_stop._asdict()),
        },
    }
    man.update(extra)
    return man


# --------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------


def solve(
    problem: Problem,
    *,
    recorder=None,
    early_stop: EarlyStop | None = None,
    fine_grids=FINE_GRIDS,
    verbose: int = 0,
) -> SolveResult:
    """Run one optimization under the §3.1 configuration.

    *recorder* is an already-constructed :class:`curve_opt.recorder.Recorder`;
    when given, every ``checkpoint_every`` iterations its ``append`` is called and
    ``close`` is called at the end with the stop reason and the fine-grid
    re-evaluation. This module performs no I/O of its own.
    """
    if problem.hessian_mode not in HESSIAN_MODES:
        raise ValueError(f"hessian_mode must be one of {HESSIAN_MODES}")
    if problem.gate not in ("projection", "linear_constraint"):
        raise ValueError(f"unknown gate handling {problem.gate!r}")
    floor = MAXITER_FLOOR[problem.layer]
    if problem.maxiter < floor and not problem.fixed_budget_survey:
        raise ValueError(
            f"maxiter={problem.maxiter} is below the {problem.layer} floor of {floor} "
            "(_plan.md §3.1); use fixed_budget_survey=True only for a censored survey"
        )

    T, M, N = problem.T, problem.M, problem.N_grid
    g = basis.gate_row(M, T)
    coeffs0 = np.asarray(problem.coeffs0, dtype=float)

    if problem.layer == "general":
        # No gate elimination: with Omega_y != 0 the gate is three nonlinear
        # equations, so it joins the constraint block instead.
        a0, P = np.zeros(2 * M), np.eye(2 * M)
        z0 = np.concatenate([coeffs0, np.asarray(problem.coeffs0_b, dtype=float)])
    elif problem.gate == "projection":
        a0, P = _gate_projection(M, T, problem.theta)
        # least-squares start in the reduced variables: the component of the
        # ansatz that survives gate elimination. Nothing else is changed.
        z0 = P.T @ (coeffs0 - a0)
    else:
        a0, P = np.zeros(M), np.eye(M)
        z0 = coeffs0.copy()
    nz = P.shape[1]

    def to_coeffs(x):
        return a0 + P @ np.asarray(x[:nz], dtype=float)

    def split(x):
        """Solver variables -> ``(a, b)``; ``b`` is None in the planar layer."""
        full = to_coeffs(x)
        return (full[:M], full[M:]) if problem.layer == "general" else (full, None)

    def peak_of(x):
        a, b = split(x)
        return metrics.peak(a, T, problem.n_peak, b=b)

    x0 = (
        np.concatenate([z0, [peak_of(z0) / T]]) if problem.uses_epigraph else np.asarray(z0)
    )

    fun, jac, hess = _objective(problem, a0, P)
    constraints = [_nonlinear_constraint(problem, a0, P)]
    if problem.layer == "planar" and problem.gate == "linear_constraint":
        constraints.append(LinearConstraint(np.hstack([g, np.zeros(len(x0) - M)])[None, :],
                                            problem.theta, problem.theta))
    if problem.uses_epigraph:
        constraints.append(_epigraph_constraints(problem, a0, P))
    constraints.extend(_extra_constraint_objects(problem, a0, P, len(x0)))

    objective_kwargs = (
        {"jac": jac, "hess": hess}
        if problem.hessian_mode in ("gauss_newton", "objective_only")
        else {"jac": jac, "hess": BFGS()}
    )

    U_target_3d = propagate.target_x(problem.theta) if problem.layer == "general" else None

    # ---- callback: record, then decide whether to stop ------------------
    state = {"n": 0, "checkpoints": 0, "stalled": 0, "prev": None, "early": False}

    def evaluate(x):
        """Cost terms, the equality block and the gate residual at *x*."""
        a, b = split(x)
        terms = metrics.cost_terms(a, T, N, b=b)
        if problem.layer == "general":
            eq = np.asarray(propagate.equality_residuals(a, b, T, N, U_target_3d))
            gate_res = float(np.max(np.abs(eq[:3])))
            return terms, eq[3:], gate_res
        eq = np.asarray(geometry.equality_residuals(a, T, N))
        return terms, eq, abs(float(g @ a) - problem.theta)

    def callback(xk, res=None):
        state["n"] += 1
        a, b = split(xk)
        terms, eq, gate_res = evaluate(xk)
        s = float(xk[nz]) if problem.uses_epigraph else None
        total = terms.energy + (problem.lam * T * s if s is not None else 0.0)

        if recorder is not None and (state["n"] - 1) % problem.checkpoint_every == 0:
            n_closure = 3 if problem.layer == "general" else 2
            recorder.append(
                state["n"] - 1,
                a,
                b,
                cost_total=total,
                cost_energy=terms.energy,
                cost_curv=terms.curv,
                cost_peak=terms.peak,
                res_gate=gate_res,
                res_closure=eq[:n_closure],
                res_area=eq[n_closure:],
            )
            state["checkpoints"] += 1

        if early_stop is not None:
            prev = state["prev"]
            if prev is not None:
                improvement = abs(prev - total) / max(abs(prev), 1e-300)
                state["stalled"] = state["stalled"] + 1 if improvement < early_stop.tol else 0
            state["prev"] = total
            if state["stalled"] >= early_stop.patience:
                state["early"] = True
                return True
        return False

    t_start = time.perf_counter()
    res = minimize(
        fun,
        x0,
        method="trust-constr",
        constraints=constraints,
        callback=callback,
        options={
            "maxiter": problem.maxiter,
            "gtol": problem.gtol,
            "xtol": problem.xtol,
            "verbose": verbose,
        },
        **objective_kwargs,
    )
    wall = time.perf_counter() - t_start

    coeffs = to_coeffs(res.x)
    if state["early"]:
        stop_reason = "early_stop"
    elif res.status in (1, 2):
        stop_reason = "converged"
    elif res.nit >= problem.maxiter or res.status == 0:
        stop_reason = "maxiter"
    else:
        stop_reason = "failed"

    coeffs_a, coeffs_b = split(res.x)
    fine = (
        reevaluate(coeffs_a, T, fine_grids, coeffs_b=coeffs_b, theta=problem.theta)
        if fine_grids
        else {}
    )
    terms, eq, gate_res = evaluate(res.x)
    s_final = float(res.x[nz]) if problem.uses_epigraph else None

    out = SolveResult(
        coeffs=coeffs,
        stop_reason=stop_reason,
        status=int(res.status),
        nit=int(res.nit),
        wall_clock_s=wall,
        n_checkpoints=state["checkpoints"],
        terms=terms,
        gate_residual=gate_res,
        equality_residuals=eq,
        epigraph_s=s_final,
        fine_grid=fine,
        scipy_message=str(res.message),
        fixed_budget_survey=bool(problem.fixed_budget_survey),
    )

    if recorder is not None:
        recorder.close(
            status=stop_reason,
            nit=out.nit,
            wall_clock_s=wall,
            scipy_status=out.status,
            scipy_message=out.scipy_message,
            hessian_mode=problem.hessian_mode,
            gate_handling=problem.gate,
            gate_residual=out.gate_residual,
            equality_residuals=out.equality_residuals,
            epigraph_s=s_final,
            fine_grid=fine,
            terms=out.terms._asdict(),
            void_for_claims=out.void_for_claims,
        )
    return out
