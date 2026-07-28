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

from curve_opt import basis, geometry, metrics

__all__ = [
    "EarlyStop",
    "HESSIAN_MODES",
    "N_CLAIM",
    "N_SURVEY",
    "Problem",
    "SolveResult",
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
    """Starting coefficients, normally the projected ansatz (step-0 output)."""
    N_grid: int = N_CLAIM
    lam: float = 0.0
    """Weight of ``T * s`` in the objective. ``0`` disables the epigraph block."""
    n_peak: int = 400
    """Number of grid points carrying the epigraph peak inequalities."""
    gate: str = "projection"
    hessian_mode: str = "gauss_newton"
    maxiter: int = 1500
    gtol: float = 1e-12
    xtol: float = 1e-14
    checkpoint_every: int = 1
    target_gate: dict | None = None
    ansatz: dict | None = None
    theta_before: float | None = None
    """The ansatz's *measured* total turning, recorded alongside the exact branch."""

    @property
    def uses_epigraph(self) -> bool:
        return self.lam != 0.0


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

    @property
    def void_for_claims(self) -> bool:
        return self.stop_reason != "converged"


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
    ``blockdiag(T^2 P^T P, 0)`` -- ``T^2 I`` in the reduced block.
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
    """closure (2) + area (1) as one hard equality block, Jacobian by ``jax.jacrev``."""
    T, N = problem.T, problem.N_grid
    nz = P.shape[1]
    n = nz + (1 if problem.uses_epigraph else 0)
    a0_j, P_j = jnp.asarray(a0), jnp.asarray(P)

    def residual(x):
        a = a0_j + P_j @ x[:nz]
        return geometry.equality_residuals(a, T, N)

    fun = jax.jit(residual)
    jac = jax.jit(jax.jacrev(residual))

    if problem.hessian_mode == "default":
        hess = BFGS()
    else:
        zeros = csr_matrix((n, n))

        def hess(x, v):  # Gauss-Newton: drop the constraint curvature
            return zeros

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
    t_peak = np.linspace(0.0, problem.T, problem.n_peak + 2)[1:-1]
    S = basis.design_matrix(t_peak, problem.T, problem.M)
    SP, Sa0 = S @ P, S @ a0
    ones = np.ones((problem.n_peak, 1))
    A = np.vstack([np.hstack([SP, -ones]), np.hstack([-SP, -ones])])
    ub = np.concatenate([-Sa0, Sa0])
    return LinearConstraint(A, -np.inf, ub)


# --------------------------------------------------------------------------
# evaluation helpers
# --------------------------------------------------------------------------


def reevaluate(coeffs, T: float, grids=FINE_GRIDS) -> dict:
    """Re-evaluate the residuals on finer grids -- mandatory for a claim.

    ``_plan.md`` §6.4 item 2: a residual claim must survive 4..25x refinement. The
    energy invariant is analytic and grid-free, so it is reported once; closure
    and area are re-integrated per grid.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    out = {"energy": basis.energy_invariant(coeffs, T), "grids": {}}
    for N in grids:
        out["grids"][str(N)] = {
            "closure": geometry.closure_invariant(coeffs, T, N),
            "area": geometry.area_invariant(coeffs, T, N),
            "max_equality_residual": float(
                np.max(np.abs(np.asarray(geometry.equality_residuals(coeffs, T, N))))
            ),
        }
    return out


def manifest_for(problem: Problem, *, early_stop: EarlyStop | None = None, **extra) -> dict:
    """Build the manifest dict for *problem*. Pure data -- the recorder writes it."""
    man = {
        "ansatz": dict(problem.ansatz or {}),
        "M": problem.M,
        "T": problem.T,
        "target_gate": dict(problem.target_gate or {"name": "unspecified"}),
        "N_grid": problem.N_grid,
        "winding_branch": float(problem.theta),
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
    if problem.maxiter < 1500:
        raise ValueError(
            f"maxiter={problem.maxiter} is below the planar floor of 1500 (_plan.md §3.1); "
            "a run that stops on the cap cannot enter a quantitative claim"
        )

    T, M, N = problem.T, problem.M, problem.N_grid
    g = basis.gate_row(M, T)
    coeffs0 = np.asarray(problem.coeffs0, dtype=float)

    if problem.gate == "projection":
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

    x0 = np.concatenate([z0, [metrics.peak(to_coeffs(z0), T, problem.n_peak) / T]]) if (
        problem.uses_epigraph
    ) else z0

    fun, jac, hess = _objective(problem, a0, P)
    constraints = [_nonlinear_constraint(problem, a0, P)]
    if problem.gate == "linear_constraint":
        constraints.append(LinearConstraint(np.hstack([g, np.zeros(len(x0) - M)])[None, :],
                                            problem.theta, problem.theta))
    if problem.uses_epigraph:
        constraints.append(_epigraph_constraints(problem, a0, P))

    objective_kwargs = (
        {"jac": jac, "hess": hess}
        if problem.hessian_mode in ("gauss_newton", "objective_only")
        else {"jac": jac, "hess": BFGS()}
    )

    # ---- callback: record, then decide whether to stop ------------------
    state = {"n": 0, "checkpoints": 0, "stalled": 0, "prev": None, "early": False}

    def callback(xk, res=None):
        state["n"] += 1
        a = to_coeffs(xk)
        terms = metrics.cost_terms(a, T, N)
        eq = np.asarray(geometry.equality_residuals(a, T, N))
        s = float(xk[nz]) if problem.uses_epigraph else None
        total = terms.energy + (problem.lam * T * s if s is not None else 0.0)

        if recorder is not None and (state["n"] - 1) % problem.checkpoint_every == 0:
            recorder.append(
                state["n"] - 1,
                a,
                cost_total=total,
                cost_energy=terms.energy,
                cost_curv=terms.curv,
                cost_peak=terms.peak,
                res_gate=abs(float(g @ a) - problem.theta),
                res_closure=eq[:2],
                res_area=eq[2:],
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

    fine = reevaluate(coeffs, T, fine_grids) if fine_grids else {}
    terms = metrics.cost_terms(coeffs, T, N)
    eq = np.asarray(geometry.equality_residuals(coeffs, T, N))
    s_final = float(res.x[nz]) if problem.uses_epigraph else None

    out = SolveResult(
        coeffs=coeffs,
        stop_reason=stop_reason,
        status=int(res.status),
        nit=int(res.nit),
        wall_clock_s=wall,
        n_checkpoints=state["checkpoints"],
        terms=terms,
        gate_residual=abs(float(g @ coeffs) - problem.theta),
        equality_residuals=eq,
        epigraph_s=s_final,
        fine_grid=fine,
        scipy_message=str(res.message),
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
        )
    return out
