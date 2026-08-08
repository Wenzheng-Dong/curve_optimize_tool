"""F05b acceptance: the R guard hard inequality on ``BudgetProblem`` (``general`` layer).

Task brief §4, three criteria:

1. added, and enforced at a real solve: ``max|tau| <= rho * |Delta| / 2``
   (measured via :attr:`curve_opt.optimize.BudgetSolveResult.tau_peak`).
2. the ``rcp+3D`` construction's decomposition ratio improves substantially
   from the pre-guard 168x (checked in the dev log against a fresh
   ``solve_budget`` run, not here -- that comparison needs the full F05
   Monte-Carlo machinery of ``_dev_logs/F05_budget_validation.py``, which is
   an analysis script, not a pytest module, same split F05 used).
3. the (gate, c1) constraint Jacobian stays full rank at a converged point
   with the guard active (F04 acceptance (2)'s rank check, re-run).

Marked ``slow`` throughout except the two constraint-plumbing unit tests,
which need no solve.
"""

from __future__ import annotations

import numpy as np
import pytest

from curve_opt import basis, device, optimize, parametrization

DEV = device.DEFAULT_DEVICE
T_PHYS = DEV.gate_time


# --------------------------------------------------------------------------
# constraint plumbing (fast, no solve)
# --------------------------------------------------------------------------


def test_tau_guard_bound_is_rho_times_half_delta():
    problem = optimize.BudgetProblem(
        M=6, T=T_PHYS, theta=np.pi,
        coeffs0_a=np.zeros(6), coeffs0_c=np.zeros(6),
    )
    assert problem.rho == pytest.approx(0.5)
    assert problem.tau_guard_bound == pytest.approx(0.5 * abs(DEV.delta) / 2.0)
    # the numeric value quoted to leader (task brief report):
    assert problem.tau_guard_bound == pytest.approx(0.25 * abs(DEV.delta), rel=1e-12)


def test_tau_design_matrix_matches_tau_of_coeffs():
    """The R guard's constant ``S_tau`` (a jacrev of ``tau_of_coeffs`` at ``c=0``)
    must reproduce ``tau_of_coeffs`` exactly for *any* ``c`` -- it is a linear map,
    so agreement at one nonzero point away from the basepoint it was built at is
    already a full check.
    """
    M, N = 8, 50
    rng = np.random.default_rng(0)
    c = rng.normal(size=M)
    S_tau = optimize._tau_design_matrix(M, T_PHYS, N)
    assert S_tau.shape == (N, M)
    direct = np.asarray(parametrization.tau_of_coeffs(c, T_PHYS, N))
    assert np.allclose(S_tau @ c, direct, atol=1e-12, rtol=1e-10)


def test_tau_guard_constraint_sign_convention():
    """A deliberately oversized ``c`` violates the constraint at ``r=0`` and
    satisfies it once ``r`` is set to (or past) the true ``max|tau|`` -- checks
    the *direction* of the inequality, not just that the matrix has the right
    shape.
    """
    M = 6
    problem = optimize.BudgetProblem(
        M=M, T=T_PHYS, theta=np.pi,
        coeffs0_a=np.zeros(M), coeffs0_c=np.zeros(M), n_peak=40,
    )
    lc = optimize._budget_tau_guard_constraint(problem)
    n = problem.n_vars

    c = np.zeros(M)
    c[0] = 5.0  # large -- guaranteed to push max|tau| well past any small r
    tau = np.asarray(parametrization.tau_of_coeffs(c, T_PHYS, problem.n_peak))
    true_max = float(np.max(np.abs(tau)))
    assert true_max > 0.0

    def x_with(r):
        x = np.zeros(n)
        x[M : 2 * M] = c
        x[-1] = r
        return x

    violated = lc.A @ x_with(0.0)
    assert np.max(violated) > 0.0  # r too small -- some row exceeds the ub=0 cap

    satisfied = lc.A @ x_with(true_max * 1.001)
    assert np.max(satisfied) <= 1e-9  # r past the true max -- every row feasible


# --------------------------------------------------------------------------
# ★ acceptance (1)/(3): a real small-scale solve, guard active by default
# --------------------------------------------------------------------------


def _small_budget_problem(**kw):
    M = 6
    a0 = basis.min_norm_gate_only(np.pi, T_PHYS, M)
    c0 = np.zeros(M)
    kw.setdefault("device", DEV)
    kw.setdefault("gate_level", "two_level")
    kw.setdefault("N_grid", 300)
    kw.setdefault("n_peak", 80)
    kw.setdefault("fixed_budget_survey", True)
    return optimize.BudgetProblem(M=M, T=T_PHYS, theta=np.pi, coeffs0_a=a0, coeffs0_c=c0, **kw)


@pytest.fixture(scope="module")
def small_guarded_solution():
    """One converged small-scale (M=6) budget-mode solve with the R guard
    active at its default rho -- mirrors ``test_f04_budget.py``'s
    ``small_budget_solution`` fixture (same scale, same gate route), reused
    by both criteria (1) and (3) below.
    """
    problem = _small_budget_problem(maxiter=3000, hessian_mode="default")
    res = optimize.solve_budget(problem, verbose=0)
    assert res.stop_reason == "converged", res.scipy_message
    return problem, res


@pytest.mark.slow
def test_solved_point_respects_the_tau_guard(small_guarded_solution):
    problem, res = small_guarded_solution
    assert res.tau_peak <= problem.tau_guard_bound + 1e-8
    print(
        f"\nM=6 guarded solve: tau_peak={res.tau_peak:.4e}, "
        f"bound={problem.tau_guard_bound:.4e}, |c|={np.linalg.norm(res.coeffs_c):.3f}"
    )


@pytest.mark.slow
def test_gate_and_c1_jacobian_still_full_rank_with_tau_guard(small_guarded_solution):
    """F04 acceptance (2)'s rank check, re-run with the guard active (task
    brief F05b acceptance ③): the extra ``r`` column is all-zero in this
    block (gate/c1 do not depend on ``r``), so the achievable rank is
    unaffected -- checked directly, not assumed.
    """
    problem, res = small_guarded_solution
    x = np.concatenate(
        [res.coeffs_a, res.coeffs_c, [res.Phi_0, res.phi_vz, res.s, res.r]]
    )
    assert x.shape == (problem.n_vars,)

    J_gate = optimize._budget_gate_constraint(problem).jac(x)
    J_c1 = optimize._budget_c1_constraint(problem).A
    J = np.vstack([J_gate, J_c1])

    n_constraints = 3 + 2  # gate (3) + c1 start/end (2)
    assert J.shape == (n_constraints, problem.n_vars)
    rank = np.linalg.matrix_rank(J, tol=1e-8)
    assert rank == n_constraints

    sv = np.linalg.svd(J, compute_uv=False)
    assert sv.min() > 1e-3

    # the added column (r) is exactly zero in this block, by construction.
    assert np.all(J[:, -1] == 0.0)
