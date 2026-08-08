"""F06b acceptance: the hard fluence cap on ``BudgetProblem`` (``general`` layer).

Task brief §4.1: exactly two tests.

1. Independent reconciliation -- the value :func:`curve_opt.optimize._budget_fluence_constraint`
   implies for ``int_0^T kappa^2 dt`` must agree with a *second, independent* path:
   :func:`curve_opt.parametrization.design_curve`'s own ``kappa`` samples, numerically
   integrated on a fine grid. This is the test the task brief calls out as specifically
   targeting the ``T`` unit-conversion factor (``_budget_fluence_constraint``'s docstring:
   ``basis.energy_invariant`` returns ``T`` times the physical fluence, so the constraint
   divides by one extra factor of ``T`` beyond the ``cap`` normalization) -- a wrong power
   of ``T`` here would still pass a "constraint has the right shape/sign" test but silently
   move the cap by a factor of 50 (this project's ``T``).
2. Jacobian-vs-central-difference agreement, plus the "columns other than ``a`` are exactly
   zero" claim from the same docstring (``c``, ``Phi_0``, ``phi_vz``, ``s``, ``r`` do not
   enter the fluence).

``fluence_cap=None``'s backward compatibility is already covered by every pre-F06b test in
this suite (none of them pass ``fluence_cap``, so the default path is exercised on every
run of the full suite) -- not re-tested here (task brief §4.1: "不必新写").
"""

from __future__ import annotations

import numpy as np
import pytest

from curve_opt import device, optimize, parametrization

DEV = device.DEFAULT_DEVICE
T_PHYS = DEV.gate_time


def _small_fluence_problem(cap: float = 1.0) -> optimize.BudgetProblem:
    M = 6
    rng = np.random.default_rng(0)
    a0 = rng.normal(size=M)
    c0 = np.zeros(M)
    return optimize.BudgetProblem(
        M=M, T=T_PHYS, theta=np.pi,
        coeffs0_a=a0, coeffs0_c=c0,
        device=DEV, n_peak=40, fluence_cap=cap,
    )


def _random_x(problem: optimize.BudgetProblem, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=problem.n_vars) * 0.3
    x[-1] = abs(x[-1])  # r >= 0 is the only physically meaningful sign for the R guard slack
    return x


def test_fluence_constraint_matches_independent_design_curve_integral():
    """★ independent reconciliation, specifically targets the T unit-conversion factor."""
    M = 6
    cap = 2.345  # arbitrary, deliberately not one of the three run values
    problem = _small_fluence_problem(cap=cap)
    con = optimize._budget_fluence_constraint(problem)

    rng = np.random.default_rng(2)
    a = rng.normal(size=M) * 0.7
    c = np.zeros(M)
    x = np.zeros(problem.n_vars)
    x[:M] = a
    x[M : 2 * M] = c

    # what the constraint claims int_0^T kappa^2 dt is:
    implied_fluence = (float(con.fun(x)[0]) + 1.0) * cap

    # independent path: parametrization.design_curve's own kappa, fine grid, direct quadrature.
    N_fine = 20_000
    dc = parametrization.design_curve(a, c, 0.0, T_PHYS, N_fine)
    kappa = np.asarray(dc.kappa)
    fine_fluence = float(np.sum(kappa**2) * dc.dt)

    assert implied_fluence == pytest.approx(fine_fluence, rel=1e-8)


def test_fluence_constraint_jacobian_matches_central_difference_and_is_a_only():
    problem = _small_fluence_problem(cap=1.579)
    con = optimize._budget_fluence_constraint(problem)
    M, n = problem.M, problem.n_vars
    x0 = _random_x(problem)

    J_analytic = con.jac(x0)
    assert J_analytic.shape == (1, n)

    eps = 1e-6
    J_fd = np.zeros((1, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J_fd[0, i] = (con.fun(x0 + dx)[0] - con.fun(x0 - dx)[0]) / (2 * eps)

    assert np.allclose(J_analytic, J_fd, rtol=1e-6, atol=1e-8)
    # columns beyond a (c, Phi_0, phi_vz, s, r) are exactly zero (docstring claim)
    assert np.all(J_analytic[0, M:] == 0.0)
