"""F04 acceptance: the full-cost error-budget objective (``curve_opt.budget``)
and the budget-mode solver (``curve_opt.optimize.BudgetProblem``/``solve_budget``).

Five criteria, task brief §3:

1. weights zero + energy objective + ``gate_level="two_level"`` reproduces
   Step 08 after the §3.1 unit conversion -- the "old/new machine" gate.
2. the (gate, c1) constraint Jacobian is full rank at a converged point.
3. the Gauss-Newton objective/constraint Hessian is re-verified on the new
   constraint set, against a ``default`` (BFGS-both) control.
4. ``C3^planar = 1.6449 epsilon^2``, evaluated on the design curve.
5. the solved peak vs the Fenchel/hardware budget ``T Omega_max = 15.708``,
   and whether C1 is peak-limited.

Marked ``slow`` throughout: every criterion but (4) needs an actual
``trust-constr`` solve (or a converged point to differentiate at). Run at a
deliberately small ``M`` (6, not the project default 20) and a coarse
``N_grid`` -- the point of these tests is that the *machinery* (objective,
constraints, Jacobian rank, GN vs default) is correct, not that they
reproduce a publication-grade M=20 claim; the dev log carries a larger-scale
exploratory run separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from curve_opt import basis, budget, device, gate, optimize, parametrization
from conftest import requires_novera

DEV = device.DEFAULT_DEVICE
T_PHYS = DEV.gate_time
XPI = {"name": "X(pi)", "matrix": [[0, 1], [1, 0]]}


# --------------------------------------------------------------------------
# budget.py units: weights, the Gaussian delta_z^4 assumption, and the
# evaluation-object discipline (design curve vs broadcast waveform).
# --------------------------------------------------------------------------


def test_weights_from_device_match_the_plan_2_5_orders_of_magnitude():
    w = budget.weights_from_device(DEV)
    assert w.c1 == pytest.approx(6.5797e-8, rel=1e-3)
    assert w.c3 == pytest.approx(2.6667e-4, rel=1e-4)
    assert w.c4 == pytest.approx(1.0 / 3.0)


def test_delta_z_fourth_moment_uses_the_gaussian_kurtosis_factor():
    """``<delta_z^4> = 3 <delta_z^2>^2`` -- task brief §2.2's explicit instruction.

    Differs from ``_plan_full_cost.md`` §2.5's own illustrative table by
    exactly that factor of 3 (2.60e-14 -> 7.79e-14 ns^-4); see
    ``curve_opt.budget``'s module docstring and the F04 dev log.
    """
    w = budget.weights_from_device(DEV)
    dz2 = DEV.static_detuning_rate ** 2
    assert w.c2 == pytest.approx(3.0 * dz2 ** 2 / 6.0, rel=1e-12)
    assert w.c2 == pytest.approx(7.7927e-14, rel=1e-3)


def test_design_and_broadcast_waveforms_differ_when_leakage_correction_is_nonzero():
    """The single easiest mistake in this step (CLAUDE.md, task brief §2.1):
    confirms :func:`curve_opt.budget.design_chain` and
    :func:`curve_opt.budget.broadcast_chain` are genuinely two different
    forward passes, not the same one called twice.
    """
    M = 6
    a = basis.naive_coeffs(np.pi, T_PHYS, M)
    c = 0.3 * np.ones(M)
    curve = parametrization.design_curve(a, c, 0.0, T_PHYS, 400)
    design_om_x, design_om_y = budget.design_waveform_of_curve(curve)
    broadcast_om_x, broadcast_om_y = budget.broadcast_waveform(a, c, 0.0, T_PHYS, DEV.delta, 400)
    assert not np.allclose(np.asarray(design_om_y), np.asarray(broadcast_om_y), atol=1e-6)


# --------------------------------------------------------------------------
# ★ acceptance (4): the planar C3 floor, evaluated on the design curve.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("M", [6, 20])
def test_c3_planar_floor_matches_the_analytic_1_6449_epsilon_squared(M):
    """``_plan_full_cost.md`` §2.5b: waveform-independent, a function of the
    gate angle alone. Checked at two different ``a`` (naive and min-norm) and
    two ``M`` to show it really is waveform-independent, not a coincidence of
    one particular coefficient vector.
    """
    theta = np.pi
    c = np.zeros(M)
    analytic = budget.c3_planar_floor(theta, DEV.control_error)
    assert analytic == pytest.approx(1.6449 * DEV.control_error ** 2, rel=1e-3)
    assert analytic == pytest.approx(6.5797e-4, rel=1e-3)

    for a in (basis.naive_coeffs(theta, T_PHYS, M), basis.min_norm_gate_only(theta, T_PHYS, M)):
        terms = budget.budget_terms(a, c, 0.0, T_PHYS, DEV, N=2000)
        assert float(terms.c3) == pytest.approx(analytic, rel=5e-4)


# --------------------------------------------------------------------------
# the JAX three-level propagator: unitary, and cross-checked against Novera.
# --------------------------------------------------------------------------


def test_three_level_propagator_is_unitary():
    """Not :func:`curve_opt.propagate.unitarity_error` -- that helper is
    hardcoded to a 2x2 identity (its own module's SU(2) chain), so this
    checks the 3x3 case directly."""
    M = 6
    a = basis.naive_coeffs(np.pi, T_PHYS, M)
    c = np.zeros(M)
    om_x, om_y = budget.broadcast_waveform(a, c, 0.0, T_PHYS, DEV.delta, 400)
    U3 = np.asarray(gate.three_level_propagator(om_x, om_y, T_PHYS, DEV.delta))
    err = np.max(np.abs(np.conj(U3).T @ U3 - np.eye(3)))
    assert err < 1e-10


@requires_novera
def test_three_level_propagator_matches_novera_three_level_gate():
    """Independent numpy/scipy reference, on a differently-discretized grid
    (edge-averaged vs midpoint) -- agreement is approximate, not bit for bit
    (the same caveat F01/F03 documented for ``leakage_amplitude``).
    """
    from curve_opt import novera

    N = 2000
    t_mid = (np.arange(N) + 0.5) * (T_PHYS / N)
    amp = 0.3 * DEV.rabi_max_rate
    om_x_mid = amp * np.sin(np.pi * t_mid / T_PHYS)
    om_y_mid = 0.2 * amp * np.cos(np.pi * t_mid / T_PHYS) * np.sin(2 * np.pi * t_mid / T_PHYS)

    U3_jax = np.asarray(gate.three_level_propagator(om_x_mid, om_y_mid, T_PHYS, DEV.delta))

    t_edge = np.linspace(0.0, T_PHYS, N + 1)
    om_x_edge = np.interp(t_edge, t_mid, om_x_mid)
    om_y_edge = np.interp(t_edge, t_mid, om_y_mid)
    pulse = novera.build_pulse(t_edge, om_x_edge, om_y_edge, T_PHYS, device=DEV)
    calibration = {"detuning": -DEV.static_detuning, "scale": 1.0, "frame_phase": 0.0}
    U3_novera = np.asarray(
        novera.three_level_gate(pulse, epsilon=0.0, stride=1, calibration=calibration)
    )
    assert np.max(np.abs(U3_jax - U3_novera)) < 5e-6


# --------------------------------------------------------------------------
# ★ acceptance (1): weights zero + energy objective + gate_level="two_level"
# reproduces Step 08 after the §3.1 unit conversion. This is the *pre-existing*
# ``Problem``/``solve`` machinery (untouched by F04), run at the device's own
# physical T and M rather than the energy era's T=1 -- the master gate that
# the new physical-unit machinery agrees with the old dimensionless one.
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_old_machine_at_physical_T_reproduces_step08_after_unit_conversion():
    M = DEV.n_modes  # 20, matching Step 08's headline M=20 case
    problem = optimize.Problem(
        M=M,
        T=T_PHYS,
        theta=np.pi,
        target_gate=XPI,
        N_grid=optimize.N_CLAIM,
        coeffs0=basis.naive_coeffs(np.pi, T_PHYS, M),
    )
    res = optimize.solve(problem)
    assert res.stop_reason == "converged", res.scipy_message

    # energy and the peak *invariant* are dimensionless (basis.py's own
    # "invariant" framing) and must reproduce Step 08's M=20 numbers exactly,
    # unaffected by switching from T=1 to T=50 ns.
    assert res.terms.energy == pytest.approx(81.7690, abs=5e-4)
    assert res.terms.peak == pytest.approx(22.7482, abs=5e-3)

    fine = res.fine_grid["grids"]["500000"]
    assert fine["max_equality_residual"] <= 1e-8

    # §3.1's conversion: max|Omega|_phys = peak / T. Step 08's closed +
    # zero-area solution needs 0.4550 rad/ns, 1.448x this device's hardware
    # ceiling (0.31416 rad/ns) -- exactly the §2.6b finding that motivates
    # F04's hard peak *inequality* rather than treating closure as hard.
    peak_phys = res.terms.peak / T_PHYS
    assert peak_phys == pytest.approx(0.45496, rel=2e-3)
    assert peak_phys > DEV.rabi_max_rate
    assert peak_phys / DEV.rabi_max_rate == pytest.approx(1.448, rel=2e-3)


# --------------------------------------------------------------------------
# budget-mode fixture: one small-scale converged solve, reused by (2), (5)
# and (part of) the GN-vs-default comparison in (3).
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
def small_budget_solution():
    """One converged small-scale (M=6) budget-mode solve, ``hessian_mode="default"``.

    ``gate_level="two_level"`` (task brief §2.3: the fast warm-start channel)
    keeps this test-suite-scale solve well under a minute per criterion; the
    three-level route is exercised separately (unitarity + Novera cross-check
    above) and is not what criteria (2)/(3)/(5) are about.
    """
    problem = _small_budget_problem(maxiter=3000, hessian_mode="default")
    res = optimize.solve_budget(problem, verbose=0)
    assert res.stop_reason == "converged", res.scipy_message
    return problem, res


# --------------------------------------------------------------------------
# ★ acceptance (2): the (gate, c1) constraint Jacobian is full rank at the
# solution.
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_gate_and_c1_constraint_jacobian_is_full_rank_at_the_solution(small_budget_solution):
    problem, res = small_budget_solution
    # F05b: the solver vector grew a second epigraph slack (r, the R guard's
    # max|tau| bound) -- the gate/c1 block does not depend on it, but the
    # vector still has to be the right length for jax.jacrev to trace.
    x = np.concatenate([res.coeffs_a, res.coeffs_c, [res.Phi_0, res.phi_vz, res.s, res.r]])

    J_gate = optimize._budget_gate_constraint(problem).jac(x)
    J_c1 = optimize._budget_c1_constraint(problem).A
    J = np.vstack([J_gate, J_c1])

    n_constraints = 3 + 2  # gate (3) + c1 start/end (2)
    assert J.shape == (n_constraints, problem.n_vars)
    rank = np.linalg.matrix_rank(J, tol=1e-8)
    assert rank == n_constraints

    # not a knife-edge full rank: singular values stay well clear of zero.
    sv = np.linalg.svd(J, compute_uv=False)
    assert sv.min() > 1e-3


# --------------------------------------------------------------------------
# ★ acceptance (5): solved peak vs the Fenchel/hardware budget, and whether
# C1 is peak-limited.
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_solved_peak_vs_fenchel_budget_and_c1_floor(small_budget_solution):
    _problem, res = small_budget_solution

    assert DEV.peak_budget == pytest.approx(15.708, rel=1e-3)

    peak_dimless = res.peak_phys * T_PHYS
    # the peak epigraph is a hard inequality against Omega_max, not against
    # the Fenchel bound -- so the *solved* peak sits at the hardware ceiling
    # (Omega_max, peak_budget = T*Omega_max), not necessarily at the looser
    # Fenchel floor of 2 pi.
    assert res.peak_phys <= DEV.rabi_max_rate + 1e-6
    assert peak_dimless == pytest.approx(DEV.peak_budget, rel=0.05)

    # C1 is far from zero relative to machine precision (contrast the gate
    # residual, ~1e-16): closed *and* zero-area is not reachable within the
    # peak ceiling at this M, so C1 is peak-limited rather than driven to the
    # floor an unconstrained-peak solve would reach.
    assert res.terms.c1 > 1e-8
    assert res.gate_residual.max() < 1e-10


# --------------------------------------------------------------------------
# ★ acceptance (3): Gauss-Newton re-verified on the new (gate + c1 + peak)
# constraint set, against a ``default`` (BFGS-both) control.
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_gauss_newton_speedup_on_the_budget_constraint_set(small_budget_solution):
    """★ AGENTS.md discipline 3: keep the ``default`` control, report the ratio.

    Reuses the already-solved ``default`` run from the module fixture and
    solves the identical problem again with ``hessian_mode="gauss_newton"``,
    same starting point, same grid, same ``maxiter`` budget.

    ★ Honest negative result (task brief's "重新验证", not "confirm"): unlike
    the energy-time era (Step 08/12, 5.9x..25.3x), Gauss-Newton does *not*
    speed this problem up. At this scale (M=6, N_grid=300) ``default``
    (BFGS on both the objective and the constraint) reaches
    ``stop_reason="converged"`` in 2444 iterations / ~150s; ``gauss_newton``
    (the analytic ``2 J_R^T J_R`` objective Hessian + a zeroed constraint
    Hessian) still has not satisfied trust-constr's own convergence test
    after 3000 iterations (``stop_reason="maxiter"``) at a comparable wall
    clock per iteration, though it has driven the cost down by a comparable
    factor and the gate residual to ~1e-9. See the F04 dev log for the
    candidate explanation (dropping the residual's own curvature loses more
    here than it did for the old quadratic-objective problem, where the
    *objective* Hessian was exact rather than an approximation) and the
    higher-``maxiter`` control that confirms GN eventually converges, just
    slower, not stuck.
    """
    problem_default, res_default = small_budget_solution
    problem_gn = problem_default._replace(hessian_mode="gauss_newton")

    res_gn = optimize.solve_budget(problem_gn, verbose=0)

    # GN substantially reduces the cost and nearly satisfies the gate even
    # though it does not reach ``converged`` in the same iteration budget --
    # not stuck/diverging, just not faster here. Order-of-magnitude sanity
    # (not a tight bound): within 5x of default's converged total cost.
    assert res_gn.terms.total < 5.0 * res_default.terms.total
    assert res_gn.gate_residual.max() < 1e-6

    print(
        f"\nGN vs default (M=6, N_grid=300): "
        f"nit {res_gn.nit} ({res_gn.stop_reason}) vs {res_default.nit} ({res_default.stop_reason}), "
        f"wall {res_gn.wall_clock_s:.1f}s vs {res_default.wall_clock_s:.1f}s, "
        f"total cost {res_gn.terms.total:.3e} vs {res_default.terms.total:.3e}"
    )
