"""Step 08 acceptance: the trust-constr wrapper.

Claim-run criteria (``_plan.md`` §6.4): reproduce 83.75 at M=12 and 81.77 at
M=20, CONVERGED, with the fine-grid (N=5e5) residual <= 1e-8; measure the
Gauss-Newton speedup; leave a complete RunRecord behind.

The heavy claim-runs are marked ``slow``; run them with ``pytest -m slow`` or
plain ``pytest`` (they are included by default, ~1 min total).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from curve_opt import basis, geometry, metrics, optimize, plotting, recorder

T = 1.0
XPI = {"name": "X(pi)", "matrix": [[0, 1], [1, 0]]}


def problem(M=12, **kw):
    kw.setdefault("N_grid", optimize.N_CLAIM)
    kw.setdefault("coeffs0", basis.naive_coeffs(np.pi, T, M))
    return optimize.Problem(M=M, T=T, theta=np.pi, target_gate=XPI, **kw)


# --------------------------------------------------------------------------
# ★ v4.2: the winding branch is snapped to the exact theoretical value
# --------------------------------------------------------------------------


def test_snap_picks_the_branch_and_returns_the_exact_value():
    for k in (-1, 0, 1, 2):
        measured = np.pi + 2 * np.pi * k + 3e-4  # a plausible ansatz gate error
        exact, got_k = optimize.snap_winding_branch(measured, np.pi)
        assert got_k == k
        assert exact == np.pi + 2 * np.pi * k  # exact float, not approximately


def test_snap_is_exactly_pi_for_the_rcp_gate_error():
    """rcp_lemniscate(pi) measures 3.1418139; the constraint RHS must be pi itself.

    ★ The point of the v4.2 verdict: the optimization exists partly to remove this
    2.2e-4 gate error, so writing the measured value into the constraint would
    define the error as the target.
    """
    exact, k = optimize.snap_winding_branch(3.141813905776, np.pi)
    assert exact == np.pi
    assert k == 0


def test_snap_refuses_an_ansatz_that_is_not_on_any_branch():
    """alpha_3d turns 12.43 rad -- no branch of X(pi) is near that.

    Snapping anyway would silently pair an identity-gate 3D family with an X(pi)
    target and hide that the two are unrelated.
    """
    with pytest.raises(ValueError, match="does not implement the target gate"):
        optimize.snap_winding_branch(12.42520, np.pi)
    exact, k = optimize.snap_winding_branch(12.42520, np.pi, max_distance=4.0)
    assert k == 1 and exact == pytest.approx(3 * np.pi)


def test_manifest_separates_the_exact_branch_from_the_measured_turning():
    man = optimize.manifest_for(
        problem(theta_before=3.141813905776), early_stop=optimize.EarlyStop()
    )
    assert man["winding_branch"] == np.pi
    assert man["theta_before"] == 3.141813905776
    assert man["solver"]["early_stop"]["patience"] == 5


# --------------------------------------------------------------------------
# gate elimination
# --------------------------------------------------------------------------


@pytest.mark.parametrize("M", [8, 12, 20])
def test_affine_projection_satisfies_the_gate_identically(M):
    a0, P = optimize._gate_projection(M, T, np.pi)
    g = basis.gate_row(M, T)
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = a0 + P @ rng.normal(size=M - 1)
        assert basis.gate_angle(a, T) == pytest.approx(np.pi, abs=1e-13)


def test_projection_basis_is_orthonormal_so_the_hessian_stays_T2I():
    """``P^T P = I`` is what keeps the analytic objective Hessian of §3.1 valid."""
    M = 12
    _, P = optimize._gate_projection(M, T, np.pi)
    assert P.shape == (M, M - 1)
    assert np.allclose(P.T @ P, np.eye(M - 1), atol=1e-13)
    fun, jac, hess = optimize._objective(problem(M=M), *optimize._gate_projection(M, T, np.pi))
    assert np.allclose(hess(np.zeros(M - 1)), T**2 * np.eye(M - 1), atol=1e-13)


def test_objective_gradient_matches_finite_differences():
    M = 12
    a0, P = optimize._gate_projection(M, T, np.pi)
    fun, jac, _ = optimize._objective(problem(M=M), a0, P)
    z = np.random.default_rng(1).normal(size=M - 1)
    h = 1e-7
    fd = np.array(
        [(fun(z + h * e) - fun(z - h * e)) / (2 * h) for e in np.eye(M - 1)]
    )
    assert np.allclose(jac(z), fd, rtol=1e-6)


def test_configuration_floors_are_enforced():
    with pytest.raises(ValueError, match="below the planar floor"):
        optimize.solve(problem(maxiter=300))
    with pytest.raises(ValueError, match="hessian_mode"):
        optimize.solve(problem(hessian_mode="magic"))
    with pytest.raises(ValueError, match="unknown gate handling"):
        optimize.solve(problem(gate="hope"))


def test_fixed_budget_survey_explicitly_allows_a_subfloor_cap():
    p = problem(maxiter=3, fixed_budget_survey=True, N_grid=optimize.N_SURVEY)
    result = optimize.solve(p, fine_grids=())
    assert result.stop_reason == "maxiter"
    assert result.void_for_claims
    assert optimize.manifest_for(p)["solver"]["fixed_budget_survey"] is True


# --------------------------------------------------------------------------
# ★ acceptance: reproduce step00d
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "M,energy,curv,peak",
    [(12, 83.7479, 6.6379, 22.632), (20, 81.7690, 6.6524, 22.748)],
)
def test_reproduces_step00d_claim_run(M, energy, curv, peak):
    """★ CONVERGED + fine-grid residual <= 1e-8, energy/curv/peak as recorded."""
    res = optimize.solve(problem(M=M))
    assert res.stop_reason == "converged", res.scipy_message
    assert res.nit < 1500
    assert res.terms.energy == pytest.approx(energy, abs=5e-4)
    assert res.terms.curv == pytest.approx(curv, abs=5e-4)
    assert res.terms.peak == pytest.approx(peak, abs=5e-3)

    fine = res.fine_grid["grids"]["500000"]
    assert fine["max_equality_residual"] <= 1e-8, fine
    assert abs(fine["closure"]) <= 1e-8
    assert abs(fine["area"]) <= 1e-8
    assert res.gate_residual <= 1e-13


@pytest.mark.slow
def test_fine_grid_residual_matches_the_recorded_verify_best_values():
    """step00d's verify_best printed 8.03e-9 / 5.86e-9 at 1e5 and 8.35e-9 / 6.10e-9 at 5e5."""
    res = optimize.solve(problem(M=12))
    g1 = res.fine_grid["grids"]["100000"]
    g5 = res.fine_grid["grids"]["500000"]
    assert g1["closure"] == pytest.approx(8.03e-9, rel=0.05)
    assert abs(g1["area"]) == pytest.approx(5.86e-9, rel=0.05)
    assert g5["closure"] == pytest.approx(8.35e-9, rel=0.05)
    assert abs(g5["area"]) == pytest.approx(6.10e-9, rel=0.05)


@pytest.mark.slow
def test_optimizing_on_a_coarse_grid_fails_the_fine_grid_criterion():
    """★ The fine-grid residual is a property of N_grid, not only of convergence.

    N=4000 converges just as cleanly but leaves 2e-7 on the truth grid, 25x worse
    than N=20000 -- which is why N_CLAIM exists.
    """
    coarse = optimize.solve(problem(N_grid=optimize.N_SURVEY))
    assert coarse.stop_reason == "converged"
    assert coarse.terms.energy == pytest.approx(83.7479, abs=5e-4)
    worst = coarse.fine_grid["grids"]["500000"]["max_equality_residual"]
    assert worst > 1e-8
    assert worst == pytest.approx(2.0e-7, rel=0.5)


@pytest.mark.slow
def test_both_gate_routes_reach_the_same_optimum():
    """§3.1 offers projection or LinearConstraint; they must agree on the answer."""
    a = optimize.solve(problem(gate="projection"))
    b = optimize.solve(problem(gate="linear_constraint", maxiter=3000))
    assert a.terms.energy == pytest.approx(b.terms.energy, rel=1e-6)
    assert a.gate_residual <= 1e-13
    assert b.gate_residual <= 1e-9  # a constraint, not an identity
    assert np.allclose(a.coeffs, b.coeffs, atol=1e-6)


@pytest.mark.slow
def test_rcp_start_needs_more_iterations_than_naive_but_lands_identically():
    """step00d §1: the ansatz moves the iteration count, not the endpoint."""
    from curve_opt import ansatz

    ansatz.register_builtin_families(overwrite=True)
    naive = optimize.solve(problem(M=12))
    circle = optimize.solve(
        problem(M=12, coeffs0=ansatz.project_named("circle_arc", M=12, T=T).coeffs_a)
    )
    assert circle.terms.energy == pytest.approx(naive.terms.energy, rel=1e-7)
    assert circle.nit > naive.nit


# --------------------------------------------------------------------------
# ★ acceptance: Gauss-Newton speedup
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_gauss_newton_speedup_in_the_step00d_configuration():
    """★ >= 2x against ``default``, measured where the claim was made.

    The 5.9..25.3x of step00d/00e was measured with the gate as a
    LinearConstraint and no analytic Hessians. Reproduced here: ``default`` needs
    ~450 iterations (step00d recorded 477) against ~40 for Gauss-Newton.

    With the gate eliminated by affine projection the speedup largely disappears,
    because elimination already removes the ill-conditioning -- see the dev log.
    """
    common = {"gate": "linear_constraint", "maxiter": 3000}
    optimize.solve(problem(N_grid=2000, **common), fine_grids=())  # warm the jit

    timings = {}
    for mode in ("gauss_newton", "default"):
        start = time.perf_counter()
        res = optimize.solve(problem(hessian_mode=mode, **common), fine_grids=())
        timings[mode] = (time.perf_counter() - start, res)
        assert res.stop_reason == "converged"
        assert res.terms.energy == pytest.approx(83.7479, abs=5e-4)

    gn_wall, gn = timings["gauss_newton"]
    df_wall, df = timings["default"]
    assert df.nit / gn.nit >= 2.0, (gn.nit, df.nit)
    assert df_wall / gn_wall >= 2.0, (gn_wall, df_wall)
    assert df.nit == pytest.approx(450, rel=0.35)  # step00d recorded 477


@pytest.mark.slow
def test_the_constraint_hessian_is_what_buys_the_speedup():
    """★ §3.1 as written: supplying only the objective Hessian buys nothing.

    Step 08 reported the opposite, from this very test. That was wrong: the mode
    dispatch in ``_nonlinear_constraint`` sent both ``gauss_newton`` and
    ``objective_only`` down the zeroed-Hessian branch, so the two modes were
    literally the same code and measuring them against each other measured
    nothing. Fixed in Step 12; the numbers now reproduce step00e's hesstest table
    row by row (rcp/M=12: 138 / 351 / 348 here against its 133 / 361 / 360).
    """
    common = {"gate": "linear_constraint", "maxiter": 3000}
    results = {
        mode: optimize.solve(problem(hessian_mode=mode, **common), fine_grids=())
        for mode in optimize.HESSIAN_MODES
    }
    for mode, res in results.items():
        assert res.terms.energy == pytest.approx(83.7479, abs=5e-4), mode
        assert res.stop_reason == "converged", mode

    gn = results["gauss_newton"].nit
    # the analytic objective Hessian on its own lands with the default, not with GN
    assert results["objective_only"].nit > 5 * gn
    assert results["default"].nit > 5 * gn
    assert results["objective_only"].nit == pytest.approx(results["default"].nit, rel=0.30)


def test_the_three_hessian_modes_are_actually_distinct_configurations():
    """Guard for the dispatch bug above: same objective, three different setups.

    Cheap and structural -- it inspects what is handed to scipy rather than how
    fast it runs, so it fails immediately if two modes are ever collapsed again.
    """
    from scipy.optimize import BFGS

    M = 12
    a0, P = optimize._gate_projection(M, T, np.pi)
    constraint_hess = {
        mode: optimize._nonlinear_constraint(problem(M=M, hessian_mode=mode), a0, P).hess
        for mode in optimize.HESSIAN_MODES
    }
    assert callable(constraint_hess["gauss_newton"])
    assert not isinstance(constraint_hess["gauss_newton"], BFGS)
    for mode in ("objective_only", "default"):
        assert isinstance(constraint_hess[mode], BFGS), mode

    objective_hess = {
        mode: optimize._objective(problem(M=M, hessian_mode=mode), a0, P)[2]
        for mode in optimize.HESSIAN_MODES
    }
    # both Hessian-supplying modes expose the same analytic objective Hessian; what
    # distinguishes them from each other is only the constraint side checked above
    # (whether `default` hands the objective to scipy is decided in solve()).
    for mode in ("gauss_newton", "objective_only"):
        assert np.allclose(objective_hess[mode](np.zeros(M - 1)), T**2 * np.eye(M - 1)), mode


# --------------------------------------------------------------------------
# epigraph (smoke; the sweep is Step 10)
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_epigraph_trades_peak_against_energy_and_keeps_the_hard_constraints():
    """★ lambda != 0 goes through the auxiliary variable and linear inequalities."""
    base = optimize.solve(problem(N_grid=optimize.N_SURVEY), fine_grids=())
    out = {}
    for lam in (0.05, 0.2, 1.0):
        res = optimize.solve(problem(N_grid=optimize.N_SURVEY, lam=lam), fine_grids=())
        out[lam] = res
        assert res.stop_reason == "converged"
        assert res.epigraph_s is not None
        assert res.terms.peak < base.terms.peak  # peak really comes down
        assert res.terms.energy > base.terms.energy - 1e-9  # and energy pays for it
        assert np.max(np.abs(res.equality_residuals)) < 1e-12  # hard constraints hold
        assert res.gate_residual <= 1e-13

    assert out[1.0].terms.peak < out[0.2].terms.peak < out[0.05].terms.peak
    assert out[1.0].terms.peak == pytest.approx(19.44, abs=0.05)


@pytest.mark.slow
def test_epigraph_variable_is_the_peak_up_to_the_peak_grid_resolution():
    """The reported peak sits just above ``T s``; the gap is the coarse peak grid.

    Step 05 asks the report and the constraint to share a grid, which the epigraph
    cannot afford literally (2 N inequality rows at N=20000). Measured shortfall
    at n_peak=400: about 1e-4 relative.
    """
    res = optimize.solve(problem(N_grid=optimize.N_SURVEY, lam=0.2, n_peak=400), fine_grids=())
    ratio = res.terms.peak / (T * res.epigraph_s)
    assert 1.0 <= ratio < 1.001, ratio


def test_lambda_zero_does_not_build_the_epigraph_block():
    p = problem()
    assert not p.uses_epigraph
    assert optimize.manifest_for(p)["objective"]["peak_grid_N"] is None
    assert optimize.manifest_for(p)["objective"]["terms"] == ["energy"]


# --------------------------------------------------------------------------
# early stop
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_early_stop_reports_itself_and_stops_sooner():
    full = optimize.solve(problem(N_grid=optimize.N_SURVEY), fine_grids=())
    stopped = optimize.solve(
        problem(N_grid=optimize.N_SURVEY),
        early_stop=optimize.EarlyStop(tol=1e-3, patience=3),
        fine_grids=(),
    )
    assert stopped.stop_reason == "early_stop"
    assert stopped.void_for_claims
    assert stopped.nit < full.nit
    assert full.stop_reason == "converged"
    assert not full.void_for_claims


# --------------------------------------------------------------------------
# ★ acceptance: a complete RunRecord, and plotting from it alone
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_solve_leaves_a_complete_runrecord_that_plots(tmp_path):
    """★ Every run goes through the recorder; plotting needs nothing else."""
    p = problem(M=12, ansatz={"name": "naive", "source": "builtin"}, theta_before=np.pi)
    run_id = recorder.make_run_id("Xpi", "naive", "M12_lam0_test", date="20260728")
    rec = recorder.Recorder(run_id, optimize.manifest_for(p), root=tmp_path)
    res = optimize.solve(p, recorder=rec)

    loaded = recorder.load(run_id, root=tmp_path)
    assert loaded.n_checkpoints == res.n_checkpoints > 5
    assert loaded.manifest["stop_reason"] == "converged"
    assert loaded.manifest["winding_branch"] == np.pi
    assert loaded.manifest["solver"]["hessian_mode"] == "gauss_newton"
    assert loaded.final["void_for_claims"] is False
    assert loaded.final["fine_grid"]["grids"]["500000"]["max_equality_residual"] <= 1e-8

    # the stored trajectory is self-consistent, bit for bit
    assert recorder.verify_history(loaded, atol=0.0) == {
        "cost_energy": 0.0,
        "cost_curv": 0.0,
        "cost_peak": 0.0,
    }
    # It ends at the recorded optimum -- but the energy *rises* on the way there.
    # ★ The naive start satisfies only the gate (energy 12.18); becoming feasible
    # for closure and zero area is what costs 83.75. A cost history from an
    # infeasible start is therefore not a descent, and any "how fast did it
    # converge" reading has to account for that (_plan.md §8.1).
    energies = np.asarray(loaded.history["cost_energy"], dtype=float)
    assert energies[-1] == pytest.approx(83.7479, abs=5e-4)
    assert energies[0] == pytest.approx(12.176, abs=1e-3)
    assert energies[-1] > energies[0]
    # what does fall monotonically-ish is the constraint residual
    res_closure = np.abs(np.asarray(loaded.history["res_closure"], dtype=float)).max(axis=1)
    assert res_closure[0] > 1e-2
    assert res_closure[-1] < 1e-12

    assert plotting.plot_history(loaded) is not None
    assert plotting.plot_waveform(loaded, checkpoint=0) is not None


@pytest.mark.slow
def test_optimize_itself_writes_nothing(tmp_path):
    """The architecture rule, checked behaviourally as well as by AST.

    Without a recorder the solve leaves no trace on disk at all -- ``_runs/`` is
    untouched, so a caller cannot accidentally produce an unrecorded artefact.
    """
    before = set(recorder.list_runs())
    optimize.solve(problem(N_grid=optimize.N_SURVEY), fine_grids=())
    assert set(recorder.list_runs()) == before


def test_reevaluate_reports_grid_free_energy_and_per_grid_residuals():
    a = basis.naive_coeffs(np.pi, T, 12)
    out = optimize.reevaluate(a, T, grids=(4000, 20000))
    assert out["energy"] == basis.energy_invariant(a, T)
    assert set(out["grids"]) == {"4000", "20000"}
    for entry in out["grids"].values():
        assert entry["closure"] == pytest.approx(0.472, abs=1e-3)


def test_result_advertises_whether_it_may_be_claimed():
    res = optimize.SolveResult(
        coeffs=np.zeros(3),
        stop_reason="maxiter",
        status=0,
        nit=1500,
        wall_clock_s=1.0,
        n_checkpoints=0,
        terms=metrics.CostTerms(0.0, 0.0, 0.0),
        gate_residual=0.0,
        equality_residuals=np.zeros(3),
        epigraph_s=None,
        fine_grid={},
        scipy_message="",
    )
    assert res.void_for_claims
    assert res._replace(stop_reason="converged").void_for_claims is False


# --------------------------------------------------------------------------
# Step 14a: the §2.3 row-7 constraints (bandwidth cap, C1 switch-on/off)
# --------------------------------------------------------------------------


def test_bandwidth_form_matches_numerical_quadrature():
    """The analytic diagonal form against midpoint quadrature of omega_dot^2."""
    rng = np.random.default_rng(7)
    for M in (3, 12):
        a = rng.normal(size=M)
        for T_ in (1.0, 2.5):
            N = 200_000
            t = (np.arange(N) + 0.5) * T_ / N
            numeric = float(np.sum(basis.omega_dot(a, t, T_) ** 2) * (T_ / N)) * T_**3
            assert basis.bandwidth_invariant(a, T_) == pytest.approx(numeric, rel=1e-8)


def test_bandwidth_gradient_and_hessian_are_exact():
    rng = np.random.default_rng(8)
    M = 9
    a = rng.normal(size=M)
    grad = basis.bandwidth_gradient(a, T)
    eps = 1e-6
    for i in range(M):
        step = np.zeros(M)
        step[i] = eps
        fd = (basis.bandwidth_invariant(a + step, T) - basis.bandwidth_invariant(a - step, T)) / (
            2 * eps
        )
        assert grad[i] == pytest.approx(fd, rel=1e-6)
    # the form is quadratic, so the Hessian reproduces it exactly from any point
    H = basis.bandwidth_hessian(M, T)
    assert 0.5 * a @ H @ a == pytest.approx(basis.bandwidth_invariant(a, T), rel=1e-12)


def test_bandwidth_of_the_general_layer_adds_the_two_components():
    rng = np.random.default_rng(9)
    M = 6
    a, b = rng.normal(size=M), rng.normal(size=M)
    both = basis.bandwidth_invariant(np.concatenate([a, b]), T, M=M)
    assert both == pytest.approx(
        basis.bandwidth_invariant(a, T) + basis.bandwidth_invariant(b, T), rel=1e-14
    )


def test_c1_row_reproduces_omega_dot_at_the_ends():
    rng = np.random.default_rng(10)
    M = 11
    a = rng.normal(size=M)
    for T_ in (1.0, 3.0):
        assert basis.c1_row(M, T_, "start") @ a == pytest.approx(
            float(basis.omega_dot(a, [0.0], T_)[0]), rel=1e-12
        )
        assert basis.c1_row(M, T_, "end") @ a == pytest.approx(
            float(basis.omega_dot(a, [T_], T_)[0]), rel=1e-12
        )


def test_c1_rows_coincide_up_to_sign_on_the_odd_subspace():
    """★ The structural fact behind the Step 14a design: on odd harmonics the
    two ends are the same condition, so 'C1 at both ends' costs one dof there."""
    M = 12
    a = np.zeros(M)
    a[0::2] = np.random.default_rng(11).normal(size=M // 2)  # odd harmonics only
    r0, rT = basis.c1_row(M, T, "start"), basis.c1_row(M, T, "end")
    assert r0 @ a == pytest.approx(-(rT @ a), rel=1e-12)


def test_empty_extra_constraints_changes_nothing():
    base = problem(M=8, N_grid=4000, maxiter=1500)
    with_empty = base._replace(extra_constraints=())
    r1 = optimize.solve(base, fine_grids=())
    r2 = optimize.solve(with_empty, fine_grids=())
    assert np.allclose(r1.coeffs, r2.coeffs, atol=0.0, rtol=0.0)


def test_c1_constraint_is_met_exactly_and_costs_energy():
    """C1 is a linear equality: it holds to machine precision, and it is not free."""
    free = optimize.solve(problem(M=12, N_grid=4000), fine_grids=())
    spec = ({"kind": "c1", "ends": ("start", "end")},)
    constrained = optimize.solve(
        problem(M=12, N_grid=4000, extra_constraints=spec), fine_grids=()
    )
    assert constrained.stop_reason == "converged"
    res = optimize.extra_constraint_residuals(
        problem(M=12, extra_constraints=spec), constrained.coeffs
    )
    assert res["c1_0"] < 1e-9
    # the hard equality block is untouched by the new row
    assert np.max(np.abs(constrained.equality_residuals)) < 1e-10
    assert constrained.terms.energy > free.terms.energy


def test_bandwidth_cap_binds_and_the_hard_equalities_survive():
    """★ The cap has a feasibility floor: the minimum bandwidth reachable on the
    gate+closure+area manifold is ``0.884 * B_free`` at M=12 (step14a §2), so a
    cap below that makes the problem infeasible, not merely tight. 0.94 is inside
    the window."""
    free = optimize.solve(problem(M=12, N_grid=4000), fine_grids=())
    free_bw = basis.bandwidth_invariant(free.coeffs, T)
    bound = 0.94 * free_bw
    spec = ({"kind": "bandwidth", "bound": bound},)
    capped = optimize.solve(problem(M=12, N_grid=4000, extra_constraints=spec), fine_grids=())
    assert capped.stop_reason == "converged"
    assert basis.bandwidth_invariant(capped.coeffs, T) <= bound * (1 + 1e-7)
    assert np.max(np.abs(capped.equality_residuals)) < 1e-10
    assert capped.terms.energy > free.terms.energy  # a binding cap costs energy


def test_unknown_extra_constraint_is_refused():
    with pytest.raises(ValueError, match="unknown extra constraint"):
        optimize.solve(problem(M=8, N_grid=2000, extra_constraints=({"kind": "smooth"},)),
                       fine_grids=())


def test_manifest_carries_the_extra_constraints_verbatim():
    spec = ({"kind": "bandwidth", "bound": 1234.5}, {"kind": "c1", "ends": ("start",)})
    man = optimize.manifest_for(problem(extra_constraints=spec))
    assert man["extra_constraints"] == [
        {"kind": "bandwidth", "bound": 1234.5},
        {"kind": "c1", "ends": ("start",)},
    ]
