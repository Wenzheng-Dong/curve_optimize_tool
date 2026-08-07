"""F05-pre acceptance: :mod:`curve_opt.recorder`'s schema extended to budget mode.

Task brief §3, three criteria:

1. one real small-scale ``solve_budget`` run, recorded through
   ``Recorder(..., schema="budget")``, round-trips bit-for-bit
   (:func:`curve_opt.recorder.verify_history`) and reproduces the four cost
   terms plus the coefficients from disk alone, matching the in-memory
   :class:`curve_opt.optimize.BudgetSolveResult`.
2. the pre-existing ``schema_version`` 1 records (83 at the time of writing)
   are still readable exactly as before, and :mod:`curve_opt.plotting` still
   works on them.
3. the budget manifest carries the whole device, the weights' actual values,
   ``gate_level``, ``layer`` and the pinned upstream hash -- checked by one
   assertion against :data:`curve_opt.recorder.REQUIRED_MANIFEST_KEYS_BUDGET`,
   not eyeballed.

Marked ``slow``: criterion 1 needs an actual ``trust-constr`` call (a handful
of iterations at ``M=6``, per the task brief -- "小规模即可"), not a
publication-grade solve.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from curve_opt import basis, budget, device, novera, optimize, plotting, recorder

DEV = device.DEFAULT_DEVICE
T_PHYS = DEV.gate_time


def _tiny_budget_problem(**kw):
    """M=6, a coarse grid, a handful of iterations -- exercises the schema,
    not a claim (mirrors ``tests/test_f04_budget.py``'s ``_small_budget_problem``,
    but deliberately capped at a low ``maxiter`` so the smoke test stays fast)."""
    M = 6
    a0 = basis.min_norm_gate_only(np.pi, T_PHYS, M)
    c0 = np.zeros(M)
    kw.setdefault("device", DEV)
    kw.setdefault("gate_level", "two_level")
    kw.setdefault("N_grid", 200)
    kw.setdefault("n_peak", 40)
    kw.setdefault("maxiter", 15)
    kw.setdefault("checkpoint_every", 1)
    kw.setdefault("fixed_budget_survey", True)
    return optimize.BudgetProblem(M=M, T=T_PHYS, theta=np.pi, coeffs0_a=a0, coeffs0_c=c0, **kw)


# --------------------------------------------------------------------------
# ★ acceptance 1: a real solve_budget run, full round trip, bit-for-bit
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_a_real_solve_budget_run_round_trips_bit_for_bit(tmp_path):
    problem = _tiny_budget_problem()
    manifest = optimize.budget_manifest_for(problem)
    run_id = recorder.make_run_id("Xpi", "f05pre", "budget_smoke", date="20260807")
    rec = recorder.Recorder(run_id, manifest, root=tmp_path, schema="budget")

    result = optimize.solve_budget(problem, recorder=rec, verbose=0)

    loaded = recorder.load(run_id, root=tmp_path)
    assert loaded.manifest["schema_version"] == 2
    # maxiter=15 with no convergence expected at this scale -- the point of
    # this run is the recorder plumbing, not a claim (fixed_budget_survey=True
    # marks it so, and it is void_for_claims either way per BudgetSolveResult).
    assert loaded.n_checkpoints == problem.maxiter
    assert loaded.manifest["stop_reason"] == result.stop_reason

    # the last checkpoint's coefficients are exactly the solver's final point:
    # trust-constr's callback fires on the same iterate `minimize` returns.
    assert np.array_equal(loaded.history["coeffs_a"][-1], result.coeffs_a)
    assert np.array_equal(loaded.history["coeffs_c"][-1], result.coeffs_c)
    last_cost = {
        "c1": float(loaded.history["cost_c1"][-1]),
        "c2": float(loaded.history["cost_c2"][-1]),
        "c3": float(loaded.history["cost_c3"][-1]),
        "c4": float(loaded.history["cost_c4"][-1]),
    }
    assert last_cost["c1"] == float(result.terms.c1)
    assert last_cost["c2"] == float(result.terms.c2)
    assert last_cost["c3"] == float(result.terms.c3)
    assert last_cost["c4"] == float(result.terms.c4)

    # final.json's redundant final_terms block is exactly the last history row.
    assert loaded.final["final_terms"] == {
        "cost_total": float(loaded.history["cost_total"][-1]),
        "cost_c1": last_cost["c1"],
        "cost_c2": last_cost["c2"],
        "cost_c3": last_cost["c3"],
        "cost_c4": last_cost["c4"],
    }

    # ★ the double-channel check, offline: recompute cost_c1..c4 from the
    # *stored* coefficients + the *stored* device, bit-for-bit against what
    # was live-computed and appended during the solve.
    deviations = recorder.verify_history(loaded, atol=0.0)
    assert deviations == {"cost_c1": 0.0, "cost_c2": 0.0, "cost_c3": 0.0, "cost_c4": 0.0}

    # the extra F05-pre columns are the right shape: res_gate is the 3-component
    # polar-decomposition residual, res_c1_ends the 2-component endpoint residual.
    assert loaded.history["res_gate"].shape == (problem.maxiter, 3)
    assert loaded.history["res_c1_ends"].shape == (problem.maxiter, 2)
    assert loaded.history["epigraph_s"].shape == (problem.maxiter,)
    assert not np.any(np.isnan(loaded.history["epigraph_s"]))

    # the stored device reconstructs the exact Device the run used.
    assert recorder._device_from_manifest(loaded.manifest["device"]) == DEV


@pytest.mark.slow
def test_recorder_refuses_append_budget_on_an_energy_schema_recorder(tmp_path):
    """Guard rail: the two histories do not share a row shape, so mixing them
    must raise rather than silently write the wrong columns."""
    problem = _tiny_budget_problem(maxiter=1)
    manifest = optimize.budget_manifest_for(problem)
    manifest_energy = {
        "ansatz": {},
        "M": 6,
        "T": T_PHYS,
        "target_gate": {"name": "X(pi)"},
        "N_grid": 200,
        "winding_branch": np.pi,
        "objective": {},
        "solver": {},
    }
    rec_energy = recorder.Recorder("bad", manifest_energy, root=tmp_path)
    with pytest.raises(RuntimeError, match="append_budget"):
        rec_energy.append_budget(
            0, np.zeros(6), np.zeros(6),
            cost_total=0.0, cost_c1=0.0, cost_c2=0.0, cost_c3=0.0, cost_c4=0.0,
            Phi_0=0.0, phi_vz=0.0, res_gate=np.zeros(3), res_c1_ends=np.zeros(2),
        )

    rec_budget = recorder.Recorder("good", manifest, root=tmp_path, schema="budget")
    with pytest.raises(RuntimeError, match="append"):
        rec_budget.append(
            0, np.zeros(6), np.zeros(6),
            cost_total=0.0, cost_energy=0.0, cost_curv=0.0, cost_peak=0.0,
            res_gate=0.0, res_closure=np.zeros(2), res_area=np.zeros(1),
        )


# --------------------------------------------------------------------------
# ★ acceptance 2: pre-existing schema_version=1 records still read exactly
# --------------------------------------------------------------------------


def test_three_existing_energy_era_records_are_unaffected():
    run_ids = recorder.list_runs()
    assert len(run_ids) >= 3, "expected the pre-existing _runs/ store to be populated"
    rng = random.Random(0)
    sample = rng.sample(run_ids, 3)

    for run_id in sample:
        rec = recorder.load(run_id)
        assert rec.manifest.get("schema_version", 1) == 1
        assert set(rec.history) == set(recorder.HISTORY_COLUMNS)

        deviations = recorder.verify_history(rec, atol=0.0)
        assert deviations == {"cost_energy": 0.0, "cost_curv": 0.0, "cost_peak": 0.0}

        # plotting.py must still work unmodified on these records.
        assert plotting.plot_history(rec) is not None
        if rec.step0:
            assert plotting.plot_step0(rec) is not None
        assert plotting.plot_waveform(rec, checkpoint=-1) is not None


def test_schema_version_default_is_still_1_for_a_manifest_without_the_key(tmp_path):
    """Records written before ``schema_version`` existed as a field default
    to the energy path -- the same behaviour :func:`verify_history` had
    before F05-pre, now expressed as an explicit dispatch default."""
    man = {
        "ansatz": {},
        "M": 4,
        "T": 1.0,
        "target_gate": {"name": "X(pi)"},
        "N_grid": 100,
        "winding_branch": np.pi,
        "objective": {},
        "solver": {},
    }
    rec = recorder.Recorder("no_version_probe", man, root=tmp_path)
    a = basis.naive_coeffs(np.pi, 1.0, M=4)
    from curve_opt import metrics

    terms = metrics.cost_terms(a, 1.0, 100)
    rec.append(
        0, a, cost_total=terms.energy, cost_energy=terms.energy,
        cost_curv=terms.curv, cost_peak=terms.peak, res_gate=0.0,
        res_closure=np.zeros(2), res_area=np.zeros(1),
    )
    rec.close(status="synthetic", nit=0, wall_clock_s=0.0)
    loaded = recorder.load("no_version_probe", root=tmp_path)
    assert loaded.manifest["schema_version"] == 1
    assert recorder.verify_history(loaded, atol=0.0)["cost_energy"] == 0.0


# --------------------------------------------------------------------------
# ★ acceptance 3: manifest field completeness, asserted not eyeballed
# --------------------------------------------------------------------------


def test_budget_manifest_carries_device_weights_gate_level_layer_and_upstream_hash():
    problem = _tiny_budget_problem(maxiter=1)
    man = optimize.budget_manifest_for(problem)

    missing = [k for k in recorder.REQUIRED_MANIFEST_KEYS_BUDGET if k not in man]
    assert missing == []

    # the whole device, round-tripping to the exact instance used.
    assert man["device"] == DEV.to_manifest()
    assert recorder._device_from_manifest(man["device"]) == DEV

    # the weights' actual numeric values, not a placeholder.
    expected_weights = budget.weights_from_device(DEV)._asdict()
    assert man["objective"]["weights"] == expected_weights
    assert man["objective"]["weights"]["c1"] == pytest.approx(6.5797e-8, rel=1e-3)

    assert man["gate_level"] == problem.gate_level == "two_level"
    assert man["layer"] in ("planar", "planar_drag", "general")
    assert man["novera_git_hash"] == novera.UPSTREAM_GIT_HASH
    assert man["novera_git_hash"] == "59fb6166e105a44064b440865b8fed3415258ad9"


def test_recorder_refuses_a_budget_manifest_missing_a_required_key(tmp_path):
    problem = _tiny_budget_problem(maxiter=1)
    man = optimize.budget_manifest_for(problem)
    del man["device"]
    with pytest.raises(ValueError, match="device"):
        recorder.Recorder("bad_budget_run", man, root=tmp_path, schema="budget")
    assert not (tmp_path / "bad_budget_run").exists()
