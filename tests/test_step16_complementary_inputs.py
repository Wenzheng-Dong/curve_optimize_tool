"""Mechanical tests for the Step 16 complementary-input experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "_dev_logs" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


step16 = load_module("step16_complementary_inputs", "step16_complementary_inputs.py")
analysis = load_module("step16_analyse", "step16_analyse.py")


def test_comparison_group_contains_two_complements_two_rcp_and_one_baseline():
    roles = [spec.role for spec in step16.STARTS]
    assert roles.count("complementary_raw") == 1
    assert roles.count("complementary_projected") == 1
    assert roles.count("gate_near_target") == 2
    assert roles.count("baseline") == 1
    assert [spec.name for spec in step16.STARTS] == [
        "alpha_3d",
        "tilted_lemniscate",
        "rcp_lemniscate",
        "rcp_petal",
        "naive",
    ]


def test_every_problem_uses_the_identical_step12_general_layer_triple():
    report = step16.ansatz.project_named(
        "naive", M=step16.M, T=step16.T, theta=step16.THETA
    )
    for spec in step16.STARTS:
        intake = {
            f"projected_{key}_residual": 0.0
            for key in ("gate", "closure", "area", "max")
        }
        problem = step16.build_problem(spec, report, intake)
        assert problem.layer == "general"
        assert problem.N_grid == step16.optimize.N_CLAIM_3D
        assert problem.maxiter == 6000
        assert problem.hessian_mode == "gauss_newton"
        assert problem.fixed_budget_survey is False
        np.testing.assert_array_equal(problem.coeffs0_b, np.zeros(step16.M))


def test_first_reach_is_horizontal_and_right_censored():
    iterations = np.array([0, 5, 10, 15])
    values = np.array([1.0, 1e-2, 1e-6, 1e-10])
    assert analysis.first_reach(iterations, values, 1e-3) == 10
    assert analysis.first_reach(iterations, values, 1e-9) == 15
    assert analysis.first_reach(iterations, values, 1e-12) is None


def test_offline_analysis_has_no_optimizer_import_or_solve_call():
    source = (ROOT / "_dev_logs" / "step16_analyse.py").read_text()
    assert "from curve_opt import optimize" not in source
    assert "optimize.solve" not in source
