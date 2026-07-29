"""Mechanical tests for the Step 15 signed-branch experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "step15_gate_branches", ROOT / "_dev_logs" / "step15_gate_branches.py"
)
step15 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(step15)


def test_every_signed_branch_is_the_declared_gate_modulo_global_phase():
    for branch in step15.BRANCHES:
        assert branch.theta == branch.target_angle + 2 * np.pi * branch.branch_k
        assert step15.branch_is_physical_target(branch)


def test_petal_high_member_uses_negative_signed_branch():
    branch = next(b for b in step15.BRANCHES if b.key == "Rxpi4_petal_high")
    assert branch.theta == -7 * np.pi / 4
    assert branch.branch_k == -1
    assert branch.native_members == (("rcp_petal", "7pi/4", True),)


def test_complex_target_matrix_payload_round_trips():
    matrix = step15.target_matrix(np.pi / 2)
    payload = step15.matrix_payload(matrix)
    rebuilt = np.asarray(payload["matrix_real"]) + 1j * np.asarray(payload["matrix_imag"])
    np.testing.assert_allclose(rebuilt, matrix, atol=0, rtol=0)


def test_coefficient_clustering_uses_connected_components():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0008])
    c = np.array([1.0, 0.0016])
    far = np.array([-1.0, 0.0])
    groups = step15.cluster_coefficients([a, b, c, far], rtol=1e-3)
    assert groups == [[0, 1, 2], [3]]


def test_diagnostic_problem_is_always_void_for_claims():
    spec = step15.BRANCHES[0]
    report = step15.ansatz.project_named("naive", M=step15.M, T=step15.T, theta=spec.theta)
    diagnostic = step15.build_problem(spec, report, diagnostic=True)
    claim = step15.build_problem(spec, report, diagnostic=False)
    assert diagnostic.fixed_budget_survey is True
    assert diagnostic.N_grid == step15.optimize.N_SURVEY
    assert claim.fixed_budget_survey is False
    assert claim.N_grid == step15.optimize.N_CLAIM
