"""Step 05 acceptance: the individual cost terms and the naive normalization.

The acceptance criterion is the naive baseline's analytic values
(``int |Omega| dt = pi``, ``energy * L = pi^4 / 8``); the step00d optimum then
pins the two grid-dependent terms against recorded data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from curve_opt import basis, metrics

BEST_COEFFS = Path(__file__).resolve().parents[1] / "_dev_logs" / "best_planar_coeffs.json"


def best(key):
    return np.array(json.loads(BEST_COEFFS.read_text())[key]["coeffs"])


def rng(seed=0):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# ★ acceptance: the naive baseline's analytic values
# --------------------------------------------------------------------------


@pytest.mark.parametrize("T", [1.0, 2.7, 40.0])
def test_naive_analytic_values(T):
    """★ int |Omega| dt = pi, energy * L = pi^4 / 8, T Omega_max = pi^2 / 2."""
    a = basis.naive_coeffs(np.pi, T, M=12)
    terms = metrics.cost_terms(a, T, N=100_000)
    assert terms.curv == pytest.approx(np.pi, abs=1e-6)
    assert terms.energy == pytest.approx(np.pi**4 / 8, rel=1e-13)
    assert terms.peak == pytest.approx(np.pi**2 / 2, rel=1e-8)
    # ...and against the plan's table, at its printed precision
    assert terms.curv == pytest.approx(3.1416, abs=1e-4)
    assert terms.energy == pytest.approx(12.18, abs=5e-3)
    assert terms.peak == pytest.approx(4.93, abs=5e-3)


@pytest.mark.parametrize("theta", [np.pi, np.pi / 2, 5 * np.pi / 2, np.pi / 4])
def test_naive_reference_matches_its_own_numerics_on_every_branch(theta):
    """The analytic reference must equal the numerically evaluated naive mode.

    Run across winding branches (pi/2 and 5 pi/2 are the same gate, different
    branches) because the reference is per branch by construction.
    """
    T = 1.0
    a = basis.naive_coeffs(theta, T, M=8)
    ref = metrics.naive_reference(theta)
    got = metrics.cost_terms(a, T, N=100_000)
    assert got.energy == pytest.approx(ref.energy, rel=1e-13)
    assert got.curv == pytest.approx(ref.curv, abs=1e-6)
    assert got.peak == pytest.approx(ref.peak, rel=1e-8)


def test_naive_normalizes_to_one():
    T = 1.0
    a = basis.naive_coeffs(np.pi, T, M=12)
    norm = metrics.normalize_to_naive(metrics.cost_terms(a, T, N=100_000), np.pi)
    assert norm.energy == pytest.approx(1.0, rel=1e-12)
    assert norm.curv == pytest.approx(1.0, abs=1e-6)
    assert norm.peak == pytest.approx(1.0, rel=1e-8)


def test_naive_reference_scales_as_theta_squared_and_theta():
    """energy ~ theta^2 while curv and peak ~ theta -- the branch dependence."""
    lo, hi = metrics.naive_reference(np.pi / 2), metrics.naive_reference(5 * np.pi / 2)
    assert hi.energy / lo.energy == pytest.approx(25.0, rel=1e-12)
    assert hi.curv / lo.curv == pytest.approx(5.0, rel=1e-12)
    assert hi.peak / lo.peak == pytest.approx(5.0, rel=1e-12)
    assert lo.energy == pytest.approx(basis.energy_bound_truncated(np.pi / 2, M=1), rel=1e-13)


# --------------------------------------------------------------------------
# regression against the recorded optimum
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,energy,curv,peak",
    [("M12", 83.75, 6.638, 22.63), ("M20", 81.77, 6.652, 22.75)],
)
def test_step00d_optimum_reproduces_plan_table_7_1(key, energy, curv, peak):
    """All three terms of §7.1, from the converged coefficients."""
    T = 1.0
    a = best(key)
    terms = metrics.cost_terms(a, T, N=500_000)
    assert terms.energy == pytest.approx(energy, abs=5e-3)
    assert terms.curv == pytest.approx(curv, abs=5e-4)
    assert terms.peak == pytest.approx(peak, abs=5e-3)


def test_robust_solution_pays_curvature_above_the_gate_angle():
    """curv >= |theta|, and the excess is the doubling back that robustness costs.

    The optimum spends 6.638 of total curvature to deliver a gate angle of pi
    (2.11x), which is the geometric statement of the price of closure + zero
    area. Also confirms the kink in ``int |Omega| dt`` is real: the signed field
    crosses zero five times, it is not a hypothetical corner case.
    """
    T = 1.0
    a = best("M12")
    theta = basis.gate_angle(a, T)
    curv = metrics.total_curvature(a, T, N=100_000)
    assert curv > abs(theta)
    assert curv / abs(theta) == pytest.approx(2.11, abs=0.01)

    om = basis.omega(a, np.linspace(0.0, T, 4001), T)
    assert int(np.sum(np.diff(np.sign(om)) != 0)) == 5


def test_curv_equals_theta_when_omega_never_changes_sign():
    """Equality case of the triangle inequality: the naive mode is single-signed."""
    T = 1.0
    a = basis.naive_coeffs(np.pi, T, M=1)
    assert metrics.total_curvature(a, T, N=100_000) == pytest.approx(
        abs(basis.gate_angle(a, T)), abs=1e-6
    )


# --------------------------------------------------------------------------
# smoothness contract: what is smooth, what only looks smooth
# --------------------------------------------------------------------------


def test_energy_is_exact_and_grid_independent():
    """The only smooth term: closed form, identical on any grid, same as basis."""
    T = 1.0
    a = best("M12")
    assert metrics.energy_invariant(a, T) == basis.energy_invariant(a, T)
    assert metrics.cost_terms(a, T, N=100).energy == metrics.cost_terms(a, T, N=100_000).energy


def test_curv_converges_but_not_at_a_clean_second_order():
    """The kink shows up as an erratic convergence rate, hence report-only.

    Measured ratios on the optimum are 4.8 / 7.7 / 4.7 rather than a steady 4:
    the error depends on where the zero crossings fall relative to cell
    boundaries. It still converges, so the reported value is trustworthy at
    N >= 4000 (1e-7), but this term must not be handed to a smooth optimizer.
    """
    T = 1.0
    a = best("M12")
    ref = metrics.total_curvature(a, T, N=1_000_000)
    errs = [abs(metrics.total_curvature(a, T, N) - ref) for N in (1000, 2000, 4000, 8000)]
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    assert errs[-1] < 1e-6


def test_peak_grid_max_under_reports_and_increases_with_N():
    """The grid maximum approaches the continuum maximum from below."""
    T = 1.0
    a = best("M12")
    vals = [metrics.peak(a, T, N) for N in (1000, 4000, 20_000, 200_000)]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    assert vals[-1] - vals[0] < 1e-3  # the shortfall is small, but it has a sign


def test_smoothed_curvature_brackets_the_kinked_one():
    """The surrogate decreases monotonically to int |Omega| dt as delta -> 0."""
    T, N = 1.0, 20_000
    a = best("M12")
    exact = metrics.total_curvature(a, T, N)
    vals = [metrics.smoothed_total_curvature(a, T, N, delta=d) for d in (1e-1, 1e-2, 1e-3, 1e-5)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    assert all(v >= exact for v in vals)
    assert vals[0] <= exact + 1e-1 * T + 1e-12
    assert vals[-1] == pytest.approx(exact, abs=1e-4)


# --------------------------------------------------------------------------
# general (3D) inputs, invariance, floor
# --------------------------------------------------------------------------


def test_magnitude_uses_the_curvature_when_omega_y_is_present():
    """|Omega| = sqrt(Omega_x^2 + Omega_y^2): equal components give sqrt(2) x."""
    T, N, M = 1.0, 20_000, 6
    a = rng(1).normal(size=M)
    terms_planar = metrics.cost_terms(a, T, N)
    terms_3d = metrics.cost_terms(a, T, N, b=a)
    assert terms_3d.curv == pytest.approx(np.sqrt(2) * terms_planar.curv, rel=1e-12)
    assert terms_3d.peak == pytest.approx(np.sqrt(2) * terms_planar.peak, rel=1e-12)
    assert terms_3d.energy == pytest.approx(2.0 * terms_planar.energy, rel=1e-14)


def test_energy_with_b_matches_the_concatenated_closed_form():
    T, M = 1.0, 10
    a, b = rng(2).normal(size=M), rng(3).normal(size=M)
    assert metrics.energy_invariant(a, T, b=b) == pytest.approx(
        basis.energy_invariant(np.concatenate([a, b]), T), rel=1e-15
    )


@pytest.mark.parametrize("T2", [2.7, 40.0])
def test_all_terms_are_scale_invariant(T2):
    a = best("M12")
    N = 100_000
    t1 = metrics.cost_terms(a, 1.0, N)
    t2 = metrics.cost_terms(a * 1.0 / T2, T2, N)
    assert t2.energy == pytest.approx(t1.energy, rel=1e-12)
    assert t2.curv == pytest.approx(t1.curv, rel=1e-12)
    assert t2.peak == pytest.approx(t1.peak, rel=1e-12)


def test_robustness_floor():
    assert metrics.clamp_robustness(1e-17) == metrics.ROBUSTNESS_FLOOR
    assert metrics.clamp_robustness(-1e-17) == metrics.ROBUSTNESS_FLOOR
    assert metrics.clamp_robustness(0.5) == 0.5
    assert metrics.clamp_robustness(-0.5) == 0.5
    assert metrics.clamp_robustness(1e-6, floor=1e-9) == pytest.approx(1e-6)


def test_grid_size_must_be_positive():
    with pytest.raises(ValueError):
        metrics.total_curvature(np.ones(3), 1.0, N=0)
