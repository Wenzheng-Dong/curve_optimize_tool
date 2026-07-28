"""Step 03 acceptance: the closed forms in :mod:`curve_opt.basis`.

The arbiter for the gate row and the fluence is an independent midpoint
quadrature written locally in this file -- deliberately not
:mod:`curve_opt.geometry`, so that Step 03 stands on its own. Step 04 then
cross-checks geometry against these same closed forms.
"""

from __future__ import annotations

import numpy as np
import pytest

from curve_opt import basis

T_CASES = [1.0, 2.7, 40.0]


def midpoint(f, T, N):
    """int_0^T f(t) dt by the midpoint rule, O(dt^2)."""
    dt = T / N
    t = (np.arange(N) + 0.5) * dt
    return float(np.sum(f(t)) * dt)


def rng(seed=0):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# gate row
# --------------------------------------------------------------------------


@pytest.mark.parametrize("T", T_CASES)
def test_gate_row_closed_form(T):
    M = 12
    g = basis.gate_row(M, T)
    n = np.arange(1, M + 1)
    expected = np.where(n % 2 == 1, 2.0 * T / (n * np.pi), 0.0)
    assert np.allclose(g, expected, rtol=0, atol=0)


def test_even_harmonics_are_free_shape_knobs():
    """Even harmonics must not move the gate at all -- exactly, not approximately."""
    T, M = 1.0, 12
    a = rng().normal(size=M)
    theta0 = basis.gate_angle(a, T)
    a_shaken = a.copy()
    a_shaken[1::2] += 5.0 * rng(1).normal(size=a[1::2].size)  # n = 2, 4, 6, ...
    assert basis.gate_angle(a_shaken, T) == pytest.approx(theta0, abs=1e-15)


@pytest.mark.parametrize("M", [1, 8, 12, 20])
def test_gate_angle_matches_quadrature_at_T1(M):
    """★ Step 03 acceptance: closed form vs numerical theta(T), <= 1e-9 at T = 1.

    T = 1 is the project convention (every step00 script sets ``T = 1.0``), and
    the criterion is stated as an absolute tolerance, so it is checked here at
    that convention. The midpoint error of mode n is ``T n pi / (12 N^2)``,
    which is *proportional to T* -- see the relative-tolerance test below and
    the note in the dev log.
    """
    T = 1.0
    a = rng(2).normal(size=M)
    theta_closed = basis.gate_angle(a, T)
    theta_num = midpoint(lambda t: basis.omega(a, t, T), T, N=200_000)
    assert abs(theta_closed - theta_num) <= 1e-9


@pytest.mark.parametrize("T", T_CASES)
@pytest.mark.parametrize("M", [1, 8, 12, 20])
def test_gate_angle_matches_quadrature_relative(T, M):
    """Scale-free form of the same check: relative agreement is T-independent.

    theta itself grows like T (the gate row is proportional to T), so an
    absolute tolerance is a T-dependent statement while a relative one is not.
    """
    a = rng(2).normal(size=M)
    theta_closed = basis.gate_angle(a, T)
    theta_num = midpoint(lambda t: basis.omega(a, t, T), T, N=200_000)
    assert abs(theta_closed - theta_num) <= 1e-9 * max(1.0, abs(theta_closed))


def test_gate_angle_quadrature_error_is_second_order():
    """The disagreement above is quadrature error, and it must fall like dt^2."""
    T, M = 1.0, 12
    a = rng(3).normal(size=M)
    exact = basis.gate_angle(a, T)
    errs = [abs(midpoint(lambda t: basis.omega(a, t, T), T, N) - exact) for N in (500, 1000, 2000)]
    ratios = [errs[i] / errs[i + 1] for i in range(2)]
    assert all(3.7 < r < 4.3 for r in ratios), (errs, ratios)


# --------------------------------------------------------------------------
# waveform
# --------------------------------------------------------------------------


@pytest.mark.parametrize("T", T_CASES)
def test_endpoints_vanish_identically(T):
    a = rng(4).normal(size=12)
    assert basis.omega(a, [0.0, T], T) == pytest.approx([0.0, 0.0], abs=1e-13)


def test_design_matrix_is_the_linear_map():
    T, M = 2.7, 9
    t = np.linspace(0.0, T, 17)
    a = rng(5).normal(size=M)
    assert np.allclose(basis.design_matrix(t, T, M) @ a, basis.omega(a, t, T), atol=0)


def test_omega_dot_matches_finite_differences():
    T, M = 1.0, 8
    a = rng(6).normal(size=M)
    t = np.linspace(0.05, T - 0.05, 11)
    h = 1e-6
    fd = (basis.omega(a, t + h, T) - basis.omega(a, t - h, T)) / (2 * h)
    assert np.allclose(basis.omega_dot(a, t, T), fd, rtol=1e-7, atol=1e-7)


# --------------------------------------------------------------------------
# energy: orthogonality identity and derivatives
# --------------------------------------------------------------------------


@pytest.mark.parametrize("T", T_CASES)
def test_energy_invariant_equals_quadrature(T):
    """(int Omega^2 dt) * L from orthogonality vs from the grid."""
    a = rng(7).normal(size=12)
    num = midpoint(lambda t: basis.omega(a, t, T) ** 2, T, N=200_000) * T
    assert basis.energy_invariant(a, T) == pytest.approx(num, rel=1e-9)


def test_energy_invariant_covers_both_components_via_concatenation():
    T, M = 1.0, 10
    a, b = rng(8).normal(size=M), rng(9).normal(size=M)
    num = T * (
        midpoint(lambda t: basis.omega(a, t, T) ** 2, T, 200_000)
        + midpoint(lambda t: basis.omega(b, t, T) ** 2, T, 200_000)
    )
    assert basis.energy_invariant(np.concatenate([a, b]), T) == pytest.approx(num, rel=1e-9)


@pytest.mark.parametrize("T", T_CASES)
def test_energy_gradient_and_hessian(T):
    c = rng(10).normal(size=14)
    h = 1e-7
    fd = np.array(
        [
            (
                basis.energy_invariant(c + h * e, T) - basis.energy_invariant(c - h * e, T)
            )
            / (2 * h)
            for e in np.eye(c.size)
        ]
    )
    assert np.allclose(basis.energy_gradient(c, T), fd, rtol=1e-6)
    assert np.allclose(basis.energy_hessian(c.size, T), T**2 * np.eye(c.size), atol=0)


# --------------------------------------------------------------------------
# analytic bounds and the winding branch
# --------------------------------------------------------------------------


def test_naive_baseline_realizes_the_gate_and_hits_the_M1_bound():
    """The normalization reference, checked against its two analytic values."""
    T = 1.0
    a = basis.naive_coeffs(np.pi, T, M=12)
    assert basis.gate_angle(a, T) == pytest.approx(np.pi, abs=1e-14)
    assert basis.energy_invariant(a, T) == pytest.approx(np.pi**4 / 8, rel=1e-14)
    assert basis.energy_invariant(a, T) == pytest.approx(12.176, abs=1e-3)
    assert basis.energy_bound_truncated(np.pi, M=1) == pytest.approx(np.pi**4 / 8, rel=1e-14)


@pytest.mark.parametrize("T", T_CASES)
def test_naive_amplitude_is_T_independent_as_an_invariant(T):
    a = basis.naive_coeffs(np.pi, T, M=1)
    assert a[0] == pytest.approx(np.pi**2 / (2 * T), rel=1e-14)
    assert basis.energy_invariant(a, T) == pytest.approx(np.pi**4 / 8, rel=1e-14)


@pytest.mark.parametrize("T", T_CASES)
def test_min_norm_gate_only_attains_the_truncated_bound(T):
    theta, M = np.pi, 20
    a = basis.min_norm_gate_only(theta, T, M)
    assert basis.gate_angle(a, T) == pytest.approx(theta, rel=1e-14)
    assert basis.energy_invariant(a, T) == pytest.approx(
        basis.energy_bound_truncated(theta, M), rel=1e-13
    )


def test_truncated_bound_decreases_to_the_continuum_bound():
    theta = np.pi
    vals = [basis.energy_bound_truncated(theta, M) for M in (1, 4, 8, 12, 20, 40, 200, 2000)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    assert vals[-1] == pytest.approx(basis.energy_bound(theta), rel=1e-3)
    assert basis.energy_bound(theta) == pytest.approx(9.8696, abs=1e-4)


def test_truncated_bound_reproduces_step00d_numbers():
    """★ Reproduces measured values without re-running any optimization.

    step00d reported 9.97 for the gate-only optimum at M = 40 and 9.87 as the
    M -> infinity bound; the closed form must land on both.
    """
    assert basis.energy_bound_truncated(np.pi, M=40) == pytest.approx(9.97, abs=5e-3)
    assert basis.energy_bound(np.pi) == pytest.approx(9.87, abs=5e-3)


def test_bound_is_a_bound_for_random_gate_feasible_coefficients():
    T, M, theta = 1.0, 12, np.pi
    g = basis.gate_row(M, T)
    r = rng(11)
    for _ in range(200):
        a = r.normal(size=M)
        a += (theta - g @ a) * g / (g @ g)  # project onto the gate hyperplane
        assert basis.gate_angle(a, T) == pytest.approx(theta, rel=1e-12)
        assert basis.energy_invariant(a, T) >= basis.energy_bound_truncated(theta, M) - 1e-12
        assert basis.energy_invariant(a, T) >= basis.energy_bound(theta) - 1e-12


def test_bound_scales_as_theta_squared_across_winding_branches():
    """★ _plan.md §8.3: same gate R_x(pi/2), branches pi/2 and 5 pi/2, 25x apart."""
    low = basis.energy_bound(np.pi / 2)
    high = basis.energy_bound(5 * np.pi / 2)
    assert low == pytest.approx(2.4674, abs=1e-4)
    assert high == pytest.approx(61.685, abs=1e-3)
    assert high / low == pytest.approx(25.0, rel=1e-12)


def test_M_must_be_positive():
    with pytest.raises(ValueError):
        basis.gate_row(0, 1.0)
