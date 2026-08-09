"""Step 04 acceptance: the planar geometry chain.

Arbiters, in order of strength:

1. closed forms for constant curvature -- semicircle => X(pi), full circle =>
   identity, with analytic closure and area (the plan's acceptance criterion);
2. the step00d converged optimum, replayed through this chain: energy 83.75 /
   81.77 and residuals at the 1e-9 level must come back out, which tests the
   whole chain against data this module did not produce;
3. grid-refinement order, which is what distinguishes a correct quadrature from
   a grid artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from curve_opt import basis, geometry

BEST_COEFFS = Path(__file__).resolve().parent / "data" / "best_planar_coeffs.json"


def rng(seed=0):
    return np.random.default_rng(seed)


def const_curvature_chain(kappa, T, N):
    """Constant ``Omega_x = kappa`` -- a circle of radius 1/kappa.

    Not representable in the sine basis (endpoints must vanish), which is why it
    enters as samples through :func:`curve_opt.geometry.chain`.
    """
    return geometry.chain(np.full(N, float(kappa)), T)


# --------------------------------------------------------------------------
# structural identities: unit speed, planarity, equation counts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("T", [1.0, 2.7])
def test_tangent_is_unit_speed_hence_L_equals_T(T):
    a = rng(1).normal(size=12)
    Tvec = np.asarray(geometry.tangent(a, T, N=2000))
    assert np.allclose(np.linalg.norm(Tvec, axis=1), 1.0, atol=1e-15)
    arc_length = np.sum(np.linalg.norm(Tvec, axis=1)) * (T / 2000)
    assert arc_length == pytest.approx(T, rel=1e-14)


def test_curve_stays_in_the_yz_plane():
    """r_x = 0 identically => closure is 2 equations, not 3."""
    a = rng(2).normal(size=12)
    r_edge, r_mid = geometry.curve(a, 1.0, N=2000)
    assert np.all(np.asarray(r_edge)[:, 0] == 0.0)
    assert np.all(np.asarray(r_mid)[:, 0] == 0.0)


def test_area_vector_has_only_an_x_component():
    """=> zero area is 1 equation in the planar layer."""
    a = rng(3).normal(size=12)
    area_vec = np.asarray(geometry.area(a, 1.0, N=2000))
    assert np.allclose(area_vec[1:], 0.0, atol=1e-15)


def test_equality_residuals_are_two_closure_plus_one_area():
    a = rng(4).normal(size=12)
    T, N = 1.0, 2000
    res = np.asarray(geometry.equality_residuals(a, T, N))
    assert res.shape == (3,)
    r_end = np.asarray(geometry.closure(a, T, N))
    area_vec = np.asarray(geometry.area(a, T, N))
    assert res[0] == pytest.approx(r_end[1] / T, rel=1e-14)
    assert res[1] == pytest.approx(r_end[2] / T, rel=1e-14)
    assert res[2] == pytest.approx(area_vec[0] / T**2, rel=1e-14)


def test_curvature_is_the_signed_field_omega():
    """kappa = |dTvec/dt| = |Omega_x|, i.e. Omega is the signed curvature."""
    T, N = 1.0, 20000
    a = rng(5).normal(size=8)
    Tvec = np.asarray(geometry.tangent(a, T, N))
    dt = T / N
    speed = np.linalg.norm(np.diff(Tvec, axis=0), axis=1) / dt
    om = np.asarray(geometry.omega_samples(a, T, N))
    om_between = 0.5 * (om[1:] + om[:-1])
    assert np.allclose(speed, np.abs(om_between), atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------
# ★ acceptance: constant curvature closed forms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("T", [1.0, 2.7])
@pytest.mark.parametrize("turns", [0.5, 1.0, 2.0])
def test_constant_curvature_theta_is_exact(T, turns):
    """theta(t) = kappa t is reproduced to machine precision, not to O(dt^2).

    The half-step correction is exact for a constant integrand, so the only
    grid error left in the circle cases is the outer quadrature for r.
    """
    kappa = 2 * np.pi * turns / T
    c = const_curvature_chain(kappa, T, N=5000)
    assert float(c.theta_edge[-1]) == pytest.approx(kappa * T, rel=1e-14)
    t_mid, _ = geometry.midpoint_grid(T, 5000)
    assert np.allclose(np.asarray(c.theta_mid), kappa * t_mid, rtol=1e-13)


@pytest.mark.parametrize("T", [1.0, 2.7])
def test_semicircle_is_X_pi(T):
    """★ Plan acceptance: semicircle => X(pi), with analytic closure 2/pi.

    kappa T = pi, radius R = 1/kappa. The curve is half a circle in the y-z
    plane, so it ends a diameter away: r(T) = (0, 2R, 0), closure/L = 2/pi.
    A semicircle is *not* closed -- it is not first-order robust, which is the
    honest statement about every arc-type ansatz.
    """
    kappa = np.pi / T
    c = const_curvature_chain(kappa, T, N=200_000)
    assert float(c.theta_edge[-1]) == pytest.approx(np.pi, rel=1e-14)

    R = 1.0 / kappa
    r_end = np.asarray(c.closure)
    assert r_end[1] == pytest.approx(2 * R, rel=1e-10)
    assert r_end[2] == pytest.approx(0.0, abs=1e-10 * R)
    assert float(np.linalg.norm(r_end) / T) == pytest.approx(2 / np.pi, rel=1e-10)


@pytest.mark.parametrize("T", [1.0, 2.7])
def test_full_circle_is_identity_and_closes(T):
    """★ Plan acceptance: full circle => identity, closed, area = -1/(2 pi) L^2.

    kappa T = 2 pi gives the gate exp(-i 2 pi sigma_x / 2) = -I, the identity up
    to a global phase. The circle closes (first-order robust) but its area is
    maximally non-zero: for a closed curve ``int r x r_dot dt`` is twice the
    signed enclosed area, i.e. -2 pi R^2 here, so area/L^2 = -1/(2 pi) exactly.
    This value fixes the sign convention of the module.
    """
    kappa = 2 * np.pi / T
    c = const_curvature_chain(kappa, T, N=200_000)
    assert float(c.theta_edge[-1]) == pytest.approx(2 * np.pi, rel=1e-14)
    assert float(np.linalg.norm(c.closure)) / T <= 1e-14
    assert float(c.area[0]) == pytest.approx(-2 * np.pi / kappa**2, rel=1e-10)
    assert float(c.area[0]) / T**2 == pytest.approx(-1 / (2 * np.pi), rel=1e-10)


def test_circle_has_constant_radius():
    """Geometric self-check: every point sits at R from the centre (0, R, 0)."""
    T, kappa, N = 1.0, 2 * np.pi, 20_000
    c = const_curvature_chain(kappa, T, N)
    R = 1.0 / kappa
    r = np.asarray(c.r_mid)
    d = np.linalg.norm(r - np.array([0.0, R, 0.0]), axis=1)
    assert np.allclose(d, R, rtol=1e-8)


# --------------------------------------------------------------------------
# cross-module and quadrature order
# --------------------------------------------------------------------------


@pytest.mark.parametrize("M", [1, 8, 12, 20])
def test_numeric_gate_angle_matches_the_linear_row(M):
    """Step 03/04 cross-check: quadrature theta(T) vs the exact row g . a."""
    T = 1.0
    a = rng(6).normal(size=M)
    assert abs(geometry.gate_angle_numeric(a, T, N=200_000) - basis.gate_angle(a, T)) <= 1e-9


def test_gate_angle_error_is_second_order():
    T, M = 1.0, 12
    a = rng(7).normal(size=M)
    exact = basis.gate_angle(a, T)
    errs = [abs(geometry.gate_angle_numeric(a, T, N) - exact) for N in (500, 1000, 2000)]
    ratios = [errs[i] / errs[i + 1] for i in range(2)]
    assert all(3.7 < r < 4.3 for r in ratios), (errs, ratios)


def test_closure_and_area_are_second_order_in_the_grid():
    """The whole chain, not just the first link, must converge like dt^2."""
    T, M = 1.0, 12
    a = rng(8).normal(size=M)
    ref = np.asarray(geometry.equality_residuals(a, T, N=256_000))
    errs = [
        float(np.linalg.norm(np.asarray(geometry.equality_residuals(a, T, N)) - ref))
        for N in (1000, 2000, 4000)
    ]
    ratios = [errs[i] / errs[i + 1] for i in range(2)]
    assert all(3.6 < r < 4.4 for r in ratios), (errs, ratios)


def test_right_endpoint_rule_is_second_order_on_the_gate_integral_only():
    """Where the quadrature red line does *not* bite, and why.

    ``Omega`` vanishes at both endpoints in this basis, so for the gate integral
    the right-endpoint sum equals the trapezoid sum bit for bit -- it is already
    O(dt^2) there. The first-order failure of _plan.md §6.2 therefore cannot be
    reproduced on the gate; it lives in the curve integrals, see the next test.
    """
    T, M, N = 1.0, 12, 1000
    a = rng(9).normal(size=M)
    dt = T / N
    right = float(np.sum(basis.omega(a, (np.arange(N) + 1.0) * dt, T)) * dt)
    t_edge = np.linspace(0.0, T, N + 1)
    f = basis.omega(a, t_edge, T)
    trapz = float((np.sum(f) - 0.5 * f[0] - 0.5 * f[-1]) * dt)
    assert right == pytest.approx(trapz, abs=1e-15)


def test_first_order_quadrature_loses_an_order_on_the_curve_integrals():
    """★ Numerical discipline 1, localized: the loss is in ``int sin/cos theta``.

    Those integrands do not vanish at the endpoints (``cos theta(0) = 1``), so a
    right-endpoint rule is genuinely O(dt) there while midpoint is O(dt^2). This
    is the mechanism behind the grid artifact of _plan.md §6.2: the residual the
    optimizer minimizes becomes dominated by discretization error, and refining
    the grid cannot repair the rule.
    """
    T, M = 1.0, 12
    a = rng(9).normal(size=M)

    def right_endpoint_closure(N):
        dt = T / N
        om = basis.omega(a, (np.arange(N) + 1.0) * dt, T)
        th = np.cumsum(om) * dt
        return float(np.hypot(np.sum(np.sin(th)) * dt, np.sum(np.cos(th)) * dt) / T)

    ref = geometry.closure_invariant(a, T, N=512_000)
    first = [abs(right_endpoint_closure(N) - ref) for N in (1000, 2000, 4000)]
    mid = [abs(geometry.closure_invariant(a, T, N) - ref) for N in (1000, 2000, 4000)]

    assert all(1.8 < first[i] / first[i + 1] < 2.2 for i in range(2)), first
    assert all(3.8 < mid[i] / mid[i + 1] < 4.2 for i in range(2)), mid
    assert first[-1] / mid[-1] > 100  # ~540x at N = 4000


# --------------------------------------------------------------------------
# regression against step00d (data this module did not produce)
# --------------------------------------------------------------------------


def test_naive_baseline_reproduces_plan_table_7_1():
    """closure/L = 0.472 and |area|/L^2 = 0.295 from _plan.md §7.1.

    The plan's table prints the area magnitude (step00's ``verify_best.py``
    wrapped it in ``abs()``); the signed value in this module's convention is
    negative. See the dev log.
    """
    T = 1.0
    a = basis.naive_coeffs(np.pi, T, M=12)
    assert geometry.gate_angle_numeric(a, T, N=100_000) == pytest.approx(np.pi, abs=1e-9)
    assert geometry.closure_invariant(a, T, N=100_000) == pytest.approx(0.472, abs=5e-4)
    assert abs(geometry.area_invariant(a, T, N=100_000)) == pytest.approx(0.295, abs=5e-4)
    assert geometry.area_invariant(a, T, N=100_000) < 0


@pytest.mark.parametrize(
    "key,M,energy_expected",
    [("M12", 12, 83.75), ("M20", 20, 81.77)],
)
def test_step00d_optimum_replays_through_this_chain(key, M, energy_expected):
    """★ Strongest arbitration available without running an optimization.

    The converged coefficients of step00d are fed through this independent
    implementation; the energy invariant (analytic) and the three residuals
    (this chain) must come back at the values §7.1 recorded on the truth grid.
    """
    T = 1.0
    coeffs = np.array(json.loads(BEST_COEFFS.read_text())[key]["coeffs"])
    assert coeffs.size == M

    assert basis.energy_invariant(coeffs, T) == pytest.approx(energy_expected, abs=5e-3)
    assert basis.gate_angle(coeffs, T) == pytest.approx(np.pi, abs=1e-9)

    res = np.asarray(geometry.equality_residuals(coeffs, T, N=500_000))
    assert np.max(np.abs(res)) <= 1e-7, res
    assert geometry.closure_invariant(coeffs, T, N=500_000) <= 1e-7
    assert abs(geometry.area_invariant(coeffs, T, N=500_000)) <= 1e-7


# --------------------------------------------------------------------------
# invariance and differentiability (what Step 08 will rely on)
# --------------------------------------------------------------------------


def test_invariants_are_scale_invariant_in_T():
    """Rescaling duration with a_n -> a_n T/T' leaves every invariant fixed."""
    a = rng(10).normal(size=12)
    T1, T2 = 1.0, 3.7
    a2 = a * T1 / T2
    N = 20_000
    assert geometry.gate_angle_numeric(a2, T2, N) == pytest.approx(
        geometry.gate_angle_numeric(a, T1, N), rel=1e-12
    )
    assert geometry.closure_invariant(a2, T2, N) == pytest.approx(
        geometry.closure_invariant(a, T1, N), rel=1e-12
    )
    assert geometry.area_invariant(a2, T2, N) == pytest.approx(
        geometry.area_invariant(a, T1, N), rel=1e-12
    )


def test_equality_residuals_are_jax_differentiable():
    """jax.jacrev must produce the (3, M) constraint Jacobian Step 08 needs."""
    T, M, N = 1.0, 12, 2000
    a = rng(11).normal(size=M)
    jac = np.asarray(jax.jacrev(lambda c: geometry.equality_residuals(c, T, N))(a))
    assert jac.shape == (3, M)
    assert np.all(np.isfinite(jac))

    h = 1e-6
    for j in range(0, M, 4):
        e = np.zeros(M)
        e[j] = 1.0
        fd = (
            np.asarray(geometry.equality_residuals(a + h * e, T, N))
            - np.asarray(geometry.equality_residuals(a - h * e, T, N))
        ) / (2 * h)
        assert np.allclose(jac[:, j], fd, rtol=1e-5, atol=1e-8)


def test_grid_size_must_be_positive():
    with pytest.raises(ValueError):
        geometry.midpoint_grid(1.0, 0)
