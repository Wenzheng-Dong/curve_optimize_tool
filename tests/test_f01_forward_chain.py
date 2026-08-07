"""F01 acceptance: tantrix area, leakage amplitude, d(leakage)/d(delta).

Split out from ``test_propagate.py`` (task brief §4 offers either; this file
is the "new" choice, kept separate because it is the full-cost era's own
acceptance gate and pulls in the Novera cross-check machinery
``test_propagate.py`` does not need).

★ Read this file's bottom section (``KNOWN ISSUE``) before trusting a
green run at face value: two of the F01 acceptance criteria (grid convergence
and the Novera cross-check, both at *non-planar* points) are blocked by a
pre-existing bug in :func:`curve_opt.propagate.chain`'s ``U_mid`` construction,
discovered while writing this file. See ``_dev_logs/F01_forward_chain.md`` for
the full diagnosis; the affected tests are marked ``xfail`` with a pointer
back to this comment rather than silently weakened.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from curve_opt import basis, device, propagate
from conftest import requires_novera

DEV = device.DEFAULT_DEVICE
T = DEV.gate_time
DELTA = DEV.delta


def rng(seed: int):
    return np.random.default_rng(seed)


def _fit_order(Ns, errs) -> float:
    """Least-squares log-log slope of *errs* vs *Ns*, returned as a positive order.

    ``err ~ C * N^{-p}`` -- midpoint quadrature (AGENTS.md discipline #1) claims
    ``p = 2``.
    """
    log_n = np.log(np.asarray(Ns, dtype=float))
    log_e = np.log(np.asarray(errs, dtype=float))
    slope, _ = np.polyfit(log_n, log_e, 1)
    return float(-slope)


# --------------------------------------------------------------------------
# Four planar waveforms for the tantrix-area target (criterion (1))
# --------------------------------------------------------------------------


def _naive_single_mode():
    M = 8
    a = basis.naive_coeffs(np.pi, T, M)
    return a, np.zeros(M)


def _multimode_random():
    M = 12
    a = rng(20).normal(size=M) * 0.08
    return a, np.zeros(M)


def _zero_crossing():
    """Explicit sign change of Omega_x at an interior point."""
    M = 6
    a = np.zeros(M)
    a[0], a[1], a[2] = 0.05, -0.09, 0.04
    return a, np.zeros(M)


def _petal_like():
    """A higher-harmonic, multi-lobe shape ("rcp-like" diversity, planar)."""
    M = 10
    a = np.zeros(M)
    a[1] = 0.06
    a[3] = -0.05
    a[5] = 0.03
    a[8] = -0.02
    return a, np.zeros(M)


PLANAR_CASES = {
    "naive_single_mode": _naive_single_mode(),
    "multimode_random": _multimode_random(),
    "zero_crossing": _zero_crossing(),
    "petal_like": _petal_like(),
}


@pytest.mark.parametrize("name", sorted(PLANAR_CASES))
def test_planar_tantrix_area_matches_same_grid_midpoint_theta(name):
    """(1)(a): A_T = [-theta_numeric/2, 0, 0] to <= 1e-12 on the *same* grid.

    theta_numeric is the midpoint-quadrature gate angle (same rule, same grid,
    same dt as tantrix_area's own sum) -- this is an algebraic identity of the
    planar formula (T . Omega = 0 identically), not a physics claim, so it
    should hold at essentially machine precision.
    """
    a, b = PLANAR_CASES[name]
    N = 4000
    om_x, _ = propagate.omega_samples(a, b, T, N)
    theta_numeric = float(np.sum(np.asarray(om_x)) * (T / N))

    At = np.asarray(propagate.tantrix_area_of_coeffs(a, b, T, N=N))
    assert At[0] == pytest.approx(-theta_numeric / 2.0, abs=1e-12)
    assert abs(At[1]) <= 1e-12
    assert abs(At[2]) <= 1e-12


@pytest.mark.parametrize("name", sorted(PLANAR_CASES))
def test_planar_tantrix_area_converges_to_the_exact_gate_angle(name):
    """(1)(b): vs basis.gate_angle (the exact closed form), O(dt^2) convergence.

    theta_numeric itself carries O(dt^2) quadrature error relative to the
    exact theta = g . a, so this comparison converges rather than matching
    exactly at any fixed N -- unlike the same-grid check above.
    """
    a, b = PLANAR_CASES[name]
    theta_exact = basis.gate_angle(a, T)
    target = -theta_exact / 2.0

    Ns = (500, 1000, 2000, 4000)
    errs = [
        abs(float(propagate.tantrix_area_of_coeffs(a, b, T, N=N)[0]) - target) for N in Ns
    ]
    assert all(e > 0 for e in errs), "identically zero error would hide the convergence check"
    order = _fit_order(Ns, errs)
    assert order == pytest.approx(2.0, abs=0.15), (name, Ns, errs, order)


# --------------------------------------------------------------------------
# Leakage amplitude closed form (criterion (2))
# --------------------------------------------------------------------------


def _constant_drive_closed_form(omega0: float, delta: float, T: float):
    """Hand-derived closed form for a constant Omega_x = omega0, Omega_y = 0.

    U_c(s)[1, 0] = -i sin(omega0 s / 2), U_c(s)[1, 1] = cos(omega0 s / 2)
    (the standard single-axis rotation), so

        Lambda_0 = (1/sqrt2) int_0^T omega0 exp(i delta s) (-i sin(omega0 s/2)) ds
        Lambda_1 = (1/sqrt2) int_0^T omega0 exp(i delta s) cos(omega0 s/2) ds

    Both elementary but left to :func:`scipy.integrate.quad` (rtol 1e-12)
    rather than hand-carried through trig identities, since quad is the
    independent arbiter the acceptance criterion asks for.
    """

    def integrand0(s):
        return omega0 * np.exp(1j * delta * s) * (-1j * np.sin(omega0 * s / 2.0))

    def integrand1(s):
        return omega0 * np.exp(1j * delta * s) * np.cos(omega0 * s / 2.0)

    re0, _ = quad(lambda s: integrand0(s).real, 0, T, limit=400, epsabs=1e-13, epsrel=1e-12)
    im0, _ = quad(lambda s: integrand0(s).imag, 0, T, limit=400, epsabs=1e-13, epsrel=1e-12)
    re1, _ = quad(lambda s: integrand1(s).real, 0, T, limit=400, epsabs=1e-13, epsrel=1e-12)
    im1, _ = quad(lambda s: integrand1(s).imag, 0, T, limit=400, epsabs=1e-13, epsrel=1e-12)
    L0 = (re0 + 1j * im0) / np.sqrt(2.0)
    L1 = (re1 + 1j * im1) / np.sqrt(2.0)
    return L0, L1


def test_leakage_amplitude_matches_the_constant_drive_closed_form():
    """(2): constant Omega_x = Omega0 -- hand-derived closed form, quad-verified."""
    omega0 = 0.9 * DEV.rabi_max_rate  # comfortably inside the hardware ceiling
    L0_exact, L1_exact = _constant_drive_closed_form(omega0, DELTA, T)

    N = 40_000
    om_x = np.full(N, omega0)
    om_y = np.zeros(N)
    lam = np.asarray(propagate.leakage_amplitude_of_samples(om_x, om_y, T, DELTA))
    L0 = lam[0] + 1j * lam[1]
    L1 = lam[2] + 1j * lam[3]
    assert abs(L0 - L0_exact) < 5e-8
    assert abs(L1 - L1_exact) < 5e-8


def test_leakage_amplitude_constant_drive_converges_second_order():
    omega0 = 0.9 * DEV.rabi_max_rate
    L0_exact, _ = _constant_drive_closed_form(omega0, DELTA, T)

    Ns = (500, 1000, 2000, 4000)
    errs = []
    for N in Ns:
        om_x = np.full(N, omega0)
        om_y = np.zeros(N)
        lam = np.asarray(propagate.leakage_amplitude_of_samples(om_x, om_y, T, DELTA))
        errs.append(abs((lam[0] + 1j * lam[1]) - L0_exact))
    order = _fit_order(Ns, errs)
    assert order == pytest.approx(2.0, abs=0.05), (Ns, errs, order)


def test_leakage_amplitude_matches_quad_for_a_single_sine_mode():
    """(2): second independent target -- one sine harmonic, via quad(rtol<=1e-12).

    Here U_c(s) does not have an elementary closed form (Omega_x(s) is not
    constant), so the "closed form" being checked against is the propagator
    truth read off :func:`propagate.chain` at a very fine grid, and quad
    supplies the outer integral independently at rtol <= 1e-12.
    """
    M = 1
    amplitude = 0.7 * DEV.rabi_max_rate
    a = np.array([amplitude])
    N_fine = 40_000
    om_x_fine, om_y_fine = (np.asarray(x) for x in propagate.omega_samples(a, np.zeros(M), T, N_fine))
    c = propagate.chain(om_x_fine, om_y_fine, T)
    t_mid_fine = (np.arange(N_fine) + 0.5) * (T / N_fine)
    u10_fine = np.asarray(c.U_mid[:, 1, 0])

    def integrand_re(s):
        omega = amplitude * np.sin(np.pi * s / T)
        u10 = np.interp(s, t_mid_fine, u10_fine.real) + 1j * np.interp(s, t_mid_fine, u10_fine.imag)
        return (omega * np.exp(1j * DELTA * s) * u10).real

    def integrand_im(s):
        omega = amplitude * np.sin(np.pi * s / T)
        u10 = np.interp(s, t_mid_fine, u10_fine.real) + 1j * np.interp(s, t_mid_fine, u10_fine.imag)
        return (omega * np.exp(1j * DELTA * s) * u10).imag

    re0, _ = quad(integrand_re, 0, T, limit=400, epsabs=1e-13, epsrel=1e-12)
    im0, _ = quad(integrand_im, 0, T, limit=400, epsabs=1e-13, epsrel=1e-12)
    L0_quad = (re0 + 1j * im0) / np.sqrt(2.0)

    lam = np.asarray(propagate.leakage_amplitude_of_coeffs(a, np.zeros(M), T, DELTA, N=N_fine))
    L0_ours = lam[0] + 1j * lam[1]
    # quad integrates a linearly-interpolated u10, so this is bounded by the
    # interpolation error on top of the (already tiny at N_fine) quadrature error.
    assert abs(L0_ours - L0_quad) < 1e-6


# --------------------------------------------------------------------------
# Grid convergence (criterion (3)) -- planar/collinear inputs only. See the
# module docstring's KNOWN ISSUE: non-planar convergence is blocked upstream.
# --------------------------------------------------------------------------


def test_tantrix_area_converges_second_order_on_a_planar_case():
    a, b = _petal_like()
    theta_exact = basis.gate_angle(a, T)
    target = np.array([-theta_exact / 2.0, 0.0, 0.0])
    Ns = (500, 1000, 2000, 4000)
    errs = [
        float(np.linalg.norm(np.asarray(propagate.tantrix_area_of_coeffs(a, b, T, N=N)) - target))
        for N in Ns
    ]
    order = _fit_order(Ns, errs)
    assert order == pytest.approx(2.0, abs=0.15), (Ns, errs, order)


def test_leakage_amplitude_converges_second_order_on_a_collinear_case():
    """Collinear (Omega_y = ratio * Omega_x) drives commute, same as planar."""
    M = 8
    a = rng(30).normal(size=M) * 0.05
    b = 0.6 * a
    ref = np.asarray(propagate.leakage_amplitude_of_coeffs(a, b, T, DELTA, N=200_000))
    Ns = (1000, 2000, 4000, 8000)
    errs = [
        float(np.linalg.norm(np.asarray(propagate.leakage_amplitude_of_coeffs(a, b, T, DELTA, N=N)) - ref))
        for N in Ns
    ]
    order = _fit_order(Ns, errs)
    assert order == pytest.approx(2.0, abs=0.15), (Ns, errs, order)


def test_leakage_amplitude_ddelta_converges_second_order_on_a_planar_case():
    a, b = _petal_like()
    ref = np.asarray(propagate.leakage_amplitude_ddelta_of_coeffs(a, b, T, DELTA, N=200_000))
    Ns = (1000, 2000, 4000, 8000)
    errs = [
        float(
            np.linalg.norm(
                np.asarray(propagate.leakage_amplitude_ddelta_of_coeffs(a, b, T, DELTA, N=N)) - ref
            )
        )
        for N in Ns
    ]
    order = _fit_order(Ns, errs)
    assert order == pytest.approx(2.0, abs=0.15), (Ns, errs, order)


@pytest.mark.xfail(
    reason=(
        "KNOWN UPSTREAM BUG (not F01's formulas): propagate.chain()'s U_mid = "
        "U_left @ half_step has the multiplication order reversed relative to "
        "this project's own 'newest on the left' convention (see the module "
        "docstring and _dev_logs/F01_forward_chain.md). Invisible for "
        "commuting (planar/collinear) fields -- every other test in this file "
        "uses those -- but for genuinely non-commuting drives it degrades "
        "U_mid, and everything built on it (tangent, area, closure, and this "
        "function), from O(dt^2) to O(dt). Verified fix: swap to "
        "`half_step @ U_left`; not applied here because it touches pre-F01, "
        "widely depended-on code outside this step's scope. Reported to "
        "leader, not patched unilaterally."
    ),
    strict=True,
)
def test_tantrix_area_convergence_on_a_genuinely_noncommuting_case():
    M = 8
    a = rng(31).normal(size=M) * 3.0
    b = rng(32).normal(size=M) * 2.0
    assert propagate.noncommutativity(a, b, T, 2000) > 0.0  # genuinely non-planar
    ref = np.asarray(propagate.tantrix_area_of_coeffs(a, b, T, N=200_000))
    Ns = (1000, 2000, 4000, 8000)
    errs = [
        float(np.linalg.norm(np.asarray(propagate.tantrix_area_of_coeffs(a, b, T, N=N)) - ref))
        for N in Ns
    ]
    order = _fit_order(Ns, errs)
    assert order == pytest.approx(2.0, abs=0.15), (Ns, errs, order)


# --------------------------------------------------------------------------
# Gradients (criterion (4)) -- must be at a non-planar, non-commuting point.
# jax.jacrev differentiates whatever chain() computes exactly, so this is
# unaffected by the U_mid ordering bug above (it checks AD correctness, not
# physical accuracy of the underlying approximation).
# --------------------------------------------------------------------------


def _noncommuting_point(M=6):
    a = rng(40).normal(size=M) * 3.0
    b = rng(41).normal(size=M) * 2.0
    assert propagate.noncommutativity(a, b, T, 500) > 0.0
    return a, b


def _central_difference_jacobian(f, x0, h=1e-6):
    x0 = np.asarray(x0, dtype=float)
    out0 = np.asarray(f(x0))
    jac = np.zeros((out0.size, x0.size))
    for j in range(x0.size):
        e = np.zeros_like(x0)
        e[j] = h
        plus = np.asarray(f(x0 + e)).ravel()
        minus = np.asarray(f(x0 - e)).ravel()
        jac[:, j] = (plus - minus) / (2 * h)
    return jac


def test_tantrix_area_jacobian_matches_finite_differences_at_a_noncommuting_point():
    import jax

    a, b = _noncommuting_point()
    M = a.shape[0]
    N = 500

    def f(x):
        return propagate.tantrix_area_of_coeffs(x[:M], x[M:], T, N=N)

    x0 = np.concatenate([a, b])
    jac_ad = np.asarray(jax.jacrev(f)(x0))
    jac_fd = _central_difference_jacobian(f, x0)
    assert jac_ad.shape == (3, 2 * M)
    assert np.allclose(jac_ad, jac_fd, rtol=1e-5, atol=1e-7)


def test_leakage_amplitude_jacobian_matches_finite_differences_at_a_noncommuting_point():
    import jax

    a, b = _noncommuting_point()
    M = a.shape[0]
    N = 500

    def f(x):
        return propagate.leakage_amplitude_of_coeffs(x[:M], x[M:], T, DELTA, N=N)

    x0 = np.concatenate([a, b])
    jac_ad = np.asarray(jax.jacrev(f)(x0))
    jac_fd = _central_difference_jacobian(f, x0)
    assert jac_ad.shape == (4, 2 * M)
    assert np.allclose(jac_ad, jac_fd, rtol=1e-5, atol=1e-7)


def test_leakage_amplitude_ddelta_jacobian_matches_finite_differences_at_a_noncommuting_point():
    import jax

    a, b = _noncommuting_point()
    M = a.shape[0]
    N = 500

    def f(x):
        return propagate.leakage_amplitude_ddelta_of_coeffs(x[:M], x[M:], T, DELTA, N=N)

    x0 = np.concatenate([a, b])
    jac_ad = np.asarray(jax.jacrev(f)(x0))
    jac_fd = _central_difference_jacobian(f, x0)
    assert jac_ad.shape == (4, 2 * M)
    assert np.allclose(jac_ad, jac_fd, rtol=1e-5, atol=1e-6)


def test_leakage_amplitude_ddelta_matches_jax_grad_of_leakage_amplitude():
    """(4), second half: analytic d/d(delta) vs jax.grad, <= 1e-10.

    Checked at a non-planar point too -- both differentiate the same smooth
    integrand (one symbolically, one by autodiff), so this equality does not
    depend on chain()'s approximation accuracy at all, only on internal
    consistency of the two derivative routes.
    """
    import jax

    a, b = _noncommuting_point()
    N = 500

    def lam_j(delta, j):
        return propagate.leakage_amplitude_of_coeffs(a, b, T, delta, N=N)[j]

    analytic = np.asarray(propagate.leakage_amplitude_ddelta_of_coeffs(a, b, T, DELTA, N=N))
    for j in range(4):
        ad = float(jax.grad(lam_j, argnums=0)(DELTA, j))
        assert ad == pytest.approx(float(analytic[j]), abs=1e-10)


# --------------------------------------------------------------------------
# Novera cross-check (criterion (5)) -- planar and non-commuting, each with
# grid refinement. Bypasses to_physical entirely (task brief §2.4): pulse
# dicts are built directly in *this project's* physical units and fed to
# db_helpers.ideal_propagators, which does no renormalization of its own.
# --------------------------------------------------------------------------


def _novera_style_amplitude(om_x_fn, om_y_fn, T, delta, N):
    import db_helpers  # noqa: PLC0415 -- only importable when requires_novera passes

    t = np.linspace(0.0, T, N + 1)
    om_x = om_x_fn(t)
    om_y = om_y_fn(t)
    pulse = {
        "time_over_Tg": t,  # physical ns here, not the [0, 1] normalized convention
        "Tg_omega_x": om_x,  # physical rad/ns
        "Tg_omega_y": om_y,
        "Tg_delta": np.zeros_like(t),
    }
    u10 = db_helpers.ideal_propagators(pulse)[:, 1, 0]
    drive = om_x + 1j * om_y
    integrand = drive * np.exp(1j * delta * t) * u10
    return -1j / np.sqrt(2.0) * np.trapezoid(integrand, t)


@requires_novera
@pytest.mark.parametrize(
    "name,a,b",
    [
        ("planar_naive_xpi", basis.naive_coeffs(np.pi, T, 12), np.zeros(12)),
    ],
)
def test_novera_cross_check_planar(name, a, b):
    """novera_amplitude == -1j * Lambda_0(ours): magnitude + phase(-pi/2) + O(dt^2)."""

    def om_x_fn(t):
        return basis.omega(a, t, T)

    def om_y_fn(t):
        return basis.omega(b, t, T)

    prev_diff = None
    for N in (2000, 8000):  # 4x refinement
        novera_amp = _novera_style_amplitude(om_x_fn, om_y_fn, T, DELTA, N)
        lam = np.asarray(propagate.leakage_amplitude_of_coeffs(a, b, T, DELTA, N=N))
        L0 = lam[0] + 1j * lam[1]
        predicted = -1j * L0
        rel_mag = abs(abs(novera_amp) - abs(predicted)) / abs(predicted)
        phase_diff = float(np.angle(novera_amp / L0))
        diff = abs(novera_amp - predicted)
        assert rel_mag < 2e-3
        assert phase_diff == pytest.approx(-np.pi / 2.0, abs=1e-4)
        if prev_diff is not None:
            assert diff < prev_diff / 3.0  # expect ~16x at 4x refinement; 3x is a safe floor
        prev_diff = diff


@requires_novera
def test_novera_cross_check_noncommuting_reports_the_known_degradation():
    """Same comparison on a non-commuting drive.

    ★ Per the module docstring's KNOWN ISSUE, the *difference* between our
    scheme and Novera's does not shrink at the same 4x-per-4x-refinement rate
    here, because our own U_mid carries the pre-existing O(dt) bug on
    non-commuting inputs -- Novera's own edge-averaged scheme does not share
    that bug (confirmed independently in the diagnosis). This test does not
    assert O(dt^2) convergence; it only pins the leading-order agreement
    (magnitude, phase) that does not depend on it, and prints the residual so
    a future fix's effect is visible in the pytest output.
    """
    M = 12
    a = rng(50).normal(size=M) * 3.0
    b = rng(51).normal(size=M) * 2.0
    assert propagate.noncommutativity(a, b, T, 2000) > 1.0

    def om_x_fn(t):
        return basis.omega(a, t, T)

    def om_y_fn(t):
        return basis.omega(b, t, T)

    N = 8000
    novera_amp = _novera_style_amplitude(om_x_fn, om_y_fn, T, DELTA, N)
    lam = np.asarray(propagate.leakage_amplitude_of_coeffs(a, b, T, DELTA, N=N))
    L0 = lam[0] + 1j * lam[1]
    predicted = -1j * L0
    rel_mag = abs(abs(novera_amp) - abs(predicted)) / abs(predicted)
    phase_diff = float(np.angle(novera_amp / L0))
    print(f"\n[F01 known-issue] non-commuting Novera cross-check at N={N}: "
          f"rel_mag_diff={rel_mag:.3e} phase_diff={phase_diff:.6f} "
          f"(target -pi/2={-np.pi/2:.6f})")
    # Loose bounds: leading-order agreement only, see the docstring above.
    assert rel_mag < 0.05
    assert phase_diff == pytest.approx(-np.pi / 2.0, abs=0.05)
