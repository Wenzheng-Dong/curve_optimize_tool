"""F02 acceptance: parametrization.py's three design-control layers + DRAG/Stark readout.

Four acceptance criteria (task brief §3) plus one structural self-check (task brief's "另加一条"):

 (1) Delta -> infinity recovers the identity readout map, E_out -> kappa exp(i Phi).
 (2) A c1-violating ``a`` hard-errors through the DRAG layers (not a silent nonzero endpoint);
     a c1-satisfying ``a`` gives an endpoint <= 1e-12.
 (3) kappa -> -kappa, Phi -> Phi + pi leaves the broadcast waveform invariant to <= 1e-14.
 (4) Gaussian + DRAG (raw samples, tau = 0) reproduces the textbook Omega_y = -Omega_x_dot /
     Delta and the Stark-shift Phi_dot_prog = -kappa^2 / (2 Delta), checked against a reference
     computed independently in this file (not by calling back into parametrization.py's own
     formula).
 (structural) ``general_waveform`` at ``c = 0, Phi_0 = 0`` matches ``planar_drag_waveform`` bit
     for bit (<= 1e-14) -- the three layers share one code path.

★ (5, added 2026-08-07, F05-redo) A *non-circular* check of the Stark-reprogramming sign
    (AGENTS.md's "quality guardrails" §5th entry, the postmortem of the 2026-08-07 sign bug).
    Criterion (4) above is a *circular* check for this particular sign: its reference formula
    was transcribed from the same plan-text expression that the implementation was transcribed
    from, so when that text carried the wrong sign, both sides were wrong identically and the
    test stayed green. ``test_stark_coefficient_scan_is_minimized_at_the_derived_sign`` below
    instead asks a purely physical question that no transcription error can make pass
    spuriously: run the *actual* three-level propagator on a naive-X(pi)+DRAG waveform, scan
    an independent (beta, Stark-coefficient) grid, and assert the zero-noise total infidelity
    is minimized at the theoretically derived point by a wide margin over its neighbours.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from curve_opt import basis, device, gate, parametrization, propagate

DEV = device.DEFAULT_DEVICE
T = DEV.gate_time
DELTA = DEV.delta
M = DEV.n_modes


def rng(seed: int):
    return np.random.default_rng(seed)


def _project_onto_c1(a, T: float) -> np.ndarray:
    """Project *a* onto the subspace where both :func:`parametrization.c1_residuals` vanish.

    Least-norm correction: solve the 2xM system ``R a_proj = 0`` for the component of *a*
    along ``R``'s row space, where ``R`` stacks ``basis.c1_row(..., 'start')`` and
    ``basis.c1_row(..., 'end')``. Used to build a *satisfying* test case for acceptance (2).
    """
    a = np.asarray(a, dtype=float)
    Mloc = a.shape[0]
    R = np.stack([basis.c1_row(Mloc, T, "start"), basis.c1_row(Mloc, T, "end")])
    correction = R.T @ np.linalg.solve(R @ R.T, R @ a)
    return a - correction


# --------------------------------------------------------------------------
# (1) Delta -> infinity recovers the identity map
# --------------------------------------------------------------------------


def test_delta_to_infinity_recovers_identity_readout():
    """E_out -> kappa exp(i Phi) as Delta -> infinity, isolated from grid quadrature error.

    Comparing against the *closed-form* ``curve.Phi`` would also pick up the O(dt^2) error of
    the midpoint-cumulative integral of ``tau`` (a separate, already-accepted discretization
    error, not the Delta -> infinity claim under test). Instead the reference is built through
    the *same* midpoint-cumulative machinery with the DRAG/Stark corrections set to their exact
    Delta -> infinity limit (kappa_y = 0, Phi_dot_prog = tau), so the only thing left in the
    residual is the 1/Delta correction itself.
    """
    a = _project_onto_c1(rng(1).normal(size=M) * 0.02, T)
    c = rng(2).normal(size=M) * 0.01
    Phi_0 = 0.37

    N = 2000
    curve = parametrization.design_curve(a, c, Phi_0, T, N)

    delta_huge = 1.0e14  # rad/ns -- unphysical, but this is a math-limit check on the formula
    Ox, Oy = parametrization.readout_waveform(
        curve.kappa, curve.kappa_dot, curve.tau, Phi_0, delta_huge, T
    )
    E_out = np.asarray(Ox) + 1j * np.asarray(Oy)

    Phi_prog_identity = Phi_0 + parametrization._midpoint_cumulative(curve.tau, curve.dt)
    E_identity = np.asarray(curve.kappa) * np.exp(1j * np.asarray(Phi_prog_identity))

    resid = np.max(np.abs(E_out - E_identity))
    print(f"F02 (1) Delta->inf residual: {resid:.3e}")
    assert resid <= 1e-12


# --------------------------------------------------------------------------
# (2) c1 guard: hard error when violated, <= 1e-12 endpoint when satisfied
# --------------------------------------------------------------------------


def test_c1_violation_hard_errors_not_silently_nonzero():
    a_bad = rng(3).normal(size=M) * 0.02
    r0, rT = parametrization.c1_residuals(a_bad, T)
    print(f"F02 (2) violating a: kappa_dot(0)={r0:.3e}, kappa_dot(T)={rT:.3e}")
    assert abs(r0) > 1e-6 and abs(rT) > 1e-6  # generic random a is nowhere near c1

    # the physical consequence: the broadcast Omega_y(0) this would produce is not zero.
    kappa_y0_would_be = -r0 / DELTA
    assert abs(kappa_y0_would_be) > 1e-6

    with pytest.raises(ValueError, match="c1"):
        parametrization.planar_drag_waveform(a_bad, T, DELTA)
    with pytest.raises(ValueError, match="c1"):
        parametrization.general_waveform(a_bad, np.zeros(M), 0.0, T, DELTA)


def test_c1_satisfied_gives_endpoint_at_most_1e_minus_12():
    a_good = _project_onto_c1(rng(4).normal(size=M) * 0.02, T)
    r0, rT = parametrization.c1_residuals(a_good, T)
    print(f"F02 (2) satisfying a: kappa_dot(0)={r0:.3e}, kappa_dot(T)={rT:.3e}")

    # does not raise
    parametrization.check_c1(a_good, T)

    kappa_y0 = -r0 / DELTA
    kappa_yT = -rT / DELTA
    print(f"F02 (2) endpoint kappa_y: t=0 -> {kappa_y0:.3e}, t=T -> {kappa_yT:.3e}")
    assert abs(kappa_y0) <= 1e-12
    assert abs(kappa_yT) <= 1e-12

    # end-to-end through the guarded layer entry point (does not raise)
    Ox, Oy = parametrization.planar_drag_waveform(a_good, T, DELTA, N=2000)
    assert Ox.shape == Oy.shape == (2000,)


# --------------------------------------------------------------------------
# (3) kappa -> -kappa, Phi -> Phi + pi invariance
# --------------------------------------------------------------------------


def test_kappa_sign_phi_shift_pi_invariance():
    a = _project_onto_c1(rng(5).normal(size=M) * 0.02, T)
    c = rng(6).normal(size=M) * 0.01
    Phi_0 = -1.1
    N = 2000

    Ox1, Oy1 = parametrization.general_waveform(a, c, Phi_0, T, DELTA, N=N)
    Ox2, Oy2 = parametrization.general_waveform(-a, c, Phi_0 + np.pi, T, DELTA, N=N)

    diff = np.max(np.abs(np.asarray(Ox1) - np.asarray(Ox2)))
    diff = max(diff, np.max(np.abs(np.asarray(Oy1) - np.asarray(Oy2))))
    print(f"F02 (3) kappa -> -kappa, Phi -> Phi+pi max diff: {diff:.3e}")
    assert diff <= 1e-14


# --------------------------------------------------------------------------
# (4) Gaussian + DRAG reproduces the textbook planar formulas
# --------------------------------------------------------------------------


def test_gaussian_drag_reproduces_textbook_quadrature_and_stark_phase():
    N = 2000
    t_mid = (np.arange(N) + 0.5) * (T / N)
    t0, sigma = T / 2.0, T / 8.0
    amplitude = DEV.rabi_max_rate

    kappa = amplitude * np.exp(-((t_mid - t0) ** 2) / (2.0 * sigma**2))
    kappa_dot_ref = -((t_mid - t0) / sigma**2) * kappa  # exact analytic Gaussian derivative
    tau = np.zeros(N)

    kappa_y, phi_dot_prog = parametrization.drag_stark_quadratures(kappa, kappa_dot_ref, tau, DELTA)

    # independent reference formulas, written directly here (not via parametrization.py)
    omega_y_textbook = -kappa_dot_ref / DELTA
    stark_phase_rate_textbook = -(kappa**2) / (2.0 * DELTA)

    err_quadrature = np.max(np.abs(np.asarray(kappa_y) - omega_y_textbook))
    err_stark = np.max(np.abs(np.asarray(phi_dot_prog) - stark_phase_rate_textbook))
    print(f"F02 (4) DRAG quadrature max err: {err_quadrature:.3e}")
    print(f"F02 (4) Stark phase-rate max err: {err_stark:.3e}")
    assert err_quadrature <= 1e-12
    assert err_stark <= 1e-12


# --------------------------------------------------------------------------
# structural self-check: general at Phi = 0 == planar_drag, bit for bit
# --------------------------------------------------------------------------


def test_general_layer_at_phi_zero_matches_planar_drag_layer():
    a = _project_onto_c1(rng(7).normal(size=M) * 0.02, T)
    N = 2000

    Ox_pd, Oy_pd = parametrization.planar_drag_waveform(a, T, DELTA, N=N)
    Ox_g, Oy_g = parametrization.general_waveform(a, np.zeros(M), 0.0, T, DELTA, N=N)

    diff = max(
        np.max(np.abs(np.asarray(Ox_pd) - np.asarray(Ox_g))),
        np.max(np.abs(np.asarray(Oy_pd) - np.asarray(Oy_g))),
    )
    print(f"F02 (structural) general(Phi=0) vs planar_drag max diff: {diff:.3e}")
    assert diff <= 1e-14


# --------------------------------------------------------------------------
# internal sanity (not a formal F02 acceptance criterion): the analytic
# kappa_dot must be the exact derivative of kappa, checked against a plain
# central finite difference -- guards the "kappa_dot must be analytic, never
# a finite difference" redline by cross-checking the analytic formula itself.
# --------------------------------------------------------------------------


def test_kappa_dot_analytic_matches_finite_difference_sanity():
    a = rng(8).normal(size=M) * 0.02
    N = 4000
    t_mid, dt = (np.arange(N) + 0.5) * (T / N), T / N

    kappa_dot_analytic = np.asarray(parametrization.kappa_dot_of_coeffs(a, T, N))

    S = basis.design_matrix(t_mid, T, M)
    kappa = S @ a
    kappa_dot_fd = np.gradient(kappa, dt)

    err = np.max(np.abs(kappa_dot_analytic - kappa_dot_fd))
    print(f"F02 (sanity) kappa_dot analytic vs finite-difference max err: {err:.3e}")
    assert err <= 1e-4  # np.gradient is only O(dt^2) accurate; loose bound by design


# --------------------------------------------------------------------------
# (5) ★ non-circular Stark-sign check (2026-08-07, F05-redo; AGENTS.md's 5th
# quality guardrail: verification must not be circular, and must cover every
# channel the correction touches). Deliberately does *not* reuse
# drag_stark_quadratures's kappa_y/Phi_dot_prog split with a single "delta" --
# beta (DRAG amplitude scale) and the Stark coefficient are varied
# *independently* here, which the module's own function does not expose, so
# there is no way this test's grid construction can silently inherit the
# module's sign.
# --------------------------------------------------------------------------


_PAULIS_4 = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)


def _nielsen_avg_gate_fidelity(B: np.ndarray, target: np.ndarray) -> float:
    """Standard Nielsen average-gate-fidelity formula, leakage counted as population loss.

    Independent of, and unrelated to, the Stark-reprogramming formula under test here -- this
    is the metric the scan is minimized *against*, not a reference for the quantity being
    verified. Reimplemented locally (rather than imported) so this test has no dependency on
    any other script's copy of the same standard formula.
    """
    total = 0.0
    for pauli in _PAULIS_4:
        rotated = target @ pauli @ target.conj().T
        total += np.trace(rotated @ B @ pauli @ B.conj().T).real
    return float((total + 4.0) / 12.0)


def _naive_x_pi_drag_infidelity(beta: float, stark_coeff: float, N: int = 2000) -> float:
    """Zero-noise total infidelity of a naive-X(pi) waveform under a (beta, stark_coeff) readout.

    Builds the broadcast waveform directly from the design-curve samples (not through
    :func:`parametrization.drag_stark_quadratures`, which only exposes a single ``delta`` and
    cannot vary the DRAG amplitude and the Stark phase independently) -- ``beta`` scales the
    ``kappa_y`` DRAG quadrature, ``stark_coeff`` scales the ``kappa^2 / (2 Delta)`` phase term
    directly (i.e. ``stark_coeff = -1`` is the theoretically derived, corrected sign;
    ``stark_coeff = +1`` is the 2026-08-07 bug). Runs the actual three-level propagator
    (``curve_opt.gate``) -- not a perturbative shortcut -- and calibrates ``phi_vz`` (the free
    virtual-Z knob the gate condition already allows) before reporting ``1 - Fbar``.
    """
    # min-norm c1 projection, then rescaled back to theta = pi exactly (module docstring's
    # helper _project_onto_c1 alone does not preserve the gate angle -- see
    # c1_project_and_rescale's docstring pattern in _dev_logs/F05_budget_validation.py; without
    # the rescale this test's gate-angle residual (~1e-5) drowns out the ~1e-6 sign signal it
    # is trying to measure)
    a_proj = _project_onto_c1(basis.naive_coeffs(np.pi, T, M), T)
    a = a_proj * (np.pi / basis.gate_angle(a_proj, T))
    curve = parametrization.design_curve(a, np.zeros(M), 0.0, T, N)
    kappa, kappa_dot, dt = curve.kappa, curve.kappa_dot, curve.dt

    kappa_y = -beta * kappa_dot / DELTA
    phi_dot_prog = stark_coeff * (kappa**2) / (2.0 * DELTA)
    Phi_prog = parametrization._midpoint_cumulative(phi_dot_prog, dt)
    envelope = (kappa + 1j * kappa_y) * np.exp(1j * np.asarray(Phi_prog))
    om_x, om_y = np.real(envelope), np.imag(envelope)

    U3 = np.asarray(gate.three_level_propagator(om_x, om_y, T, DELTA, delta_z=0.0))
    B = np.asarray(gate.qubit_block(U3))
    U_target = np.asarray(propagate.target_x(np.pi))

    def neg_fid(phi):
        rz = np.array(
            [[np.exp(-1j * phi / 2), 0.0], [0.0, np.exp(1j * phi / 2)]], dtype=complex
        )
        return -_nielsen_avg_gate_fidelity(B, rz @ U_target)

    res = minimize_scalar(neg_fid, bounds=(-np.pi, np.pi), method="bounded", options={"xatol": 1e-13})
    return 1.0 - (-res.fun)


def test_stark_coefficient_scan_is_minimized_at_the_derived_sign():
    """Physical, non-circular criterion: total infidelity is minimized at (beta, stark) = (1, -1).

    No formula transcribed from the plan text is used anywhere in this test (module docstring
    §5) -- the assertion is purely "the physically correct point is the best point on the
    grid, by a wide margin", which a matching sign error on both sides of a comparison cannot
    fake.
    """
    betas = [0.0, 0.5, 1.0, 1.5]
    starks = [-2.0, -1.0, 0.0, 1.0, 2.0]

    grid = {(b, s): _naive_x_pi_drag_infidelity(b, s) for b in betas for s in starks}
    for (b, s), val in sorted(grid.items(), key=lambda kv: kv[1]):
        print(f"F02 (5) beta={b:.1f} stark={s:+.1f}  1-Fbar={val:.6e}")

    best_point, best_val = min(grid.items(), key=lambda kv: kv[1])
    print(f"F02 (5) minimum at {best_point}: {best_val:.6e}")
    assert best_point == (1.0, -1.0)

    neighbours = [
        grid[(0.5, -1.0)],
        grid[(1.5, -1.0)],
        grid[(1.0, 0.0)],
        grid[(1.0, -2.0)],
    ]
    min_neighbour = min(neighbours)
    print(f"F02 (5) closest neighbour: {min_neighbour:.6e}  ratio={min_neighbour / best_val:.1f}x")
    assert min_neighbour >= 100.0 * best_val
