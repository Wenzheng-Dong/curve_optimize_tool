"""Step 11 acceptance: the SU(2) propagator and the nine 3D constraints.

This file is the admission gate of ``_plan.md`` §6.4 item 3, and it is written so
that the gate cannot be passed by accident:

* the arbitration cases are **non-commuting** -- ``||[H_1, H_2]|| != 0`` is
  asserted before anything is compared, because a planar case has commuting
  Hamiltonians at all times and cannot see a multiplication-order error at all;
* the arbiters are **independent second implementations**: an explicit numpy
  product of ``scipy.linalg.expm`` factors, and qutip's ODE ``propagator``;
* the **wrong** order is run too, and required to fail. A test that only checks
  the correct code cannot tell whether it is a gate or a decoration -- step00e's
  3.7e-2 bug survived every check that lacked one of these three ingredients.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from curve_opt import basis, geometry, propagate

T = 1.0
SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

#: Three deliberately non-commuting piecewise-constant segments (Omega_x, Omega_y),
#: scaled so the commutator is O(1) rather than a perturbation.
SEGMENTS = ((4.0, 1.2), (0.8, 5.6), (-3.6, 2.8))


def rng(seed=0):
    return np.random.default_rng(seed)


def phase_aligned_diff(A, B) -> float:
    """max |A - B| after removing the global phase -- the physical comparison."""
    A, B = np.asarray(A), np.asarray(B)
    overlap = np.trace(A.conj().T @ B)
    phase = np.exp(1j * np.angle(overlap)) if abs(overlap) > 1e-300 else 1.0
    return float(np.max(np.abs(A * phase - B)))


def segment_samples(N_per_segment: int):
    """Midpoint samples of the piecewise-constant control, segment-aligned.

    With the segment boundaries on cell boundaries every cell has a genuinely
    constant Hamiltonian, so the discretization is *exact* and the comparison
    against the explicit product tests the ordering alone, with no truncation
    error to hide behind.
    """
    om_x = np.repeat([s[0] for s in SEGMENTS], N_per_segment)
    om_y = np.repeat([s[1] for s in SEGMENTS], N_per_segment)
    return om_x, om_y


def numpy_explicit_product(om_x, om_y, T_total: float):
    """``U = U_N ... U_1`` built one factor at a time with expm. Independent arbiter."""
    dt = T_total / len(om_x)
    U = np.eye(2, dtype=complex)
    for ox, oy in zip(om_x, om_y):
        U = expm(-1j * dt * 0.5 * (ox * SX + oy * SY)) @ U
    return U


def qutip_propagator(om_x, om_y, T_total: float):
    """qutip's ODE integration of the same piecewise-constant control."""
    import qutip

    dt = T_total / len(om_x)
    edges = np.arange(len(om_x) + 1) * dt

    def coeff(values):
        def f(t, args=None):
            idx = int(np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(values) - 1))
            return float(values[idx])

        return f

    H = [[0.5 * qutip.sigmax(), coeff(om_x)], [0.5 * qutip.sigmay(), coeff(om_y)]]
    result = qutip.propagator(
        H, T_total, options={"atol": 1e-13, "rtol": 1e-11, "nsteps": 500_000}
    )
    U = result[-1] if isinstance(result, list) else result
    return np.asarray(U.full())


def wrong_order_propagator(om_x, om_y, T_total: float):
    """The bug of ``_plan.md`` §6.3: ``associative_scan(jnp.matmul, ...)``.

    Kept in the test suite on purpose. It is the only way to demonstrate that the
    arbitration actually discriminates -- and to document what the failure looks
    like if anyone reintroduces it.
    """
    om_x = jnp.asarray(om_x, dtype=jnp.float64)
    om_y = jnp.asarray(om_y, dtype=jnp.float64)
    dt = T_total / om_x.shape[0]
    steps = propagate._cell_propagators(om_x, om_y, dt, 1.0)
    return np.asarray(jax.lax.associative_scan(jnp.matmul, steps)[-1])


# --------------------------------------------------------------------------
# ★ the admission gate: non-commuting case, two independent arbiters
# --------------------------------------------------------------------------


def test_the_arbitration_case_is_genuinely_non_commuting():
    """Asserted first: without this the whole file proves nothing about ordering."""
    H = [0.5 * (ox * SX + oy * SY) for ox, oy in SEGMENTS]
    commutators = [
        np.linalg.norm(H[i] @ H[j] - H[j] @ H[i]) for i in range(3) for j in range(i + 1, 3)
    ]
    assert min(commutators) > 1.0, commutators


def test_non_commuting_matches_the_explicit_numpy_product_exactly():
    """★ Segment-aligned piecewise-constant control: agreement must be exact."""
    om_x, om_y = segment_samples(400)
    ours = np.asarray(propagate.chain(om_x, om_y, T).U_edge[-1])
    reference = numpy_explicit_product(om_x, om_y, T)
    # 1200 sequential matrix products, so the floor is accumulation of rounding,
    # not discretization: 9.3e-16 at N=300 and 2.3e-14 at N=1200.
    assert phase_aligned_diff(ours, reference) < 1e-13
    assert np.max(np.abs(ours - reference)) < 1e-13  # not even a phase apart


def test_non_commuting_matches_qutip_sesolve():
    """★ Acceptance: <= 5e-6 against an independent ODE integration."""
    om_x, om_y = segment_samples(200)
    ours = np.asarray(propagate.chain(om_x, om_y, T).U_edge[-1])
    reference = qutip_propagator(om_x, om_y, T)
    deviation = phase_aligned_diff(ours, reference)
    assert deviation <= 5e-6, deviation
    assert deviation < 1e-8  # in fact far better: the discretization is exact here


def test_smooth_non_commuting_sine_control_matches_qutip():
    """★ The acceptance criterion on a control the project actually optimizes.

    A sine series with a large ``Omega_y``: now the piecewise-constant stepping is
    an approximation, and the residual is the O(dt^2) truncation step00e measured
    at 4.5e-6 for N = 4000.
    """
    M, N = 6, 4000
    a = rng(1).normal(size=M) * 4.0
    b = rng(2).normal(size=M) * 3.0
    om_x, om_y = (np.asarray(x) for x in propagate.omega_samples(a, b, T, N))
    assert propagate.noncommutativity(a, b, T, N) > 1.0  # genuinely non-planar

    ours = np.asarray(propagate.propagator(a, b, T, N))
    reference = qutip_propagator(om_x, om_y, T)
    deviation = phase_aligned_diff(ours, reference)
    assert deviation <= 5e-6, deviation


@pytest.mark.parametrize("n_per_seg", [100, 200, 400])
def test_the_wrong_order_is_caught_by_the_non_commuting_case(n_per_seg):
    """★ Negative control: the reversed scan must deviate at the 1e-2 level.

    step00e measured 3.7e-2 for its segments. The magnitude depends on the case;
    what matters is that it is nowhere near a rounding error, and that it does not
    shrink as the grid is refined -- it is a different problem, not a worse
    approximation.
    """
    om_x, om_y = segment_samples(n_per_seg)
    reference = numpy_explicit_product(om_x, om_y, T)
    wrong = wrong_order_propagator(om_x, om_y, T)
    deviation = phase_aligned_diff(wrong, reference)
    assert deviation > 1e-2, deviation


def test_a_planar_case_cannot_see_the_order_error_at_all():
    """★ Exactly why a planar case may not serve as the admission gate.

    With ``Omega_y = 0`` every H(t) commutes, so the reversed product is the *same
    matrix*. Any validation suite built only on planar cases will pass while the
    physics is wrong -- which is what happened in step00e.
    """
    om_x, _ = segment_samples(300)
    om_y = np.zeros_like(om_x)
    H = [0.5 * ox * SX for ox in {*om_x}]
    assert max(np.linalg.norm(h1 @ h2 - h2 @ h1) for h1 in H for h2 in H) < 1e-15

    right = np.asarray(propagate.chain(om_x, om_y, T).U_edge[-1])
    wrong = wrong_order_propagator(om_x, om_y, T)
    assert phase_aligned_diff(wrong, right) < 1e-13  # indistinguishable
    assert phase_aligned_diff(right, numpy_explicit_product(om_x, om_y, T)) < 1e-14


# --------------------------------------------------------------------------
# planar limit against the L1 implementation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("M", [8, 12])
def test_planar_limit_reproduces_the_L1_geometry(M):
    """★ Acceptance: <= 1e-12 against :mod:`curve_opt.geometry` when b = 0."""
    N = 4000
    a = rng(3).normal(size=M)
    b = np.zeros(M)

    c3 = propagate.chain(*propagate.omega_samples(a, b, T, N), T)
    c1 = geometry.chain(geometry.omega_samples(a, T, N), T)

    assert np.max(np.abs(np.asarray(c3.tangent) - np.asarray(c1.tangent))) <= 1e-12
    assert np.max(np.abs(np.asarray(c3.r_mid) - np.asarray(c1.r_mid))) <= 1e-12
    assert np.max(np.abs(np.asarray(c3.closure) - np.asarray(c1.closure))) <= 1e-12
    assert np.max(np.abs(np.asarray(c3.area) - np.asarray(c1.area))) <= 1e-12
    # the x-components must be identically zero in the planar limit
    assert abs(float(c3.closure[0])) <= 1e-13
    assert np.max(np.abs(np.asarray(c3.area)[1:])) <= 1e-13


def test_planar_limit_gate_angle_matches_the_linear_row():
    """The propagator's rotation angle vs the exact gate row of the planar layer."""
    M, N = 12, 20_000
    a = basis.naive_coeffs(np.pi, T, M)
    U = propagate.propagator(a, np.zeros(M), T, N)
    angle, axis = propagate.gate_rotation(U)
    assert float(angle) == pytest.approx(np.pi, abs=1e-8)
    assert np.allclose(np.asarray(axis), [1.0, 0.0, 0.0], atol=1e-9)
    assert float(angle) == pytest.approx(basis.gate_angle(a, T), abs=1e-8)


def test_planar_limit_of_the_nine_residuals_matches_the_three():
    """The 9-vector reduces to the planar 3 plus six machine zeros."""
    M, N = 12, 4000
    a = rng(4).normal(size=M) * 2.0
    nine = np.asarray(propagate.equality_residuals(a, np.zeros(M), T, N))
    three = np.asarray(geometry.equality_residuals(a, T, N))
    assert nine.shape == (9,)
    # closure y,z and area x are the planar equations
    assert nine[4] == pytest.approx(three[0], abs=1e-12)
    assert nine[5] == pytest.approx(three[1], abs=1e-12)
    assert nine[6] == pytest.approx(three[2], abs=1e-12)
    # closure x and area y,z vanish identically for a planar control
    assert abs(nine[3]) <= 1e-13
    assert np.max(np.abs(nine[7:])) <= 1e-13


# --------------------------------------------------------------------------
# order of accuracy, unitarity, gate residual algebra
# --------------------------------------------------------------------------


def test_convergence_is_second_order_in_the_grid():
    """★ O(dt^2): the midpoint discipline of §6.2 applies to the propagator too."""
    M = 5
    a, b = rng(5).normal(size=M) * 3.0, rng(6).normal(size=M) * 2.0
    reference = np.asarray(propagate.propagator(a, b, T, 64_000))
    errs = [
        phase_aligned_diff(np.asarray(propagate.propagator(a, b, T, N)), reference)
        for N in (500, 1000, 2000)
    ]
    ratios = [errs[i] / errs[i + 1] for i in range(2)]
    assert all(3.6 < r < 4.4 for r in ratios), (errs, ratios)


@pytest.mark.parametrize("N", [200, 4000])
def test_every_propagator_is_unitary(N):
    M = 6
    a, b = rng(7).normal(size=M) * 5.0, rng(8).normal(size=M) * 5.0
    c = propagate.chain(*propagate.omega_samples(a, b, T, N), T)
    for U in (c.U_edge[-1], c.U_mid[N // 2], c.U_edge[0]):
        assert propagate.unitarity_error(U) < 1e-13
    assert np.allclose(np.linalg.norm(np.asarray(c.tangent), axis=1), 1.0, atol=1e-13)


def test_zero_field_gives_the_identity_and_a_straight_line():
    N = 100
    zero = np.zeros(N)
    c = propagate.chain(zero, zero, T)
    assert np.max(np.abs(np.asarray(c.U_edge[-1]) - np.eye(2))) < 1e-15
    # tangent stays on +z, so the curve is the straight segment of length T
    assert np.allclose(np.asarray(c.tangent), np.array([0.0, 0.0, 1.0]), atol=1e-15)
    assert float(c.closure[2]) == pytest.approx(T, rel=1e-14)


def test_sinc_form_is_exact_and_differentiable_at_zero_field():
    """The ``sin(phi)/|Omega|`` factor must not need an epsilon guard.

    A cell with |Omega| = 0 occurs on the way to every solution (Omega_y -> 0 and
    Omega_x crosses zero), so both the value and the gradient have to be finite
    there.
    """
    N = 50

    def endpoint(scale):
        om = jnp.full(N, scale)
        return jnp.real(jnp.trace(propagate.chain(om, jnp.zeros(N), T).U_edge[-1]))

    assert np.isfinite(float(endpoint(0.0)))
    grad = float(jax.grad(endpoint)(0.0))
    assert np.isfinite(grad)
    assert abs(grad) < 1e-12  # d/dscale of cos(scale T / 2) at 0


def test_gate_residual_vanishes_exactly_on_the_target_up_to_phase():
    U = propagate.target_x(np.pi)
    assert np.max(np.abs(np.asarray(propagate.gate_residual(U, U)))) < 1e-15
    for phase in (1.0, -1.0, np.exp(0.7j)):
        assert np.max(np.abs(np.asarray(propagate.gate_residual(phase * U, U)))) < 1e-15


def test_gate_residual_is_nonzero_for_a_wrong_gate():
    target = propagate.target_x(np.pi)
    for angle in (np.pi / 2, np.pi + 0.01, 3 * np.pi / 2):
        res = np.asarray(propagate.gate_residual(propagate.target_x(angle), target))
        assert np.max(np.abs(res)) > 1e-4, angle
    # 3 pi is the same gate on the next winding branch: residual vanishes again
    res = np.asarray(propagate.gate_residual(propagate.target_x(3 * np.pi), target))
    assert np.max(np.abs(res)) < 1e-14


def test_gate_rotation_round_trips():
    for angle in (0.3, np.pi / 2, np.pi, 4.0):
        for axis in ([1.0, 0, 0], [0, 1.0, 0], [0.6, 0.8, 0.0]):
            U = propagate.su2_rotation(np.array(axis), angle)
            got_angle, got_axis = propagate.gate_rotation(U)
            assert float(got_angle) == pytest.approx(angle, abs=1e-12)
            assert np.allclose(
                np.asarray(got_axis), np.array(axis) / np.linalg.norm(axis), atol=1e-12
            )


def test_nine_residuals_are_jax_differentiable():
    """The (9, 2M) Jacobian Step 12 needs, checked against finite differences."""
    M, N = 6, 500
    a, b = rng(9).normal(size=M), rng(10).normal(size=M)

    def residual(x):
        return propagate.equality_residuals(x[:M], x[M:], T, N)

    x0 = np.concatenate([a, b])
    jac = np.asarray(jax.jacrev(residual)(x0))
    assert jac.shape == (9, 2 * M)
    assert np.all(np.isfinite(jac))
    h = 1e-6
    for j in (0, M - 1, M, 2 * M - 1):
        e = np.zeros(2 * M)
        e[j] = 1.0
        fd = (np.asarray(residual(x0 + h * e)) - np.asarray(residual(x0 - h * e))) / (2 * h)
        assert np.allclose(jac[:, j], fd, rtol=1e-5, atol=1e-8)


def test_noncommutativity_is_exactly_zero_for_planar_and_grows_with_omega_y():
    """★ And it must be zero for a *sign-changing* planar control too.

    That is where step00e's arctan2 torsion fails: the same input below makes it
    report 2 pi because Omega_x crosses zero twice, even though every Hamiltonian
    in the control commutes with every other. The collinearity form returns 0.0.
    """
    M, N = 8, 2000
    a = rng(11).normal(size=M) * 3.0
    om_x, _ = propagate.omega_samples(a, np.zeros(M), T, N)
    assert int(np.sum(np.diff(np.sign(np.asarray(om_x))) != 0)) >= 2  # it does change sign
    assert propagate.noncommutativity(a, np.zeros(M), T, N) == 0.0  # exactly

    # the discredited measure, kept here only as the demonstration
    phase = np.arctan2(np.zeros(N), np.asarray(om_x))
    assert np.sum(np.abs(np.diff(np.unwrap(phase)))) == pytest.approx(2 * np.pi, abs=1e-9)

    previous = 0.0
    for scale in (0.1, 0.5, 1.0):
        value = propagate.noncommutativity(a, rng(12).normal(size=M) * 3.0 * scale, T, N)
        assert value > previous
        previous = value


def test_noncommutativity_vanishes_for_any_collinear_two_component_drive():
    """Anti-parallel counts as commuting: a fixed direction with a signed amplitude."""
    M, N = 6, 1000
    a = rng(13).normal(size=M) * 2.0
    for ratio in (0.5, -2.0, 3.0):
        assert propagate.noncommutativity(a, ratio * a, T, N) == pytest.approx(0.0, abs=1e-24)


# --------------------------------------------------------------------------
# Step 06 leftover: the 3D families' post-projection robustness and gate angle
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family,cct_angle",
    [
        ("alpha_3d", 0.00098),
        ("zeng_clifford_3d", 4.18828),
        ("lissajous_3d", 5.89043),
        ("helix", 1.77160),
    ],
)
def test_gate_angle_of_a_3d_source_matches_upstreams_own_value(family, cct_angle):
    """★ Independent cross-check of the whole chain against cct's rotation_angle.

    And the correctness fix it enables: ``int Omega_x dt`` is the gate angle only
    for a planar control. Reading it off a 3D one reports alpha_3d as turning
    12.43 rad when the unitary it produces is the identity.
    """
    from conftest import requires_cct  # noqa: PLC0415

    if requires_cct.args[0]:  # skipif condition is True => upstream missing
        pytest.skip("upstream cct not importable")
    from curve_opt import ansatz  # noqa: PLC0415

    ansatz.register_builtin_families(overwrite=True)
    ansatz.register_cct_families()
    src = ansatz.source(family)
    report = ansatz.project(src, M=12, T=T, N=4000)

    assert report.theta_before == pytest.approx(cct_angle, abs=5e-4), report.theta_before
    assert report.theta_before == pytest.approx(
        float(src.metadata["cct_rotation_angle"]), abs=5e-4
    )
    assert not report.is_planar


def test_projection_destroys_the_robustness_of_the_closed_3d_families():
    """★ Fills the n/a cells Step 06 had to leave, and the answer is not benign.

    alpha_3d and lissajous_3d are closed and zero-area as curves to 1e-17, but
    their M=12 sine projection is not: closure lands at the 1e-2 level. With
    relRMS around 0.9 and 0.65 the projected control is simply a different pulse,
    which is exactly the caveat the forced tier carries -- the effective ansatz is
    the projected object, not the published curve.
    """
    from conftest import requires_cct  # noqa: PLC0415

    if requires_cct.args[0]:
        pytest.skip("upstream cct not importable")
    from curve_opt import ansatz  # noqa: PLC0415

    ansatz.register_builtin_families(overwrite=True)
    ansatz.register_cct_families()
    for family in ("alpha_3d", "lissajous_3d"):
        report = ansatz.project(ansatz.source(family), M=12, T=T, N=4000)
        assert report.closure_before < 1e-15, family
        assert report.area_before < 1e-8, family
        assert report.closure_after is not None and report.area_after is not None, family
        assert report.closure_after > 1e-3, (family, report.closure_after)
        assert report.area_after > 1e-3, (family, report.area_after)
