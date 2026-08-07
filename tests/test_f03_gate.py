"""F03 acceptance: the polar-decomposition gate residual (criterion (3), plus
the correctness of :mod:`curve_opt.gate`'s closed-form polar decomposition
that criterion (2) leans on).

Self-contained: nothing here needs upstream Novera (criterion (3) is
explicitly scoped that way -- ``_dev_logs/F03_task_brief.md`` says the
Jacobian check just needs "a known input that realizes U_target", not a real
three-level propagation). ``tests/test_f03_novera.py`` covers the
Novera-dependent criteria (1), (2)'s Lambda-vs-P cross-check and (4).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import polar as scipy_polar

import jax
import jax.numpy as jnp

from curve_opt import gate, propagate

SX = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
SY = jnp.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=jnp.complex128)
SZ = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
I2 = jnp.eye(2, dtype=jnp.complex128)


def rng(seed: int):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# polar_decompose correctness -- independent arbiter (scipy.linalg.polar),
# not the jax closed form under test. This underlies criterion (2)'s leakage
# read-off (1 - tr(P^2)/2), so it has to be right on its own before that
# comparison means anything.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_polar_decompose_matches_scipy_and_reconstructs_B(seed):
    r = rng(seed)
    B = (r.normal(size=(2, 2)) + 1j * r.normal(size=(2, 2))) * 0.7

    W, P = gate.polar_decompose(B)
    W, P = np.asarray(W), np.asarray(P)
    W_ref, P_ref = scipy_polar(B, side="right")

    assert np.allclose(W, W_ref, atol=1e-13)
    assert np.allclose(P, P_ref, atol=1e-13)
    assert np.allclose(B, W @ P, atol=1e-13)
    assert np.allclose(np.conj(W).T @ W, np.eye(2), atol=1e-13)
    assert np.allclose(P, np.conj(P).T, atol=1e-13)  # Hermitian
    assert np.all(np.linalg.eigvalsh(P) >= -1e-12)  # PSD


def test_polar_decompose_is_identity_when_B_is_already_unitary():
    """B unitary => P = I exactly, W = B exactly -- the no-leakage limit."""
    theta = 0.83
    B = propagate.su2_rotation(jnp.array([0.3, -0.6, 0.7]), theta)
    W, P = gate.polar_decompose(B)
    assert np.allclose(np.asarray(P), np.eye(2), atol=1e-13)
    assert np.allclose(np.asarray(W), np.asarray(B), atol=1e-13)


def test_polar_margin_equals_det_P():
    r = rng(5)
    B = (r.normal(size=(2, 2)) + 1j * r.normal(size=(2, 2))) * 0.5
    _, P = gate.polar_decompose(B)
    margin = float(gate.polar_margin(B))
    det_P = float(jnp.real(jnp.linalg.det(P)))
    assert margin == pytest.approx(det_P, abs=1e-13)


def test_leakage_population_is_zero_for_a_unitary_B():
    B = propagate.su2_rotation(jnp.array([1.0, 0.0, 0.0]), 1.1)
    _, P = gate.polar_decompose(B)
    assert float(gate.leakage_population(P)) == pytest.approx(0.0, abs=1e-13)


# --------------------------------------------------------------------------
# Why gate_residual_vector normalizes W's phase before reusing
# propagate.gate_residual: a worked false-zero example (module docstring's
# derivation, checked numerically here rather than only asserted in prose).
# --------------------------------------------------------------------------


def test_su2_only_residual_formula_has_a_false_zero_that_normalization_avoids():
    """V = i*(cos(chi) I - i sin(chi) sigma_x), chi != 0: cos(phase) = 0 exactly,
    so the bare SU(2)-only formula reports zero despite V being far from +-I.
    """
    chi = 0.6
    V = 1j * (jnp.cos(chi) * I2 - 1j * jnp.sin(chi) * SX)
    false_zero = propagate.gate_residual(V, I2)
    assert np.allclose(np.asarray(false_zero), 0.0, atol=1e-12)

    fixed = gate.gate_residual_vector(V, I2, 0.0)
    assert float(jnp.linalg.norm(fixed)) > 0.1  # genuinely detects the mismatch


# --------------------------------------------------------------------------
# Criterion (3): vector residual, zero at a known solution, full-rank
# Jacobian there (w.r.t. both phi_vz and the "control parameters" feeding B).
# --------------------------------------------------------------------------

U_TARGET = propagate.target_x(jnp.pi)


def _guarded_small_rotation(n):
    """exp(-i n.sigma/2), analytic at n = 0 (matches propagate.py's own guard
    pattern for the |Omega| -> 0 singularity: *both* the sinc coefficient and
    the rotation angle itself must be built from the ``jnp.where``-guarded
    magnitude, or the unselected branch's dummy value leaks into ``cos``
    through the shared ``half`` -- exactly the bug this test caught during
    development (``_dev_logs/F03_novera_polar.md``).
    """
    n = jnp.asarray(n)
    sq = jnp.sum(n**2)
    positive = sq > 0.0
    mag = jnp.sqrt(jnp.where(positive, sq, 1.0))
    half = jnp.where(positive, 0.5 * mag, 0.0)
    coeff = jnp.where(positive, jnp.sin(half) / mag, 0.5)
    drive = n[0] * SX + n[1] * SY + n[2] * SZ
    return jnp.cos(half) * I2 - 1j * coeff * drive


def _synthetic_B(params):
    """A toy differentiable stand-in for "the qubit block of a three-level
    unitary as a function of control parameters": a small rotation away from
    U_target (``params[:3]``) times a scalar leakage shrink (``params[3]``).
    Exercises gate.py's own math (polar decomposition + residual), not an
    actual three-level propagator -- that cross-check is criterion (2),
    ``tests/test_f03_novera.py``, and needs upstream.
    """
    n = params[:3]
    leak = params[3]
    return (1.0 - leak) * (U_TARGET @ _guarded_small_rotation(n))


def _residual_of_x(x):
    phi_vz = x[0]
    B = _synthetic_B(x[1:])
    W, P = gate.polar_decompose(B)
    return gate.gate_residual_vector(W, U_TARGET, phi_vz)


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


def test_gate_residual_vector_is_zero_at_a_known_solution():
    x0 = jnp.zeros(5)
    residual = _residual_of_x(x0)
    assert np.allclose(np.asarray(residual), 0.0, atol=1e-12)


def test_gate_residual_vector_jacobian_is_full_rank_at_the_solution():
    """The central claim: unlike a scalar 1 - Fbar = 0 constraint (whose
    gradient vanishes exactly at the solution), this vector residual's
    Jacobian w.r.t. (phi_vz, control params) stays rank 3 there.
    """
    x0 = np.zeros(5)
    jac_ad = np.asarray(jax.jacrev(_residual_of_x)(jnp.asarray(x0)))
    jac_fd = _central_difference_jacobian(_residual_of_x, x0)
    assert jac_ad.shape == (3, 5)
    assert np.allclose(jac_ad, jac_fd, rtol=1e-5, atol=1e-7)
    assert np.linalg.matrix_rank(jac_ad, tol=1e-8) == 3
    assert np.linalg.matrix_rank(jac_fd, tol=1e-6) == 3


def test_gate_residual_vector_jacobian_stays_full_rank_at_a_generic_point():
    """Not just an isolated fluke at the solution: full rank nearby too."""
    x0 = np.array([0.2, 0.05, -0.03, 0.02, 0.01])
    jac_ad = np.asarray(jax.jacrev(_residual_of_x)(jnp.asarray(x0)))
    assert np.linalg.matrix_rank(jac_ad, tol=1e-8) == 3
