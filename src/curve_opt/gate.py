"""Polar-decomposition gate residual: the full-cost era's hard gate constraint.

``_plan_full_cost.md`` §2.3a establishes that a three-level qubit block
``B = U_3(T)[:2, :2]`` is not unitary once leakage is present (``Lambda != 0``),
so "the gate" is not something a design curve can be read off any more -- it
has to be *measured* on the actual three-level propagator, after separating
the coherent rotation from the leakage-induced loss of norm. The polar
decomposition does exactly that::

    B = W P,     P = (B^dagger B)^(1/2) >= 0,     W = B P^{-1} in U(2)

``W`` is the coherent gate actually implemented (leakage's norm loss
stripped out); ``P`` is the leakage factor, with ``1 - (1/2) tr(P^2)`` the
average population lost to |2> -- see :func:`leakage_population`, whose
leading order is ``(1/2) ||Lambda||^2`` (:func:`curve_opt.propagate.leakage_amplitude`,
checked in ``tests/test_f03_gate.py``'s criterion (2)).

Hard gate constraint (§2.3a, §4.2 decision 1)
-----------------------------------------------
``W = R_z(Phi_vz) U_target``, up to a global phase, with ``Phi_vz`` a free
optimization variable (the virtual-Z calibration, free in hardware). This is
the *only* hard constraint of the full-cost era, and it must stay a genuine
vector residual: turning it into the scalar ``1 - Fbar(W, U_target) = 0``
would have a vanishing gradient exactly at the solution (Jacobian rank drops
to 0 where trust-constr needs it to be full rank) -- AGENTS.md's numerical
red line, and the whole reason this module exists instead of one line calling
:func:`curve_opt.metrics` fidelity.

★ ``W`` in general lives in ``U(2)``, not ``SU(2)``: ``B`` is an arbitrary
complex matrix (the qubit block of a three-level unitary that is not
required to be special-unitary), and the polar factor ``W`` inherits
whatever phase ``det B`` carries.
:func:`curve_opt.propagate.gate_residual`'s axis-angle formula only detects
"``V`` equals the identity up to a *sign*" when its two arguments are
genuinely in ``SU(2)`` -- that module's own ``chain()`` only ever builds
``SU(2)`` propagators (traceless drive Hamiltonian), so it never needed to
handle this. A worked check: for a general unitary ``V = e^(i phase)(cos(chi)
I - i sin(chi) n.sigma)``, ``-Im tr(V sigma_a)/2 = sin(chi) n_a cos(phase)``,
which vanishes whenever ``cos(phase) = 0`` regardless of ``chi, n`` -- a false
zero the ``SU(2)``-only formula cannot see. :func:`gate_residual_vector`
therefore normalizes ``W`` to its own canonical ``SU(2)`` representative
first (``W_tilde = W / sqrt(det W)``, principal branch; ``|det W| = 1``
exactly since ``W`` is unitary) and reuses
:func:`curve_opt.propagate.gate_residual` on ``(W_tilde, R_z(Phi_vz)
U_target)``, both now genuinely in ``SU(2)``. The global phase absorbed by
``sqrt(det W)`` is exactly the unobservable phase the "up to a global phase"
clause refers to; ``Phi_vz`` is the separate, physical virtual-Z knob and is
not part of that gauge.

Near-singular ``B``: not a removable singularity
--------------------------------------------------
Unlike ``propagate.py``'s ``|Omega| = 0`` guard (a genuine coordinate
singularity of the propagator parametrization, smoothed away exactly by a
``sinc`` rewrite -- see that module's docstring), ``B`` losing rank is a
*physical* breakdown: it means the pulse has leaked essentially all of its
population out of the qubit subspace, so "the coherent gate it implements" is
not well-defined at all, by any convention. There is no smooth formula to
paper over that point, and this device's regime (``eta = Omega_max/|Delta| =
0.25``) never approaches it -- leakage populations here run at a few percent.
So :func:`polar_decompose` does not special-case it: at exact ``det B = 0``
the closed form divides by zero and returns ``inf``/``nan``, on purpose
rather than silently. Callers must check :func:`polar_margin` (``|det B|``,
which equals ``det P``) before trusting :func:`gate_residual_vector`'s value
or gradient there; this is a diagnostic health check in the same spirit as
``propagate.unitarity_error``, not a differentiable relaxation.

Closed-form polar decomposition (2x2, jax-differentiable)
-----------------------------------------------------------
No eigendecomposition (not differentiable at degenerate eigenvalues, and
unnecessary here). By Cayley-Hamilton, for Hermitian PSD ``M = B^dagger B``
with ``S = a M + b I`` solving ``S^2 = M``::

    S = P = (M + sqrt(det M) I) / sqrt(tr M + 2 sqrt(det M))

(checked against ``jax.jacrev``-through-:func:`polar_decompose` and against
an eigendecomposition reference in ``tests/test_f03_gate.py``). ``P^{-1}``
then uses the 2x2 adjugate/determinant formula, with ``det P = sqrt(det M) =
|det B|`` (the same quantity :func:`polar_margin` reports) rather than a
second matrix inversion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from curve_opt.propagate import gate_residual as _su2_gate_residual
from curve_opt.propagate import su2_rotation

__all__ = [
    "DRIVE_X3",
    "DRIVE_Y3",
    "gate_residual_vector",
    "leakage_population",
    "polar_decompose",
    "polar_gate_residual",
    "polar_margin",
    "qubit_block",
    "rz",
    "three_level_bare",
    "three_level_propagator",
]


def qubit_block(U3):
    """Rows/cols 0, 1 of a >=2-level unitary -- the computational-subspace block ``B``.

    Row/column convention matches ``propagate.py``'s Novera cross-check
    (``U_c(s)[1, j]`` = row 1, i.e. levels are indexed ``0, 1, 2, ...`` in the
    obvious order) and Novera's own ``gate[:2, :2]`` slicing
    (``db_transmon.block_gate_fidelity``).
    """
    U3 = jnp.asarray(U3)
    return U3[:2, :2]


def rz(phi):
    """``R_z(phi) = exp(-i phi Z / 2)``, the free virtual-Z calibration.

    Reuses :func:`curve_opt.propagate.su2_rotation` rather than rederiving the
    same formula -- one source for the SU(2) rotation convention.
    """
    return su2_rotation(jnp.array([0.0, 0.0, 1.0]), phi)


def polar_margin(B) -> jnp.ndarray:
    """``|det B|`` -- the polar-decomposition validity margin, in ``[0, 1]``.

    Equals ``det P`` exactly (see the module docstring): zero iff ``B`` has
    lost a full unit of qubit-subspace population, at which point
    :func:`polar_decompose` and :func:`gate_residual_vector` are undefined
    (division by zero, returned as ``inf``/``nan`` on purpose -- see the
    module docstring's "near-singular B" section). Callers must check this
    *before* trusting either function's output or gradient near such a point.
    """
    B = jnp.asarray(B)
    return jnp.abs(jnp.linalg.det(B))


def polar_decompose(B):
    """Closed-form polar decomposition of a 2x2 complex matrix: ``(W, P)``.

    ``B = W P``, ``P = (B^dagger B)^(1/2)`` Hermitian PSD, ``W = B P^{-1}``
    unitary. See the module docstring for the closed form and its validity
    range (:func:`polar_margin`).
    """
    B = jnp.asarray(B, dtype=jnp.complex128)
    M = jnp.conj(B).T @ B
    det_M = jnp.maximum(jnp.real(jnp.linalg.det(M)), 0.0)  # PSD; clamp roundoff
    sqrt_det_M = jnp.sqrt(det_M)
    tr_M = jnp.real(jnp.trace(M))
    denom = jnp.sqrt(tr_M + 2.0 * sqrt_det_M)
    eye2 = jnp.eye(2, dtype=jnp.complex128)
    P = (M + sqrt_det_M * eye2) / denom
    # det(P) = sqrt(det M) = |det B| for PSD M's unique PSD square root --
    # the same quantity polar_margin reports, computed here without a second
    # call to jnp.linalg.det.
    det_P = sqrt_det_M
    P_inv = jnp.array([[P[1, 1], -P[0, 1]], [-P[1, 0], P[0, 0]]]) / det_P
    W = B @ P_inv
    return W, P


def leakage_population(P) -> jnp.ndarray:
    """``1 - (1/2) tr(P^2)``: average population leaked out of the qubit subspace.

    Leading order equals ``(1/2) ||Lambda||^2``
    (:func:`curve_opt.propagate.leakage_amplitude`) -- criterion (2) of
    ``tests/test_f03_gate.py``.
    """
    P = jnp.asarray(P)
    return jnp.real(1.0 - 0.5 * jnp.trace(P @ P))


def gate_residual_vector(W, U_target, phi_vz):
    """3 real equations pinning ``W = R_z(phi_vz) U_target`` up to a global phase.

    See the module docstring ("Hard gate constraint") for the derivation of
    why ``W`` -- generically in ``U(2)``, not ``SU(2)`` -- is first normalized
    to its canonical ``SU(2)`` representative ``W / sqrt(det W)`` before
    reusing :func:`curve_opt.propagate.gate_residual`. ``phi_vz`` is a free
    optimization variable, not part of the global-phase gauge that
    normalization absorbs.

    Returns shape ``(3,)``, exactly zero (to floating precision) iff ``W``
    realizes ``U_target`` up to the free virtual-Z and an unobservable global
    phase.
    """
    W = jnp.asarray(W, dtype=jnp.complex128)
    det_W = jnp.linalg.det(W)
    W_tilde = W / jnp.sqrt(det_W)
    target = rz(phi_vz) @ jnp.asarray(U_target, dtype=jnp.complex128)
    return _su2_gate_residual(W_tilde, target)


def polar_gate_residual(U3, U_target, phi_vz):
    """One-shot: :func:`qubit_block` -> :func:`polar_decompose` -> :func:`gate_residual_vector`.

    Returns ``(residual (3,), W (2, 2), P (2, 2))`` so a caller also gets the
    leakage factor for free (:func:`leakage_population`, or C4 bookkeeping)
    without a second call to :func:`polar_decompose`.
    """
    B = qubit_block(U3)
    W, P = polar_decompose(B)
    residual = gate_residual_vector(W, U_target, phi_vz)
    return residual, W, P


# --------------------------------------------------------------------------
# F04 addition: a JAX-differentiable three-level propagator
# --------------------------------------------------------------------------
# ``curve_opt.novera.three_level_gate`` (F03) is the reference three-level
# simulation, but it is a plain-numpy loop over ``scipy.linalg.expm`` (see its
# docstring / upstream ``db_transmon.three_level_gate``) -- opaque to
# ``jax.grad``. F04's hard gate constraint needs an analytic Jacobian of
# :func:`polar_gate_residual` with respect to the pulse coefficients for
# ``trust-constr``, so this module adds its own three-level midpoint chain,
# built the same way :func:`curve_opt.propagate.chain` builds the two-level
# one (piecewise-constant cells at their midpoint value,
# ``jax.lax.associative_scan`` for the ordered product, newest on the left --
# the same numerical contract), but with ``jax.scipy.linalg.expm`` standing in
# for the closed-form ``sinc`` rewrite: no 2x2-style closed form exists for a
# driven three-level ladder, and unlike the two-level ``1/|Omega|`` division
# ``expm`` has no coordinate singularity at zero drive, so no guard is needed.
# Cross-checked against :func:`curve_opt.novera.three_level_gate` in
# ``tests/test_f04_budget.py`` -- a genuinely independent numpy/scipy
# reference on a differently-discretized grid (edge-averaged vs midpoint), so
# agreement is only approximate, the same caveat F01/F03 already documented
# for ``leakage_amplitude`` vs Novera's own.
#
# The Hamiltonian matches ``db_transmon`` term for term (git hash
# ``59fb616``): ``H = diag(0, 0, delta) + Omega_x DRIVE_X + Omega_y DRIVE_Y``
# with the ladder operator ``a`` such that ``a|1> = |0>``, ``a|2> =
# sqrt(2)|1>``. Deliberately *no* static detuning, amplitude scale or frame
# rotation term: those are ``db_transmon``'s *calibration* knobs
# (``_calibration_terms``), not part of the nominal design Hamiltonian G is
# measured against (``delta = curve_opt.device.Device.delta``, the
# anharmonicity -- the noise moments delta_z, epsilon enter only through
# C1/C3's *weights* in :mod:`curve_opt.budget`, never through this
# propagator).

_LOWER3 = jnp.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, jnp.sqrt(2.0)], [0.0, 0.0, 0.0]], dtype=jnp.complex128
)
#: ``H_drive = Omega_x * DRIVE_X3 + Omega_y * DRIVE_Y3`` -- matches
#: ``db_transmon.DRIVE_X`` / ``DRIVE_Y`` term for term (``_LOWER3`` is real, so
#: ``.conj().T`` and ``.T`` agree here).
DRIVE_X3 = 0.5 * (_LOWER3 + jnp.conj(_LOWER3).T)
DRIVE_Y3 = 0.5j * (jnp.conj(_LOWER3).T - _LOWER3)


def three_level_bare(delta) -> jnp.ndarray:
    """``diag(0, 0, delta)`` -- the undriven three-level Hamiltonian.

    ``delta`` is the anharmonicity (``curve_opt.device.Device.delta``), not
    the static-detuning noise moment -- see the section docstring above.
    """
    return jnp.diag(jnp.stack([0.0 * delta, 0.0 * delta, delta]).astype(jnp.complex128))


def _three_level_cell_propagators(om_x, om_y, delta, dt: float) -> jnp.ndarray:
    """``exp(-i dt H(Omega_x_k, Omega_y_k))`` per cell, shape ``(N, 3, 3)``.

    ``jax.scipy.linalg.expm`` in place of :mod:`curve_opt.propagate`'s closed
    ``sinc`` form (section docstring): no ``|Omega| = 0`` guard is needed here
    because ``expm`` has no coordinate singularity at zero drive (unlike the
    ``1 / |Omega|`` division the two-level closed form removes).
    """
    om_x = jnp.asarray(om_x, dtype=jnp.float64)
    om_y = jnp.asarray(om_y, dtype=jnp.float64)
    H = (
        three_level_bare(delta)[None, :, :]
        + om_x[:, None, None] * DRIVE_X3[None, :, :]
        + om_y[:, None, None] * DRIVE_Y3[None, :, :]
    )
    return jax.vmap(lambda h: jax.scipy.linalg.expm(-1j * h * dt))(H)


def three_level_propagator(om_x_mid, om_y_mid, T: float, delta) -> jnp.ndarray:
    """``U_3(T)``, shape ``(3, 3)``, from midpoint samples of the broadcast waveform.

    Same "newest on the left" contract as :func:`curve_opt.propagate.chain`
    (that module's numerical-contract discipline): ``associative_scan``
    combines batches of ``(3, 3)`` propagators with the later step on the
    left. Only the final propagator is returned -- G only ever needs
    ``U_3(T)``, not the intermediate trajectory the two-level chain keeps for
    the space curve.
    """
    om_x = jnp.asarray(om_x_mid, dtype=jnp.float64)
    om_y = jnp.asarray(om_y_mid, dtype=jnp.float64)
    N = om_x.shape[0]
    dt = T / N
    steps = _three_level_cell_propagators(om_x, om_y, delta, dt)
    # ★ newest on the left, same contract as propagate.chain's U_edge scan.
    U_edge = jax.lax.associative_scan(lambda A, B: jnp.einsum("kij,kjl->kil", B, A), steps)
    return U_edge[-1]
