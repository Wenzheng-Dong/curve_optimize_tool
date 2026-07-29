"""General (L2) layer: JAX SU(2) propagator and the 3D constraints.

Used when ``Omega_y != 0``, i.e. for 3D ansatz inputs and for the future
curve-family members proposed by the user. Same public constraint API as the
planar layer (``gate`` / ``closure`` / ``area``) so the optimizer never has to
know which layer it is on; in the planar limit the two implementations must
agree to <= 1e-12 (regression test, Step 11).

Numerical contract -- the multiplication order is a *physics* bug if wrong
-------------------------------------------------------------------------
``U(t_k) = U_k U_{k-1} ... U_1``, newest on the left::

    Uc = jax.lax.associative_scan(lambda A, B: B @ A, Us)   # correct
    Uc = jax.lax.associative_scan(jnp.matmul, Us)           # reversed order

The wrong version is invisible on planar test cases (everything commutes) and
was caught only on a non-commuting case, where it deviated by 3.7e-2 from an
explicit numpy product and from qutip ``sesolve`` (``_plan.md`` §6.3).
Therefore: **any new propagator code must pass a non-commuting case plus an
independent second implementation (numpy / qutip) before use.** A planar case
does not count as that gate.

Time stepping uses the same midpoint rule as :mod:`curve_opt.geometry`. Float64
is enabled process-wide by importing :mod:`curve_opt` (JAX defaults to float32,
and every residual claim here lives at 1e-16).

Construction
------------
On cell ``k`` the drive is held constant at its midpoint value, so the exact
one-cell propagator is

    U_k = exp(-i (dt / 2) (Omega_x sigma_x + Omega_y sigma_y))
        = cos(phi) I - i sin(phi) / |Omega| * (Omega_x sigma_x + Omega_y sigma_y)

with ``phi = |Omega| dt / 2``. The apparent ``1 / |Omega|`` singularity is
removed exactly rather than regularized: ``sin(phi) / |Omega| = (dt / 2)
sinc(phi / pi)``, which is analytic at ``|Omega| = 0`` and differentiable there.
That matters in practice, not just in principle -- the optimizer drives
``Omega_y`` to zero and ``Omega_x`` crosses zero at interior points, so cells
with ``|Omega| ~ 0`` occur on the way to every solution, and an
``|Omega| + 1e-300`` guard would return a finite value with a meaningless
gradient.

Edge and midpoint propagators come from one scan: ``U_edge`` is the scan over
full steps, and ``U_mid = U_edge[k-1] @ (half step)``. The space curve then
follows the same midpoint quadrature as the planar layer, applied to the Bloch
components of ``U^dagger sigma_z U``.

The nine equality constraints
-----------------------------
``gate`` (3) + ``closure`` (3) + ``area`` (3), against ``_plan.md`` §2.3's count
for the general case. The gate block is the axis-angle mismatch of
``V = U_target^dagger U(T)``: with ``V = cos(chi) I - i sin(chi) n . sigma``,

    -Im tr(V sigma_a) / 2 = sin(chi) n_a ,

which vanishes for every component exactly when ``V = +-I``, i.e. when the gate
is reached up to a global phase. Using ``sin(chi) n`` rather than ``chi n``
keeps the residual an analytic function of the coefficients.

★ All three second-order area components must vanish, and for purely static
``sigma_z`` noise they can: the second-order Magnus term is proportional to
``int r x r_dot dt`` component-wise (``_plan.md`` §10, closed by derivation in
step01b).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from curve_opt import basis, geometry

__all__ = [
    "Chain3D",
    "PAULI",
    "area",
    "area_invariant",
    "chain",
    "closure",
    "closure_invariant",
    "equality_residuals",
    "gate_residual",
    "gate_rotation",
    "noncommutativity",
    "omega_samples",
    "propagator",
    "su2_rotation",
    "target_x",
    "unitarity_error",
]

SX = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
SY = jnp.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=jnp.complex128)
SZ = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
I2 = jnp.eye(2, dtype=jnp.complex128)

#: Stacked as (x, y, z) so that Bloch components come out in the same order the
#: planar layer uses for the space curve.
PAULI = jnp.stack([SX, SY, SZ])


def su2_rotation(axis, angle: float):
    """``exp(-i angle (axis . sigma) / 2)`` for a unit *axis*."""
    axis = jnp.asarray(axis, dtype=jnp.float64)
    axis = axis / jnp.linalg.norm(axis)
    half = 0.5 * angle
    return jnp.cos(half) * I2 - 1j * jnp.sin(half) * jnp.tensordot(axis, PAULI, axes=1)


def target_x(theta: float):
    """``R_x(theta) = exp(-i theta sigma_x / 2)`` -- the X-branch target gate."""
    return su2_rotation(jnp.array([1.0, 0.0, 0.0]), theta)


def omega_samples(a, b, T: float, N: int = geometry.N_DEFAULT):
    """Both field components at the cell midpoints, ``(Omega_x, Omega_y)``.

    Shares :func:`curve_opt.basis.design_matrix` with every other layer, so the
    basis formula exists in exactly one place.
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    t_mid, _ = geometry.midpoint_grid(T, N)
    S = jnp.asarray(basis.design_matrix(t_mid, T, a.shape[0]))
    return S @ a, S @ b


class Chain3D(NamedTuple):
    """One pass of the general chain on a given ``(Omega_x, Omega_y)`` sampling."""

    T: float
    dt: float
    omega_x: jnp.ndarray
    omega_y: jnp.ndarray
    U_edge: jnp.ndarray
    """(N, 2, 2) -- the propagator at the right edge of each cell; ``U_edge[-1]`` is U(T)."""
    U_mid: jnp.ndarray
    """(N, 2, 2) -- the propagator at each cell midpoint."""
    tangent: jnp.ndarray
    r_mid: jnp.ndarray
    closure: jnp.ndarray
    area: jnp.ndarray


def _cell_propagators(om_x, om_y, dt: float, fraction: float):
    """``exp(-i (fraction dt / 2) (Omega_x sigma_x + Omega_y sigma_y))`` per cell.

    Two separate things have to be handled at ``|Omega| = 0``, and only one of them
    is the division:

    * ``sin(phi) / |Omega|`` is written as ``(fraction dt / 2) sinc(phi / pi)``,
      exact and analytic at zero -- no epsilon guard;
    * ★ ``|Omega| = sqrt(Omega_x^2 + Omega_y^2)`` itself has an **infinite
      derivative** at the origin, so differentiating through it returns ``nan``
      even when the value is finite. That is the actual trap: the sinc rewrite
      alone still produced ``nan`` gradients. The magnitude is therefore taken of
      a ``jnp.where``-guarded argument, which makes the gradient flow through the
      constant branch at the origin and yields the correct value there (the true
      derivative of the propagator at zero field comes from the drive factor,
      which is linear, not from the ``cos`` factor, whose derivative does vanish).

    Cells with ``|Omega| ~ 0`` are not a corner case: the optimizer drives
    ``Omega_y`` to zero while ``Omega_x`` crosses zero at interior points, so they
    occur on the way to every solution.
    """
    drive = om_x[:, None, None] * SX + om_y[:, None, None] * SY
    squared = om_x**2 + om_y**2
    positive = squared > 0.0
    magnitude = jnp.sqrt(jnp.where(positive, squared, 1.0))
    step = 0.5 * fraction * dt
    phi = jnp.where(positive, magnitude * step, 0.0)
    return jnp.cos(phi)[:, None, None] * I2 - 1j * (step * jnp.sinc(phi / jnp.pi))[
        :, None, None
    ] * drive


def chain(om_x_mid, om_y_mid, T: float) -> Chain3D:
    """Run the general chain on midpoint samples of both field components.

    Parametrization-free, exactly as :func:`curve_opt.geometry.chain` is: the
    non-commuting arbitration cases of Step 11 are piecewise-constant controls
    that no sine series can represent, so they have to enter as samples.
    """
    om_x = jnp.asarray(om_x_mid, dtype=jnp.float64)
    om_y = jnp.asarray(om_y_mid, dtype=jnp.float64)
    N = om_x.shape[0]
    dt = T / N

    steps = _cell_propagators(om_x, om_y, dt, 1.0)
    # ★ newest on the left. jnp.matmul here would silently give the reverse order,
    # which is invisible unless the case is non-commuting (_plan.md §6.3).
    U_edge = jax.lax.associative_scan(lambda A, B: B @ A, steps)
    U_left = jnp.concatenate([I2[None], U_edge[:-1]])
    U_mid = U_left @ _cell_propagators(om_x, om_y, dt, 0.5)

    # Bloch components of U^dagger sigma_z U at the midpoints: the space curve's
    # unit tangent, same object the planar layer builds by trigonometry.
    U_dag = jnp.conj(jnp.swapaxes(U_mid, 1, 2))
    adjoint = jnp.einsum("kij,jl,klm->kim", U_dag, SZ, U_mid)
    tangent = jnp.real(jnp.einsum("kim,ami->ka", adjoint, PAULI)) / 2.0

    r_mid = jnp.cumsum(tangent, axis=0) * dt - 0.5 * tangent * dt
    return Chain3D(
        T=T,
        dt=dt,
        omega_x=om_x,
        omega_y=om_y,
        U_edge=U_edge,
        U_mid=U_mid,
        tangent=tangent,
        r_mid=r_mid,
        closure=jnp.sum(tangent, axis=0) * dt,
        area=jnp.sum(jnp.cross(r_mid, tangent), axis=0) * dt,
    )


def _chain_of_coeffs(a, b, T: float, N: int) -> Chain3D:
    om_x, om_y = omega_samples(a, b, T, N)
    return chain(om_x, om_y, T)


def propagator(a, b, T: float, N: int = geometry.N_DEFAULT):
    """``U(T)`` for the sine-series coefficients ``(a, b)``."""
    return _chain_of_coeffs(a, b, T, N).U_edge[-1]


def gate_residual(U_final, U_target):
    """``sin(chi) n``, shape (3,): zero exactly when the gate is reached up to phase.

    See the module docstring. ``U_target`` may be any SU(2) matrix; the residual
    is blind to a global phase, which is the physically correct equivalence.
    """
    V = jnp.conj(jnp.asarray(U_target)).T @ jnp.asarray(U_final)
    return -jnp.imag(jnp.einsum("ij,aji->a", V, PAULI)) / 2.0


def gate_rotation(U):
    """Return ``(angle, axis)`` of an SU(2) matrix, with ``angle`` in ``[0, 2 pi)``.

    Reporting helper: ``angle`` is the rotation the pulse actually performs, which
    is what identifies the winding branch of a 3D ansatz.
    """
    U = jnp.asarray(U)
    cos_half = jnp.real(jnp.trace(U)) / 2.0
    vec = -jnp.imag(jnp.einsum("ij,aji->a", U, PAULI)) / 2.0
    sin_half = jnp.linalg.norm(vec)
    angle = 2.0 * jnp.arctan2(sin_half, cos_half)
    axis = jnp.where(sin_half > 1e-14, vec / jnp.where(sin_half > 0, sin_half, 1.0), jnp.zeros(3))
    return angle, axis


def closure(a, b, T: float, N: int = geometry.N_DEFAULT):
    """``r(T)``, shape (3,) -- three equations in the general case, not two."""
    return _chain_of_coeffs(a, b, T, N).closure


def area(a, b, T: float, N: int = geometry.N_DEFAULT):
    """``int_0^T r x r_dot dt``, shape (3,) -- all three components must vanish."""
    return _chain_of_coeffs(a, b, T, N).area


def closure_invariant(a, b, T: float, N: int = geometry.N_DEFAULT) -> float:
    """``|r(T)| / L`` with ``L = T`` (unit speed holds here too: ``|tangent| = 1``)."""
    return float(jnp.linalg.norm(closure(a, b, T, N)) / T)


def area_invariant(a, b, T: float, N: int = geometry.N_DEFAULT) -> float:
    """``|area| / L^2`` -- a norm, since a general curve has no privileged component."""
    return float(jnp.linalg.norm(area(a, b, T, N)) / T**2)


def equality_residuals(a, b, T: float, N: int = geometry.N_DEFAULT, U_target=None):
    """The nine general equality constraints, in invariant form.

    ``[gate (3), closure / L (3), area / L^2 (3)]``. One function so ``jax.jacrev``
    produces the whole (9, 2M) Jacobian in a single pass -- which is also the
    dominant cost of the 3D layer (step00e measured the constraint Jacobian at
    37..180 ms against 0.07 ms for the objective, i.e. the propagator's forward
    evaluation is *not* the bottleneck).
    """
    if U_target is None:
        U_target = target_x(jnp.pi)
    c = _chain_of_coeffs(a, b, T, N)
    return jnp.concatenate(
        [
            gate_residual(c.U_edge[-1], U_target),
            c.closure / T,
            c.area / T**2,
        ]
    )


def unitarity_error(U) -> float:
    """``max |U^dagger U - I|`` -- a cheap health check on any propagator output."""
    U = jnp.asarray(U)
    return float(jnp.max(jnp.abs(jnp.conj(U).T @ U - I2)))


def noncommutativity(a, b, T: float, N: int = geometry.N_DEFAULT) -> float:
    """RMS commutator over all pairs of times -- zero **exactly** iff every H(t) commutes.

    Two drives commute iff their ``(Omega_x, Omega_y)`` vectors are collinear
    (parallel *or* anti-parallel), since
    ``[H_j, H_k] = (i / 2) (Omega_x^j Omega_y^k - Omega_y^j Omega_x^k) sigma_z``.
    Averaging that squared cross product over all pairs collapses to a 2x2
    second-moment determinant,

        noncommutativity = T^2 sqrt(2 (<Ox^2><Oy^2> - <Ox Oy>^2)) ,

    which is O(N) to evaluate, has no singularity, and vanishes identically for a
    collinear control.

    ⚠️ This is **not** step00e's diagnostic, which integrated ``|dPhi/dt|`` with
    ``Phi = arctan2(Omega_y, Omega_x)``. That quantity is contaminated by the
    signed-curvature convention this project deliberately adopts (``_plan.md``
    §2.2b): a strictly planar control whose ``Omega_x`` changes sign has ``Phi``
    jump by ``pi`` at each crossing, and the arctan2 measure reports ``2 pi`` of
    "non-commutativity" for a control in which every Hamiltonian commutes with
    every other. Measured on the M=8 planar case of the Step 11 tests: arctan2 says
    6.2832, this says exactly 0.0. So the absolute values of step00e's
    non-commutativity column are not reproduced here by construction; the ordering
    of its inputs and its conclusion (cost saturates near 2x) are what carry over.
    """
    om_x, om_y = omega_samples(a, b, T, N)
    mxx = jnp.mean(om_x**2)
    myy = jnp.mean(om_y**2)
    mxy = jnp.mean(om_x * om_y)
    determinant = jnp.maximum(mxx * myy - mxy**2, 0.0)
    return float(T**2 * jnp.sqrt(2.0 * determinant))
