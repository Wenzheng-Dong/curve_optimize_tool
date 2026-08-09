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

★ This contract binds ``U_mid`` exactly as much as it binds the ``U_edge``
scan -- ``U_mid[k]`` is ``(half step at cell k) @ U_left[k]``, half step
(later) on the left. From Step 11 through F01, ``chain()`` had this backwards
(``U_left @ half_step``), degrading ``U_mid`` -- and everything built from it
(``tangent``, ``r_mid``, ``closure``, ``area``, F01's ``tantrix_area`` /
``leakage_amplitude``) -- from O(dt^2) to O(dt) on any genuinely non-commuting
input, while being byte-identical on planar/collinear inputs (where it commutes
and the order does not matter). It went undetected because the only existing
``U_mid`` test (``test_every_propagator_is_unitary``) checks unitarity, and
``U_1 U_2`` is exactly as unitary as ``U_2 U_1`` for unitary ``U_1, U_2`` --
that check cannot see an ordering error at all. Fixed and regression-tested in
F01b (see ``_dev_logs/F01b_umid_order.md``); the ``U_edge`` scan itself was
always correct, which is why endpoint-only quantities (``propagator()``,
``gate_residual`` on planar/general layers alike) were never affected.

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
full steps, and ``U_mid = (half step) @ U_edge[k-1]`` (half step, being later in
time, on the left -- see the ordering contract above). The space curve then
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

F01 (full-cost era): tantrix area and leakage amplitude
---------------------------------------------------------
Two more quantities read off the *same* forward scan -- no second propagator
call, ever (``_plan_full_cost.md`` §2.2, ``AGENTS.md`` full-cost discipline).
Both take a :class:`Chain3D` (or, via the ``_of_coeffs`` / ``_of_samples``
wrappers, coefficients or raw ``(Omega_x, Omega_y)`` samples) and do pure
elementwise algebra on its ``omega_x``, ``omega_y``, ``tangent`` and ``U_mid``
fields -- nothing here differentiates or re-propagates.

``tantrix_area`` (C3, evaluated on the *design* curve, ``_plan_full_cost.md``
§2.3/§2.5b)::

    A_T = (1/2) int_0^T [T (T . Omega) - Omega] ds ,   Omega = (Omega_x, Omega_y, 0)

Sign pinned by ``_plan_full_cost.md`` §2.2 (``T_dot = T x Omega``, *not*
``Omega x T``): in the planar limit ``T = (0, sin theta, cos theta)`` has
``T_x = 0`` so ``T . Omega = 0`` identically, and the bracket collapses to
``-Omega``, giving ``A_T = -(theta / 2) x_hat`` -- the closed form of §2.5b.
No special-casing is needed for that limit; it falls out of the general
formula because ``T . Omega`` is already zero there.

``leakage_amplitude`` (C4, evaluated on the *broadcast* waveform,
``_plan_full_cost.md`` §2.2 recollection A)::

    Lambda_j = (1 / sqrt(2)) int_0^T [Omega_x(s) + i Omega_y(s)]
               exp(i Delta s) U_c(s)[1, j] ds ,   j = 0, 1

returned as the 4 real components ``[Re L0, Im L0, Re L1, Im L1]`` in that
fixed order (later modules depend on it). ``Delta = alpha`` (negative, rad/ns,
``curve_opt.device.Device.delta``); the exponent is ``exp(+i Delta s)``.
``U_c(s)[1, j]`` is ``chain.U_mid[:, 1, j]`` -- row 1, i.e. ``<1|U_c(s)|j>`` --
matching the row/column convention Novera's ``ideal_propagators`` uses
(``pulse-shape-Novera`` git hash ``59fb616``: ``u10 = ideal_propagators(...)[:, 1, 0]``).
No ``-i`` prefactor here (that is the v5.1 §2.2 form); Novera's
``leakage_amplitude`` carries one, so ``novera_amplitude == -1j * Lambda_0``
up to the two secondary differences documented at the call sites in
``tests/test_f01_forward_chain.py`` (Novera integrates by trapezoid with edge
propagators, this module by midpoint; ``to_physical`` is deliberately bypassed
in that comparison).

``leakage_amplitude_ddelta`` is the analytic ``d Lambda / d Delta`` -- ``U_c``
does not depend on ``Delta``, so differentiating under the integral only
multiplies the integrand by ``i s``::

    dLambda_j/dDelta = (1 / sqrt(2)) int_0^T [Omega_x + i Omega_y] (i s)
                        exp(i Delta s) U_c(s)[1, j] ds

checked against ``jax.grad`` of :func:`leakage_amplitude` w.r.t. ``delta`` in
the test suite (should agree to <= 1e-10, since both differentiate the same
smooth integrand -- one symbolically, one by autodiff through the quadrature).
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
    "leakage_amplitude",
    "leakage_amplitude_ddelta",
    "leakage_amplitude_ddelta_of_coeffs",
    "leakage_amplitude_ddelta_of_samples",
    "leakage_amplitude_of_coeffs",
    "leakage_amplitude_of_samples",
    "noncommutativity",
    "omega_samples",
    "propagator",
    "su2_rotation",
    "tantrix_area",
    "tantrix_area_of_coeffs",
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
    # ★ same "newest on the left" contract as U_edge above: the half-step (later
    # in time) goes on the left of U_left (earlier). Reversed order was a
    # pre-existing (Step 11) bug -- see F01b's dev log and the module docstring's
    # "Numerical contract" section for the O(dt) -> O(dt^2) diagnosis.
    U_mid = _cell_propagators(om_x, om_y, dt, 0.5) @ U_left

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


# --------------------------------------------------------------------------
# F01: tantrix area (C3) and leakage amplitude (C4) -- see the module
# docstring section "F01 (full-cost era)". Both are pure reductions of an
# already-built Chain3D: no propagator is re-run.
# --------------------------------------------------------------------------


def _mid_times(c: Chain3D) -> jnp.ndarray:
    """Cell-midpoint times ``(k + 1/2) dt`` for ``k = 0 .. N-1``.

    Determined entirely by ``c.dt`` and ``N = len(c.omega_x)``, i.e. exactly
    :func:`curve_opt.geometry.midpoint_grid`'s grid -- recomputed here rather
    than stored on :class:`Chain3D` because it is pure arithmetic on fields
    the chain already carries, and adding a field would touch every
    positional construction of the namedtuple.
    """
    N = c.omega_x.shape[0]
    return (jnp.arange(N, dtype=jnp.float64) + 0.5) * c.dt


def tantrix_area(c: Chain3D) -> jnp.ndarray:
    """``A_T = (1/2) int_0^T [T (T . Omega) - Omega] ds``, shape (3,).

    C3 (``_plan_full_cost.md`` §2.3): the leading-order amplitude-error
    invariant, evaluated on the **design** curve. Pure algebra on ``c.tangent``
    and the field samples ``c.omega_x, c.omega_y`` -- midpoint sum, no
    derivative, no second propagator call (§2.2 recollection B). Planar limit:
    ``T . Omega = 0`` identically (``T_x = 0``), so this reduces to
    ``-(theta / 2) x_hat`` with no special-casing, matching §2.5b's closed
    form.
    """
    zero = jnp.zeros_like(c.omega_x)
    omega = jnp.stack([c.omega_x, c.omega_y, zero], axis=-1)
    dot = jnp.sum(c.tangent * omega, axis=-1, keepdims=True)
    integrand = c.tangent * dot - omega
    return 0.5 * jnp.sum(integrand, axis=0) * c.dt


def tantrix_area_of_coeffs(a, b, T: float, N: int = geometry.N_DEFAULT) -> jnp.ndarray:
    """:func:`tantrix_area` from sine-series coefficients -- runs :func:`chain` once."""
    return tantrix_area(_chain_of_coeffs(a, b, T, N))


def _leakage_amplitude_complex(c: Chain3D, delta: float) -> jnp.ndarray:
    """``Lambda`` as a complex 2-vector ``[Lambda_0, Lambda_1]`` -- shared core
    of :func:`leakage_amplitude` and the real/imaginary bookkeeping the public
    API exposes. See the module docstring for the formula.
    """
    s = _mid_times(c)
    envelope = c.omega_x + 1j * c.omega_y
    phase = jnp.exp(1j * jnp.asarray(delta, dtype=jnp.float64) * s)
    weight = envelope * phase * c.dt
    row1 = c.U_mid[:, 1, :]  # (N, 2): U_c(s)[1, 0], U_c(s)[1, 1]
    return jnp.sum(weight[:, None] * row1, axis=0) / jnp.sqrt(2.0)


def _to_real4(lam: jnp.ndarray) -> jnp.ndarray:
    """Complex 2-vector -> ``[Re L0, Im L0, Re L1, Im L1]`` -- the fixed order
    the module docstring pins.
    """
    return jnp.stack(
        [jnp.real(lam[0]), jnp.imag(lam[0]), jnp.real(lam[1]), jnp.imag(lam[1])]
    )


def leakage_amplitude(c: Chain3D, delta: float) -> jnp.ndarray:
    """C4 leakage amplitude, shape (4,): ``[Re L0, Im L0, Re L1, Im L1]``.

    ``Lambda_j = (1/sqrt2) int_0^T [Omega_x + i Omega_y] exp(i delta s)
    U_c(s)[1, j] ds``, ``j = 0, 1``. Evaluated on the **broadcast** waveform
    (``_plan_full_cost.md`` §2.3, "求值对象"): pass a ``Chain3D`` built from
    the played-out ``(Omega_x, Omega_y)``, not necessarily the design curve.
    See the module docstring for the Novera cross-check (``-1j`` prefactor
    difference) and the sign/exponent convention.
    """
    return _to_real4(_leakage_amplitude_complex(c, delta))


def leakage_amplitude_of_coeffs(
    a, b, T: float, delta: float, N: int = geometry.N_DEFAULT
) -> jnp.ndarray:
    """:func:`leakage_amplitude` from sine-series coefficients -- runs :func:`chain` once."""
    return leakage_amplitude(_chain_of_coeffs(a, b, T, N), delta)


def leakage_amplitude_of_samples(om_x_mid, om_y_mid, T: float, delta: float) -> jnp.ndarray:
    """:func:`leakage_amplitude` from raw ``(Omega_x, Omega_y)`` midpoint samples.

    Parametrization-free, like :func:`chain` itself -- the entry point F02's
    DRAG/Stark-corrected broadcast waveform (not expressible as sine-series
    coefficients) will use. Runs :func:`chain` once.
    """
    return leakage_amplitude(chain(om_x_mid, om_y_mid, T), delta)


def leakage_amplitude_ddelta(c: Chain3D, delta: float) -> jnp.ndarray:
    """Analytic ``d Lambda / d delta``, shape (4,), same real-component order.

    ``U_c`` does not depend on ``delta``, so this is the same integral with an
    extra ``(i s)`` factor -- see the module docstring. Cross-checked against
    ``jax.grad`` of :func:`leakage_amplitude` in the test suite.
    """
    s = _mid_times(c)
    envelope = c.omega_x + 1j * c.omega_y
    phase = jnp.exp(1j * jnp.asarray(delta, dtype=jnp.float64) * s)
    weight = envelope * (1j * s) * phase * c.dt
    row1 = c.U_mid[:, 1, :]
    lam = jnp.sum(weight[:, None] * row1, axis=0) / jnp.sqrt(2.0)
    return _to_real4(lam)


def leakage_amplitude_ddelta_of_coeffs(
    a, b, T: float, delta: float, N: int = geometry.N_DEFAULT
) -> jnp.ndarray:
    """:func:`leakage_amplitude_ddelta` from sine-series coefficients."""
    return leakage_amplitude_ddelta(_chain_of_coeffs(a, b, T, N), delta)


def leakage_amplitude_ddelta_of_samples(
    om_x_mid, om_y_mid, T: float, delta: float
) -> jnp.ndarray:
    """:func:`leakage_amplitude_ddelta` from raw ``(Omega_x, Omega_y)`` midpoint samples."""
    return leakage_amplitude_ddelta(chain(om_x_mid, om_y_mid, T), delta)


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
