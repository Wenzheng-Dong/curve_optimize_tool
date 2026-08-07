"""Three design-control layers plus the DRAG/Stark readout map (F02, full-cost era).

Data flow (``_plan_full_cost.md`` §4.1)::

    free params (a, c, Phi_0)
       -> design control (kappa, tau)      -- C1 / C2 / C3 evaluated *here*
       -> DRAG/Stark readout (Omega_x, Omega_y)  -- G / C4 / peak evaluated *here*

``kappa`` and ``Phi`` (``tau = Phi_dot``) are the free design curve's polar speed and heading;
the readout map turns them into the broadcast two-quadrature waveform that actually drives the
three-level system, correcting for leakage (DRAG, the ``kappa_y`` term) and for the AC-Stark
shift the drive itself induces (the ``Phi_prog`` phase term).

Parametrization decision (leader-pinned, task brief §2.1 -- not re-litigated here)
-----------------------------------------------------------------------------------
The design control is parametrized directly as ``(kappa, Phi)``, not as ``(Omega_x, Omega_y)``
with ``tau`` recovered by differentiating and dividing::

    tau = (Omega_x Omega_y_dot - Omega_y Omega_x_dot) / (Omega_x^2 + Omega_y^2)

That quotient is singular at ``kappa -> 0``, and the P (peak/periodicity -- here, sine-basis)
constraint *forces* ``kappa(0) = kappa(T) = 0`` identically, so both endpoints of *every* curve
sit exactly on the singularity. Parametrizing ``(kappa, Phi)`` directly removes the division
altogether: ``tau = Phi_dot`` is read off a second Fourier series and is analytic everywhere.

Basis choice for kappa (unchanged from the rest of the project)
-----------------------------------------------------------------
``kappa(t) = sum_n a_n sin(n pi t / T)``, i.e. :func:`curve_opt.geometry.omega_samples` applied
to the coefficient vector ``a`` -- reused verbatim rather than reimplemented, since "kappa" and
"Omega_x" are the same sine series in this parametrization (the planar layer *is* the ``a``-only
sine series; ``general`` just adds the second series ``c`` on top).

★ Basis choice for Phi (this module's own decision, per task brief §2.1's "you choose the basis
and justify it")
-----------------------------------------------------------------------------------------------
::

    Phi(t) = Phi_0 + sum_n c_n (cos(n pi t / T) - 1)

Two properties motivate this over any other Fourier family:

1. **``tau = Phi_dot`` comes out as the *same* sine series structure as kappa**, just
   differentiated and re-signed::

       tau(t) = -sum_n c_n (n pi / T) sin(n pi t / T)

   so no new basis machinery is needed for tau beyond what :func:`kappa_dot_of_coeffs` already
   needs for kappa_dot (a cosine derivative matrix) -- the two derivative matrices differ only in
   which series they came from.

2. **``Phi(0) = Phi_0`` is a structural identity**, not an imposed constraint: every term
   ``cos(0) - 1 = 0``, so ``Phi_0`` is *exactly* the design phase at ``t = 0`` for any ``c``,
   matching the readout formula's own use of ``Phi_0`` as the integration constant of
   ``Phi_prog`` (§2.3 below). No endpoint condition is needed on ``Phi`` itself (task brief
   §2.1: "Phi 本身无端点约束") -- and none is imposed here; ``Phi(T)`` is left free.

A further, unplanned bonus of this choice (documented because it changes what the ``c1``
guard in §2.4 has to check): since ``tau``'s series is a *sine* series like kappa's, it
inherits the same identity ``tau(0) = tau(T) = 0`` for *every* ``c`` -- with no extra linear
constraint required. Only ``kappa_dot`` (built from ``a``, a *cosine* derivative) can be nonzero
at the endpoints, so the ``c1`` guard below only ever needs to constrain ``a``, never ``c``.

### 2.3 Readout map (tex Stage 4, ``_plan_full_cost.md`` §4.1)

::

    kappa_y       = -kappa_dot / (Delta + tau)
    Phi_dot_prog  = tau - kappa^2 / (2 (Delta + tau))
    E_out         = (kappa + i kappa_y) exp(i Phi_0 + i int_0^s Phi_dot_prog ds')

``kappa_y`` is the DRAG leakage-cancelling quadrature; ``Phi_dot_prog`` is the programmed phase
rate, which *cancels* the AC-Stark shift the drive induces on the 0-1 transition (the SW
reduction renormalizes the design heading rate to ``tau_eff = tau + kappa^2 / (2 (Delta + tau))``,
``_plan_full_cost.md`` §2.3a; to cancel that shift the *played* rate must satisfy
``Phi_dot_prog + kappa^2 / (2 (Delta + tau)) = tau``, hence the minus sign -- §4.1's
2026-08-07 correction, see also the second-order 1->2 shift ``-Omega^2 / (2 Delta)``, itself
negative). The planar layer is the ``tau = 0`` special case: ``kappa_y = -kappa_dot / Delta``,
``Phi_dot_prog = -kappa^2 / (2 Delta)`` -- the textbook DRAG form, checked independently in
``tests/test_f02_parametrization.py`` (acceptance ④, and the non-circular Stark-coefficient
scan added 2026-08-07).

``int_0^s Phi_dot_prog`` is accumulated by the same midpoint cumulative rule as
:mod:`curve_opt.geometry`'s space curve (:func:`_midpoint_cumulative` below mirrors
``geometry._cumulative``, restricted to the values at cell midpoints since that is all the
readout map needs).

★ **Signed kappa is safe.** Unlike a Frenet ``kappa = |Omega_x|`` (which kinks at every zero
crossing), this map is exactly invariant under ``kappa -> -kappa, Phi -> Phi + pi``: negating
``kappa`` negates both ``kappa_dot`` and ``kappa_y``, ``Phi_dot_prog`` is unchanged (only
``kappa^2`` enters), so ``Phi_prog -> Phi_prog + pi`` and ``E_out = (kappa + i kappa_y)
e^{i Phi_prog}`` picks up two independent sign flips that cancel exactly. No special-casing is
needed to use *signed* ``kappa`` throughout -- checked as acceptance ③.

### 2.4 The ``c1`` guard is an entry ticket, not optional polish

``kappa_y`` is proportional to ``kappa_dot``, and on the sine basis ``kappa_dot(0) =
sum_n a_n (n pi / T) != 0`` in general -- so the broadcast ``Omega_y`` does not vanish at the
endpoints unless ``a`` satisfies the ``c1`` condition (``basis.c1_row(M, T, 'start'/'end') . a
= 0``, i.e. ``sum_n n a_n = 0`` and ``sum_n (-1)^n n a_n = 0``). Without it, the leakage
invariant ``Lambda`` becomes basis-dependent -- not a well-defined design target (CLAUDE.md
numerical discipline, DRAG readout section). This module does not optimize under that
constraint (F04's job); it only provides :func:`check_c1` / :func:`c1_residuals` and makes the
two DRAG-readout layer entry points (:func:`planar_drag_waveform`, :func:`general_waveform`)
call :func:`check_c1` eagerly, raising rather than silently returning a basis-dependent result.

★ ``check_c1``/``c1_residuals`` convert their input to a concrete numpy array and branch in
plain Python -- by design (a hard *raise*, not a differentiable penalty), but that means these
two entry points are not safe to put under ``jax.grad``/``jax.jit`` as written. F04, which
*will* need to differentiate through the readout map during optimization, will have to keep the
``c1`` check outside the traced objective (e.g. validate the coefficients once per iteration
from plain Python, not embed the check inside the function being differentiated) -- flagged for
F04, not resolved here (F02 does no optimization).

Layers (task brief §2.2 table)
-------------------------------
* ``planar``       -- free params ``a``;         design ``Phi = 0``; no readout.
* ``planar_drag``  -- free params ``a``;         design ``Phi = 0``; DRAG/Stark readout, ``tau
  = 0`` special case.
* ``general``      -- free params ``a, c, Phi_0``; true 3D design; full DRAG/Stark readout.

``general_waveform`` at ``c = 0, Phi_0 = 0`` must reduce to ``planar_drag_waveform`` bit for bit
(structural self-check, task brief §3 "另加一条"): both compute ``kappa``/``kappa_dot`` through
the same matrix product, and ``tau`` is either literally ``jnp.zeros_like(kappa)``
(``planar_drag_waveform``) or ``tau_of_coeffs(0, ...)`` which multiplies the same zero
coefficients through the derivative matrix and is therefore also exactly zero -- no rounding
difference is expected, let alone tolerated at only 1e-14.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from curve_opt import basis, geometry

__all__ = [
    "DesignCurve",
    "c1_residuals",
    "check_c1",
    "design_curve",
    "drag_stark_quadratures",
    "general_waveform",
    "kappa_dot_of_coeffs",
    "phi_shape_of_coeffs",
    "planar_drag_waveform",
    "planar_waveform",
    "readout_waveform",
    "tau_of_coeffs",
]


def _harmonics(M: int) -> np.ndarray:
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")
    return np.arange(1, M + 1, dtype=float)


def _cos_matrix(t, T: float, M: int) -> np.ndarray:
    """``C[k, n] = cos(n pi t_k / T)`` for ``n = 1..M``.

    The companion of :func:`curve_opt.basis.design_matrix` (the sine matrix): together they hold
    the sine series and its exact term-by-term derivative, since ``d/dt sin(n pi t / T) =
    (n pi / T) cos(n pi t / T)``. Kept local to this module (plain numpy, same pattern as
    ``basis.design_matrix``) since F02 is scoped to this one file.
    """
    t = np.asarray(t, dtype=float)
    n = _harmonics(M)
    return np.cos(np.pi * np.outer(t, n) / T)


def kappa_dot_of_coeffs(a, T: float, N: int = geometry.N_DEFAULT) -> jnp.ndarray:
    """Analytic ``kappa_dot`` at the cell midpoints: ``sum_n a_n (n pi / T) cos(n pi t / T)``.

    The exact term-by-term derivative of the sine series -- never a finite difference (task
    brief pitfall 2). The derivative matrix depends only on the grid and ``M`` (numpy), so the
    matrix-vector product with ``a`` stays differentiable, exactly as
    :func:`curve_opt.geometry.omega_samples` does for ``kappa`` itself.
    """
    a = jnp.asarray(a)
    M = a.shape[0]
    t_mid, _ = geometry.midpoint_grid(T, N)
    n = _harmonics(M)
    Cdot = _cos_matrix(t_mid, T, M) * (n * np.pi / T)[None, :]
    return jnp.asarray(Cdot) @ a


def tau_of_coeffs(c, T: float, N: int = geometry.N_DEFAULT) -> jnp.ndarray:
    """Analytic ``tau = Phi_dot`` at the cell midpoints: ``-sum_n c_n (n pi/T) sin(n pi t/T)``.

    Derivative of :func:`phi_shape_of_coeffs`'s cosine series (module docstring §"basis choice
    for Phi"). Vanishes at ``t = 0`` and ``t = T`` identically for *any* ``c`` -- the sine
    factor, same structural zero the ``kappa`` series itself has -- so no ``c1``-style
    constraint on ``c`` is ever needed (unlike ``a``, see :func:`check_c1`).
    """
    c = jnp.asarray(c)
    Mc = c.shape[0]
    t_mid, _ = geometry.midpoint_grid(T, N)
    n = _harmonics(Mc)
    S = jnp.asarray(basis.design_matrix(t_mid, T, Mc)) * (n * np.pi / T)
    return -(S @ c)


def phi_shape_of_coeffs(c, T: float, N: int = geometry.N_DEFAULT) -> jnp.ndarray:
    """Analytic ``Phi - Phi_0`` at the cell midpoints: ``sum_n c_n (cos(n pi t/T) - 1)``.

    ``Phi_shape(0) = 0`` identically (every term is ``cos(0) - 1 = 0``) -- an algebraic
    identity, not an imposed constraint, so ``Phi_0`` (added by the caller, see
    :func:`design_curve`) is exactly the design phase at ``t = 0`` for any ``c``.
    """
    c = jnp.asarray(c)
    Mc = c.shape[0]
    t_mid, _ = geometry.midpoint_grid(T, N)
    Cm1 = jnp.asarray(_cos_matrix(t_mid, T, Mc) - 1.0)
    return Cm1 @ c


class DesignCurve(NamedTuple):
    """One pass of the ``general``-layer design control at the cell midpoints.

    ``kappa``/``kappa_dot`` come from the sine series ``a``; ``tau``/``Phi`` come from the
    shifted-cosine series ``c`` plus the scalar ``Phi_0``. Parametrization-free downstream of
    the coefficients: everything past this point (:func:`readout_waveform`) only ever sees the
    four sampled arrays.
    """

    T: float
    dt: float
    t_mid: jnp.ndarray
    kappa: jnp.ndarray
    kappa_dot: jnp.ndarray
    tau: jnp.ndarray
    Phi: jnp.ndarray


def design_curve(a, c, Phi_0: float, T: float, N: int = geometry.N_DEFAULT) -> DesignCurve:
    """Build the ``general``-layer design curve's midpoint samples from ``(a, c, Phi_0)``.

    ``kappa`` reuses :func:`curve_opt.geometry.omega_samples` verbatim (the two are the same
    sine series in this parametrization); ``kappa_dot``, ``tau``, ``Phi`` come from the
    functions above. C1/C2/C3 (task brief / CLAUDE.md "each cost on the object its own
    derivation names") are evaluated on this object, never on the readout output.
    """
    t_mid, dt = geometry.midpoint_grid(T, N)
    kappa = geometry.omega_samples(a, T, N)
    kappa_dot = kappa_dot_of_coeffs(a, T, N)
    tau = tau_of_coeffs(c, T, N)
    Phi = Phi_0 + phi_shape_of_coeffs(c, T, N)
    return DesignCurve(
        T=T, dt=dt, t_mid=jnp.asarray(t_mid), kappa=kappa, kappa_dot=kappa_dot, tau=tau, Phi=Phi
    )


def drag_stark_quadratures(kappa_mid, kappa_dot_mid, tau_mid, delta: float):
    """``(kappa_y, Phi_dot_prog)`` -- the two DRAG/Stark corrections, before phase accumulation.

    ``kappa_y = -kappa_dot / (delta + tau)`` is the DRAG leakage-cancelling quadrature;
    ``Phi_dot_prog = tau - kappa^2 / (2 (delta + tau))`` is the programmed phase rate (design
    heading rate minus the correction that cancels the AC-Stark shift the drive itself induces --
    module docstring §2.3's 2026-08-07 sign correction). Split out from
    :func:`readout_waveform` so the two textbook DRAG quantities can be checked directly against
    an independent reference (``tests/test_f02_parametrization.py`` acceptance ④), without also
    exercising the phase-accumulation machinery.

    No guard against ``delta + tau == 0``: physically ``|tau| << |Delta|`` is required anyway
    for the perturbative readout map itself to be valid (``eta`` small, ``_plan_full_cost.md``
    §2.5c) -- an explicit near-resonance guard is a design question for whoever imposes bounds
    on ``tau`` (F04), not this module's forward map. Flagged, not fixed, here.
    """
    kappa_mid = jnp.asarray(kappa_mid)
    kappa_dot_mid = jnp.asarray(kappa_dot_mid)
    tau_mid = jnp.asarray(tau_mid)
    denom = delta + tau_mid
    kappa_y = -kappa_dot_mid / denom
    phi_dot_prog = tau_mid - kappa_mid**2 / (2.0 * denom)
    return kappa_y, phi_dot_prog


def _midpoint_cumulative(f_mid, dt: float) -> jnp.ndarray:
    """Midpoint cumulative integral of *f_mid*, returned at the cell midpoints only.

    Same convention as ``curve_opt.geometry._cumulative``'s ``F_mid`` (edges are not needed
    here): ``F_mid[k] = int_0^{(k+1/2) dt} f dt``.
    """
    F_edge = jnp.concatenate([jnp.zeros(1), jnp.cumsum(f_mid) * dt])
    return F_edge[:-1] + 0.5 * dt * f_mid


def readout_waveform(kappa_mid, kappa_dot_mid, tau_mid, Phi_0: float, delta: float, T: float):
    """DRAG/Stark readout map: design ``(kappa, kappa_dot, tau)`` -> broadcast ``(Omega_x,
    Omega_y)``.

    Parametrization-free (takes raw midpoint samples, like
    :func:`curve_opt.propagate.chain`/:func:`curve_opt.geometry.chain`), so the coefficient
    layers below (:func:`planar_drag_waveform`, :func:`general_waveform`) are the only callers
    that need to know about ``a``/``c``. ``Delta -> infinity`` recovers the identity map exactly
    (module docstring §2.3, acceptance ①): both DRAG corrections vanish like ``1/Delta`` and
    ``Phi_prog -> Phi_0 + int tau = Phi`` (the design phase).
    """
    kappa_mid = jnp.asarray(kappa_mid)
    N = kappa_mid.shape[0]
    dt = T / N
    kappa_y, phi_dot_prog = drag_stark_quadratures(kappa_mid, kappa_dot_mid, tau_mid, delta)
    Phi_prog = Phi_0 + _midpoint_cumulative(phi_dot_prog, dt)
    envelope = (kappa_mid + 1j * kappa_y) * jnp.exp(1j * Phi_prog)
    return jnp.real(envelope), jnp.imag(envelope)


def c1_residuals(a, T: float) -> tuple[float, float]:
    """``(kappa_dot(0), kappa_dot(T))`` in closed form, via :func:`curve_opt.basis.c1_row`.

    Zero at both ends iff ``a`` satisfies the ``c1`` condition (module docstring §2.4). Exact --
    no grid, since ``basis.c1_row(M, T, end) . a`` *is* ``kappa_dot`` evaluated at the literal
    endpoint (``basis.c1_row``'s own docstring), not the nearest midpoint-grid sample (which
    would only be O(dt) close to the true endpoint value and could not certify a 1e-12 claim).
    """
    a_np = np.asarray(a, dtype=float)
    M = a_np.shape[0]
    r0 = float(basis.c1_row(M, T, "start") @ a_np)
    rT = float(basis.c1_row(M, T, "end") @ a_np)
    return r0, rT


def check_c1(a, T: float, tol: float = 1e-9) -> None:
    """Raise ``ValueError`` unless both :func:`c1_residuals` are within *tol* of zero.

    A hard error, not a warning (task brief §2.4/§validation acceptance ②): a DRAG-readout
    layer built from an ``a`` that violates this would produce a broadcast ``Omega_y`` whose
    endpoint value is basis-dependent, which is not a well-defined design target (module
    docstring §2.4). Called eagerly by :func:`planar_drag_waveform` and
    :func:`general_waveform`; not jit/grad-safe as written (module docstring §2.4 note).
    """
    r0, rT = c1_residuals(a, T)
    if abs(r0) > tol or abs(rT) > tol:
        raise ValueError(
            f"c1 endpoint condition violated: kappa_dot(0) = {r0:.6e}, "
            f"kappa_dot(T) = {rT:.6e} (tol = {tol:.1e}). A DRAG/Stark readout layer built from "
            "this 'a' would have a basis-dependent broadcast Omega_y endpoint (module "
            "docstring 'The c1 guard is an entry ticket'). Project 'a' onto the c1-orthogonal "
            "subspace (curve_opt.basis.c1_row(M, T, 'start'/'end')) before calling a DRAG layer."
        )


def planar_waveform(a, T: float, N: int = geometry.N_DEFAULT):
    """``planar`` layer: ``Omega_y = 0`` identically, no DRAG/Stark readout (table in module
    docstring). Broadcast waveform equals the design curve verbatim.
    """
    kappa = geometry.omega_samples(a, T, N)
    return kappa, jnp.zeros_like(kappa)


def planar_drag_waveform(
    a, T: float, delta: float, N: int = geometry.N_DEFAULT, c1_tol: float = 1e-9
):
    """``planar_drag`` layer: design ``Phi = 0`` (``tau = 0``), DRAG/Stark readout.

    The ``tau = 0`` special case of :func:`readout_waveform`: ``kappa_y = -kappa_dot / Delta``,
    ``Phi_dot_prog = -kappa^2 / (2 Delta)`` -- the textbook planar-DRAG form (acceptance ④).
    Raises via :func:`check_c1` if ``a`` does not satisfy the ``c1`` endpoint condition.
    """
    check_c1(a, T, tol=c1_tol)
    kappa = geometry.omega_samples(a, T, N)
    kappa_dot = kappa_dot_of_coeffs(a, T, N)
    tau = jnp.zeros_like(kappa)
    return readout_waveform(kappa, kappa_dot, tau, 0.0, delta, T)


def general_waveform(
    a,
    c,
    Phi_0: float,
    T: float,
    delta: float,
    N: int = geometry.N_DEFAULT,
    c1_tol: float = 1e-9,
):
    """``general`` layer: true 3D design curve ``(a, c, Phi_0)``, full DRAG/Stark readout.

    At ``c = 0, Phi_0 = 0`` this must reduce to :func:`planar_drag_waveform` bit for bit
    (module docstring, task brief's "structural self-check"). Raises via :func:`check_c1` if
    ``a`` does not satisfy the ``c1`` endpoint condition (``c`` never needs it, module
    docstring §2.4).
    """
    check_c1(a, T, tol=c1_tol)
    curve = design_curve(a, c, Phi_0, T, N)
    return readout_waveform(curve.kappa, curve.kappa_dot, curve.tau, Phi_0, delta, T)
