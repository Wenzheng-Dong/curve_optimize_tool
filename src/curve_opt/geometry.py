"""Planar (L1) geometry chain: Omega_x => theta => T-vector => r => closure/area.

Valid only when ``Omega_y == 0``, where the control Hamiltonian commutes with
itself at all times and no propagator is needed: the gate angle is an integral,
the space curve is a pure quadrature. This is the main workhorse layer -- under
pure Z noise the optimum is planar (``_plan.md`` §7.3).

Written with ``jax.numpy`` so the constraint functions are differentiable
(``jax.jacrev`` supplies the constraint Jacobian, ``_plan.md`` §3.1). The basis
itself comes from :mod:`curve_opt.basis`: the design matrix depends on the grid
and on ``M`` but never on the coefficients, so a plain numpy matrix multiplied
by a traced coefficient vector stays differentiable, and the basis formula
lives in exactly one place.

The chain, and where the conventions are pinned
----------------------------------------------
With ``H_c = (1/2) Omega_x(t) sigma_x`` the control propagator is
``U_c(t) = exp(-i theta(t) sigma_x / 2)``, ``theta(t) = int_0^t Omega_x dt'``.
The error-frame integrand of ``_plan.md`` §2.1 is then

    U_c^dagger sigma_z U_c = cos(theta) sigma_z + sin(theta) sigma_y

(differentiate in theta and use ``[sigma_x, sigma_z] = -2 i sigma_y``), so with
``r . sigma = int_0^t U_c^dagger sigma_z U_c dt'`` the space curve is

    tangent  Tvec(t) = (0, sin(theta(t)), cos(theta(t)))      (components x,y,z)
    curve    r(t)    = (0, int_0^t sin(theta), int_0^t cos(theta))

Consequences, all of them checked in ``tests/test_geometry.py``:

* ``|Tvec| = 1`` identically -- unit speed, hence arc length ``L = T``. Every
  invariant here divides by ``T``, never by a separately integrated length.
* The curve lies in the ``y-z`` plane: ``r_x = 0`` identically. Closure
  ``r(T) = 0`` is therefore **2** equations, not 3.
* Curvature ``kappa = |dTvec/dt| = |theta_dot| = |Omega_x|``, i.e. ``Omega`` is
  the *signed* curvature: a sign change of ``Omega_x`` is an ordinary zero
  crossing, not a jump of the phase (that is why the series is placed on
  ``Omega_{x,y}`` and not on ``(Omega, Phi)``; ``_plan.md`` §2.2).
* The area vector ``int_0^T r x r_dot dt`` has only an x-component,
  ``int (r_y cos(theta) - r_z sin(theta)) dt``, so zero area is **1** equation.
  Planar totals: 2 + 1 nonlinear equalities on top of the linear gate row,
  matching the count in ``_plan.md`` §2.3.

Sign and normalization convention (pinned here, tested against closed forms)
---------------------------------------------------------------------------
``area`` is ``int_0^T r x r_dot dt`` verbatim -- **not** the enclosed area. For
a closed curve it equals twice the signed enclosed-area vector; the sign follows
the traversal orientation. A full circle traversed by constant ``Omega_x > 0``
gives ``area_x / L^2 = -1 / (2 pi)`` exactly, which is the reference value the
convention is fixed by. Both signs occur in real data (the naive baseline is
positive, rcp is negative), so the sign is reported, never taken absolute.

Reported invariants (``_plan.md`` §2.4): ``closure / L`` and ``area / L^2``.

Numerical contract
------------------
* Quadrature is **midpoint**, ``O(dt^2)``. First-order right-endpoint rules
  create *grid artifacts*: the optimizer exploits the discretization error and
  reports a 1e-9 residual where the true residual is 1.28e-2 (``_plan.md``
  §6.2). Refining the grid does not rescue a low-order rule.
* First-order Z cancellation <=> ``r(T) = 0`` (closure).
* Second-order <=> ``int_0^T r x r_dot dt = 0`` (projected area).
* Both cumulative integrals run on cell midpoints; values at cell edges come
  from the running sum, and the midpoint value of the next cell adds the exact
  half-step ``dt * f_k / 2``. For constant ``Omega`` this reproduces
  ``theta(t) = kappa t`` to machine precision, and the residual grid error is
  then purely the ``O(dt^2)`` of the outer quadrature.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from curve_opt import basis

__all__ = [
    "Chain",
    "area",
    "area_invariant",
    "chain",
    "closure",
    "closure_invariant",
    "curve",
    "equality_residuals",
    "gate_angle_numeric",
    "midpoint_grid",
    "omega_samples",
    "tangent",
    "theta_of_t",
]

#: Default grid for the planar layer. step00d used N = 4000 for optimization and
#: 4..25x refinements for the re-evaluation of any claim (``_plan.md`` §6.4).
N_DEFAULT = 4000


def midpoint_grid(T: float, N: int = N_DEFAULT):
    """Return ``(t_mid, dt)`` with ``t_mid_k = (k + 1/2) dt``, ``dt = T / N``.

    Cell midpoints, the only sample locations this module integrates on.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    dt = T / N
    return (np.arange(N) + 0.5) * dt, dt


def omega_samples(a, T: float, N: int = N_DEFAULT):
    """Sample ``Omega_x`` at the cell midpoints.

    The design matrix is built in numpy (it depends only on the grid, ``T`` and
    ``M``); only the matrix-vector product runs through JAX, which keeps the
    result differentiable in *a*.
    """
    a = jnp.asarray(a)
    t_mid, _ = midpoint_grid(T, N)
    S = jnp.asarray(basis.design_matrix(t_mid, T, a.shape[0]))
    return S @ a


def _cumulative(f_mid, dt):
    """Midpoint cumulative integral of *f_mid*, at cell edges and cell midpoints.

    Returns ``(F_edge, F_mid)`` where ``F_edge[j] = int_0^{j dt} f dt`` for
    ``j = 0..N`` and ``F_mid[k] = int_0^{(k + 1/2) dt} f dt``.
    """
    F_edge = jnp.concatenate([jnp.zeros(1), jnp.cumsum(f_mid) * dt])
    F_mid = F_edge[:-1] + 0.5 * dt * f_mid
    return F_edge, F_mid


class Chain(NamedTuple):
    """One pass of the geometry chain on a given ``Omega_x`` sampling.

    Attributes are the successive links: ``theta`` (edges and midpoints), the
    unit ``tangent`` at midpoints, the curve ``r`` (edges and midpoints), and the
    two robustness quantities ``closure = r(T)`` and ``area = int r x r_dot dt``.
    """

    T: float
    dt: float
    omega: jnp.ndarray
    theta_edge: jnp.ndarray
    theta_mid: jnp.ndarray
    tangent: jnp.ndarray
    r_edge: jnp.ndarray
    r_mid: jnp.ndarray
    closure: jnp.ndarray
    area: jnp.ndarray


def chain(om_mid, T: float) -> Chain:
    """Run the whole chain on ``Omega_x`` sampled at the cell midpoints.

    Parametrization-free: the sine series enters only through *om_mid*. That is
    what lets the acceptance tests drive this with a constant curvature (a
    circle), which the sine basis cannot represent -- constant ``Omega`` is
    exactly the ``forced`` tier of ``_plan.md`` §2.2b, so it has to come in as
    samples rather than as coefficients.

    ``N`` is inferred from ``len(om_mid)`` and ``dt = T / N``.
    """
    om_mid = jnp.asarray(om_mid)
    N = om_mid.shape[0]
    dt = T / N

    theta_edge, theta_mid = _cumulative(om_mid, dt)

    zero = jnp.zeros_like(theta_mid)
    Tvec = jnp.stack([zero, jnp.sin(theta_mid), jnp.cos(theta_mid)], axis=-1)

    r_edge = jnp.concatenate([jnp.zeros((1, 3)), jnp.cumsum(Tvec, axis=0) * dt], axis=0)
    r_mid = r_edge[:-1] + 0.5 * dt * Tvec

    return Chain(
        T=T,
        dt=dt,
        omega=om_mid,
        theta_edge=theta_edge,
        theta_mid=theta_mid,
        tangent=Tvec,
        r_edge=r_edge,
        r_mid=r_mid,
        closure=r_edge[-1],
        area=jnp.sum(jnp.cross(r_mid, Tvec), axis=0) * dt,
    )


def _chain_of_coeffs(a, T: float, N: int) -> Chain:
    return chain(omega_samples(a, T, N), T)


def theta_of_t(a, T: float, N: int = N_DEFAULT):
    """Accumulated rotation angle ``theta(t) = int_0^t Omega_x dt'``.

    Returns ``(theta_edge, theta_mid)``; ``theta_edge[-1]`` is the gate angle.
    """
    c = _chain_of_coeffs(a, T, N)
    return c.theta_edge, c.theta_mid


def gate_angle_numeric(a, T: float, N: int = N_DEFAULT) -> float:
    """Gate angle by quadrature -- the numerical counterpart of ``g . a``.

    Agreement with :func:`curve_opt.basis.gate_angle` to <= 1e-9 (at T = 1,
    N large) is the acceptance criterion of Step 03; the optimizer never uses
    this function, it imposes the exact linear row instead.
    """
    theta_edge, _ = theta_of_t(a, T, N)
    return float(theta_edge[-1])


def tangent(a, T: float, N: int = N_DEFAULT):
    """Unit tangent ``(0, sin theta, cos theta)`` at the cell midpoints, shape (N, 3)."""
    return _chain_of_coeffs(a, T, N).tangent


def curve(a, T: float, N: int = N_DEFAULT):
    """Space curve ``r(t)``. Returns ``(r_edge, r_mid)`` of shapes (N+1, 3), (N, 3)."""
    c = _chain_of_coeffs(a, T, N)
    return c.r_edge, c.r_mid


def closure(a, T: float, N: int = N_DEFAULT):
    """``r(T)``, shape (3,). Zero <=> first-order Z noise cancels.

    The x-component is identically zero in the planar layer, so the two
    non-trivial components are the actual equality constraints.
    """
    return _chain_of_coeffs(a, T, N).closure


def area(a, T: float, N: int = N_DEFAULT):
    """``int_0^T r x r_dot dt``, shape (3,). Zero <=> second order cancels too.

    Only the x-component is non-trivial in the planar layer. See the module
    docstring for the sign and normalization convention: this is the raw
    integral, twice the signed enclosed area for a closed curve, and its sign is
    reported rather than removed.

    cct's docs concede that the second-order condition has no API there
    (``_plan.md`` §2.1) -- this function is the point of the whole tool.
    """
    return _chain_of_coeffs(a, T, N).area


def closure_invariant(a, T: float, N: int = N_DEFAULT) -> float:
    """``|r(T)| / L`` with ``L = T``. Non-negative, scale-invariant."""
    return float(jnp.linalg.norm(closure(a, T, N)) / T)


def area_invariant(a, T: float, N: int = N_DEFAULT) -> float:
    """Signed ``area_x / L^2`` with ``L = T`` -- the planar reporting form.

    Signed on purpose: the naive baseline gives -0.295, rcp gives -3.3e-3, and
    collapsing that to a magnitude would throw away the orientation.
    """
    return float(area(a, T, N)[0] / T**2)


def equality_residuals(a, T: float, N: int = N_DEFAULT):
    """The three non-linear planar equality constraints, in invariant form.

    ``[r_y(T) / L, r_z(T) / L, area_x / L^2]`` -- 2 closure + 1 zero area, on top
    of the *linear* gate row handled exactly by :mod:`curve_opt.basis`. Written
    as one function so ``jax.jacrev`` produces the whole constraint Jacobian in
    a single pass, and normalized so that all three enter the solver's
    tolerance on the same footing.

    These are hard equality constraints, never soft penalties: they reach
    1e-16..1e-17 in practice, and no finite penalty weight holds the gate
    (LESSONS §4).
    """
    c = _chain_of_coeffs(a, T, N)
    return jnp.array([c.closure[1] / T, c.closure[2] / T, c.area[0] / T**2])
