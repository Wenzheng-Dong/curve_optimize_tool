"""Sine basis: coefficients <-> waveforms, the gate row, analytic bounds.

Home of every closed-form expression in the project. Depends on numpy only
(no JAX, no scipy) so that the analytic formulas stay independently checkable
against the differentiable implementations.

Conventions
-----------
* Basis functions ``sin(n pi t / T)``, ``n = 1..M``; endpoints vanish
  identically for any coefficient vector -- an identity, never a penalty.
* Even harmonics contribute exactly zero to the gate angle; only odd harmonics
  move the gate. This split is the skeleton of the optimization structure.
* Gate row (planar): ``g_n = (T / (n pi)) * (1 - (-1)**n)``, so that the gate
  condition reads ``g . a = theta`` -- linear in the coefficients.
* Analytic bound: ``(int Omega^2 dt) * L >= theta_gate**2``, evaluated *per
  winding branch* (theta and theta + 2 pi are the same gate but different
  constraint surfaces; see ``_plan.md`` §8.3).

Coefficient vectors
-------------------
A planar problem carries ``a`` of length ``M``. The general case carries ``a``
and ``b``. The energy functions here take a single *flat* coefficient vector,
which is ``a`` in the planar case and the concatenation ``[a, b]`` in the
general case: the fluence only ever sees the sum of squared coefficients,

    int_0^T Omega^2 dt = int_0^T (Omega_x^2 + Omega_y^2) dt
                       = (T / 2) * (|a|^2 + |b|^2)

by orthogonality of the basis on ``[0, T]``. That identity is exact, so no
quadrature enters the objective at all.

Unit conventions: ``t`` and ``T`` share one time unit, ``Omega`` is a rate
(rad / time), the gate angle ``theta`` and all returned invariants are
dimensionless.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "design_matrix",
    "energy_bound",
    "energy_bound_truncated",
    "energy_gradient",
    "energy_hessian",
    "energy_invariant",
    "gate_angle",
    "gate_row",
    "min_norm_gate_only",
    "naive_coeffs",
    "odd_inverse_square_sum",
    "omega",
    "omega_dot",
]


def _harmonics(M: int) -> np.ndarray:
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")
    return np.arange(1, M + 1, dtype=float)


def design_matrix(t, T: float, M: int) -> np.ndarray:
    """Return ``S[k, n] = sin((n + 1) pi t_k / T)`` for ``n = 0..M-1``.

    The linear map from coefficients to waveform samples. Used for least-squares
    projection of an ansatz (Step 06) and for the epigraph peak constraints,
    which are linear in the coefficients precisely because this matrix exists.
    """
    t = np.asarray(t, dtype=float)
    n = _harmonics(M)
    return np.sin(np.pi * np.outer(t, n) / T)


def omega(a, t, T: float) -> np.ndarray:
    """Evaluate ``sum_n a_n sin(n pi t / T)`` at times *t*.

    Works for either field component: pass ``a`` for ``Omega_x``, ``b`` for
    ``Omega_y``. Zero at ``t = 0`` and ``t = T`` by construction.
    """
    a = np.asarray(a, dtype=float)
    return design_matrix(t, T, a.size) @ a


def omega_dot(a, t, T: float) -> np.ndarray:
    """Analytic time derivative ``sum_n a_n (n pi / T) cos(n pi t / T)``.

    Not used by the current constraint set; it is the closed form the future
    bandwidth term ``int Omega_dot^2 dt`` and the C1 switch-on condition
    ``sum_n n a_n = 0`` are built on (``_plan.md`` §2.3 row 7).
    """
    a = np.asarray(a, dtype=float)
    t = np.asarray(t, dtype=float)
    n = _harmonics(a.size)
    return np.cos(np.pi * np.outer(t, n) / T) @ (a * n * np.pi / T)


def gate_row(M: int, T: float) -> np.ndarray:
    """Return ``g`` with ``g_n = (T / (n pi)) (1 - (-1)^n)``, so ``g . a = theta``.

    Derivation: ``theta = int_0^T Omega_x dt`` and
    ``int_0^T sin(n pi t / T) dt = (T / (n pi)) (1 - cos(n pi))``, which is
    ``2T / (n pi)`` for odd ``n`` and exactly ``0`` for even ``n``. Hence the
    even harmonics are free shape knobs: they can be moved without touching the
    gate. In the planar layer this makes the gate condition a *linear* equality,
    imposed exactly (by affine elimination or a ``LinearConstraint``) rather
    than by quadrature.
    """
    n = _harmonics(M)
    return (T / (n * np.pi)) * (1.0 - (-1.0) ** n)


def gate_angle(a, T: float) -> float:
    """Gate angle ``theta = int_0^T Omega_x dt`` from the closed form ``g . a``.

    Exact -- no grid. :func:`curve_opt.geometry.gate_angle_numeric` computes the
    same quantity by quadrature; the two agreeing to <= 1e-9 is the acceptance
    criterion of Step 03.
    """
    a = np.asarray(a, dtype=float)
    return float(gate_row(a.size, T) @ a)


def energy_invariant(coeffs, T: float) -> float:
    """Scale-invariant fluence ``(int_0^T Omega^2 dt) * L`` with ``L = T``.

    Equals ``T^2 / 2 * |coeffs|^2`` exactly (orthogonality). Pass ``a`` for a
    planar problem or the concatenation ``[a, b]`` in general -- see the module
    docstring. This is the objective of the optimization: a convex quadratic,
    i.e. the problem is a minimum-norm problem in disguise.
    """
    c = np.asarray(coeffs, dtype=float)
    return float(0.5 * T**2 * (c @ c))


def energy_gradient(coeffs, T: float) -> np.ndarray:
    """Gradient of :func:`energy_invariant`: ``T^2 * coeffs``."""
    c = np.asarray(coeffs, dtype=float)
    return T**2 * c


def energy_hessian(n_coeffs: int, T: float) -> np.ndarray:
    """Hessian of :func:`energy_invariant`: the constant matrix ``T^2 I``.

    Handed to ``trust-constr`` as the analytic objective Hessian. On its own it
    buys nothing -- the constraint Hessian must be supplied together with it
    (zeroed = Gauss-Newton); the pair is what gave 5.9..25.3x (``_plan.md``
    §3.1).
    """
    return T**2 * np.eye(n_coeffs)


def energy_bound(theta: float) -> float:
    """Continuum lower bound ``(int Omega^2 dt) L >= theta^2``.

    Cauchy-Schwarz on ``theta = int_0^T Omega_x dt``:
    ``theta^2 <= (int Omega_x^2 dt) * T``. Equality would need constant
    ``Omega``, which the sine basis cannot produce (its endpoints vanish), so
    the bound is approached but never attained -- see
    :func:`energy_bound_truncated`.

    The absolute yardstick for the cost of robustness: for X(pi) on the pi
    branch the gate alone costs 9.87, gate + closure + zero area costs 81.77 =
    8.3 theta^2, and the rcp ansatz pays 33 theta^2.

    ★ theta is the *winding branch*, not the gate: theta and theta + 2 pi
    implement the same gate but this bound grows like theta^2 (R_x(pi/2):
    2.47 on the pi/2 branch vs 61.7 on the 5 pi/2 branch). Never report a cost
    multiple without the branch it was computed on (``_plan.md`` §8.3).
    """
    return float(theta) ** 2


def odd_inverse_square_sum(M: int) -> float:
    """``sum over odd n <= M of 1 / n^2``, which tends to ``pi^2 / 8``."""
    n = _harmonics(M)
    odd = n[n % 2 == 1]
    return float(np.sum(1.0 / odd**2))


def energy_bound_truncated(theta: float, M: int) -> float:
    """Lower bound at truncation order *M*: ``pi^2 theta^2 / (8 S_M)``.

    The gate-only minimum-norm solution is ``a = theta g / |g|^2``, whose energy
    is ``T^2 theta^2 / (2 |g|^2)``. With
    ``|g|^2 = (4 T^2 / pi^2) S_M`` and ``S_M`` from
    :func:`odd_inverse_square_sum` this reduces to ``pi^2 theta^2 / (8 S_M)``,
    independent of ``T`` as an invariant must be.

    Sanity: ``M = 1`` gives ``pi^2 theta^2 / 8`` -- exactly the naive single-mode
    baseline -- and ``S_M -> pi^2 / 8`` recovers :func:`energy_bound`. For
    ``theta = pi`` this predicts 12.18 at M=1, 9.97 at M=40 and 9.87 in the
    limit, which is what step00d measured.

    This is a bound *on the gate constraint alone*. Closure and zero area push
    the achievable value far above it (81.77 for X(pi)).
    """
    return np.pi**2 * float(theta) ** 2 / (8.0 * odd_inverse_square_sum(M))


def min_norm_gate_only(theta: float, T: float, M: int) -> np.ndarray:
    """Minimum-energy coefficients subject to the gate condition only.

    ``a = theta g / |g|^2``: the projection of the origin onto the gate
    hyperplane. Not robust (closure and area are ignored) -- it is the reference
    point that :func:`energy_bound_truncated` evaluates, and a legitimate
    "constraint-blind" ansatz for the comparison study.
    """
    g = gate_row(M, T)
    return float(theta) * g / (g @ g)


def naive_coeffs(theta: float, T: float, M: int = 1) -> np.ndarray:
    """The naive single-mode baseline: ``a_1 = pi theta / (2 T)``, rest zero.

    ``Omega_x = (pi theta / 2T) sin(pi t / T)`` realizes the gate angle *theta*
    exactly, with no free parameters. For ``theta = pi`` this is
    ``(pi^2 / 2T) sin(pi t / T)``, energy invariant ``pi^4 / 8 = 12.18``.

    Chosen as the normalization reference because it is ansatz-independent,
    analytic, and therefore fair by construction. Each winding branch gets its
    own naive reference at its own *theta* (``_plan.md`` §2.2b, §8.3).
    """
    a = np.zeros(M, dtype=float)
    a[0] = np.pi * float(theta) / (2.0 * T)
    return a
