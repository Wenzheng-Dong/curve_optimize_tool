"""Individual cost terms, scale-invariant, plus the naive normalization.

Each cost is its own function. The total objective is only a combinator over
them. That is what makes any weight combination decomposable after the fact:
:mod:`curve_opt.recorder` stores every term separately, and
:mod:`curve_opt.plotting` re-derives them from the stored coefficients
(``_plan.md`` §4.3 item 4, §5.1 ``history.npz``).

Scope: rows 4-6 of the constraint table (``_plan.md`` §2.3) -- total curvature,
fluence, peak. The robustness residuals (closure, area) belong to
:mod:`curve_opt.geometry`; this module does not duplicate them.

Scale-invariant forms only (``_plan.md`` §2.4). With unit speed, ``L = T``:

    energy   (int_0^T Omega^2 dt) * L      dimensionless
    curv      int_0^T |Omega| dt           dimensionless already
    peak      T * max|Omega|               dimensionless

where ``Omega = |Omega|`` means ``|Omega_x|`` in the planar layer and
``sqrt(Omega_x^2 + Omega_y^2)`` in general -- the curvature of the space curve
in both cases.

Reference point: the naive single mode ``Omega_x = (pi theta / 2T) sin(pi t/T)``
is ansatz-independent and fully analytic,

    energy = pi^2 theta^2 / 8,   curv = theta,   peak = pi theta / 2

(for ``theta = pi``: 12.176, 3.1416, 4.9348), and therefore fair by
construction. Each winding branch gets its own naive reference at its own theta
(``_plan.md`` §8.3) -- :func:`naive_reference` takes theta for exactly that
reason, and normalizing across branches is a category error.

Smoothness contract
-------------------
* ``peak = max |Omega|`` is **not** differentiable in the coefficients (the
  argmax switches, and the absolute value kinks). It is reported here, but it
  enters the objective only through the epigraph form built in
  :mod:`curve_opt.optimize` (``_plan.md`` §3). ★ The reported peak and the
  epigraph constraint must share one grid, or the number in the record will not
  be the number the solver bounded; :func:`peak` therefore samples the same
  midpoint grid as :mod:`curve_opt.geometry`, and the grid maximum is a *lower*
  bound on the continuum maximum (O(dt^2) below it).
* ``int |Omega| dt`` kinks wherever the signed Omega_x crosses zero, which is
  the normal case, not an edge case: the optimized planar solution has
  ``curv = 6.638`` against a gate angle of ``pi``, and the excess is exactly the
  back-and-forth. Report-only quantity; :func:`smoothed_total_curvature` is the
  ``int sqrt(Omega^2 + delta^2) dt`` surrogate for the day it must be optimized.
* ``energy`` is the only one of the three that is smooth -- indeed an exact
  convex quadratic with no quadrature at all (see :mod:`curve_opt.basis`).

Robustness floor: :func:`clamp_robustness` clamps normalized robustness
measures below 1e-3 so that a machine-zero cannot spuriously beat an
already-robust solution (``_plan.md`` §2.4). It applies to *reported,
normalized* robustness ratios, never to the hard equality constraints
themselves, which are driven to 1e-16.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from curve_opt import basis

__all__ = [
    "CostTerms",
    "ROBUSTNESS_FLOOR",
    "clamp_robustness",
    "cost_terms",
    "energy_invariant",
    "naive_reference",
    "normalize_to_naive",
    "peak",
    "smoothed_total_curvature",
    "total_curvature",
]

#: See ``_plan.md`` §2.4.
ROBUSTNESS_FLOOR = 1e-3

#: Same default grid as :mod:`curve_opt.geometry`, deliberately: the peak report
#: and the epigraph constraints must be built on one grid.
N_DEFAULT = 4000


def _midpoints(T: float, N: int):
    """Return ``(t_mid, dt)`` -- the same grid :mod:`curve_opt.geometry` uses."""
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    dt = T / N
    return (np.arange(N) + 0.5) * dt, dt


def _magnitude(a, T: float, N: int, b=None):
    """``(|Omega|, dt)`` on the midpoint grid: the curvature of the space curve."""
    t, dt = _midpoints(T, N)
    om_x = basis.omega(a, t, T)
    if b is None:
        return np.abs(om_x), dt
    return np.hypot(om_x, basis.omega(b, t, T)), dt


def energy_invariant(a, T: float, b=None) -> float:
    """``(int Omega^2 dt) * L``. Delegates to the exact closed form.

    One implementation only -- :func:`curve_opt.basis.energy_invariant` -- so the
    stored value and any recomputation from coefficients agree bit for bit,
    which is what the <= 1e-12 double-channel check of ``_plan.md`` §5.2 needs.
    """
    coeffs = np.asarray(a) if b is None else np.concatenate([np.asarray(a), np.asarray(b)])
    return basis.energy_invariant(coeffs, T)


def total_curvature(a, T: float, N: int = N_DEFAULT, b=None) -> float:
    """``int_0^T |Omega| dt``, the total curvature. Dimensionless as it stands.

    Bounded below by ``|theta|`` (triangle inequality), with equality iff
    ``Omega_x`` never changes sign. The gap measures how much the pulse doubles
    back -- the geometric price of closure and zero area.
    """
    mag, dt = _magnitude(a, T, N, b)
    return float(np.sum(mag) * dt)


def smoothed_total_curvature(a, T: float, N: int = N_DEFAULT, b=None, delta: float = 1e-3) -> float:
    """``int sqrt(Omega^2 + delta^2) dt`` -- the differentiable surrogate.

    Overestimates :func:`total_curvature` by at most ``delta * T`` and decreases
    monotonically to it as ``delta -> 0``. Only needed if total curvature ever
    has to enter the objective; the plan keeps it as a report quantity.
    """
    mag, dt = _magnitude(a, T, N, b)
    return float(np.sum(np.sqrt(mag**2 + delta**2)) * dt)


def peak(a, T: float, N: int = N_DEFAULT, b=None) -> float:
    """``T * max|Omega|`` on the midpoint grid -- the hardware ceiling proxy.

    Non-smooth in the coefficients; see the module docstring. The grid maximum
    slightly *under*-reports the continuum maximum, so a solution that satisfies
    the epigraph bound on this grid can exceed it between grid points by
    O(dt^2); with N = 4000 that is far below any hardware margin, but it is the
    reason the report and the constraint must share the grid.
    """
    mag, _ = _magnitude(a, T, N, b)
    return float(T * np.max(mag))


class CostTerms(NamedTuple):
    """The three objective-relevant terms, all scale-invariant.

    Mirrors ``cost_energy`` / ``cost_curv`` / ``cost_peak`` in the
    ``history.npz`` schema of ``_plan.md`` §5.1.
    """

    energy: float
    curv: float
    peak: float


def cost_terms(a, T: float, N: int = N_DEFAULT, b=None) -> CostTerms:
    """Evaluate all three terms at once -- what the recorder checkpoints."""
    return CostTerms(
        energy=energy_invariant(a, T, b),
        curv=total_curvature(a, T, N, b),
        peak=peak(a, T, N, b),
    )


def naive_reference(theta: float, T: float = 1.0) -> CostTerms:
    """Analytic cost terms of the naive single mode at gate angle *theta*.

    ``energy = pi^2 theta^2 / 8``, ``curv = |theta|``, ``peak = pi |theta| / 2``.
    All three are independent of ``T``, as invariants must be. This is the
    normalization reference of ``_plan.md`` §2.4, and it is per winding branch:
    calling it with theta = pi/2 and with theta = 5 pi/2 gives two different
    references for the same gate, which is the point.
    """
    theta = float(theta)
    return CostTerms(
        energy=np.pi**2 * theta**2 / 8.0,
        curv=abs(theta),
        peak=np.pi * abs(theta) / 2.0,
    )


def normalize_to_naive(terms: CostTerms, theta: float) -> CostTerms:
    """Divide each term by the naive reference at the same *theta*.

    Ratios only ever compare within one winding branch (``_plan.md`` §8.3);
    across branches, report absolute invariants instead.
    """
    ref = naive_reference(theta)
    return CostTerms(
        energy=terms.energy / ref.energy,
        curv=terms.curv / ref.curv,
        peak=terms.peak / ref.peak,
    )


def clamp_robustness(value: float, floor: float = ROBUSTNESS_FLOOR) -> float:
    """Clamp a normalized robustness measure from below at *floor*.

    Prevents a machine-zero residual from producing an unbounded improvement
    ratio and thereby "beating" a solution that is already robust
    (``_plan.md`` §2.4). Never applied to the hard equality constraints.
    """
    return float(max(abs(float(value)), floor))
