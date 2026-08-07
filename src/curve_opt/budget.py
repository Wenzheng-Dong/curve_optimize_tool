"""The full-cost era's soft objective: C1-C4, weighted by device noise moments (F04).

``_plan_full_cost.md`` §2.4::

    C = (1/6)<delta_z^2>  ||r(T)||^2        (C1, closure)
      + (1/6)<delta_z^4>  ||A_r||^2          (C2, curve area)
      + (2/3)<epsilon^2>  ||A_T||^2          (C3, tantrix area)
      + (1/3)             ||Lambda||^2       (C4, leakage)

The gate (G), the c1 endpoint condition (P) and the peak ceiling are *not*
here -- they are hard constraints, assembled in :mod:`curve_opt.optimize`.
This module only owns the four *soft* budget terms and the weights that turn
noise-characterization numbers into them (task brief §1.1, §2.4a).

★ Evaluation object discipline (CLAUDE.md, task brief §2.1) -- the single
easiest thing to get wrong in this step: C1/C2/C3 are evaluated on the
**design curve**, C4 on the **broadcast** (DRAG/Stark-readout) waveform. Two
different forward passes, both built here (:func:`design_chain`,
:func:`broadcast_chain`) so no caller can accidentally share one chain
between the two.

The design curve is *not* :func:`curve_opt.parametrization.design_curve`'s
``(kappa, tau, Phi)`` fields fed to a Frenet formula -- module F02's whole
point (its own docstring, "Parametrization decision") is that ``kappa`` and
``Phi`` are literally the polar magnitude/heading of a *complex envelope*,
``kappa * exp(i Phi) = Omega_x_design + i Omega_y_design``. So the design
curve's own (uncorrected) two-quadrature field is::

    Omega_x_design = kappa * cos(Phi),   Omega_y_design = kappa * sin(Phi)

feeding :func:`curve_opt.propagate.chain` exactly as any other
``(Omega_x, Omega_y)`` sampling would (:func:`design_waveform_of_curve`,
:func:`design_chain`). This is the effective two-level model C1/C2/C3 are
derived in (``_plan_full_cost.md`` §2.5b): SW reduction guarantees that once
the *broadcast* waveform is played out, the effective curvature/torsion come
back to exactly this designed pair -- see that section for why DRAG's
``Omega_y`` correction must *not* be fed to C1/C2/C3 (the "DRAG buys no
amplitude robustness" result the plan corrected in v5.1).

Gauss-Newton structure
-----------------------
Every one of C1-C4 is a *weighted sum of squares* of a residual block, so the
whole objective is ``||R(x)||^2`` for the 13-component
:func:`residual_vector` (``sqrt(w_i) *`` each block, concatenated) --
:mod:`curve_opt.optimize` builds the Gauss-Newton objective Hessian
``2 J_R^T J_R`` directly from this one function, the same "drop the
residual's own second derivative" approximation the constraint block has
used since Step 08 (``_plan.md`` §3.1), now applied to the objective because
it is literally in that sum-of-squares form.

``<delta_z^4>``: a Gaussian assumption, stated once
-----------------------------------------------------
C2's weight needs the fourth moment of the quasi-static Z-noise distribution,
which the device only characterizes to second order (``static_detuning`` is a
Ramsey/echo-measured *rate*, i.e. a standard deviation, not a full
distribution). :func:`weights_from_device` assumes delta_z is (quasi-static,
zero-mean) Gaussian, so ``<delta_z^4> = 3 <delta_z^2>^2`` -- the standard
kurtosis identity, adopted here per the task brief (§2.2) rather than derived
from data. ★ This makes C2's weight ``2.60e-14 -> 7.79e-14`` ns^-4 relative to
the illustrative number in ``_plan_full_cost.md`` §2.5's table, which computes
``(1/6)<delta_z^2>^2`` (i.e. implicitly *not* applying the Gaussian kurtosis
factor of 3); the F04 task brief explicitly asks for the Gaussian assumption
to be implemented and documented, so that is what this module does -- the
discrepancy with §2.5's table is flagged in ``_dev_logs/F04_budget_objective.md``
for leader disposition rather than silently reconciled in either direction.
Because C2's contribution is ~1e-8 (three orders below C1 and six below C3;
§2.5's own "free rider" characterization), the factor-of-3 choice does not
change any qualitative conclusion.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from curve_opt import geometry, parametrization, propagate
from curve_opt.device import DEFAULT_DEVICE, Device

__all__ = [
    "BudgetTerms",
    "Weights",
    "broadcast_chain",
    "broadcast_waveform",
    "budget_terms",
    "c3_planar_floor",
    "design_chain",
    "design_waveform_of_curve",
    "residual_vector",
    "weights_from_device",
]


class Weights(NamedTuple):
    """The four budget weights, each a noise-moment prefactor (§2.4/§2.4a).

    Not tunable knobs: a set of weights *is* a noise-environment assumption
    (§2.4a) -- change ``device.static_detuning``/``control_error`` to explore
    one, never these numbers directly.
    """

    c1: float
    """``(1/6) <delta_z^2>``, rad^2/ns^2."""
    c2: float
    """``(1/6) <delta_z^4>``, rad^4/ns^4 -- Gaussian assumption, module docstring."""
    c3: float
    """``(2/3) <epsilon^2>``, dimensionless."""
    c4: float
    """``1/3`` exactly -- fixed by the budget derivation, not a device moment."""


def weights_from_device(device: Device = DEFAULT_DEVICE) -> Weights:
    """Build :class:`Weights` from a device's characterized noise moments.

    ``<delta_z^4> = 3 <delta_z^2>^2`` (Gaussian quasi-static assumption,
    module docstring). ``device.static_detuning_rate`` and
    ``device.control_error`` are the two characterization numbers this
    project sweeps (F06) -- the weights change with them, deterministically.
    """
    dz2 = device.static_detuning_rate ** 2
    dz4 = 3.0 * dz2 ** 2
    eps2 = device.control_error ** 2
    return Weights(c1=dz2 / 6.0, c2=dz4 / 6.0, c3=2.0 * eps2 / 3.0, c4=1.0 / 3.0)


def design_waveform_of_curve(curve: parametrization.DesignCurve):
    """``(Omega_x, Omega_y)`` of the *design* curve: ``kappa * (cos Phi, sin Phi)``.

    The complex envelope ``kappa * exp(i Phi)`` *is* ``Omega_x_design + i
    Omega_y_design`` in the effective two-level model the design control
    parametrizes (module docstring). Not the broadcast/readout waveform --
    that is :func:`broadcast_waveform`, a different function on purpose (the
    evaluation-object discipline this module exists to keep straight).
    """
    envelope = curve.kappa * jnp.exp(1j * curve.Phi)
    return jnp.real(envelope), jnp.imag(envelope)


def design_chain(a, c, Phi_0: float, T: float, N: int = geometry.N_DEFAULT) -> propagate.Chain3D:
    """The design curve's forward chain -- what C1/C2/C3 are evaluated on.

    ``design_curve(a, c, Phi_0, ...) -> design_waveform_of_curve -> chain``.
    One call builds everything :func:`residual_vector` needs from the design
    side; no second propagator call is hidden anywhere downstream.
    """
    curve = parametrization.design_curve(a, c, Phi_0, T, N)
    om_x, om_y = design_waveform_of_curve(curve)
    return propagate.chain(om_x, om_y, T)


def broadcast_waveform(a, c, Phi_0: float, T: float, delta, N: int = geometry.N_DEFAULT):
    """``(Omega_x, Omega_y)`` actually played out: the DRAG/Stark-corrected readout.

    Builds the design curve and calls
    :func:`curve_opt.parametrization.readout_waveform` directly --
    deliberately *not* :func:`curve_opt.parametrization.general_waveform`,
    which eagerly calls ``check_c1`` (numpy conversion + a Python ``raise``,
    not jit/grad-safe -- ``_plan_full_cost.md`` §4.3 note 2, task brief §2.4
    known hazard 1). The ``c1`` condition is validated once, outside any
    traced path, when :mod:`curve_opt.optimize` builds the constraint set
    (the linear ``P`` row is the actual enforcement mechanism; this function
    must stay differentiable end to end for both the objective and the gate
    constraint to use it).
    """
    curve = parametrization.design_curve(a, c, Phi_0, T, N)
    return parametrization.readout_waveform(curve.kappa, curve.kappa_dot, curve.tau, Phi_0, delta, T)


def broadcast_chain(
    a, c, Phi_0: float, T: float, delta, N: int = geometry.N_DEFAULT
) -> propagate.Chain3D:
    """The broadcast waveform's forward chain -- what C4 (and G, and peak) use."""
    om_x, om_y = broadcast_waveform(a, c, Phi_0, T, delta, N)
    return propagate.chain(om_x, om_y, T)


def residual_vector(
    a, c, Phi_0: float, T: float, device: Device = DEFAULT_DEVICE, N: int = geometry.N_DEFAULT
) -> jnp.ndarray:
    """The 13-component ``R`` with ``C = R . R`` (module docstring's GN structure).

    Order: ``sqrt(w1) r(T)`` (3), ``sqrt(w2) A_r`` (3), ``sqrt(w3) A_T`` (3),
    ``sqrt(w4) Lambda`` (4) -- matching ``_plan_full_cost.md`` §4.3's residual
    count (13) and the C1/C2/C3/C4 order everywhere else in the project.
    """
    w = weights_from_device(device)
    dchain = design_chain(a, c, Phi_0, T, N)
    r_T = dchain.closure
    A_r = 0.5 * dchain.area
    A_T = propagate.tantrix_area(dchain)

    bchain = broadcast_chain(a, c, Phi_0, T, device.delta, N)
    Lambda = propagate.leakage_amplitude(bchain, device.delta)

    return jnp.concatenate(
        [
            jnp.sqrt(w.c1) * r_T,
            jnp.sqrt(w.c2) * A_r,
            jnp.sqrt(w.c3) * A_T,
            jnp.sqrt(w.c4) * Lambda,
        ]
    )


class BudgetTerms(NamedTuple):
    """The four weighted budget terms plus their total, and the raw invariants.

    Mirrors :class:`curve_opt.metrics.CostTerms`'s "decomposable after the
    fact" role for the full-cost era: :mod:`curve_opt.recorder` (once
    extended for a budget-mode run) would checkpoint ``c1..c4``/``total``
    exactly like ``cost_energy``/``cost_curv``/``cost_peak`` today. The raw
    invariants (``r_T``, ``A_r``, ``A_T``, ``Lambda``) are the unweighted
    physical quantities, useful for reporting independent of any weight
    choice.
    """

    c1: float
    c2: float
    c3: float
    c4: float
    total: float
    r_T: jnp.ndarray
    A_r: jnp.ndarray
    A_T: jnp.ndarray
    Lambda: jnp.ndarray


def budget_terms(
    a, c, Phi_0: float, T: float, device: Device = DEFAULT_DEVICE, N: int = geometry.N_DEFAULT
) -> BudgetTerms:
    """Evaluate all four budget terms plus their sum, decomposed for reporting.

    ``R = residual_vector(...)`` sliced back into its four blocks; each term
    equals ``sum(R_block ** 2)`` exactly (the weight is folded into ``R`` as
    ``sqrt(w)``, so slicing and squaring recovers ``w * ||block||^2`` bit for
    bit -- no second formula, no risk of the two disagreeing).
    """
    w = weights_from_device(device)
    dchain = design_chain(a, c, Phi_0, T, N)
    r_T = dchain.closure
    A_r = 0.5 * dchain.area
    A_T = propagate.tantrix_area(dchain)

    bchain = broadcast_chain(a, c, Phi_0, T, device.delta, N)
    Lambda = propagate.leakage_amplitude(bchain, device.delta)

    c1 = w.c1 * jnp.sum(r_T ** 2)
    c2 = w.c2 * jnp.sum(A_r ** 2)
    c3 = w.c3 * jnp.sum(A_T ** 2)
    c4 = w.c4 * jnp.sum(Lambda ** 2)
    total = c1 + c2 + c3 + c4
    return BudgetTerms(c1=c1, c2=c2, c3=c3, c4=c4, total=total, r_T=r_T, A_r=A_r, A_T=A_T, Lambda=Lambda)


def c3_planar_floor(theta: float, control_error: float) -> float:
    """Analytic C3 floor of the ``planar``/``planar_drag`` layers (§2.5b).

    ``C3^planar = (2/3) epsilon^2 (theta/2)^2 = epsilon^2 theta^2 / 6`` --
    waveform-independent, a function of the gate angle alone (§2.5b's
    "DRAG buys no amplitude robustness" result). At ``epsilon = 2%``,
    ``theta = pi`` this is ``1.6449 epsilon^2 = 6.58e-4`` -- the acceptance
    target of task brief criterion (4).
    """
    return float(control_error) ** 2 * float(theta) ** 2 / 6.0
