"""Ansatz intake: any curve -> sine coefficients + a step-0 error report.

A *registry* interface. A new curve family -- a cct ``curve_families`` member,
a raw ``Omega`` sampling, or a 3D space curve ``r(t)`` -- only has to provide
samples or a closed form; it then gets the projection, the step-0 report and
admission to the optimizer for free, without touching the core modules
(``_plan.md`` §4.3 item 2).

Projection protocol (locked, ``_plan.md`` §2.2b)
-----------------------------------------------
The canonical protocol is **plain least squares, with no constraint repair**.
Variants that also fix the gate angle or the closure are *different protocols*
and must be labelled as such in the run manifest -- otherwise the choice of
repair becomes a confounder in the ansatz comparison, since different repairs
hand the optimizer different starting points and can flip the conclusion.

Every projection produces a **step-0 error** (relative RMS of Omega, plus the
gate / closure / area values before and after projection). It is quantified,
recorded and stored as iteration 0 of the run history -- not swept under the
rug as an implementation detail.

Representability tiers
----------------------
* ``exact``   -- natively a sine series: ``rcp_petal`` / ``rcp_lemniscate``.
  relRMS ~1e-4 at M=12, pure truncation.
* ``approx``  -- endpoint curvature zero or nearly zero: ``triangle_pulse_arc``
  (exactly 0, cusp => coefficients decay like 1/n^2), ``gaussian_arc``
  (~2.1% of peak).
* ``forced``  -- ``Omega(0) != 0``, a hard conflict with the basis:
  ``circle_arc`` and all cct 3D closed families. Gibbs-type endpoint error, the
  waveform deforms qualitatively. Still projected -- but any scientific reading
  must state that the effective ansatz is the *projected* object.

Each ansatz is optimized in its own native winding branch (the constraint
right-hand side theta is that ansatz's own total rotation angle), to avoid the
violent deformation of a cross-branch projection (``_plan.md`` §8.3).

Planned API (Step 06)
---------------------
``register(name, sampler, tier)``, ``project(name, M, T)`` ->
``(coeffs, Step0Report)``, ``list_ansatz()``.
"""

from __future__ import annotations

__all__: list[str] = []
