"""Figures, drawn from a RunRecord and from nothing else.

Hard rule (``_plan.md`` §4.3 item 1): this module reads a RunRecord through
:mod:`curve_opt.recorder` and **never re-runs an optimization** to produce a
figure. It therefore must not import :mod:`curve_opt.optimize` or
``scipy.optimize``; ``tests/test_architecture.py`` enforces that mechanically.

Acceptance for the record system (``_plan.md`` §5.2) is stated as three figures
that must be producible from ``<run_id>/`` alone:

1. the waveforms ``Omega_x(t)``, ``Omega_y(t)`` at any checkpoint;
2. the iteration history of the total cost, of each individual cost term, and of
   each constraint residual;
3. the step-0 comparison, waveform before vs after projection.

Plus a cross-check of the stored values against the ones recomputed from the
stored coefficients (agreement <= 1e-12), and Pareto fronts with degenerate
solutions marked hollow-x.

Figure files are named ``stepNN_<shortname>_<figname>.png``.

Planned API (Step 07)
---------------------
``plot_waveform(record, iter=...)``, ``plot_history(record)``,
``plot_step0(record)``, ``plot_pareto(records)``.
"""

from __future__ import annotations

__all__: list[str] = []
