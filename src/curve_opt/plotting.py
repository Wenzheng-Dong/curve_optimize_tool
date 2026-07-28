"""Figures, drawn from a RunRecord and from nothing else.

Hard rule (``_plan.md`` §4.3 item 1): a figure is drawn from stored run data and
**never** from a fresh optimization. Here that is a structural property rather
than a promise -- this module imports no optimizer, no solver, and no filesystem
or path API at all, not even :mod:`curve_opt.recorder`. Its functions take a
record *object* (anything exposing ``manifest`` / ``history`` / ``step0`` /
``final``), so there is no expression in this file that can reach the disk. Move
the run store away and these functions cannot be called at all, because nothing
can load a record for them; ``tests/test_architecture.py`` and
``tests/test_recorder.py`` check both halves of that.

What the record must support (``_plan.md`` §5.2 acceptance):

1. :func:`plot_waveform` -- ``Omega_x(t)``, ``Omega_y(t)`` at any checkpoint,
   rebuilt from the stored coefficients;
2. :func:`plot_history` -- the iteration history of the total cost, of each
   individual cost term, and of each constraint residual;
3. :func:`plot_step0` -- the step-0 comparison, waveform before vs after
   projection.

:func:`plot_pareto` adds the front, with degenerate solutions marked hollow-x.

Every figure is annotated with the run id and ``stop_reason``: a survey run that
stopped early may not be read as a converged one (``_plan.md`` §3.1). A record
whose manifest says ``synthetic`` gets a diagonal watermark -- fabricated data
must never be mistakable for a measurement.

Figure files are named ``stepNN_<shortname>_<figname>.png``. Path handling is the
caller's business: pass an already-built path as ``save_path``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from curve_opt import basis

__all__ = [
    "annotate_provenance",
    "plot_history",
    "plot_pareto",
    "plot_step0",
    "plot_waveform",
    "waveform_at",
]

_COST_STYLE = {
    "cost_total": ("k", "total"),
    "cost_energy": ("C0", "energy $(\\int\\Omega^2dt)L$"),
    "cost_curv": ("C1", "curvature $\\int|\\Omega|dt$"),
    "cost_peak": ("C2", "peak $T\\Omega_{max}$"),
}


def _manifest(record) -> dict:
    return record.manifest


def _finish(fig, record, save_path=None):
    annotate_provenance(fig, record)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def annotate_provenance(fig, record) -> None:
    """Stamp run id, stop reason and the synthetic watermark onto *fig*."""
    man = _manifest(record)
    stop = man.get("stop_reason", "unknown")
    fig.text(
        0.005,
        0.005,
        f"{man.get('run_id', '?')}  |  stop_reason={stop}  |  "
        f"M={man.get('M', '?')} T={man.get('T', '?')} N={man.get('N_grid', '?')}",
        fontsize=6,
        color="0.35",
    )
    if man.get("synthetic", False):
        fig.text(
            0.5,
            0.5,
            "SYNTHETIC",
            fontsize=52,
            color="red",
            alpha=0.13,
            ha="center",
            va="center",
            rotation=30,
            zorder=10,
        )


def waveform_at(record, checkpoint: int = -1, n_points: int = 2000):
    """Rebuild ``(t, Omega_x, Omega_y)`` from the coefficients stored at *checkpoint*.

    This is the operational meaning of "the coefficients alone reconstruct
    everything" (``_plan.md`` §5.1): the waveform is not stored, it is recomputed
    from ``coeffs_a`` / ``coeffs_b``.
    """
    history = record.history
    if not history:
        raise ValueError(f"run {record.run_id} has no history to plot")
    T = float(_manifest(record)["T"])
    a = np.asarray(history["coeffs_a"])[checkpoint]
    b = np.asarray(history["coeffs_b"])[checkpoint]
    t = np.linspace(0.0, T, n_points)
    return t, basis.omega(a, t, T), basis.omega(b, t, T)


def plot_waveform(record, checkpoint: int = -1, save_path=None, n_points: int = 2000):
    """★ §5.2(a): the control waveform at any checkpoint, from the record alone."""
    t, om_x, om_y = waveform_at(record, checkpoint, n_points)
    iters = np.asarray(record.history["iter"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(t, om_x, "C0", label=r"$\Omega_x$")
    if np.any(om_y):
        ax.plot(t, om_y, "C3", label=r"$\Omega_y$")
    else:
        ax.plot([], [], "C3", label=r"$\Omega_y \equiv 0$ (planar)")
    ax.axhline(0.0, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\Omega$")
    ax.set_title(f"waveform at checkpoint {checkpoint} (iter {int(iters[checkpoint])})")
    ax.legend(fontsize=8)
    return _finish(fig, record, save_path)


def plot_history(record, save_path=None):
    """★ §5.2(b): total cost, every individual term, every constraint residual.

    Two panels sharing the iteration axis. The residual panel is where a run is
    judged: hard equalities are supposed to reach 1e-16, and a curve that flattens
    out at 1e-6 is the signature of a grid artifact rather than of convergence.
    """
    history = record.history
    if not history:
        raise ValueError(f"run {record.run_id} has no history to plot")
    iters = np.asarray(history["iter"])

    fig, (ax_cost, ax_res) = plt.subplots(
        2, 1, figsize=(6.8, 6.0), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0]}
    )

    for key, (color, label) in _COST_STYLE.items():
        if key in history:
            values = np.asarray(history[key], dtype=float)
            ax_cost.plot(iters, np.abs(values), color=color, label=label, lw=1.4)
    ax_cost.set_yscale("log")
    ax_cost.set_ylabel("cost (scale-invariant)")
    ax_cost.legend(fontsize=8, ncol=2)
    ax_cost.set_title("optimization history (total and per-term)")
    ax_cost.grid(alpha=0.25)

    for key, color, label in (
        ("res_gate", "C4", "gate"),
        ("res_closure", "C5", "closure"),
        ("res_area", "C6", "area"),
    ):
        if key not in history:
            continue
        values = np.abs(np.asarray(history[key], dtype=float))
        if values.ndim > 1:
            values = values.max(axis=1)
        ax_res.plot(iters, np.maximum(values, 1e-18), color=color, label=label, lw=1.4)
    ax_res.set_yscale("log")
    ax_res.set_xlabel("iteration")
    ax_res.set_ylabel("|constraint residual|")
    ax_res.legend(fontsize=8)
    ax_res.grid(alpha=0.25)
    return _finish(fig, record, save_path)


def plot_step0(record, save_path=None):
    """★ §5.2(c): the projection error -- waveform before vs after, plus the table.

    The left panel is the ansatz as it arrived against its least-squares
    projection; the right panel lists what the projection did to the gate angle,
    the closure and the area. Values that are unavailable (a non-planar ansatz has
    no after-projection robustness until the Step 11 propagator exists) are shown
    as ``n/a`` rather than as a zero.
    """
    step0 = record.step0
    if not step0:
        raise ValueError(f"run {record.run_id} has no step0 data to plot")

    times = np.asarray(step0["times"])
    fig, (ax, ax_txt) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [2, 1]})

    ax.plot(times, np.asarray(step0["omega_x_before"]), "0.45", lw=2.0, label=r"$\Omega_x$ ansatz")
    ax.plot(times, np.asarray(step0["omega_x_after"]), "C0", lw=1.3, label=r"$\Omega_x$ projected")
    if np.any(np.asarray(step0["omega_y_before"])):
        ax.plot(times, np.asarray(step0["omega_y_before"]), "0.75", lw=2.0, label=r"$\Omega_y$ ansatz")
        ax.plot(times, np.asarray(step0["omega_y_after"]), "C3", lw=1.3, label=r"$\Omega_y$ projected")
    ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\Omega$")
    ax.set_title("step 0: ansatz vs least-squares projection")
    ax.legend(fontsize=8)

    def _val(key, fmt="{:+.4e}"):
        if not bool(step0.get(f"{key}_available", np.array(True))):
            return "n/a"
        value = float(np.asarray(step0[key]))
        return "n/a" if np.isnan(value) else fmt.format(value)

    ansatz_meta = _manifest(record).get("ansatz", {})
    lines = [
        f"ansatz    : {ansatz_meta.get('name', '?')}",
        f"tier      : {ansatz_meta.get('tier_declared', '?')}"
        f" / measured {ansatz_meta.get('tier_measured', '?')}",
        f"protocol  : {ansatz_meta.get('projection_protocol', '?')}",
        "",
        f"relRMS    : {float(np.asarray(step0['rel_rms'])):.4e}",
        f"endpoint  : {float(np.asarray(step0['endpoint_ratio'])):.4e}  (|Om(0)|/peak)",
        f"winding   : {float(np.asarray(step0['winding_branch'])):+.6f}",
        "",
        "              before        after",
        f"theta    : {float(np.asarray(step0['theta_before'])):+.6f}   "
        f"{float(np.asarray(step0['theta_after'])):+.6f}",
        f"closure/L: {_val('closure_before')}  {_val('closure_after')}",
        f"area/L^2 : {_val('area_before')}  {_val('area_after')}",
    ]
    ax_txt.axis("off")
    ax_txt.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=7.5, va="top")
    return _finish(fig, record, save_path)


def plot_pareto(records, x_key: str = "cost_peak", y_key: str = "cost_energy", save_path=None):
    """The Pareto front over several records; degenerate points hollow-x.

    A point is marked degenerate when its run is not ``converged`` or when its
    manifest says ``synthetic`` -- exactly the runs whose numbers may not be read
    as claims (``_plan.md`` §3.1, §6.4).
    """
    records = list(records)
    if not records:
        raise ValueError("plot_pareto needs at least one record")
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for rec in records:
        x = float(np.asarray(rec.history[x_key])[-1])
        y = float(np.asarray(rec.history[y_key])[-1])
        stop = rec.manifest.get("stop_reason")
        degenerate = stop != "converged" or rec.manifest.get("synthetic", False)
        ax.plot(
            x,
            y,
            marker="x" if degenerate else "o",
            mfc="none" if degenerate else None,
            color="0.4" if degenerate else "C0",
            ms=7,
        )
        ax.annotate(rec.run_id, (x, y), fontsize=6, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title("Pareto front (hollow x = not a claim-run)")
    ax.grid(alpha=0.25)
    return _finish(fig, records[0], save_path)
