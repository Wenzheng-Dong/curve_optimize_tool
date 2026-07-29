#!/usr/bin/env python
"""Build X(pi) cost-trajectory comparisons from existing RunRecords only.

No optimizer is imported or called.  The exact projected curve-family input is
read from ``step0.npz``; intermediate points come from ``history.npz``; endpoint
performance comes from ``final.json``.  L1/M=12 and L2/M=14 are rendered
separately because their constraint maps and grids are different experiments.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402

from curve_opt import basis, geometry, metrics, propagate, recorder  # noqa: E402

OUT = ROOT / "output"
THRESHOLD = 1e-9
METRICS = (
    ("energy", "energy  $(\\int |\\Omega|^2dt)L$"),
    ("curv", "curvature  $\\int |\\Omega|dt$"),
    ("peak", "peak  $T\\Omega_{max}$"),
)

GROUPS = {
    "l1_m12": {
        "title": "X(pi), planar L1, M=12, N=20000",
        "run_ids": (
            "20260729_Xpi_rcp_lemniscate_M12_w0_claim",
            "20260729_Xpi_rcp_petal_M12_w0_claim",
            "20260729_Xpi_triangle_pulse_arc_M12_w0_claim",
            "20260729_Xpi_gaussian_arc_M12_w0_claim",
            "20260729_Xpi_treble_clef_loop_M12_w0_claim",
            "20260729_Xpi_circle_arc_M12_w0_claim",
            "20260729_Xpi_min_norm_gate_only_M12_w0_claim",
            "20260729_Xpi_naive_M12_w0_claim",
        ),
    },
    "l2_m14": {
        "title": "X(pi), general L2, M=14, N=4000",
        "run_ids": (
            "20260729_Xpi_alpha_3d_M14_L2_step16",
            "20260729_Xpi_tilted_lemniscate_M14_L2_step16",
            "20260729_Xpi_rcp_lemniscate_M14_L2_step16",
            "20260729_Xpi_rcp_petal_M14_L2_step16",
            "20260729_Xpi_naive_M14_L2_step16",
        ),
    },
}


def short_label(record) -> str:
    return str(record.manifest["ansatz"]["name"])


def load_group(key: str) -> list:
    records = [recorder.load(run_id) for run_id in GROUPS[key]["run_ids"]]
    signatures = {
        (
            record.manifest["target_gate"]["name"],
            record.manifest["layer"],
            record.manifest["M"],
            record.manifest["T"],
            record.manifest["N_grid"],
            record.manifest["objective"]["lambda"],
            record.manifest["solver"]["hessian_mode"],
        )
        for record in records
    }
    if len(signatures) != 1:
        raise ValueError(f"{key} mixes incomparable solver configurations: {signatures}")
    for record in records:
        if record.manifest["stop_reason"] != "converged":
            raise ValueError(f"{record.run_id} is not converged")
        if not record.step0 or not record.history or not record.final:
            raise ValueError(f"{record.run_id} is incomplete")
        recorder.verify_history(record, atol=0.0)
    return records


def start_terms(record) -> metrics.CostTerms:
    a = np.asarray(record.step0["coeffs_a"], dtype=float)
    b = np.asarray(record.step0["coeffs_b"], dtype=float)
    b_arg = b if record.manifest["layer"] == "general" else None
    return metrics.cost_terms(
        a,
        float(record.manifest["T"]),
        int(record.manifest["N_grid"]),
        b=b_arg,
    )


def start_residuals(record) -> tuple[float, float, float]:
    a = np.asarray(record.step0["coeffs_a"], dtype=float)
    b = np.asarray(record.step0["coeffs_b"], dtype=float)
    T = float(record.manifest["T"])
    N = int(record.manifest["N_grid"])
    theta = float(record.manifest["winding_branch"])
    if record.manifest["layer"] == "general":
        values = np.asarray(
            propagate.equality_residuals(
                a, b, T, N, propagate.target_x(theta)
            ),
            dtype=float,
        )
        gate = float(np.max(np.abs(values[:3])))
        closure = float(np.max(np.abs(values[3:6])))
        area = float(np.max(np.abs(values[6:9])))
    else:
        values = np.asarray(geometry.equality_residuals(a, T, N), dtype=float)
        gate = abs(float(basis.gate_row(len(a), T) @ a) - theta)
        closure = float(np.max(np.abs(values[:2])))
        area = float(np.max(np.abs(values[2:])))
    return gate, closure, area


def trajectory(record) -> list[dict]:
    """Exact projected input, stored checkpoints, and final performance."""
    name = short_label(record)
    rows = []
    initial = start_terms(record)
    rows.append(
        {
            "input": name,
            "stage": "projected_input",
            "display_step": 0,
            "stored_iteration": "",
            "energy": initial.energy,
            "curv": initial.curv,
            "peak": initial.peak,
        }
    )
    history = record.history
    for index, stored_iteration in enumerate(np.asarray(history["iter"], dtype=int)):
        rows.append(
            {
                "input": name,
                "stage": "checkpoint",
                "display_step": int(stored_iteration) + 1,
                "stored_iteration": int(stored_iteration),
                "energy": float(history["cost_energy"][index]),
                "curv": float(history["cost_curv"][index]),
                "peak": float(history["cost_peak"][index]),
            }
        )
    final = record.final["terms"]
    rows.append(
        {
            "input": name,
            "stage": "final",
            "display_step": int(record.final["nit"]) + 1,
            "stored_iteration": int(record.final["nit"]),
            "energy": float(final["energy"]),
            "curv": float(final["curv"]),
            "peak": float(final["peak"]),
        }
    )
    return rows


def residual_history(record) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start_gate, start_closure, start_area = start_residuals(record)
    iterations = np.concatenate(
        [[0], np.asarray(record.history["iter"], dtype=int) + 1]
    )

    def scalar(key: str, initial: float) -> np.ndarray:
        values = np.abs(np.asarray(record.history[key], dtype=float))
        if values.ndim > 1:
            values = values.max(axis=1)
        return np.concatenate([[initial], values])

    gate = scalar("res_gate", start_gate)
    closure = scalar("res_closure", start_closure)
    area = scalar("res_area", start_area)
    return iterations, gate, closure, area


def first_reach(iterations, values, threshold: float = THRESHOLD) -> int | None:
    hits = np.flatnonzero(np.asarray(values) <= threshold)
    return None if hits.size == 0 else int(np.asarray(iterations)[hits[0]])


def provenance(fig, records, group_title: str) -> None:
    commits = sorted({record.manifest["git_commit"][:8] for record in records})
    fig.text(
        0.005,
        0.005,
        f"RunRecord-only offline rendering | {group_title} | commits={','.join(commits)}",
        fontsize=6,
        color="0.35",
    )


def plot_trajectories(key: str, records: list, trajectories: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(records)))
    for record, color in zip(records, colors):
        name = short_label(record)
        rows = trajectories[name]
        x = np.asarray([row["display_step"] for row in rows], dtype=float)
        for ax, (metric, label) in zip(axes, METRICS):
            y = np.asarray([row[metric] for row in rows], dtype=float)
            ax.plot(x, y, color=color, lw=1.35, alpha=0.9, label=name)
            ax.scatter(x[0], y[0], s=32, facecolors="white", edgecolors=[color], zorder=4)
            ax.scatter(x[-1], y[-1], s=45, color=color, marker="*", zorder=4)
            ax.set_yscale("log")
            ax.set_xlabel("recorded optimization step (0 = projected input)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.22)
    axes[0].set_title("Energy objective trajectory")
    axes[1].set_title("Curvature performance trajectory")
    axes[2].set_title("Peak performance trajectory")
    axes[-1].legend(fontsize=6.7, loc="best")
    fig.suptitle(
        GROUPS[key]["title"]
        + "\nopen circle = exact projected input; star = final performance",
        fontsize=12,
    )
    provenance(fig, records, GROUPS[key]["title"])
    fig.tight_layout(rect=(0, 0.025, 1, 0.92))
    fig.savefig(OUT / f"xpi_{key}_cost_trajectories.png", dpi=170)
    plt.close(fig)


def plot_initial_final(key: str, records: list, trajectories: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.2), sharey=True)
    names = [short_label(record) for record in records]
    y = np.arange(len(names))
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(records)))
    for ax, (metric, label) in zip(axes, METRICS):
        panel_values = []
        for index, (name, color) in enumerate(zip(names, colors)):
            initial = trajectories[name][0][metric]
            final = trajectories[name][-1][metric]
            panel_values.extend([initial, final])
            ax.plot([initial, final], [index, index], color=color, lw=2.0, alpha=0.75)
            ax.scatter(initial, index, s=42, facecolors="white", edgecolors=[color], zorder=3)
            ax.scatter(final, index, s=48, color=color, marker="*", zorder=3)
        ax.set_xscale("log")
        # A narrow log range makes Matplotlib label many minor ticks and causes
        # overlap (especially the L2 energy panel). Four explicit geometric ticks
        # retain the log comparison while staying legible.
        ticks = np.geomspace(min(panel_values), max(panel_values), 4)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:.3g}"))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xlabel(label)
        ax.grid(alpha=0.22, axis="x")
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_title("Energy")
    axes[1].set_title("Curvature")
    axes[2].set_title("Peak")
    fig.suptitle(
        GROUPS[key]["title"]
        + "\npaired initial → final comparison (open circle → star)",
        fontsize=12,
    )
    provenance(fig, records, GROUPS[key]["title"])
    fig.tight_layout(rect=(0, 0.025, 1, 0.92))
    fig.savefig(OUT / f"xpi_{key}_initial_final.png", dpi=170)
    plt.close(fig)


def plot_feasibility_heatmap(key: str, records: list, summary_rows: list[dict]) -> None:
    names = [short_label(record) for record in records]
    by_name = {row["input"]: row for row in summary_rows if row["group"] == key}
    columns = ("gate_reach_1e-9", "robustness_reach_1e-9", "all_reach_1e-9", "nit")
    labels = ("gate <= 1e-9", "robustness <= 1e-9", "all <= 1e-9", "termination")
    values = np.asarray(
        [
            [
                np.nan if by_name[name][column] == "" else float(by_name[name][column])
                for column in columns
            ]
            for name in names
        ]
    )
    fig, ax = plt.subplots(figsize=(8.2, max(3.5, 0.55 * len(names) + 1.7)))
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, aspect="auto", cmap="viridis_r")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = "censored" if np.isnan(values[i, j]) else str(int(values[i, j]))
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if not np.isnan(values[i, j]) and values[i, j] > np.nanmax(values) / 2 else "black",
            )
    ax.set_xticks(np.arange(len(labels)), labels, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_title(GROUPS[key]["title"] + "\nfirst recorded step reaching feasibility")
    fig.colorbar(image, ax=ax, label="recorded optimization step")
    provenance(fig, records, GROUPS[key]["title"])
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(OUT / f"xpi_{key}_feasibility_heatmap.png", dpi=170)
    plt.close(fig)


def build_summary(key: str, records: list, trajectories: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for record in records:
        name = short_label(record)
        trace = trajectories[name]
        iterations, gate, closure, area = residual_history(record)
        robustness = np.maximum(closure, area)
        overall = np.maximum(gate, robustness)
        row = {
            "group": key,
            "input": name,
            "run_id": record.run_id,
            "implementation_commit": record.manifest["git_commit"],
            "layer": record.manifest["layer"],
            "M": record.manifest["M"],
            "N_grid": record.manifest["N_grid"],
            "stop_reason": record.manifest["stop_reason"],
            "nit": int(record.final["nit"]),
            "wall_clock_s": float(record.final["wall_clock_s"]),
            "gate_reach_1e-9": first_reach(iterations, gate),
            "robustness_reach_1e-9": first_reach(iterations, robustness),
            "all_reach_1e-9": first_reach(iterations, overall),
        }
        for metric, _ in METRICS:
            initial = float(trace[0][metric])
            final = float(trace[-1][metric])
            row[f"initial_{metric}"] = initial
            row[f"final_{metric}"] = final
            row[f"final_over_initial_{metric}"] = final / initial
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_summary = []
    all_trajectories = []
    audit = {}
    for key in GROUPS:
        records = load_group(key)
        trajectories = {
            short_label(record): trajectory(record) for record in records
        }
        summary = build_summary(key, records, trajectories)
        all_summary.extend(summary)
        for name, rows in trajectories.items():
            all_trajectories.extend(
                {"group": key, "run_id": next(r.run_id for r in records if short_label(r) == name), **row}
                for row in rows
            )
        plot_trajectories(key, records, trajectories)
        plot_initial_final(key, records, trajectories)
        plot_feasibility_heatmap(key, records, summary)
        audit[key] = {
            "title": GROUPS[key]["title"],
            "n_records": len(records),
            "run_ids": [record.run_id for record in records],
            "implementation_commits": sorted(
                {record.manifest["git_commit"] for record in records}
            ),
            "all_converged": all(
                record.manifest["stop_reason"] == "converged" for record in records
            ),
            "history_verified_atol": 0.0,
        }

    write_csv(OUT / "xpi_cost_summary.csv", all_summary)
    write_csv(OUT / "xpi_cost_trajectories.csv", all_trajectories)
    (OUT / "xpi_cost_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"wrote {len(all_summary)} input summaries and "
        f"{len(all_trajectories)} trajectory points without running optimization"
    )


if __name__ == "__main__":
    main()
