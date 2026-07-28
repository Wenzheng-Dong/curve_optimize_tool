#!/usr/bin/env python
"""Audit every registered ansatz family: what its robustness conditions measure.

Two stages, deliberately separated so that no number is ever transcribed by hand:

    measure   ->  audit.csv          (the only place numbers are produced)
    render    ->  TABLE.md, *.png    (read back from audit.csv, nothing recomputed)

``python make_report.py`` does both. ``python make_report.py --render-only``
does the second stage alone, which is also the check that the rendering really
reads from the stored data: delete nothing, change nothing, and the table and
figures come back identical.

Requires the upstream ``curvecontroltoolbox`` clone on PYTHONPATH for the cct
families (read-only; the audit never writes there). Without it, only the built-in
families are audited and the missing rows are reported as skipped.

    PYTHONPATH=../../../curvecontroltoolbox/src python make_report.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
CCT = REPO.parent / "curvecontroltoolbox" / "src"
if CCT.is_dir():
    sys.path.insert(0, str(CCT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from curve_opt import ansatz, basis, geometry  # noqa: E402

CSV_PATH = HERE / "audit.csv"
TABLE_PATH = HERE / "TABLE.md"

#: Fixed for the whole audit. M = 12 is the project default (_plan.md §2.2);
#: T = 1 is the convention every scale-invariant quantity is reported at.
M = 12
T = 1.0
N_GRID = 20_000
TARGET_THETA = np.pi  # X(pi), the first target gate of _plan.md §8.3

#: Families to audit, with the keyword arguments that select a member.
FAMILIES = [
    ("naive", {"theta": np.pi}),
    ("min_norm_gate_only", {"theta": np.pi}),
    ("rcp_lemniscate", {"turning_label": "pi"}),
    ("rcp_petal", {"turning_label": "pi"}),
    ("triangle_pulse_arc", {}),
    ("gaussian_arc", {}),
    ("circle_arc", {"theta": np.pi}),
    ("flora_petal", {}),
    ("tilted_lemniscate", {}),
    ("treble_clef_loop", {}),
    ("alpha_3d", {}),
    ("zeng_clifford_3d", {}),
    ("lissajous_3d", {}),
    ("helix", {}),
]

FIELDS = [
    "family",
    "tier_declared",
    "tier_measured",
    "planar",
    "endpoint_ratio",
    "rel_rms",
    "theta_measured",
    "winding_k",
    "gate_error_vs_branch",
    "on_pi_branch",
    "closure_source",
    "area_source",
    "closure_projected",
    "area_projected",
    "gate_error_projected",
    "closure_gate_fixed",
    "area_gate_fixed",
    "energy_projected",
    "energy_gate_fixed",
    "energy_bound_own_angle",
    "peak_projected",
    "curv_projected",
]


def gate_corrected(a, theta):
    """Minimum-norm shift onto the gate hyperplane: ``a + (theta - g.a) g / |g|^2``.

    ★ This is a *different projection protocol* from the canonical one
    (``_plan.md`` §2.2b locks plain least squares with no repair). It appears here
    only as a diagnostic, to answer how much the "free" exact gate correction
    costs in closure and area, and its columns are labelled ``*_gate_fixed``
    throughout. It is never used to start an optimization.
    """
    g = basis.gate_row(len(a), T)
    return a + (theta - g @ a) * g / (g @ g)


def measure() -> list[dict]:
    ansatz.register_builtin_families(overwrite=True)
    try:
        ansatz.register_cct_families()
        have_cct = True
    except ImportError as exc:
        print(f"! upstream cct not importable, auditing built-ins only: {exc}")
        have_cct = False

    rows = []
    for name, kwargs in FAMILIES:
        if name not in dict(ansatz.list_ansatz()):
            print(f"  skipped {name} (needs upstream cct: {have_cct})")
            continue
        src = ansatz.source(name, **kwargs)
        rep = ansatz.project(src, M, T, N_GRID)

        two_pi = 2.0 * np.pi
        k = int(np.round((rep.theta_before - TARGET_THETA) / two_pi))
        branch = TARGET_THETA + two_pi * k
        gate_error = abs(rep.theta_before - branch)

        row = {
            "family": name,
            "tier_declared": rep.tier_declared,
            "tier_measured": rep.tier_measured,
            "planar": int(rep.is_planar),
            "endpoint_ratio": rep.endpoint_ratio,
            "rel_rms": rep.rel_rms,
            "theta_measured": rep.theta_before,
            "winding_k": k,
            "gate_error_vs_branch": gate_error,
            "on_pi_branch": int(gate_error < 0.5),
            "closure_source": rep.closure_before,
            "area_source": rep.area_before,
            "closure_projected": rep.closure_after,
            "area_projected": rep.area_after,
            "gate_error_projected": abs(rep.theta_after - branch),
            # Both field components: for a non-planar family the fluence, the
            # curvature and the peak are all |Omega| = hypot(Omega_x, Omega_y),
            # never Omega_x alone (that under-reports, and can even put the
            # reported energy below the analytic bound of its own gate angle).
            "energy_projected": basis.energy_invariant(
                np.concatenate([rep.coeffs_a, rep.coeffs_b]), T
            ),
            "peak_projected": float(T * np.max(np.hypot(rep.omega_x_after, rep.omega_y_after))),
            "curv_projected": float(
                np.trapezoid(np.hypot(rep.omega_x_after, rep.omega_y_after), rep.times)
            ),
            "energy_bound_own_angle": float(rep.theta_after) ** 2,
            "closure_gate_fixed": None,
            "area_gate_fixed": None,
            "energy_gate_fixed": None,
        }

        if rep.is_planar:
            fixed = gate_corrected(rep.coeffs_a, branch)
            row["closure_gate_fixed"] = geometry.closure_invariant(fixed, T, N_GRID)
            row["area_gate_fixed"] = geometry.area_invariant(fixed, T, N_GRID)
            row["energy_gate_fixed"] = basis.energy_invariant(fixed, T)

        rows.append(row)
        print(f"  audited {name}")
    return rows


def write_csv(rows) -> None:
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in FIELDS})
    print(f"wrote {CSV_PATH.relative_to(REPO)}  ({len(rows)} families)")


def read_csv() -> list[dict]:
    with CSV_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key in ("family", "tier_declared", "tier_measured"):
                continue
            row[key] = None if value == "" else float(value)
    return rows


# --------------------------------------------------------------------------
# rendering -- reads audit.csv only
# --------------------------------------------------------------------------


def _fmt(value, spec="{:+.3e}"):
    return "n/a" if value is None else spec.format(value)


def render_table(rows) -> None:
    lines = [
        "<!-- GENERATED by make_report.py from audit.csv -- do not edit by hand -->",
        "",
        f"# Curve-family audit (M = {M}, T = {T:g}, N = {N_GRID}, target X(pi))",
        "",
        "All quantities are scale-invariant (`_plan.md` §2.4): `closure` is",
        "`|r(T)| / L`, `area` is the signed `area_x / L^2` in the planar layer and",
        "`|area| / L^2` for a space curve, `energy` is `(int Omega^2 dt) L`.",
        "",
        "## 1. Representability and projection error",
        "",
        "| family | tier (declared / measured) | planar | endpoint / peak | relRMS |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['family']}` | {r['tier_declared']} / {r['tier_measured']} "
            f"| {'yes' if r['planar'] else 'no'} | {_fmt(r['endpoint_ratio'], '{:.3e}')} "
            f"| {_fmt(r['rel_rms'], '{:.3e}')} |"
        )

    lines += [
        "",
        "## 2. Gate angle and winding branch",
        "",
        "`theta` is the total turning of the ansatz as it arrives. `k` identifies the",
        "winding branch of the target, `theta_target + 2 pi k`; `gate error` is the",
        "distance to that branch. A family flagged `off-branch` does not implement",
        "X(pi) on any branch -- it is audited for completeness, and every gate-related",
        "number in its row should be read as a distance to an unrelated target.",
        "",
        "| family | theta | k | gate error vs branch | on a branch of X(pi)? |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['family']}` | {r['theta_measured']:+.6f} | {int(r['winding_k'])} "
            f"| {_fmt(r['gate_error_vs_branch'], '{:.2e}')} "
            f"| {'yes' if r['on_pi_branch'] else '**off-branch**'} |"
        )

    lines += [
        "",
        "## 3. Robustness conditions: as it arrives, after projection, after a gate fix",
        "",
        "`source` is the ansatz itself. `projected` is after the canonical protocol",
        "(plain least squares onto M sine harmonics, no repair). `gate-fixed` adds the",
        "minimum-norm shift onto the gate hyperplane -- a **different protocol**, shown",
        "here only to price the 'free' exact gate correction. Non-planar families have",
        "no post-projection robustness values until the SU(2) propagator of Step 11:",
        "rebuilding r(t) from (a, b) needs it.",
        "",
        "| family | closure source | closure proj. | closure gate-fixed | area source | area proj. | area gate-fixed |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['family']}` | {_fmt(r['closure_source'], '{:.2e}')} "
            f"| {_fmt(r['closure_projected'], '{:.2e}')} "
            f"| {_fmt(r['closure_gate_fixed'], '{:.2e}')} "
            f"| {_fmt(r['area_source'])} | {_fmt(r['area_projected'])} "
            f"| {_fmt(r['area_gate_fixed'])} |"
        )

    lines += [
        "",
        "## 4. Cost of the projected ansatz",
        "",
        "What the optimizer starts from. `energy` is the objective; the analytic lower",
        f"bound for the X(pi) branch is `theta^2 = {TARGET_THETA**2:.4f}` and the",
        "converged optimum under gate + closure + zero area is 83.75 at M = 12",
        "(`_dev_logs/step08_optimize.md`).",
        "",
        "`bound` is `theta^2` evaluated at the *realized* gate angle of the projected",
        "ansatz, not at the target: a projection that misses the gate is allowed to sit",
        "below the target's bound without violating anything, and comparing it against",
        "the wrong bound is the easiest way to read a violation that is not there.",
        "",
        "| family | energy | bound at own angle | energy gate-fixed | int abs Omega dt | T Omega_max |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['family']}` | {_fmt(r['energy_projected'], '{:.4f}')} "
            f"| {_fmt(r['energy_bound_own_angle'], '{:.4f}')} "
            f"| {_fmt(r['energy_gate_fixed'], '{:.4f}')} "
            f"| {_fmt(r['curv_projected'], '{:.4f}')} "
            f"| {_fmt(r['peak_projected'], '{:.4f}')} |"
        )
    TABLE_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLE_PATH.relative_to(REPO)}")


def render_figures(rows) -> None:
    planar = [r for r in rows if r["planar"]]

    # --- figure 1: what the gate fix costs in closure and area ---------
    fig, (ax_c, ax_a) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    names = [r["family"] for r in planar]
    x = np.arange(len(names))
    floor = 1e-18
    for ax, key, title in (
        (ax_c, "closure", "closure  $|r(T)|/L$"),
        (ax_a, "area", "|area| $/L^2$"),
    ):
        for offset, stage, color, marker in (
            (-0.22, "source", "0.45", "o"),
            (0.0, "projected", "C0", "s"),
            (0.22, "gate_fixed", "C3", "^"),
        ):
            vals = [
                max(abs(r[f"{key}_{stage}"]), floor) if r[f"{key}_{stage}"] is not None else np.nan
                for r in planar
            ]
            ax.scatter(x + offset, vals, c=color, marker=marker, s=42, label=stage, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7.5)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Planar families: robustness residuals as they arrive, after least-squares "
        "projection, and after an exact gate fix",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(HERE / "fig1_gate_fix_cost.png", dpi=150, bbox_inches="tight")

    # --- figure 2: projection error vs endpoint conflict ---------------
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    styles = {"exact": ("C2", "o"), "approx": ("C0", "s"), "forced": ("C3", "^")}
    for tier, (color, marker) in styles.items():
        sel = [r for r in rows if r["tier_declared"] == tier]
        if not sel:
            continue
        ax.scatter(
            [max(r["endpoint_ratio"], 1e-16) for r in sel],
            [max(r["rel_rms"], 1e-16) for r in sel],
            c=color,
            marker=marker,
            s=55,
            label=f"declared {tier}",
            zorder=3,
        )
    for r in rows:
        ax.annotate(
            r["family"],
            (max(r["endpoint_ratio"], 1e-16), max(r["rel_rms"], 1e-16)),
            fontsize=6,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax.axvline(0.10, color="0.6", ls="--", lw=1.0)
    ax.text(0.11, 2e-16, "forced threshold", fontsize=7, color="0.4", rotation=90)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"endpoint conflict  $|\Omega(0)|/\max|\Omega|$")
    ax.set_ylabel(r"projection error  relRMS of $\Omega$")
    ax.set_title("Endpoint conflict does not predict projection error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(HERE / "fig2_tier_vs_relrms.png", dpi=150, bbox_inches="tight")

    # --- figure 3: starting cost of each projected ansatz --------------
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    sel = sorted(
        [r for r in rows if r["on_pi_branch"] and r["energy_projected"] is not None],
        key=lambda r: r["energy_projected"],
    )
    ax.barh(
        [r["family"] for r in sel],
        [r["energy_projected"] for r in sel],
        color=["C2" if r["tier_declared"] == "exact" else "C0" for r in sel],
    )
    ax.axvline(TARGET_THETA**2, color="k", ls=":", lw=1.2)
    ax.text(TARGET_THETA**2, -0.6, r" bound $\theta^2$", fontsize=8)
    ax.axvline(83.7479, color="C3", ls="--", lw=1.2)
    ax.text(83.7479, -0.6, " optimum 83.75", fontsize=8, color="C3")
    ax.set_xscale("log")
    ax.set_xlabel(r"energy  $(\int\Omega^2dt)L$  of the projected ansatz")
    ax.set_title("Starting cost, X(pi) branch families only")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(HERE / "fig3_starting_energy.png", dpi=150, bbox_inches="tight")
    print("wrote fig1_gate_fix_cost.png, fig2_tier_vs_relrms.png, fig3_starting_energy.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="skip measurement; rebuild TABLE.md and the figures from audit.csv",
    )
    args = parser.parse_args()

    if not args.render_only:
        write_csv(measure())
    rows = read_csv()
    render_table(rows)
    render_figures(rows)


if __name__ == "__main__":
    main()
