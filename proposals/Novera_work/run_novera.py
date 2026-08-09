"""Solve X(pi) arms C/D/E/F on the Novera device (N2).

    python run_novera.py                # C D E F, in that order
    python run_novera.py C D            # just these
    python run_novera.py --dry-run      # print the plan, solve nothing

Fast path (leader-decided, see the task brief and README.md Sec.4.1): every arm
warm-starts from the *same-letter* arm of the **default-device** X(pi) solve
(``_runs/`` behind ``gatespec.SPECS["Xpi"]``), not from the registered C -> D ->
E -> F chain on this device. alpha moved 0.76% and M is unchanged, so the
default-device optimum sits right next to the Novera optimum; the coefficients
``(a, c)`` are basis coefficients in the shared ``(T, M)`` sine basis, so
carrying them across devices is legitimate (nothing about the basis changed).

Stop budget (README Sec.4.3): ``maxiter`` is converted from a wall-clock budget
using the measured per-iteration cost on the default device (C 0.169 s, D 0.239
s, E 0.82 s, F 0.518 s) and a budget of 600/900/900/900 s -- this is an upper
bound the warm-started solve is expected to undercut, not a target.

If the run does not end CONVERGED with a clean c1 endpoint at its literal final
point, the last checkpoint (recorded every ``checkpoint_every`` iterations) that
clears both the c1 entry ticket (``max|res_c1| <= 1e-9``, the DRAG/Stark
readout's requirement) and a small gate-residual norm is used instead, and the
arm is reported ``status="budget_stop"`` -- never upgraded to "converged". No
feasible checkpoint => the arm is reported as a failure, not silently degraded.

Solving does *not* touch ``configs/runs.json``: that registry's contract is
"the run whose *last* history row IS the arm", which does not hold for a
rewound arm's original run (its last row is the literal final iterate, which
failed the c1/gate-residual filter, not the accepted checkpoint). A separate
pass, ``--register-accepted``, reads the summary JSON this script writes and,
for each accepted arm, writes a small *derived* run whose only checkpoint is
the accepted one (:func:`write_accepted_run`) and registers *that* under
``Xpi_novera``. See the N2 dev log Sec.7 for the discussion.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GATELIB = HERE.parent / "_gatelib"
sys.path.insert(0, str(GATELIB))

REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from curve_opt import recorder  # noqa: E402

import armcore  # noqa: E402
from armspec import ARM_NAMES, ARM_RECIPES, replace  # noqa: E402
from gatespec import SPECS  # noqa: E402
from gatelib import GateLab  # noqa: E402

CHAIN = ("C", "D", "E", "F")

#: Wall-clock budget per arm (README Sec.4.3), s.
WALL_BUDGET_S = {"C": 600.0, "D": 900.0, "E": 900.0, "F": 900.0}
#: Measured per-iteration cost on the default device (README Sec.4.3), s/iter.
SEC_PER_ITER = {"C": 0.169, "D": 0.239, "E": 0.82, "F": 0.518}
#: floor(budget / sec_per_iter), matching the task brief's own numbers.
BUDGET_MAXITER = {k: int(WALL_BUDGET_S[k] / SEC_PER_ITER[k]) for k in CHAIN}
#: The task brief rounds these to C 3500 / D 3700 / E 1100 / F 1700; use the
#: rounded values so the recorded run manifest matches the brief exactly.
BUDGET_MAXITER = {"C": 3500, "D": 3700, "E": 1100, "F": 1700}

#: c1 endpoint entry ticket (parametrization.check_c1 / solve_arm's own guard).
C1_TOL = 1e-9
#: Gate-residual norm below which a c1-clean checkpoint counts as "clean" too.
#: Not specified numerically by the task brief; chosen two decades below the
#: scale F06c/F06f runs show a checkpoint's gate residual actually reaching once
#: c1 clears (~1e-8), so it is a generous, not a hair-trigger, cut.
GATE_RES_TOL = 1e-6

DEVICE_CONFIG = REPO_ROOT / "configs" / "device_novera.json"


def arm_from_checkpoint(lab: GateLab, arm_key: str, run_id: str, hist, idx: int) -> armcore.Input:
    """Build an :class:`armcore.Input` from one ``history.npz`` row (not the last one)."""
    it = int(hist["iter"][idx])
    arm = armcore.Input(
        ARM_NAMES[arm_key], "general",
        hist["coeffs_a"][idx], hist["coeffs_c"][idx], float(hist["Phi_0"][idx]),
        f"solve_budget solution for {lab.spec.name}, run {run_id}, rewound to "
        f"checkpoint iter={it} (status=budget_stop -- the literal final point did "
        f"not clear the c1/gate-residual checkpoint filter).",
        device=lab.DEV,
    )
    arm.run_id = run_id
    arm.c1_residual = np.asarray(hist["res_c1_ends"][idx])
    arm.checkpoint_iter = it
    return arm


def rewind(lab: GateLab, arm_key: str, run_id: str) -> tuple[armcore.Input | None, dict]:
    """Scan ``history.npz`` backwards for the last c1-clean, gate-residual-clean row.

    Reuses :meth:`GateLab.checkpoint_trace` for the c1 filtering (its docstring:
    checkpoints violating the c1 entry ticket are dropped) by pointing this
    lab's *in-memory* ``spec.run_ids`` at the just-solved run -- nothing is
    written to ``configs/runs.json``.
    """
    run_dir = recorder.RUNS_DIR / run_id
    lab.spec = dataclasses.replace(lab.spec, run_ids={**lab.spec.run_ids, arm_key: run_id})
    trace = lab.checkpoint_trace(arm_key)
    info = {"n_checkpoints_c1_clean": int(len(trace["iter"]))}
    if len(trace["iter"]) == 0:
        return None, info
    hist = dict(np.load(run_dir / "history.npz"))
    order = np.argsort(trace["iter"])[::-1]  # most advanced c1-clean checkpoint first
    for j in order:
        gate_res_norm = float(trace["gate_residual_norm"][j])
        if gate_res_norm <= GATE_RES_TOL:
            it = int(trace["iter"][j])
            hidx = int(np.where(hist["iter"] == it)[0][0])
            info["gate_residual_norm"] = gate_res_norm
            return arm_from_checkpoint(lab, arm_key, run_id, hist, hidx), info
    info["gate_residual_norm"] = float(np.min(trace["gate_residual_norm"]))
    return None, info


def win_count(sweep_here: np.ndarray, sweep_b: np.ndarray) -> tuple[int, int]:
    wins = int(np.sum(sweep_here < sweep_b))
    return wins, sweep_here.size


def write_accepted_run(orig_run_id: str, it: int) -> str:
    """Write a derived run whose only checkpoint IS the accepted point (leader,
    2026-08-09: "把被接受的 checkpoint 落成一条它自己的 run, 然后登记那条").

    ``configs/runs.json``'s own ``_README`` states the contract:
    "run_ids: the run whose last checkpoint IS the arm." A budget-stopped,
    rewound arm's accepted point is *not* its parent run's last checkpoint (the
    parent's literal final iterate failed the c1 entry ticket and/or the
    gate-residual filter, or simply postdates the accepted one -- Sec.2.3 of
    this module's docstring / the N2 dev log). Registering the parent run
    directly would therefore make ``GateLab.load_optimized`` silently read the
    wrong point. This writes a one-checkpoint derived record instead -- the
    accepted checkpoint's coefficients and cost/residual columns, copied
    *verbatim* out of the parent's ``history.npz`` row (not recomputed), so
    :func:`curve_opt.recorder.verify_history` reproduces them exactly (they are
    a pure function of the coefficients, ``T``, ``N_grid`` and the device, none
    of which changed). The parent run is left untouched on disk as the
    provenance record (the full optimization trajectory the point was picked
    from).

    ``recorder.STOP_REASONS`` has no ``'budget_stop'`` entry (the vocabulary
    this project's dev logs use for this situation); the derived run is closed
    with the nearest valid label, ``'maxiter'`` -- which is also literally true
    of the parent run this checkpoint is drawn from -- so ``void_for_claims``
    still comes out ``True`` and no claim of formal convergence is implied.
    ``'budget_stop'`` and the checkpoint iteration are spelled out in
    ``manifest["ansatz"]["derived_note"]`` instead, in words.
    """
    orig_dir = recorder.RUNS_DIR / orig_run_id
    hist = dict(np.load(orig_dir / "history.npz"))
    hidx = int(np.where(hist["iter"] == it)[0][0])
    orig_manifest = json.loads((orig_dir / "manifest.json").read_text())

    manifest = dict(orig_manifest)
    manifest["ansatz"] = dict(orig_manifest["ansatz"])
    manifest["ansatz"]["derived_from_run"] = orig_run_id
    manifest["ansatz"]["derived_checkpoint_iter"] = it
    manifest["ansatz"]["derived_note"] = (
        f"This run's only checkpoint IS the accepted arm: coefficients and "
        f"cost/residual columns copied verbatim from {orig_run_id}'s "
        f"history.npz row iter={it} -- the last checkpoint that cleared both "
        f"the c1 entry ticket (max|res_c1| <= {C1_TOL:g}) and the gate-residual "
        f"filter (<= {GATE_RES_TOL:g}) after {orig_run_id} stopped at scipy "
        f"stop_reason='maxiter' without reaching a usable point at its literal "
        f"final iterate. Called status='budget_stop' in the N2 dev log's "
        f"vocabulary; recorded here under this schema's nearest valid "
        f"STOP_REASONS label, 'maxiter', which is also literally true of the "
        f"run this checkpoint is drawn from. See {orig_run_id} (left untouched) "
        f"for the full optimization trajectory this point was picked from."
    )
    for key in ("run_id", "timestamp", "git_commit", "git_branch", "schema_version",
               "stop_reason"):
        manifest.pop(key, None)

    derived_id = f"{orig_run_id}_accepted"
    rec = recorder.Recorder(derived_id, manifest, schema="budget", exist_ok=True)
    epi = hist["epigraph_s"][hidx]
    rec.append_budget(
        it, hist["coeffs_a"][hidx], hist["coeffs_c"][hidx],
        cost_total=float(hist["cost_total"][hidx]),
        cost_c1=float(hist["cost_c1"][hidx]),
        cost_c2=float(hist["cost_c2"][hidx]),
        cost_c3=float(hist["cost_c3"][hidx]),
        cost_c4=float(hist["cost_c4"][hidx]),
        Phi_0=float(hist["Phi_0"][hidx]),
        phi_vz=float(hist["phi_vz"][hidx]),
        res_gate=hist["res_gate"][hidx],
        res_c1_ends=hist["res_c1_ends"][hidx],
        epigraph_s=None if np.isnan(epi) else float(epi),
    )
    rec.close(
        status="maxiter", nit=it, wall_clock_s=0.0,
        gate_residual=np.asarray(hist["res_gate"][hidx]).tolist(),
        c1_residual=np.asarray(hist["res_c1_ends"][hidx]).tolist(),
        note="derived record -- see manifest.ansatz.derived_from_run/derived_note",
    )
    return derived_id


def register_accepted(summary_path: Path) -> dict:
    """Re-entrant pass: for every ``budget_stop``/``converged`` arm in a summary
    JSON, write its derived run (:func:`write_accepted_run`) and register it in
    ``configs/runs.json`` under ``"Xpi_novera"``. Does not re-solve anything --
    reads the existing ``_runs/<run_id>/history.npz`` the original solve wrote.
    """
    import gatespec

    results = json.loads(Path(summary_path).read_text())
    registered = {}
    for entry in results:
        if entry.get("status") not in ("budget_stop", "converged"):
            continue
        arm_key, run_id, it = entry["arm"], entry["run_id"], entry["selected_iter"]
        derived_id = write_accepted_run(run_id, it)
        gatespec.register_run("Xpi_novera", arm_key, derived_id)
        registered[arm_key] = derived_id
        print(f"arm {arm_key}: derived run {derived_id} "
              f"(from {run_id} checkpoint iter={it}) registered as "
              f"Xpi_novera/{arm_key}", flush=True)
    return registered


def patch_warm_start_provenance(run_id: str, arm_key: str, warm_arm: armcore.Input) -> None:
    """Correct ``manifest.json``'s ``ansatz`` block after the fact.

    ``GateLab.solve_arm`` always names the run and its ``ansatz.source`` from
    ``recipe.warm`` (the registry recipe's own arm letter, e.g. ``"B"`` for arm
    C), *regardless* of what ``warm_arm=`` was actually passed in. That is
    correct when the two coincide (the normal on-device chain) but wrong here
    -- every N2 arm warm-starts from the *default-device* arm of the **same**
    letter (task brief Sec.2), not from this device's own chain, and the task
    brief is explicit that this must not be misrepresented ("不许假装走了正规
    链"). ``recipe.warm`` itself is left untouched (the task brief says the
    recipe stays "原样" for C/D/E), so the fix is a metadata addendum, applied
    here right after the run lands rather than by hand afterwards.
    """
    path = recorder.RUNS_DIR / run_id / "manifest.json"
    m = json.loads(path.read_text())
    m["ansatz"]["warm_start_provenance"] = (
        f"N2 fast path: warm-started from the DEFAULT device's X(pi) arm "
        f"{arm_key!r} solution (run {getattr(warm_arm, 'run_id', None)!r}), not "
        f"from this device's own arm {arm_key!r} lineage. ansatz.name's "
        f"'_warm_<letter>' suffix is the registry recipe's warm-arm label and "
        f"does NOT describe what was actually passed as warm_arm= here -- this "
        f"field is the corrected record."
    )
    path.write_text(json.dumps(m, indent=1))


def solve_one(lab: GateLab, ref: dict, arm_key: str, *, maxiter: int | None,
             run_id: str | None, dry_run: bool) -> dict:
    # recipe stays None (-> the registry recipe, unmodified) for C/D/E, so the
    # run manifest's ansatz.recipe_overridden is accurate; only F needs the
    # e0_cap override below, so only F passes an explicit recipe object.
    recipe = None
    if arm_key == "F":
        # X(pi)'s literal e0_cap=7.2e-6 *meant* "arm B's zero-noise level for
        # this gate" -- alpha moved, so measure it on this device instead of
        # inheriting the default-device literal (task brief Sec.2, armspec.py
        # docstring).
        recipe = replace(ARM_RECIPES["F"], e0_cap="arm_B")
    warm_arm = ref[arm_key]
    maxiter = BUDGET_MAXITER[arm_key] if maxiter is None else maxiter

    t0 = time.perf_counter()
    arm, result = lab.solve_arm(arm_key, warm_arm=warm_arm, recipe=recipe,
                                maxiter=maxiter, run_id=run_id, verbose=1,
                                dry_run=dry_run)
    wall = time.perf_counter() - t0
    if dry_run:
        return {"arm": arm_key, "dry_run": True}
    if arm is None:
        return {"arm": arm_key, "status": "failed", "reason": "solve_arm returned None"}

    run_id = arm.run_id
    patch_warm_start_provenance(run_id, arm_key, warm_arm)
    max_c1 = float(np.max(np.abs(arm.c1_residual)))
    entry = {
        "arm": arm_key,
        "run_id": run_id,
        "warm_start": f"default-device Xpi arm {arm_key} "
                       f"(run {getattr(warm_arm, 'run_id', None)})",
        "wall_clock_s": wall,
        "scipy_stop_reason": result.stop_reason,
        "nit": int(result.nit),
        "maxiter_budget": maxiter,
        "final_max_res_c1": max_c1,
    }

    accepted = None
    if result.stop_reason == "converged" and max_c1 <= C1_TOL:
        gate_res_norm = float(np.linalg.norm(np.asarray(result.gate_residual)))
        if gate_res_norm <= GATE_RES_TOL:
            accepted = arm
            entry.update(status="converged", selected_iter=int(result.nit),
                         max_res_c1=max_c1, gate_residual_norm=gate_res_norm,
                         rewound=False)

    if accepted is None:
        accepted, rewind_info = rewind(lab, arm_key, run_id)
        entry["rewind_info"] = rewind_info
        if accepted is None:
            entry.update(status="failed", rewound=True,
                         reason="no checkpoint cleared both the c1 entry ticket "
                                 "and the gate-residual filter")
            print(f"!! arm {arm_key}: NO feasible checkpoint -- not registered.",
                  flush=True)
            return entry
        entry.update(status="budget_stop", rewound=True,
                     selected_iter=accepted.checkpoint_iter,
                     max_res_c1=float(np.max(np.abs(accepted.c1_residual))),
                     gate_residual_norm=rewind_info["gate_residual_norm"])

    # -- characterize the accepted point + arm B on this device --------------
    row = lab.characterize(accepted)
    arm_b = lab.build_gaussian_drag()
    row_b = lab.characterize(arm_b)
    entry["baseline_infidelity"] = row["baseline_infidelity"]
    entry["baseline_infidelity_armB"] = row_b["baseline_infidelity"]
    entry["leakage_population"] = row["leakage_population"]
    entry["phi_vz"] = row["phi_vz"]

    sweep_here = lab.sweep(accepted, row["phi_vz"])
    sweep_b = lab.sweep(arm_b, row_b["phi_vz"])
    wins, total = win_count(sweep_here, sweep_b)
    entry["grid_wins_vs_B"] = f"{wins}/{total}"

    print(f"\n>>> arm {arm_key}: status={entry['status']} "
          f"selected_iter={entry.get('selected_iter')} wall={wall:.1f}s "
          f"max|res_c1|={entry['max_res_c1']:.3e} "
          f"gate_res_norm={entry['gate_residual_norm']:.3e} "
          f"baseline_infid={row['baseline_infidelity']:.4e} "
          f"(armB={row_b['baseline_infidelity']:.4e}) "
          f"grid_wins={wins}/{total}\n", flush=True)
    return entry


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # nargs='*' + choices mis-validates an empty/default selection as a single
    # invalid "choice" (argparse quirk), so choices is checked by hand below.
    p.add_argument("arms", nargs="*", default=None)
    p.add_argument("--maxiter", type=int, default=None,
                   help="override every arm's budget-derived maxiter")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=str(HERE / "novera_solve_summary.json"))
    p.add_argument("--register-accepted", action="store_true",
                   help="do not solve anything; read --out's summary JSON, write each "
                        "accepted arm's derived one-checkpoint run, and register it in "
                        "configs/runs.json under Xpi_novera")
    args = p.parse_args(argv)
    arms = args.arms or list(CHAIN)
    bad = [a for a in arms if a not in CHAIN]
    if bad:
        p.error(f"invalid arm(s) {bad}; choose from {list(CHAIN)}")

    if args.register_accepted:
        register_accepted(Path(args.out))
        return

    device = recorder.load_device(DEVICE_CONFIG)
    print(f"device: anharmonicity={device.anharmonicity}, t1={device.t1}, "
          f"t2_echo={device.t2_echo}, n_modes={device.n_modes}", flush=True)

    lab_ref = GateLab(SPECS["Xpi"])
    ref = lab_ref.build_arms()
    print(f"default-device Xpi arms available: {sorted(ref)}", flush=True)

    lab = GateLab(SPECS["Xpi_novera"], device=device)

    results = []
    for arm_key in arms:
        entry = solve_one(lab, ref, arm_key, maxiter=args.maxiter, run_id=None,
                          dry_run=args.dry_run)
        results.append(entry)

    if not args.dry_run:
        Path(args.out).write_text(json.dumps(results, indent=2, default=float))
        print(f"\nsummary written to {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
