"""Golden regression: prove this tree reproduces the recorded baseline bit for bit.

    python golden/check_golden.py            # all four gates, rtol=0 atol=0
    python golden/check_golden.py Xpi Xpi2   # a subset
    python golden/check_golden.py --rebase   # overwrite the baseline (intentional
                                             # numerics change ONLY -- say why in
                                             # the commit that carries it)

The baseline npz files under ``golden/baseline/`` were captured on the exploration
tree (branch ``dev/full_cost_optimizer``, F09) with ``_dev_logs/F08_golden.py``'s
worker, one file per gate, ~300k numbers total: every ``characterize()`` scalar,
the full 7x7 noise sweeps, design and export waveforms, coefficients, checkpoint
traces and the rotated gates' covariance checks.  Those numbers were themselves
cross-checked against QuTiP while that layer existed, so **bit-for-bit agreement
with this file is what carries the independent arbitration** into a tree that no
longer ships a second integrator.

The comparison is ``np.array_equal`` -- rtol=0, atol=0.  A pure refactor must
reproduce every number exactly; a 1e-16 drift means an expression was re-associated
(the classic: ``T/6`` becoming ``T*(1/6)``), not that the tolerance is too tight.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

BASELINE = Path(__file__).resolve().parent / "baseline"

GATE_KEYS = ("Xpi", "Xpi2", "Ypi", "Ypi2")


# ---------------------------------------------------------------------------
# capture -- the same quantities, key for key, as the F08/F09 golden worker
# ---------------------------------------------------------------------------


def capture(gate_key: str) -> dict[str, np.ndarray]:
    from curve_opt.gates.gatelib import GateLab, bind
    from curve_opt.gates.gatespec import SPECS

    ns: dict = {}
    bind(GateLab(SPECS[gate_key]), ns)
    al = SimpleNamespace(**ns)

    blob: dict[str, np.ndarray] = {}

    def put(key, val):
        blob[key] = np.asarray(val)

    # -- gate-level constants ----------------------------------------------
    put("_ARM_ORDER", np.array(list(al.ARM_ORDER), dtype="U8"))
    put("_N_DESIGN", al.N_DESIGN)
    put("_N_SIM", al.N_SIM)
    put("_N_EXPORT", al.N_EXPORT)
    put("_FLUENCE_FLOOR", al.FLUENCE_FLOOR)
    put("_THETA", al.THETA)
    put("_EPS_GRID", al.EPS_GRID)
    put("_DELTA_MHZ_GRID", al.DELTA_MHZ_GRID)
    put("_U_TARGET_re", np.real(al.U_TARGET))
    put("_U_TARGET_im", np.imag(al.U_TARGET))
    dev = al.DEV
    for f in ("anharmonicity", "rabi_max", "t1", "t2_echo", "static_detuning",
              "control_error", "sample_rate", "gate_time", "n_modes"):
        put(f"_dev_{f}", getattr(dev, f))
    for f in ("anharmonicity_rate", "rabi_max_rate", "static_detuning_rate", "delta",
              "eta", "peak_budget", "fenchel_min_time", "gamma1", "gamma_phi",
              "decoherence_floor", "resonant_harmonic"):
        put(f"_devd_{f}", getattr(dev, f))

    # -- per-arm -----------------------------------------------------------
    arms = al.build_arms()
    scalar_names = None
    for k in al.ARM_ORDER:
        arm = arms[k]
        row = al.characterize(arm)
        if scalar_names is None:
            scalar_names = sorted(row)
        for name in sorted(row):
            put(f"{k}/row/{name}", row[name])
        put(f"{k}/coeffs_a", np.asarray(arm.a, dtype=float))
        put(f"{k}/coeffs_c", np.asarray(arm.c, dtype=float))
        put(f"{k}/Phi_0", float(arm.Phi_0))
        ox, oy = al.broadcast_of(arm, al.N_EXPORT)
        put(f"{k}/bcast_export_x", ox); put(f"{k}/bcast_export_y", oy)
        ox, oy = al.broadcast_of(arm, al.N_DESIGN)
        put(f"{k}/bcast_design_x", ox); put(f"{k}/bcast_design_y", oy)
        dc = al.design_of(arm, al.N_DESIGN)
        put(f"{k}/kappa", np.asarray(dc.kappa)); put(f"{k}/tau", np.asarray(dc.tau))
        put(f"{k}/sweep", al.sweep(arm, row["phi_vz"]))
        znc = al.zero_noise_consistency(arm, row["phi_vz"], row["baseline_infidelity"])
        for name in sorted(znc):
            put(f"{k}/znc/{name}", znc[name])
        put(f"{k}/c3_pred", al.c3_prediction(arm))
        print(f"  {gate_key:>5s} arm {k}: ok", flush=True)
    put("_scalar_names", np.array(scalar_names, dtype="U48"))

    # -- checkpoint traces -------------------------------------------------
    targets = [(k, lambda k=k: al.checkpoint_trace(k))
               for k in ("C", "D", "E", "F") if al.LAB.run_dir(k) is not None]
    if al.LAB.trace_run_dir("E") is not None:
        targets.append(("E_trace", lambda: al.checkpoint_trace("E", trace=True)))
    for key, fn in targets:
        tr = fn()
        for name in sorted(tr):
            put(f"trace/{key}/{name}", tr[name])
        print(f"  {gate_key:>5s} trace {key}: ok ({len(tr['iter'])} ckpt)", flush=True)

    # -- covariance check (rotated gates only) -----------------------------
    if al.LAB.spec.is_rotated:
        for k in al.ARM_ORDER:
            cc = al.covariance_check(k)
            for name in sorted(cc):
                v = cc[name]
                if isinstance(v, dict):
                    for sub in sorted(v):
                        put(f"cov/{k}/{name}/{sub}", v[sub])
                elif isinstance(v, str):
                    put(f"cov/{k}/{name}", np.array(v, dtype="U40"))
                else:
                    put(f"cov/{k}/{name}", v)
            print(f"  {gate_key:>5s} cov {k}: ok", flush=True)

    return blob


def baseline_path(gate_key: str) -> Path:
    from curve_opt.gates.gatespec import SPECS

    return BASELINE / f"{SPECS[gate_key].folder}.npz"


# ---------------------------------------------------------------------------
# compare -- rtol=0, atol=0
# ---------------------------------------------------------------------------


def compare(gate_key: str, blob: dict[str, np.ndarray]) -> list[str]:
    path = baseline_path(gate_key)
    if not path.exists():
        return [f"{gate_key}: baseline missing: {path}"]
    ref = np.load(path, allow_pickle=False)
    problems: list[str] = []
    kb, ka = set(ref.files), set(blob)
    for k in sorted(kb - ka):
        problems.append(f"{gate_key}: key vanished: {k}")
    for k in sorted(ka - kb):
        problems.append(f"{gate_key}: key appeared: {k}")
    for k in sorted(kb & ka):
        vb, va = ref[k], blob[k]
        if vb.shape != va.shape:
            problems.append(f"{gate_key}:{k}: shape {vb.shape} -> {va.shape}")
            continue
        if vb.dtype.kind in "US":
            if not np.array_equal(vb, va):
                problems.append(f"{gate_key}:{k}: {vb.tolist()!r} -> {va.tolist()!r}")
            continue
        if not np.array_equal(vb, va):
            d = np.abs(np.asarray(va, float) - np.asarray(vb, float))
            den = np.maximum(np.abs(np.asarray(vb, float)), 1e-300)
            problems.append(f"{gate_key}:{k}: max|d| = {np.max(d):.3e}, "
                            f"max rel = {np.max(d / den):.3e}")
    return problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gates", nargs="*", default=list(GATE_KEYS),
                   help="subset of gates to check (default: all four)")
    p.add_argument("--rebase", action="store_true",
                   help="overwrite golden/baseline/ with this tree's numbers")
    args = p.parse_args()

    problems: list[str] = []
    n_keys = n_elem = 0
    for gate_key in args.gates:
        print(f"[golden] {gate_key}", flush=True)
        blob = capture(gate_key)
        n_keys += len(blob)
        n_elem += sum(int(v.size) for v in blob.values())
        if args.rebase:
            BASELINE.mkdir(parents=True, exist_ok=True)
            np.savez(baseline_path(gate_key), **blob)
            print(f"  -> rebased {baseline_path(gate_key).name} ({len(blob)} keys)")
        else:
            problems += compare(gate_key, blob)

    print(f"\nchecked {n_keys} keys ({n_elem} numbers) across {len(args.gates)} gate(s)")
    if args.rebase:
        print("BASELINE REWRITTEN -- commit it together with the change that justified it.")
        return
    if problems:
        print(f"\n!! {len(problems)} MISMATCH(ES) -- rtol=0, atol=0:")
        for pr in problems:
            print(f"   {pr}")
        raise SystemExit(1)
    print("ALL BIT-FOR-BIT IDENTICAL")


if __name__ == "__main__":
    main()
