#!/usr/bin/env python3
"""Regenerate the Novera single-qubit pi-gate delivery package at any sample rate.

    python make_delivery.py                                  # 2.4 GSa/s -> delivery/
    python make_delivery.py --sample-rate 1.0                # 1 GSa/s   -> delivery_1GSa/
    python make_delivery.py --sample-rate 1.0 --outdir foo    # explicit destination

Why this script exists
----------------------
The 2026-08-09 delivery package (``delivery/``) was written by an ad-hoc script
that was never committed, so when the experimentalist asked for 1 GSa/s instead
of 2.4 GSa/s there was nothing to re-run. This is that script, committed, with
the sample rate as its one knob.

How the resample is done -- and how it is NOT done
--------------------------------------------------
The exported CSV is **not** a resampling of the 120-point file. Every arm is a
set of sine-basis coefficients ``(a, c, Phi_0)`` defining a *continuous* design
curve; ``armcore.Input.broadcast(N)`` evaluates that continuous object (through
the DRAG/Stark readout map, for the layers that have one) on an ``N``-point
midpoint grid. Changing the sample rate therefore changes ``N`` and re-evaluates
the analytic curve -- no interpolation error is introduced, and the design curve
itself is untouched.

``N = round(sample_rate * gate_time)``: 2.4 GSa/s x 50 ns -> 120 samples,
1.0 GSa/s x 50 ns -> 50 samples. This is exactly ``GateLab.N_EXPORT``
(``gatelib.py`` ~line 191), which already reads the device's own
``sample_rate``, so overriding that one field is the whole mechanism. The device
default in ``curve_opt.device`` stays 2.4 -- we pass a
``dataclasses.replace``'d copy rather than editing the config layer, so the F08
golden regression is untouched.

Midpoint sampling, not cell-averaging
-------------------------------------
Checked both on 2026-08-10 at N = 50: cell-averaging (the exact zeroth Magnus
term of each hold cell) helps arms C/D/F's nominal infidelity by ~20% but hurts
arm E by 6x and hurts arm F's worst-case-over-the-noise-grid, i.e. it is not a
uniform win. Midpoint sampling is kept -- it is what the 2.4 GSa/s package
shipped, what the CSV header documents (``t_ns`` is the CELL MIDPOINT), and what
``geometry.midpoint_grid`` gives every other consumer in this project.

phi_vz is re-fitted on the played waveform
------------------------------------------
Deviation from the 2026-08-09 package, deliberate. That one froze ``phi_vz``
from the fine (N = 4000) design curve and then reported the coarse waveform's
infidelity against that frozen frame. Here ``phi_vz`` is fitted on the exact
samples the file contains. Reasons:

* the header instructs the reader to apply ``phi_vz`` to *this* file, so the
  number must be that file's own frame -- otherwise we ship a known frame error;
* ``verify_waveform.py`` cross-checks the header's ``phi_vz`` against its own
  best fit from the CSV. Re-fitting makes that check agree to the grid step;
  freezing makes it disagree by ~0.1 deg at 1 GSa/s for no reason.

This is not the "free recalibration" the project red line forbids: that rule is
about re-fitting ``phi_vz`` *per noise point* inside a robustness sweep. Here it
is fitted once, at zero noise, and then frozen for the whole grid -- exactly
what ``GateLab.characterize`` does and exactly what a calibration run does.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "proposals" / "_gatelib"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax.numpy as jnp  # noqa: E402

import armcore  # noqa: E402
from armspec import SLUGS  # noqa: E402
from curve_opt import gate, recorder  # noqa: E402
from gatelib import GateLab, delta_rate  # noqa: E402
from gatespec import SPECS, rotate_drive  # noqa: E402

TWO_PI = 2.0 * np.pi
DEVICE_CONFIG = REPO_ROOT / "configs" / "device_novera.json"

#: Delivered gates: package sub-directory -> the ``SPECS`` key that solves it.
GATES = {"Xpi": "Xpi_novera", "Ypi": "Ypi_novera"}

#: Human-facing arm labels. Presentation text only -- the physics of each arm
#: lives in ``armspec.ARM_RECIPES``; these two strings are what the CSV header
#: and the README call it.
ARM_LABELS = {
    "A": ("Gaussian", "Gaussian (planar baseline)"),
    "B": ("Gaussian + DRAG", "Gaussian + DRAG (standard baseline)"),
    "C": ("Optimized (3D)", "Optimized 3D, unguarded"),
    "D": ("Opt + fluence cap", "Optimized 3D + fluence cap"),
    "E": ("Opt AD-exact", "Optimized 3D, exact-gate arm"),
    "F": ("Opt AD-robust", "Optimized 3D, robust arm"),
}

#: The robustness grid quoted in every header. Deliberately the *delivery* grid
#: (+-5% amplitude, +-0.2 MHz detuning), not ``gatelib``'s wider EPS/DELTA grids.
EPS_GRID = np.array([-0.05, -0.02, 0.0, 0.02, 0.05])
DELTA_MHZ_GRID = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])

#: CSV columns, in order. ``Omega_*`` are the physical angular drive; everything
#: to the left of them is a convenience rescaling of the same two numbers.
COLUMNS = ("t_ns", "I_norm", "Q_norm", "rabi_x_MHz", "rabi_y_MHz",
           "rabi_envelope_MHz", "phase_rad", "Omega_x_rad_per_ns",
           "Omega_y_rad_per_ns")

TARGET_NAMES = {"Xpi": "X(pi)", "Ypi": "Y(pi)"}


# ---------------------------------------------------------------------------
# physics: one arm, one sample rate
# ---------------------------------------------------------------------------


def evaluate(lab: GateLab, om_x, om_y, *, phi_vz: float | None = None) -> dict:
    """Everything the header quotes, measured on *these exact samples*.

    ``phi_vz=None`` fits it (the canonical file); passing a value evaluates
    against that frozen frame instead (used to score the vz-free variant, whose
    correct answer is ``phi_vz = 0``).
    """
    om_x, om_y = np.asarray(om_x, dtype=float), np.asarray(om_y, dtype=float)
    U3 = np.asarray(gate.three_level_propagator(
        jnp.asarray(om_x), jnp.asarray(om_y), lab.T, lab.DEV.delta))
    B = np.asarray(gate.qubit_block(U3))
    if phi_vz is None:
        phi_vz, fid = armcore.fit_phi_vz(B, lab.U_TARGET)
    else:
        fid = float(armcore.fidelity_jax(jnp.asarray(B),
                                         jnp.asarray(armcore.rz_matrix(phi_vz) @ lab.U_TARGET)))
    fid_no_vz = float(armcore.fidelity_jax(jnp.asarray(B), jnp.asarray(lab.U_TARGET)))

    # Robustness at a frame frozen by the zero-noise fit above -- re-fitting per
    # noise point would hand the arm a free recalibration (project red line).
    fn = lab._make_mc_fn(om_x, om_y, phi_vz)
    DZ, EPS = np.meshgrid(delta_rate(DELTA_MHZ_GRID), EPS_GRID, indexing="ij")
    grid = np.asarray(fn(jnp.asarray(DZ.ravel()), jnp.asarray(EPS.ravel()))).reshape(DZ.shape)

    return {
        "phi_vz": float(phi_vz),
        "infidelity": 1.0 - fid,
        "infidelity_without_vz": 1.0 - fid_no_vz,
        # tr(P^2) == tr(B^dag B) for the polar factor, so this is the header's
        # "1 - tr(B^dag B)/2" written the way characterize_zero_noise writes it.
        "leakage": float(1.0 - 0.5 * np.real(np.trace(B.conj().T @ B))),
        "peak_rabi_MHz": float(np.max(np.hypot(om_x, om_y))) / TWO_PI * 1e3,
        # grid[detuning, eps] -> transpose to [eps, detuning], the header's layout
        "grid": grid.T,
    }


# ---------------------------------------------------------------------------
# on-disk format
# ---------------------------------------------------------------------------


def grid_block(grid: np.ndarray, indent: str = "#   ") -> list[str]:
    """The ``1 - Fbar over (amplitude error, drive detuning)`` table, header style."""
    lines = [f"{indent}1 - Fbar over (amplitude error, drive detuning):",
             f"{indent}      detuning ->  "
             + "  ".join(f"{d:+.2f}MHz" for d in DELTA_MHZ_GRID)]
    for i, eps in enumerate(EPS_GRID):
        lines.append(f"{indent}  eps={eps:+.2f}     "
                     + "  ".join(f"{v:.2e}" for v in grid[i]))
    return lines


def header(gate_dir: str, arm_key: str, res: dict, meta: dict, *, vzfree: bool) -> str:
    """The ``#``-commented preamble. Self-describing by contract: a reader with
    only the CSV must be able to play it and to check it."""
    target = TARGET_NAMES[gate_dir]
    _, description = ARM_LABELS[arm_key]
    dev, N, dt = meta["device"], meta["n_samples"], meta["dt_ns"]
    rule = "# " + "=" * 74
    L = [rule,
         f"# {target} gate for the Novera transmon  --  arm {arm_key}: {description}",
         rule, "# "]

    if vzfree:
        psi_deg = np.degrees(meta["psi"])
        L += [
            "# *** THIS IS THE VZ-FREE VARIANT: play it as-is, NO virtual Z needed. ***",
            "# ",
            f"#   The whole drive phase has been rotated by psi = -phi_vz/2 = {psi_deg:+.6f} deg.",
            "#   This is an EXACT identity for a pi rotation about an in-plane axis:",
            f"#       rotating the drive phase by psi maps  W -> R_z(phi_vz + 2*psi) . {target},",
            f"#   so psi = -phi_vz/2 leaves exactly {target}. Robustness is unchanged (R_z commutes",
            f"#   with the noise terms). Verified: 1 - Fbar = {res['infidelity_without_vz']:.4e} "
            "with no virtual Z,",
            f"#   matching the canonical file's {meta['canonical_infidelity']:.4e}.",
        ]
    else:
        phi = res["phi_vz"]
        L += [
            "# *** THIS PULSE ALONE IS NOT THE TARGET GATE. READ THE NEXT 6 LINES. ***",
            "# ",
            f"#   The pulse realises   W = R_z(phi_vz) . {target}   with",
            f"#       phi_vz = {phi:.12f} rad = {np.degrees(phi):.6f} deg",
            "#   You MUST apply phi_vz as a virtual-Z (frame update) AFTER the pulse.",
            f"#   Playing this file without it gives 1 - Fbar = {res['infidelity_without_vz']:.4e}"
            f"  (instead of {res['infidelity']:.4e}).",
            "#   Alternative: use the companion file *_vzfree.csv, which has the whole drive",
            f"#   phase pre-rotated by psi = -phi_vz/2 = {np.degrees(meta['psi']):+.6f} deg "
            "and needs NO virtual Z.",
        ]

    L += ["# ",
          "# -- conventions " + "-" * 58,
          f"#   target = {target} = exp(-i*pi*(n.sigma)/2), n at azimuth "
          f"phi_axis = {np.degrees(meta['phi_axis'])} deg in the equatorial plane",
          f"#   provenance: {meta['provenance']}",
          "#   H = Omega_x*(a+a^dag)/2 + Omega_y*i(a^dag-a)/2 + Delta*|2><2|,  "
          f"Delta/2pi = {dev['anharmonicity']} GHz",
          "#   Omega is ANGULAR (rad/ns). rabi_*_MHz = Omega/(2*pi) in MHz. "
          "I_norm = Omega_x/Omega_max.",
          f"#   Omega_max/2pi = {dev['rabi_max'] * 1e3} MHz "
          "(full-scale reference for I_norm/Q_norm).",
          "#   phase_rad = atan2(Omega_y, Omega_x). If your IQ convention has the opposite phase",
          "#   sense, negate Q -- see README, it changes the answer by 4 orders of magnitude.",
          "#   Drive is ON RESONANCE with the bare f01. No frequency offset is required",
          "#   (the AC-Stark shift is already built into the pulse shape).",
          "# ",
          "# -- playback " + "-" * 61,
          f"#   {N} samples, sample-and-hold, dt = {dt:.9f} ns  ->  "
          f"{meta['sample_rate_GSa']:.4f} GSa/s",
          f"#   total duration T = {meta['T_ns']:.3f} ns.  t_ns is the CELL MIDPOINT (t0 = dt/2).",
          f"#   peak |Omega|/2pi = {res['peak_rabi_MHz']:.4f} MHz  "
          f"({100 * res['peak_rabi_MHz'] / (dev['rabi_max'] * 1e3):.2f} % of Omega_max)",
          "# ",
          "# -- simulated performance on this device " + "-" * 34,
          f"#   three-level coherent 1 - Fbar   = "
          f"{(res['infidelity_without_vz'] if vzfree else res['infidelity']):.4e}",
          f"#   leakage 1 - tr(B^dag B)/2       = {res['leakage']:.4e}",
          f"#   T1 = {dev['t1'] / 1e3} us, T2echo = {dev['t2_echo'] / 1e3} us -> "
          f"decoherence floor {meta['decoherence_floor']:.3e} for a {meta['T_ns']:.0f} ns gate",
          "# "]
    L += grid_block(res["grid"])
    L += ["# ",
          "# -- provenance " + "-" * 59,
          f"#   run_id = {meta['run_id']}   status = {meta['status']}",
          f"#   git_commit = {meta['git_commit']}",
          f"#   device = Novera: alpha/2pi = {dev['anharmonicity']} GHz, "
          f"Omega_max/2pi = {dev['rabi_max'] * 1e3} MHz,",
          f"#            T1 = {dev['t1']} ns, T2echo = {dev['t2_echo']} ns, "
          f"sample_rate = {meta['sample_rate_GSa']} GSa/s",
          rule]
    return "\n".join(L) + "\n"


def write_csv(path: Path, t_ns, om_x, om_y, rabi_max_rate: float, head: str) -> None:
    om_x, om_y = np.asarray(om_x), np.asarray(om_y)
    env = np.hypot(om_x, om_y)
    table = np.column_stack([
        t_ns, om_x / rabi_max_rate, om_y / rabi_max_rate,
        om_x / TWO_PI * 1e3, om_y / TWO_PI * 1e3, env / TWO_PI * 1e3,
        np.arctan2(om_y, om_x), om_x, om_y,
    ])
    with path.open("w") as fh:
        fh.write(head)
        fh.write(",".join(COLUMNS) + "\n")
        for row in table:
            fh.write(",".join(f"{v:.12e}" for v in row) + "\n")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def build(sample_rate: float, outdir: Path) -> dict:
    from curve_opt import geometry

    dev24 = recorder.load_device(DEVICE_CONFIG)
    dev = dataclasses.replace(dev24, sample_rate=float(sample_rate))
    commit = git_commit()

    package = {
        "package": "Novera single-qubit pi-gate waveform delivery",
        "device": {
            "name": "Novera",
            "anharmonicity_GHz": dev.anharmonicity,
            "rabi_max_GHz": dev.rabi_max,
            "t1_ns": dev.t1,
            "t2_echo_ns": dev.t2_echo,
            "sample_rate_GSa": dev.sample_rate,
            "gate_time_ns": dev.gate_time,
        },
        "conventions": {
            "hamiltonian": "H = Omega_x*(a+a^dag)/2 + Omega_y*1j*(a^dag-a)/2 + Delta*|2><2|",
            "omega_units": "rad/ns (angular); rabi_*_MHz = Omega/(2*pi)*1e3",
            "i_norm": f"Omega_x / Omega_max, Omega_max/2pi = {dev.rabi_max * 1e3} MHz",
            "time_grid": "cell midpoints, sample-and-hold, t0 = dt/2, T = N*dt",
            "phi_vz": ("the played pulse realises W = R_z(phi_vz).U_target; apply phi_vz "
                       "as a virtual Z after the pulse, or use the *_vzfree.csv variant"),
            "drive_frequency": "on resonance with the bare f01; no offset needed",
            "phi_vz_fit": ("fitted on the exported samples themselves, not inherited from a "
                           "finer grid -- see make_delivery.py's module docstring"),
        },
        "gates": {},
    }

    for gate_dir, spec_key in GATES.items():
        spec = SPECS[spec_key]
        lab = GateLab(spec, device=dev)
        arms = lab.build_arms()
        N = lab.N_EXPORT
        t_ns, _ = geometry.midpoint_grid(lab.T, N)
        t_ns = np.asarray(t_ns)
        dt = lab.T / N
        wf_dir = outdir / gate_dir / "waveforms"
        wf_dir.mkdir(parents=True, exist_ok=True)

        provenance = ("exact frame rotation of X(pi) by phi_axis = pi/2 "
                      "(not an independent solve)" if spec.source_key
                      else "solved independently on this device")
        gate_entry = {
            "gate": TARGET_NAMES[gate_dir],
            "phi_axis_rad": lab.phi_axis,
            "provenance": provenance,
            "git_commit": commit,
            "arms": {},
        }

        for key in lab.ARM_ORDER:
            arm = arms[key]
            slug = SLUGS[key]
            om_x, om_y = lab.broadcast_of(arm, N)
            res = evaluate(lab, om_x, om_y)
            psi = -0.5 * res["phi_vz"]
            vx, vy = rotate_drive(om_x, om_y, psi)
            # The vz-free variant's claim is "no virtual Z", so it is scored at
            # phi_vz = 0 exactly, never re-fitted.
            res_vz = evaluate(lab, vx, vy, phi_vz=0.0)

            run_id = getattr(arm, "run_id", None)
            status = None
            if run_id is not None:
                final = recorder.RUNS_DIR / run_id / "final.json"
                if final.exists():
                    status = json.loads(final.read_text()).get("status")

            meta = {
                "device": {"anharmonicity": dev.anharmonicity, "rabi_max": dev.rabi_max,
                           "t1": dev.t1, "t2_echo": dev.t2_echo},
                "n_samples": N, "dt_ns": dt, "sample_rate_GSa": dev.sample_rate,
                "T_ns": lab.T, "phi_axis": lab.phi_axis, "provenance": provenance,
                "decoherence_floor": dev.decoherence_floor,
                "run_id": run_id, "status": status, "git_commit": commit,
                "psi": psi, "canonical_infidelity": res["infidelity"],
            }

            name = f"{gate_dir}_novera_{slug}"
            csv_path, vz_path = wf_dir / f"{name}.csv", wf_dir / f"{name}_vzfree.csv"
            write_csv(csv_path, t_ns, om_x, om_y, dev.rabi_max_rate,
                      header(gate_dir, key, res, meta, vzfree=False))
            write_csv(vz_path, t_ns, vx, vy, dev.rabi_max_rate,
                      header(gate_dir, key, res_vz, meta, vzfree=True))

            label, description = ARM_LABELS[key]
            gate_entry["arms"][key] = {
                "name": label, "description": description, "slug": slug,
                "run_id": run_id, "status": status,
                "csv": f"waveforms/{name}.csv",
                "csv_vzfree": f"waveforms/{name}_vzfree.csv",
                "phi_vz_rad": res["phi_vz"], "phi_vz_deg": np.degrees(res["phi_vz"]),
                "drive_phase_offset_psi_rad": psi,
                "drive_phase_offset_psi_deg": np.degrees(psi),
                "infidelity_with_vz": res["infidelity"],
                "infidelity_without_vz": res["infidelity_without_vz"],
                "infidelity_vzfree_variant": res_vz["infidelity_without_vz"],
                "leakage": res["leakage"],
                "peak_rabi_MHz": res["peak_rabi_MHz"],
                "T_ns": lab.T, "n_samples": N, "dt_ns": dt,
                "sample_rate_GSa": dev.sample_rate,
                "omega_x_rad_per_ns": np.asarray(om_x).tolist(),
                "omega_y_rad_per_ns": np.asarray(om_y).tolist(),
                "omega_x_vzfree_rad_per_ns": np.asarray(vx).tolist(),
                "omega_y_vzfree_rad_per_ns": np.asarray(vy).tolist(),
                "robustness_grid": {
                    "eps": EPS_GRID.tolist(),
                    "detuning_MHz": DELTA_MHZ_GRID.tolist(),
                    "infidelity": res["grid"].tolist(),
                },
            }
            print(f"  {gate_dir} arm {key}: {N} samples, 1-F = {res['infidelity']:.4e}, "
                  f"grid max = {res['grid'].max():.4e}, peak = {res['peak_rabi_MHz']:.3f} MHz, "
                  f"phi_vz = {np.degrees(res['phi_vz']):+.4f} deg")
        package["gates"][gate_dir] = gate_entry

    (outdir / "delivery_metadata.json").write_text(
        json.dumps(package, indent=1, default=float))
    src_verify = HERE / "delivery" / "verify_waveform.py"
    if src_verify.exists():
        shutil.copy2(src_verify, outdir / "verify_waveform.py")
    return package


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample-rate", type=float, default=2.4,
                   help="AWG sample rate in GSa/s (default 2.4, as delivered 2026-08-09)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="destination (default delivery/ at 2.4, delivery_<rate>GSa/ otherwise)")
    args = p.parse_args(argv)

    outdir = args.outdir
    if outdir is None:
        outdir = HERE / ("delivery" if args.sample_rate == 2.4
                         else f"delivery_{args.sample_rate:g}GSa")
    outdir = outdir if outdir.is_absolute() else HERE / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"sample_rate = {args.sample_rate} GSa/s  ->  {outdir}")
    build(args.sample_rate, outdir)
    print(f"\nwrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
