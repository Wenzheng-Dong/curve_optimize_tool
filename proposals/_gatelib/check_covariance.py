"""Verify that a rotated gate really is its X-axis source gate's exact image.

    python check_covariance.py Ypi
    python check_covariance.py Ypi2

For every available arm this compares, against the source gate,

    * the four budget costs C1..C4,
    * the zero-noise true ``1 - Fbar``,
    * the whole ``(delta_z, epsilon)`` sweep, point by point.

The frame-rotation argument in ``gatespec``'s module docstring predicts agreement
to round-off.  Anything above ``TOL`` is a bug in the rotation plumbing, not
physics -- and if it ever becomes physics (IQ imbalance, quadrature-dependent
crosstalk, a noise channel that does not commute with the number operator), this
script is what tells you the Y folders must switch to ``solve_mode="independent"``.

Runs in well under a minute; no optimizer time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from gatelib import GateLab  # noqa: E402
from gatespec import SPECS  # noqa: E402

RTOL, ATOL = 1e-8, 1e-13
"""Mixed criterion: a quantity passes if ``|diff| <= ATOL + RTOL * |reference|``.

The rotation itself is exact to machine precision -- measured on arm B of Y(pi),
``C1, C2, C3`` agree to 0.0 and ``C4`` to 1.1e-12 relative.  The floor on the
*infidelity* comparison comes from somewhere else: ``f05.fit_phi_vz`` is a bounded
scalar minimization sitting in a shallow minimum, so it reproduces ``phi_vz`` only to
~1e-7 absolute.  On Y(pi)'s arm B (``1 - Fbar = 7e-6``) that is 7e-10 relative; on
Y(pi/2)'s arm B (``1 - Fbar = 1e-7``) the same ~5e-15 *absolute* wobble reads as 5e-8
relative.  A pure relative tolerance would therefore have to be loosened until it
stopped testing anything, which is why ``ATOL`` carries the small-infidelity arms and
``RTOL`` the rest.  Both sit far below any difference real physics would make.
"""


def _passes(abs_diff: float, ref: float) -> bool:
    return abs_diff <= ATOL + RTOL * abs(ref)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gate", choices=[k for k, s in SPECS.items() if s.is_rotated])
    p.add_argument("--device", default=None,
                   help="path to a device config JSON for curve_opt.recorder.load_device "
                        "(N3). Defaults to None, i.e. the module-level DEV -- unchanged "
                        "behavior for every gate that does not need this. A rotated gate "
                        "solved on a non-default device (e.g. Ypi_novera) MUST pass its "
                        "device here, or this script silently checks it against the "
                        "wrong Hamiltonian.")
    args = p.parse_args()

    device_obj = None
    if args.device is not None:
        REPO_ROOT = HERE.parent.parent
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from curve_opt import recorder  # noqa: E402 (only needed on this path)
        device_obj = recorder.load_device(args.device)

    lab = GateLab(SPECS[args.gate], device=device_obj)
    print(f"=== {lab.spec.name}  vs  its source gate  "
          f"(pass: |diff| <= {ATOL:.0e} + {RTOL:.0e} |ref|) ===")
    failures = []
    for key in lab.ARM_ORDER:
        r = lab.covariance_check(key)
        # (label, |diff|, reference magnitude) -- the sweep is referenced to the
        # zero-noise baseline, the smallest value anywhere on its grid.
        checks = [("baseline", r["baseline_abs_diff"], r["baseline_there"]),
                  ("sweep", r["sweep_max_abs_diff"], r["baseline_there"])]
        checks += [(c, r["cost_abs_diffs"][c],
                    r["cost_abs_diffs"][c] / r["cost_rel_diffs"][c]
                    if r["cost_rel_diffs"][c] > 0 else 0.0)
                   for c in ("c1", "c2", "c3", "c4")]
        costs = "  ".join(f"{c}:{r['cost_rel_diffs'][c]:.1e}" for c in ("c1", "c2", "c3", "c4"))
        print(f"arm {key}: 1-Fbar {r['baseline_here']:.6e} vs {r['baseline_there']:.6e}"
              f"  (abs {r['baseline_abs_diff']:.1e}, rel {r['baseline_rel_diff']:.1e})")
        print(f"        sweep max abs {r['sweep_max_abs_diff']:.1e} "
              f"(rel {r['sweep_max_rel_diff']:.1e})   cost rel diffs  {costs}")
        print(f"        phi_vz refit spread {r['phi_vz_abs_diff']:.1e} rad "
              f"(not part of the verdict -- see covariance_check's docstring)")
        failures += [f"{key}.{n}" for n, d, ref in checks if not _passes(d, ref)]

    if failures:
        print(f"\nFAIL: {', '.join(failures)}")
        raise SystemExit(1)
    print("\nPASS - every arm is its source gate's exact frame rotation.")


if __name__ == "__main__":
    main()
