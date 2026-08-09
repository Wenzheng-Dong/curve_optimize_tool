"""X(pi/2) construction runs -- a CLI over ``GateLab.solve_arm``.

    python run_construction.py C     # plain budget objective, no fluence cap
    python run_construction.py D     # + hard fluence cap at 2 F0   (F06b recipe)
    python run_construction.py E     # AD-exact objective, cap 4 F0 (F06e recipe)
    python run_construction.py F     # AD-robust objective + e0 fuse (F06f recipe)
    python run_construction.py C --dry-run     # build the problem, print the manifest

The recipes themselves live in ``_gatelib/armspec.py::ARM_RECIPES``, and the notebook's
in-place solve cell calls the same ``solve_arm``, so there is exactly one definition of
what each arm *is*.  Use this script for the long solves (arm D runs to maxiter 20000)
where a notebook kernel holding the run is a liability; use the notebook cell when you
want to solve and look at the result in one place.  Either way the run is recorded to
``_runs/<run_id>/`` and everything downstream reads it back from there.

Only two things differ from the X(pi) runs this mirrors
(``_dev_logs/F06min_construction.py``, ``F06b_construction.py``, ``F06e_construction.py``):
``theta = pi/2``, and the fluence cap is quoted against ``F0 = theta**2 / T`` -- the
square-pulse floor *for this angle*.  Keeping the hard-coded ``pi**2 / T`` would turn
"cap = 2 F0" into "cap = 8 F0" and silently switch the validity guard off.  Nothing else
is retuned; in particular the budget weights are conditional on the assumed noise
moments, so changing them would be changing the noise model rather than the gate.

What to watch for that the X(pi) runs could not show
----------------------------------------------------
A pi/2 Gaussian carries half the area, so its peak sits near 24% of ``Omega_max`` instead
of 48%.  The peak epigraph is therefore far from active at the warm start and the
optimizer has roughly twice the amplitude headroom it had for X(pi) -- the same direction
that drove arm C out of the surrogate's domain of validity.  Expect the fluence cap, not
the peak bound, to be the binding guard; the ``fluence / F0`` and ``peak_phys`` lines
printed at the end of each run are how you check that.

A finished run registers itself in ``configs/runs.json`` (``--no-register`` opts out),
so both this folder's notebook and ``_Ypi2``'s -- which reads the same entry -- pick
the arm up on their next kernel restart. No source edit, and no chance of the two
notebooks disagreeing about which run an arm came from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import armlib as al  # noqa: E402
from gatelib import ARM_RECIPES  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arm", choices=sorted(ARM_RECIPES))
    p.add_argument("--run-id", default=None,
                   help="override the default <date>_Xpi2_arm<X> run id")
    p.add_argument("--maxiter", type=int, default=None,
                   help="override the recipe's maxiter")
    p.add_argument("--dry-run", action="store_true",
                   help="build the problem, print the manifest, solve nothing")
    p.add_argument("--no-register", action="store_true",
                   help="do not write the finished run into configs/runs.json")
    args = p.parse_args()

    arm, result = al.solve_arm(args.arm, run_id=args.run_id, maxiter=args.maxiter,
                               dry_run=args.dry_run, verbose=1)
    if args.dry_run or arm is None:
        return
    if args.no_register:
        print(f"\n--no-register: run {arm.run_id} left out of the registry; add it "
              f"with gatespec.register_run({al.GATE_KEY!r}, {args.arm!r}, "
              f"{arm.run_id!r}) when you want the notebooks to see it.")
    else:
        al.register_solved(args.arm, arm.run_id)


if __name__ == "__main__":
    main()
