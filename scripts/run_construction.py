"""Construction runs -- a CLI over ``GateLab.solve_arm`` for any registered gate.

    python scripts/run_construction.py Xpi2 C            # plain budget objective
    python scripts/run_construction.py Xpi2 D            # + hard fluence cap (F06b recipe)
    python scripts/run_construction.py Xpi2 E            # AD-exact objective   (F06e recipe)
    python scripts/run_construction.py Xpi2 F            # AD-robust + e0 fuse  (F06f recipe)
    python scripts/run_construction.py Xpi2 C --dry-run  # build the problem, print manifest

The recipes live in ``curve_opt.gates.armspec::ARM_RECIPES`` -- one definition of
what each arm *is*, shared with any interactive session that calls the same
``solve_arm``.  A finished run is recorded to ``_runs/<run_id>/`` and registered in
``configs/runs.json`` (``--no-register`` opts out), so every downstream reader picks
the arm up without a source edit.

Y gates (``rotate_from_x`` specs) refuse to solve here by design: they inherit their
arms from the corresponding X gate by exact frame rotation -- solve the X gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from curve_opt.gates.armspec import ARM_RECIPES  # noqa: E402
from curve_opt.gates.gatelib import GateLab  # noqa: E402
from curve_opt.gates.gatespec import SPECS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gate", choices=sorted(SPECS))
    p.add_argument("arm", choices=sorted(ARM_RECIPES))
    p.add_argument("--run-id", default=None,
                   help="override the default <date>_<gate>_arm<X> run id")
    p.add_argument("--maxiter", type=int, default=None,
                   help="override the recipe's maxiter")
    p.add_argument("--dry-run", action="store_true",
                   help="build the problem, print the manifest, solve nothing")
    p.add_argument("--no-register", action="store_true",
                   help="do not write the finished run into configs/runs.json")
    args = p.parse_args()

    lab = GateLab(SPECS[args.gate])
    arm, result = lab.solve_arm(args.arm, run_id=args.run_id, maxiter=args.maxiter,
                                dry_run=args.dry_run, verbose=1)
    if args.dry_run or arm is None:
        return
    if args.no_register:
        print(f"\n--no-register: run {arm.run_id} left out of the registry; add it "
              f"with gatespec.register_run({args.gate!r}, {args.arm!r}, "
              f"{arm.run_id!r}) when you want downstream readers to see it.")
    else:
        lab.register_solved(args.arm, arm.run_id)


if __name__ == "__main__":
    main()
