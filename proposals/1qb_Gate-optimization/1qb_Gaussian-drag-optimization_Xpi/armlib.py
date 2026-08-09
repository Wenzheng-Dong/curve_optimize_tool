"""``armlib`` for the X(pi) proposal -- a thin shim over ``proposals/_gatelib``.

This folder is the original single-gate proposal, and until F08a it carried its own
475-line copy of the arm machinery while ``_Xpi2``/``_Ypi``/``_Ypi2`` had already
moved to the shared ``_gatelib``.  The two copies each held their own run-id table,
and they had already drifted apart once (arm E pointed at round 1 here and round 2
there).  There is now one implementation; this file only binds it to X(pi).

``solve_mode = "independent"``: arms C/D/E/F come from this folder's own recorded
``solve_budget`` runs, registered in ``_gatelib/gatespec.py``'s ``SPECS["Xpi"]``.
Arm E additionally registers a ``trace_run_ids`` entry -- its round-1 run, which is
kept for the Figure-4 descent arc only and backs no arm.

``ARM_ORDER`` is decided at *import* time from which of X(pi)'s ``_runs/``
directories exist, so restart the kernel after a new solve lands.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Find _gatelib by walking up, not by a fixed number of parents: these folders
# were regrouped once already and every hard-coded depth broke at that moment.
_here = Path(__file__).resolve()
for _up in _here.parents:
    if (_up / "_gatelib" / "gatespec.py").exists():
        sys.path.insert(0, str(_up / "_gatelib"))
        break
else:
    raise ImportError(f"no _gatelib/ found above {_here}")

from gatespec import SPECS            # noqa: E402
from gatelib import GateLab, bind     # noqa: E402

GATE_KEY = "Xpi"
LAB = bind(GateLab(SPECS[GATE_KEY]), globals())
