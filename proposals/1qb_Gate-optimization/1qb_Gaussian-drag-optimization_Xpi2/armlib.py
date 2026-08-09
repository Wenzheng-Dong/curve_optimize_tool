"""``armlib`` for the X(pi/2) proposal -- a thin shim over ``proposals/_gatelib``.

All the machinery (arm construction, Table-1 evaluation, noise sweep, checkpoint
trace, on-disk export) is gate-generic and lives in ``_gatelib/gatelib.py``; this
file only binds it to this folder's target gate, so the notebook keeps calling
``al.characterize(...)``, ``al.ARM_ORDER`` and friends exactly as the X(pi)
notebook does.

``solve_mode = "independent"``: arms C/D/E come from this folder's own
``solve_budget`` runs (``run_construction.py``).  ``optimize.BudgetProblem``
already accepts an arbitrary ``theta`` against ``propagate.target_x``, so a
different rotation angle needs no ``src/`` change -- only the fluence floor moves
(``theta**2 / T``; see ``_gatelib/gatespec.py``).

``ARM_ORDER`` is decided at *import* time from which ``_runs/`` directories
exist, so restart the kernel after a new solve lands.
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

GATE_KEY = "Xpi2"
LAB = bind(GateLab(SPECS[GATE_KEY]), globals())
