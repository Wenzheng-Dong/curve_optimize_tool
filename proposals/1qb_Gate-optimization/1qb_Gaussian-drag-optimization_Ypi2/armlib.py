"""``armlib`` for the Y(pi/2) proposal -- a thin shim over ``proposals/_gatelib``.

All the machinery is gate-generic and lives in ``_gatelib/gatelib.py``; this file
only binds it to this folder's target gate.

``solve_mode = "rotate_from_x"``: every arm is the corresponding X(pi/2) arm
played with the drive phase advanced by ``pi/2`` (exact frame rotation of the
three-level model including both noise channels -- see ``_gatelib/gatespec.py``).
This folder therefore only populates its optimized arms once the X(pi/2) solves
in ``1qb_Gaussian-drag-optimization_Xpi2`` have landed; arms A and B work
immediately.

``ARM_ORDER`` is decided at *import* time from which of X(pi/2)'s ``_runs/``
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

GATE_KEY = "Ypi2"
LAB = bind(GateLab(SPECS[GATE_KEY]), globals())
