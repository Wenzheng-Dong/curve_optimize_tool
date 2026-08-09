"""``armlib`` for the Y(pi) proposal -- a thin shim over ``proposals/_gatelib``.

All the machinery is gate-generic and lives in ``_gatelib/gatelib.py``; this file
only binds it to this folder's target gate.

``solve_mode = "rotate_from_x"``: every arm is the corresponding X(pi) arm played
with the drive phase advanced by ``pi/2``.  That is an *exact* frame rotation of
the three-level model -- including both noise channels -- so no separate
optimization is run here.  The derivation is in ``_gatelib/gatespec.py``'s module
docstring; ``covariance_check()`` verifies it numerically, and that check is this
folder's actual deliverable.

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

GATE_KEY = "Ypi"
LAB = bind(GateLab(SPECS[GATE_KEY]), globals())
