"""``armlib`` for the Novera X(pi) proposal -- ``_gatelib``, bound to the Novera device.

Same shim pattern as ``1qb_Gaussian-drag-optimization_Xpi/armlib.py`` (F08a), with one
addition every default-device folder does not need: the ``device=`` argument (N1).
``_gatelib`` captures ``device.DEFAULT_DEVICE`` as a module global, so without this the
gate would silently build against the *wrong* chip -- the exact trap
``proposals/Novera_work/README.md`` and the N3 task brief both call out by name. This
file is the one place that binding happens; every notebook cell downstream reads
``al.DEV``, never the default.

``solve_mode = "independent"``: arms C/D/E/F are the N2 fast-path solves, registered in
``_gatelib/gatespec.py``'s ``SPECS["Xpi_novera"]`` via ``configs/runs.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Find _gatelib by walking up, not by a fixed number of parents (F08a's own lesson --
# the gate folders have already been regrouped once).
_here = Path(__file__).resolve()
for _up in _here.parents:
    if (_up / "_gatelib" / "gatespec.py").exists():
        REPO_ROOT = _up.parent
        sys.path.insert(0, str(_up / "_gatelib"))
        sys.path.insert(0, str(REPO_ROOT / "src"))
        break
else:
    raise ImportError(f"no _gatelib/ found above {_here}")

from curve_opt import recorder        # noqa: E402
from gatespec import SPECS            # noqa: E402
from gatelib import GateLab, bind     # noqa: E402

GATE_KEY = "Xpi_novera"
DEVICE_CONFIG = REPO_ROOT / "configs" / "device_novera.json"

NOVERA = recorder.load_device(DEVICE_CONFIG)
assert NOVERA.anharmonicity == -0.19848, (
    f"device_novera.json did not load the measured anharmonicity: got "
    f"{NOVERA.anharmonicity}, expected -0.19848 -- this armlib would silently bind "
    f"the wrong chip.")

LAB = bind(GateLab(SPECS[GATE_KEY], device=NOVERA), globals())
