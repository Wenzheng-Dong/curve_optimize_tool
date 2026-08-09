"""``armlib`` for the Novera Y(pi) proposal -- ``_gatelib``, bound to the Novera device.

Mirrors ``Novera_Xpi/armlib.py``; see that file's docstring for why ``device=`` is
threaded explicitly instead of relying on ``_gatelib``'s module-default device.

``solve_mode = "rotate_from_x"``: every arm is X(pi) [Novera]'s solved arm with the
drive phase advanced by ``phi_axis = pi/2`` -- no separate optimization (module
docstring of ``gatespec.py``; verified numerically by ``check_covariance.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

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

GATE_KEY = "Ypi_novera"
DEVICE_CONFIG = REPO_ROOT / "configs" / "device_novera.json"

NOVERA = recorder.load_device(DEVICE_CONFIG)
assert NOVERA.anharmonicity == -0.19848, (
    f"device_novera.json did not load the measured anharmonicity: got "
    f"{NOVERA.anharmonicity}, expected -0.19848 -- this armlib would silently bind "
    f"the wrong chip.")

LAB = bind(GateLab(SPECS[GATE_KEY], device=NOVERA), globals())
