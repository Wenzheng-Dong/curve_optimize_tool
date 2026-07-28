"""curve-opt: constrained sine-series pulse design for noise-robust quantum gates.

The control fields are expanded in a sine series on ``[0, T]``::

    Omega_x(t) = sum_n a_n sin(n pi t / T)
    Omega_y(t) = sum_n b_n sin(n pi t / T)

and the optimization runs at the *control* level over the coefficients
``(a, b)``, with the gate / closure / zero-area conditions imposed as hard
equality constraints. See ``_plan.md`` for the physics conventions (§2), the
problem statement (§3), the module layout (§4) and the run-record schema (§5).

Importing this package enables float64 in JAX process-wide. This is not a
convenience: JAX defaults to float32, and every residual claim in this project
lives at 1e-9..1e-17, which float32 cannot represent. The switch must happen
before the first JAX array is created, so it belongs here -- at import time of
the package root -- and nowhere else.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"

#: Module names of the architecture in ``_plan.md`` §4.1, in dependency order
#: (input layer -> core -> optimize/record -> presentation).
MODULES = (
    "basis",
    "geometry",
    "propagate",
    "metrics",
    "ansatz",
    "optimize",
    "recorder",
    "plotting",
)

__all__ = ["MODULES", "__version__", "x64_enabled"]


def x64_enabled() -> bool:
    """Return whether JAX float64 is active in this process.

    Every entry point that can produce a numerical claim should be able to
    assert this cheaply; see ``tests/test_env.py``.
    """
    return bool(jax.config.jax_enable_x64)
