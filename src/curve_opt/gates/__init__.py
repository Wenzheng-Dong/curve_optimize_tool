"""Gate-level configuration and construction layer (F09: adopted from ``proposals/_gatelib``).

Four modules, in dependency order:

    gatespec   which target gates exist (theta, phi_axis, run registry)
    armspec    what each arm *is* (objective / caps / guards / grids / warm starts)
    armcore    the arm abstraction and the fidelity primitives (polar decomposition,
               ``fit_phi_vz``, ``fidelity_jax``) shared by every construction
    gatelib    ``GateLab`` -- build / solve / characterize / sweep / export one gate

Until F09 these lived in the git-ignored ``proposals/_gatelib`` and were reached
through per-folder ``sys.path`` shims. They are now a normal subpackage: import as

    from curve_opt.gates import gatelib, gatespec
    lab = gatelib.GateLab(gatespec.SPECS["Xpi"])
"""

from __future__ import annotations
