"""One place that says what each arm *is* -- baselines and optimized recipes alike.

Before F08a the answer was split three ways: arms C/D/E were a dict-of-dicts in
``gatelib.py`` carrying four keys; arm F had no recipe at all and existed only as a
hand-written ``BudgetProblem`` in ``_dev_logs/F06f_construction.py``; and the
Gaussian baseline's shape parameters were literals inside
``GateLab.build_gaussian_drag``.  This module holds all six.

Why arm F forced the rewrite
----------------------------
The old four-key recipe (``cap_mult / objective_mode / maxiter / warm``) could not
express F, which additionally needs ``rho=0.8``, ``N_grid=n_peak=120``,
``e0_cap=7.2e-6`` and ``fluence_cap=None``.  So ``solve_arm('F')`` raised, and
X(pi/2) / Y(pi/2) could not grow an F arm at all -- a structural gap, not a missing
run.  :class:`ArmRecipe` therefore carries **every** ``BudgetProblem`` field that
varies between arms; adding a seventh arm should never again require touching
``gatelib``.

What is *not* here
------------------
Anything that is a property of the gate (``theta``, ``phi_axis``, the run registry)
lives in :mod:`gatespec`; anything that is a property of the device or the noise
hypothesis lives in :mod:`curve_opt.device`.  A recipe is the third thing: the
*method* by which an arm is produced, held fixed across gates on purpose, so that
"arm D of X(pi/2)" means the same construction as "arm D of X(pi)".

Scaling with the gate
---------------------
Two recipe numbers are quoted *relative* to the gate rather than absolutely:

``cap_mult``
    multiplies this gate's fluence floor ``F0 = theta**2 / T``, never a hard-coded
    ``pi**2 / T``.  Keeping the absolute number would turn "cap = 2 F0" into
    "cap = 8 F0" for a pi/2 gate and switch the validity guard off silently.
``e0_cap = "arm_B"``
    means "measure arm B's zero-noise ``1 - Fbar`` for this gate and use that".
    X(pi)'s arm F is registered with the literal ``7.2e-6`` it was actually solved
    with, so the recorded run stays reproducible from the registry; new gates should
    prefer ``"arm_B"``, which is what that number *meant*.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["ArmRecipe", "BaselineRecipe", "ARM_RECIPES", "BASELINE_RECIPES",
           "ARM_NAMES", "SLUGS", "ARM_COLORS", "BASELINE_ARMS", "OPTIMIZED_ARMS",
           "ALL_ARMS", "recipe_for", "replace"]


# ---------------------------------------------------------------------------
# baselines (arms A and B) -- constructed, not solved
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineRecipe:
    """A closed-form baseline arm: the truncated Gaussian, with or without DRAG."""

    key: str
    name: str
    layer: str
    """``"planar"`` (arm A: ``Omega_y == 0``) or ``"planar_drag"`` (arm B)."""

    sigma_divisor: float = 6.0
    r"""Gaussian width as ``sigma = T / sigma_divisor``.

    ⚠️ Stored as a **divisor**, and the call site must keep writing ``T /
    sigma_divisor``, because ``T / 6.0`` and ``T * (1/6.0)`` are *different
    doubles* at ``T = 50`` (8.333333333333334 vs 8.333333333333332). Rewriting the
    division as a multiplication would perturb every downstream number in the
    baseline by one ulp and silently invalidate every comparison against the
    recorded F05/F06 values.
    """

    n_fit: int = 8000
    """Grid the Gaussian is least-squares fitted onto before truncation to ``M`` modes."""

    note: str = ""


BASELINE_RECIPES: dict[str, BaselineRecipe] = {
    "A": BaselineRecipe(
        key="A", name="Gaussian", layer="planar",
        note=("Arm B's Gaussian kappa played *without* the DRAG/Stark readout map: "
              "Omega_y == 0, no DRAG, no Stark correction. The no-leakage-correction "
              "reference point."),
    ),
    "B": BaselineRecipe(
        key="B", name="Gaussian + DRAG", layer="planar_drag",
        note=("Truncated Gaussian (sigma = T / 6) least-squares fitted to M sine "
              "modes, c1-projected and rescaled so the *truncated numerical* area "
              "equals theta exactly, then played through the DRAG/Stark readout map. "
              "The industry baseline every optimized arm is scored against."),
    ),
}


# ---------------------------------------------------------------------------
# optimized arms (C..F) -- solved through GateLab.solve_arm
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmRecipe:
    """Everything ``solve_budget`` needs that varies from one optimized arm to the next.

    Every field maps onto a :class:`curve_opt.optimize.BudgetProblem` field of the
    same meaning, except :attr:`cap_mult` (scaled by the gate's fluence floor to give
    ``fluence_cap``) and :attr:`warm` (resolved to starting coefficients).  Fields
    whose default matches ``BudgetProblem``'s own default are still spelled out here,
    so that reading one recipe tells you the whole problem without a second lookup.
    """

    key: str
    name: str
    warm: str
    """Arm key this one warm-starts from."""

    objective_mode: str
    """``"budget"`` | ``"ad_exact"`` | ``"ad_robust"`` -- see ``BudgetProblem``."""

    maxiter: int

    cap_mult: float | None = None
    """Hard fluence cap as a multiple of **this gate's** ``F0 = theta**2 / T``.
    ``None`` disables the constraint."""

    e0_cap: float | str | None = None
    """Hard cap on the zero-noise infidelity: a float, ``"arm_B"`` (measure it),
    or ``None`` (no fuse). Only meaningful with ``objective_mode="ad_robust"``,
    whose objective no longer prices the zero-noise term."""

    rho: float = 0.5
    """F05b R guard: ``max|tau| <= rho * |Delta| / 2``."""

    N_grid: int = 300
    """Grid the objective and the gate/e0 constraints are evaluated on."""

    n_peak: int = 120
    """Coarser grid for the peak and R-guard epigraph inequalities."""

    hessian_mode: str = "default"
    gate_level: str = "three_level"
    checkpoint_every: int = 100
    note: str = ""


ARM_RECIPES: dict[str, ArmRecipe] = {
    "C": ArmRecipe(
        key="C", name="Optimized (3D)", warm="B",
        objective_mode="budget", maxiter=6000,
        cap_mult=None,
        note=("F06-min recipe: general layer, Gaussian+DRAG warm start, "
              "gate_level='three_level' throughout, hessian_mode='default', hard "
              "constraints = gate residual (vector, polar decomposition) + c1 endpoints "
              "+ peak epigraph + R guard. No fluence cap -- the deliberately unguarded "
              "arm, kept so a new gate shows whether it reproduces the "
              "surrogate-vs-truth divergence X(pi) had."),
    ),
    "D": ArmRecipe(
        key="D", name="Opt + fluence cap", warm="B",
        objective_mode="budget", maxiter=20000,
        cap_mult=2.0,
        note=("F06b recipe: arm C's problem plus the hard fluence cap "
              "int kappa^2 dt <= 2 F0 with F0 = theta^2 / T. The cap is the validity "
              "guard that keeps the achieved eta near the minimum the gate requires."),
    ),
    "E": ArmRecipe(
        key="E", name="Opt AD-exact", warm="D",
        objective_mode="ad_exact", maxiter=1500,
        cap_mult=4.0,
        note=("F06e recipe: same hard-constraint set as D, but the objective is the "
              "second-order noise expansion of the *true* three-level expected "
              "infidelity via forward-mode AD (optimize._budget_objective_ad) -- no "
              "design-curve surrogate, hence immune to the tau != 0 readout-map "
              "truncation F06d located. Cap loosened to 4 F0: with the exact objective "
              "the cap guards against higher-than-second-order terms only, it is not "
              "the primary regularizer."),
    ),
    "F": ArmRecipe(
        key="F", name="Opt AD-robust", warm="E",
        objective_mode="ad_robust", maxiter=6000,
        cap_mult=None,
        e0_cap=7.2e-6,
        rho=0.8,
        N_grid=120, n_peak=120,
        note=("F06f recipe: minimize the noise *sensitivity* alone -- the same "
              "second-order expansion as E with the zero-noise term dropped -- so the "
              "advantage grows with the error rather than concentrating at zero noise. "
              "Evaluated on the 120-point AWG playback grid (three-level propagation of "
              "a piecewise-constant waveform is exact there, which kills E's "
              "sampling-loss artefact). Both surrogate-era walls come down: no fluence "
              "cap (the surrogate is not what is being optimized) and the R guard opens "
              "to rho = 0.8. The e0 cap replaces them as the fuse that stops leakage "
              "regrowing while the objective no longer prices it; X(pi) uses the "
              "literal 7.2e-6 it was solved with (= arm B's zero-noise level), new "
              "gates should use e0_cap='arm_B'."),
    ),
}

BASELINE_ARMS = ("A", "B")
OPTIMIZED_ARMS = ("C", "D", "E", "F")
ALL_ARMS = BASELINE_ARMS + OPTIMIZED_ARMS

ARM_NAMES: dict[str, str] = {
    **{k: v.name for k, v in BASELINE_RECIPES.items()},
    **{k: v.name for k, v in ARM_RECIPES.items()},
}

SLUGS = {"A": "A_gaussian", "B": "B_gaussian_drag", "C": "C_optimized",
         "D": "D_optimized_fluence_cap", "E": "E_optimized_ad_exact",
         "F": "F_optimized_ad_robust"}
"""On-disk filename stem for each arm's exported waveform."""

ARM_COLORS = {"A": "#888888", "B": "#1f77b4", "C": "#d62728", "D": "#2ca02c",
              "E": "#9467bd", "F": "#ff7f0e"}


def recipe_for(arm_key: str) -> ArmRecipe:
    """The optimized-arm recipe, with a message that names the alternatives."""
    try:
        return ARM_RECIPES[arm_key]
    except KeyError:
        if arm_key in BASELINE_RECIPES:
            raise ValueError(
                f"arm {arm_key!r} is a closed-form baseline "
                f"({BASELINE_RECIPES[arm_key].name}), not a solve -- build it with "
                f"GateLab.build_gaussian_drag()/build_arm_a()") from None
        raise ValueError(
            f"no recipe for arm {arm_key!r}; have {sorted(ARM_RECIPES)}") from None
