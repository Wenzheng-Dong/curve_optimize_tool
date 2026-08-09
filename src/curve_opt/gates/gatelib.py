"""Gate-generic arm construction, evaluation and noise sweep.

This is the historical ``1qb_Gaussian-drag-optimization/armlib.py`` with the
target gate lifted out of module scope and into a :class:`gatespec.GateSpec`.
Instantiate one :class:`GateLab` per gate (``GateLab(SPECS["Xpi"])``); the
:func:`bind` helper publishes its methods into a flat namespace for callers that
keep the original ``al.characterize(...)`` call style (the golden checker does).

Arms (unchanged in meaning from the X(pi) proposal)::

    A  "Gaussian"          layer `planar`       truncated Gaussian, no DRAG
    B  "Gaussian + DRAG"   layer `planar_drag`  same kappa, DRAG/Stark readout
    C  "Optimized (3D)"    layer `general`      solve_budget, warm started from B
    D  "Opt + fluence cap" layer `general`      C plus the hard fluence cap
    E  "Opt AD-exact"      layer `general`      objective_mode="ad_exact"
    F  "Opt AD-robust"     layer `general`      objective_mode="ad_robust" + e0 fuse

C..F appear only if the corresponding ``run_ids`` entry is set *and* the run
directory exists, so a fresh gate's notebook runs top-to-bottom on A/B alone
before any optimizer time has been spent.

What each arm *is* -- baseline shape parameters, objective, caps, guards, grids --
lives in :mod:`armspec`, not here (F08a). Before that split, arm F had no recipe at
all and could only be loaded, never solved, for any gate.

What is reused rather than rebuilt
----------------------------------
:mod:`armcore` supplies ``Input`` (the arm abstraction: name / layer /
``(a, c, Phi_0)`` / ``.broadcast`` / ``.predicted_terms``, including the arm-A
``C4`` evaluation-object trap), ``c1_project_and_rescale``, ``fit_phi_vz``,
``rz_matrix`` and ``fidelity_jax`` -- the F05 primitives, moved out of the dev log
by F08c so that proposals no longer import a frozen archive at runtime.  What F05
hard-codes at module scope -- ``THETA`` and ``U_TARGET`` -- is re-implemented here
as gate-parametric methods (:meth:`GateLab.build_gaussian_drag`,
:meth:`GateLab.characterize_zero_noise`, :meth:`GateLab._make_mc_fn`) so that
numbers remain term-for-term comparable with the accepted F05/F06 dev-log values
on the X(pi) gate while other angles work at all.

Evaluation-object rule (project red line): ``C1/C2/C3`` on the **design curve**;
gate residual, ``C4`` and the peak on the **broadcast (played)** waveform.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

from curve_opt import basis, budget, device, gate, geometry, parametrization, propagate
from curve_opt.gates import armcore, gatespec
from curve_opt.gates.gatespec import (  # noqa: F401
    DEV, GateSpec, REPO_ROOT, SPECS, T, rotate_drive, source_spec,
)
from curve_opt.gates.armspec import (
    ARM_COLORS, ARM_NAMES, ARM_RECIPES, BASELINE_ARMS, BASELINE_RECIPES,
    ArmRecipe, BaselineRecipe, OPTIMIZED_ARMS, SLUGS, recipe_for,
)

N_DESIGN = 4000
"""Design/broadcast quadrature grid for every reported deterministic quantity."""

N_SIM = 300
"""Propagator grid for the noise sweep (147 grid points x 5 arms)."""

N_EXPORT = int(round(DEV.sample_rate * DEV.gate_time))  # 2.4 GS/s * 50 ns = 120

EPS_GRID = np.array([-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05])
DELTA_MHZ_GRID = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
"""Static detuning in MHz. The design assumption is 0.1 MHz rms, so this reaches 5 sigma."""


def delta_rate(delta_mhz) -> np.ndarray:
    """MHz -> rad/ns angular detuning (``delta_z = 2 pi f``)."""
    return 2.0 * np.pi * np.asarray(delta_mhz, dtype=float) * 1e-3


# ---------------------------------------------------------------------------
# a rotated-frame arm (the Y gates)
# ---------------------------------------------------------------------------


class RotatedInput(armcore.Input):
    """An X-axis arm played with the drive phase advanced by ``phi_axis``.

    The design curve ``(a, c, Phi_0)`` is *not* touched: the rotation is a frame
    change on the played quadratures, so ``kappa``, ``tau`` and therefore
    ``C1, C2, C3`` are literally the source arm's.  ``C4 ~ |Lambda|^2`` and the
    peak ``|Omega|`` are rotation invariant as well, so the whole Table-1 row is
    inherited -- which is the claim the Y notebooks verify numerically rather
    than assert (see ``gatespec`` module docstring).
    """

    def __init__(self, source: armcore.Input, phi_axis: float, name: str | None = None):
        super().__init__(
            name if name is not None else source.name,
            source.layer,
            source.a,
            source.c,
            source.Phi_0,
            f"{source.note}  Played with the drive phase advanced by "
            f"phi_axis={phi_axis:.6f} rad -- exact frame rotation of the "
            f"corresponding X-axis arm, not an independent solve.",
        )
        self.phi_axis = float(phi_axis)
        self.source = source
        self.run_id = getattr(source, "run_id", None)
        if hasattr(source, "solver_phi_vz"):
            self.solver_phi_vz = source.solver_phi_vz

    def broadcast(self, N: int = N_DESIGN):
        return rotate_drive(*super().broadcast(N), self.phi_axis)


# ---------------------------------------------------------------------------
# the lab
# ---------------------------------------------------------------------------


def bind(lab: "GateLab", ns: dict) -> "GateLab":
    """Publish *lab*'s methods and constants into a folder's ``armlib`` namespace.

    Lets each notebook keep the flat ``al.characterize(...)`` / ``al.ARM_ORDER``
    call style of the original X(pi) proposal while the gate itself lives in one
    :class:`GateLab`.  ``ARM_ORDER`` is a snapshot taken at import time (it depends
    on which runs exist on disk); re-import after a new solve lands.
    """
    for name in ("DEV", "T", "theta", "phi_axis", "U_TARGET", "FLUENCE_FLOOR",
                 "N_DESIGN", "N_SIM", "N_EXPORT", "ARM_COLORS",
                 "EPS_GRID", "DELTA_MHZ_GRID", "spec", "folder"):
        ns[name] = getattr(lab, name)
    ns["THETA"] = lab.theta
    ns["GATE"] = lab.spec
    ns["LAB"] = lab
    ns["ARM_ORDER"] = lab.ARM_ORDER
    ns["ARM_NAMES"] = ARM_NAMES
    ns["SLUGS"] = SLUGS
    ns["ARM_RECIPES"] = ARM_RECIPES
    ns["BASELINE_RECIPES"] = BASELINE_RECIPES
    ns["OPTIMIZED_ARMS"] = OPTIMIZED_ARMS
    ns["delta_rate"] = delta_rate
    ns["rotate_drive"] = rotate_drive
    for name in ("run_dir", "trace_run_dir",
                 "build_gaussian_drag", "build_arm_a", "load_optimized",
                 "build_arms", "design_of", "broadcast_of", "geometry_of",
                 "characterize_zero_noise", "characterize",
                 "solve_arm", "register_solved",
                 "sweep", "zero_noise_consistency", "c3_prediction",
                 "checkpoint_trace", "covariance_check", "export_arms"):
        ns[name] = getattr(lab, name)
    return lab


class GateLab:
    """Everything the notebook for one target gate needs."""

    def __init__(self, spec: GateSpec, n_design: int = N_DESIGN, n_sim: int = N_SIM):
        self.spec = spec
        self.DEV = DEV
        self.T = T
        self.theta = float(spec.theta)
        self.phi_axis = float(spec.phi_axis)
        self.U_TARGET = spec.U_target
        self.FLUENCE_FLOOR = spec.fluence_floor
        self.N_DESIGN = n_design
        self.N_SIM = n_sim
        self.N_EXPORT = N_EXPORT
        self.ARM_COLORS = ARM_COLORS
        self.EPS_GRID = EPS_GRID
        self.DELTA_MHZ_GRID = DELTA_MHZ_GRID
        self.folder = gatespec.gate_dir(spec)

    # -- run directories ---------------------------------------------------

    def run_dir(self, arm_key: str) -> Path | None:
        """``_runs/<run_id>`` for an optimized arm, or ``None`` if not solved yet."""
        sp = source_spec(self.spec) if self.spec.is_rotated else self.spec
        run_id = (sp.run_ids or {}).get(arm_key)
        if not run_id:
            return None
        d = REPO_ROOT / "_runs" / run_id
        return d if d.exists() else None

    def trace_run_dir(self, arm_key: str) -> Path | None:
        """``_runs/<run_id>`` for a *trajectory-only* run (``GateSpec.trace_run_ids``).

        Not inherited through ``source_key``: a convergence history belongs to the
        gate that ran the solve, so the rotated gates return ``None`` here.
        """
        run_id = (self.spec.trace_run_ids or {}).get(arm_key)
        if not run_id:
            return None
        d = REPO_ROOT / "_runs" / run_id
        return d if d.exists() else None

    @property
    def ARM_ORDER(self) -> tuple[str, ...]:
        """A, B, plus whichever optimized arms actually have a run on disk."""
        return BASELINE_ARMS + tuple(k for k in OPTIMIZED_ARMS
                                     if self.run_dir(k) is not None)

    # -- arm construction --------------------------------------------------

    def build_gaussian_drag(self) -> armcore.Input:
        """Arm B: truncated Gaussian (sigma = T/6) fitted to the sine basis, plus DRAG.

        F05's ``build_gaussian_drag`` with ``THETA`` lifted to ``self.theta``.  The
        amplitude is set so the *truncated numerical* area equals theta exactly
        (not the infinite-Gaussian closed form, which is off by the tail).
        """
        rec = BASELINE_RECIPES["B"]
        N_fit = rec.n_fit
        t = (np.arange(N_fit) + 0.5) * (self.T / N_fit)
        # ⚠️ Must stay a *division*: T / 6.0 and T * (1/6.0) are different doubles at
        # T = 50 ns. See BaselineRecipe.sigma_divisor.
        sigma = self.T / rec.sigma_divisor
        shape = np.exp(-0.5 * ((t - self.T / 2.0) / sigma) ** 2)
        om_gauss = shape * (self.theta / np.trapezoid(shape, t))
        S = basis.design_matrix(t, self.T, DEV.n_modes)
        a_fit, *_ = np.linalg.lstsq(S, om_gauss, rcond=None)
        a = armcore.c1_project_and_rescale(a_fit, self.T, self.theta)
        rel_rms = float(np.linalg.norm(S @ a - om_gauss) / np.linalg.norm(om_gauss))
        return armcore.Input(
            rec.name, rec.layer, a, np.zeros(DEV.n_modes), 0.0,
            f"Truncated Gaussian (sigma=T/{rec.sigma_divisor:g}) for "
            f"theta={self.theta:.6f}, least-squares "
            f"fit to M={DEV.n_modes} sine modes (relRMS after c1 projection = "
            f"{rel_rms:.2e}), DRAG/Stark readout. The industry baseline.",
        )

    def build_arm_a(self, arm_b: armcore.Input) -> armcore.Input:
        """Arm A: arm B's Gaussian kappa played *without* the DRAG/Stark readout."""
        rec = BASELINE_RECIPES["A"]
        return armcore.Input(
            rec.name, rec.layer, arm_b.a, arm_b.c, arm_b.Phi_0,
            f"Truncated Gaussian (sigma=T/{BASELINE_RECIPES['B'].sigma_divisor:g}) "
            f"fitted to M={DEV.n_modes} sine modes, "
            f"c1-projected and rescaled to theta={self.theta:.6f}. "
            "Omega_y == 0: no DRAG, no Stark correction.",
        )

    def load_optimized(self, arm_key: str) -> armcore.Input | None:
        """Load arm C/D/E from its recorded run's **last checkpoint**.

        ``final.json`` does not persist coefficients, so the arm is taken from the
        last row of ``history.npz``.  That is an iterate, not the literal final
        point -- stated here rather than glossed over; the X(pi) run's checkpoint
        agreed with ``final.json`` to 2e-5 relative.
        """
        run_dir = self.run_dir(arm_key)
        if run_dir is None:
            return None
        hist = np.load(run_dir / "history.npz")
        arm = armcore.Input(
            ARM_NAMES[arm_key], "general",
            hist["coeffs_a"][-1], hist["coeffs_c"][-1], float(hist["Phi_0"][-1]),
            f"solve_budget general-layer solution for theta={self.theta:.6f}, loaded "
            f"from {run_dir.name} checkpoint iter={int(hist['iter'][-1])}.",
        )
        arm.run_id = run_dir.name
        arm.solver_phi_vz = float(hist["phi_vz"][-1])
        return arm

    def build_arms(self, resolve_c: bool = False) -> dict[str, armcore.Input]:
        """All available arms for this gate, in ``ARM_ORDER``.

        For a ``rotate_from_x`` spec every arm is built on the source gate's axis
        and then wrapped in :class:`RotatedInput`.

        *resolve_c* is accepted for backwards compatibility with the X(pi) notebook's
        ``RESOLVE_ARM_C`` switch, which predates the recorder-always contract. Every
        arm now comes off disk unconditionally -- ``solve_arm`` records its result and
        the notebook reads it back -- so re-solving here would produce an arm that is
        *not* the one the figures cite. Passing ``True`` therefore raises rather than
        silently doing something different from what the flag used to mean.
        """
        if resolve_c:
            raise ValueError(
                "build_arms(resolve_c=True) is no longer supported: arms are loaded "
                "from their recorded _runs/ directory so that the notebook and the "
                "dev logs cite the same solve. To re-solve, call "
                "solve_arm('C') explicitly -- it records to _runs/ and you then "
                "register the new run_id in the gate registry.")
        arm_b = self.build_gaussian_drag()
        arms: dict[str, armcore.Input] = {"A": self.build_arm_a(arm_b), "B": arm_b}
        for k in OPTIMIZED_ARMS:
            arm = self.load_optimized(k)
            if arm is not None:
                arms[k] = arm
        if self.spec.is_rotated:
            arms = {k: RotatedInput(v, self.phi_axis, v.name) for k, v in arms.items()}
        return arms

    # -- solving an optimized arm -----------------------------------------

    def solve_arm(self, arm_key: str, *, run_id: str | None = None,
                  maxiter: int | None = None, warm_arm=None, n_grid: int | None = None,
                  n_peak: int | None = None, recipe: ArmRecipe | None = None,
                  verbose: int = 1, dry_run: bool = False):
        """Run one optimized arm's ``solve_budget`` and return ``(arm, result)``.

        The single entry point for producing arms C/D/E/F, used both by
        ``run_construction.py`` and by the notebook's in-place solve cell -- the recipe
        lives in :data:`armspec.ARM_RECIPES`, so the two cannot drift apart.

        *n_grid* / *n_peak* / *maxiter* default to the recipe's own values and are
        overridable per call (a solver budget is not a physics knob).

        *recipe* replaces the registry entry wholesale, for a gate that genuinely
        needs a different construction -- e.g. an ``e0_cap`` of ``"arm_B"`` where
        X(pi) uses the literal number it was solved with, or a warm start that the
        default lineage cannot supply. Use ``dataclasses.replace(ARM_RECIPES[k], ...)``
        so the deviation is one visible line. ⚠️ The result is then **not** the same
        construction as the registry's arm of that name; the difference lands in the
        run manifest's ``ansatz`` block, and it belongs in the dev log too. Prefer
        editing the registry when the change should apply to every gate.

        Always goes through the recorder into ``_runs/<run_id>/``: the project requires
        every optimization run to be on disk and every plot to be read back from a
        RunRecord, and a solve that only exists in a kernel's memory satisfies neither.
        Chaining in a session is still fine -- pass the previous arm as *warm_arm* --
        but the result is persisted either way, so the notebook re-executes off disk
        afterwards instead of re-solving.

        *warm_arm* defaults to the recipe's warm start resolved through
        :meth:`build_arms`, which requires that arm's ``run_id`` to be registered in
        ``gatespec``.  In a session where you have just solved it, pass the in-memory
        arm instead and skip the registration round-trip.
        """
        from curve_opt import optimize, recorder  # local: heavy, and only needed here

        if self.spec.is_rotated:
            raise ValueError(
                f"{self.spec.name} inherits its arms from {self.spec.source_key} by exact "
                f"frame rotation -- solve there, not here (see _gatelib/README.md)."
            )
        cfg = recipe_for(arm_key) if recipe is None else recipe
        cap = None if cfg.cap_mult is None else cfg.cap_mult * self.FLUENCE_FLOOR
        maxiter = maxiter or cfg.maxiter
        n_grid = cfg.N_grid if n_grid is None else n_grid
        n_peak = cfg.n_peak if n_peak is None else n_peak

        available = None
        if warm_arm is None:
            available = self.build_arms()
            if cfg.warm not in available:
                raise RuntimeError(
                    f"arm {arm_key} warm-starts from arm {cfg.warm}, which is not "
                    f"available for {self.spec.name} (have {sorted(available)}). Solve it "
                    f"first, then either register its run_id in the gate registry or "
                    f"pass it here as warm_arm=."
                )
            warm_arm = available[cfg.warm]
        warm_row = self.characterize(warm_arm, n_grid)

        # ``e0_cap="arm_B"`` means "arm B's zero-noise level for *this* gate" -- the
        # thing X(pi)'s literal 7.2e-6 was. Measured on the solver's own grid so the
        # fuse and the constrained quantity are the same number.
        e0_cap = cfg.e0_cap
        if isinstance(e0_cap, str):
            if e0_cap != "arm_B":
                raise ValueError(
                    f"arm {arm_key}: e0_cap must be a float, None or 'arm_B', "
                    f"got {e0_cap!r}")
            if available is None:
                available = self.build_arms()
            e0_cap = float(self.characterize(available["B"], n_grid)["baseline_infidelity"])
        # A planar_drag warm start enters the general layer at c = 0 by construction.
        c0 = (np.zeros(DEV.n_modes) if warm_arm.layer == "planar_drag"
              else np.asarray(warm_arm.c, dtype=float))

        problem = optimize.BudgetProblem(
            M=DEV.n_modes,
            T=self.T,
            theta=self.theta,
            coeffs0_a=np.asarray(warm_arm.a, dtype=float),
            coeffs0_c=c0,
            Phi_0_0=float(warm_arm.Phi_0),
            phi_vz_0=float(warm_row["phi_vz"]),
            device=DEV,
            gate_level=cfg.gate_level,
            N_grid=n_grid,
            n_peak=n_peak,
            rho=cfg.rho,
            hessian_mode=cfg.hessian_mode,
            maxiter=maxiter,
            checkpoint_every=cfg.checkpoint_every,
            fluence_cap=cap,
            e0_cap=e0_cap,
            objective_mode=cfg.objective_mode,
            ansatz={"name": f"{self.spec.key}_arm{arm_key}_warm_{cfg.warm}",
                    "source": cfg.note,
                    "recipe_overridden": recipe is not None},
            target_gate=self.spec.target_gate_manifest(),
        )

        run_id = run_id or f"{time.strftime('%Y%m%d')}_{self.spec.key}_arm{arm_key}"
        cap_str = "none" if cap is None else f"{cfg.cap_mult:g} F0 = {cap:.6f}"
        if verbose:
            print(f"=== {self.spec.name} arm {arm_key} ===", flush=True)
            print(f"    theta          = {self.theta:.6f} rad", flush=True)
            print(f"    F0             = {self.FLUENCE_FLOOR:.6f} rad^2/ns  (theta^2 / T)",
                  flush=True)
            print(f"    fluence cap    = {cap_str}", flush=True)
            print(f"    e0 cap         = "
                  f"{'none' if e0_cap is None else f'{e0_cap:.4e}'}"
                  f"{'  (= arm B, measured)' if isinstance(cfg.e0_cap, str) else ''}",
                  flush=True)
            print(f"    objective_mode = {cfg.objective_mode}", flush=True)
            print(f"    R guard        = rho {cfg.rho:g} -> max|tau| <= "
                  f"{problem.tau_guard_bound:.6f} rad/ns", flush=True)
            print(f"    grids          = N_grid {n_grid}, n_peak {n_peak}", flush=True)
            print(f"    warm start     = arm {cfg.warm} "
                  f"({warm_arm.layer}), maxiter = {maxiter}", flush=True)
            print(f"    run_id         = {run_id}", flush=True)
        if dry_run:
            print("\n--dry-run: manifest only, nothing solved.")
            print(optimize.budget_manifest_for(problem))
            # The problem is returned (not None) so a caller can diff it against the
            # manifest of a historical run -- that is how F08a proved the recipes
            # reproduce the constructions that were originally hand-written in
            # _dev_logs/F06*_construction.py.
            return None, problem

        rec = recorder.Recorder(run_id, optimize.budget_manifest_for(problem),
                                schema="budget", exist_ok=True)
        t0 = time.perf_counter()
        result = optimize.solve_budget(problem, recorder=rec, verbose=verbose)
        wall = time.perf_counter() - t0

        arm = armcore.Input(
            ARM_NAMES[arm_key], "general", result.coeffs_a, result.coeffs_c,
            float(result.Phi_0),
            f"solve_budget solution for {self.spec.name}, fresh run {run_id}, "
            f"stop_reason={result.stop_reason!r}, nit={result.nit}.",
        )
        arm.run_id = run_id
        arm.solver_phi_vz = float(result.phi_vz)

        # ★ The c1 endpoint condition is an equality constraint, so a solve that stopped
        # before satisfying it produces an arm the DRAG/Stark readout layer will refuse
        # (parametrization.check_c1, "the c1 guard is an entry ticket"). Without this
        # check the failure surfaces much later, as a ValueError deep inside broadcast()
        # -- which reads like a plumbing bug rather than "the solve is not finished".
        c1_res = np.array([basis.c1_row(DEV.n_modes, self.T, "start") @ np.asarray(arm.a),
                           basis.c1_row(DEV.n_modes, self.T, "end") @ np.asarray(arm.a)])
        arm.c1_residual = c1_res
        if np.max(np.abs(c1_res)) > 1e-9:
            print(f"\n!! arm {arm_key} does NOT satisfy the c1 endpoint constraint "
                  f"(|res| = {np.max(np.abs(c1_res)):.3e} > 1e-9).\n"
                  f"   stop_reason={result.stop_reason!r} at nit={result.nit}: the solve "
                  f"stopped before its equality constraints closed.\n"
                  f"   The run is on disk, but this arm cannot be broadcast -- "
                  f"characterize()/export will raise. Re-run with a larger maxiter.",
                  flush=True)
            return arm, result

        if verbose:
            fl = self.geometry_of(arm, self.N_DESIGN)
            print(f"\nstop_reason = {result.stop_reason!r}, nit = {result.nit}, "
                  f"wall = {wall:.1f}s", flush=True)
            print(f"terms         = {result.terms}", flush=True)
            print(f"gate_residual = {np.asarray(result.gate_residual)}", flush=True)
            print(f"peak_phys     = {result.peak_phys:.6f} "
                  f"({result.peak_phys / DEV.rabi_max_rate:.1%} of Omega_max)", flush=True)
            print(f"tau_peak      = {result.tau_peak:.6f} "
                  f"(guard {fl['tau_guard_bound']:.6f})", flush=True)
            print(f"fluence       = {fl['fluence']:.6f} rad^2/ns = "
                  f"{fl['fluence_over_floor']:.3f} F0", flush=True)
        return arm, result

    def register_solved(self, arm_key: str, run_id: str) -> Path:
        """Record a finished solve in ``configs/runs.json`` so notebooks pick it up.

        Replaces the "paste the run_id back into gatespec.py" step the old workflow
        ended on -- the registry is data, and a landed run should not need a source
        edit to become visible.
        """
        path = gatespec.register_run(self.spec.key, arm_key, run_id)
        # ★ GateSpec is frozen, so register_run puts a *new* object into SPECS and
        # this lab's captured self.spec would otherwise keep the stale run_ids --
        # ARM_ORDER and run_dir would go on claiming the arm does not exist, in the
        # very session that just solved it. Re-point at the refreshed spec.
        self.spec = gatespec.SPECS[self.spec.key]
        print(f"registered {self.spec.key}/{arm_key} -> {run_id} in {path}", flush=True)
        return path

    # -- per-arm evaluation ------------------------------------------------

    def design_of(self, arm, N: int | None = None) -> parametrization.DesignCurve:
        return parametrization.design_curve(arm.a, arm.c, arm.Phi_0, self.T,
                                            N or self.N_DESIGN)

    def broadcast_of(self, arm, N: int | None = None):
        """The *played* waveform -- the only object G / C4 / the peak ever see."""
        om_x, om_y = arm.broadcast(N or self.N_DESIGN)
        return np.asarray(om_x), np.asarray(om_y)

    def geometry_of(self, arm, N: int | None = None) -> dict:
        dc = self.design_of(arm, N)
        kappa, tau = np.asarray(dc.kappa), np.asarray(dc.tau)
        fluence = float(np.sum(kappa**2) * dc.dt)
        return {
            "fluence": fluence,
            "fluence_over_floor": fluence / self.FLUENCE_FLOOR,
            "max_abs_tau": float(np.max(np.abs(tau))),
            "tau_guard_bound": 0.5 * abs(DEV.delta) / 2.0,
            "max_abs_kappa": float(np.max(np.abs(kappa))),
        }

    def characterize_zero_noise(self, arm, N: int | None = None) -> dict:
        """F05's zero-noise calibration, against *this* gate's ``U_target``.

        Returns the calibrated ``phi_vz``, the raw ``1 - Fbar``, and its split into
        a leakage-population piece and a coherent residual beyond calibration.
        """
        om_x, om_y = self.broadcast_of(arm, N)
        U3 = np.asarray(gate.three_level_propagator(om_x, om_y, self.T, DEV.delta))
        B = np.asarray(gate.qubit_block(U3))
        W, P = gate.polar_decompose(B)
        W, P = np.asarray(W), np.asarray(P)
        phi_vz, fid_raw = armcore.fit_phi_vz(B, self.U_TARGET)
        gate_res = np.asarray(gate.gate_residual_vector(
            jnp.asarray(W), jnp.asarray(self.U_TARGET), phi_vz))
        return {
            "phi_vz": phi_vz,
            "baseline_infidelity": 1.0 - fid_raw,
            "leakage_population": float(1.0 - 0.5 * np.real(np.trace(P @ P))),
            "gate_residual_norm_at_phi_vz": float(np.linalg.norm(gate_res)),
            "polar_margin": float(gate.polar_margin(B)),
        }

    def characterize(self, arm, N: int | None = None) -> dict:
        """Everything Table 1 needs for one arm, all at ``delta_z = epsilon = 0``.

        ``phi_vz`` is calibrated here, once, and frozen for the whole sweep: re-fitting
        it per noise point would hand every arm a free recalibration.
        """
        N = N or self.N_DESIGN
        om_x, om_y = self.broadcast_of(arm, N)
        row = dict(self.characterize_zero_noise(arm, N))
        row.update(arm.predicted_terms(N))   # C1..C4 (arm A's C4 trap handled by F05)
        row.update(self.geometry_of(arm, N))
        peak = float(np.max(np.hypot(om_x, om_y)))
        row["peak"] = peak
        row["peak_frac"] = peak / DEV.rabi_max_rate
        row["decoherence_floor"] = DEV.decoherence_floor
        row["predicted_plus_decoh"] = row["total"] + DEV.decoherence_floor
        row["true_plus_decoh"] = row["baseline_infidelity"] + DEV.decoherence_floor
        row["ratio_true_over_predicted"] = row["baseline_infidelity"] / row["total"]
        return row

    # -- noise sweep -------------------------------------------------------

    def _make_mc_fn(self, om_x, om_y, phi_vz: float):
        """Jitted, vmapped ``(delta_z batch, epsilon batch) -> 1 - Fbar batch``.

        F05's ``_make_mc_fn`` with the module-level ``U_TARGET`` replaced by this
        gate's.  The target is built from ``phi_vz`` once, so the calibration is
        frozen by construction.
        """
        om_x, om_y = jnp.asarray(om_x), jnp.asarray(om_y)
        target = jnp.asarray(armcore.rz_matrix(phi_vz) @ self.U_TARGET, dtype=jnp.complex128)

        def one(delta_z, epsilon):
            scale = 1.0 + epsilon
            U3 = gate.three_level_propagator(om_x * scale, om_y * scale, self.T,
                                             DEV.delta, delta_z=delta_z)
            return 1.0 - armcore.fidelity_jax(gate.qubit_block(U3), target)

        return jax.jit(jax.vmap(one))

    def sweep(self, arm, phi_vz: float, eps_grid=None, delta_mhz_grid=None,
              N: int | None = None) -> np.ndarray:
        """True ``1 - Fbar`` on the ``(delta, epsilon)`` grid at a *frozen* ``phi_vz``.

        Shape ``(len(delta_mhz_grid), len(eps_grid))``.
        """
        eps_grid = self.EPS_GRID if eps_grid is None else eps_grid
        delta_mhz_grid = self.DELTA_MHZ_GRID if delta_mhz_grid is None else delta_mhz_grid
        om_x, om_y = self.broadcast_of(arm, N or self.N_SIM)
        fn = self._make_mc_fn(om_x, om_y, phi_vz)
        DZ, EPS = np.meshgrid(delta_rate(delta_mhz_grid),
                              np.asarray(eps_grid, dtype=float), indexing="ij")
        vals = np.asarray(fn(jnp.asarray(DZ.ravel()), jnp.asarray(EPS.ravel())))
        return vals.reshape(DZ.shape)

    def zero_noise_consistency(self, arm, phi_vz: float, baseline: float,
                               N: int | None = None) -> dict:
        """What the ``N_DESIGN -> N_SIM`` grid change costs at the sweep's ``(0, 0)`` point."""
        om_x, om_y = self.broadcast_of(arm, N or self.N_SIM)
        fn = self._make_mc_fn(om_x, om_y, phi_vz)
        val = float(np.asarray(fn(jnp.asarray([0.0]), jnp.asarray([0.0])))[0])
        return {"sweep_at_zero": val, "characterize_baseline": baseline,
                "rel_diff": abs(val - baseline) / max(baseline, 1e-300)}

    def c3_prediction(self, arm, eps_grid=None, N: int | None = None) -> np.ndarray:
        """The budget's own prediction for the epsilon slice: ``C3`` rescaled to each epsilon.

        ``C3 = (2/3) <eps^2> |A_T|^2`` is exactly quadratic in epsilon, so the whole
        slice follows from one design-curve evaluation.
        """
        eps_grid = self.EPS_GRID if eps_grid is None else eps_grid
        A_T = np.asarray(propagate.tantrix_area(
            budget.design_chain(arm.a, arm.c, arm.Phi_0, self.T, N or self.N_DESIGN)))
        return (2.0 / 3.0) * np.asarray(eps_grid, dtype=float) ** 2 * float(np.sum(A_T**2))

    # -- optimization trajectory (Fig. 4) ----------------------------------

    def checkpoint_trace(self, arm_key: str = "C", N: int | None = None,
                         trace: bool = False) -> dict:
        """Per-checkpoint surrogate terms plus the *true* ``1 - Fbar`` recomputed offline.

        The true value is not in the RunRecord -- the optimizer never computes it.
        ``phi_vz`` is re-fitted at each iterate (the calibration a real experiment
        would perform on that pulse).  Checkpoints whose recorded ``c1`` endpoint
        residual exceeds the readout layer's entry ticket (1e-9) are dropped: a
        DRAG/Stark broadcast of such an iterate is not well defined.

        ``trace=True`` reads the arm's trajectory-only run instead of the run its arm
        comes from (:meth:`trace_run_dir`) -- for X(pi) arm E that is round 1, which
        carries the D -> E descent arc round 2's converged tail does not show.
        """
        run_dir = self.trace_run_dir(arm_key) if trace else self.run_dir(arm_key)
        if run_dir is None:
            kind = "trajectory-only run" if trace else "recorded run"
            raise FileNotFoundError(
                f"arm {arm_key} has no {kind} for {self.spec.key}")
        N = N or self.N_SIM
        hist = dict(np.load(run_dir / "history.npz"))
        ok = np.abs(hist["res_c1_ends"]).max(axis=1) < 1e-9
        if not ok.all():
            dropped = np.asarray(hist["iter"])[~ok]
            print(f"checkpoint_trace({run_dir.name}): dropped {len(dropped)} checkpoint(s) "
                  f"violating the c1 entry ticket (iters {dropped.tolist()})")
            hist = {k: v[ok] if getattr(v, "shape", ()) and len(v) == len(ok) else v
                    for k, v in hist.items()}
        n = len(hist["iter"])
        true_infid, leak = np.empty(n), np.empty(n)
        fluence, max_tau, peak = np.empty(n), np.empty(n), np.empty(n)
        for k in range(n):
            a, c, Phi_0 = hist["coeffs_a"][k], hist["coeffs_c"][k], float(hist["Phi_0"][k])
            om_x, om_y = parametrization.general_waveform(a, c, Phi_0, self.T, DEV.delta, N)
            om_x, om_y = np.asarray(om_x), np.asarray(om_y)
            if self.spec.is_rotated:
                om_x, om_y = rotate_drive(om_x, om_y, self.phi_axis)
            U3 = gate.three_level_propagator(jnp.asarray(om_x), jnp.asarray(om_y),
                                             self.T, DEV.delta)
            B = np.asarray(gate.qubit_block(U3))
            _phi, fid = armcore.fit_phi_vz(B, self.U_TARGET)
            true_infid[k] = 1.0 - fid
            _W, P = gate.polar_decompose(jnp.asarray(B))
            leak[k] = float(1.0 - 0.5 * np.real(np.trace(np.asarray(P) @ np.asarray(P))))
            dc = parametrization.design_curve(a, c, Phi_0, self.T, N)
            fluence[k] = float(np.sum(np.asarray(dc.kappa) ** 2) * dc.dt)
            max_tau[k] = float(np.max(np.abs(np.asarray(dc.tau))))
            peak[k] = float(np.max(np.hypot(om_x, om_y)))
        return {
            "iter": np.asarray(hist["iter"]),
            "surrogate_total": np.asarray(hist["cost_total"]),
            **{f"surrogate_{k}": np.asarray(hist[f"cost_{k}"]) for k in ("c1", "c2", "c3", "c4")},
            "true_infidelity": true_infid,
            "leakage_population": leak,
            "fluence_over_floor": fluence / self.FLUENCE_FLOOR,
            "max_abs_tau": max_tau,
            "peak_frac": peak / DEV.rabi_max_rate,
            "gate_residual_norm": np.linalg.norm(np.asarray(hist["res_gate"]), axis=1),
        }

    # -- covariance check (the Y folders' actual deliverable) --------------

    def covariance_check(self, arm_key: str = "B", N: int | None = None) -> dict:
        """Numerically verify that this rotated gate is the source gate's exact image.

        Compares the true ``1 - Fbar`` of the rotated arm against ``U_target`` with
        the source arm's against ``R_x(theta)``, on the full ``(delta, epsilon)``
        grid.  The frame-rotation argument (``gatespec`` module docstring) predicts
        agreement to round-off.

        ★ Both sweeps are run at the **source arm's** frozen ``phi_vz``, on purpose.
        The rotation maps the calibration problem onto itself exactly -- fidelity is
        invariant under simultaneous conjugation, so the two gates have the *same*
        optimal ``phi_vz`` -- but each side reaches it through its own bounded scalar
        minimization, which lands ~1e-7 apart.  Freezing one value isolates the
        rotation, which is what this check is for; the calibration discrepancy is
        reported separately as ``phi_vz_abs_diff`` rather than silently inflating the
        sweep residual (on arm D of Y(pi) it inflated it to 9e-10, three decades above
        everything else on the page).
        """
        if not self.spec.is_rotated:
            raise ValueError(f"{self.spec.key} is not a rotated gate")
        src = GateLab(source_spec(self.spec), self.N_DESIGN, self.N_SIM)
        here_arm, there_arm = self.build_arms()[arm_key], src.build_arms()[arm_key]
        here_row, there_row = self.characterize(here_arm, N), src.characterize(there_arm, N)
        phi_frozen = there_row["phi_vz"]
        s_here = self.sweep(here_arm, phi_frozen)
        s_there = src.sweep(there_arm, phi_frozen)
        denom = np.maximum(np.abs(s_there), 1e-300)
        base_abs = abs(here_row["baseline_infidelity"] - there_row["baseline_infidelity"])
        sweep_abs = np.abs(s_here - s_there)
        return {
            "arm": arm_key,
            "source_gate": src.spec.name,
            "baseline_here": here_row["baseline_infidelity"],
            "baseline_there": there_row["baseline_infidelity"],
            "phi_vz_abs_diff": abs(here_row["phi_vz"] - there_row["phi_vz"]),
            "baseline_abs_diff": float(base_abs),
            "baseline_rel_diff": base_abs / max(there_row["baseline_infidelity"], 1e-300),
            "sweep_max_abs_diff": float(np.max(sweep_abs)),
            "sweep_max_rel_diff": float(np.max(sweep_abs / denom)),
            "cost_abs_diffs": {c: abs(here_row[c] - there_row[c]) for c in ("c1", "c2", "c3", "c4")},
            "cost_rel_diffs": {c: abs(here_row[c] - there_row[c])
                               / max(abs(there_row[c]), 1e-300)
                               for c in ("c1", "c2", "c3", "c4")},
        }

    # -- on-disk contract --------------------------------------------------

    def export_arms(self, arms: dict, rows: dict, outdir: Path | None = None) -> dict:
        """Write the 120-point CSVs, the coefficient JSONs and ``manifest.json``."""
        import subprocess

        from curve_opt import recorder  # local: the only module allowed to touch disk

        outdir = self.folder if outdir is None else Path(outdir)
        wf_dir = outdir / "waveforms"
        wf_dir.mkdir(parents=True, exist_ok=True)
        t_mid, _ = geometry.midpoint_grid(self.T, self.N_EXPORT)
        try:
            commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            commit = "unknown"

        manifest = {
            "git_commit": commit,
            "N_export": self.N_EXPORT,
            "T_ns": self.T,
            "gate": {"key": self.spec.key, "name": self.spec.name,
                     "theta": self.theta, "phi_axis": self.phi_axis,
                     "solve_mode": self.spec.solve_mode,
                     "source_gate": self.spec.source_key,
                     "fluence_floor": self.FLUENCE_FLOOR},
            "device": {"anharmonicity": DEV.anharmonicity, "rabi_max": DEV.rabi_max,
                       "gate_time": DEV.gate_time, "n_modes": DEV.n_modes,
                       "static_detuning": DEV.static_detuning,
                       "control_error": DEV.control_error},
            "arms": {},
        }
        for key in self.ARM_ORDER:
            arm, row, slug = arms[key], rows[key], SLUGS[key]
            om_x, om_y = self.broadcast_of(arm, self.N_EXPORT)
            csv_path = wf_dir / f"{slug}_{self.N_EXPORT}pt.csv"
            recorder.export_waveform_csv(csv_path, t_mid, om_x, om_y)
            # 120-point sanity reading of the exported waveform, same propagator
            # as characterize(); an external cross-integrator is not part of this
            # tree (the golden baseline plays that role instead).
            U3_120 = gate.three_level_propagator(jnp.asarray(om_x), jnp.asarray(om_y),
                                                 self.T, DEV.delta)
            _phi_120, fid_120 = armcore.fit_phi_vz(
                np.asarray(gate.qubit_block(U3_120)), self.U_TARGET)
            infid_120 = 1.0 - fid_120
            json_path = wf_dir / f"{slug}.json"
            json_path.write_text(json.dumps({
                "arm": key, "name": arm.name, "layer": arm.layer, "note": arm.note,
                "gate": self.spec.key,
                "coeffs_a": np.asarray(arm.a).tolist(),
                "coeffs_c": np.asarray(arm.c).tolist(),
                "Phi_0": float(arm.Phi_0), "phi_axis": self.phi_axis,
                "phi_vz_frozen": float(row["phi_vz"]),
                "theta": self.theta, "M": DEV.n_modes, "T_ns": self.T,
                "run_id": getattr(arm, "run_id", None),
            }, indent=1))
            manifest["arms"][key] = {
                "name": arm.name, "layer": arm.layer,
                "csv": str(csv_path.relative_to(outdir)),
                "json": str(json_path.relative_to(outdir)),
                "run_id": getattr(arm, "run_id", None),
                "infidelity_120pt": float(infid_120),
                "infidelity_fine": float(row["baseline_infidelity"]),
                "sampling_loss_ratio": float(infid_120 / row["baseline_infidelity"]),
            }
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        return manifest
