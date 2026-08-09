"""Target-gate specification shared by the four single-qubit proposal folders.

One :class:`GateSpec` fixes everything that distinguishes ``X(pi)`` from
``X(pi/2)``, ``Y(pi)`` and ``Y(pi/2)``:

    theta     the rotation angle -- the *only* physically new degree of freedom
    phi_axis  the azimuth of the rotation axis in the equatorial plane

Why the axis is not a new physics problem
-----------------------------------------
The three-level model is

    H(t) = Delta |2><2| + (1/2) [Omega_x(t) X_3 + Omega_y(t) Y_3]  (+ noise)

with the ladder operators ``X_3, Y_3``.  Let ``N = diag(0, 1, 2)`` be the number
operator and ``V(phi) = exp(i phi N)``.  Then

    V H(Omega_x, Omega_y) V^dag = H(Omega_x cos phi - Omega_y sin phi,
                                    Omega_x sin phi + Omega_y cos phi)

because ``V X_3 V^dag`` mixes the quadratures exactly as a planar rotation, and
``|2><2|`` commutes with ``N``.  Both noise channels survive the conjugation
untouched: the quasi-static ``delta_z`` couples through a diagonal operator
proportional to ``N`` (commutes with ``V``), and the amplitude error is a scalar
multiplying ``Omega``.  Hence, for a *fixed* gate time and device,

    * the optimal waveform for ``R_{n(phi)}(theta)`` is the optimal waveform for
      ``R_x(theta)`` with the drive phase advanced by ``phi``;
    * every design-curve cost ``C_1..C_4``, the true three-level ``1 - Fbar``,
      the leakage population and the whole ``(delta_z, epsilon)`` sweep are
      **numerically identical**, point for point.

So the Y folders exist to *present* the Y waveforms and to *check* that
covariance numerically -- not to re-discover the same optimum twice.  They run
with ``SOLVE_MODE = "rotate_from_x"``.  If a future model breaks the symmetry
(IQ imbalance, quadrature-dependent crosstalk, a non-diagonal noise channel),
flip that switch to ``"independent"``; that path additionally needs
``propagate.target_axis(theta, phi)`` in ``src/`` -- see the folder READMEs.

What *does* change with theta
-----------------------------
``fluence_floor = theta**2 / T`` -- the square-pulse lower bound on
``int_0^T kappa^2 dt`` for a rotation by ``theta``.  The F06b fluence cap is
quoted as a multiple of this floor, so the multiple keeps its meaning ("how far
above the cheapest possible pulse are we allowed to go", i.e. how far the
achieved ``eta`` may exceed the minimum) across gates only if the floor itself
scales.  Hard-coding ``pi**2 / T`` would make ``cap = 2 F0`` mean ``8 F0`` for a
pi/2 gate and switch the validity guard off without saying so.

Everything else is a device constant and is *not* rescaled: the peak bound
``Omega_max``, the R guard ``max|tau| <= 0.5 |Delta| / 2``, the gate time
``T = 50 ns``, and the budget weights ``w_1..w_4`` (which are conditional on the
assumed noise moments -- rescaling them would be changing the noise model, not
the gate).
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROPOSALS = HERE.parent
REPO_ROOT = PROPOSALS.parent
for _p in (REPO_ROOT / "src",):   # F08c: no more _dev_logs on the proposal path
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from curve_opt import device, propagate  # noqa: E402

DEV = device.DEFAULT_DEVICE
T = DEV.gate_time


# ---------------------------------------------------------------------------
# the spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateSpec:
    """One target gate and the folder that owns it."""

    key: str
    """Short identifier, also the proposal-folder suffix (``Xpi``, ``Xpi2``, ...)."""

    name: str
    """Human-readable gate name, e.g. ``"X(pi/2)"``."""

    latex: str
    r"""Display form for notebook titles, e.g. ``r"X(\pi/2)"``."""

    theta: float
    """Rotation angle in rad -- the winding branch the gate constraint is snapped to."""

    phi_axis: float
    """Azimuth of the rotation axis in the equatorial plane (0 = x, pi/2 = y)."""

    folder: str
    """Proposal directory **name** -- not a path.

    Resolved by searching under ``proposals/`` (:func:`gate_dir`), so regrouping the
    gate folders into subdirectories does not break anything. They were moved once
    already (2026-08-08, into ``1qb_Gate-optimization/``), which broke every
    fixed-depth ``parents[n]`` in the package; searching by name is what stops the
    next move from doing the same.
    """

    solve_mode: str = "independent"
    """``"independent"``: run ``solve_budget`` in this folder.
    ``"rotate_from_x"``: take the solved arms from :attr:`source_key`'s folder and
    advance the drive phase by :attr:`phi_axis` (exact -- module docstring)."""

    source_key: str | None = None
    """Which gate the arms are rotated from, when ``solve_mode == "rotate_from_x"``."""

    run_ids: dict = field(default_factory=dict)
    """``{"C": run_id | None, "D": ..., "E": ...}`` -- the recorded ``_runs/`` solves
    backing the optimized arms.  ``None`` (or a missing key) means "not solved yet";
    :class:`gatelib.GateLab` then simply omits that arm, so the notebook runs
    end-to-end with A and B alone before any optimizer time is spent."""

    trace_run_ids: dict = field(default_factory=dict)
    """Runs kept only for their *optimization trajectory*, not for an arm.

    X(pi)'s ``{"E": "20260810_F06e_adexact_construction"}`` is arm E's **round 1**:
    round 2 is the converged solve and supplies the arm, but round 1 carries the
    D -> E descent arc the Figure-4 panel plots.  Before F08a this run had nowhere
    to live in the registry and survived only as ``ARM_E_RUN_R1`` in the X(pi)
    folder's own ``armlib`` -- one of the duplicated entries this step removes.

    ⚠️ Unlike :attr:`run_ids`, this is **not** inherited through :attr:`source_key`.
    A trajectory belongs to the notebook of the gate that actually ran the solve;
    the rotated gates present arms, not convergence histories.
    """

    # -- derived -----------------------------------------------------------

    @property
    def axis(self) -> np.ndarray:
        return np.array([np.cos(self.phi_axis), np.sin(self.phi_axis), 0.0])

    @property
    def U_target(self) -> np.ndarray:
        """``exp(-i theta (n . sigma) / 2)`` with ``n`` the equatorial axis.

        Equal to ``rz(phi_axis) @ target_x(theta) @ rz(-phi_axis)`` -- the same
        conjugation the waveform rotation implements.
        """
        return np.asarray(propagate.su2_rotation(self.axis, self.theta))

    @property
    def U_target_x(self) -> np.ndarray:
        """The unrotated ``R_x(theta)`` the solver actually constrains against."""
        return np.asarray(propagate.target_x(self.theta))

    def fluence_floor(self, dev: device.Device | None = None) -> float:
        """Square-pulse lower bound on ``int kappa^2 dt`` for this angle (module docstring).

        *dev* defaults to the module-level :data:`DEV` (``None``), so every existing
        caller that wants the default device keeps getting the same number; pass a
        non-default :class:`curve_opt.device.Device` to get the floor for that gate
        time instead (N1: a non-default ``Device`` was previously unreachable from
        this layer).
        """
        d = DEV if dev is None else dev
        return self.theta**2 / d.gate_time

    @property
    def is_rotated(self) -> bool:
        return self.solve_mode == "rotate_from_x"

    def target_gate_manifest(self) -> dict:
        """The ``BudgetProblem.target_gate`` provenance blob."""
        return {
            "name": self.name,
            "theta": float(self.theta),
            "phi_axis": float(self.phi_axis),
            "matrix_real": np.real(self.U_target).tolist(),
            "matrix_imag": np.imag(self.U_target).tolist(),
        }


# ---------------------------------------------------------------------------
# frame rotation (the Y arms)
# ---------------------------------------------------------------------------


def rotate_drive(om_x, om_y, phi: float):
    """Advance the drive phase by *phi*: ``(Omega_x + i Omega_y) -> e^{i phi} (...)``.

    Exact frame change, not an approximation -- see the module docstring.
    """
    c, s = np.cos(phi), np.sin(phi)
    om_x, om_y = np.asarray(om_x), np.asarray(om_y)
    return c * om_x - s * om_y, s * om_x + c * om_y


# ---------------------------------------------------------------------------
# the four gates
# ---------------------------------------------------------------------------

SPECS: dict[str, GateSpec] = {
    "Xpi": GateSpec(
        key="Xpi",
        name="X(pi)",
        latex=r"X(\pi)",
        theta=np.pi,
        phi_axis=0.0,
        folder="1qb_Gaussian-drag-optimization_Xpi",
        solve_mode="independent",
        # run_ids / trace_run_ids come from configs/runs.json -- see REGISTRY_PATH.
    ),
    "Xpi2": GateSpec(
        key="Xpi2",
        name="X(pi/2)",
        latex=r"X(\pi/2)",
        theta=np.pi / 2,
        phi_axis=0.0,
        folder="1qb_Gaussian-drag-optimization_Xpi2",
        solve_mode="independent",
        # filled in by run_construction.py as each solve lands -- configs/runs.json
    ),
    "Ypi": GateSpec(
        key="Ypi",
        name="Y(pi)",
        latex=r"Y(\pi)",
        theta=np.pi,
        phi_axis=np.pi / 2,
        folder="1qb_Gaussian-drag-optimization_Ypi",
        solve_mode="rotate_from_x",
        source_key="Xpi",
    ),
    "Ypi2": GateSpec(
        key="Ypi2",
        name="Y(pi/2)",
        latex=r"Y(\pi/2)",
        theta=np.pi / 2,
        phi_axis=np.pi / 2,
        folder="1qb_Gaussian-drag-optimization_Ypi2",
        solve_mode="rotate_from_x",
        source_key="Xpi2",
    ),
    "Xpi_novera": GateSpec(
        key="Xpi_novera",
        name="X(pi) [Novera]",
        latex=r"X(\pi)_{\rm Novera}",
        theta=np.pi,
        phi_axis=0.0,
        folder="Novera_Xpi",
        solve_mode="independent",
        # run_ids come from configs/runs.json, same as every other gate.
    ),
    "Ypi_novera": GateSpec(
        key="Ypi_novera",
        name="Y(pi) [Novera]",
        latex=r"Y(\pi)_{\rm Novera}",
        theta=np.pi,
        phi_axis=np.pi / 2,
        folder="Novera_Ypi",
        solve_mode="rotate_from_x",
        source_key="Xpi_novera",
    ),
}


# ---------------------------------------------------------------------------
# locating a gate's folder
# ---------------------------------------------------------------------------

_DIR_CACHE: dict[str, Path] = {}


def gate_dir(sp: "GateSpec | str") -> Path:
    """The proposal directory for a gate, found by name anywhere under ``proposals/``.

    Deliberately a search rather than ``PROPOSALS / name``: the four gate folders
    have already been regrouped once, and every fixed-depth path in this package
    broke at that moment. A name search costs one cached directory walk and
    survives arbitrary regrouping.

    Raises if the name is missing or ambiguous -- both are real problems that
    should surface here rather than as a mysterious empty export directory.
    """
    name = sp if isinstance(sp, str) else sp.folder
    if name in _DIR_CACHE:
        return _DIR_CACHE[name]
    hits = [p for p in PROPOSALS.rglob(name)
            if p.is_dir() and "__pycache__" not in p.parts]
    if not hits:
        raise FileNotFoundError(
            f"no directory named {name!r} under {PROPOSALS}. If the folder was "
            f"renamed, update the matching GateSpec.folder in gatespec.py.")
    if len(hits) > 1:
        raise ValueError(
            f"{len(hits)} directories named {name!r} under {PROPOSALS}: "
            f"{[str(h) for h in hits]}. Gate folder names must be unique.")
    _DIR_CACHE[name] = hits[0]
    return hits[0]


# ---------------------------------------------------------------------------
# the run registry (configs/runs.json)
# ---------------------------------------------------------------------------

REGISTRY_PATH = REPO_ROOT / "configs" / "runs.json"
"""Where ``run_ids`` / ``trace_run_ids`` actually live.

Which recorded run backs which arm is **data**: it changes every time a solve
lands, and before F08b the way to update it was to hand-edit the ``SPECS``
literal above (``run_construction.py``'s docstring literally said "paste its
run_id into gatespec.py").  It is the one part of this configuration layer that
belongs in a file a script can write, so the specs above declare no run ids at
all and this file is the single source.

Everything with a *derivation* behind it stays in Python -- the device constants
(:mod:`curve_opt.device`) and what each arm is (:mod:`armspec`) -- because JSON
has nowhere to put the argument for why a number is what it is.
"""


def load_registry(path: Path | None = None) -> dict:
    """Read the run registry; ``{}`` (with a warning) if the file is absent."""
    path = REGISTRY_PATH if path is None else Path(path)
    if not path.exists():
        warnings.warn(
            f"run registry {path} not found: every gate falls back to baselines "
            f"A and B only. This is the intended behaviour for a fresh checkout, "
            f"but if you expected optimized arms, that file is why they are missing.",
            stacklevel=2)
        return {}
    raw = json.loads(path.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def apply_registry(specs: dict, registry: dict) -> dict:
    """Return *specs* with ``run_ids`` / ``trace_run_ids`` filled in from *registry*.

    An entry naming a gate or a field that does not exist is an error, not a
    silent no-op: a typo'd key that quietly registers nothing would present as
    "the solve did not land", which is exactly the debugging dead end this
    registry exists to remove.
    """
    known_fields = {"run_ids", "trace_run_ids"}
    out = dict(specs)
    for gate_key, entry in registry.items():
        if gate_key not in out:
            raise ValueError(
                f"run registry names unknown gate {gate_key!r}; "
                f"known gates are {sorted(out)}")
        unknown = set(entry) - known_fields
        if unknown:
            raise ValueError(
                f"run registry entry {gate_key!r} has unknown field(s) "
                f"{sorted(unknown)}; expected a subset of {sorted(known_fields)}")
        out[gate_key] = replace(
            out[gate_key],
            run_ids=dict(entry.get("run_ids", {})),
            trace_run_ids=dict(entry.get("trace_run_ids", {})),
        )
    return out


SPECS = apply_registry(SPECS, load_registry())


def register_run(gate_key: str, arm_key: str, run_id: str, *,
                 trace: bool = False, path: Path | None = None) -> Path:
    """Record ``gate_key/arm_key -> run_id`` in the registry, on disk and in memory.

    Called by ``run_construction.py`` when a solve finishes, so a landed run is
    picked up by every notebook without anyone editing Python.  Preserves the
    file's ``_README`` block and key order.
    """
    path = REGISTRY_PATH if path is None else Path(path)
    if gate_key not in SPECS:
        raise ValueError(f"unknown gate {gate_key!r}; known gates are {sorted(SPECS)}")
    raw = json.loads(path.read_text()) if path.exists() else {}
    entry = raw.setdefault(gate_key, {})
    entry.setdefault("run_ids" if not trace else "trace_run_ids", {})
    entry["trace_run_ids" if trace else "run_ids"][arm_key] = run_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2) + "\n")
    SPECS.update(apply_registry(SPECS, {gate_key: entry}))
    return path


# ---------------------------------------------------------------------------


def spec(key: str) -> GateSpec:
    return SPECS[key]


def source_spec(sp: GateSpec) -> GateSpec:
    """The X-axis gate a rotated spec inherits its arms from."""
    if not sp.is_rotated:
        raise ValueError(f"{sp.key} solves independently; it has no source gate")
    return SPECS[sp.source_key]
