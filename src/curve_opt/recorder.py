"""RunRecord I/O -- the only module in the package that touches the filesystem.

Every optimization run goes through here (``_plan.md`` §5). Traceability is a
hard requirement, and the mechanical guarantee is the rule in §4.3 item 1:
:mod:`curve_opt.plotting` may only read a RunRecord, never re-run an
optimization. If a figure can be drawn from the stored data alone, the schema is
adequate; if it cannot, the schema is broken.

The guarantee is enforced by construction, not by good intentions:
:mod:`curve_opt.plotting` imports neither this module nor any path/filesystem
API, and its functions take a :class:`RunRecord` *object*. There is no code path
by which a figure can reach the disk, so a figure that exists is a figure whose
data was loaded through :func:`load`.

Layout -- ``<RUNS_DIR>/<run_id>/`` (git-ignored local asset)
-----------------------------------------------------------
``run_id = <date>_<gate>_<ansatz>_<label>``, e.g.
``20260801_Xpi_rcp_lem_M12_lam0``.

* ``manifest.json`` -- the run's self-description; without it the data cannot be
  interpreted, so :class:`Recorder` refuses to create a run whose manifest is
  missing any key of :data:`REQUIRED_MANIFEST_KEYS`. run_id, timestamp, git
  commit, branch; ansatz (name, source, original parameters, representability
  tier, projection protocol); M, T, target gate (name + matrix), N_grid;
  ``winding_branch`` (the **exact** branch value ``theta_target + 2 pi k`` -- the
  gate name alone does not define the problem, since theta and theta + 2 pi are
  the same gate but have lower bounds differing by theta^2; the ansatz's own
  *measured* turning is a separate field, ``theta_before``, because the constraint
  right-hand side must not carry the error the run is meant to remove);
  objective (terms, lambda or
  epsilon-constraint value, epigraph flag); solver (method, maxiter, GN Hessian
  on/off, tol, checkpoint_every); ``stop_reason`` in
  :data:`STOP_REASONS` -- a ``maxiter`` run is void by default.
* ``step0.npz`` -- the Fourier-projection error: raw Omega samples, projected
  coefficients, relRMS, and gate_err / closure / area before *and* after.
* ``history.npz`` -- one row per ``checkpoint_every`` iterations: ``iter``,
  ``coeffs_a[K, M]``, ``coeffs_b[K, M]`` (these alone reconstruct everything),
  ``cost_total``, the individual ``cost_energy`` / ``cost_curv`` / ``cost_peak``
  stored redundantly, and the residual trajectories ``res_gate`` /
  ``res_closure`` / ``res_area``.
* ``final.json`` -- status, nit, wall clock, per-term endpoint values, fine-grid
  re-evaluation if performed.

The redundant cost columns are a *check*, not a convenience:
:func:`verify_history` recomputes them from the stored coefficients and requires
**bit-for-bit** equality (``_plan.md`` §5.2 asks for 1e-12; the terms are
deterministic functions of the coefficients and the recorded grid, so anything
above exact zero means the record and its metadata disagree -- a wrong N_grid, a
transposed array, a truncated dtype).

Synthetic records
-----------------
``manifest["synthetic"] = True`` marks a fabricated run (schema validation,
plotting development). It is carried into every figure as a visible watermark:
fake data must never be mistakable for a real optimization.

Schema versions (F05-pre)
--------------------------
``manifest["schema_version"]`` says which of the two shapes above a record
uses:

* ``1`` (``schema="energy"``, the default) -- everything above: ``coeffs_a``/
  ``coeffs_b``, ``cost_energy``/``cost_curv``/``cost_peak``, ``res_closure``/
  ``res_area``. Every one of the 83 records written before F05-pre is version
  1 and is read by the exact code path that always existed -- nothing about
  it changed.
* ``2`` (``schema="budget"``) -- the full-cost era's :func:`curve_opt.optimize.solve_budget`
  run: ``coeffs_a``/``coeffs_c`` (F04's design-curve series, not ``a``/``b``),
  ``cost_c1``..``cost_c4``, ``Phi_0``/``phi_vz``, the 3-component polar-decomposition
  gate residual ``res_gate``, the 2-component endpoint residual ``res_c1_ends``,
  and the epigraph slack ``epigraph_s``. Written through :meth:`Recorder.append_budget`
  rather than :meth:`Recorder.append` (the two histories do not share a row
  shape, so one method silently accepting both would risk mixing them);
  verified through the same :func:`verify_history` entry point, dispatched on
  ``schema_version``.

A :class:`Recorder` is one or the other for its whole life (``schema=`` at
construction); :func:`load` never needs to be told which -- the manifest
already carries it.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

from curve_opt import budget, metrics
from curve_opt.device import Device

__all__ = [
    "HISTORY_COLUMNS",
    "HISTORY_COLUMNS_BUDGET",
    "load_device",
    "REPO_ROOT",
    "REQUIRED_MANIFEST_KEYS",
    "REQUIRED_MANIFEST_KEYS_BUDGET",
    "RUNS_DIR",
    "RUNS_DIRNAME",
    "SCHEMA_VERSIONS",
    "Recorder",
    "RunRecord",
    "STOP_REASONS",
    "delete_run",
    "git_context",
    "list_runs",
    "load",
    "make_run_id",
    "verify_history",
]

#: Root of the raw run data, relative to the repository root. Git-ignored on
#: purpose: it is a local asset, self-describing through each manifest.
#: Presentation copies live in ``results/``, selected by the user and always
#: re-plotted from here -- numbers are never transcribed by hand.
RUNS_DIRNAME = "_runs"

#: Repository root = three levels up from this file (src/curve_opt/recorder.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Absolute path of the run store. Nothing outside this module may build it.
RUNS_DIR = REPO_ROOT / RUNS_DIRNAME

#: A run that hit ``maxiter`` is void for quantitative claims (``_plan.md`` §6.4);
#: ``early_stop`` is legitimate for survey runs but must be annotated on figures.
STOP_REASONS = ("converged", "early_stop", "maxiter", "failed", "synthetic")

#: Without these the record cannot be interpreted, so writing one is refused.
REQUIRED_MANIFEST_KEYS = (
    "ansatz",
    "M",
    "T",
    "target_gate",
    "N_grid",
    "winding_branch",
    "objective",
    "solver",
)

#: F05-pre (schema_version 2): a budget-mode run additionally needs the full
#: device (weights and the noise moments they came from are not
#: reconstructible without it), which physical readout layer the run used,
#: which gate route measured G, and the pinned upstream commit the three-level
#: propagator's physics is cross-checked against -- none of these have an
#: energy-era counterpart, so they are additional keys, not replacements.
REQUIRED_MANIFEST_KEYS_BUDGET = REQUIRED_MANIFEST_KEYS + (
    "device",
    "gate_level",
    "layer",
    "novera_git_hash",
)

#: Columns of ``history.npz``, energy schema (``schema_version`` 1). ``coeffs_*``
#: are the primary data; the cost and residual columns are redundant and
#: cross-checked by :func:`verify_history`.
HISTORY_COLUMNS = (
    "iter",
    "coeffs_a",
    "coeffs_b",
    "cost_total",
    "cost_energy",
    "cost_curv",
    "cost_peak",
    "res_gate",
    "res_closure",
    "res_area",
)

#: Columns of ``history.npz``, budget schema (``schema_version`` 2, F05-pre).
#: ``coeffs_a``/``coeffs_c`` are :mod:`curve_opt.parametrization`'s design-curve
#: series (kappa/tau), not the ``a``/``b`` waveform coefficients of the energy
#: era. ``res_gate`` is the 3-component polar-decomposition residual
#: (:func:`curve_opt.gate.polar_gate_residual`); ``res_c1_ends`` is the
#: 2-component free-linear-row endpoint residual (start, end) that enforces
#: the DRAG ``c1`` guard on ``a`` (module docstring of
#: :mod:`curve_opt.optimize`, F04 section). ``epigraph_s`` is NaN when the
#: peak epigraph slack was not part of that checkpoint's solver vector.
HISTORY_COLUMNS_BUDGET = (
    "iter",
    "coeffs_a",
    "coeffs_c",
    "cost_total",
    "cost_c1",
    "cost_c2",
    "cost_c3",
    "cost_c4",
    "Phi_0",
    "phi_vz",
    "res_gate",
    "res_c1_ends",
    "epigraph_s",
)

#: ``schema=`` string -> ``manifest["schema_version"]`` integer. The version
#: on disk is what :func:`load`/:func:`verify_history` dispatch on; the string
#: is only the human-facing spelling at construction time.
SCHEMA_VERSIONS = {"energy": 1, "budget": 2}

#: The cost columns redundantly stored at each checkpoint, by schema version --
#: what :meth:`Recorder.close` puts in ``final["final_terms"]`` and what
#: :func:`verify_history` recomputes.
_COST_COLUMNS_BY_VERSION = {
    1: ("cost_total", "cost_energy", "cost_curv", "cost_peak"),
    2: ("cost_total", "cost_c1", "cost_c2", "cost_c3", "cost_c4"),
}

_HISTORY_COLUMNS_BY_VERSION = {1: HISTORY_COLUMNS, 2: HISTORY_COLUMNS_BUDGET}


class RunRecord(NamedTuple):
    """One run, as read back from disk. Plain dicts so the npz round-trip is 1:1."""

    run_id: str
    manifest: dict
    history: dict
    step0: dict
    final: dict

    @property
    def is_synthetic(self) -> bool:
        return bool(self.manifest.get("synthetic", False))

    @property
    def n_checkpoints(self) -> int:
        return int(np.asarray(self.history["iter"]).size) if self.history else 0


def git_context() -> dict:
    """Current commit and branch, or ``"unknown"`` -- never raises.

    Recorded automatically so that no run can be written without provenance:
    asking the caller to remember it is how manifests end up incomplete.
    """
    out = {}
    for key, args in (
        ("git_commit", ["rev-parse", "HEAD"]),
        ("git_branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
    ):
        try:
            out[key] = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - provenance must never break a run
            out[key] = "unknown"
    return out


def make_run_id(gate: str, ansatz: str, label: str, date: str | None = None) -> str:
    """``<date>_<gate>_<ansatz>_<label>`` with ``date`` defaulting to today (UTC)."""
    stamp = date or datetime.now(timezone.utc).strftime("%Y%m%d")
    parts = [stamp, gate, ansatz, label]
    if any(("/" in p or "\\" in p or not p) for p in parts):
        raise ValueError(f"run_id components must be non-empty and path-free, got {parts}")
    return "_".join(parts)


def _jsonable(value):
    """Make numpy/jax array types JSON-serializable without changing any value.

    F05-pre: :class:`curve_opt.budget.BudgetTerms` (passed through
    :meth:`Recorder.close`'s ``**extra`` by :func:`curve_opt.optimize.solve_budget`)
    carries raw ``jax.Array`` fields (``r_T``/``A_r``/``A_T``/``Lambda``), which
    the ``np.ndarray``/``np.generic`` cases above never matched -- a real
    budget-mode run could not close before this. The duck-typed ``.tolist()``
    branch below covers ``jax.Array`` (and anything else array-like) without
    this filesystem-only module taking on a ``jax`` import of its own.
    """
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return _jsonable(value.tolist())
    return value


class Recorder:
    """Writes one RunRecord. The only writer of ``_runs/``.

    Usage (energy schema, ``schema_version`` 1 -- the 83 pre-F05-pre records
    and every default call site)::

        rec = Recorder(run_id, manifest)
        rec.write_step0(step0_report)
        rec.append(iter=0, coeffs_a=a0, ...)      # from the solver callback
        rec.close(status="converged", nit=477, wall_clock_s=31.5)

    Usage (budget schema, ``schema_version`` 2, F05-pre --
    :func:`curve_opt.optimize.solve_budget`)::

        rec = Recorder(run_id, manifest, schema="budget")
        rec.append_budget(iter=0, coeffs_a=a0, coeffs_c=c0, ...)
        rec.close(status="converged", nit=..., wall_clock_s=...)

    *schema* is fixed for the Recorder's whole life and decides both which of
    :meth:`append`/:meth:`append_budget` is usable and which
    ``REQUIRED_MANIFEST_KEYS*`` set the manifest is checked against; ``append``
    buffers in memory and flushes ``history.npz`` every ``flush_every``
    checkpoints, so a crashed run still leaves usable data.
    """

    def __init__(
        self,
        run_id: str,
        manifest: dict,
        root: Path | str = RUNS_DIR,
        *,
        flush_every: int = 50,
        exist_ok: bool = False,
        schema: str = "energy",
    ):
        if schema not in SCHEMA_VERSIONS:
            raise ValueError(f"schema must be one of {sorted(SCHEMA_VERSIONS)}, got {schema!r}")
        required = REQUIRED_MANIFEST_KEYS_BUDGET if schema == "budget" else REQUIRED_MANIFEST_KEYS
        missing = [k for k in required if k not in manifest]
        if missing:
            raise ValueError(
                f"manifest is missing {missing}; without them the record cannot be "
                "interpreted (_plan.md §5.1)"
            )
        self.run_id = run_id
        self.path = Path(root) / run_id
        if self.path.exists() and not exist_ok:
            raise FileExistsError(f"{self.path} already exists; pass exist_ok=True to overwrite")
        self.path.mkdir(parents=True, exist_ok=True)
        self.flush_every = int(flush_every)
        self.schema = schema
        self._schema_version = SCHEMA_VERSIONS[schema]
        self._history_columns = _HISTORY_COLUMNS_BY_VERSION[self._schema_version]
        self._cost_columns = _COST_COLUMNS_BY_VERSION[self._schema_version]

        self.manifest = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **git_context(),
            "synthetic": bool(manifest.get("synthetic", False)),
            "stop_reason": None,
            **manifest,
            # placed last, after **manifest: the schema this Recorder instance
            # actually writes with governs what append/close/flush do, so the
            # caller's manifest dict (which may carry no opinion, or the wrong
            # one) may never override it.
            "schema_version": self._schema_version,
        }
        self._rows: list[dict] = []
        self._closed = False
        self._write_manifest()

    # -- writing ----------------------------------------------------------

    def _write_manifest(self):
        (self.path / "manifest.json").write_text(
            json.dumps(_jsonable(self.manifest), indent=2, sort_keys=True) + "\n"
        )

    def write_step0(self, report) -> None:
        """Serialize a :class:`curve_opt.ansatz.Step0Report` to ``step0.npz``.

        Accepts any object exposing the report's fields; ``None`` entries (the
        after-projection robustness values of a non-planar ansatz, pending the
        Step 11 propagator) are stored as NaN so the schema stays rectangular,
        and the ``*_available`` flags say which is which.
        """
        fields = {
            "coeffs_a": report.coeffs_a,
            "coeffs_b": report.coeffs_b,
            "times": report.times,
            "omega_x_before": report.omega_x_before,
            "omega_x_after": report.omega_x_after,
            "omega_y_before": report.omega_y_before,
            "omega_y_after": report.omega_y_after,
            "rel_rms": report.rel_rms,
            "rel_rms_x": report.rel_rms_x,
            "endpoint_ratio": report.endpoint_ratio,
            "theta_before": report.theta_before,
            "theta_after": report.theta_after,
            "closure_before": report.closure_before,
            "area_before": report.area_before,
            "closure_after": report.closure_after,
            "area_after": report.area_after,
            "winding_branch": report.winding_branch,
            "is_planar": report.is_planar,
        }
        payload = {}
        for key, value in fields.items():
            if value is None:
                payload[key] = np.array(np.nan)
                payload[f"{key}_available"] = np.array(False)
            else:
                payload[key] = np.asarray(value)
                if key.endswith("_after") and key.startswith(("closure", "area")):
                    payload[f"{key}_available"] = np.array(True)
        np.savez(self.path / "step0.npz", **payload)

        self.manifest.setdefault("ansatz", {})
        if isinstance(self.manifest["ansatz"], dict):
            self.manifest["ansatz"].update(
                {
                    "projection_protocol": report.protocol,
                    "tier_declared": report.tier_declared,
                    "tier_measured": report.tier_measured,
                    "tier_mismatch": bool(report.tier_mismatch),
                    "step0_rel_rms": float(report.rel_rms),
                }
            )
        # ★ The measured total turning goes to ``theta_before``; ``winding_branch``
        # is NOT touched. Under the v4.2 semantics of §5.1 the branch value is the
        # *exact* theta_target + 2 pi k that the gate constraint uses, while the
        # quadrature estimate of the ansatz's own turning is a separate field.
        # Overwriting winding_branch here (as this method did before Step 08) would
        # record the ansatz's gate error as the target -- for rcp_lemniscate that is
        # 2.2e-4, precisely the error the optimization exists to remove.
        self.manifest["theta_before"] = float(report.winding_branch)
        self._write_manifest()

    def append(
        self,
        iter: int,
        coeffs_a,
        coeffs_b=None,
        *,
        cost_total: float,
        cost_energy: float,
        cost_curv: float,
        cost_peak: float,
        res_gate: float,
        res_closure,
        res_area,
    ) -> None:
        """Record one energy-schema checkpoint. Called from the solver callback, never by hand."""
        if self._closed:
            raise RuntimeError(f"recorder for {self.run_id} is already closed")
        if self.schema != "energy":
            raise RuntimeError(
                f"recorder for {self.run_id} uses schema={self.schema!r}; append() is the "
                "energy-schema writer -- use append_budget() instead"
            )
        a = np.asarray(coeffs_a, dtype=float)
        b = np.zeros_like(a) if coeffs_b is None else np.asarray(coeffs_b, dtype=float)
        self._rows.append(
            {
                "iter": int(iter),
                "coeffs_a": a,
                "coeffs_b": b,
                "cost_total": float(cost_total),
                "cost_energy": float(cost_energy),
                "cost_curv": float(cost_curv),
                "cost_peak": float(cost_peak),
                "res_gate": float(res_gate),
                "res_closure": np.atleast_1d(np.asarray(res_closure, dtype=float)),
                "res_area": np.atleast_1d(np.asarray(res_area, dtype=float)),
            }
        )
        if len(self._rows) % self.flush_every == 0:
            self.flush()

    def append_budget(
        self,
        iter: int,
        coeffs_a,
        coeffs_c,
        *,
        cost_total: float,
        cost_c1: float,
        cost_c2: float,
        cost_c3: float,
        cost_c4: float,
        Phi_0: float,
        phi_vz: float,
        res_gate,
        res_c1_ends,
        epigraph_s: float | None = None,
    ) -> None:
        """Record one budget-schema checkpoint (F05-pre, ``schema_version`` 2).

        *coeffs_a*/*coeffs_c* are the design-curve series (kappa/tau), not the
        energy-era ``a``/``b`` waveform coefficients. *res_gate* is the
        3-component polar-decomposition residual and *res_c1_ends* the
        2-component (start, end) endpoint residual -- both vectors, unlike the
        energy schema's scalar ``res_gate``, per the task brief's "极分解 3
        分量" / "端点线性行". *epigraph_s* is ``None`` when the checkpoint's
        solver vector carried no peak slack; stored as NaN so the column stays
        rectangular (:meth:`write_step0`'s ``None``-as-NaN convention).
        """
        if self._closed:
            raise RuntimeError(f"recorder for {self.run_id} is already closed")
        if self.schema != "budget":
            raise RuntimeError(
                f"recorder for {self.run_id} uses schema={self.schema!r}; append_budget() "
                "requires Recorder(..., schema='budget')"
            )
        a = np.asarray(coeffs_a, dtype=float)
        c = np.asarray(coeffs_c, dtype=float)
        self._rows.append(
            {
                "iter": int(iter),
                "coeffs_a": a,
                "coeffs_c": c,
                "cost_total": float(cost_total),
                "cost_c1": float(cost_c1),
                "cost_c2": float(cost_c2),
                "cost_c3": float(cost_c3),
                "cost_c4": float(cost_c4),
                "Phi_0": float(Phi_0),
                "phi_vz": float(phi_vz),
                "res_gate": np.atleast_1d(np.asarray(res_gate, dtype=float)),
                "res_c1_ends": np.atleast_1d(np.asarray(res_c1_ends, dtype=float)),
                "epigraph_s": np.nan if epigraph_s is None else float(epigraph_s),
            }
        )
        if len(self._rows) % self.flush_every == 0:
            self.flush()

    def flush(self) -> None:
        """Write ``history.npz`` from the buffer. Idempotent."""
        if not self._rows:
            return
        payload = {}
        for key in self._history_columns:
            payload[key] = np.stack([np.asarray(r[key]) for r in self._rows])
        np.savez(self.path / "history.npz", **payload)

    def close(self, *, status: str, nit: int, wall_clock_s: float, **extra) -> Path:
        """Flush the history and write ``final.json``.

        *status* is the ``stop_reason`` and must be one of :data:`STOP_REASONS`;
        it lands in the manifest too, because whether a run may enter a
        quantitative claim is a property of the run, not of a side file.
        """
        if status not in STOP_REASONS:
            raise ValueError(f"status must be one of {STOP_REASONS}, got {status!r}")
        self.flush()
        final = {
            "status": status,
            "nit": int(nit),
            "wall_clock_s": float(wall_clock_s),
            "n_checkpoints": len(self._rows),
            "void_for_claims": status in ("maxiter", "failed", "synthetic"),
            **extra,
        }
        if self._rows:
            last = self._rows[-1]
            final["final_terms"] = {k: last[k] for k in self._cost_columns}
        (self.path / "final.json").write_text(
            json.dumps(_jsonable(final), indent=2, sort_keys=True) + "\n"
        )
        self.manifest["stop_reason"] = status
        self._write_manifest()
        self._closed = True
        return self.path


# -- reading --------------------------------------------------------------


def _read_npz(path: Path) -> dict:
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def load(run_id: str, root: Path | str = RUNS_DIR) -> RunRecord:
    """Read a RunRecord back. Raises if the run or its manifest is absent.

    This is the *only* door to stored run data. Nothing is regenerated, nothing
    is defaulted: a missing directory is an error, not an empty record, so a
    figure can never be silently produced from thin air.
    """
    path = Path(root) / run_id
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no RunRecord at {path} (manifest.json missing). Nothing is recomputed "
            "here: run data is the only source for figures (_plan.md §4.3)."
        )
    manifest = json.loads(manifest_path.read_text())
    final_path = path / "final.json"
    return RunRecord(
        run_id=run_id,
        manifest=manifest,
        history=_read_npz(path / "history.npz"),
        step0=_read_npz(path / "step0.npz"),
        final=json.loads(final_path.read_text()) if final_path.is_file() else {},
    )


def list_runs(root: Path | str = RUNS_DIR) -> tuple[str, ...]:
    """Run ids present under *root*, sorted. Empty when the store does not exist."""
    root = Path(root)
    if not root.is_dir():
        return ()
    return tuple(sorted(p.name for p in root.iterdir() if (p / "manifest.json").is_file()))


def delete_run(run_id: str, root: Path | str = RUNS_DIR) -> None:
    """Remove one run directory. Only ever used by tests and explicit cleanup."""
    path = Path(root) / run_id
    if path.is_dir():
        shutil.rmtree(path)


def _device_from_manifest(device_manifest: dict) -> Device:
    """Reconstruct a :class:`curve_opt.device.Device` from a stored manifest block.

    ``Device.to_manifest()`` (``device.py``) stores the dataclass fields plus a
    ``"derived"`` sub-dict of read-only properties; only the former are
    constructor arguments, so the derived block is dropped here rather than
    fed back in as a keyword ``Device`` does not accept.
    """
    return Device(**{k: v for k, v in device_manifest.items() if k != "derived"})


def load_device(path) -> Device:
    """Build a :class:`~curve_opt.device.Device` from a JSON config file (F08b).

    The file may be flat (``{"gate_time": 40.0}``) or grouped the way
    :meth:`Device.groups` emits (``{"hardware": {...}, "noise": {...},
    "design": {...}}``), and needs to name only the fields it overrides -- every
    other field keeps the documented default from ``device.py``. A
    :meth:`Device.to_manifest` blob works too, so a device can be lifted straight
    out of a RunRecord.

    Lives here rather than on ``Device`` because ``_plan.md`` §4.2 makes this the
    only module permitted to read the filesystem; ``Device.from_dict`` does all
    the actual work and stays a pure function.

    An unknown field is a ``ValueError``, not a silent no-op -- see
    :meth:`Device.from_dict`.
    """
    return Device.from_dict(json.loads(Path(path).read_text()))


def _verify_history_energy(record: RunRecord, *, atol: float) -> dict:
    """``schema_version`` 1: recompute ``cost_energy``/``cost_curv``/``cost_peak``."""
    T = float(record.manifest["T"])
    N = int(record.manifest["N_grid"])
    a_all = np.asarray(record.history["coeffs_a"])
    b_all = np.asarray(record.history["coeffs_b"])
    is_general = record.manifest.get("layer") == "general"

    deviations = {"cost_energy": 0.0, "cost_curv": 0.0, "cost_peak": 0.0}
    for k in range(a_all.shape[0]):
        a, b = a_all[k], b_all[k]
        # Layer is problem metadata, not a property that can be inferred from one
        # checkpoint.  A general-layer trajectory may begin exactly at b == 0 or
        # converge back to it; recomputing that row through the planar arithmetic
        # path changes floating-point summation order and can break the exact
        # redundant-channel check even though the record is sound (Step 16).
        b_arg = b if is_general else None
        terms = metrics.cost_terms(a, T, N, b=b_arg)
        for column, value in (
            ("cost_energy", terms.energy),
            ("cost_curv", terms.curv),
            ("cost_peak", terms.peak),
        ):
            dev = abs(float(record.history[column][k]) - value)
            deviations[column] = max(deviations[column], dev)

    bad = {k: v for k, v in deviations.items() if v > atol}
    if bad:
        raise AssertionError(
            f"run {record.run_id}: stored cost terms disagree with values recomputed "
            f"from the stored coefficients at N_grid={N}, T={T}: {bad} (atol={atol})"
        )
    return deviations


def _verify_history_budget(record: RunRecord, *, atol: float) -> dict:
    """``schema_version`` 2: recompute ``cost_c1``..``cost_c4`` via :func:`curve_opt.budget.budget_terms`.

    Uses the *stored* device block (:func:`_device_from_manifest`), not the
    process's :data:`curve_opt.device.DEFAULT_DEVICE`, so a record survives
    even if the default device is later changed (F06 sweeps it).
    """
    T = float(record.manifest["T"])
    N = int(record.manifest["N_grid"])
    dev = _device_from_manifest(record.manifest["device"])
    a_all = np.asarray(record.history["coeffs_a"])
    c_all = np.asarray(record.history["coeffs_c"])
    Phi_0_all = np.asarray(record.history["Phi_0"])

    deviations = {"cost_c1": 0.0, "cost_c2": 0.0, "cost_c3": 0.0, "cost_c4": 0.0}
    for k in range(a_all.shape[0]):
        terms = budget.budget_terms(a_all[k], c_all[k], float(Phi_0_all[k]), T, dev, N)
        for column, value in (
            ("cost_c1", terms.c1),
            ("cost_c2", terms.c2),
            ("cost_c3", terms.c3),
            ("cost_c4", terms.c4),
        ):
            dev_max = abs(float(record.history[column][k]) - float(value))
            deviations[column] = max(deviations[column], dev_max)

    bad = {k: v for k, v in deviations.items() if v > atol}
    if bad:
        raise AssertionError(
            f"run {record.run_id}: stored budget terms disagree with values recomputed "
            f"from the stored coefficients at N_grid={N}, T={T}: {bad} (atol={atol})"
        )
    return deviations


def verify_history(record: RunRecord, *, atol: float = 0.0) -> dict:
    """Recompute the stored cost terms from the stored coefficients.

    ``_plan.md`` §5.2 asks for agreement to 1e-12; the default here is **exact**
    (``atol=0.0``), because each term is a deterministic function of the
    coefficients and of ``N_grid`` from the manifest. Any deviation therefore
    means the record disagrees with its own metadata rather than that arithmetic
    drifted -- a wrong grid size, a transposed coefficient array, a dtype
    narrowed on the way to disk.

    Dispatches on ``record.manifest["schema_version"]`` (F05-pre): ``1`` uses
    :func:`curve_opt.metrics.cost_terms` (energy schema, unchanged since Step
    07); ``2`` uses :func:`curve_opt.budget.budget_terms` (budget schema).
    Records written before ``schema_version`` existed as a field default to
    ``1``, so every pre-F05-pre record takes the original path untouched.

    Returns the maximum absolute deviation per term and raises
    :class:`AssertionError` when one exceeds *atol*.
    """
    if not record.history:
        raise ValueError(f"run {record.run_id} has no history to verify")
    version = int(record.manifest.get("schema_version", 1))
    if version == 2:
        return _verify_history_budget(record, atol=atol)
    return _verify_history_energy(record, atol=atol)


# ---------------------------------------------------------------------------
# waveform export (F09: moved here from the removed qutip layer -- the recorder
# is the only module allowed to touch the filesystem)
# ---------------------------------------------------------------------------


def export_waveform_csv(path, t_mid, om_x, om_y) -> None:
    """Write ``(time_ns, Omega_x_rad_per_ns, Omega_y_rad_per_ns)`` to *path*.

    One row per AWG sample (cell-midpoint convention, matching every other
    grid in this project -- each row is the constant value the pulse holds
    over that sample's ``dt`` interval, the natural reading of an "AWG
    sample").
    """
    t_mid = np.asarray(t_mid, dtype=float)
    om_x = np.asarray(om_x, dtype=float)
    om_y = np.asarray(om_y, dtype=float)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_ns", "Omega_x_rad_per_ns", "Omega_y_rad_per_ns"])
        for t, x, y in zip(t_mid, om_x, om_y):
            writer.writerow([f"{t:.12e}", f"{x:.12e}", f"{y:.12e}"])
