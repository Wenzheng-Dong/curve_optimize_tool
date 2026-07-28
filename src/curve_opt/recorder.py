"""RunRecord I/O -- the only module in the package that touches the filesystem.

Every optimization run goes through here (``_plan.md`` §5). Traceability is a
hard requirement, and the mechanical guarantee is the rule in §4.3 item 1:
:mod:`curve_opt.plotting` may only read a RunRecord, never re-run an
optimization. If a figure can be drawn from the stored data alone, the schema is
adequate; if it cannot, the schema is broken.

Layout -- ``<RUNS_DIR>/<run_id>/`` (git-ignored local asset)
-----------------------------------------------------------
``run_id = <date>_<gate>_<ansatz>_<label>``, e.g.
``20260801_Xpi_rcp_lem_M12_lam0``.

* ``manifest.json`` -- the run's self-description; without it the data cannot be
  interpreted. run_id, timestamp, git commit, branch; ansatz (name, source,
  original parameters, representability tier, projection protocol); M, T,
  target gate (name + matrix), N_grid; ``winding_branch`` (the numerical value
  of theta -- the gate name alone does not define the problem, since theta and
  theta + 2 pi are the same gate but have lower bounds differing by theta^2);
  objective (terms, lambda or epsilon-constraint value, epigraph flag); solver
  (method, maxiter, GN Hessian on/off, tol, checkpoint_every); ``stop_reason``
  in {converged, early_stop, maxiter} -- a ``maxiter`` run is void by default.
* ``step0.npz`` -- the Fourier-projection error: raw Omega samples, projected
  coefficients, relRMS, and gate_err / closure / area before *and* after.
* ``history.npz`` -- one row per ``checkpoint_every`` iterations: ``iter``,
  ``coeffs_a[K, M]``, ``coeffs_b[K, M]`` (these alone reconstruct everything),
  ``cost_total``, the individual ``cost_energy`` / ``cost_curv`` / ``cost_peak``
  stored redundantly, and the residual trajectories ``res_gate`` /
  ``res_closure`` / ``res_area``.
* ``final.json`` -- status, nit, wall clock, per-term endpoint values, fine-grid
  re-evaluation if performed.

Planned API (Step 07)
---------------------
``Recorder(run_id, manifest).write_step0/append/close()``,
``load(run_id) -> RunRecord``.
"""

from __future__ import annotations

from pathlib import Path

#: Root of the raw run data, relative to the repository root. Git-ignored on
#: purpose: it is a local asset, self-describing through each manifest.
#: Presentation copies live in ``results/``, selected by the user and always
#: re-plotted from here -- numbers are never transcribed by hand.
RUNS_DIRNAME = "_runs"

#: Repository root = three levels up from this file (src/curve_opt/recorder.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Absolute path of the run store. Nothing outside this module may build it.
RUNS_DIR = REPO_ROOT / RUNS_DIRNAME

__all__ = ["REPO_ROOT", "RUNS_DIR", "RUNS_DIRNAME"]
