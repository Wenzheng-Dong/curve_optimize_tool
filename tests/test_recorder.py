"""Step 07 acceptance: RunRecord write / read / plot, and the two hard guarantees.

Every test writes into ``tmp_path``, never into the real ``_runs/``.

The acceptance criteria of ``_plan.md`` §5.2 are:

1. from ``<run_id>/`` alone, produce the waveform at any checkpoint, the cost and
   residual history, and the step-0 before/after comparison;
2. the stored per-term costs agree with values recomputed from the stored
   coefficients (the plan asks 1e-12; here it is required to be **exact**);
3. presentation material is re-plotted from records, never transcribed.

Schema validation uses a *fabricated* trajectory built from the step00d converged
coefficients, as the plan suggests. It is marked ``synthetic`` in the manifest,
which makes it void for claims and puts a watermark on every figure.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from curve_opt import ansatz, basis, geometry, metrics, plotting, recorder

BEST_COEFFS = Path(__file__).resolve().parents[1] / "_dev_logs" / "best_planar_coeffs.json"
T = 1.0
N_GRID = 4000


def _fake_trajectory(K: int = 25, M: int = 12):
    """A plausible descent: naive start -> the step00d optimum, K checkpoints.

    Not an optimization -- a deliberately fabricated path whose only purpose is to
    exercise the schema. Costs and residuals are evaluated from the coefficients
    with the production functions, so the double-channel check is a check on
    serialization, not on arithmetic.
    """
    end = np.array(json.loads(BEST_COEFFS.read_text())["M12"]["coeffs"])
    start = basis.naive_coeffs(np.pi, T, M=M)
    g = basis.gate_row(M, T)
    rows = []
    for k in range(K):
        s = k / (K - 1)
        a = (1 - s) * start + s * end
        terms = metrics.cost_terms(a, T, N_GRID)
        res = np.asarray(geometry.equality_residuals(a, T, N_GRID))
        rows.append(
            {
                "iter": k,
                "coeffs_a": a,
                "cost_total": terms.energy,
                "cost_energy": terms.energy,
                "cost_curv": terms.curv,
                "cost_peak": terms.peak,
                "res_gate": abs(float(g @ a) - np.pi),
                "res_closure": res[:2],
                "res_area": res[2:],
            }
        )
    return rows


def _manifest(**overrides) -> dict:
    man = {
        "ansatz": {"name": "naive", "source": "builtin", "params": {"theta": np.pi}},
        "M": 12,
        "T": T,
        "target_gate": {"name": "X(pi)", "matrix": [[0, 1], [1, 0]]},
        "N_grid": N_GRID,
        "winding_branch": float(np.pi),
        "objective": {"terms": ["energy"], "lambda": 0.0, "epigraph": False},
        "solver": {"method": "trust-constr", "maxiter": 1500, "gn_hessian": True},
        "synthetic": True,
    }
    man.update(overrides)
    return man


@pytest.fixture
def written(tmp_path):
    """A complete synthetic run on disk; returns (root, run_id).

    ``root`` is a *subdirectory* of tmp_path so that the relocation probe can move
    the store somewhere genuinely outside it.
    """
    root = tmp_path / "runs"
    run_id = recorder.make_run_id("Xpi", "naive", "M12_lam0_synthetic", date="20260728")
    rec = recorder.Recorder(run_id, _manifest(), root=root, flush_every=10)

    ansatz.register_builtin_families(overwrite=True)
    rec.write_step0(ansatz.project_named("naive", M=12, T=T, theta=np.pi))
    for row in _fake_trajectory():
        rec.append(**row)
    rec.close(status="synthetic", nit=24, wall_clock_s=0.0, note="fabricated for schema check")
    return root, run_id


# --------------------------------------------------------------------------
# schema round trip
# --------------------------------------------------------------------------


def test_run_id_format_and_validation():
    assert recorder.make_run_id("Xpi", "rcp_lem", "M12_lam0", date="20260801") == (
        "20260801_Xpi_rcp_lem_M12_lam0"
    )
    with pytest.raises(ValueError):
        recorder.make_run_id("Xpi", "a/b", "lab")
    with pytest.raises(ValueError):
        recorder.make_run_id("Xpi", "", "lab")


def test_four_files_are_written(written):
    root, run_id = written
    path = root / run_id
    for name in ("manifest.json", "step0.npz", "history.npz", "final.json"):
        assert (path / name).is_file(), name


def test_manifest_carries_everything_needed_to_interpret_the_run(written):
    root, run_id = written
    man = recorder.load(run_id, root=root).manifest
    for key in recorder.REQUIRED_MANIFEST_KEYS:
        assert key in man, key
    # provenance is recorded automatically, not left to the caller
    assert man["git_commit"] and man["git_branch"]
    assert man["timestamp"].startswith("20")
    assert man["stop_reason"] == "synthetic"
    # winding_branch is a quadrature estimate on the source sampling, not a
    # closed form: a sampled ansatz has no exact gate angle (see ansatz.py).
    assert man["winding_branch"] == pytest.approx(np.pi, abs=1e-8)
    assert man["ansatz"]["projection_protocol"] == ansatz.PROTOCOL


def test_manifest_missing_keys_is_refused(tmp_path):
    """§5.1: without the manifest the data cannot be read, so it is mandatory."""
    incomplete = _manifest()
    del incomplete["winding_branch"]
    with pytest.raises(ValueError, match="winding_branch"):
        recorder.Recorder("bad_run", incomplete, root=tmp_path)
    assert not (tmp_path / "bad_run").exists()


def test_bad_stop_reason_is_refused(tmp_path):
    rec = recorder.Recorder("r", _manifest(), root=tmp_path)
    with pytest.raises(ValueError, match="status must be one of"):
        rec.close(status="looks_fine", nit=1, wall_clock_s=1.0)


def test_existing_run_is_not_silently_overwritten(tmp_path):
    recorder.Recorder("r", _manifest(), root=tmp_path)
    with pytest.raises(FileExistsError):
        recorder.Recorder("r", _manifest(), root=tmp_path)
    recorder.Recorder("r", _manifest(), root=tmp_path, exist_ok=True)


def test_history_columns_match_the_schema(written):
    root, run_id = written
    rec = recorder.load(run_id, root=root)
    assert set(rec.history) == set(recorder.HISTORY_COLUMNS)
    K = rec.n_checkpoints
    assert K == 25
    assert rec.history["coeffs_a"].shape == (K, 12)
    assert rec.history["coeffs_b"].shape == (K, 12)
    assert rec.history["res_closure"].shape == (K, 2)
    assert rec.history["res_area"].shape == (K, 1)
    assert rec.history["coeffs_a"].dtype == np.float64


def test_coefficients_round_trip_bit_for_bit(written):
    """The coefficients are the primary data; nothing may be lost writing them."""
    root, run_id = written
    stored = recorder.load(run_id, root=root).history["coeffs_a"]
    expected = np.stack([row["coeffs_a"] for row in _fake_trajectory()])
    assert np.array_equal(stored, expected)


def test_final_json_marks_a_synthetic_run_void_for_claims(written):
    root, run_id = written
    final = recorder.load(run_id, root=root).final
    assert final["status"] == "synthetic"
    assert final["void_for_claims"] is True
    assert final["n_checkpoints"] == 25
    assert final["note"] == "fabricated for schema check"
    assert set(final["final_terms"]) == {"cost_total", "cost_energy", "cost_curv", "cost_peak"}


def test_flush_leaves_usable_history_before_close(tmp_path):
    """A crashed run must not lose everything: history is flushed periodically."""
    rec = recorder.Recorder("partial", _manifest(), root=tmp_path, flush_every=5)
    for row in _fake_trajectory(K=12):
        rec.append(**row)
    loaded = recorder.load("partial", root=tmp_path)  # no close() at all
    assert loaded.n_checkpoints == 10  # two flushes of five
    assert loaded.final == {}
    assert loaded.manifest["stop_reason"] is None


def test_list_runs_and_delete(written):
    root, run_id = written
    assert recorder.list_runs(root=root) == (run_id,)
    recorder.delete_run(run_id, root=root)
    assert recorder.list_runs(root=root) == ()
    assert recorder.list_runs(root=root / "nope") == ()


def test_runs_dir_default_points_at_the_repository_store():
    assert recorder.RUNS_DIR == recorder.REPO_ROOT / "_runs"
    assert (recorder.REPO_ROOT / "pyproject.toml").is_file()


# --------------------------------------------------------------------------
# ★ acceptance 2: the double-channel cost check, exact
# --------------------------------------------------------------------------


def test_stored_costs_equal_values_recomputed_from_coefficients(written):
    """★ §5.2: stored vs recomputed, required here to be bit-identical."""
    root, run_id = written
    deviations = recorder.verify_history(recorder.load(run_id, root=root), atol=0.0)
    assert deviations == {"cost_energy": 0.0, "cost_curv": 0.0, "cost_peak": 0.0}


def test_verify_history_uses_manifest_layer_when_general_b_is_exactly_zero(tmp_path):
    """A zero Omega_y checkpoint is still general-layer data (Step 16 regression)."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=14) * 10.0
    b = np.zeros(14)
    terms = metrics.cost_terms(a, T, N_GRID, b=b)
    rec = recorder.Recorder(
        "general_zero_b",
        _manifest(M=14, layer="general"),
        root=tmp_path,
    )
    rec.append(
        iter=0,
        coeffs_a=a,
        coeffs_b=b,
        cost_total=terms.energy,
        cost_energy=terms.energy,
        cost_curv=terms.curv,
        cost_peak=terms.peak,
        res_gate=1.0,
        res_closure=np.ones(3),
        res_area=np.ones(3),
    )
    rec.close(status="synthetic", nit=0, wall_clock_s=0.0)
    loaded = recorder.load("general_zero_b", root=tmp_path)
    assert recorder.verify_history(loaded, atol=0.0) == {
        "cost_energy": 0.0,
        "cost_curv": 0.0,
        "cost_peak": 0.0,
    }


def test_verify_history_catches_a_tampered_cost(written):
    """Negative control: the check must actually be able to fail."""
    root, run_id = written
    path = root / run_id / "history.npz"
    with np.load(path) as data:
        payload = {k: data[k] for k in data.files}
    payload["cost_energy"][7] += 1e-9
    np.savez(path, **payload)
    with pytest.raises(AssertionError, match="cost_energy"):
        recorder.verify_history(recorder.load(run_id, root=root))


def test_verify_history_catches_a_wrong_grid_in_the_manifest(written):
    """The grid-dependent terms tie the record to its own metadata.

    Recording N_grid = 4000 but having evaluated the costs on another grid is
    exactly the kind of silent inconsistency that makes a record uninterpretable
    later; the check notices because curv and peak depend on N.
    """
    root, run_id = written
    manifest_path = root / run_id / "manifest.json"
    man = json.loads(manifest_path.read_text())
    man["N_grid"] = 1000
    manifest_path.write_text(json.dumps(man))
    with pytest.raises(AssertionError) as excinfo:
        recorder.verify_history(recorder.load(run_id, root=root))
    assert "cost_curv" in str(excinfo.value) or "cost_peak" in str(excinfo.value)
    # energy is analytic and grid-free, so it is *not* what fails
    assert "'cost_energy': 0.0" not in str(excinfo.value)


def test_energy_column_matches_the_analytic_closed_form(written):
    """Cross-check against basis, not just against metrics."""
    root, run_id = written
    rec = recorder.load(run_id, root=root)
    for k in range(rec.n_checkpoints):
        a = rec.history["coeffs_a"][k]
        assert rec.history["cost_energy"][k] == basis.energy_invariant(a, T)


def test_last_checkpoint_reproduces_the_step00d_optimum(written):
    """The fabricated path ends at the recorded optimum; the record shows it."""
    root, run_id = written
    rec = recorder.load(run_id, root=root)
    assert float(rec.history["cost_energy"][-1]) == pytest.approx(83.75, abs=5e-3)
    assert float(rec.history["cost_curv"][-1]) == pytest.approx(6.638, abs=5e-4)
    assert float(rec.history["cost_peak"][-1]) == pytest.approx(22.63, abs=5e-3)
    # The residual is grid-limited: 2e-7 on the N=4000 optimization grid, 8.4e-9
    # when the same coefficients are re-evaluated on the truth grid. That gap is
    # precisely why a claim needs a fine-grid re-evaluation (_plan.md §6.4).
    a_last = rec.history["coeffs_a"][-1]
    assert np.max(np.abs(rec.history["res_closure"][-1])) < 1e-6
    assert geometry.closure_invariant(a_last, T, N=500_000) < 1e-8


# --------------------------------------------------------------------------
# step0 serialization
# --------------------------------------------------------------------------


def test_step0_round_trip_planar(written):
    root, run_id = written
    step0 = recorder.load(run_id, root=root).step0
    assert float(step0["rel_rms"]) < 1e-13  # naive is exact in the basis
    assert bool(step0["closure_after_available"])
    assert float(step0["theta_before"]) == pytest.approx(np.pi, abs=1e-8)


def test_step0_stores_unavailable_after_values_as_nan_not_zero(tmp_path):
    """An unavailable value must be storable as "unknown", never as 0.0.

    Storing 0.0 for a missing robustness value would read as "perfectly robust".
    Until Step 11 this was the everyday case (a non-planar ansatz had no
    after-projection closure without the propagator); now that the propagator
    exists the situation is constructed explicitly, because the *recorder's*
    contract still has to hold for any field that is ever unavailable.
    """
    ansatz.register_builtin_families(overwrite=True)
    report = ansatz.project_named("naive", M=8, T=1.0, theta=np.pi)._replace(
        closure_after=None, area_after=None
    )
    assert report.closure_after is None

    rec = recorder.Recorder("r_unavailable", _manifest(), root=tmp_path)
    rec.write_step0(report)
    rec.close(status="synthetic", nit=0, wall_clock_s=0.0)
    step0 = recorder.load("r_unavailable", root=tmp_path).step0
    assert np.isnan(float(step0["closure_after"]))
    assert not bool(step0["closure_after_available"])
    assert np.isnan(float(step0["area_after"]))
    assert not np.isnan(float(step0["closure_before"]))


def test_step0_of_a_3d_ansatz_now_carries_real_after_values(tmp_path):
    """The counterpart: Step 11 filled these in, and they must round-trip."""
    src = ansatz.AnsatzSource(
        name="fake3d",
        tier="forced",
        T=1.0,
        times=np.linspace(0, 1, 501),
        omega_x=np.sin(np.pi * np.linspace(0, 1, 501)),
        omega_y=0.5 * np.ones(501),
        positions=np.column_stack(
            [np.linspace(0, 1, 501), np.zeros(501), np.linspace(0, 1, 501)]
        ),
    )
    report = ansatz.project(src, M=8, T=1.0)
    assert report.closure_after is not None and report.area_after is not None

    rec = recorder.Recorder("r3d", _manifest(), root=tmp_path)
    rec.write_step0(report)
    rec.close(status="synthetic", nit=0, wall_clock_s=0.0)
    step0 = recorder.load("r3d", root=tmp_path).step0
    assert bool(step0["closure_after_available"])
    assert float(step0["closure_after"]) == pytest.approx(report.closure_after, rel=1e-12)


# --------------------------------------------------------------------------
# ★ acceptance 1 + the "move the store away" probe
# --------------------------------------------------------------------------


def test_three_required_figures_from_the_record_alone(written, tmp_path):
    """★ §5.2(a)(b)(c): all three figures, drawn from ``<run_id>/`` and nothing else."""
    root, run_id = written
    rec = recorder.load(run_id, root=root)
    out = tmp_path / "figs"
    out.mkdir()

    for fname, fig in (
        ("step07_record_waveform.png", plotting.plot_waveform(rec, checkpoint=-1,
                                                             save_path=out / "w.png")),
        ("step07_record_history.png", plotting.plot_history(rec, save_path=out / "h.png")),
        ("step07_record_step0.png", plotting.plot_step0(rec, save_path=out / "s.png")),
    ):
        assert fig is not None, fname
    assert (out / "w.png").stat().st_size > 5000
    assert (out / "h.png").stat().st_size > 5000
    assert (out / "s.png").stat().st_size > 5000


def test_waveform_is_rebuilt_from_coefficients_not_stored(written):
    """§5.1's promise: the coefficients alone reconstruct the waveform."""
    root, run_id = written
    rec = recorder.load(run_id, root=root)
    t, om_x, om_y = plotting.waveform_at(rec, checkpoint=-1)
    a = rec.history["coeffs_a"][-1]
    assert np.array_equal(om_x, basis.omega(a, t, T))
    assert not np.any(om_y)
    assert "omega_x" not in rec.history  # the waveform is not in the record at all


def test_any_checkpoint_can_be_plotted(written):
    root, run_id = written
    rec = recorder.load(run_id, root=root)
    first = plotting.waveform_at(rec, checkpoint=0)[1]
    last = plotting.waveform_at(rec, checkpoint=-1)[1]
    mid = plotting.waveform_at(rec, checkpoint=12)[1]
    assert not np.allclose(first, last)
    assert not np.allclose(mid, last)


def test_synthetic_records_are_watermarked(written):
    """Fabricated data must be visibly fabricated on every figure."""
    root, run_id = written
    rec = recorder.load(run_id, root=root)
    assert rec.is_synthetic
    fig = plotting.plot_history(rec)
    texts = [t.get_text() for t in fig.texts]
    assert "SYNTHETIC" in texts
    assert any("stop_reason=synthetic" in t for t in texts)


def test_loading_a_missing_run_raises_instead_of_regenerating(tmp_path):
    """Nothing is recomputed on a cache miss -- there is no cache and no fallback."""
    with pytest.raises(FileNotFoundError, match="Nothing is recomputed"):
        recorder.load("never_written", root=tmp_path)


def test_plotting_is_unreachable_without_the_store(written, tmp_path):
    """★ The probe: move ``_runs`` away and no figure can be produced.

    After the store is gone, :func:`recorder.load` raises -- and since plotting
    takes a record object and imports neither the recorder nor any path API (see
    test_architecture.py), there is no other way to obtain one. A figure existing
    therefore proves its data was read from disk.

    An already-loaded record keeps working, which is the correct behaviour and the
    point of the schema: the record, once read, is sufficient on its own.
    """
    root, run_id = written
    in_memory = recorder.load(run_id, root=root)

    shutil.move(str(root / run_id), str(tmp_path / "moved_away"))  # outside `root`

    with pytest.raises(FileNotFoundError):
        recorder.load(run_id, root=root)
    assert recorder.list_runs(root=root) == ()

    # the loaded record is self-sufficient: all three figures still come out
    assert plotting.plot_waveform(in_memory) is not None
    assert plotting.plot_history(in_memory) is not None
    assert plotting.plot_step0(in_memory) is not None
    assert recorder.verify_history(in_memory, atol=0.0)["cost_energy"] == 0.0

    # and plotting exposes no way to fetch a record itself
    for attr in ("load", "RunRecord", "RUNS_DIR", "Path", "open"):
        assert not hasattr(plotting, attr), attr


def test_plotting_refuses_an_empty_record(tmp_path):
    rec = recorder.Recorder("empty", _manifest(), root=tmp_path)
    rec.close(status="failed", nit=0, wall_clock_s=0.0)
    loaded = recorder.load("empty", root=tmp_path)
    with pytest.raises(ValueError, match="no history"):
        plotting.plot_history(loaded)
    with pytest.raises(ValueError, match="no step0"):
        plotting.plot_step0(loaded)


def test_pareto_marks_non_claim_runs_hollow(written):
    root, run_id = written
    rec = recorder.load(run_id, root=root)
    fig = plotting.plot_pareto([rec])
    (line,) = [ln for ln in fig.axes[0].lines if ln.get_marker() == "x"]
    assert line.get_markerfacecolor() == "none"
