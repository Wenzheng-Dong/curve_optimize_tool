"""Step 10 acceptance, as a regression test on the recorded front.

The sweep itself is a data-generating script (`_dev_logs/step10_pareto.py`); what
belongs in the test suite is the *claim discipline* the recorded front has to
satisfy, so that a later change which quietly breaks it fails here rather than in
a figure nobody re-reads. Everything is read from the tracked artefacts, nothing
is recomputed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

DEV_LOGS = Path(__file__).resolve().parents[1] / "_dev_logs"
FRONT_CSV = DEV_LOGS / "step10_pareto_front.csv"
POSED_JSON = DEV_LOGS / "step10_lambda_wellposedness.json"


@pytest.fixture(scope="module")
def front():
    with FRONT_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key not in ("run_id", "stop_reason"):
                row[key] = None if value == "" else float(value)
    return sorted(rows, key=lambda r: r["lam"])


def test_every_front_point_is_a_claim_run(front):
    """★ CONVERGED, optimized on the claim grid, and re-evaluated on fine grids."""
    assert len(front) >= 8
    for row in front:
        assert row["stop_reason"] == "converged", row["run_id"]
        assert row["nit"] < 1500, row["run_id"]
        assert row["fine_5e5"] <= 1e-8, (row["run_id"], row["fine_5e5"])
        assert row["fine_1e5"] <= 1e-8, (row["run_id"], row["fine_1e5"])
        assert row["max_equality_residual_opt_grid"] <= 1e-12, row["run_id"]
        assert row["gate_residual"] <= 1e-13, row["run_id"]


def test_front_is_monotone_in_both_objectives(front):
    """★ peak falls and energy rises with lambda -- otherwise it is not a front."""
    peaks = [r["peak"] for r in front]
    energies = [r["energy"] for r in front]
    assert all(peaks[i] > peaks[i + 1] for i in range(len(peaks) - 1)), peaks
    assert all(energies[i] < energies[i + 1] for i in range(len(energies) - 1)), energies


def test_front_anchors_on_the_step08_optimum(front):
    """lambda = 0 must be the pure-energy claim of Step 08."""
    anchor = front[0]
    assert anchor["lam"] == 0.0
    assert anchor["energy"] == pytest.approx(83.7479, abs=5e-4)
    assert anchor["peak"] == pytest.approx(22.632, abs=5e-3)


def test_no_point_is_flagged_degenerate(front):
    """The front carries no hollow-x point: none failed a discipline check."""
    assert [r["run_id"] for r in front if r["degenerate"]] == []


def test_epigraph_variable_tracks_the_reported_peak(front):
    """T s is the constrained peak; the reported one may exceed it only marginally."""
    for row in front:
        if row["peak_over_Ts"] is None:
            assert row["lam"] == 0.0
            continue
        assert 1.0 <= row["peak_over_Ts"] < 1.001, (row["run_id"], row["peak_over_Ts"])


def test_high_frequency_content_stays_bounded_but_grows(front):
    """The degeneracy the risk register warns about, quantified.

    Buying a lower peak with high-frequency ripple would show up as a rising tail
    fraction. It does rise with lambda, and at the largest lambda swept it is
    already within a factor of ~2 of the 0.10 flag -- which is where a sweep that
    pushed further would start producing solutions that are numerically fine and
    experimentally useless.
    """
    tails = [r["tail_fraction"] for r in front]
    assert max(tails) < 0.10
    assert tails[-1] > tails[0]


def test_lambda_nonzero_is_well_posed_from_every_start():
    """★ The v4.1 open risk: step00d's evidence covered lambda = 0 only."""
    posed = json.loads(POSED_JSON.read_text())
    assert posed["config"]["gate"] == "projection"
    assert posed["config"]["hessian_mode"] == "gauss_newton"
    assert len(posed["lambdas"]) >= 3
    for lam, entry in posed["lambdas"].items():
        assert entry["n_converged"] == len(entry["starts"]) >= 5, lam
        assert entry["objective_spread_rel"] < 1e-9, (lam, entry["objective_spread_rel"])
        assert entry["max_coeff_distance"] < 1e-5, (lam, entry["max_coeff_distance"])
        for label, run in entry["starts"].items():
            assert run["stop_reason"] == "converged", (lam, label)
            assert run["max_equality_residual"] < 1e-12, (lam, label)
