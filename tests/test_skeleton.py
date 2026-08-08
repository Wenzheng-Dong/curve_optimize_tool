"""Skeleton gate: the module layout of ``_plan.md`` §4.1 exists and imports."""

from __future__ import annotations

import importlib

import pytest

import curve_opt


def test_module_list_matches_the_plan():
    """``_plan.md`` §4.1 plus F01's ``device`` (``_plan_full_cost.md`` §4.4),
    the lowest-level dependency, ahead of ``basis``; F02-F04 add
    ``parametrization``/``novera``/``gate``/``budget`` ahead of ``optimize``,
    which depends on all four (``_plan_full_cost.md`` §4.4).
    """
    assert curve_opt.MODULES == (
        "device",
        "basis",
        "geometry",
        "propagate",
        "metrics",
        "ansatz",
        "parametrization",
        "novera",
        "gate",
        "budget",
        "optimize",
        "recorder",
        "plotting",
    )


@pytest.mark.parametrize("name", curve_opt.MODULES)
def test_module_imports(name):
    mod = importlib.import_module(f"curve_opt.{name}")
    assert mod.__doc__, f"{name} must document its responsibility and contracts"


def test_runs_dir_is_declared_but_not_created_on_import():
    """The recorder owns the path; importing it must not touch the disk."""
    from curve_opt import recorder

    assert recorder.RUNS_DIR.name == "_runs"
    assert (recorder.REPO_ROOT / "pyproject.toml").is_file()
