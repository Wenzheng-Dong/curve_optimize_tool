"""Architecture gate: the layering rules of ``_plan.md`` §4.2 / §4.3, by AST.

These rules are cheap to state and easy to erode once the modules fill up, so
they are checked mechanically from Step 02 on, while the modules are still
stubs. The tests are guards for the steps that follow, not evidence about the
current (empty) implementation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import curve_opt

SRC = Path(curve_opt.__file__).parent

#: Only the recorder may perform filesystem I/O (§4.2: "the only module that
#: touches _runs/"). ``__future__`` is not an import in the relevant sense.
FS_MODULES = {"os", "os.path", "pathlib", "shutil", "json", "pickle", "tempfile", "glob"}
FS_CALLS = {"open", "savez", "savez_compressed", "save", "load", "loadtxt", "savetxt"}


def _imports(tree):
    """Yield the top-level module name of every import in *tree*."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module.split(".")[0]
            elif node.level > 0 and node.module:
                yield f".{node.module.split('.')[0]}"


def _called_names(tree):
    """Yield the called attribute/function names in *tree* (``np.save`` -> save)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                yield func.id
            elif isinstance(func, ast.Attribute):
                yield func.attr


def _tree(name):
    return ast.parse((SRC / f"{name}.py").read_text())


@pytest.mark.parametrize("name", [m for m in curve_opt.MODULES if m != "recorder"])
def test_only_recorder_does_filesystem_io(name):
    tree = _tree(name)
    bad_imports = FS_MODULES.intersection(_imports(tree))
    assert not bad_imports, (
        f"{name}.py imports {sorted(bad_imports)}; only recorder.py may do I/O "
        "(_plan.md §4.2). Route it through curve_opt.recorder."
    )
    bad_calls = FS_CALLS.intersection(_called_names(tree))
    assert not bad_calls, f"{name}.py calls {sorted(bad_calls)}; I/O belongs in recorder.py"


def test_plotting_never_reruns_an_optimization():
    """§4.3 item 1 -- figures come from a RunRecord, not from a fresh solve."""
    imported = set(_imports(_tree("plotting")))
    forbidden = {"optimize", ".optimize", "scipy"}
    assert not forbidden.intersection(imported), (
        "plotting.py may not import the optimizer or scipy.optimize: figures are "
        "drawn from stored RunRecord data only (_plan.md §4.3)."
    )


def test_basis_stays_analytic_numpy_only():
    """§4.2 -- basis.py is the home of the closed forms; numpy only.

    Keeping JAX out of it is what makes it usable as an independent arbiter for
    the differentiable implementations.
    """
    imported = set(_imports(_tree("basis")))
    assert "jax" not in imported
    assert "scipy" not in imported


def test_geometry_and_propagate_do_not_import_the_optimizer():
    """Core computation must not depend on the solver layer (no cycles)."""
    for name in ("basis", "geometry", "propagate", "metrics"):
        imported = set(_imports(_tree(name)))
        assert "optimize" not in imported and ".optimize" not in imported, name
        assert "recorder" not in imported and ".recorder" not in imported, name


def test_x64_switch_lives_only_in_the_package_root():
    """Rule 0 must be enabled once, at import of the package -- not scattered."""
    hits = [
        p.name
        for p in sorted(SRC.glob("*.py"))
        if "jax_enable_x64" in p.read_text()
    ]
    assert hits == ["__init__.py"], f"jax_enable_x64 mentioned in {hits}"
