"""F01 acceptance criterion (6): no bare device-parameter literals outside ``device.py``.

AGENTS.md numerical discipline #10: ``T``, ``alpha``, ``Omega_max``, ``T1``/``T2``,
``delta_z``, ``epsilon`` may only live in :mod:`curve_opt.device`. This module
scans source files with Python's own ``ast`` module (the same tool
``test_architecture.py`` uses) for numeric literals that equal one of the
frozen ``Device`` field values, type-matched (a bare ``float`` literal only
matches the forbidden *float* set, a bare ``int`` literal only the forbidden
*int* set) so that unrelated integers/floats that happen to share a decimal
representation with a device constant are not falsely flagged -- see the
docstring of :func:`_scan_source` for the one case this mattered in practice
(``recorder.py``'s ``flush_every: int = 50``, unrelated to ``gate_time =
50.0``).

Scan scope
----------
``src/curve_opt/*.py`` except ``device.py`` itself (it *is* the source of these
numbers). No project "script directory" currently contains any full-cost-era
code: ``_dev_logs/*.py`` are frozen, pre-full-cost energy-era analysis scripts
(would be pure noise: they use unrelated ``T = 1`` conventions throughout, and
touching them is out of this step's scope), and `output/build_xpi_cost_comparison.py`
is a pre-existing energy-era artifact that already contains a *legitimate*
collision (a matplotlib layout margin ``rect=(0, 0.02, 1, 0.95)``, where
``0.02`` is exactly the default ``control_error`` -- confirmed by running this
scanner over it and reading the hit). If a `scripts/` directory appears in a
later full-cost step, glob it in here too -- ``SCRIPT_GLOBS`` below is where
that goes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import curve_opt
from curve_opt.device import DEFAULT_DEVICE

SRC = Path(curve_opt.__file__).parent

#: Any top-level "scripts" directory for the full-cost era; empty for now (see
#: the module docstring). Kept as a real glob, not a placeholder, so a future
#: step only has to create the directory for this test to start covering it.
SCRIPT_GLOBS = tuple((SRC.parent.parent / "scripts").glob("*.py"))

#: float(x) and int(x) both round-trip these literals exactly (they are short
#: decimal / exponent forms), so exact equality is safe -- no tolerance needed.
FORBIDDEN_FLOAT = {
    abs(DEFAULT_DEVICE.anharmonicity),  # 0.200 -- "-0.200" is UnaryOp(USub, Constant(0.200))
    DEFAULT_DEVICE.rabi_max,  # 0.050
    DEFAULT_DEVICE.t1,  # 60000.0 (== t2_echo)
    DEFAULT_DEVICE.static_detuning,  # 1e-4
    0.01,
    0.02,  # DEFAULT_DEVICE.control_error
    0.03,  # the other two swept control_error values (_plan_full_cost.md §3.2)
    DEFAULT_DEVICE.sample_rate,  # 2.4
    DEFAULT_DEVICE.gate_time,  # 50.0
    float(DEFAULT_DEVICE.n_modes),  # 20.0
}
FORBIDDEN_INT = {
    int(DEFAULT_DEVICE.t1),  # 60000
    DEFAULT_DEVICE.n_modes,  # 20
}


def _scan_source(source: str) -> list[tuple[int, object]]:
    """Return ``[(lineno, value), ...]`` for every forbidden numeric literal.

    Type-matched on purpose: ``50`` (int) is not flagged against the forbidden
    *float* ``50.0`` (``gate_time``), so an unrelated ``flush_every: int = 50``
    checkpoint interval elsewhere in the package does not false-positive. This
    was not a hypothetical design choice -- it is exactly what
    ``recorder.py``'s ``Recorder.__init__`` needed when this scanner was first
    run over the real tree.
    """
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        value = node.value
        if isinstance(value, bool):  # bool is an int subclass; not a device literal
            continue
        if isinstance(value, float) and value in FORBIDDEN_FLOAT:
            hits.append((node.lineno, value))
        elif isinstance(value, int) and value in FORBIDDEN_INT:
            hits.append((node.lineno, value))
    return hits


def _scan_file(path: Path) -> list[tuple[int, object]]:
    return _scan_source(path.read_text())


SCANNED_FILES = tuple(
    sorted(p for p in SRC.glob("*.py") if p.name != "device.py")
) + SCRIPT_GLOBS


@pytest.mark.parametrize("path", SCANNED_FILES, ids=lambda p: p.name)
def test_no_hardcoded_device_literals(path):
    hits = _scan_file(path)
    assert not hits, (
        f"{path.name} contains bare literal(s) matching a curve_opt.device.Device "
        f"field: {hits}. Import curve_opt.device.DEFAULT_DEVICE (or Device) instead "
        "of spelling the number (AGENTS.md numerical discipline #10)."
    )


def test_the_scan_actually_finds_at_least_one_forbidden_literal_per_type():
    """Negative self-test: the scanner must be able to catch a real violation.

    Without this, a scanner that (by a bug) matches nothing would make every
    test above vacuously pass. Covers both the float and the int branch.
    """
    fake_source_float = "rabi_max_hardcoded = 0.050\n"
    hits = _scan_source(fake_source_float)
    assert hits == [(1, 0.050)]

    fake_source_int = "n_modes_hardcoded = 20\n"
    hits = _scan_source(fake_source_int)
    assert hits == [(1, 20)]

    fake_source_negative = "anharmonicity_hardcoded = -0.200\n"
    hits = _scan_source(fake_source_negative)
    assert hits == [(1, 0.200)]  # UnaryOp(USub, Constant(0.200)): the Constant is positive

    # sanity: an unrelated number must NOT be flagged
    fake_source_clean = "n = 4000\nm = 0.5\n"
    assert _scan_source(fake_source_clean) == []


def test_the_scan_type_matching_avoids_the_recorder_flush_every_false_positive():
    """Documents the one real collision found and how type-matching resolves it."""
    source = "flush_every: int = 50\n"  # int 50, unrelated to gate_time = 50.0
    assert _scan_source(source) == []


def test_scanned_files_cover_every_module_except_device():
    names = {p.name for p in SCANNED_FILES if p.parent == SRC}
    expected = {f"{m}.py" for m in curve_opt.MODULES if m != "device"}
    assert expected <= names
