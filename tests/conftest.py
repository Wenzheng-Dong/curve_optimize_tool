"""Test-side fixtures. Locating upstream cct lives here, never in the package.

``curve_opt.ansatz`` imports ``curvecontroltoolbox`` lazily and does no path
handling at all -- upstream is read-only, is not a dependency of the ``curve``
environment, and the architecture guards forbid filesystem code outside
``recorder.py``. So the search for the sibling clone happens in the test harness
instead, and tests that need upstream skip cleanly when it is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pytest

# Figures are produced head-less in tests; must precede any pyplot import.
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Sibling clone of the personal cct remote (``_plan.md`` §1). Read-only.
CCT_SRC = REPO_ROOT.parent / "curvecontroltoolbox" / "src"


def _cct_available() -> bool:
    if not (CCT_SRC / "curvecontroltoolbox" / "curve_families.py").is_file():
        return False
    if str(CCT_SRC) not in sys.path:
        sys.path.insert(0, str(CCT_SRC))
    try:
        import curvecontroltoolbox  # noqa: F401
    except ImportError:
        return False
    return True


CCT_AVAILABLE = _cct_available()

requires_cct = pytest.mark.skipif(
    not CCT_AVAILABLE, reason=f"upstream curvecontroltoolbox not importable from {CCT_SRC}"
)


@pytest.fixture(scope="session")
def cct_registered():
    """Register the cct-backed ansatz families once per session."""
    from curve_opt import ansatz

    return ansatz.register_cct_families()


#: Sibling clone of the read-only Novera transmon layer (F01,
#: ``_plan_full_cost.md`` §4.2 decision 4). Its ``db_helpers`` module has no
#: package ``__init__.py`` -- it is a plain script directory added to
#: ``sys.path``, matching how the upstream notebook itself imports it.
NOVERA_SRC = (
    REPO_ROOT.parent / "pulse-shape-Novera" / "proposals" / "1qb-DB" / "1qb-DB-demo"
)


def _novera_available() -> bool:
    if not (NOVERA_SRC / "db_helpers.py").is_file():
        return False
    if str(NOVERA_SRC) not in sys.path:
        sys.path.insert(0, str(NOVERA_SRC))
    try:
        import db_helpers  # noqa: F401
    except ImportError:
        return False
    return True


NOVERA_AVAILABLE = _novera_available()

requires_novera = pytest.mark.skipif(
    not NOVERA_AVAILABLE, reason=f"upstream pulse-shape-Novera not importable from {NOVERA_SRC}"
)
