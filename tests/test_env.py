"""Environment gate: the acceptance criterion of Step 02.

If any of these fail, no numerical result from this repository is meaningful.
"""

from __future__ import annotations

import sys

import pytest


def test_core_stack_importable():
    """`python -c "import jax, numpy, scipy, qutip"` as a test."""
    import jax
    import numpy
    import qutip
    import scipy

    for mod in (jax, numpy, scipy, qutip):
        assert mod.__version__


def test_python_at_least_311():
    assert sys.version_info >= (3, 11)


def test_version_floors():
    """Pinned in environment.yml; all step00 numbers were produced on these."""
    import jax
    import numpy
    import qutip
    import scipy

    def tup(v):
        return tuple(int(x) for x in v.split(".")[:2])

    assert tup(jax.__version__) >= (0, 4)
    assert tup(numpy.__version__) >= (2, 0)
    assert tup(scipy.__version__) >= (1, 14)
    assert tup(qutip.__version__) >= (5, 0)


def test_x64_enabled_by_importing_the_package():
    """Numerical discipline rule 0: float64 must be on process-wide.

    Importing :mod:`curve_opt` is the single place that switches it on, so a
    bare ``import curve_opt`` has to be sufficient.
    """
    import curve_opt

    assert curve_opt.x64_enabled()


def test_x64_actually_produces_float64_arrays():
    """The config flag is necessary but not sufficient -- check real dtypes."""
    import curve_opt  # noqa: F401  (import for the side effect)
    import jax.numpy as jnp

    assert jnp.zeros(3).dtype == jnp.float64
    assert jnp.asarray(1.0).dtype == jnp.float64
    # Resolution that float32 does not have (its eps is ~1.2e-7): under float32
    # this sum collapses to 1.0 exactly, which is how the missing x64 switch
    # silently destroys every residual claim in this project.
    assert float(jnp.asarray(1.0) + jnp.asarray(1e-12)) != 1.0
    # A residual of 1e-17 must survive as a value in its own right.
    assert float(jnp.asarray(1e-17)) == 1e-17


def test_absent_dependencies_stay_absent():
    """The `curve` env deliberately has no diffrax / equinox / jaxopt.

    Guards against a module silently starting to depend on something the
    documented environment does not provide.
    """
    for name in ("diffrax", "equinox", "jaxopt"):
        with pytest.raises(ImportError):
            __import__(name)
