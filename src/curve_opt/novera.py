"""Thin adapter onto pulse-shape-Novera's three-level transmon layer (F03).

``_plan_full_cost.md`` §4.2 decision 4: the three-level simulation is reused
from ``Gate_construction/pulse-shape-Novera/proposals/1qb-DB/1qb-DB-demo``
(``db_helpers.py`` + ``db_transmon.py``), not rewritten -- that repository is
**read-only** for this project. Every function below is transcribed rather
than reimplemented: it lazily imports the upstream module and delegates
straight through, adding at most a device-object translation
(:func:`to_novera_device`) or an argument-format fix
(:func:`build_pulse`, for the ``to_physical`` incompatibility below).

Upstream reference and reproducibility
---------------------------------------
Pinned at git hash ``59fb6166e105a44064b440865b8fed3415258ad9`` (short form
``59fb616``, matching ``curve_opt.device``'s own docstring reference) --
:data:`UPSTREAM_GIT_HASH`. This is a frozen reference for the manifest (F04
wires it in; this step only has to make the constant exist and be correct),
not a live check against the sibling clone's current ``HEAD``.

No filesystem code here (``sys.path`` handling included)
-----------------------------------------------------------
Exactly the discipline ``curve_opt.ansatz`` already uses for the other
read-only upstream (``curvecontroltoolbox``, see that module's "Upstream
data" docstring section) and enforced by ``tests/test_architecture.py``'s
filesystem-I/O guard: this module never touches ``sys.path``, and every
upstream import happens lazily inside :func:`_require_novera`, so importing
``curve_opt.novera`` itself never requires upstream to be present. Locating
the sibling clone and inserting it onto ``sys.path`` is the *caller's* job --
for the test suite, ``tests/conftest.py``'s ``requires_novera`` marker
already does this, and tests using it skip cleanly (rather than fail) when
upstream is not importable.

The ``to_physical`` incompatibility, and how :func:`build_pulse` sidesteps it
---------------------------------------------------------------------------
Every upstream ``db_transmon`` function below that takes a ``pulse`` dict
internally calls ``db_transmon.to_physical``, which reconstructs a gate time
from the waveform's own **peak**::

    Tg = max|Tg*Omega(s)| / Omega_max

That is Novera's own gauge choice (every waveform runs at the same peak
Rabi rate, so its shape alone sets its duration) and is incompatible with
this project's, which fixes ``T = 50`` ns regardless of how far below
``Omega_max`` a given waveform's peak sits (F01 hit exactly this mismatch;
``_dev_logs/F01_forward_chain.md`` documents it, and
``_dev_logs/BUILDER_BRIEF.md`` §6 flags it as a standing trap). Reusing
``to_physical`` unmodified on this project's own waveforms would silently
compress or stretch time by the ratio ``Omega_max / max|Omega|``, which is
generally not 1.

Rather than reimplement ``to_physical`` (that would be "rewriting the
physics", exactly what this module is not supposed to do) or duplicate its
math with a different gate-time source, :func:`build_pulse` exploits the one
place ``to_physical`` reads its peak from -- the ``Tg_omega_envelope`` field
-- and sets that field to a *constant*, ``T * Omega_max``, independent of the
waveform's actual samples. ``to_physical`` then reconstructs
``Tg = (T*Omega_max)/Omega_max = T`` exactly, and the accompanying
``Tg_omega_x``/``Tg_omega_y`` fields are pre-scaled by ``T`` so that dividing
by that reconstructed ``Tg`` gives back the original physical
``omega_x``/``omega_y`` unchanged. ``Tg_omega_envelope`` is not read anywhere
else in the functions this module wraps (checked against every call site in
``db_transmon.py`` at git hash ``59fb616``: only ``to_physical`` reads it, to
build the unused ``physical["omega"]`` field). This is a documented gauge fix
at the call boundary, not a change to any upstream formula.

Transcribed vs. added
----------------------
``TransmonDevice`` / ``three_level_propagation`` / ``leakage_amplitude`` /
``gate_channel`` / ``subspace_gate_fidelity`` / ``block_gate_fidelity`` /
``best_frame_phase`` / ``calibrate`` are exactly the list
``_plan_full_cost.md`` §4.2 decision 4 names. One addition beyond that list:
:func:`three_level_gate`, upstream's coherent 3x3 propagator (no dissipator).
It is needed because ``three_level_propagation`` only returns the final
*state* reached from ``|0>`` (populations), not the full unitary -- and
:mod:`curve_opt.gate`'s polar decomposition needs the qubit *block* of the
full ``U_3(T)``, which only ``three_level_gate`` provides. Recorded here
rather than silently added, per the task brief's "existing functions are
whatever is actually there; missing/added ones go in the dev log."
"""

from __future__ import annotations

import numpy as np

from curve_opt.device import DEFAULT_DEVICE, Device

__all__ = [
    "UPSTREAM_GIT_HASH",
    "best_frame_phase",
    "block_gate_fidelity",
    "build_pulse",
    "calibrate",
    "gate_channel",
    "leakage_amplitude",
    "subspace_gate_fidelity",
    "three_level_gate",
    "three_level_propagation",
    "to_novera_device",
    "transmon_device_cls",
]

#: Full commit of the pinned upstream checkout (``_plan_full_cost.md`` §4.2
#: decision 4, ``_dev_logs/F03_task_brief.md`` §2.1). Frozen reference for the
#: manifest, not a live check -- see the module docstring.
UPSTREAM_GIT_HASH = "59fb6166e105a44064b440865b8fed3415258ad9"


def _require_novera():
    """Lazily import the two upstream modules; raise an actionable error if absent.

    Mirrors ``curve_opt.ansatz._require_cct`` exactly: upstream is read-only,
    not a dependency of the ``curve`` environment, and this module performs no
    ``sys.path`` handling of its own (see the module docstring).
    """
    try:
        import db_helpers  # noqa: PLC0415
        import db_transmon  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the caller's env
        raise ImportError(
            "pulse-shape-Novera is not importable. It is upstream, read-only and "
            "not a dependency of the `curve` environment: put "
            "`pulse-shape-Novera/proposals/1qb-DB/1qb-DB-demo` on sys.path to use "
            "curve_opt.novera (tests/conftest.py's requires_novera fixture does "
            "this for the test suite)."
        ) from exc
    return db_helpers, db_transmon


def transmon_device_cls():
    """Return ``db_transmon.TransmonDevice`` itself (lazy import)."""
    _, db_transmon = _require_novera()
    return db_transmon.TransmonDevice


def to_novera_device(device: Device = DEFAULT_DEVICE):
    """Map this project's :class:`curve_opt.device.Device` onto a ``TransmonDevice``.

    Field-for-field, except ``gate_time`` and ``n_modes`` (this project's own
    design choices -- ``_plan_full_cost.md`` §3.2 -- absent from
    ``TransmonDevice``). ``control_error`` carries over as given: this
    project's default is 2%, Novera's own module-level default is fixed at
    3% (``curve_opt.device``'s docstring documents that deviation); passing
    ``device`` explicitly reproduces whichever value is wanted.
    """
    _, db_transmon = _require_novera()
    return db_transmon.TransmonDevice(
        anharmonicity=device.anharmonicity,
        rabi_max=device.rabi_max,
        t1=device.t1,
        t2_echo=device.t2_echo,
        sample_rate=device.sample_rate,
        static_detuning=device.static_detuning,
        control_error=device.control_error,
    )


def build_pulse(times, omega_x, omega_y, T: float, device: Device = DEFAULT_DEVICE):
    """A Novera pulse dict whose ``to_physical()`` reconstructs *this* ``(T, omega)``.

    See the module docstring's "``to_physical`` incompatibility" section for
    the constant-envelope gauge fix this implements. ``times`` runs over
    ``[0, T]`` (physical ns); ``omega_x``, ``omega_y`` are the physical drive
    in rad/ns on that same grid. ``device`` only supplies ``rabi_max_rate``
    (this project's own :class:`~curve_opt.device.Device`, not a
    ``TransmonDevice`` -- no upstream import needed to build the dict, only
    to consume it).
    """
    times = np.asarray(times, dtype=float)
    omega_x = np.asarray(omega_x, dtype=float)
    omega_y = np.asarray(omega_y, dtype=float)
    peak_budget = T * device.rabi_max_rate
    return {
        "time_over_Tg": times / T,
        "Tg_omega_x": omega_x * T,
        "Tg_omega_y": omega_y * T,
        "Tg_omega_envelope": np.full_like(times, peak_budget),
        "phase_rad": np.arctan2(omega_y, omega_x),
        "Tg_delta": np.zeros_like(times),
    }


# --------------------------------------------------------------------------
# transcribed functions -- thin pass-throughs, one per name in the task brief
# (plus three_level_gate, see the module docstring). Each docstring points at
# the upstream one rather than repeating it.
# --------------------------------------------------------------------------


def three_level_propagation(pulse, device=None, coupling: float = 1.0, stride: int = 1):
    """``db_transmon.three_level_propagation`` -- see its docstring.

    ``device`` defaults to :func:`to_novera_device` of
    :data:`curve_opt.device.DEFAULT_DEVICE` (not upstream's own 3%-error
    default); pass an explicit ``device`` for a bit-for-bit comparison
    against calling ``db_transmon`` directly with the same argument.
    """
    _, db_transmon = _require_novera()
    device = to_novera_device() if device is None else device
    return db_transmon.three_level_propagation(pulse, device, coupling, stride)


def three_level_gate(pulse, epsilon: float = 0.0, device=None, stride: int = 1, calibration=None):
    """``db_transmon.three_level_gate`` -- see its docstring.

    Not in the task brief's function list; added because it is the only
    upstream function that returns the full three-level unitary (see the
    module docstring's "Transcribed vs. added" section).
    """
    _, db_transmon = _require_novera()
    device = to_novera_device() if device is None else device
    return db_transmon.three_level_gate(pulse, epsilon, device, stride, calibration)


def leakage_amplitude(pulse, device=None):
    """``db_transmon.leakage_amplitude`` -- see its docstring.

    Not to be confused with :func:`curve_opt.propagate.leakage_amplitude`,
    this project's own perturbative C4 formula, evaluated on the *broadcast*
    waveform via the design-control chain -- the two are cross-checked
    against each other (up to a documented ``-1j`` prefactor and quadrature
    scheme difference) in ``tests/test_f01_forward_chain.py``, and against
    the exact three-level propagator (this module's :func:`three_level_gate`)
    in ``tests/test_f03_gate.py``'s criterion (2).
    """
    _, db_transmon = _require_novera()
    device = to_novera_device() if device is None else device
    return db_transmon.leakage_amplitude(pulse, device)


def gate_channel(
    pulse, epsilon: float = 0.0, device=None, stride: int = 1,
    dissipation: bool = True, calibration=None,
):
    """``db_transmon.gate_channel`` -- see its docstring."""
    _, db_transmon = _require_novera()
    device = to_novera_device() if device is None else device
    return db_transmon.gate_channel(pulse, epsilon, device, stride, dissipation, calibration)


def subspace_gate_fidelity(channel, target=None):
    """``db_transmon.subspace_gate_fidelity`` -- see its docstring.

    ``target`` defaults to ``db_helpers.U_TARGET`` (``-i X``, the X(pi) gate;
    matches ``curve_opt.propagate.target_x(pi)`` exactly).
    """
    db_helpers, db_transmon = _require_novera()
    target = db_helpers.U_TARGET if target is None else target
    return db_transmon.subspace_gate_fidelity(channel, target)


def block_gate_fidelity(gate, target=None, leakage_corrected: bool = False):
    """``db_transmon.block_gate_fidelity`` -- see its docstring."""
    db_helpers, db_transmon = _require_novera()
    target = db_helpers.U_TARGET if target is None else target
    return db_transmon.block_gate_fidelity(gate, target, leakage_corrected)


def best_frame_phase(gate, target=None, coarse: int = 181, refine: int = 61, leakage_corrected: bool = False):
    """``db_transmon.best_frame_phase`` -- see its docstring."""
    db_helpers, db_transmon = _require_novera()
    target = db_helpers.U_TARGET if target is None else target
    return db_transmon.best_frame_phase(gate, target, coarse, refine, leakage_corrected)


def calibrate(pulse, device=None, stride: int = 4, target=None, scale_window=None, detuning_window=None):
    """``db_transmon.calibrate`` -- see its docstring.

    ``scale_window``/``detuning_window`` default to ``None`` here rather than
    spelling out upstream's own default bounds (``(0.8, 1.25)`` and
    ``(-0.030, 0.030)``): ``-0.030``/``0.030`` collide, digit for digit, with
    the swept ``control_error`` values ``curve_opt.device`` reserves
    (``tests/test_device_hardcoding.py``'s scan flags exactly this kind of
    coincidence -- its own docstring documents an identical case in
    ``recorder.py``). Reading the real defaults off ``db_transmon.calibrate``'s
    own signature keeps this a true pass-through with no literal duplicated
    here at all, and never drifts if upstream's defaults change.
    """
    db_helpers, db_transmon = _require_novera()
    device = to_novera_device() if device is None else device
    target = db_helpers.U_TARGET if target is None else target
    import inspect

    defaults = inspect.signature(db_transmon.calibrate).parameters
    if scale_window is None:
        scale_window = defaults["scale_window"].default
    if detuning_window is None:
        detuning_window = defaults["detuning_window"].default
    return db_transmon.calibrate(pulse, device, stride, target, scale_window, detuning_window)
