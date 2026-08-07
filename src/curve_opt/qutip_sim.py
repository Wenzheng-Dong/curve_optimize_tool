"""F07: an independent QuTiP validation path for the jax three-level chain.

★ Independence discipline (task brief §2/§4, CLAUDE.md "F07"). This module's
entire reason to exist is that it must **not** call
:func:`curve_opt.gate.three_level_propagator` (or anything built from it) to
compute the actual time evolution -- doing so would be circular verification
of exactly the kind that let the Stark-map sign error (F05-redo) survive
three earlier checks. Every operator here is rebuilt from QuTiP's own ladder
primitives (``qutip.destroy(3)``, ``qutip.num(3)``), and every propagation
uses QuTiP's own solvers (``qutip.propagator``, ``qutip.mesolve``) -- a
genuinely second numerical implementation of the same physics, not a wrapper
around the first.

What *is* reused from the rest of the package: :mod:`curve_opt.device` (the
one source of hardware numbers, ``AGENTS.md`` discipline #10) and
:mod:`curve_opt.budget`/:mod:`curve_opt.parametrization` to *generate* the
input waveform samples this module consumes -- generating a pulse shape is
not propagating it, so that reuse does not create the circularity the brief
warns about; it is exactly the same thing a real AWG would be handed.

★ Deliberately **not** listed in :data:`curve_opt.MODULES`. That tuple is the
core optimize/record pipeline DAG ``tests/test_architecture.py`` enforces
"only recorder.py touches disk" over; this module's CSV export
(:func:`export_waveform_csv`/:func:`load_waveform_csv`) is a first-class F07
deliverable (task brief §2.2, the 2.4 GS/s AWG waveform), not a core-pipeline
side effect, and this module sits outside that DAG entirely (nothing in
``gate``/``budget``/``optimize`` imports it, and it only reaches back into
``device``). Flagged for leader disposition in ``_dev_logs/F07_end_to_end.md``
rather than silently added to or silently exempted from that rule.

Hamiltonian convention (must match ``curve_opt.gate`` bit for bit, checked
independently in ``tests/test_f07_qutip_sim.py`` rather than assumed)
------------------------------------------------------------------------
``H = diag(0, 0, Delta) + (delta_z / 2)(I - 2 N) + Omega_x * DRIVE_X + Omega_y
* DRIVE_Y``, with the ladder operator ``a`` such that ``a|1> = |0>``,
``a|2> = sqrt(2)|1>`` (``curve_opt.gate.three_level_bare``/``DRIVE_X3``/
``DRIVE_Y3``'s docstrings). ``qutip.destroy(3)`` is *exactly* this ``a`` --
QuTiP's own truncated bosonic ladder operator uses the same convention, so
:data:`DRIVE_X3`/:data:`DRIVE_Y3` below are built independently from the
qutip primitive, not imported from :mod:`curve_opt.gate`.

Collapse operators (task brief acceptance ②) match
``pulse-shape-Novera/proposals/1qb-DB/1qb-DB-demo/db_transmon.collapse_operators``
(read-only upstream reference, git hash ``59fb616`` -- consulted for the
physics convention only, not imported): relaxation ``sqrt(Gamma1) * a``
(the ladder structure gives the 2-1 transition twice the 1-0 rate, matching a
real transmon) and pure dephasing ``sqrt(2 Gamma_phi) * N`` with ``N =
qutip.num(3)``, normalized so the two-level limit reproduces ``1/T2 =
1/(2 T1) + Gamma_phi`` -- exactly :attr:`curve_opt.device.Device.gamma_phi`'s
own split.

Average gate fidelity of a leaky, possibly-dissipative channel
-----------------------------------------------------------------
The channel-vs-target average gate fidelity (:func:`channel_infidelity`) uses
Nielsen's operator-sum formula generalized to a channel that need not be
unitary, via a linearity trick: the (Schrodinger-picture) master equation is
linear in its input regardless of positivity, so feeding the *qubit* Pauli
basis ``{I, X, Y, Z}`` (zero-padded to the full 3-level space) directly to
``qutip.mesolve`` as the "initial state" and reading back the top-left 2x2
block of the output at ``t = T`` computes ``E(P)`` -- the channel's own
action on that operator, with leakage counted as loss exactly the way
:func:`curve_opt.gate.leakage_population` does (the amplitude that would have
gone to the padded zero on ``|2>`` never re-enters the extracted block).
Setting ``c_ops = []`` reduces this exactly to ``B P B^dagger`` for the true
coherent unitary block ``B`` (matrix identity, checked directly in
``tests/test_f07_qutip_sim.py``), so :func:`channel_infidelity` at zero
dissipation must agree with :func:`coherent_infidelity` to solver tolerance.

★ Verified discrepancy, flagged for leader disposition (not fixed here,
out of F07's scope): the commonly-quoted "decoherence floor"
``(Gamma1 + Gamma_phi) * T`` (``curve_opt.device.Device.decoherence_floor``,
``_plan_full_cost.md`` sec 0/2.5) is **not** what a rigorous average-gate-
fidelity calculation gives for an idle qubit under these two collapse
operators. Three independent routes -- a hand-derived Kraus-operator
expansion of amplitude damping to ``O(p)``, this module's own
:func:`channel_infidelity` tomography, and ``qutip.average_gate_fidelity``
called directly on ``qutip.propagator``'s output superoperator for a bare
two-level idle channel -- all agree the correct leading-order result is
``(Gamma1 + Gamma_phi) * T / 3``, a factor of 3 below the plan's constant
(measured ``4.165e-4`` vs. the plan's ``1.25e-3``, at this device's T1 = T2echo
= 60 us, T = 50 ns). See ``_dev_logs/F07_end_to_end.md`` sec "Lindblad
discrepancy" for the full derivation and numbers this module's own dev-log
script measured on both table inputs.
"""

from __future__ import annotations

import csv
from typing import NamedTuple

import numpy as np
import qutip as qt
from scipy.optimize import minimize_scalar

from curve_opt.device import DEFAULT_DEVICE, Device

__all__ = [
    "DRIVE_X3",
    "DRIVE_Y3",
    "QUBIT_PAULIS_2",
    "QUBIT_PAULIS_3",
    "bare_hamiltonian",
    "block_gate_fidelity",
    "channel_infidelity",
    "channel_qubit_blocks",
    "coherent_infidelity",
    "coherent_unitary",
    "collapse_operators",
    "export_waveform_csv",
    "fit_phi_vz",
    "load_waveform_csv",
    "rz_matrix",
    "waveform_hamiltonian",
]

# --------------------------------------------------------------------------
# operators -- built from qutip's own primitives, independent of curve_opt.gate
# --------------------------------------------------------------------------

_A3 = qt.destroy(3)
_N3 = qt.num(3)
_I3 = qt.qeye(3)

#: ``H_drive = Omega_x * DRIVE_X3 + Omega_y * DRIVE_Y3``, same convention as
#: ``curve_opt.gate.DRIVE_X3``/``DRIVE_Y3`` (module docstring), rebuilt here
#: from ``qutip.destroy(3)`` rather than imported.
DRIVE_X3 = 0.5 * (_A3 + _A3.dag())
DRIVE_Y3 = 0.5j * (_A3.dag() - _A3)

#: Qubit-subspace Pauli basis ``{I, X, Y, Z}``, plain 2x2 (target-side factor
#: of :func:`channel_infidelity`'s Nielsen sum) and zero-padded to 3 levels
#: (the "generalized state" fed to :func:`channel_qubit_blocks`).
QUBIT_PAULIS_2 = (
    np.eye(2, dtype=complex),
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    np.diag([1.0, -1.0]).astype(complex),
)
QUBIT_PAULIS_3 = tuple(np.pad(p, (0, 1)) for p in QUBIT_PAULIS_2)


def bare_hamiltonian(delta: float, delta_z: float = 0.0) -> qt.Qobj:
    """``diag(0, 0, delta) + (delta_z / 2)(I - 2 N)`` -- matches
    ``curve_opt.gate.three_level_bare`` (module docstring), independently
    reassembled from ``qutip.num(3)`` rather than imported.
    """
    return qt.Qobj(np.diag([0.0, 0.0, delta]).astype(complex)) + (0.5 * delta_z) * (
        _I3 - 2 * _N3
    )


def collapse_operators(device: Device = DEFAULT_DEVICE) -> list[qt.Qobj]:
    """Lindblad collapse operators: relaxation + pure dephasing (module docstring).

    ``[sqrt(Gamma1) * a, sqrt(2 Gamma_phi) * N]``, ``Gamma1``/``Gamma_phi``
    read from *device* (never a bare literal here -- ``AGENTS.md`` discipline
    #10, enforced by ``tests/test_device_hardcoding.py``).
    """
    return [np.sqrt(device.gamma1) * _A3, np.sqrt(2.0 * device.gamma_phi) * _N3]


def waveform_hamiltonian(t_mid, om_x, om_y, T: float, delta: float, delta_z: float = 0.0):
    """Build QuTiP's list-format time-dependent ``H`` plus its ``tlist``.

    ``t_mid``/``om_x``/``om_y`` are cell-midpoint samples (any of this
    project's own grids, e.g. ``curve_opt.budget.broadcast_waveform``'s
    output, or a waveform freshly loaded back from a CSV export). The sine
    basis plus the c1 endpoint condition (``CLAUDE.md`` numerical red line)
    force ``Omega_x(0) = Omega_x(T) = Omega_y(0) = Omega_y(T) = 0`` exactly,
    so the sample array is zero-padded at both ends before handing it to
    QuTiP's own array-format time dependence (cubic-spline interpolated
    internally by ``QobjEvo`` from ``tlist``) -- this covers the full
    ``[0, T]`` domain and never calls into ``curve_opt.propagate``/``gate``
    for the interpolation or the dynamics.
    """
    t_mid = np.asarray(t_mid, dtype=float)
    om_x = np.asarray(om_x, dtype=float)
    om_y = np.asarray(om_y, dtype=float)
    times = np.concatenate([[0.0], t_mid, [T]])
    ox = np.concatenate([[0.0], om_x, [0.0]])
    oy = np.concatenate([[0.0], om_y, [0.0]])
    H = [bare_hamiltonian(delta, delta_z), [DRIVE_X3, ox], [DRIVE_Y3, oy]]
    return H, times


def coherent_unitary(t_mid, om_x, om_y, T: float, delta: float, delta_z: float = 0.0) -> np.ndarray:
    """``U_3(T)``, shape ``(3, 3)``, via ``qutip.propagator`` (QuTiP's own ODE solver).

    No leakage/dissipation channel by construction (``qutip.propagator`` with
    no ``c_ops``): the zero-noise, coherent-only counterpart of
    :func:`curve_opt.gate.three_level_propagator`, computed by an entirely
    different numerical method (adaptive-step ODE integration vs. a
    piecewise-constant-cell ``expm`` chain) -- exactly the second
    implementation acceptance (1) needs.
    """
    H, times = waveform_hamiltonian(t_mid, om_x, om_y, T, delta, delta_z)
    U = qt.propagator(H, times[-1], args={}, tlist=times)
    return np.asarray(U.full())


def rz_matrix(phi: float) -> np.ndarray:
    """``R_z(phi) = exp(-i phi Z / 2)`` -- the free virtual-Z calibration, plain numpy."""
    return np.array([[np.exp(-1j * phi / 2), 0.0], [0.0, np.exp(1j * phi / 2)]], dtype=complex)


def block_gate_fidelity(B: np.ndarray, target: np.ndarray) -> float:
    """Nielsen's average gate fidelity, leakage counted as loss (module docstring).

    Reimplemented locally (same formula as ``_dev_logs/F05_budget_validation.py``'s
    ``block_gate_fidelity`` / upstream ``db_transmon.block_gate_fidelity`` -- a
    textbook formula, not propagator machinery, so re-deriving it here keeps
    this module a genuinely standalone check rather than importing a helper
    from the jax-era analysis scripts).
    """
    total = 0.0
    for pauli in QUBIT_PAULIS_2:
        rotated = target @ pauli @ target.conj().T
        total += np.trace(rotated @ B @ pauli @ B.conj().T).real
    return float((total + 4.0) / 12.0)


def fit_phi_vz(B: np.ndarray, U_target: np.ndarray) -> tuple[float, float]:
    """One-time virtual-Z calibration maximizing zero-noise fidelity: ``(phi_vz, fidelity)``."""

    def neg_fid(phi):
        return -block_gate_fidelity(B, rz_matrix(phi) @ U_target)

    res = minimize_scalar(neg_fid, bounds=(-np.pi, np.pi), method="bounded", options={"xatol": 1e-13})
    return float(res.x), float(-res.fun)


def coherent_infidelity(t_mid, om_x, om_y, T: float, delta: float, U_target, delta_z: float = 0.0):
    """``1 - Fbar`` at one noise point, with ``phi_vz`` calibrated fresh at that point.

    Returns ``(one_minus_fbar, phi_vz, B)``. Calibrating fresh (rather than
    reusing a frozen ``phi_vz``) is correct only for a *single* zero-noise
    evaluation (acceptance (1)) -- a Monte-Carlo sweep must calibrate once at
    zero noise and freeze ``phi_vz`` for every noisy sample (``_dev_logs/
    F05_budget_validation.py``'s ``mc_measure`` docstring explains why); that
    sweep lives in ``_dev_logs/F07_end_to_end.py``, not in this module.
    """
    U3 = coherent_unitary(t_mid, om_x, om_y, T, delta, delta_z)
    B = U3[:2, :2]
    phi_vz, fid = fit_phi_vz(B, U_target)
    return 1.0 - fid, phi_vz, B


def channel_qubit_blocks(t_mid, om_x, om_y, T: float, delta: float, c_ops) -> list[np.ndarray]:
    """``[E(I), E(X), E(Y), E(Z)]``, each a ``(2, 2)`` qubit block (module docstring).

    Feeds each zero-padded qubit Pauli directly to ``qutip.mesolve`` as the
    "initial state" and reads back the top-left block at ``t = T``. Valid
    for any ``c_ops`` (including ``[]``, the coherent-only limit) because the
    master equation is linear in its input regardless of positivity -- this
    is not an approximation, it is the same linear map ``rho -> E(rho)``
    evaluated at a non-physical (Hermitian, not PSD) input.
    """
    H, times = waveform_hamiltonian(t_mid, om_x, om_y, T, delta, 0.0)
    blocks = []
    for P in QUBIT_PAULIS_3:
        result = qt.mesolve(H, qt.Qobj(P), times, c_ops=c_ops)
        rho_T = np.asarray(result.states[-1].full())
        blocks.append(rho_T[:2, :2])
    return blocks


def channel_infidelity(t_mid, om_x, om_y, T: float, delta: float, U_target, phi_vz: float, c_ops) -> float:
    """``1 - Fbar`` of the (possibly dissipative) channel against a *frozen* ``phi_vz`` target.

    ``c_ops = []`` must agree with :func:`coherent_infidelity` at the same
    ``phi_vz`` to solver tolerance (module docstring; checked in
    ``tests/test_f07_qutip_sim.py``).
    """
    target = rz_matrix(phi_vz) @ np.asarray(U_target, dtype=complex)
    blocks = channel_qubit_blocks(t_mid, om_x, om_y, T, delta, c_ops)
    total = 0.0
    for P2, EP in zip(QUBIT_PAULIS_2, blocks):
        rotated = target @ P2 @ target.conj().T
        total += np.trace(rotated @ EP).real
    return 1.0 - (total + 4.0) / 12.0


# --------------------------------------------------------------------------
# waveform export (task brief §2.2): AWG-rate CSV round trip
# --------------------------------------------------------------------------


class WaveformCSV(NamedTuple):
    """One round-tripped waveform: ``(times, omega_x, omega_y)``, all ``ns``/``rad/ns``."""

    times: np.ndarray
    omega_x: np.ndarray
    omega_y: np.ndarray


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


def load_waveform_csv(path) -> WaveformCSV:
    """Read back a file written by :func:`export_waveform_csv`."""
    times, om_x, om_y = [], [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            times.append(float(row[0]))
            om_x.append(float(row[1]))
            om_y.append(float(row[2]))
    return WaveformCSV(np.asarray(times), np.asarray(om_x), np.asarray(om_y))
