"""1qb-DB: deterministic benchmarking of the full-cost arms on a three-level transmon.

The proposal ``1qb_Gaussian-drag-optimization/`` measures **one** gate and reports
``1 - Fbar`` at ``1e-6``--``1e-8``.  Nothing in a laboratory reads a number that small
off a single gate.  1qb-DB is the amplifier: repeat the gate pair ``XX`` for ``n``
cycles and read the survival probability ``P_0(n)``, which turns a per-gate rotation
error ``theta_g`` into a slow, directly visible oscillation ``cos^2(n theta_g / 2)``.
The experiment therefore does not measure fidelity at all -- it measures the *coherent*
part of the error, amplified by ``2n``, which is exactly the part the budget's ``C3``
term claims to remove.

Migrated from ``pulse-shape-Novera/proposals/1qb-DB``
----------------------------------------------------
The protocol, the transmon error model and the readout definitions are Novera's
(``1qb-DB-demo/db_transmon.py``, git ``59fb616``); the device constants are already
shared -- ``curve_opt.device.Device`` was derived from Novera's ``TransmonDevice``.
Three things are deliberately **different**:

1.  **No gauge fixing.**  Novera's waveforms are dimensionless, so it has to choose a
    gauge, and it chooses equal peak drive -- which hands each waveform a different
    ``T_g`` (22 ns for the Gaussian, 100 ns for the composite) and therefore a
    different decoherence envelope.  Every arm here lives at ``T = 50`` ns by
    construction (``device.gate_time``), so the envelope is common and the DB
    comparison isolates the control error with nothing to subtract.
2.  **No amplitude/frequency recalibration by default.**  Novera's waveforms are
    two-level designs replayed on a transmon, so they need a Rabi + Ramsey
    calibration before they are X gates at all.  Arms B--E here are already exact
    three-level ``X(pi)`` up to the solved virtual Z, so the default calibration
    (:data:`CAL_VZ`) turns *only* the free knob, the virtual Z, frozen once at
    ``delta_z = epsilon = 0`` per the proposal's calibration rule.  The full
    three-knob laboratory protocol (:data:`CAL_LAB`) is available and is used as a
    *check*: on arms B--E it must return ``scale ~ 1`` and ``detuning ~ 0``.
3.  **Arm A is not dropped.**  A bare Gaussian is not an X gate on a transmon
    (``1 - Fbar = 1.7e-3``), and showing it saturate the DB trace within a handful of
    cycles is the calibration-free statement of what DRAG buys.

What the experiment can and cannot settle
-----------------------------------------
Novera's transmon notebook found that the two-level amplitude-error closure
*under-predicts* the transmon per-gate error by two orders of magnitude for every
waveform designed to be robust, because ``epsilon`` also rescales the Stark shift
``Omega^2 / 4|Delta|`` and the drive-frequency knob cancels that only at
``epsilon = 0``.  That residual is first order in ``epsilon`` however well the curve
closes.  Arms C/D/E were optimized against the **three-level** model, and arm E
against the AD-exact expected infidelity, so this project has a specific reason to
expect the gap to be smaller here -- but "expect" is the operative word, and this
module measures the per-cycle angle rather than inferring it.  The two are reported
side by side in :func:`angle_table`; a large ratio is a result, not a bug.

★ Which component of the closure the sequence actually amplifies
----------------------------------------------------------------
One DB cycle is ``G G`` with ``G = R_n(theta) E``, ``E = exp(-i eps R . sigma / 2)``
the first-order error rotation and ``R = R(T)`` the closure vector of the amplitude
error curve.  For a ``pi`` rotation about ``n``,

    G^2 = R_n(2 pi) exp(-i eps (R + R') . sigma / 2) + O(eps^2),
    R' = conjugate of R by R_n(pi) = 2 (n . R) n - R,

so ``R + R' = 2 (n . R) n``: **only the component of the closure along the rotation
axis survives the pair, doubled; the transverse part is echoed away.**  A pulse can
therefore have a badly open error curve and still give a flat DB trace, and the
converse -- the sequence is not a general robustness meter, it is a meter for one
projection of it.  :func:`error_curve` reports ``closure_axis`` alongside ``closure``
for exactly this reason, and the first-order DB prediction is
``theta_cycle = 2 eps |n . R(T)|``.

Layout
------
``dblib`` is the only module; the notebook orchestrates and plots.  Arms come from
``proposals/_gatelib`` (so this folder is gate-generic: point :data:`GATE_KEY` at
``"Xpi2"`` / ``"Ypi"`` / ``"Ypi2"`` and it runs unchanged, provided that gate's arms
are solved), and the waveform actually propagated is the ``N_PLAY = 120``-sample
AWG-rate broadcast -- what the hardware emits, not an idealized continuum.

Numerical discipline (``AGENTS.md``, ``CLAUDE.md``)
--------------------------------------------------
``jax_enable_x64`` on (via ``curve_opt``'s ``__init__``); midpoint quadrature
everywhere; ``B @ A`` propagator order; device constants read from
``curve_opt.device``, never spelled here.  The coherent path reuses
``curve_opt.gate.three_level_propagator``; the dissipative path integrates the
Liouvillian assembled from ``curve_opt.qutip_sim``'s operators, and
:func:`check_coherent_limit` verifies the two agree when the dissipator is off --
they are separate implementations and that agreement is a real check, not a tautology.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROPOSALS = HERE.parent
REPO_ROOT = PROPOSALS.parent
for _p in (REPO_ROOT / "src", PROPOSALS / "_gatelib"):   # F08c: no _dev_logs
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import jax  # noqa: E402  (curve_opt.__init__ turns on x64 -- import it first)
import jax.numpy as jnp  # noqa: E402
import jax.scipy.linalg  # noqa: E402
from scipy.linalg import expm  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from curve_opt import device, gate, geometry, propagate, qutip_sim  # noqa: E402

from gatespec import SPECS  # noqa: E402
from gatelib import ARM_COLORS, ARM_NAMES, GateLab  # noqa: E402

__all__ = [
    "ARM_COLORS", "ARM_NAMES", "CAL_LAB", "CAL_VZ", "DB_SIGNAL_THRESHOLD", "DEV",
    "GATE_KEY", "N_CYCLES", "N_PLAY", "T", "TEST_EPS", "DBArm",
    "angle_table", "calibrate", "calibrate_all", "check_calibration_is_free",
    "check_closure_prediction", "check_coherent_limit", "check_grid_convergence",
    "check_expm_backends", "check_independent_integrator",
    "check_sequence_composition", "cycle_angle",
    "cycle_error_vector", "db_configs", "db_readout",
    "db_sequence", "db_traces", "error_curve", "gate_channel", "geometry_table",
    "DETUNING_SCAN_MHZ", "axial_table", "detuning_halfwidth", "detuning_scan",
    "load_arms",
    "plot_db_traces", "plot_detuning_scan", "plot_error_curves",
    "plot_played_waveforms", "plot_signal_vs_epsilon", "print_axial_table",
    "print_db_table", "print_detuning_scan", "print_gate_table",
    "print_geometry_table", "run_all_checks", "signal_vs_epsilon",
    "subspace_gate_fidelity", "three_level_gate",
]

_jax_jit = jax.jit

DEV = device.DEFAULT_DEVICE
T = DEV.gate_time

#: Which gate this folder benchmarks.  ``_gatelib`` makes every other gate a
#: one-line change, but only ``Xpi`` has solved C/D/E runs today.
GATE_KEY = "Xpi"

#: The played grid: one sample per AWG tick, ``2.4 GS/s * 50 ns = 120``.  The DB
#: sequence propagates *this*, not a fine-grid idealization -- the sampling is part
#: of what is being compared (``check_grid_convergence`` measures what it costs).
N_PLAY = int(round(DEV.sample_rate * DEV.gate_time))

#: DB cycles.  One cycle is the gate *pair*, so ``60`` cycles is 120 gates and, at
#: ``T = 50`` ns, ``6`` us of sequence -- a tenth of ``T1``.
N_CYCLES = 60

#: Amplitude errors injected in the DB sequence.  ``0.03`` is the headline (Novera's
#: value, so the two projects' traces are directly comparable); the design
#: assumption behind ``w_3`` is ``device.control_error = 2%`` and is a *different*
#: quantity -- design epsilon shapes the waveform, test epsilon probes it.
TEST_EPS = 0.03

#: A DB signal saturates: once the accumulated rotation error passes ``pi/2`` the
#: trace covers the full range of ``P_0`` and stops ordering the waveforms.  The
#: cycle count at which the signal first crosses this threshold does not saturate.
DB_SIGNAL_THRESHOLD = 0.05

#: Calibration knob sets.  ``CAL_VZ`` is the default and the honest one for arms
#: B--E: the virtual Z is the only knob those waveforms were designed to need.
CAL_VZ = ("frame_phase",)
CAL_LAB = ("scale", "detuning", "frame_phase")

_A3 = np.diag([1.0, np.sqrt(2.0)], 1).astype(complex)
_DRIVE_X3 = np.asarray(qutip_sim.DRIVE_X3.full())
_DRIVE_Y3 = np.asarray(qutip_sim.DRIVE_Y3.full())
_KET0 = np.diag([1.0, 0.0, 0.0]).astype(complex)

_PAULI = np.array([
    [[0.0, 1.0], [1.0, 0.0]],
    [[0.0, -1.0j], [1.0j, 0.0]],
    [[1.0, 0.0], [0.0, -1.0]],
], dtype=complex)


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------


@dataclass
class DBArm:
    """One arm, reduced to what the DB experiment needs.

    ``om_x``/``om_y`` are the **played** waveform at :data:`N_PLAY` cell midpoints,
    in rad/ns -- the object every number in this module is computed on.  Everything
    else is provenance or a cross-reference to the single-gate proposal.
    """

    key: str
    name: str
    layer: str
    om_x: np.ndarray
    om_y: np.ndarray
    T: float = T
    phi_vz_design: float = 0.0
    """The virtual Z the single-gate proposal froze, calibrated on its ``N_DESIGN``
    grid.  The DB calibration re-fits on the played grid; the two must agree to the
    sampling loss, which :func:`check_calibration_is_free` reports."""
    run_id: str | None = None
    source: object | None = None
    """The underlying ``armcore.Input``, kept so the played waveform can
    be rebuilt at any ``N`` (grid-convergence check) without touching disk."""
    device: "device.Device | None" = None
    """The :class:`curve_opt.device.Device` this arm was built and broadcast against
    (N3, additive: ``None`` falls back to the module-level :data:`DEV` wherever this
    is read, so every pre-N3 call site that never set it keeps behaving exactly as
    before). :func:`load_arms` always fills this in from the :class:`GateLab` it
    built the arm with -- reading the module global here instead would silently put
    the *default* device's ``delta``/T1/T2 into a non-default-device DB run, which is
    exactly the bug class this project's device layer exists to rule out."""

    @property
    def n(self) -> int:
        return self.om_x.shape[0]

    @property
    def dt(self) -> float:
        return self.T / self.n

    @property
    def t_mid(self) -> np.ndarray:
        return (np.arange(self.n) + 0.5) * self.dt

    @property
    def peak(self) -> float:
        return float(np.max(np.hypot(self.om_x, self.om_y)))

    def rebuilt(self, n: int) -> "DBArm":
        """The same arm broadcast on a different grid -- the convergence handle."""
        om_x, om_y = self.source.broadcast(n)
        return DBArm(self.key, self.name, self.layer, np.asarray(om_x), np.asarray(om_y),
                     self.T, self.phi_vz_design, self.run_id, self.source, self.device)


@dataclass
class DBLab:
    """The gate under test plus its arms.  Everything else in the module is a function
    of ``(arms, U_target, axis)``, so a different gate is a different :class:`DBLab`."""

    gate_key: str = GATE_KEY
    n_play: int = N_PLAY
    arms: dict = field(default_factory=dict)
    U_target: np.ndarray = None
    axis: np.ndarray = None
    theta: float = 0.0
    lab: GateLab = None

    @property
    def dev(self) -> "device.Device":
        """The device every arm in this lab was built against (``self.lab.DEV``).

        N3: the single place every ``db: DBLab``-taking function should read the
        device from, instead of the module-level :data:`DEV` -- see :func:`load_arms`.
        """
        return self.lab.DEV


def load_arms(gate_key: str = GATE_KEY, n_play: int | None = None,
             dev: "device.Device | None" = None) -> DBLab:
    """Build every available arm and broadcast it onto the AWG grid.

    Arms come from ``_gatelib.GateLab`` -- the same construction the single-gate
    proposal reports, so a number here is comparable term for term with its Table 1
    rather than merely similar.  ``ARM_ORDER`` is decided by which ``_runs/``
    directories exist, so a gate whose C/D/E have not been solved loads A and B alone
    and every function below still runs.

    *dev* is N3's additive device parameter, mirroring ``_gatelib.GateLab``'s own
    ``device=`` (N1): ``None`` (the default) builds against the module-level
    :data:`DEV`, bit for bit as before; passing a non-default
    :class:`curve_opt.device.Device` (e.g. via ``curve_opt.recorder.load_device``)
    builds and characterizes every arm against that device instead, and the
    resulting :class:`DBLab`/:class:`DBArm` objects carry it so downstream functions
    stop reading the module global.

    *n_play* defaults to ``None``, resolved to ``lab.N_EXPORT`` (``sample_rate *
    gate_time`` for *this* device, N1) rather than the module constant :data:`N_PLAY`
    -- identical to :data:`N_PLAY` when *dev* is ``None`` (Novera in particular keeps
    the default ``sample_rate``/``gate_time``, so this makes no numeric difference
    here, but hard-coding the default-device grid size would be wrong the day a
    device config changes either field).
    """
    lab = GateLab(SPECS[gate_key], device=dev)
    n_play = lab.N_EXPORT if n_play is None else n_play
    built = lab.build_arms()
    arms: dict[str, DBArm] = {}
    for key in lab.ARM_ORDER:
        src = built[key]
        om_x, om_y = lab.broadcast_of(src, n_play)
        run_dir = lab.run_dir(key)
        arms[key] = DBArm(
            key=key,
            name=ARM_NAMES.get(key, src.name),
            layer=src.layer,
            om_x=np.asarray(om_x, dtype=float),
            om_y=np.asarray(om_y, dtype=float),
            T=lab.T,
            phi_vz_design=lab.characterize_zero_noise(src)["phi_vz"],
            run_id=run_dir.name if run_dir is not None else None,
            source=src,
            device=lab.DEV,
        )
    return DBLab(gate_key=gate_key, n_play=n_play, arms=arms,
                 U_target=np.asarray(lab.U_TARGET), axis=np.asarray(SPECS[gate_key].axis),
                 theta=float(lab.theta), lab=lab)


# ---------------------------------------------------------------------------
# error-curve geometry (two-level, on the played waveform)
# ---------------------------------------------------------------------------


def error_curve(arm: DBArm, noise: str = "amplitude", axis=None) -> dict:
    """First-order error curve ``R(t)`` of one played waveform.

    An error Hamiltonian ``H_err = eps G(t)`` enters the first Magnus term as
    ``-i eps int U_0^dag G U_0 dt``; writing ``U_0^dag G U_0 = g . sigma / 2`` defines
    ``R(t) = int_0^t g ds``.  ``|R(T)|`` is the first-order error rotation angle per
    unit ``eps`` and ``(1/2) oint R x dR`` its signed area vector, which carries the
    leading second-order term.

    ``noise="amplitude"`` uses ``G = H_c`` -- the multiplicative drive error the DB
    sequence injects.  ``noise="dephasing"`` uses ``G = sigma_z / 2``, the unit-speed
    SCQC curve, and is reported only as a reminder that the two channels are
    different questions: a pulse can close one and not the other.

    ★ ``closure_axis = |n . R(T)|`` is the DB-relevant projection (module docstring):
    the ``XX`` pair echoes the transverse part away, so the first-order per-cycle
    angle is ``2 eps closure_axis``, *not* ``2 eps closure``.

    The ideal propagators come from ``curve_opt.propagate.chain``'s midpoint
    ``U_mid`` -- the project's own two-level chain, midpoint quadrature, not a
    second implementation of it.

    Returns
    -------
    dict
        ``time`` (N+1, cell edges), ``curve`` (N+1, 3), ``closure``,
        ``closure_vector`` (3,), ``closure_axis``, ``area`` (3,), ``length``.
    """
    ch = propagate.chain(jnp.asarray(arm.om_x), jnp.asarray(arm.om_y), arm.T)
    U_mid = np.asarray(ch.U_mid)
    if noise == "amplitude":
        coeff = np.stack([arm.om_x, arm.om_y, np.zeros_like(arm.om_x)], axis=1)
    elif noise == "dephasing":
        coeff = np.zeros((arm.n, 3))
        coeff[:, 2] = 1.0
    else:
        raise ValueError(f"unknown noise channel: {noise!r}")

    # g_b = (1/2) Tr[sigma_b U^dag (c . sigma) U]: the SO(3) adjoint acting on c.
    U_dag = np.conj(np.swapaxes(U_mid, 1, 2))
    operator = np.einsum("na,aij->nij", coeff, _PAULI)
    toggled = np.einsum("nij,njk,nkl->nil", U_dag, operator, U_mid)
    tangent = 0.5 * np.real(np.einsum("aij,nji->na", _PAULI, toggled))

    increments = tangent * arm.dt
    curve = np.vstack([np.zeros(3), np.cumsum(increments, axis=0)])
    mid = 0.5 * (curve[:-1] + curve[1:])
    area = 0.5 * np.sum(np.cross(mid, increments), axis=0)
    closure_vector = curve[-1] - curve[0]
    axis = np.asarray([1.0, 0.0, 0.0]) if axis is None else np.asarray(axis, dtype=float)
    return {
        "time": np.arange(arm.n + 1) * arm.dt,
        "curve": curve,
        "closure": float(np.linalg.norm(closure_vector)),
        "closure_vector": closure_vector,
        "closure_axis": float(abs(np.dot(axis, closure_vector))),
        "area": area,
        "length": float(np.sum(np.linalg.norm(increments, axis=1))),
    }


def geometry_table(db: DBLab, noise: str = "amplitude") -> dict:
    """:func:`error_curve` for every arm, plus the first-order DB prediction."""
    rows = {}
    for key, arm in db.arms.items():
        row = error_curve(arm, noise, db.axis)
        row["predicted_cycle_angle"] = 2.0 * TEST_EPS * row["closure_axis"]
        rows[key] = row
    return rows


def axial_table(db: DBLab, eps: float = TEST_EPS, delta_z: float = None) -> dict:
    """★ Both channels' *axial* closure, side by side -- the table this proposal exists for.

    The ``XX`` echo keeps only the component of an error-rotation axis along ``n``
    (module docstring).  That applies to **every** first-order channel, not just the
    one being injected:

    * amplitude, ``G = H_c``: axial closure ``|n . R_amp(T)|`` [dimensionless],
      per-cycle angle ``2 eps |n . R_amp(T)|``;
    * quasi-static Z, ``G = sigma_z / 2``: axial closure ``|n . R_z(T)|`` [ns],
      per-cycle angle ``2 delta_z |n . R_z(T)|``.

    A single-gate infidelity weights the *full* closure isotropically -- ``C_1`` is
    ``w_1 |r(T)|^2``, ``C_3`` is ``w_3 |A_T|^2`` -- and with
    ``delta_z / 2pi = 0.1`` MHz the ``C_1`` term is small enough that an optimizer can
    leave the Z curve wide open for free.  The sequence does not agree: it amplifies
    the axial part of *both* channels by ``2n``, and a term worth ``5e-5`` per gate is
    worth a full swing of ``P_0`` after 120 of them.

    So this table is the one that says which channel a given arm's DB trace is actually
    reporting on -- and whether the experiment as specified isolates the amplitude
    error at all.  :func:`detuning_scan` measures the same statement with a real knob.
    """
    delta_z = db.dev.static_detuning_rate if delta_z is None else delta_z
    rows = {}
    for key, arm in db.arms.items():
        amp = error_curve(arm, "amplitude", db.axis)
        dep = error_curve(arm, "dephasing", db.axis)
        angle_amp = 2.0 * eps * amp["closure_axis"]
        angle_z = 2.0 * delta_z * dep["closure_axis"]
        rows[key] = {
            "closure_amp": amp["closure"],
            "axial_amp": amp["closure_axis"],
            "angle_amp": angle_amp,
            "closure_z_ns": dep["closure"],
            "axial_z_ns": dep["closure_axis"],
            "angle_z": angle_z,
            "z_over_amp": angle_z / angle_amp if angle_amp > 0 else float("inf"),
        }
    return rows


def print_axial_table(db: DBLab, rows=None, eps: float = TEST_EPS,
                      delta_z: float = None) -> dict:
    """Print :func:`axial_table`."""
    delta_z = db.dev.static_detuning_rate if delta_z is None else delta_z
    rows = rows or axial_table(db, eps, delta_z)
    w = max(len(a.name) for a in db.arms.values()) + 2
    print(f"axial (echo-surviving) closure of both channels, at eps = {eps:.0%} and "
          f"delta_z/2pi = {delta_z / device.TWO_PI * 1e3:.1f} MHz")
    head = (f"{'arm':4s}{'waveform':{w}s}{'|R_amp|':>11s}{'|n.R_amp|':>12s}"
            f"{'angle/cyc':>12s}{'|R_z| [ns]':>12s}{'|n.R_z| [ns]':>14s}"
            f"{'angle/cyc':>12s}{'Z/amp':>9s}")
    print(head)
    print("-" * len(head))
    for key, arm in db.arms.items():
        r = rows[key]
        print(f"{key:4s}{arm.name:{w}s}{r['closure_amp']:11.4f}{r['axial_amp']:12.4f}"
              f"{r['angle_amp']:12.4e}{r['closure_z_ns']:12.4f}{r['axial_z_ns']:14.4f}"
              f"{r['angle_z']:12.4e}{r['z_over_amp']:9.2f}")
    print("Z/amp > 1 means the DB trace is reporting the quasi-static Z channel, not the\n"
          "amplitude error the protocol is nominally about (README section 0);\n"
          "detuning_scan measures that, with the drive frequency as the knob.")
    return rows


#: Deliberate drive-frequency offsets for :func:`detuning_scan`, MHz.  Range: the
#: device's own unknown offset is 0.1 MHz rms, so ``+-0.5`` reaches 5 sigma either way,
#: matching the single-gate proposal's sweep.
#:
#: ★ Spacing: **fine on purpose.**  A 120-gate sequence is an interferometer on the Z
#: channel -- the reference ``P_0`` runs through a full fringe every
#: ``pi / (2 n |n . r(T)|)`` of detuning, which for arms D and E is about 0.35 MHz.  The
#: 7-point grid this started with (``+-0.1, +-0.3, +-0.5``) aliased those fringes
#: completely and drew straight lines through them, which reads as a smooth curve and is
#: not one.  41 points resolve them; the channel integrator is batched
#: (:func:`_expm_batch`), so the whole scan costs seconds.
DETUNING_SCAN_MHZ = np.linspace(-0.5, 0.5, 41)


def detuning_scan(db: DBLab, cycles: int = N_CYCLES, eps: float = TEST_EPS,
                  mhz_grid=None, calibrations=None) -> dict:
    """``P_0(N)`` vs. a deliberately applied drive-frequency offset, full model.

    ★ **The measurable way to ask "which channel is this trace reporting?".**  The
    tempting answer is to rerun with ``delta_z`` deleted from the Hamiltonian and
    compare -- but no instrument has that switch, so the comparison is a statement about
    a model, not about a device, and it does not belong in a results table.  A drive
    frequency, by contrast, *is* a knob.  Detuning it on purpose adds to the unknown
    static offset, and the resulting curve is the arm's Z-channel sensitivity measured
    the way an experiment would measure it.

    Two things come out of the same scan:

    * ``p0_ref`` vs. detuning at ``eps = 0`` -- how fast the sequence loses ``P_0`` to a
      frequency error alone.  A curve that is sharply peaked at the nominal point is an
      arm whose ``|n . r(T)|`` is large, i.e. whose Z-noise closure the ``XX`` echo does
      not remove (:func:`axial_table` predicts the curvature);
    * ``signal`` vs. detuning -- whether the amplitude-error discriminator survives once
      the drive is off resonance, which is what decides if the DB reading at the nominal
      point is about ``eps`` at all.

    Returns ``{"mhz", "cycles", "eps", <arm>: {"p0_ref", "signal", "envelope_loss"}}``,
    each an array over the grid.
    """
    mhz_grid = DETUNING_SCAN_MHZ if mhz_grid is None else np.asarray(mhz_grid, float)
    cals = calibrations or calibrate_all(db)
    out = {"mhz": mhz_grid, "cycles": cycles, "eps": eps}
    for key, arm in db.arms.items():
        p0_ref, signal = [], []
        for mhz in mhz_grid:
            dz = db.dev.static_detuning_rate + device.TWO_PI * float(mhz) * 1e-3
            ref = db_sequence(gate_channel(arm, 0.0, dz, True, cals[key]), cycles)[:, 0]
            worst = 0.0
            for sign in (+1.0, -1.0):
                trace = db_sequence(
                    gate_channel(arm, sign * eps, dz, True, cals[key]), cycles)[:, 0]
                worst = max(worst, float(np.max(np.abs(trace - ref))))
            p0_ref.append(float(ref[-1]))
            signal.append(worst)
        out[key] = {"p0_ref": np.array(p0_ref), "signal": np.array(signal),
                    "envelope_loss": 1.0 - np.array(p0_ref)}
    return out


def detuning_halfwidth(mhz, p0_ref, level: float = 0.5) -> float:
    """Smallest ``|detuning|`` at which the reference ``P_0`` first drops below *level*.

    A single number for "how far off resonance may the drive sit before this sequence
    stops working", linearly interpolated between grid points.  ``nan`` means the
    reference never fell that far anywhere on the scanned range, which is the good case.

    Reported instead of a row of raw samples because the reference curve is a *fringe
    pattern*, not a smooth bump (see :data:`DETUNING_SCAN_MHZ`): a table of samples
    invites reading a straight line through an oscillation, whereas the first crossing
    is well defined however fast the fringes run.
    """
    mhz = np.asarray(mhz, float)
    p0 = np.asarray(p0_ref, float)
    order = np.argsort(np.abs(mhz))
    mhz, p0 = mhz[order], p0[order]
    below = np.flatnonzero(p0 < level)
    if not below.size:
        return float("nan")
    k = int(below[0])
    if k == 0:
        return 0.0
    x0, x1 = abs(mhz[k - 1]), abs(mhz[k])
    y0, y1 = p0[k - 1], p0[k]
    if y0 == y1:
        return float(x1)
    return float(x0 + (x1 - x0) * (y0 - level) / (y0 - y1))


def print_detuning_scan(db: DBLab, scan=None, cycles: int = N_CYCLES,
                        eps: float = TEST_EPS, calibrations=None,
                        level: float = 0.5) -> dict:
    """Print :func:`detuning_scan` as one row per arm.

    Summary numbers rather than the raw grid: at 120 gates the reference ``P_0`` is an
    interference fringe pattern in the detuning, so printing samples of it would invite
    reading a curve through an oscillation.  What is well defined is where it *first*
    fails (:func:`detuning_halfwidth`) and how much the amplitude-error signal moves
    across the scan -- the figure shows the fringes themselves.
    """
    scan = scan or detuning_scan(db, cycles, eps, None, calibrations)
    w = max(len(a.name) for a in db.arms.values()) + 2
    mhz = scan["mhz"]
    nominal = int(np.argmin(np.abs(mhz)))
    print(f"drive-frequency scan over {mhz[0]:+.2f} .. {mhz[-1]:+.2f} MHz in "
          f"{len(mhz)} steps ({scan['cycles']} XX cycles, full model)")
    head = (f"{'arm':4s}{'waveform':{w}s}{'P0 nominal':>12s}{f'df to P0<{level:.1f}':>15s}"
            f"{'signal nom.':>13s}{'signal min':>12s}{'signal max':>12s}")
    print(head)
    print("-" * len(head))
    rows = {}
    for key, arm in db.arms.items():
        s = scan[key]
        hw = detuning_halfwidth(mhz, s["p0_ref"], level)
        rows[key] = hw
        hw_txt = ("  > range" if np.isnan(hw)
                  else "at nominal" if hw == 0.0 else f"{hw:.3f} MHz")
        print(f"{key:4s}{arm.name:{w}s}{s['p0_ref'][nominal]:12.4f}{hw_txt:>15s}"
              f"{s['signal'][nominal]:13.4f}{s['signal'].min():12.4f}"
              f"{s['signal'].max():12.4f}")
    print()
    print(f"P0 nominal     = P0(N) of the eps = 0 reference at the nominal drive frequency;\n"
          f"df to P0<{level:.1f}   = how far the drive may be detuned before the reference "
          f"falls below {level:.1f}\n"
          f"                 ('> range' = never, within the scanned +-{mhz[-1]:.2f} MHz, "
          f"which is the good case;\n"
          f"                  'at nominal' = already below it with no detuning applied "
          f"at all);\n"
          f"signal nom/min/max = the eps = +-{eps:.0%} discriminator at the nominal point "
          f"and across the scan.\n"
          f"A small df is an arm whose Z-noise closure the XX echo does not remove -- "
          f"axial_table's\n|n.R_z| predicts it.  A signal that swings across the scan is "
          f"one whose DB reading is\nnot about eps alone at that working point.")
    scan["halfwidth_mhz"] = rows
    return scan


# ---------------------------------------------------------------------------
# three-level propagation: coherent gate, dissipative channel
# ---------------------------------------------------------------------------


def _wrap(angle: float) -> float:
    """Wrap an angle difference into ``(-pi, pi]`` -- virtual Z is defined mod 2 pi."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _unpack(calibration) -> tuple[float, float, float]:
    """``(amplitude scale, calibration detuning [rad/ns], frame phase [rad])``."""
    cal = calibration or {}
    return (
        float(cal.get("scale", 1.0)),
        device.TWO_PI * float(cal.get("detuning", 0.0)),
        float(cal.get("frame_phase", 0.0)),
    )


def frame_rotation(phase: float) -> np.ndarray:
    """``exp(-i phase N)`` on the ladder -- a virtual Z, free in hardware.

    ★ Sign convention, measured rather than assumed.  The qubit block of this matrix is
    ``diag(1, e^{-i phase}) = e^{-i phase/2} R_z(-phase)``, and ``qutip_sim.fit_phi_vz``
    returns the ``phi_vz`` with ``B = R_z(phi_vz) U_target``.  So

        frame_rotation(f) B = (phase) x R_z(phi_vz - f) U_target,

    and the calibrated frame phase equals ``+phi_vz``, not minus it.
    :func:`check_calibration_is_free` pins the identity; the residual it reports is the
    played-grid (``N_PLAY``) vs. design-grid (``N_DESIGN``) sampling loss, nothing else.
    """
    return np.diag(np.exp(-1j * phase * np.array([0.0, 1.0, 2.0])))


def _frame_super(phase: float) -> np.ndarray:
    """The same rotation as a column-stacking superoperator: ``U* (x) U``."""
    f = frame_rotation(phase)
    return np.kron(f.conj(), f)


def three_level_gate(arm: DBArm, epsilon: float = 0.0, delta_z: float = 0.0,
                     calibration=None) -> np.ndarray:
    """Coherent ``3x3`` propagator of one calibrated gate.

    Delegates to ``curve_opt.gate.three_level_propagator`` -- the repo's validated
    piecewise-constant midpoint chain -- so the coherent limit of the DB channel is
    computed by the same code every other proposal quotes.

    Reads ``arm.device`` (N3), falling back to the module-level :data:`DEV` only if
    the arm was built by hand without one -- every arm :func:`load_arms` produces
    always has it set.
    """
    dev = DEV if arm.device is None else arm.device
    scale, det_rate, frame = _unpack(calibration)
    amp = scale * (1.0 + epsilon)
    U3 = np.asarray(gate.three_level_propagator(
        jnp.asarray(amp * arm.om_x), jnp.asarray(amp * arm.om_y),
        arm.T, dev.delta, delta_z + det_rate))
    return frame_rotation(frame) @ U3


def _liouvillian_parts(delta_z: float, dissipation: bool,
                       dev: "device.Device | None" = None) -> dict:
    """The Liouvillian as a constant plus two control terms.

    ``L`` is linear in ``H`` and ``H`` is linear in ``(Omega_x, Omega_y)``, so one
    gate is three fixed ``9x9`` matrices and two scalars per cell -- which is what
    makes a 6-configuration, 5-arm sweep take seconds.  The Hamiltonian pieces and
    the collapse operators come from ``curve_opt.qutip_sim``, so the dissipative path
    shares its conventions with the repo's independent QuTiP checker.

    *dev* (N3) defaults to the module-level :data:`DEV`, unchanged from before.
    """
    import qutip as qt

    dev = DEV if dev is None else dev
    c_ops = qutip_sim.collapse_operators(dev) if dissipation else []
    bare = qutip_sim.bare_hamiltonian(dev.delta, delta_z)
    return {
        "constant": np.asarray(qt.liouvillian(bare, c_ops).full()),
        "omega_x": np.asarray(qt.liouvillian(qutip_sim.DRIVE_X3).full()),
        "omega_y": np.asarray(qt.liouvillian(qutip_sim.DRIVE_Y3).full()),
    }


def gate_channel(arm: DBArm, epsilon: float = 0.0, delta_z: float = 0.0,
                 dissipation: bool = True, calibration=None) -> np.ndarray:
    """One calibrated gate as a ``9x9`` quantum channel on the three-level state.

    Ordered product of midpoint Liouvillian exponentials on the played grid -- the
    same discretization the coherent propagator uses, so the two are comparable digit
    by digit (:func:`check_coherent_limit`).  Because the result is the *channel*,
    an ``n``-cycle sequence is a matrix power rather than a re-integration.

    ``epsilon`` multiplies the *calibrated* amplitude: it is the unknown residual an
    experiment is left with after calibrating, which is what the DB sequence measures.
    The dissipator and the static detuning are deliberately not scaled by it.

    Returns a column-stacking (QuTiP) vectorization: ``vec(rho') = channel @ vec(rho)``.

    All ``N`` cell exponentials are taken in one batched, jitted call
    (:func:`_expm_batch`) and then reduced in order.  A per-cell ``scipy.linalg.expm``
    loop is ~30x slower here and made the detuning scan the dominant cost of the whole
    notebook; the two agree to ``~1e-15`` and :func:`check_coherent_limit` re-verifies
    the batched path against the independent jax propagator on every run.
    """
    scale, det_rate, frame = _unpack(calibration)
    amp = scale * (1.0 + epsilon)
    parts = _liouvillian_parts(delta_z + det_rate, dissipation, dev=arm.device)
    generators = (parts["constant"]
                  + (amp * arm.om_x)[:, None, None] * parts["omega_x"]
                  + (amp * arm.om_y)[:, None, None] * parts["omega_y"])
    steps = _expm_batch(generators * arm.dt)
    channel = np.eye(9, dtype=complex)
    for step in steps:
        channel = step @ channel  # ★ B @ A: newest on the left
    return _frame_super(frame) @ channel


@_jax_jit
def _expm_batch_jax(generators):
    """``expm`` of a stack of matrices, vectorized.  ``x64`` is on via ``curve_opt``."""
    return jax.vmap(jax.scipy.linalg.expm)(generators)


def _expm_batch(generators: np.ndarray) -> np.ndarray:
    """``(N, 9, 9) -> (N, 9, 9)``, one exponential per cell.

    jax's vmapped ``expm``, jitted once and reused for every channel of every arm.  The
    fallback is the obvious ``scipy`` loop, kept so the module still runs if jax's
    ``expm`` is unavailable -- and used by :func:`check_expm_backends` to confirm the
    two agree, since replacing a per-cell ``scipy`` call with a batched jax one is a
    change of numerical implementation and gets checked like one.
    """
    try:
        return np.asarray(_expm_batch_jax(jnp.asarray(generators)))
    except Exception:  # pragma: no cover -- fallback path
        return np.stack([expm(g) for g in np.asarray(generators)])


def apply_channel(channel: np.ndarray, operator: np.ndarray) -> np.ndarray:
    """Act with a column-stacking superoperator on a ``3x3`` operator."""
    flat = np.asarray(operator, dtype=complex).ravel(order="F")
    return (channel @ flat).reshape(3, 3, order="F")


def subspace_gate_fidelity(channel: np.ndarray, target: np.ndarray) -> float:
    """Leakage-aware average gate fidelity of a three-level channel.

    Nielsen's formula on ``{I, X, Y, Z}`` embedded in three levels, with the output
    projected back onto the qubit block: population that ends in ``|2>`` lowers the
    trace and is charged as error.  Leakage is a loss, not a relabelling.
    """
    total = 0.0
    for p3, p2 in zip(qutip_sim.QUBIT_PAULIS_3, qutip_sim.QUBIT_PAULIS_2):
        out = apply_channel(channel, p3)[:2, :2]
        total += np.trace(target @ p2 @ target.conj().T @ out).real
    return float((total + 4.0) / 12.0)


def block_gate_fidelity(U3: np.ndarray, target: np.ndarray,
                        leakage_corrected: bool = False) -> float:
    """Average gate fidelity of a coherent three-level gate's qubit block.

    ``leakage_corrected=True`` rescales the block to unit average norm first, leaving
    only the rotation error *inside* the qubit subspace.  That is the quantity
    :func:`calibrate` minimizes: a laboratory calibrates amplitude and frequency
    against a signal in the computational subspace, and letting the optimizer see
    leakage instead makes it chase narrow, irreproducible spectral nulls.
    """
    block = np.asarray(U3)[:2, :2]
    if leakage_corrected:
        norm = np.sqrt(0.5 * np.trace(block.conj().T @ block).real)
        block = block / max(norm, 1e-12)
    return qutip_sim.block_gate_fidelity(block, np.asarray(target))


def leakage_of(U3: np.ndarray) -> float:
    """``P_2`` after driving ``|0>`` through one coherent gate."""
    return float(abs(np.asarray(U3)[2, 0]) ** 2)


# ---------------------------------------------------------------------------
# calibration -- once, at delta_z = epsilon = 0, then frozen
# ---------------------------------------------------------------------------


def best_frame_phase(U3, target, coarse: int = 361, refine: int = 81,
                     leakage_corrected: bool = True) -> tuple[float, float]:
    """Best virtual Z for one gate, by a coarse scan plus a local refinement.

    A scan rather than a root find: the objective is periodic and the scan costs a few
    hundred ``2x2`` products, not a re-integration.
    """
    def scan(angles):
        values = [block_gate_fidelity(frame_rotation(a) @ U3, target, leakage_corrected)
                  for a in angles]
        best = int(np.argmax(values))
        return angles[best], values[best]

    grid = np.linspace(-np.pi, np.pi, coarse)
    centre, _ = scan(grid)
    span = grid[1] - grid[0]
    phase, fidelity = scan(np.linspace(centre - span, centre + span, refine))
    return float(phase), float(fidelity)


def calibrate(arm: DBArm, target, knobs=CAL_VZ,
              scale_window=(0.8, 1.25), detuning_window=(-0.030, 0.030)) -> dict:
    """Calibrate one arm against the three-level device, once, and freeze it.

    ★ Calibration protocol (proposal ``§5.3``, the easiest place to cheat by
    accident): the knobs are fitted **once, at** ``delta_z = epsilon = 0``, and held
    for the whole sequence -- that is what a real experiment does.  Re-fitting at
    every noise point hands each arm a free recalibration and flatters whichever arm
    drifts most in phase.  The static detuning is *unknown noise*, so the calibration
    is not allowed to see it either.

    ``knobs``
        :data:`CAL_VZ` (default) turns only the virtual Z: arms B--E are already
        exact three-level ``X(pi)`` by construction, so anything else would be
        re-optimizing the waveform under another name.
        :data:`CAL_LAB` adds the two knobs a laboratory really has -- drive amplitude
        (a Rabi calibration) and drive frequency (which absorbs the Stark shift) --
        bounded, because without bounds the optimizer leaves the design amplitude
        entirely (there is another ``pi`` point near ``1.6x``) and a laboratory Rabi
        calibration does not do that.  Arm A needs this to be an X gate at all; on
        B--E it must come back as a no-op, which is what
        :func:`check_calibration_is_free` asserts.

    No DRAG and no reshaping: the waveform is untouched.  The objective is the
    *leakage-corrected* rotation error, so the knobs fix the rotation and are not
    allowed to buy fidelity by hiding leakage; what is left afterwards is a cost of
    the waveform, not of the calibration.
    """
    target = np.asarray(target)
    cal = {"scale": 1.0, "detuning": 0.0, "frame_phase": 0.0}
    raw = 1.0 - best_frame_phase(three_level_gate(arm), target)[1]

    if "scale" in knobs or "detuning" in knobs:
        def rotation_error(values):
            trial = {"scale": values[0] if "scale" in knobs else 1.0,
                     "detuning": values[1] * 1e-3 if "detuning" in knobs else 0.0}
            U3 = three_level_gate(arm, calibration=trial)
            return 1.0 - best_frame_phase(U3, target)[1]

        result = minimize(
            rotation_error, np.array([1.0, 0.0]), method="Nelder-Mead",
            bounds=(scale_window, tuple(1e3 * v for v in detuning_window)),
            options={"xatol": 1e-7, "fatol": 1e-14, "maxfev": 600},
        )
        cal["scale"] = float(result.x[0]) if "scale" in knobs else 1.0
        cal["detuning"] = float(result.x[1] * 1e-3) if "detuning" in knobs else 0.0

    U3 = three_level_gate(arm, calibration=cal)
    cal["frame_phase"] = best_frame_phase(U3, target)[0] if "frame_phase" in knobs else 0.0

    calibrated = three_level_gate(arm, calibration=cal)
    cal["knobs"] = tuple(knobs)
    cal["raw_rotation_error"] = raw
    cal["rotation_error"] = 1.0 - block_gate_fidelity(calibrated, target, True)
    cal["infidelity"] = 1.0 - block_gate_fidelity(calibrated, target)
    cal["leakage"] = leakage_of(calibrated)
    cal["frame_phase_minus_phi_vz"] = _wrap(cal["frame_phase"] - arm.phi_vz_design)
    cal["at_bound"] = bool(
        ("scale" in knobs
         and min(abs(cal["scale"] - e) for e in scale_window) < 1e-3)
        or ("detuning" in knobs
            and min(abs(cal["detuning"] - e) for e in detuning_window) < 1e-6)
    )
    return cal


def calibrate_all(db: DBLab, knobs=CAL_VZ) -> dict:
    """``{arm key: calibration}``, one fit per arm."""
    return {k: calibrate(a, db.U_target, knobs) for k, a in db.arms.items()}


def print_calibration(db: DBLab, calibrations=None, knobs=CAL_VZ) -> dict:
    """What calibration had to move, and what it could not fix."""
    cals = calibrations or calibrate_all(db, knobs)
    w = max(len(a.name) for a in db.arms.values()) + 2
    print(f"calibration knobs: {', '.join(knobs)}   (fitted once at delta_z = eps = 0, then frozen)")
    head = (f"{'arm':4s}{'waveform':{w}s}{'scale':>9s}{'d_cal[MHz]':>12s}{'virtual Z':>11s}"
            f"{'rot.err before':>16s}{'after':>11s}{'1-F':>11s}{'P2':>11s}{'vz-phi_vz':>11s}")
    print(head)
    print("-" * len(head))
    for key, arm in db.arms.items():
        c = cals[key]
        print(f"{key:4s}{arm.name:{w}s}{c['scale']:9.5f}{'*' if c['at_bound'] else ' '}"
              f"{c['detuning'] * 1e3:11.4f}{c['frame_phase']:11.5f}"
              f"{c['raw_rotation_error']:16.3e}{c['rotation_error']:11.3e}"
              f"{c['infidelity']:11.3e}{c['leakage']:11.3e}"
              f"{c['frame_phase_minus_phi_vz']:11.2e}")
    if any(c["at_bound"] for c in cals.values()):
        print("* on a calibration bound: the knobs cannot make this waveform an X gate "
              "at its design amplitude.")
    print("vz-phi_vz should be ~0 (see frame_rotation's sign note); the residual is the\n"
          "played-grid vs design-grid sampling loss, not a calibration failure.\n"
          "1-F is the *coherent* single-gate residual: decoherence is waveform independent\n"
          f"and enters as the same additive floor {db.dev.decoherence_floor:.4e} for every arm,\n"
          "so it is reported separately rather than folded in.  This is a decomposition of\n"
          "one gate, not a sequence run with a term switched off -- see db_configs.")
    return cals


# ---------------------------------------------------------------------------
# per-gate rotation error -- what the sequence amplifies
# ---------------------------------------------------------------------------


def _su2(M: np.ndarray) -> np.ndarray:
    """Canonical ``SU(2)`` representative of a ``2x2`` unitary (kills the global phase)."""
    return M / np.sqrt(np.linalg.det(M))


def cycle_error_vector(arm: DBArm, epsilon: float, target, calibration=None,
                       delta_z: float = 0.0) -> np.ndarray:
    """Rotation *vector* ``theta n`` of one DB cycle's residual error.

    Polar-decomposes the qubit block so leakage is divided out -- what is left is a
    genuine rotation -- then reads ``(U_target^2)^dag W^2`` as an ``SU(2)`` element
    ``cos(theta/2) I - i sin(theta/2) n . sigma`` and returns ``theta n``.  The sign
    ambiguity ``R ~ -R`` is fixed by taking the representative with
    ``Re tr R >= 0``, i.e. ``theta`` in ``[0, pi]``.

    The *vector* rather than the angle, because that is what composes at first order
    and what a symmetric difference in ``epsilon`` can isolate: the error at
    ``epsilon = 0`` is not zero, so an angle measured at small ``epsilon`` is dominated
    by the residual rather than by the channel under test
    (:func:`check_closure_prediction`).
    """
    U3 = three_level_gate(arm, epsilon, delta_z, calibration)
    W = np.asarray(gate.polar_decompose(jnp.asarray(U3[:2, :2]))[0])
    pair = _su2(W @ W)
    target_pair = _su2(np.asarray(target) @ np.asarray(target))
    residual = target_pair.conj().T @ pair
    if np.real(np.trace(residual)) < 0.0:
        residual = -residual
    half_cos = np.clip(0.5 * np.real(np.trace(residual)), -1.0, 1.0)
    theta = 2.0 * np.arccos(half_cos)
    axis = np.array([-0.5 * np.imag(np.trace(p @ residual)) for p in _PAULI])
    norm = np.linalg.norm(axis)
    return theta * (axis / norm) if norm > 1e-300 else np.zeros(3)


def cycle_angle(arm: DBArm, epsilon: float, target, calibration=None,
                delta_z: float = 0.0) -> float:
    """Magnitude of :func:`cycle_error_vector` -- the angle in ``P_0 = cos^2(n theta/2)``.

    This is the *total* per-cycle rotation error at that ``epsilon``, residual included,
    which is what the sequence actually accumulates and therefore the right thing to put
    in :func:`angle_table`.  It is **not** the right thing to compare against a
    first-order closure at small ``epsilon`` -- see :func:`check_closure_prediction`.
    """
    return float(np.linalg.norm(
        cycle_error_vector(arm, epsilon, target, calibration, delta_z)))


def angle_table(db: DBLab, calibrations=None, eps: float = TEST_EPS,
                geom=None) -> dict:
    """Predicted vs. measured per-cycle rotation error, one row per arm.

    ★ This table is the experiment's whole logic in six columns.  ``closure_axis``
    is a property of the *design* (a two-level curve integral); ``theta measured`` is
    a property of the *device* (a three-level propagation at finite ``eps``).  Their
    ratio is how much of the transmon's error the two-level geometry does not see --
    Novera measured two orders of magnitude for its robust waveforms, because
    ``eps`` also rescales the Stark shift and the frequency knob cancels that only at
    ``eps = 0``.  Arms C/D/E were optimized against the three-level model, so the
    ratio is a genuine open question here rather than a foregone conclusion.
    """
    cals = calibrations or calibrate_all(db)
    geom = geom or geometry_table(db)
    rows = {}
    for key, arm in db.arms.items():
        predicted = 2.0 * eps * geom[key]["closure_axis"]
        measured = {s: cycle_angle(arm, s * eps, db.U_target, cals[key]) for s in (+1, -1)}
        worst = max(measured.values())
        rows[key] = {
            "closure": geom[key]["closure"],
            "closure_axis": geom[key]["closure_axis"],
            "predicted": predicted,
            "measured_plus": measured[+1],
            "measured_minus": measured[-1],
            "measured": worst,
            "ratio": worst / predicted if predicted > 0 else float("inf"),
            "cycles_to_quarter_turn": (np.pi / 2.0) / worst if worst > 0 else float("inf"),
        }
    return rows


def print_gate_table(db: DBLab, rows=None, calibrations=None, eps: float = TEST_EPS) -> dict:
    """Print :func:`angle_table`."""
    rows = rows or angle_table(db, calibrations, eps)
    w = max(len(a.name) for a in db.arms.values()) + 2
    print(f"per-cycle rotation error at eps = {eps:+.1%} (one cycle = the XX pair)")
    head = (f"{'arm':4s}{'waveform':{w}s}{'|R(T)|':>12s}{'|n.R(T)|':>12s}"
            f"{'predicted':>12s}{'measured +':>12s}{'measured -':>12s}{'ratio':>10s}"
            f"{'n to pi/2':>11s}")
    print(head)
    print("-" * len(head))
    for key, arm in db.arms.items():
        r = rows[key]
        print(f"{key:4s}{arm.name:{w}s}{r['closure']:12.4e}{r['closure_axis']:12.4e}"
              f"{r['predicted']:12.4e}{r['measured_plus']:12.4e}{r['measured_minus']:12.4e}"
              f"{r['ratio']:10.2f}{r['cycles_to_quarter_turn']:11.1f}")
    print("predicted = 2 eps |n . R(T)|, the first-order two-level estimate; only the\n"
          "component along the rotation axis survives the XX pair (module docstring).\n"
          "ratio > 1 means the transmon carries error the design curve does not see.")
    return rows


def print_geometry_table(db: DBLab, geom=None, noise: str = "amplitude") -> dict:
    """Print the error-curve invariants that rank the arms before any DB trace."""
    geom = geom or geometry_table(db, noise)
    w = max(len(a.name) for a in db.arms.values()) + 2
    print(f"{noise} error curve of the played waveform (N = {db.n_play} samples)")
    head = (f"{'arm':4s}{'waveform':{w}s}{'length':>11s}{'|R(T)|':>13s}"
            f"{'|n.R(T)|':>13s}{'|area|':>13s}")
    print(head)
    print("-" * len(head))
    for key, arm in db.arms.items():
        g = geom[key]
        print(f"{key:4s}{arm.name:{w}s}{g['length']:11.4f}{g['closure']:13.4e}"
              f"{g['closure_axis']:13.4e}{np.linalg.norm(g['area']):13.4e}")
    return geom


# ---------------------------------------------------------------------------
# the DB sequence
# ---------------------------------------------------------------------------


def db_configs(eps: float = TEST_EPS) -> tuple:
    """The three sequence configurations: ``eps = 0, +eps, -eps``, full model throughout.

    ★ **Every configuration is one an instrument can actually produce.**  An earlier
    draft carried a second family (``coherent``) with the dissipator and the static
    detuning switched off, following Novera, in order to "isolate the control-error
    signal".  That family is gone: no experiment has a knob that turns ``T1`` off, so a
    number read off it is not a measurement and must not sit in a results table next to
    ones that are.  The separation it was doing is done instead by the *difference*
    against the ``eps = 0`` reference, which an experiment gets by simply running the
    sequence without the deliberate amplitude error -- see :func:`db_readout`.

    Attribution -- "is this trace reporting the amplitude error or the Z channel?" --
    is likewise done with a real knob, by sweeping the drive frequency
    (:func:`detuning_scan`), not by deleting a term from the Hamiltonian.
    """
    return (
        ("eps=0", dict(epsilon=0.0)),
        (f"eps=+{eps:.0%}", dict(epsilon=+eps)),
        (f"eps=-{eps:.0%}", dict(epsilon=-eps)),
    )


def _labels(eps: float = TEST_EPS) -> dict:
    return {"reference": "eps=0", "signals": (f"eps=+{eps:.0%}", f"eps=-{eps:.0%}")}


def db_sequence(channel: np.ndarray, cycles: int = N_CYCLES, initial_state=None) -> np.ndarray:
    """Three-level populations after ``n`` cycles, ``n = 0 ... cycles``.

    A cycle is the gate pair, so the sequence is ``(channel @ channel)`` raised to
    ``n``; nothing is re-integrated.  The virtual Z is already inside ``channel`` as a
    left multiplication, and ``(Z U)^n`` is exactly what an instrument does when it
    shifts the phase of every subsequent drive -- so the matrix power is the physical
    sequence, not an approximation of it (:func:`check_sequence_composition`).

    Returns ``(cycles + 1, 3)``; row 0 is the input state, so the trace starts at 1.
    """
    state = _KET0.copy() if initial_state is None else np.asarray(initial_state, complex)
    pair = channel @ channel
    populations = np.empty((cycles + 1, 3))
    populations[0] = np.einsum("ii->i", state).real
    for n in range(cycles):
        state = apply_channel(pair, state)
        populations[n + 1] = np.einsum("ii->i", state).real
    return populations


def db_traces(db: DBLab, cycles: int = N_CYCLES, eps: float = TEST_EPS,
              calibrations=None, configs=None, extra_detuning_mhz: float = 0.0) -> dict:
    """Run the DB sequence for every arm in every configuration, full model throughout.

    ``extra_detuning_mhz`` is a deliberate drive-frequency offset on top of the device's
    unknown ``delta_z`` -- the one knob that lets an experiment probe the Z channel
    without pretending it can switch it off (:func:`detuning_scan`).

    Returns ``{arm key: {"gate_time_ns", "cycles", "elapsed_us", <config>: (n+1, 3)}}``.
    """
    cals = calibrations or calibrate_all(db)
    configs = configs or db_configs(eps)
    delta_z = db.dev.static_detuning_rate + device.TWO_PI * extra_detuning_mhz * 1e-3
    n = np.arange(cycles + 1)
    traces = {}
    for key, arm in db.arms.items():
        record = {"gate_time_ns": arm.T, "cycles": n, "eps": eps,
                  "detuning_mhz": extra_detuning_mhz,
                  "elapsed_us": 2.0 * n * arm.T / 1e3}
        for label, s in configs:
            channel = gate_channel(arm, epsilon=s["epsilon"], delta_z=delta_z,
                                   dissipation=True, calibration=cals[key])
            record[label] = db_sequence(channel, cycles)
        traces[key] = record
    return traces


def _cycles_to_threshold(signals, threshold: float = DB_SIGNAL_THRESHOLD) -> float:
    """Earliest cycle at which any of *signals* reaches *threshold*; ``nan`` if never."""
    crossings = [int(f[0]) for f in
                 (np.flatnonzero(np.asarray(s) >= threshold) for s in signals) if f.size]
    return float(min(crossings)) if crossings else float("nan")


def db_readout(trace: dict) -> dict:
    """The readouts of one arm's DB run.  **Every one is measurable.**

    A transmon DB trace is not one number: decoherence pulls ``P_0`` down whether or not
    the control is robust, so the control-error signal has to be separated from the
    envelope it rides on.  The separation is a *subtraction*, not a model edit -- an
    experiment runs the same sequence once without the deliberate amplitude error and
    differences the two traces, which is exactly what every entry below does.

    ``envelope_loss``
        ``1 - P_0(N)`` of the ``eps = 0`` reference run: the floor the signal must be
        seen against.  Common to every arm here, since every arm has the same ``T``.
    ``signal``
        ``|P_0(N, +-eps) - P_0(N, 0)|`` -- the discriminator, at the end of the
        sequence, worst case over the two error signs.
    ``max_signal``
        ``max_n |P_0(n, +-eps) - P_0(n, 0)|`` -- the same difference maximized over the
        sequence, which is what a fit to the whole trace would key on rather than a
        single endpoint.
    ``cycles_to_signal``
        First cycle at which that difference reaches :data:`DB_SIGNAL_THRESHOLD`.  A DB
        amplitude saturates once the accumulated error passes ``pi/2``; this column
        keeps ordering the arms after the amplitudes no longer do.  ``nan`` means the
        sequence never resolved the error at this length -- for the good arms that is
        the *expected* outcome and is itself the result, not a missing number.
    ``delta_p0``
        Peak-to-peak of the raw trace, i.e. what a plot shows before any subtraction.
    ``leakage``
        ``P_2(N)``.

    ★ Note what these readouts do **not** separate: an ``eps``-response and a
    ``delta_z``-response both show up in the same difference.  Separating *those* takes
    a real knob, and the knob is the drive frequency -- :func:`detuning_scan`.
    """
    eps = trace.get("eps", TEST_EPS)
    lab = _labels(eps)
    n = len(trace["cycles"]) - 1
    reference = trace[lab["reference"]][:, 0]
    diffs = [np.abs(trace[l][:, 0] - reference) for l in lab["signals"]]
    return {
        "gate_time_ns": trace["gate_time_ns"],
        "elapsed_us": float(trace["elapsed_us"][-1]),
        "detuning_mhz": trace.get("detuning_mhz", 0.0),
        "envelope_loss": float(1.0 - reference[-1]),
        "signal": max(float(d[n]) for d in diffs),
        "max_signal": max(float(d.max()) for d in diffs),
        "cycles_to_signal": _cycles_to_threshold(diffs),
        "delta_p0": max(float(np.ptp(trace[l][:, 0])) for l in lab["signals"]),
        "leakage": max(float(trace[l][n, 2]) for l in lab["signals"]),
    }


def print_db_table(db: DBLab, traces: dict) -> dict:
    """Print the DB readouts, one row per arm."""
    readouts = {k: db_readout(t) for k, t in traces.items()}
    n = len(next(iter(traces.values()))["cycles"]) - 1
    eps = next(iter(traces.values())).get("eps", TEST_EPS)
    det = next(iter(traces.values())).get("detuning_mhz", 0.0)
    w = max(len(a.name) for a in db.arms.values()) + 2
    pct = f"{DB_SIGNAL_THRESHOLD:.0%}"

    def cyc(v):
        return "  -" if np.isnan(v) else f"{int(v):3d}"

    extra = f", drive detuned {det:+.2f} MHz" if det else ""
    print(f"DB readout after {n} XX cycles ({2 * n} gates, "
          f"{readouts[next(iter(readouts))]['elapsed_us']:.2f} us) at eps = +-{eps:.0%}"
          f"{extra}")
    head = (f"{'arm':4s}{'waveform':{w}s}{'envelope loss':>15s}{'signal at N':>13s}"
            f"{'max signal':>13s}{f'n@{pct}':>8s}{'Delta P0':>11s}{'P2(N)':>11s}")
    print(head)
    print("-" * len(head))
    for key, arm in db.arms.items():
        r = readouts[key]
        print(f"{key:4s}{arm.name:{w}s}{r['envelope_loss']:15.3e}{r['signal']:13.3e}"
              f"{r['max_signal']:13.3e}{cyc(r['cycles_to_signal']):>8s}"
              f"{r['delta_p0']:11.3e}{r['leakage']:11.2e}")
    print()
    print(f"Full error model in every column -- no term is ever switched off.\n"
          f"envelope loss = 1 - P0(N) of the eps = 0 reference run;\n"
          f"signal at N   = |P0(N,+-eps) - P0(N,0)|, the discriminator;\n"
          f"max signal    = the same difference maximized over the sequence;\n"
          f"n@{pct}         = first cycle at which it reaches {pct} "
          f"('-' = never resolved at this length).")
    return readouts


def signal_vs_epsilon(db: DBLab, eps_grid=None, cycles: int = N_CYCLES,
                      calibrations=None, extra_detuning_mhz: float = 0.0) -> dict:
    """DB signal vs. injected ``eps``, one curve per arm, full model throughout.

    The value plotted is ``max_n |P_0(n, eps) - P_0(n, 0)|`` -- the same difference
    :func:`db_readout` reports, so the figure and the table are the same quantity.

    ★ Keep the negative values: ``C3`` is even in ``eps`` to leading order, so any
    asymmetry between ``+-eps`` reads out the odd higher-order channel, and that
    asymmetry is a result.
    """
    eps_grid = np.array([-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05]) \
        if eps_grid is None else np.asarray(eps_grid, dtype=float)
    cals = calibrations or calibrate_all(db)
    delta_z = db.dev.static_detuning_rate + device.TWO_PI * extra_detuning_mhz * 1e-3
    out = {"eps": eps_grid, "cycles": cycles, "detuning_mhz": extra_detuning_mhz}
    for key, arm in db.arms.items():
        ref = db_sequence(gate_channel(arm, 0.0, delta_z, True, cals[key]), cycles)[:, 0]
        values = []
        for e in eps_grid:
            trace = db_sequence(
                gate_channel(arm, float(e), delta_z, True, cals[key]), cycles)[:, 0]
            values.append(float(np.max(np.abs(trace - ref))))
        out[key] = np.array(values)
    return out


# ---------------------------------------------------------------------------
# validation -- right first, then accurate
# ---------------------------------------------------------------------------


def check_coherent_limit(db: DBLab, calibrations=None) -> dict:
    """(1) With the dissipator off, the channel must be ``U* (x) U`` of the coherent gate.

    Two genuinely separate implementations: ``curve_opt.gate.three_level_propagator``
    (jax, closed-form ``2x2`` cell exponentials lifted to 3 levels) against a
    ``scipy.linalg.expm`` chain on the ``9x9`` Liouvillian assembled through QuTiP.
    Agreement to ``~1e-12`` says the Liouvillian, the vectorization convention and the
    frame rotation are all consistent with the rest of the repository.
    """
    cals = calibrations or calibrate_all(db)
    rows = {}
    for key, arm in db.arms.items():
        U = three_level_gate(arm, 0.0, 0.0, cals[key])
        channel = gate_channel(arm, 0.0, 0.0, dissipation=False, calibration=cals[key])
        rows[key] = float(np.max(np.abs(channel - np.kron(U.conj(), U))))
    return {"max_abs_diff": rows,
            "pass": all(v < 1e-10 for v in rows.values()), "tol": 1e-10}


def check_expm_backends(db: DBLab, calibrations=None) -> dict:
    """(1b) The batched jax ``expm`` must agree with a per-cell ``scipy.linalg.expm`` loop.

    :func:`gate_channel` takes all ``N`` cell exponentials in one vmapped, jitted call
    because the per-cell ``scipy`` loop is ~30x slower and made the detuning scan the
    dominant cost of the notebook.  That is a change of numerical implementation, so it
    gets checked like one rather than assumed: same generators, same order of
    multiplication, two different matrix-exponential routines.
    """
    cals = calibrations or calibrate_all(db)
    dz = db.dev.static_detuning_rate
    rows = {}
    for key, arm in db.arms.items():
        parts = _liouvillian_parts(dz, True, dev=arm.device)
        loop = np.eye(9, dtype=complex)
        for ox, oy in zip(arm.om_x, arm.om_y):
            g = parts["constant"] + ox * parts["omega_x"] + oy * parts["omega_y"]
            loop = expm(g * arm.dt) @ loop
        scale, det_rate, frame = _unpack(cals[key])
        reference = _frame_super(frame) @ loop if scale == 1.0 and det_rate == 0.0 else None
        batched = gate_channel(arm, 0.0, dz, True,
                               {"frame_phase": frame} if reference is not None else cals[key])
        if reference is None:  # a scaled/detuned calibration: redo the loop with it
            parts = _liouvillian_parts(dz + det_rate, True, dev=arm.device)
            loop = np.eye(9, dtype=complex)
            for ox, oy in zip(scale * arm.om_x, scale * arm.om_y):
                g = parts["constant"] + ox * parts["omega_x"] + oy * parts["omega_y"]
                loop = expm(g * arm.dt) @ loop
            reference = _frame_super(frame) @ loop
        rows[key] = float(np.max(np.abs(batched - reference)))
    return {"max_abs_diff": rows,
            "pass": all(v < 1e-12 for v in rows.values()), "tol": 1e-12}


def _concatenated_drive(arm: DBArm, cycles: int, calibration) -> tuple:
    """``2 * cycles`` copies of the waveform, each phase-advanced by the accumulated VZ.

    An instrument does not apply a ``Z`` gate between repetitions; it shifts the phase
    of every subsequent drive.  With ``F = exp(-i f N)`` the virtual Z and ``U(Omega)``
    the gate propagator,

        (F U)^m = F^m  prod_{k=m-1..0} F^{-k} U F^{k},
        F^{-k} U(Omega) F^{k} = U(R_{+k f} Omega),

    since conjugation by ``exp(i phi N)`` rotates the drive quadratures by ``phi``
    (``_gatelib/gatespec.py``).  So gate ``k`` is played with its quadratures advanced
    by ``+k f``, and the leftover ``F^m`` is diagonal -- it cannot move a population, so
    a ``P_0`` readout needs no trailing frame rotation at all.

    ★ The sign was wrong in the first draft (``-k f``), which left
    :func:`check_sequence_composition` failing at ``1e-3`` for every arm with a
    non-zero virtual Z while arm A, whose ``f`` is exactly zero, passed at ``1e-15``.
    That asymmetry is what a composition check is for.
    """
    scale, _det_rate, frame = _unpack(calibration)
    t_all, ox_all, oy_all = [], [], []
    for k in range(2 * cycles):
        c, s = np.cos(frame * k), np.sin(frame * k)
        ox_all.append(scale * (c * arm.om_x - s * arm.om_y))
        oy_all.append(scale * (s * arm.om_x + c * arm.om_y))
        t_all.append(arm.t_mid + k * arm.T)
    return (np.concatenate(t_all), np.concatenate(ox_all), np.concatenate(oy_all))


def check_sequence_composition(db: DBLab, cycles: int = 3, calibrations=None) -> dict:
    """(2a) The matrix power must equal one long product over the concatenated sequence.

    ``db_sequence`` asserts that ``n`` cycles is ``(Lambda^2)^n``.  The independent path
    integrates the ``2n``-gate phase-advanced waveform as a *single* Liouvillian
    product, so the discretization is identical and **only** the composition and the
    frame bookkeeping are under test.  Anything above ``1e-12`` here is a bug in one of
    those two, with nothing else to blame it on.
    """
    cals = calibrations or calibrate_all(db)
    rows = {}
    for key, arm in db.arms.items():
        t_mid, om_x, om_y = _concatenated_drive(arm, cycles, cals[key])
        long_arm = DBArm(key, arm.name, arm.layer, om_x, om_y, 2 * cycles * arm.T,
                         device=arm.device)
        # the trailing virtual Z of the last gate is diagonal and cannot move a
        # population, so the concatenated run needs no frame rotation at all.
        long_channel = gate_channel(long_arm, 0.0, db.dev.static_detuning_rate, True, None)
        p0_long = float(np.einsum("ii->i", apply_channel(long_channel, _KET0)).real[0])

        channel = gate_channel(arm, 0.0, db.dev.static_detuning_rate, True, cals[key])
        p0_power = float(db_sequence(channel, cycles)[-1, 0])
        rows[key] = {"p0_matrix_power": p0_power, "p0_concatenated": p0_long,
                     "abs_diff": abs(p0_power - p0_long)}
    return {"cycles": cycles, "rows": rows,
            "pass": all(r["abs_diff"] < 1e-12 for r in rows.values()), "tol": 1e-12}


def check_independent_integrator(db: DBLab, cycles: int = 2, calibrations=None) -> dict:
    """(2b) An adaptive-ODE reading of the same sequence, by a different integrator.

    ``qutip.mesolve`` on the concatenated waveform: cubic-spline interpolation of the
    sample array and adaptive steps, against this module's piecewise-constant midpoint
    cells.  The two are *not* expected to agree to machine precision -- they differ at
    ``O(dt^2)``, and that difference is the same sampling loss the single-gate proposal
    reports (2.4% on arm C).  What is checked is that the gap stays at the size the
    grid explains, so the tolerance is loose **and the value is reported**, not merely
    thresholded.

    Deliberately at small ``cycles``: ``mesolve`` over 120 gates buys no extra
    information.
    """
    import qutip as qt

    cals = calibrations or calibrate_all(db)
    rows = {}
    for key, arm in db.arms.items():
        t_mid, om_x, om_y = _concatenated_drive(arm, cycles, cals[key])
        H, times = qutip_sim.waveform_hamiltonian(
            t_mid, om_x, om_y, 2 * cycles * arm.T, db.dev.delta, db.dev.static_detuning_rate)
        result = qt.mesolve(H, qt.Qobj(_KET0), times,
                            c_ops=qutip_sim.collapse_operators(db.dev))
        p0_qutip = float(np.real(np.asarray(result.states[-1].full())[0, 0]))

        channel = gate_channel(arm, 0.0, db.dev.static_detuning_rate, True, cals[key])
        p0_power = float(db_sequence(channel, cycles)[-1, 0])
        rows[key] = {"p0_matrix_power": p0_power, "p0_qutip": p0_qutip,
                     "abs_diff": abs(p0_power - p0_qutip)}
    return {"cycles": cycles, "rows": rows,
            "pass": all(r["abs_diff"] < 1e-3 for r in rows.values()), "tol": 1e-3,
            "note": "O(dt^2) grid difference, not an error budget -- read the values."}


def check_grid_convergence(db: DBLab, grids=(120, 240, 600), cycles: int = N_CYCLES,
                           eps: float = TEST_EPS) -> dict:
    """(3) What the AWG sampling costs, as a number rather than an assumption.

    Re-broadcasts every arm at finer grids and re-runs the coherent readout.  ``120``
    is what the hardware plays and is therefore the *reported* configuration; the
    finer grids say how far that is from the continuum the design was solved in.  The
    single-gate proposal measured a 2.4% sampling loss on arm C, so this is expected
    to be visible, not zero.
    """
    rows = {}
    for n_grid in grids:
        sub = DBLab(db.gate_key, n_grid, {k: a.rebuilt(n_grid) for k, a in db.arms.items()},
                    db.U_target, db.axis, db.theta, db.lab)
        cals = calibrate_all(sub)
        traces = db_traces(sub, cycles, eps, cals)
        rows[n_grid] = {k: db_readout(t)["max_signal"] for k, t in traces.items()}
    ref = rows[max(grids)]
    played = rows[min(grids)]
    return {"max_signal": rows,
            "rel_change": {k: abs(played[k] - ref[k]) / max(ref[k], 1e-300) for k in ref}}


def check_calibration_is_free(db: DBLab, scale_tol: float = 5e-3,
                              detuning_tol_mhz: float = 5e-2) -> dict:
    """(4) How much the full laboratory calibration can still find, arm by arm.

    Arms B--E are three-level ``X(pi)`` by construction, so a Rabi and a Ramsey
    calibration should have almost nothing to turn.  The tolerances are deliberately
    loose (``0.5%`` in amplitude, ``50`` kHz in frequency): the question is whether the
    knobs stay inside what a laboratory would call "already calibrated", not whether
    they are bit-zero.  Arm A is excluded from the verdict -- a bare Gaussian is not an
    X gate on a transmon, and that failure *is* the DRAG baseline's reason to exist.

    ★ Read ``infidelity_lab`` against ``infidelity_vz_only`` before concluding
    anything from either.  The three-knob protocol is what a laboratory actually runs,
    and a waveform whose zero-noise residual collapses under it was never limited by
    its shape -- it was limited by a calibration the single-gate comparison did not
    grant it.  Which of the two protocols the headline table should use is a decision
    about what is being claimed, not a numerical detail.
    """
    lab_cals = calibrate_all(db, CAL_LAB)
    vz_cals = calibrate_all(db, CAL_VZ)
    rows = {}
    for key in db.arms:
        c = lab_cals[key]
        rows[key] = {
            "scale_minus_1": c["scale"] - 1.0,
            "detuning_mhz": c["detuning"] * 1e3,
            "frame_phase_minus_phi_vz": c["frame_phase_minus_phi_vz"],
            "infidelity_lab": c["infidelity"],
            "infidelity_vz_only": vz_cals[key]["infidelity"],
            "gain": vz_cals[key]["infidelity"] / max(c["infidelity"], 1e-300),
        }
    designed = [k for k in db.arms if k != "A"]
    return {"rows": rows, "scale_tol": scale_tol, "detuning_tol_mhz": detuning_tol_mhz,
            "pass": all(abs(rows[k]["scale_minus_1"]) < scale_tol
                        and abs(rows[k]["detuning_mhz"]) < detuning_tol_mhz
                        for k in designed),
            "note": "arm A excluded from the verdict; 'gain' = how much a lab "
                    "calibration buys over virtual-Z only."}


def check_closure_prediction(db: DBLab, eps_small: float = 1e-3,
                             calibrations=None) -> dict:
    """(5) The first-order amplitude response must equal ``2 (n . R(T)) n``.

    ★ The residual at ``eps = 0`` is **not** zero, so an angle measured at small ``eps``
    is dominated by it, not by the channel under test -- an earlier version of this
    check compared the two directly and reported ratios of 10--20 that were entirely
    the residual.  The fix is a symmetric difference of the error *vectors*,

        v'(0) ~ [v(+eps) - v(-eps)] / (2 eps),

    which cancels every ``eps``-independent term and every even-order one.  That
    derivative is what the closure predicts, and it is compared as a vector -- the
    direction is as much a claim as the magnitude.

    ★ The transmon caveat, which is why the ratio is reported rather than asserted:
    even at first order the two-level closure misses the part of the response that
    comes from ``eps`` rescaling the Stark shift ``Omega^2/4|Delta|``.  That term is
    also first order, so a ratio above 1 here is the size of the three-level correction
    -- the very quantity Novera found to be ``~10^2`` for its robust waveforms.
    """
    cals = calibrations or calibrate_all(db)
    geom = geometry_table(db)
    rows = {}
    for key, arm in db.arms.items():
        v_plus = cycle_error_vector(arm, +eps_small, db.U_target, cals[key])
        v_minus = cycle_error_vector(arm, -eps_small, db.U_target, cals[key])
        derivative = (v_plus - v_minus) / (2.0 * eps_small)
        predicted_vec = 2.0 * np.dot(db.axis, geom[key]["closure_vector"]) * db.axis
        rows[key] = {
            "predicted_vec": predicted_vec,
            "measured_vec": derivative,
            "predicted": float(np.linalg.norm(predicted_vec)),
            "measured": float(np.linalg.norm(derivative)),
            "ratio": float(np.linalg.norm(derivative) /
                           max(np.linalg.norm(predicted_vec), 1e-300)),
            "cos_angle": float(np.dot(derivative, predicted_vec) /
                               max(np.linalg.norm(derivative) *
                                   np.linalg.norm(predicted_vec), 1e-300)),
        }
    return {"eps": eps_small, "rows": rows,
            "note": "ratio > 1 is the three-level (Stark-rescaling) correction the "
                    "two-level closure cannot see; cos_angle says whether the axis "
                    "prediction survives it."}


def run_all_checks(db: DBLab, calibrations=None, verbose: bool = True) -> dict:
    """Every check above, in order, with a one-line verdict each."""
    cals = calibrations or calibrate_all(db)
    out = {
        "coherent_limit": check_coherent_limit(db, cals),
        "expm_backends": check_expm_backends(db, cals),
        "sequence_composition": check_sequence_composition(db, 3, cals),
        "independent_integrator": check_independent_integrator(db, 2, cals),
        "calibration_is_free": check_calibration_is_free(db),
        "closure_prediction": check_closure_prediction(db, 1e-3, cals),
    }
    if verbose:
        for name, result in out.items():
            verdict = result.get("pass")
            mark = "PASS" if verdict else ("FAIL" if verdict is False else "----")
            print(f"[{mark}] {name}")
            print(f"        {result}")
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib.pyplot as plt
    return plt


#: Fixed colours for the two drive quadratures.  Deliberately *not* the arm colour:
#: ``Omega_x`` and ``Omega_y`` are different physical channels and must read the same
#: way in every panel, so arm identity is carried by the panel title instead.
QUAD_COLORS = {"x": "#0b3d91", "y": "#d1495b"}

#: Grey, dashed, in every trace figure: an ``epsilon = 0`` reference the coloured
#: curves are read *against*.  It is never a result on its own -- see
#: :func:`plot_db_traces`.
BASELINE_COLOR = "0.45"


def plot_played_waveforms(db: DBLab, figsize=(12, 3.4)):
    """Fig. 1 -- the played waveforms, with the peak ceiling drawn in.

    Markers on: the ``120``-sample grid is what the AWG emits, and the sampling is
    part of what is being compared.

    The two quadratures have fixed colours (:data:`QUAD_COLORS`) rather than the arm
    colour, because they are different physical channels and must read the same way in
    every panel: ``Omega_x`` is the drive the gate is built from, ``Omega_y`` is the
    DRAG/Stark correction -- identically zero on the ``planar`` layer (arm A), plotted
    anyway as the flat line that makes B's correction legible as a difference.  The
    dotted grey lines are the hardware ceiling ``+-Omega_max``, not data.

    One legend for the whole figure, below the panels: the panels are cramped between
    the ceiling and the curves, and with fixed quadrature colours a per-panel legend
    would repeat the same three entries five times.
    """
    plt = _plt()
    keys = list(db.arms)
    fig, axes = plt.subplots(1, len(keys), figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        arm = db.arms[key]
        ax.axhline(0.0, lw=0.6, color="0.85", zorder=0)
        ceiling = ax.axhline(+db.dev.rabi_max_rate, ls=":", lw=0.9, color="0.45",
                             label=r"$\pm\Omega_{\max}$ (hardware ceiling)")
        ax.axhline(-db.dev.rabi_max_rate, ls=":", lw=0.9, color="0.45")
        line_x, = ax.plot(arm.t_mid, arm.om_x, ".-", ms=3, lw=1.1,
                          color=QUAD_COLORS["x"], label=r"$\Omega_x$ (drive)")
        line_y, = ax.plot(arm.t_mid, arm.om_y, ".-", ms=3, lw=1.1,
                          color=QUAD_COLORS["y"],
                          label=r"$\Omega_y$ (DRAG/Stark correction)")
        flat = r"  —  $\Omega_y \equiv 0$" if arm.layer == "planar" else ""
        ax.set_title(f"{key} — {arm.name}{flat}", fontsize=9)
        ax.set_xlabel("$t$ [ns]")
    axes[0].set_ylabel(r"$\Omega$ [rad/ns]")
    fig.legend(handles=[line_x, line_y, ceiling], loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def plot_error_curves(db: DBLab, geom=None, noise: str = "amplitude", figsize=(12, 3.4)):
    """Fig. 2 -- the error curves, in 3D and in the three plane projections.

    The panel that ranks the arms *before* any DB trace is simulated -- and, read
    against :func:`angle_table`, the panel that shows where that ranking stops holding
    on a three-level device.

    Every curve starts at the origin (open circle) and ends at the filled square; the
    **gap between the two markers is the closure** ``|R(T)|``, which is the whole point
    of looking at these curves, so both markers are drawn rather than left implicit.
    The arm colours are the same as everywhere else in the notebook.
    """
    plt = _plt()
    geom = geom or geometry_table(db, noise)
    fig = plt.figure(figsize=figsize)
    ax3d = fig.add_subplot(1, 4, 1, projection="3d")
    planes = [(1, 2, "y", "z"), (2, 0, "z", "x"), (0, 1, "x", "y")]
    axes = [fig.add_subplot(1, 4, i + 2) for i in range(3)]
    for key in db.arms:
        c = geom[key]["curve"]
        color = ARM_COLORS.get(key, "k")
        ax3d.plot(c[:, 0], c[:, 1], c[:, 2], color=color, lw=1.2,
                  label=f"{key} ({geom[key]['closure']:.2f})")
        ax3d.scatter(*c[-1], color=color, s=14, marker="s")
        for ax, (i, j, li, lj) in zip(axes, planes):
            ax.plot(c[:, i], c[:, j], color=color, lw=1.2, label=key)
            ax.scatter(c[0, i], c[0, j], facecolors="none", edgecolors=color, s=18)
            ax.scatter(c[-1, i], c[-1, j], color=color, s=16, marker="s")
            ax.set_xlabel(f"$R_{li}$")
            ax.set_ylabel(f"$R_{lj}$")
            ax.set_aspect("equal", adjustable="datalim")
    unit = "ns" if noise == "dephasing" else "rad"
    ax3d.legend(fontsize=6, frameon=False, title=r"arm ($|R(T)|$)", title_fontsize=6)
    fig.suptitle(f"{noise} error curve $R(t)$ [{unit}];  "
                 r"$\circ$ = start (origin), $\blacksquare$ = $R(T)$, "
                 r"so the gap between them is the closure", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_db_traces(db: DBLab, traces: dict, figsize=(12, 3.6)):
    """Fig. 3 -- ``P_0(n)``, one panel per arm.  Full error model, nothing switched off.

    ★ **What the grey dashed line is.**  The ``epsilon = 0`` reference run -- the *same*
    sequence on the *same* device without the deliberate amplitude error, which is
    something an experiment simply does.  It is the line the coloured curves are read
    against, never a result on its own.  On a transmon it is not flat: decoherence, the
    static detuning and accumulated leakage all pull it down whether or not the control
    is robust, and the vertical gap to the coloured curves is the part that is about
    ``epsilon``.

    Every entry in :func:`db_readout` is a difference against that line, for exactly
    that reason.  Reading the raw ``P_0`` instead would credit an arm for decay it
    cannot control.
    """
    plt = _plt()
    eps = next(iter(traces.values())).get("eps", TEST_EPS)
    det = next(iter(traces.values())).get("detuning_mhz", 0.0)
    lab = _labels(eps)
    keys = list(db.arms)
    fig, axes = plt.subplots(1, len(keys), figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        t = traces[key]
        ax.plot(t["cycles"], t[lab["reference"]][:, 0], color=BASELINE_COLOR, lw=1.2,
                ls="--", label=r"$\epsilon=0$ reference")
        for label, ls, sign in zip(lab["signals"], ("-", "-."), ("+", "-")):
            ax.plot(t["cycles"], t[label][:, 0], ls, lw=1.3,
                    color=ARM_COLORS.get(key, "k"),
                    label=f"$\\epsilon={sign}{100 * eps:g}\\%$")
        ax.set_title(f"{key} — {db.arms[key].name}", fontsize=9)
        ax.set_xlabel("XX cycles $n$")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    axes[0].set_ylabel(r"$P_0(n)$")
    # ``\%`` deliberately: a bare ``%`` inside mathtext is a parse error, and
    # ``{eps:.0%}`` would emit one.
    extra = f", drive detuned ${det:+g}$ MHz" if det else ""
    fig.suptitle("full error model throughout ($T_1/T_2$ + $\\delta_z$ + leakage "
                 "+ $\\epsilon$);  grey dashed = the $\\epsilon=0$ reference run "
                 f"the signal is measured against{extra}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_detuning_scan(db: DBLab, scan: dict, figsize=(10, 3.6)):
    """Fig. 4 -- Z-channel attribution with a real knob.

    Left: ``P_0(N)`` of the ``eps = 0`` reference run vs. applied drive detuning.  Right:
    the DB signal at ``+-eps`` vs. the same axis.  Together they say whether the reading
    at the nominal working point is about the amplitude error or about the drive
    frequency -- the question the deleted ``coherent`` configuration used to answer by
    switching off a term no instrument can switch off.
    """
    plt = _plt()
    fig, (ax_ref, ax_sig) = plt.subplots(1, 2, figsize=figsize, sharex=True)
    for key in db.arms:
        color = ARM_COLORS.get(key, "k")
        label = f"{key} — {db.arms[key].name}"
        ax_ref.plot(scan["mhz"], scan[key]["p0_ref"], "-", lw=1.2, color=color,
                    label=label)
        ax_sig.semilogy(scan["mhz"], np.maximum(scan[key]["signal"], 1e-18), "-",
                        lw=1.2, color=color, label=label)
    for ax in (ax_ref, ax_sig):
        ax.axvline(0.0, lw=0.8, ls=":", color="0.5")
        ax.set_xlabel("applied drive detuning [MHz]")
        ax.grid(alpha=0.3, which="both")
    ax_ref.set_ylabel(f"$P_0(N)$ at $\\epsilon=0$,  $N={scan['cycles']}$")
    ax_sig.set_ylabel(f"DB signal at $\\epsilon=\\pm{100 * scan['eps']:g}\\%$")
    ax_ref.set_ylim(-0.02, 1.02)
    handles, labels = ax_ref.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Z-channel attribution by detuning the drive on purpose "
                 "(full model, nothing switched off).  The left panel is a fringe "
                 "pattern, not a bump:\nat 120 gates the sequence is an "
                 "interferometer on the Z channel.", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    return fig


def plot_signal_vs_epsilon(db: DBLab, scan: dict, figsize=(5.6, 3.8)):
    """Fig. 4 -- amplified signal vs. injected error, all arms on one log axis.

    The ordinate is ``max_signal`` of :func:`db_readout` -- already read against the
    ``epsilon = 0`` reference run, so a lower curve means an arm whose trace stays closer
    to that reference under the same injected error.  Curves flatten near the top because
    a DB signal saturates once the accumulated angle passes ``pi/2``.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=figsize)
    for key in db.arms:
        ax.semilogy(100 * scan["eps"], np.maximum(scan[key], 1e-18), "o-", ms=3.5,
                    color=ARM_COLORS.get(key, "k"), label=f"{key} — {db.arms[key].name}")
    ax.axvline(0.0, lw=0.6, color="0.85", zorder=0)
    ax.set_xlabel(r"injected $\epsilon$ [%]")
    ax.set_ylabel(f"DB signal $\\max_n |P_0(n,\\epsilon)-P_0(n,0)|$, "
                  f"$n \\leq {scan['cycles']}$")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# headless driver
# ---------------------------------------------------------------------------


def main(gate_key: str = GATE_KEY, cycles: int = N_CYCLES, eps: float = TEST_EPS,
         checks: bool = True) -> dict:
    """Everything the notebook shows, printed rather than plotted.

    ``conda run -n curve python proposals/1qb-DB/dblib.py`` runs it.
    """
    db = load_arms(gate_key)
    print(f"gate {SPECS[gate_key].name}, arms {tuple(db.arms)}, "
          f"played grid N = {db.n_play}, T = {T} ns\n")
    cals = print_calibration(db)
    print()
    geom = print_geometry_table(db)
    print()
    axial = print_axial_table(db, eps=eps)
    print()
    angles = print_gate_table(db, calibrations=cals, eps=eps)
    print()
    traces = db_traces(db, cycles, eps, cals)
    readouts = print_db_table(db, traces)
    print()
    scan = print_detuning_scan(db, cycles=cycles, eps=eps, calibrations=cals)
    if checks:
        print()
        run_all_checks(db, cals)
    return {"db": db, "calibrations": cals, "geometry": geom, "axial": axial,
            "angles": angles, "traces": traces, "readouts": readouts,
            "detuning_scan": scan}


if __name__ == "__main__":
    main()
