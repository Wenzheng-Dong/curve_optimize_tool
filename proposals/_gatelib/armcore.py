"""The arm abstraction and the fidelity primitives every proposal notebook runs on.

Provenance
----------
These five objects -- :class:`Input`, :func:`c1_project_and_rescale`,
:func:`fit_phi_vz`, :func:`rz_matrix`, :func:`fidelity_jax` (plus
:func:`block_gate_fidelity`, which ``fit_phi_vz`` minimizes) -- were written for
``_dev_logs/F05_budget_validation.py`` and lived there until F08c.  Four production
notebooks imported them from that file, which made a **frozen dev-log artefact** a
runtime dependency of the proposals: the archive could not be touched without
changing what the notebooks compute, and the notebooks could not be understood
without reading an archive.

They are moved here verbatim -- same expressions, same module-level ``DEV``/``T``/
``N_DESIGN`` bindings, so every number is bit-identical -- and
``F05_budget_validation.py`` now imports them back, keeping exactly one
implementation.  The dev log's own analysis code (``build_naive``, ``build_rcp``,
``mc_measure``, ``run_one_input``, ``main``) stays where it is: it is the record of
what F05 did, and nothing outside that file uses it.

Why here and not in ``src/curve_opt/``
--------------------------------------
``curve_opt`` is the optimizer kernel -- the forward chain, the budget, the solver.
These are *evaluation and presentation* helpers for the proposal layer: an arm's
name/layer/coefficients, the one-time virtual-Z calibration a lab would perform,
Nielsen's average-gate-fidelity formula.  Nothing in ``src/`` needs them, and moving
them there would widen the scope of ``tests/test_device_hardcoding.py`` and blur
what ``curve_opt`` is for.

Note on the ``T`` binding
-------------------------
:class:`Input.broadcast` reads ``self.device.gate_time`` (N1). Before N1 it read the
module-level ``T`` (= ``DEV.gate_time``) unconditionally, which meant a proposal
could never be evaluated against a non-default :class:`curve_opt.device.Device` --
``configs/device_*.json`` had no way to reach this layer. ``Input.__init__`` now
takes an optional ``device`` keyword (``None`` -> the module-level ``DEV``, exactly
the old behaviour), stored as ``self.device``. Every gate in this project still
shares the 50 ns duration on the default device, so nothing here changes unless a
caller explicitly asks for a different device.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import jax.numpy as jnp  # noqa: E402

from curve_opt import basis, budget, device, parametrization, propagate  # noqa: E402

__all__ = ["Input", "block_gate_fidelity", "c1_project_and_rescale", "fidelity_jax",
           "fit_phi_vz", "rz_matrix", "DEV", "T", "N_DESIGN", "QUBIT_PAULIS"]

DEV = device.DEFAULT_DEVICE
T = DEV.gate_time
N_DESIGN = 4000
"""Design/broadcast quadrature grid for all reported (non-MC) quantities."""

I2 = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
QUBIT_PAULIS = (I2, PAULI_X, PAULI_Y, PAULI_Z)

_PAULI4_JAX = jnp.stack(
    [
        jnp.eye(2, dtype=jnp.complex128),
        jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128),
        jnp.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=jnp.complex128),
        jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128),
    ]
)


# ---------------------------------------------------------------------------
# fidelity primitives
# ---------------------------------------------------------------------------


def rz_matrix(phi: float) -> np.ndarray:
    return np.array([[np.exp(-1j * phi / 2), 0.0], [0.0, np.exp(1j * phi / 2)]], dtype=complex)


def block_gate_fidelity(B: np.ndarray, target: np.ndarray) -> float:
    """Nielsen's average-gate-fidelity formula, leakage counted as loss.

    Identical in form to ``db_transmon.block_gate_fidelity(leakage_corrected=False)``
    (reimplemented locally rather than importing Novera, since this is meant
    to run without upstream present -- it is a standard textbook formula, not
    a piece of Novera-specific physics). ``B`` is the *raw* qubit block (not
    renormalized), so population lost to |2> lowers the fidelity exactly as
    an experiment would measure it.
    """
    total = 0.0
    for pauli in QUBIT_PAULIS:
        small = pauli
        rotated = target @ small @ target.conj().T
        total += np.trace(rotated @ B @ small @ B.conj().T).real
    return float((total + 4.0) / 12.0)


def fidelity_jax(B, target) -> jnp.ndarray:
    """jax mirror of :func:`block_gate_fidelity` for one (2, 2) block, vmap/jit-safe.

    ``jax.vmap`` is applied by the *caller* to the function that builds ``B``
    from a noise sample, not to this single-sample formula -- so ``B``/``target``
    are always plain (2, 2) here, matching :func:`block_gate_fidelity`'s numpy
    version term for term (same four-Pauli sum, unrolled -- only 4 terms, cheap
    to trace).
    """
    total = 0.0
    for i in range(4):
        pauli = _PAULI4_JAX[i]
        rotated = target @ pauli @ jnp.conj(target).T
        inner = B @ pauli @ jnp.conj(B).T
        total = total + jnp.real(jnp.trace(rotated @ inner))
    return (total + 4.0) / 12.0


def fit_phi_vz(B: np.ndarray, U_target: np.ndarray) -> tuple[float, float]:
    """One-time virtual-Z calibration: the phi_vz maximizing zero-noise fidelity.

    Mirrors what a real experiment does once, at the design point, then
    freezes for every later (noisy) shot -- phi_vz is a free hardware
    calibration knob, not something re-fit per noise sample.
    """

    def neg_fid(phi):
        return -block_gate_fidelity(B, rz_matrix(phi) @ U_target)

    res = minimize_scalar(neg_fid, bounds=(-np.pi, np.pi), method="bounded",
                          options={"xatol": 1e-13})
    return float(res.x), float(-res.fun)


def c1_project_and_rescale(a: np.ndarray, T: float, theta: float) -> np.ndarray:
    """Minimum-norm projection onto the c1 endpoint subspace, then rescaled to *theta*.

    Every input has to satisfy the c1 endpoint condition, imposed by minimum-norm
    projection. The min-norm correction (``a - A^T (A A^T)^-1 A a`` with ``A``
    stacking ``basis.c1_row(M, T, 'start'/'end')``) generally perturbs the gate
    angle slightly (both are linear functionals of ``a`` but not orthogonal to each
    other); rescaling by ``theta / gate_angle(a_proj)`` restores the exact
    winding-branch angle without reintroducing any c1 violation, since both
    ``c1_row . a`` and ``gate_row . a`` are linear (homogeneous) in ``a``.
    """
    a = np.asarray(a, dtype=float)
    M = a.shape[0]
    A = np.stack([basis.c1_row(M, T, "start"), basis.c1_row(M, T, "end")])
    correction = A.T @ np.linalg.solve(A @ A.T, A @ a)
    a_proj = a - correction
    return a_proj * (float(theta) / basis.gate_angle(a_proj, T))


# ---------------------------------------------------------------------------
# the arm
# ---------------------------------------------------------------------------


class Input:
    """One reference waveform: a name, a control layer, and its design coefficients."""

    def __init__(self, name: str, layer: str, a: np.ndarray, c: np.ndarray,
                 Phi_0: float, note: str, device: device.Device | None = None):
        self.name = name
        self.layer = layer
        self.a = np.asarray(a, dtype=float)
        self.c = np.asarray(c, dtype=float)
        self.Phi_0 = float(Phi_0)
        self.note = note
        self.device = DEV if device is None else device

    @property
    def M(self) -> int:
        return self.a.shape[0]

    def broadcast(self, N: int = N_DESIGN):
        """The *actually played* waveform -- the only thing G/C4/the MC ever see."""
        Tg = self.device.gate_time
        if self.layer == "planar":
            return parametrization.planar_waveform(self.a, Tg, N)
        if self.layer == "planar_drag":
            return parametrization.planar_drag_waveform(self.a, Tg, self.device.delta, N)
        if self.layer == "general":
            return parametrization.general_waveform(self.a, self.c, self.Phi_0, Tg,
                                                    self.device.delta, N)
        raise ValueError(self.layer)

    def predicted_terms(self, N: int = N_DESIGN) -> dict:
        """C1..C4, each on the object its own derivation names.

        ★ Project red line: C1/C2/C3 on the **design curve**, C4 and the gate on the
        **broadcast (played)** waveform.

        C1/C2/C3 come from ``budget.design_chain`` regardless of layer (the
        design curve is layer-independent by construction). C4 does **not**
        come from ``budget.budget_terms`` for the ``planar`` layer: that
        helper always applies the DRAG/Stark readout map
        (``budget.broadcast_chain`` calls ``parametrization.design_curve`` +
        ``readout_waveform`` unconditionally), which would silently predict
        the *DRAG-corrected* leakage for a waveform that is not actually
        DRAG-corrected when played. So C4 is computed here directly on
        :meth:`broadcast`'s actual output for every layer, which agrees with
        ``budget.budget_terms``'s C4 bit for bit on the ``planar_drag``/``general``
        layers (both call the same readout map) and only differs -- correctly --
        on ``planar``.
        """
        Tg = self.device.gate_time
        w = budget.weights_from_device(self.device)
        dchain = budget.design_chain(self.a, self.c, self.Phi_0, Tg, N)
        r_T = np.asarray(dchain.closure)
        A_r = 0.5 * np.asarray(dchain.area)
        A_T = np.asarray(propagate.tantrix_area(dchain))
        c1 = w.c1 * float(np.sum(r_T**2))
        c2 = w.c2 * float(np.sum(A_r**2))
        c3 = w.c3 * float(np.sum(A_T**2))

        om_x, om_y = self.broadcast(N)
        bchain = propagate.chain(om_x, om_y, Tg)
        Lambda = np.asarray(propagate.leakage_amplitude(bchain, self.device.delta))
        c4 = w.c4 * float(np.sum(Lambda**2))
        return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "total": c1 + c2 + c3 + c4}
