"""F05 acceptance support: the ``delta_z``-noisy three-level propagator.

The F05 task brief (``_dev_logs/F05_task_brief.md``) needs
:func:`curve_opt.gate.three_level_propagator` to accept quasi-static Z-noise
(``delta_z``) so the Monte-Carlo validation can inject it directly into a
JAX-differentiable, ``vmap``-able propagator instead of Novera's plain-Python
``expm`` loop. This file is the regression suite for that extension; the
validation study itself (the four inputs, the Monte Carlo, the decomposition
table) is an analysis script (``_dev_logs/F05_budget_validation.py``), not a
pytest module -- see that file and ``_dev_logs/F05_budget_validation.md``.

Three things had to be gotten right and are each checked independently here,
per the numerical red line ("new propagator code must pass a non-commuting
case plus an independent second implementation"):

1. **Backward compatibility.** ``delta_z`` defaults to ``0.0``; every
   pre-F05 caller (F04's hard gate constraint included) must see byte-identical
   output.
2. **The 1/2 factor.** ``_plan_full_cost.md`` section 2.1 literally writes
   ``H_err = delta_z sigma_z`` (no 1/2), but the tex source the C1/C2 weights
   are fit to (``_self_study/_write-up/geometric_robust_leakage_gates.tex``
   Eq. errors/Tdef/magnus) uses ``H_z = delta_z sigma_z / 2``. Using the
   plan's literal (un-halved) line would quadruple the predicted infidelity
   relative to what ``budget.weights_from_device`` computes -- a 4x
   discrepancy sits right at the F05 falsification threshold (2x), so this
   is exactly the kind of silent factor-of-N bug the diagnostic ladder's
   layer 1 exists to catch, and it is caught here *before* any Monte Carlo
   runs, not diagnosed after a false failure.
3. **The level-|2> extension.** ``sigma_z`` embedded in three levels is
   underdetermined by the (purely two-level) tex derivation; a zero-padded
   embedding turned out to be numerically wrong once leakage is present
   (``curve_opt.gate.three_level_bare``'s docstring), so this is
   cross-checked against Novera's independent ``three_level_gate`` directly,
   not just asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from curve_opt import device, gate, propagate
from conftest import requires_novera

DEV = device.DEFAULT_DEVICE
T_PHYS = DEV.gate_time


def _random_3d_waveform(N: int = 2000, amp_scale: float = 0.3):
    """The same random-ish 3D test waveform F04's Novera cross-check test uses."""
    t_mid = (np.arange(N) + 0.5) * (T_PHYS / N)
    amp = amp_scale * DEV.rabi_max_rate
    om_x_mid = amp * np.sin(np.pi * t_mid / T_PHYS)
    om_y_mid = 0.2 * amp * np.cos(np.pi * t_mid / T_PHYS) * np.sin(2 * np.pi * t_mid / T_PHYS)
    return t_mid, om_x_mid, om_y_mid


# --------------------------------------------------------------------------
# (1) backward compatibility: delta_z = 0.0 (the default) is byte-identical
# to the pre-F05 (delta-only) signature.
# --------------------------------------------------------------------------


def test_delta_z_defaults_to_zero_noise_byte_identical_to_pre_f05():
    _, om_x, om_y = _random_3d_waveform()
    U_pre = np.asarray(gate.three_level_propagator(om_x, om_y, T_PHYS, DEV.delta))
    U_explicit_zero = np.asarray(
        gate.three_level_propagator(om_x, om_y, T_PHYS, DEV.delta, delta_z=0.0)
    )
    assert np.array_equal(U_pre, U_explicit_zero)


def test_three_level_bare_defaults_to_zero_noise_byte_identical_to_pre_f05():
    H_pre = np.asarray(gate.three_level_bare(DEV.delta))
    H_explicit_zero = np.asarray(gate.three_level_bare(DEV.delta, delta_z=0.0))
    assert np.array_equal(H_pre, H_explicit_zero)


def test_noisy_propagator_stays_unitary():
    _, om_x, om_y = _random_3d_waveform()
    U = np.asarray(
        gate.three_level_propagator(om_x, om_y, T_PHYS, DEV.delta, delta_z=DEV.static_detuning_rate)
    )
    err = np.max(np.abs(np.conj(U).T @ U - np.eye(3)))
    assert err < 1e-10


# --------------------------------------------------------------------------
# (2) ★ the qubit-block 1/2 factor: independent perturbative cross-check
# against the tex Magnus-expansion formula, ``1 - Fbar = delta_z^2/6
# |r(T)-r(0)|^2``, on a non-closing planar waveform (r(T) != 0). Leakage is
# suppressed with a huge |Delta| so the qubit-block infidelity isolates the
# Z-noise effect the tex formula is about, with no leakage contamination.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("delta_z", [1e-4, 5e-4, 1e-3, 2e-3])
def test_qubit_block_infidelity_matches_the_tex_magnus_formula(delta_z):
    T = 50.0
    N = 4000
    t_mid = (np.arange(N) + 0.5) * (T / N)
    amp = 0.05  # rad/ns -- small, single half-cycle: a genuinely non-closing curve
    om_x = amp * np.sin(np.pi * t_mid / T)
    om_y = np.zeros(N)

    r_T = np.asarray(propagate.chain(om_x, om_y, T).closure)
    predicted = delta_z**2 / 6.0 * float(np.sum(r_T**2))

    huge_delta = -1.0e6  # rad/ns: leakage population is then ~0 (checked below)
    U0 = np.asarray(gate.three_level_propagator(om_x, om_y, T, huge_delta, delta_z=0.0))
    Uz = np.asarray(gate.three_level_propagator(om_x, om_y, T, huge_delta, delta_z=delta_z))
    assert abs(U0[2, 0]) ** 2 < 1e-12  # leakage genuinely negligible at this Delta

    B0, Bz = U0[:2, :2], Uz[:2, :2]
    d = 2
    fbar = (np.abs(np.trace(np.conj(B0).T @ Bz)) ** 2 + d) / (d * (d + 1))
    measured = 1.0 - fbar

    assert measured == pytest.approx(predicted, rel=5e-3)


# --------------------------------------------------------------------------
# (3) ★ the level-|2> extension: independent arbitration against Novera's
# three_level_gate, matched up to a global phase (the only gauge freedom a
# purely diagonal common shift represents) across a decade-plus range of
# delta_z. F04's own zero-noise cross-check ("calibration={'detuning':
# -DEV.static_detuning, ...}" cancels Novera's own baseline injection) is the
# delta_z=0 case here; this test sweeps delta_z on top of that same recipe.
# --------------------------------------------------------------------------


@requires_novera
@pytest.mark.parametrize("scale", [0.0, 0.01, 0.1, 0.5, 1.0, 2.0])
def test_delta_z_embedding_matches_novera_up_to_global_phase(scale):
    from curve_opt import novera

    N = 2000
    t_mid, om_x_mid, om_y_mid = _random_3d_waveform(N=N)
    delta_z = scale * DEV.static_detuning_rate

    U3_jax = np.asarray(
        gate.three_level_propagator(om_x_mid, om_y_mid, T_PHYS, DEV.delta, delta_z=delta_z)
    )

    t_edge = np.linspace(0.0, T_PHYS, N + 1)
    om_x_edge = np.interp(t_edge, t_mid, om_x_mid)
    om_y_edge = np.interp(t_edge, t_mid, om_y_mid)
    pulse = novera.build_pulse(t_edge, om_x_edge, om_y_edge, T_PHYS, device=DEV)
    # Novera's own DETUNING = -NUMBER convention already injects
    # device.static_detuning_rate by default (_calibration_terms); cancel
    # that baseline, then add delta_z (GHz, hence the /2pi) on top -- the
    # same recipe F04's zero-noise test used at delta_z = 0.
    detuning_ghz = delta_z / (2.0 * np.pi) - DEV.static_detuning
    calibration = {"detuning": detuning_ghz, "scale": 1.0, "frame_phase": 0.0}
    U3_nov = np.asarray(
        novera.three_level_gate(pulse, epsilon=0.0, stride=1, calibration=calibration)
    )

    # A diagonal common (identity-proportional) Hamiltonian shift is a pure
    # global phase on the whole propagator -- align it out before comparing
    # (gate.py's module docstring: "up to a global phase" is the physically
    # correct equivalence).
    overlap = np.trace(np.conj(U3_jax).T @ U3_nov)
    phase = overlap / np.abs(overlap)
    dist = np.max(np.abs(U3_jax - U3_nov / phase))
    # ~7e-7 is the delta_z = 0 discretization-only baseline (edge-averaged
    # vs midpoint grids, same caveat F01/F03/F04 documented); this must stay
    # flat across the whole delta_z sweep, not grow with it -- that flatness
    # is what actually distinguishes the correct (proportional) level-|2>
    # extension from the zero-padded one this test caught as wrong (module
    # docstring's "not a free/inconsequential choice" section).
    assert dist < 5e-6
