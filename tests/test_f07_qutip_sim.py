"""F07 acceptance: :mod:`curve_opt.qutip_sim` is a genuinely independent path.

Five checks, one per load-bearing claim of the module docstring:

1. the static Hamiltonian *convention* (not the propagation) matches
   ``curve_opt.gate``'s -- catches the BUILDER_BRIEF-warned convention traps
   before any dynamics is even run;
2. the zero-noise propagated infidelity matches ``curve_opt.gate.
   three_level_propagator`` (task brief acceptance (1)), with grid refinement;
3. the Lindblad-tomography channel formula reduces to the coherent Nielsen
   formula at zero dissipation (the linearity trick's own correctness);
4. the measured Lindblad-only increment on an idle 2-level toy matches
   ``qutip.average_gate_fidelity`` computed via an entirely different route
   (regression-pins the ``(Gamma1+Gamma_phi)T/3`` finding flagged for leader);
5. the CSV export round-trips exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import qutip as qt

from curve_opt import gate, geometry, propagate, qutip_sim
from curve_opt.device import DEFAULT_DEVICE

DEV = DEFAULT_DEVICE
T = DEV.gate_time
THETA = np.pi
U_TARGET = np.asarray(propagate.target_x(THETA))


def _gaussian_pulse(N: int):
    """A plain (un-DRAG-corrected) truncated Gaussian X(pi) -- a cheap, genuinely
    non-trivial waveform, independent of any project ansatz machinery."""
    t_mid, _ = geometry.midpoint_grid(T, N)
    sigma = T / 6.0
    shape = np.exp(-0.5 * ((t_mid - T / 2.0) / sigma) ** 2)
    om_x = shape * (THETA / np.trapezoid(shape, t_mid))
    return t_mid, om_x, np.zeros_like(om_x)


def test_bare_hamiltonian_matches_gate_convention():
    """Static-operator convention check only (module docstring caveat): the two
    Hamiltonians are compared as matrices, no propagation involved here."""
    delta = DEV.delta
    delta_z = DEV.static_detuning_rate
    h_qutip = np.asarray(qutip_sim.bare_hamiltonian(delta, delta_z).full())
    h_gate = np.asarray(gate.three_level_bare(delta, delta_z))
    np.testing.assert_allclose(h_qutip, h_gate, atol=1e-14)

    dx_qutip = np.asarray(qutip_sim.DRIVE_X3.full())
    dy_qutip = np.asarray(qutip_sim.DRIVE_Y3.full())
    np.testing.assert_allclose(dx_qutip, np.asarray(gate.DRIVE_X3), atol=1e-14)
    np.testing.assert_allclose(dy_qutip, np.asarray(gate.DRIVE_Y3), atol=1e-14)


@pytest.mark.parametrize("N", [100, 400])
def test_coherent_infidelity_matches_jax_chain_zero_noise(N):
    """Acceptance (1): QuTiP's own propagator vs. jax's, zero noise, converging with N."""
    t_mid, om_x, om_y = _gaussian_pulse(N)

    U3_jax = np.asarray(gate.three_level_propagator(om_x, om_y, T, DEV.delta))
    B_jax = U3_jax[:2, :2]
    phi_jax, fid_jax = qutip_sim.fit_phi_vz(B_jax, U_TARGET)
    infid_jax = 1.0 - fid_jax

    infid_qutip, phi_qutip, _ = qutip_sim.coherent_infidelity(t_mid, om_x, om_y, T, DEV.delta, U_TARGET)

    assert phi_qutip == pytest.approx(phi_jax, abs=1e-6)
    rel_diff = abs(infid_qutip - infid_jax) / infid_jax
    assert rel_diff < 5e-3, f"N={N}: jax={infid_jax:.6e} qutip={infid_qutip:.6e} reldiff={rel_diff:.3e}"


def test_channel_infidelity_reduces_to_coherent_at_zero_dissipation():
    """Module docstring's linearity claim: c_ops=[] tomography == Nielsen formula on the true B."""
    t_mid, om_x, om_y = _gaussian_pulse(150)
    infid_coherent, phi_vz, _ = qutip_sim.coherent_infidelity(t_mid, om_x, om_y, T, DEV.delta, U_TARGET)
    infid_channel = qutip_sim.channel_infidelity(t_mid, om_x, om_y, T, DEV.delta, U_TARGET, phi_vz, c_ops=[])
    assert infid_channel == pytest.approx(infid_coherent, rel=1e-3)


def test_idle_decoherence_increment_matches_qutip_builtin_average_gate_fidelity():
    """Cross-check the (Gamma1+Gamma_phi)T/3 finding against a wholly separate qutip route.

    ``qutip.average_gate_fidelity`` on ``qutip.propagator``'s output superoperator
    for a bare two-level idle channel (no drive, no third level at all) --
    the simplest possible independent computation of the same physical
    quantity, used only to pin the number this module's own tomography path
    (:func:`curve_opt.qutip_sim.channel_qubit_blocks`) reports for the driven
    3-level case (see the module docstring / dev log for the full comparison).
    """
    sm = qt.sigmam()
    n_qubit = qt.Qobj(np.diag([0.0, 1.0]).astype(complex))
    c_ops = [np.sqrt(DEV.gamma1) * sm, np.sqrt(2.0 * DEV.gamma_phi) * n_qubit]
    H0 = qt.qzero(2)
    U_super = qt.propagator(H0, T, c_ops=c_ops)
    fid = qt.average_gate_fidelity(U_super, target=qt.qeye(2))
    infid_builtin = 1.0 - fid

    expected = DEV.decoherence_floor / 3.0
    assert infid_builtin == pytest.approx(expected, rel=5e-3)
    # and it is *not* the plan's literal (Gamma1+Gamma_phi)*T constant:
    assert infid_builtin < DEV.decoherence_floor / 2.0


def test_csv_export_round_trip(tmp_path):
    t_mid, om_x, om_y = _gaussian_pulse(20)
    path = tmp_path / "waveform.csv"
    qutip_sim.export_waveform_csv(path, t_mid, om_x, om_y)
    loaded = qutip_sim.load_waveform_csv(path)
    np.testing.assert_allclose(loaded.times, t_mid, rtol=1e-10)
    np.testing.assert_allclose(loaded.omega_x, om_x, rtol=1e-10)
    np.testing.assert_allclose(loaded.omega_y, om_y, rtol=1e-10)
