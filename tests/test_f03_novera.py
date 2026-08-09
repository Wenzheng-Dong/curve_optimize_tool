"""F03 acceptance: the Novera adapter (criteria (1), (2), (4)).

Everything here needs upstream ``pulse-shape-Novera`` and is marked with
``conftest``'s ``requires_novera``, which skips cleanly (not fails) when the
sibling clone is not importable -- criterion (4). See
``_dev_logs/F03_novera_polar.md`` for a documented infrastructure caveat: in
an isolated git worktree, ``tests/conftest.py``'s sibling-clone path
resolution (anchored to the *test-harness* repo root) can miss the clone
even when it is present on the machine, which is why the numbers quoted in
that dev log were obtained by a scratch script bypassing the broken path
resolution rather than by this file's own (correctly-skipping-here) run.

Criterion (3) -- the residual's vector-ness and full Jacobian rank -- is
upstream-independent and lives in ``tests/test_f03_gate.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from curve_opt import device, gate, novera, propagate
from conftest import requires_novera

DEV = device.DEFAULT_DEVICE
T = DEV.gate_time
DELTA = DEV.delta


def _novera_default_device():
    _, db_transmon = novera._require_novera()
    return db_transmon.TransmonDevice()


def _gaussian_pulse():
    _, db_transmon = novera._require_novera()
    return db_transmon.load_imported_waveforms(["Gaussian"])["Gaussian"]


# --------------------------------------------------------------------------
# Criterion (1): bit-for-bit against db_transmon, same explicit device/target
# on both sides (the adapter's own defaults differ from upstream's -- see
# novera.py's module docstring -- so this passes the same argument to both
# rather than relying on either default).
# --------------------------------------------------------------------------


@requires_novera
def test_leakage_amplitude_matches_db_transmon_exactly():
    _, db_transmon = novera._require_novera()
    up_device = _novera_default_device()
    pulse = _gaussian_pulse()

    direct = db_transmon.leakage_amplitude(pulse, up_device)
    via = novera.leakage_amplitude(pulse, device=up_device)
    assert direct["amplitude"] == via["amplitude"]
    assert direct["population"] == via["population"]


@requires_novera
def test_three_level_propagation_matches_db_transmon_exactly():
    _, db_transmon = novera._require_novera()
    up_device = _novera_default_device()
    pulse = _gaussian_pulse()

    direct = db_transmon.three_level_propagation(pulse, up_device, 1.0, 4)
    via = novera.three_level_propagation(pulse, device=up_device, coupling=1.0, stride=4)
    assert np.array_equal(direct["populations"], via["populations"])
    assert direct["leakage"] == via["leakage"]


@requires_novera
def test_three_level_gate_matches_db_transmon_exactly():
    """Added beyond the task brief's list -- see novera.py's docstring for why."""
    _, db_transmon = novera._require_novera()
    up_device = _novera_default_device()
    pulse = _gaussian_pulse()

    direct = db_transmon.three_level_gate(pulse, 0.0, up_device, 4, None)
    via = novera.three_level_gate(pulse, epsilon=0.0, device=up_device, stride=4, calibration=None)
    assert np.array_equal(direct, via)


@requires_novera
def test_gate_channel_and_fidelities_match_db_transmon_exactly():
    _, db_transmon = novera._require_novera()
    db_helpers, _ = novera._require_novera()
    up_device = _novera_default_device()
    pulse = _gaussian_pulse()

    direct_channel = db_transmon.gate_channel(pulse, 0.0, up_device, 4, True, None)
    via_channel = novera.gate_channel(
        pulse, epsilon=0.0, device=up_device, stride=4, dissipation=True, calibration=None
    )
    assert np.array_equal(direct_channel, via_channel)

    direct_gate = db_transmon.three_level_gate(pulse, 0.0, up_device, 4, None)

    f_direct = db_transmon.subspace_gate_fidelity(direct_channel)
    f_via = novera.subspace_gate_fidelity(via_channel, target=db_helpers.U_TARGET)
    assert f_direct == f_via

    b_direct = db_transmon.block_gate_fidelity(direct_gate, db_helpers.U_TARGET, True)
    b_via = novera.block_gate_fidelity(direct_gate, target=db_helpers.U_TARGET, leakage_corrected=True)
    assert b_direct == b_via

    p_direct = db_transmon.best_frame_phase(direct_gate, db_helpers.U_TARGET)
    p_via = novera.best_frame_phase(direct_gate, target=db_helpers.U_TARGET)
    assert p_direct == p_via


@requires_novera
@pytest.mark.slow
def test_calibrate_matches_db_transmon_exactly():
    """Deterministic (Nelder-Mead on fixed inputs): bit-identical, not just close."""
    _, db_transmon = novera._require_novera()
    db_helpers, _ = novera._require_novera()
    up_device = _novera_default_device()
    pulse = _gaussian_pulse()

    direct = db_transmon.calibrate(pulse, up_device, 8, db_helpers.U_TARGET)
    via = novera.calibrate(pulse, device=up_device, stride=8, target=db_helpers.U_TARGET)
    assert direct == via


# --------------------------------------------------------------------------
# Criterion (2): polar-decomposition leakage vs propagate.leakage_amplitude,
# swept over eta. Lambda comes from curve_opt.propagate (F01, no upstream);
# the exact 3-level unitary comes from novera.three_level_gate via
# build_pulse's fixed-T gauge fix (module docstring). Small etas only
# (<= 0.08, well under this device's own eta = 0.25): a fixed-shape amplitude
# sweep can hit an accidental spectral null of Lambda at a larger, unrelated
# amplitude (measured near eta ~ 0.16 for this particular waveform -- an
# expected feature of comparing one Fourier component's overlap integral,
# not a bug), which would make "agreement" the wrong question to ask there.
# --------------------------------------------------------------------------


def _single_sine_leakage_pair(eta: float, N: int = 2000):
    """(pop_exact via polar decomposition, pop_perturbative via Lambda) for
    Omega_x(t) = eta*|Delta|*sin(pi t/T), Omega_y = 0.
    """
    amplitude = eta * abs(DELTA)
    t = np.linspace(0.0, T, N + 1)
    om_x = amplitude * np.sin(np.pi * t / T)
    om_y = np.zeros_like(t)

    t_mid = (np.arange(N) + 0.5) * (T / N)
    om_x_mid = amplitude * np.sin(np.pi * t_mid / T)
    om_y_mid = np.zeros_like(t_mid)
    lam = np.asarray(propagate.leakage_amplitude_of_samples(om_x_mid, om_y_mid, T, DELTA))
    pop_perturbative = 0.5 * float(np.sum(lam**2))

    pulse = novera.build_pulse(t, om_x, om_y, T)
    u3 = np.asarray(novera.three_level_gate(pulse, epsilon=0.0, stride=1))
    B = u3[:2, :2]
    _, P = gate.polar_decompose(B)
    pop_exact = float(gate.leakage_population(P))
    return pop_exact, pop_perturbative


@requires_novera
def test_polar_leakage_matches_perturbative_lambda_at_small_eta():
    etas = (0.02, 0.04, 0.08)
    deviations = []
    for eta in etas:
        pop_exact, pop_perturbative = _single_sine_leakage_pair(eta)
        ratio = pop_exact / pop_perturbative
        deviations.append(abs(ratio - 1.0))
        # both individually O(eta^2): population an order of magnitude below
        # eta^2 itself at these etas, and growing with eta.
        assert pop_exact == pytest.approx(pop_perturbative, rel=0.01)

    # deviation from the leading-order (perturbative) prediction should not
    # shrink slower than it grows with eta -- i.e. non-decreasing in eta.
    assert deviations[0] <= deviations[1] <= deviations[2]
    assert deviations[-1] < 0.01  # worst case (eta=0.08) still sub-percent


@requires_novera
def test_polar_leakage_and_lambda_population_both_scale_like_eta_squared():
    """Coarser, orthogonal check: population grows roughly as eta^2 for both
    routes over one octave doubling (eta: 0.02 -> 0.04), i.e. a ~4x population
    increase, not e.g. ~2x (eta^1) or ~16x (eta^4).
    """
    pop_exact_lo, pop_pert_lo = _single_sine_leakage_pair(0.02)
    pop_exact_hi, pop_pert_hi = _single_sine_leakage_pair(0.04)
    assert 3.0 < pop_exact_hi / pop_exact_lo < 5.0
    assert 3.0 < pop_pert_hi / pop_pert_lo < 5.0


# --------------------------------------------------------------------------
# Criterion (4): upstream git hash recorded (frozen reference, see the
# module docstring).
# --------------------------------------------------------------------------


def test_upstream_git_hash_is_recorded():
    assert novera.UPSTREAM_GIT_HASH == "59fb6166e105a44064b440865b8fed3415258ad9"
    assert novera.UPSTREAM_GIT_HASH.startswith("59fb616")  # short form quoted elsewhere


def test_novera_functions_skip_cleanly_without_upstream(monkeypatch):
    """Simulates upstream being unimportable: _require_novera must raise
    ImportError with an actionable message (what conftest's requires_novera
    marker turns into a skip), never a bare ImportError/AttributeError from a
    half-completed import.
    """
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name in ("db_helpers", "db_transmon"):
            raise ImportError(f"simulated: {name} not on sys.path")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    with pytest.raises(ImportError, match="pulse-shape-Novera"):
        novera._require_novera()
