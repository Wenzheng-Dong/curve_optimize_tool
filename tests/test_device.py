"""F01 acceptance: the hardware-parameter singleton and its derived quantities.

Every derived property gets its own assertion against the number
``_plan_full_cost.md`` §3.2 / ``_dev_logs/F00_plan_v5.md`` §4.3 quotes, so a
future edit to ``device.py`` cannot silently drift from the frozen values.
"""

from __future__ import annotations

import math

import pytest

from curve_opt import device as device_mod
from curve_opt.device import DEFAULT_DEVICE, Device


def test_default_device_matches_the_frozen_plan_values():
    d = DEFAULT_DEVICE
    assert d.anharmonicity == pytest.approx(-0.200)
    assert d.rabi_max == pytest.approx(0.050)
    assert d.t1 == pytest.approx(60_000.0)
    assert d.t2_echo == pytest.approx(60_000.0)
    assert d.static_detuning == pytest.approx(1.0e-4)
    assert d.control_error == pytest.approx(0.02)
    assert d.sample_rate == pytest.approx(2.4)
    assert d.gate_time == pytest.approx(50.0)
    assert d.n_modes == 20


def test_device_is_frozen():
    with pytest.raises(Exception):
        DEFAULT_DEVICE.gate_time = 100.0


def test_angular_rate_properties():
    d = DEFAULT_DEVICE
    assert d.anharmonicity_rate == pytest.approx(2.0 * math.pi * -0.200)
    assert d.rabi_max_rate == pytest.approx(2.0 * math.pi * 0.050)
    assert d.static_detuning_rate == pytest.approx(2.0 * math.pi * 1.0e-4)
    assert d.delta == pytest.approx(d.anharmonicity_rate)
    assert d.delta < 0.0  # transmon: alpha is negative


def test_eta_is_a_quarter():
    """eta = Omega_max / |Delta| = 0.25 -- _plan_full_cost.md §2.1, §2.3a(b)."""
    assert DEFAULT_DEVICE.eta == pytest.approx(0.25, abs=1e-12)


def test_peak_budget():
    """T * Omega_max_rate = 15.7080 -- _plan_full_cost.md §2.6b."""
    assert DEFAULT_DEVICE.peak_budget == pytest.approx(15.7080, abs=5e-5)


def test_fenchel_min_time():
    """2 pi / Omega_max_rate = 20.0 ns -- _plan_full_cost.md §2.6b."""
    assert DEFAULT_DEVICE.fenchel_min_time == pytest.approx(20.0, abs=1e-9)


def test_decoherence_floor():
    """(Gamma1 + Gamma_phi) * T / 3 = 4.1667e-4 -- _plan_full_cost.md §0, §2.5, §3.2.

    The 1/3 converts the population-decay probability (Gamma1+Gamma_phi)*T
    into the average gate infidelity via Nielsen's d(d+1)/2 factor for a
    qubit (d=2).
    """
    d = DEFAULT_DEVICE
    assert d.gamma1 == pytest.approx(1.0 / 60_000.0)
    assert d.gamma_phi == pytest.approx(1.0 / 60_000.0 - 0.5 / 60_000.0)
    assert d.gamma1 + d.gamma_phi == pytest.approx(2.5e-5, rel=1e-12)
    assert d.decoherence_floor == pytest.approx(4.1667e-4, rel=1e-4)


def test_resonant_harmonic_equals_n_modes():
    """n* = 2 T |alpha| = 20 -- _plan_full_cost.md §2.6, chosen equal to M."""
    d = DEFAULT_DEVICE
    assert d.resonant_harmonic == pytest.approx(20.0, abs=1e-9)
    assert d.resonant_harmonic == pytest.approx(float(d.n_modes), abs=1e-9)


def test_gamma_phi_raises_if_t2_exceeds_the_2t1_bound():
    bad = Device(t1=1000.0, t2_echo=3000.0)  # 3000 > 2*1000
    with pytest.raises(ValueError):
        _ = bad.gamma_phi


def test_to_manifest_round_trips_every_field_and_derived_quantity():
    manifest = DEFAULT_DEVICE.to_manifest()
    for name in ("anharmonicity", "rabi_max", "t1", "t2_echo", "static_detuning",
                 "control_error", "sample_rate", "gate_time", "n_modes"):
        assert manifest[name] == getattr(DEFAULT_DEVICE, name)
    assert "derived" in manifest
    for name in ("eta", "peak_budget", "fenchel_min_time", "decoherence_floor",
                  "resonant_harmonic", "delta"):
        assert manifest["derived"][name] == pytest.approx(getattr(DEFAULT_DEVICE, name))


def test_default_device_is_a_singleton_instance_of_device():
    assert isinstance(DEFAULT_DEVICE, Device)
    assert device_mod.DEFAULT_DEVICE is DEFAULT_DEVICE


def test_device_module_does_not_enable_x64_itself():
    """Rule 0 lives only in __init__.py (test_architecture.py enforces this too)."""
    import inspect

    source = inspect.getsource(device_mod)
    assert "jax_enable_x64" not in source
