"""F08 acceptance: the configuration layer on top of ``Device``.

F08 asked for one place that says what a run is made of, with the defaults marked.
Most of that lives in ``proposals/_gatelib`` (gate specs, arm recipes, the run
registry), which is outside the tested tree; what lands in ``src/`` is the ability
to *group* the device parameters by what they actually are and to *override* them
from a config mapping without editing source.

The proposal-layer half is covered by two scripts rather than by pytest, because it
needs the recorded ``_runs/`` store that this tree does not carry:

* ``_dev_logs/F08_golden.py``         -- every number, four gates, before vs after
* ``_dev_logs/F08_recipe_fidelity.py`` -- each arm recipe vs the manifest of the run
  it was actually solved with

★ The load-bearing test here is :func:`test_defaults_are_untouched_by_the_grouping`:
F08 is a refactor, so the one thing that must not have happened is a changed number.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import fields

import pytest

from curve_opt import recorder
from curve_opt.device import DEFAULT_DEVICE, GROUPS, Device


# ---------------------------------------------------------------------------
# the refactor changed no value
# ---------------------------------------------------------------------------


def test_defaults_are_untouched_by_the_grouping():
    """Every stored default, spelled out, so a metadata edit cannot move a number."""
    assert DEFAULT_DEVICE.anharmonicity == -0.200
    assert DEFAULT_DEVICE.rabi_max == 0.050
    assert DEFAULT_DEVICE.t1 == 60_000.0
    assert DEFAULT_DEVICE.t2_echo == 60_000.0
    assert DEFAULT_DEVICE.static_detuning == 1.0e-4
    assert DEFAULT_DEVICE.control_error == 0.02
    assert DEFAULT_DEVICE.sample_rate == 2.4
    assert DEFAULT_DEVICE.gate_time == 50.0
    assert DEFAULT_DEVICE.n_modes == 20


def test_declaration_order_is_still_the_positional_signature():
    """Positional construction must keep working -- ``Device(**flat)`` round-trips rely on it."""
    names = [f.name for f in fields(Device)]
    assert names == ["anharmonicity", "rabi_max", "t1", "t2_echo", "static_detuning",
                     "control_error", "sample_rate", "gate_time", "n_modes"]
    assert Device(*[getattr(DEFAULT_DEVICE, n) for n in names]) == DEFAULT_DEVICE


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------


def test_every_field_is_tagged_with_exactly_one_known_group():
    for f in fields(Device):
        assert f.metadata.get("group") in GROUPS, f"{f.name} has no valid group tag"


def test_groups_partitions_the_stored_fields():
    grouped = DEFAULT_DEVICE.groups()
    assert set(grouped) == set(GROUPS)
    flat = {k: v for g in grouped.values() for k, v in g.items()}
    assert flat == {f.name: getattr(DEFAULT_DEVICE, f.name) for f in fields(Device)}
    # the split is by *meaning*: T and M are project choices, not chip properties,
    # and the two noise moments are a hypothesis the budget weights are conditioned on
    assert set(grouped["design"]) == {"gate_time", "n_modes"}
    assert set(grouped["noise"]) == {"static_detuning", "control_error"}


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("as_config", [
    lambda d: d.groups(),
    lambda d: {f.name: getattr(d, f.name) for f in fields(Device)},
    lambda d: d.to_manifest(),          # the "derived" block must be ignored, not rejected
])
def test_from_dict_round_trips_the_default_device(as_config):
    assert Device.from_dict(as_config(DEFAULT_DEVICE)) == DEFAULT_DEVICE


def test_from_dict_overrides_only_what_it_names():
    dev = Device.from_dict({"noise": {"control_error": 0.03}})
    assert dev.control_error == 0.03
    for f in fields(Device):
        if f.name != "control_error":
            assert getattr(dev, f.name) == getattr(DEFAULT_DEVICE, f.name)


@pytest.mark.parametrize("cfg", [
    {"rabi_maxx": 1.0},                        # typo, flat
    {"hardware": {"anharmonicty": -0.2}},      # typo, inside a group
    {"nosie": {"control_error": 0.03}},        # typo in the group name itself
    {"hardware": 0.5},                         # group is not a mapping
])
def test_from_dict_rejects_anything_it_does_not_recognise(cfg):
    """A typo'd override that silently does nothing is the failure this layer prevents."""
    with pytest.raises(ValueError):
        Device.from_dict(cfg)


def test_from_dict_checks_the_t2_bound_eagerly():
    """Direct construction keeps the lazy check; a config load must fail at load time."""
    with pytest.raises(ValueError):
        Device.from_dict({"hardware": {"t1": 1000.0, "t2_echo": 3000.0}})


# ---------------------------------------------------------------------------
# the T <-> M coupling
# ---------------------------------------------------------------------------


def test_default_device_lands_the_leakage_resonance_on_the_top_harmonic():
    assert DEFAULT_DEVICE.resonant_harmonic_matched
    assert DEFAULT_DEVICE.resonant_harmonic == pytest.approx(DEFAULT_DEVICE.n_modes)


def test_breaking_the_T_M_coupling_warns_but_is_allowed():
    """``n* = 2 T |alpha| == M`` is a design choice, so moving off it is legal -- and loud.

    It must not raise: F06's sensitivity sweep and any future ``T``/``M`` scan are
    legitimate experiments that a hard error would block.
    """
    with pytest.warns(UserWarning, match="resonant harmonic"):
        dev = Device.from_dict({"design": {"gate_time": 40.0}})
    assert dev.resonant_harmonic == pytest.approx(16.0)
    assert not dev.resonant_harmonic_matched
    assert dev.to_manifest()["derived"]["resonant_harmonic_matched"] is False


def test_sweeping_the_noise_hypothesis_does_not_warn():
    """Varying epsilon is F06's whole job; it must stay silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for eps in (0.01, 0.02, 0.03):
            assert Device.from_dict({"noise": {"control_error": eps}}).control_error == eps


# ---------------------------------------------------------------------------
# reading a config file (recorder owns the I/O -- _plan.md 4.2)
# ---------------------------------------------------------------------------


def test_device_has_no_from_json_so_the_io_rule_holds():
    """The filesystem entry point is ``recorder.load_device``, not a Device method."""
    assert not hasattr(Device, "from_json")


def test_load_device_reads_a_grouped_override_file(tmp_path):
    p = tmp_path / "device_test.json"
    p.write_text(json.dumps({"design": {"n_modes": 20, "gate_time": 50.0},
                             "noise": {"control_error": 0.01}}))
    dev = recorder.load_device(p)
    assert dev.control_error == 0.01
    assert dev.gate_time == 50.0
    assert dev.anharmonicity == DEFAULT_DEVICE.anharmonicity


def test_shipped_device_default_json_matches_the_dataclass_defaults():
    """``configs/device_default.json`` is an export of the defaults, not a second source."""
    path = recorder.REPO_ROOT / "configs" / "device_default.json"
    if not path.exists():
        pytest.skip(f"{path} not present in this checkout")
    assert recorder.load_device(path) == DEFAULT_DEVICE
