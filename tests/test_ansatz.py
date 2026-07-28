"""Step 06 acceptance: registry, three-tier projection, step-0 error report.

The acceptance criteria are per-family relRMS plus the before/after constraint
table, with ``rcp`` at M=12 required to be <= 1.2e-4. Two things get hammered
here beyond that:

* that the projection really is **pure least squares with no repair** -- checked
  structurally (against an independent lstsq), by invariance to constraint
  shifts, and by the fact that the gate error survives projection instead of
  being silently fixed;
* that the ``forced`` tier is actually forced through and its deformation
  recorded, rather than quietly skipped.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import requires_cct

from curve_opt import ansatz, basis, geometry


@pytest.fixture(autouse=True)
def _builtin():
    ansatz.register_builtin_families(overwrite=True)


# --------------------------------------------------------------------------
# registry mechanics
# --------------------------------------------------------------------------


def test_builtin_registrations_present():
    names = dict(ansatz.list_ansatz())
    assert names["naive"] == "exact"
    assert names["circle_arc"] == "forced"
    assert names["min_norm_gate_only"] == "exact"


def test_register_a_new_family_needs_only_samples():
    """The registry contract: samples in, full step-0 report out, no core edits."""
    T, K = 1.0, 2001
    times = np.linspace(0.0, T, K)

    def sampler(scale: float = 1.0):
        return ansatz.AnsatzSource(
            name="_tmp",
            tier="approx",
            T=T,
            times=times,
            omega_x=scale * np.sin(np.pi * times / T) ** 3,
            omega_y=np.zeros_like(times),
            metadata={"scale": scale},
        )

    ansatz.register("_tmp", sampler, tier="approx", overwrite=True)
    try:
        rep = ansatz.project_named("_tmp", M=12, T=T, scale=2.0)
        assert rep.name == "_tmp"
        assert rep.metadata["scale"] == 2.0
        assert rep.rel_rms < 1e-6  # sin^3 is a 2-term sine series
    finally:
        ansatz.unregister("_tmp")


def test_unknown_name_and_bad_tier_are_rejected():
    with pytest.raises(KeyError):
        ansatz.source("does_not_exist")
    with pytest.raises(ValueError):
        ansatz.register("_x", lambda: None, tier="not_a_tier")
    with pytest.raises(ValueError):
        ansatz.register("naive", lambda: None, tier="exact")  # already there


# --------------------------------------------------------------------------
# ★ the protocol: pure least squares, no repair
# --------------------------------------------------------------------------


def test_projection_equals_an_independent_unconstrained_lstsq():
    """Structural proof of "no repair": same answer as a bare lstsq call."""
    src = ansatz.source("circle_arc", theta=np.pi, T=1.0)
    M, T = 12, 1.0
    rep = ansatz.project(src, M, T)

    S = basis.design_matrix(src.times, T, M)
    a_ref, *_ = np.linalg.lstsq(S, src.omega_x, rcond=None)
    assert np.allclose(rep.coeffs_a, a_ref, rtol=0, atol=0)


def test_projection_does_not_fix_the_gate():
    """A repaired protocol would return theta_after == theta_before. Ours does not.

    The circle's gate angle is pi before projection; after a pure least-squares
    fit it is not, and the difference is reported as the step-0 gate error.
    """
    rep = ansatz.project_named("circle_arc", M=12, T=1.0, theta=np.pi)
    assert rep.theta_before == pytest.approx(np.pi, rel=1e-6)
    assert abs(rep.theta_after - np.pi) > 1e-3
    assert rep.protocol == ansatz.PROTOCOL == "least_squares"


def test_projection_does_not_fix_closure_or_area():
    """The arc families are open curves; projection must leave them open."""
    rep = ansatz.project_named("circle_arc", M=12, T=1.0, theta=np.pi)
    assert rep.closure_before > 1e-2
    assert rep.closure_after > 1e-2
    assert abs(rep.area_before) > 1e-2
    assert abs(rep.area_after) > 1e-2


def test_projection_is_linear_in_the_source_amplitude():
    """Least squares is a linear map; any repair term would break homogeneity."""
    r1 = ansatz.project_named("circle_arc", M=10, T=1.0, theta=1.0)
    r2 = ansatz.project_named("circle_arc", M=10, T=1.0, theta=3.0)
    assert np.allclose(r2.coeffs_a, 3.0 * r1.coeffs_a, rtol=1e-12)


def test_relrms_is_independent_of_the_target_duration():
    """The fit happens in u = t/T_src; only the coefficient scale carries T."""
    src = ansatz.source("circle_arc", theta=np.pi, T=1.0)
    r1 = ansatz.project(src, M=12, T=1.0)
    r2 = ansatz.project(src, M=12, T=7.3)
    assert r2.rel_rms == pytest.approx(r1.rel_rms, rel=1e-12)
    assert np.allclose(r2.coeffs_a, r1.coeffs_a / 7.3, rtol=1e-12)
    assert r2.theta_before == pytest.approx(r1.theta_before, rel=1e-9)


def test_exact_sources_project_with_no_error():
    """A source already in the basis must come back bit-clean, tier ``exact``."""
    rep = ansatz.project_named("naive", M=12, T=1.0, theta=np.pi)
    assert rep.rel_rms < 1e-13
    assert rep.coeffs_a[0] == pytest.approx(np.pi**2 / 2, rel=1e-12)
    assert np.allclose(rep.coeffs_a[1:], 0.0, atol=1e-12)
    assert rep.theta_after == pytest.approx(np.pi, rel=1e-12)


def test_min_norm_source_projects_to_itself():
    rep = ansatz.project_named("min_norm_gate_only", M=12, T=1.0, theta=np.pi, M_source=12)
    assert rep.rel_rms < 1e-13
    assert np.allclose(rep.coeffs_a, basis.min_norm_gate_only(np.pi, 1.0, 12), rtol=1e-10)


# --------------------------------------------------------------------------
# tier measurement
# --------------------------------------------------------------------------


def test_endpoint_ratio_and_measured_tier_for_the_circle():
    """Constant curvature: the endpoint equals the peak, ratio exactly 1."""
    src = ansatz.source("circle_arc", theta=np.pi, T=1.0)
    assert ansatz.endpoint_ratio(src) == pytest.approx(1.0, rel=1e-12)
    assert ansatz.classify(src) == "forced"


def test_endpoint_ratio_is_zero_for_a_sine_series():
    src = ansatz.source("naive", theta=np.pi, T=1.0)
    assert ansatz.endpoint_ratio(src) == pytest.approx(0.0, abs=1e-15)
    assert ansatz.classify(src) == "approx"  # classify never claims "exact"


def test_forced_tier_is_actually_forced_through_and_deformed():
    """★ The circle is projected, and the deformation is visible and recorded.

    The endpoints must collapse to zero (the basis has no choice), the interior
    must overshoot (Gibbs), and relRMS must be large enough to make the point
    that the effective ansatz is the projected object.
    """
    rep = ansatz.project_named("circle_arc", M=12, T=1.0, theta=np.pi)
    assert rep.tier_declared == "forced"
    assert rep.tier_measured == "forced"
    assert rep.omega_x_after[0] == pytest.approx(0.0, abs=1e-12)
    assert rep.omega_x_after[-1] == pytest.approx(0.0, abs=1e-12)
    assert rep.rel_rms > 1e-2  # ~7% at M=12: qualitative deformation
    assert np.max(rep.omega_x_after) > 1.05 * np.max(rep.omega_x_before)  # overshoot


def test_forced_relrms_decreases_only_slowly_with_M():
    """Gibbs: the circle's error falls roughly like 1/sqrt(M), not geometrically."""
    src = ansatz.source("circle_arc", theta=np.pi, T=1.0)
    errs = [ansatz.project(src, M).rel_rms for M in (4, 12, 40, 120)]
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    assert errs[-1] > 1e-3  # still not small at M = 120


# --------------------------------------------------------------------------
# ★ upstream families: relRMS + before/after table
# --------------------------------------------------------------------------


@requires_cct
@pytest.mark.parametrize("family", ["rcp_lemniscate", "rcp_petal"])
def test_rcp_relrms_at_M12_meets_acceptance(cct_registered, family):
    """★ Acceptance: rcp relRMS <= 1.2e-4 at M = 12."""
    rep = ansatz.project_named(family, M=12, T=1.0, turning_label="pi")
    assert rep.tier_declared == "exact"
    assert rep.rel_rms <= 1.2e-4, rep.rel_rms
    assert rep.metadata["n_source_coefficients"] == 40


@requires_cct
def test_rcp_relrms_matches_the_recorded_M8_value(cct_registered):
    """step00d recorded 3.9e-4 at M = 8 for rcp_lemniscate."""
    rep = ansatz.project_named("rcp_lemniscate", M=8, T=1.0, turning_label="pi")
    assert rep.rel_rms == pytest.approx(3.9e-4, rel=0.05)


@requires_cct
def test_rcp_relrms_falls_geometrically_unlike_forced(cct_registered):
    """An exact family truncates; a forced one fights the endpoints."""
    src = ansatz.source("rcp_lemniscate", turning_label="pi")
    rcp = [ansatz.project(src, M).rel_rms for M in (8, 12, 20)]
    assert all(rcp[i] > rcp[i + 1] for i in range(len(rcp) - 1))
    assert rcp[-1] < 5e-5
    circle = ansatz.project(ansatz.source("circle_arc", theta=np.pi), 20).rel_rms
    assert circle > 100 * rcp[-1]


@requires_cct
def test_rcp_gate_error_is_the_recorded_2e_4(cct_registered):
    """§7.2: rcp_lemniscate(pi) misses the gate by 2.2e-4 rad before any fix."""
    rep = ansatz.project_named("rcp_lemniscate", M=20, T=1.0, turning_label="pi")
    assert abs(rep.theta_before - np.pi) == pytest.approx(2.2e-4, rel=0.15)
    assert rep.winding_branch == pytest.approx(rep.theta_before, rel=1e-12)


@requires_cct
def test_rcp_area_is_the_recorded_nonzero_value(cct_registered):
    """§7.2: measured area/L^2 = -3.30e-3 although the registry says zero_area."""
    rep = ansatz.project_named("rcp_lemniscate", M=20, T=1.0, turning_label="pi")
    assert rep.area_before == pytest.approx(-3.30e-3, rel=0.1)
    assert rep.closure_before < 1e-6


@requires_cct
@pytest.mark.parametrize(
    "family,tier,max_ratio",
    [("triangle_pulse_arc", "approx", 0.01), ("gaussian_arc", "approx", 0.05)],
)
def test_approx_tier_endpoints_are_small(cct_registered, family, tier, max_ratio):
    """Gaussian sits at ~2.1% of peak, triangle at ~4e-4 -- both approx."""
    src = ansatz.source(family)
    ratio = ansatz.endpoint_ratio(src)
    assert 0.0 <= ratio < max_ratio, ratio
    assert ansatz.classify(src) == "approx"
    rep = ansatz.project(src, M=12, T=1.0)
    assert rep.tier_declared == tier
    assert not rep.tier_mismatch


@requires_cct
def test_gaussian_endpoint_matches_the_documented_2_percent(cct_registered):
    src = ansatz.source("gaussian_arc")
    assert ansatz.endpoint_ratio(src) == pytest.approx(0.021, abs=0.004)


@requires_cct
@pytest.mark.parametrize("family", ["alpha_3d", "lissajous_3d", "helix", "zeng_clifford_3d"])
def test_3d_families_are_forced_through_and_report_what_was_done(cct_registered, family):
    """★ The forced tier is not skipped: 3D curves are rotated, inverted, fitted.

    ``rotation_applied`` records the global rotation that makes r_dot(0) = +z, a
    physical prerequisite rather than a repair. The after-projection robustness
    values were None until Step 11: rebuilding r(t) from (a, b) needs the SU(2)
    propagator, and now that it exists they are populated (see
    tests/test_propagate.py for what they turn out to be).
    """
    rep = ansatz.project_named(family, M=12, T=1.0)
    assert rep.coeffs_a.size == rep.coeffs_b.size == 12
    assert np.isfinite(rep.rel_rms)
    assert "rotation_applied" in rep.metadata
    assert rep.closure_after is not None and rep.area_after is not None
    assert np.isfinite(rep.closure_after) and np.isfinite(rep.area_after)
    assert rep.closure_before >= 0.0
    # and the gate angle now comes from the propagator, not from int Omega_x dt
    assert rep.theta_before == pytest.approx(
        float(ansatz.source(family).metadata["cct_rotation_angle"]), abs=5e-4
    )


@requires_cct
def test_alpha_3d_is_robust_but_has_the_wrong_gate(cct_registered):
    """★ Exactly the complementary input §8.2(4c) asks for.

    Closure and area are already satisfied while the gate is nowhere near the
    target -- the mirror image of rcp, which nearly hits the gate but owes area.
    """
    src = ansatz.source("alpha_3d")
    inv = geometry.invariants_of_positions(src.times, src.positions)
    assert inv.closure_invariant < 1e-12
    assert inv.area_invariant < 1e-5
    assert abs(src.metadata["cct_rotation_angle"]) < 0.01  # identity, nowhere near pi
    assert not src.is_planar


@requires_cct
def test_3d_rotation_preserves_closure_and_area(cct_registered):
    """The rotation is harmless: both robustness invariants are rotation-covariant."""
    from curvecontroltoolbox import curve_families as cf

    raw = np.asarray(cf.curve("alpha_3d").positions, dtype=float)
    rotated, R = ansatz._align_initial_tangent_to_z(raw)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    t = np.linspace(0.0, 1.0, raw.shape[0])
    before = geometry.invariants_of_positions(t, raw)
    after = geometry.invariants_of_positions(t, rotated)
    assert after.closure_invariant == pytest.approx(before.closure_invariant, rel=1e-9, abs=1e-15)
    assert after.area_invariant == pytest.approx(before.area_invariant, rel=1e-9, abs=1e-15)
    v = rotated[1] - rotated[0]
    assert v[2] / np.linalg.norm(v) == pytest.approx(1.0, abs=1e-9)


@requires_cct
def test_builtin_circle_matches_the_upstream_builder(cct_registered):
    """The analytic forced-tier member is the same curve cct builds."""
    from curvecontroltoolbox import curve_families as cf, inverse

    built = cf.curve("circle_arc", duration=1.0, turning_angle=np.pi, sample_count=4001)
    res = inverse.curve_to_control(
        inverse.prepare_curve_for_inverse_map(np.asarray(built.positions), num_samples=2001)
    )
    ours = ansatz.source("circle_arc", theta=np.pi, T=1.0)
    assert np.max(np.abs(res.pulse.omega_x)) == pytest.approx(
        float(np.max(ours.omega_x)), rel=1e-6
    )
    assert np.max(np.abs(res.pulse.omega_y)) < 1e-9


# --------------------------------------------------------------------------
# report completeness (what recorder.py will serialize in Step 07)
# --------------------------------------------------------------------------


def test_report_carries_everything_step0_npz_needs():
    rep = ansatz.project_named("circle_arc", M=12, T=1.0, theta=np.pi)
    for field in (
        "coeffs_a",
        "coeffs_b",
        "rel_rms",
        "theta_before",
        "theta_after",
        "closure_before",
        "closure_after",
        "area_before",
        "area_after",
        "omega_x_before",
        "omega_x_after",
        "winding_branch",
        "protocol",
        "tier_declared",
        "tier_measured",
    ):
        assert getattr(rep, field) is not None, field
    assert rep.omega_x_before.shape == rep.times.shape == rep.omega_x_after.shape


def test_M_must_be_positive():
    with pytest.raises(ValueError):
        ansatz.project_named("naive", M=0)


@requires_cct
def test_upsampling_before_the_inverse_map_is_refused(cct_registered):
    """★ Guard for the resampling trap: up-sampling manufactures fake curvature.

    Asking the inverse map for more samples than the builder produced
    differentiates the interpolation, which inflates max|Omega| linearly in
    num_samples. The sampler refuses instead of silently returning nonsense.
    """
    with pytest.raises(ValueError, match="exceeds"):
        ansatz.source("helix", sample_count=2001, num_samples=8001)


@requires_cct
def test_helix_curvature_is_the_declared_constant_at_any_sampling(cct_registered):
    """The trap's positive control: dense builder sampling converges, 1.5 exactly.

    cct's helix is built with constant curvature 1.5. Under the fixed protocol
    (raise sample_count, never up-sample) the extracted peak stays 1.5 for every
    sampling; under the trap it read 1.5 / 3.0 / 6.0.
    """
    peaks = []
    for K in (2001, 8001, 20001):
        src = ansatz.source("helix", sample_count=K)
        peaks.append(float(np.max(np.hypot(src.omega_x, src.omega_y))))
    assert all(p == pytest.approx(1.5, rel=1e-6) for p in peaks), peaks


@requires_cct
@pytest.mark.parametrize(
    "family,absolute_endpoint",
    [("alpha_3d", 1.18), ("lissajous_3d", 6.03), ("zeng_clifford_3d", 0.21)],
)
def test_3d_absolute_endpoint_curvature_matches_the_plan(cct_registered, family, absolute_endpoint):
    """The plan's §2.2b endpoint numbers are *absolute* curvatures; reproduce them.

    This is what settles the tier disagreement recorded in the dev log: the
    measurements agree, the criteria differ. |Omega(0)| absolute matches the plan,
    while |Omega(0)| / max|Omega| -- the scale-invariant form -- puts alpha_3d and
    zeng_clifford_3d below the forced threshold.
    """
    src = ansatz.source(family)
    got = float(np.hypot(src.omega_x[0], src.omega_y[0]))
    assert got == pytest.approx(absolute_endpoint, rel=0.10), got


@requires_cct
def test_tier_mismatch_is_reported_not_hidden(cct_registered):
    """alpha_3d is declared forced but measures approx; the report says so."""
    rep = ansatz.project_named("alpha_3d", M=12, T=1.0)
    assert rep.tier_declared == "forced"
    assert rep.tier_measured == "approx"
    assert rep.tier_mismatch is True
    assert rep.endpoint_ratio < ansatz.FORCED_THRESHOLD


def test_circle_arc_projects_parallel_to_the_gate_row():
    """★ The circle carries no information beyond its gate angle.

    A constant curvature has sine coefficients ``4C / (n pi)`` on odd harmonics,
    and the gate row is ``g_n = 2T / (n pi)`` on the same harmonics -- strictly
    proportional. So the least-squares projection of ``circle_arc`` is exactly
    parallel to ``g``, and an exact gate fix turns it into the gate-only
    minimum-norm solution itself.

    Consequence for the ansatz-comparison study: circle_arc is not an independent
    "bad ansatz". Despite its forced tier and 18% projection error it is the
    min-norm solution in disguise, which is what any conclusion drawn from it has
    to say (see results/curve_family_audit).
    """
    T, M = 1.0, 12
    a = ansatz.project_named("circle_arc", M=M, T=T, theta=np.pi).coeffs_a
    g = basis.gate_row(M, T)
    cos_angle = float(a @ g / np.linalg.norm(a) / np.linalg.norm(g))
    assert cos_angle == pytest.approx(1.0, abs=1e-14)
    assert np.max(np.abs(a[1::2])) < 1e-14  # even harmonics are absent

    fixed = a + (np.pi - g @ a) * g / (g @ g)
    assert np.allclose(fixed, basis.min_norm_gate_only(np.pi, T, M), atol=1e-7)
    assert basis.energy_invariant(fixed, T) == pytest.approx(
        basis.energy_bound_truncated(np.pi, M), rel=1e-12
    )
