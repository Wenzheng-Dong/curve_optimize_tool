"""Ansatz intake: any curve -> sine coefficients + a step-0 error report.

A *registry* interface. A new curve family -- a cct ``curve_families`` member,
a raw ``Omega`` sampling, or a 3D space curve ``r(t)`` -- only has to provide
samples or a closed form; it then gets the projection, the step-0 report and
admission to the optimizer for free, without touching the core modules
(``_plan.md`` §4.3 item 2).

Projection protocol (locked, ``_plan.md`` §2.2b)
-----------------------------------------------
The canonical protocol is **plain least squares, with no constraint repair**.
:func:`project` solves exactly one problem,

    min over (a, b) of || S a - Omega_x ||^2 + || S b - Omega_y ||^2,

where ``S`` is the sine design matrix on the source's own sample times. There is
no gate row in that system, no closure term, no penalty, no reweighting and no
post-hoc correction of any kind -- ``PROTOCOL`` is the string recorded in the
manifest, and any variant that repairs a constraint is a *different protocol*
that must be labelled as such. Otherwise the choice of repair becomes a
confounder in the ansatz comparison: different repairs hand the optimizer
different starting points and can flip the conclusion.

Every projection produces a **step-0 error** -- :class:`Step0Report` -- carrying
the relative RMS of ``Omega`` together with the gate angle, closure and area
*before and after* projection. It is quantified, recorded and stored as
iteration 0 of the run history, not swept under the rug as an implementation
detail.

Time rescaling is not a repair
------------------------------
A source has its own duration ``T_src`` (the rcp records use 50, cct's arcs
default to 1). Mapping it to the project duration ``T`` uses ``u = t / T_src``
and ``Omega -> Omega * T_src / T``, which leaves the gate angle
``theta = int Omega dt`` and every scale-invariant quantity untouched. Since
``sin(n pi t / T) = sin(n pi u)``, the least-squares fit itself happens in ``u``
and is completely independent of ``T``; only the coefficient scale carries it.

Representability tiers
----------------------
* ``exact``   -- natively a sine series: ``rcp_petal`` / ``rcp_lemniscate``.
  relRMS ~1e-4 at M=12, pure truncation.
* ``approx``  -- endpoint curvature zero or nearly zero: ``triangle_pulse_arc``
  (exactly 0, cusp => coefficients decay like 1/n^2), ``gaussian_arc``
  (~2.1% of peak).
* ``forced``  -- ``Omega(0) != 0``, a hard conflict with the basis:
  ``circle_arc`` and cct's 3D closed families. Gibbs-type endpoint error, the
  waveform deforms qualitatively. Still projected -- but any scientific reading
  must state that the effective ansatz is the *projected* object.

The tier of a registration is **declared** (it comes from the plan), while
:func:`endpoint_ratio` and :func:`classify` *measure* it as
``max(|Omega(0)|, |Omega(T)|) / max|Omega|`` against :data:`FORCED_THRESHOLD`.
:class:`Step0Report` reports both and flags a disagreement rather than silently
trusting either -- a declared tier is a claim about a family, and claims get
checked.

Each ansatz is optimized in its own native winding branch (the constraint
right-hand side theta is that ansatz's own total rotation angle), to avoid the
violent deformation of a cross-branch projection (``_plan.md`` §8.3);
:attr:`Step0Report.winding_branch` carries that number into the manifest.

Upstream data
-------------
:func:`register_cct_families` imports ``curvecontroltoolbox`` lazily. The
upstream repository is **read-only** for this project and is not a declared
dependency of the ``curve`` environment, so the import only succeeds when the
caller has put it on ``PYTHONPATH``::

    PYTHONPATH=<...>/curvecontroltoolbox/src python -c "..."

Everything registered by :func:`register_builtin_families` is pure numpy and
always available, which keeps the core test suite independent of upstream.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from curve_opt import basis, geometry

__all__ = [
    "AnsatzSource",
    "FORCED_THRESHOLD",
    "PROTOCOL",
    "Step0Report",
    "TIERS",
    "classify",
    "endpoint_ratio",
    "list_ansatz",
    "project",
    "project_named",
    "register",
    "register_builtin_families",
    "register_cct_families",
    "source",
    "unregister",
]

#: The one projection protocol of ``_plan.md`` §2.2b. Recorded in the manifest.
#: A variant that repairs the gate or the closure must use a different string.
PROTOCOL = "least_squares"

TIERS = ("exact", "approx", "forced")

#: Endpoint-to-peak ratio above which a family is measured as ``forced``.
#: Chosen from the data, which separates cleanly: triangle 4e-4, gaussian 2.1e-2,
#: then a gap up to circle 1.0 and cct's lissajous_3d 0.80.
FORCED_THRESHOLD = 0.10


class AnsatzSource(NamedTuple):
    """One ansatz as it arrives, before any projection.

    ``times`` runs from 0 to ``T``; ``omega_x`` / ``omega_y`` are the control
    fields on that sampling (``omega_y`` is all zeros for a planar source).
    ``positions`` is the space curve when the source is a curve -- it is what
    makes the *before* robustness values of a 3D family readable without the
    propagator of Step 11.
    """

    name: str
    tier: str
    T: float
    times: np.ndarray
    omega_x: np.ndarray
    omega_y: np.ndarray
    positions: np.ndarray | None = None
    metadata: dict | None = None

    @property
    def is_planar(self) -> bool:
        return not np.any(self.omega_y)


class Step0Report(NamedTuple):
    """The step-0 error of one projection: what §5.1's ``step0.npz`` stores.

    ``*_before`` are evaluated on the source sampling, ``*_after`` on the
    projected coefficients. ``theta`` is the gate angle from ``Omega_x``,
    ``closure`` is ``|r(T)| / L`` and ``area`` is the signed ``area_x / L^2`` in
    the planar layer. For a genuinely 3D source the *before* values come from the
    space curve (norms, not signed components) and the *after* values are
    ``None``: reconstructing ``r(t)`` from ``(a, b)`` needs the SU(2) propagator,
    which is Step 11.
    """

    name: str
    protocol: str
    tier_declared: str
    tier_measured: str
    tier_mismatch: bool
    M: int
    T: float
    coeffs_a: np.ndarray
    coeffs_b: np.ndarray
    rel_rms: float
    rel_rms_x: float
    rel_rms_y: float | None
    endpoint_ratio: float
    winding_branch: float
    is_planar: bool
    theta_before: float
    theta_after: float
    closure_before: float
    closure_after: float | None
    area_before: float
    area_after: float | None
    times: np.ndarray
    omega_x_before: np.ndarray
    omega_x_after: np.ndarray
    omega_y_before: np.ndarray
    omega_y_after: np.ndarray
    metadata: dict


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, tuple[Callable[..., AnsatzSource], str]] = {}


def register(name: str, sampler: Callable[..., AnsatzSource], *, tier: str, overwrite: bool = False):
    """Register *sampler* under *name* with a declared representability *tier*.

    ``sampler(**kwargs) -> AnsatzSource``. Nothing else is required of a family:
    projection, step-0 report and admission to the optimizer follow for free.
    """
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"ansatz {name!r} is already registered; pass overwrite=True")
    _REGISTRY[name] = (sampler, tier)


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def list_ansatz() -> tuple[tuple[str, str], ...]:
    """Return ``(name, declared_tier)`` for every registration, sorted by name."""
    return tuple((n, _REGISTRY[n][1]) for n in sorted(_REGISTRY))


def source(name: str, **kwargs) -> AnsatzSource:
    """Build the named ansatz. Keyword arguments go to its sampler."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown ansatz {name!r}; registered: {[n for n, _ in list_ansatz()]}")
    sampler, tier = _REGISTRY[name]
    src = sampler(**kwargs)
    return src._replace(name=name, tier=tier)


# --------------------------------------------------------------------------
# measurement of the tier
# --------------------------------------------------------------------------


def endpoint_ratio(src: AnsatzSource) -> float:
    """``max(|Omega(0)|, |Omega(T)|) / max|Omega|`` -- the conflict with the basis.

    The sine basis forces ``Omega(0) = Omega(T) = 0`` identically, so this ratio
    is exactly the size of what the projection cannot represent at the endpoints.
    """
    mag = np.hypot(src.omega_x, src.omega_y)
    peak = float(np.max(mag))
    if peak == 0.0:
        return 0.0
    return float(max(mag[0], mag[-1]) / peak)


def classify(src: AnsatzSource) -> str:
    """Measured tier: ``forced`` above :data:`FORCED_THRESHOLD`, else ``approx``.

    Never returns ``exact``: being a native sine series is a fact about how the
    family is defined, not something an endpoint ratio can establish.
    """
    return "forced" if endpoint_ratio(src) > FORCED_THRESHOLD else "approx"


# --------------------------------------------------------------------------
# the projection
# --------------------------------------------------------------------------


def _rescaled(src: AnsatzSource, T: float):
    """Map the source onto duration *T*: ``u = t / T_src``, ``Omega *= T_src / T``."""
    u = np.asarray(src.times, dtype=float) / src.T
    factor = src.T / T
    return u * T, np.asarray(src.omega_x) * factor, np.asarray(src.omega_y) * factor


def _rel_rms(residual, reference) -> float:
    ref = float(np.sqrt(np.mean(np.asarray(reference) ** 2)))
    if ref == 0.0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(residual) ** 2)) / ref)


def _planar_invariants(om_target, times, T: float, N: int):
    """closure/L and signed area/L^2 of a planar Omega_x sampling, via geometry."""
    t_mid, _ = geometry.midpoint_grid(T, N)
    om_mid = np.interp(t_mid, times, om_target)
    c = geometry.chain(om_mid, T)
    return (
        float(c.theta_edge[-1]),
        float(np.linalg.norm(c.closure) / T),
        float(c.area[0] / T**2),
    )


def project(src: AnsatzSource, M: int, T: float = 1.0, N: int = geometry.N_DEFAULT) -> Step0Report:
    """Project *src* onto ``M`` sine harmonics by plain least squares.

    No constraint is touched: the returned coefficients are the unconstrained
    least-squares minimizer and nothing more. Whatever gate error, closure or
    area the projection introduces is *reported*, not repaired -- that is the
    locked protocol of ``_plan.md`` §2.2b, and repairing it here would make the
    ansatz comparison of Step 14 uninterpretable.
    """
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")

    times, om_x, om_y = _rescaled(src, T)
    S = basis.design_matrix(times, T, M)

    # ---- the entire protocol: one unconstrained least-squares solve per field
    a, *_ = np.linalg.lstsq(S, om_x, rcond=None)
    b, *_ = np.linalg.lstsq(S, om_y, rcond=None)
    # ----

    fit_x, fit_y = S @ a, S @ b
    rel_x = _rel_rms(om_x - fit_x, om_x)
    rel_y = _rel_rms(om_y - fit_y, om_y) if np.any(om_y) else None
    rel_all = _rel_rms(
        np.concatenate([om_x - fit_x, om_y - fit_y]), np.concatenate([om_x, om_y])
    )

    theta_after = basis.gate_angle(a, T)

    if src.is_planar:
        theta_before, closure_before, area_before = _planar_invariants(om_x, times, T, N)
        # After projection the ansatz *is* the coefficient vector, so the
        # after-values are evaluated on it directly rather than on its samples.
        closure_after = float(np.linalg.norm(geometry.closure(a, T, N)) / T)
        area_after = float(geometry.area_invariant(a, T, N))
    elif src.positions is not None:
        inv = geometry.invariants_of_positions(src.times, src.positions)
        theta_before = float(np.trapezoid(om_x, times))
        closure_before, area_before = inv.closure_invariant, inv.area_invariant
        closure_after = area_after = None  # needs the SU(2) propagator -- Step 11
    else:
        raise ValueError(
            f"ansatz {src.name!r} is non-planar and carries no positions, so its "
            "before-projection robustness values cannot be evaluated"
        )

    tier_measured = classify(src)
    return Step0Report(
        name=src.name,
        protocol=PROTOCOL,
        tier_declared=src.tier,
        tier_measured=tier_measured,
        tier_mismatch=(src.tier != "exact" and src.tier != tier_measured),
        M=M,
        T=T,
        coeffs_a=a,
        coeffs_b=b,
        rel_rms=rel_all,
        rel_rms_x=rel_x,
        rel_rms_y=rel_y,
        endpoint_ratio=endpoint_ratio(src),
        winding_branch=theta_before,
        is_planar=bool(src.is_planar),
        theta_before=theta_before,
        theta_after=theta_after,
        closure_before=closure_before,
        closure_after=closure_after,
        area_before=area_before,
        area_after=area_after,
        times=times,
        omega_x_before=om_x,
        omega_x_after=fit_x,
        omega_y_before=om_y,
        omega_y_after=fit_y,
        metadata=dict(src.metadata or {}),
    )


def project_named(name: str, M: int, T: float = 1.0, N: int = geometry.N_DEFAULT, **kwargs):
    """Convenience: :func:`source` then :func:`project`."""
    return project(source(name, **kwargs), M, T, N)


# --------------------------------------------------------------------------
# built-in families (pure numpy, always available)
# --------------------------------------------------------------------------


def _naive_source(theta: float = np.pi, T: float = 1.0, K: int = 20_001) -> AnsatzSource:
    times = np.linspace(0.0, T, K)
    a = basis.naive_coeffs(theta, T, M=1)
    return AnsatzSource(
        name="naive",
        tier="exact",
        T=T,
        times=times,
        omega_x=basis.omega(a, times, T),
        omega_y=np.zeros_like(times),
        metadata={"theta": float(theta), "description": "single mode, exact X(theta)"},
    )


def _min_norm_source(theta: float = np.pi, T: float = 1.0, M_source: int = 12, K: int = 20_001):
    times = np.linspace(0.0, T, K)
    a = basis.min_norm_gate_only(theta, T, M_source)
    return AnsatzSource(
        name="min_norm_gate_only",
        tier="exact",
        T=T,
        times=times,
        omega_x=basis.omega(a, times, T),
        omega_y=np.zeros_like(times),
        metadata={
            "theta": float(theta),
            "M_source": M_source,
            "description": "gate-only minimum norm",
        },
    )


def _circle_arc_source(theta: float = np.pi, T: float = 1.0, K: int = 20_001) -> AnsatzSource:
    """Constant curvature ``kappa = theta / T`` -- the canonical ``forced`` member.

    Analytic, and identical to cct's ``circle_arc`` builder, which constructs a
    unit-speed circle in the y-z plane with ``curvature = turning_angle /
    duration``. ``tests/test_ansatz.py`` checks the two agree when upstream is
    available, so the core suite needs no upstream import to exercise the forced
    tier.
    """
    times = np.linspace(0.0, T, K)
    kappa = float(theta) / T
    return AnsatzSource(
        name="circle_arc",
        tier="forced",
        T=T,
        times=times,
        omega_x=np.full_like(times, kappa),
        omega_y=np.zeros_like(times),
        metadata={"theta": float(theta), "kappa": kappa, "description": "constant curvature"},
    )


def register_builtin_families(overwrite: bool = True) -> None:
    """Register the families that need no upstream data."""
    register("naive", _naive_source, tier="exact", overwrite=overwrite)
    register("min_norm_gate_only", _min_norm_source, tier="exact", overwrite=overwrite)
    register("circle_arc", _circle_arc_source, tier="forced", overwrite=overwrite)


# --------------------------------------------------------------------------
# cct-backed families (upstream is read-only; import is lazy)
# --------------------------------------------------------------------------


def _require_cct():
    try:
        from curvecontroltoolbox import curve_families, inverse  # noqa: PLC0415
        from curvecontroltoolbox._rcp_zerror_coefficients import (  # noqa: PLC0415
            RCP_ZERROR_CURVES,
        )
    except ImportError as exc:  # pragma: no cover - depends on the caller's env
        raise ImportError(
            "curvecontroltoolbox is not importable. It is upstream, read-only and "
            "not a dependency of the `curve` environment: put its src directory on "
            "PYTHONPATH to register the cct-backed families."
        ) from exc
    return curve_families, inverse, RCP_ZERROR_CURVES


def _align_initial_tangent_to_z(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate *positions* so that ``r_dot(0)`` points along ``+z``.

    ★ Not a convention but a physical requirement: ``r_dot = U_c^dagger sigma_z
    U_c`` and ``U_c(0) = I``, so every realizable SCQC curve starts along ``+z``.
    cct's inverse map rejects curves that do not (its 3D closed families are
    defined in other orientations), so intake applies the global rotation and
    records that it did.

    A rotation is harmless for what these families are wanted for: closure and
    the area vector rotate with the curve, so "closed" and "zero area" are
    preserved exactly. It is not a constraint repair -- it changes no residual,
    it makes the inverse map exist at all.
    """
    positions = np.asarray(positions, dtype=float)
    v = positions[1] - positions[0]
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, z)
    s = float(np.linalg.norm(axis))
    if s < 1e-14:
        R = np.eye(3) if v[2] > 0 else np.diag([1.0, -1.0, -1.0])
        return positions @ R.T, R
    axis = axis / s
    angle = float(np.arctan2(s, float(v @ z)))
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return positions @ R.T, R


def _rcp_sampler(family: str, tier: str):
    def sampler(turning_label: str = "pi", T: float = 1.0, K: int = 20_001) -> AnsatzSource:
        _, _, records = _require_cct()
        matches = [
            r for r in records if r["family"] == family and r["turning_label"] == turning_label
        ]
        if not matches:
            labels = sorted({r["turning_label"] for r in records if r["family"] == family})
            raise KeyError(f"{family} has no member {turning_label!r}; available: {labels}")
        rec = matches[0]
        # The record *is* a sine series: Omega = sum c_n sin(n pi t / T_rec).
        c = np.asarray(rec["omega_sine_coefficients"], dtype=float)
        T_rec = float(rec["omega_duration"])
        times = np.linspace(0.0, T_rec, K)
        return AnsatzSource(
            name=family,
            tier=tier,
            T=T_rec,
            times=times,
            omega_x=basis.omega(c, times, T_rec),
            omega_y=np.zeros_like(times),
            metadata={
                "turning_label": turning_label,
                "gate_angle": float(rec["gate_angle"]),
                "total_turning": float(rec["total_turning"]),
                "robustness_order": rec["robustness_order"],
                "n_source_coefficients": int(c.size),
                "cct_fit_residual": float(rec["fit_residual"]),
                "source_duration": T_rec,
            },
        )

    return sampler


def _cct_curve_sampler(cct_name: str, tier: str, default_kwargs: dict | None = None):
    """Build a sampler that goes cct curve -> rotate -> inverse map -> Omega_{x,y}.

    ★ Resampling trap (measured, see the Step 06 dev log). cct's inverse map takes
    curvature from finite differences of the *positions*, so asking
    ``prepare_curve_for_inverse_map`` for more samples than the builder produced
    differentiates the interpolation instead of the curve: ``max|Omega|`` then
    grows *proportionally to num_samples* (the helix, whose true curvature is the
    constant 1.5, reports 1.5 / 3.0 / 6.0 at num_samples 4001 / 8001 / 16001) and
    the endpoint ratio collapses to ~1e-12, manufacturing a fake smooth start.
    The fix is to make the *builder* sample densely and never up-sample after it:
    ``sample_count`` is what to increase, ``num_samples`` stays ``None``.
    """

    def sampler(sample_count: int = 20_001, num_samples: int | None = None, **kwargs):
        curve_families, inverse, _ = _require_cct()
        builder_kwargs = {**(default_kwargs or {}), "sample_count": sample_count, **kwargs}
        try:
            built = curve_families.curve(cct_name, **builder_kwargs)
        except TypeError:  # a few builders take no sample_count
            builder_kwargs.pop("sample_count")
            built = curve_families.curve(cct_name, **builder_kwargs)
        positions = np.asarray(built.positions, dtype=float)
        if num_samples is not None and num_samples > positions.shape[0]:
            raise ValueError(
                f"num_samples={num_samples} exceeds the {positions.shape[0]} samples the "
                f"{cct_name!r} builder produced. Up-sampling before the inverse map "
                "differentiates the interpolation and inflates max|Omega| linearly in "
                "num_samples; raise sample_count instead."
            )
        rotated, R = _align_initial_tangent_to_z(positions)
        rotation_applied = not np.allclose(R, np.eye(3), atol=1e-12)
        sampled = inverse.prepare_curve_for_inverse_map(rotated, num_samples=num_samples)
        result = inverse.curve_to_control(sampled)
        pulse = result.pulse
        return AnsatzSource(
            name=cct_name,
            tier=tier,
            T=float(pulse.times[-1]),
            times=np.asarray(pulse.times, dtype=float),
            omega_x=np.asarray(pulse.omega_x, dtype=float),
            omega_y=np.asarray(pulse.omega_y, dtype=float),
            positions=np.asarray(sampled.positions, dtype=float),
            metadata={
                "cct_name": cct_name,
                "builder_kwargs": builder_kwargs,
                "raw_sample_count": int(positions.shape[0]),
                "inverse_num_samples": num_samples,
                "rotation_applied": bool(rotation_applied),
                "rotation_matrix": R.tolist(),
                "cct_rotation_angle": float(result.gate.rotation_angle),
                "cct_rotation_axis": np.asarray(result.gate.rotation_axis).tolist(),
                "total_torsion": float(result.gate.total_torsion),
                "source_duration": float(pulse.times[-1]),
            },
        )

    return sampler


#: cct-backed registrations: name -> (sampler factory, declared tier).
#: Declared tiers follow ``_plan.md`` §2.2b; :func:`classify` measures them.
CCT_FAMILIES = {
    "rcp_petal": ("rcp", "exact"),
    "rcp_lemniscate": ("rcp", "exact"),
    "triangle_pulse_arc": ("curve", "approx"),
    "gaussian_arc": ("curve", "approx"),
    "alpha_3d": ("curve", "forced"),
    "lissajous_3d": ("curve", "forced"),
    "helix": ("curve", "forced"),
    "zeng_clifford_3d": ("curve", "forced"),
    "flora_petal": ("curve", "forced"),
    "tilted_lemniscate": ("curve", "forced"),
    "treble_clef_loop": ("curve", "forced"),
}

_ARC_DEFAULTS = {"duration": 1.0, "turning_angle": float(np.pi)}


def register_cct_families(overwrite: bool = True) -> tuple[str, ...]:
    """Register the upstream cct families. Returns the names registered.

    Raises :class:`ImportError` with an actionable message when cct is not on
    ``PYTHONPATH``. Upstream is never written to.
    """
    _require_cct()
    for name, (kind, tier) in CCT_FAMILIES.items():
        if kind == "rcp":
            register(name, _rcp_sampler(name, tier), tier=tier, overwrite=overwrite)
        else:
            defaults = _ARC_DEFAULTS if name.endswith("_arc") else None
            register(
                name, _cct_curve_sampler(name, tier, defaults), tier=tier, overwrite=overwrite
            )
    return tuple(CCT_FAMILIES)


register_builtin_families()
