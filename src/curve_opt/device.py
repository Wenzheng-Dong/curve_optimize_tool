"""Hardware parameters: the single source of truth for the full-cost era.

★ AGENTS.md numerical discipline #10: ``T``, ``alpha``, ``Omega_max``, ``T1``/``T2``,
``delta_z``, ``epsilon`` may only live here. No other module, script, test or
notebook may spell these numbers -- see ``tests/test_device_hardcoding.py``,
which scans ``src/curve_opt`` (this file excepted, since it *is* the source) for
exactly the literals below.

Values are taken from Novera's ``TransmonDevice``
(``pulse-shape-Novera/proposals/1qb-DB/1qb-DB-demo/db_transmon.py``, git hash
``59fb616``) with two changes recorded by ``_plan_full_cost.md`` §3.2:
``control_error`` defaults to 2% here (Novera fixes 3%; this project sweeps
{1, 2, 3}%), and this dataclass additionally owns ``gate_time`` and
``n_modes``, which are choices of this project (§2.6/§2.6b), not of the device.

Unit convention (matches Novera, so the two projects stay cross-checkable
digit for digit): frequencies are stored as ordinary frequencies in GHz;
angular rates used by the Hamiltonian come from the ``*_rate`` properties,
which multiply by ``2 pi``. Times are ns, so a rate in GHz times a time in ns
is already in the natural units of the Hamiltonian (rad/ns times ns = rad).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

__all__ = ["Device", "DEFAULT_DEVICE", "TWO_PI"]

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class Device:
    """Fixed hardware + design parameters for the full-cost era.

    Attributes
    ----------
    anharmonicity : float
        ``alpha / 2pi`` in GHz, negative for a transmon. ``Delta = alpha``
        (drive resonant on 0-1, ``_plan_full_cost.md`` §2.1) -- see the
        :attr:`delta` property, which is the symbol the rest of the codebase
        actually calls this quantity by.
    rabi_max : float
        ``Omega_max / 2pi`` in GHz. Doubles as the hardware ceiling and the
        microscopic-perturbation-validity guard (§2.5c): the peak epigraph
        constraint and :attr:`eta` both read off this one number.
    t1, t2_echo : float
        Relaxation and echo coherence times, ns.
    static_detuning : float
        ``delta_z / 2pi`` in GHz -- the quasi-static Z-noise moment C1/C2 are
        conditioned on (§2.4a).
    control_error : float
        ``epsilon`` in the multiplicative amplitude error
        ``(Omega_x, Omega_y) -> (1 + epsilon)(Omega_x, Omega_y)`` -- the
        moment C3 is conditioned on. Default 2%; swept over {1, 2, 3}% in F06.
    sample_rate : float
        AWG sample rate, GS/s. At ``gate_time = 50`` ns this is 120 samples.
    gate_time : float
        Fixed gate duration ``T``, ns. Frozen at 50 ns by the resonant-harmonic
        argument of ``_plan_full_cost.md`` §2.6 (``n* = 2 T |Delta| /(2 pi)``
        lands exactly on the truncation order) and checked feasible by the
        Fenchel bound of §2.6b (``T >= 2 pi / Omega_max = 20`` ns).
    n_modes : int
        Sine-series truncation order ``M`` (§2.6, §4.3).
    """

    anharmonicity: float = -0.200
    rabi_max: float = 0.050
    t1: float = 60_000.0
    t2_echo: float = 60_000.0
    static_detuning: float = 1.0e-4
    control_error: float = 0.02
    sample_rate: float = 2.4
    gate_time: float = 50.0
    n_modes: int = 20

    # -- angular rates: *_rate = 2 pi * the stored frequency ----------------

    @property
    def anharmonicity_rate(self) -> float:
        """``alpha`` in rad/ns (negative). Expect ``-1.256637...``."""
        return TWO_PI * self.anharmonicity

    @property
    def rabi_max_rate(self) -> float:
        """``Omega_max`` in rad/ns. Expect ``0.314159...``."""
        return TWO_PI * self.rabi_max

    @property
    def static_detuning_rate(self) -> float:
        """``delta_z`` in rad/ns."""
        return TWO_PI * self.static_detuning

    @property
    def delta(self) -> float:
        """``Delta``, the 1-2 detuning fed to :func:`curve_opt.propagate.leakage_amplitude`.

        ``Delta = alpha`` exactly, drive resonant on 0-1 (``_plan_full_cost.md``
        §2.1). This is an alias for :attr:`anharmonicity_rate` so call sites can
        spell the physics symbol used throughout the plan rather than the
        device attribute name.
        """
        return self.anharmonicity_rate

    # -- derived design/physics numbers, each a §4.3/F01 sanity check -------

    @property
    def eta(self) -> float:
        """Perturbation parameter ``Omega_max / |Delta|``. Expect ``0.25``."""
        return self.rabi_max_rate / abs(self.delta)

    @property
    def peak_budget(self) -> float:
        """Dimensionless peak budget ``T * Omega_max``.

        Comparable directly to the energy-time invariant
        ``peak = T max|Omega|`` (``_plan_full_cost.md`` §3.1, §2.6b). Expect
        ``15.7080``.
        """
        return self.gate_time * self.rabi_max_rate

    @property
    def fenchel_min_time(self) -> float:
        """Fenchel lower bound on any closed-curve gate time, ns.

        A first-order-Z-robust (closed) control has unit speed and curvature
        ``|Omega|``, so Fenchel's theorem gives ``int |Omega| dt >= 2 pi``;
        combined with ``T max|Omega| >= int |Omega| dt`` this gives
        ``T >= 2 pi / Omega_max`` (``_plan_full_cost.md`` §2.6b). Expect
        ``20.0`` ns -- the absolute floor this project's fixed ``T = 50`` ns
        sits 2.5x above.
        """
        return TWO_PI / self.rabi_max_rate

    @property
    def gamma1(self) -> float:
        """Relaxation rate ``1 / T1``, 1/ns."""
        return 1.0 / self.t1

    @property
    def gamma_phi(self) -> float:
        """Pure dephasing rate ``1/T2 - 1/(2 T1)``, 1/ns.

        Mirrors Novera's split: ``T2`` already contains the coherence loss
        relaxation alone causes, and only the remainder is genuine dephasing.
        """
        rate = 1.0 / self.t2_echo - 0.5 / self.t1
        if rate < -1e-15:
            raise ValueError(
                f"t2_echo = {self.t2_echo} ns exceeds the 2*t1 bound for "
                f"t1 = {self.t1} ns"
            )
        return max(rate, 0.0)

    @property
    def decoherence_floor(self) -> float:
        """``(Gamma1 + Gamma_phi) * T`` -- the waveform-independent infidelity floor.

        ``T`` is fixed, so this term never enters the optimization; it is a
        constant added to the total account (``_plan_full_cost.md`` §0, §2.5).
        Expect ``1.25e-3``.
        """
        return (self.gamma1 + self.gamma_phi) * self.gate_time

    @property
    def resonant_harmonic(self) -> float:
        """``n* = 2 T |Delta| / (2 pi) = 2 T |alpha|`` -- the leakage-resonant harmonic.

        The ``n``-th sine-series harmonic ``sin(n pi t / T)`` has frequency
        ``n / (2T)``; C4 needs the envelope spectrum to vanish at ``|Delta|``,
        so this is the harmonic index at which that resonance sits
        (``_plan_full_cost.md`` §2.6). Expect ``20.0``, equal to
        :attr:`n_modes` by construction -- ``T = 50`` ns was chosen exactly so
        the resonance lands on the highest harmonic the basis carries.
        """
        return 2.0 * self.gate_time * abs(self.anharmonicity)

    # -- manifest -------------------------------------------------------

    def to_manifest(self) -> dict:
        """Flat dict of every stored field plus every derived quantity above.

        For F04+'s RunRecord manifest (``_plan_full_cost.md`` §3.2: "整份
        device 进每个 RunRecord 的 manifest"). This step only has to produce
        the function and prove it round-trips the numbers -- it is not wired
        to :mod:`curve_opt.recorder` yet.
        """
        base = {f.name: getattr(self, f.name) for f in fields(self)}
        derived = {
            name: getattr(self, name)
            for name in (
                "anharmonicity_rate",
                "rabi_max_rate",
                "static_detuning_rate",
                "delta",
                "eta",
                "peak_budget",
                "fenchel_min_time",
                "gamma1",
                "gamma_phi",
                "decoherence_floor",
                "resonant_harmonic",
            )
        }
        return {**base, "derived": derived}


#: The device every full-cost-era module/test should import rather than
#: constructing its own ``Device()`` -- one instance, one set of numbers.
DEFAULT_DEVICE = Device()
