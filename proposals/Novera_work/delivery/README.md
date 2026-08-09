# Single-qubit π-gate waveforms for the Novera transmon — delivery package

X(π) and Y(π), 50 ns, 2.4 GSa/s. Generated 2026-08-09 from run set
`20260809_Xpi_novera_arm{C,D,E,F}_accepted` (git `db5b963`).

---

## ⚠️ The one thing that must not be missed

**Playing a waveform file alone does not give you the target gate.**

The optimiser solves for the qubit-block polar decomposition

```
W  =  R_z(φ_vz) · U_target
```

so every pulse leaves a residual frame rotation `φ_vz` that has to be applied as a
**virtual Z (frame update) after the pulse**. `φ_vz` is *not* small — for arm E it is
+173.7°. Forgetting it turns a 3×10⁻⁸ gate into a 0.66 error, i.e. an essentially
random gate.

Two ways to handle it — **pick one, both are exact**:

| | file | what you do |
|---|---|---|
| **(a)** | `<gate>/waveforms/*.csv` | play it, then apply `φ_vz` as a virtual-Z frame update |
| **(b)** | `<gate>/waveforms/*_vzfree.csv` | play it. **Nothing else.** |

The `_vzfree` files have the whole drive phase pre-rotated by `ψ = −φ_vz/2`. For a
π rotation about an in-plane axis this is an exact identity — rotating the drive
phase by `ψ` maps `W → R_z(φ_vz + 2ψ)·U_target` — so `ψ = −φ_vz/2` lands on the
target with no frame bookkeeping at all. Robustness is unchanged (`R_z` commutes
with both the detuning and the amplitude-error terms). **If you have no strong
preference, use (b).**

> This identity holds because the target is a **π** rotation about an equatorial
> axis. It does *not* generalise to π/2 gates — if we ship those later they will
> need the real virtual Z.

---

## What is in the package

```
README.md                  this file
delivery_metadata.json     every number below, machine-readable, incl. the raw arrays
verify_waveform.py         stand-alone re-simulation (numpy + scipy only)
Xpi/waveforms/             X(pi): 6 arms x {canonical, vzfree}
Ypi/waveforms/             Y(pi): same
```

Each CSV carries a `#`-commented header block with the full conventions, playback
parameters, simulated performance and provenance, so the file stays self-describing
even if it gets separated from this README.
Read it with e.g. `pandas.read_csv(path, comment="#")`.

---

## The arms

All at **T = 50 ns**, 120 samples, 2.4 GSa/s. `1−F̄` is the coherent three-level
average gate infidelity on the Novera device (α/2π = −198.48 MHz) with the virtual Z
applied.

| arm | what it is | 1−F̄ | leakage | peak Rabi | φ_vz | ψ = −φ_vz/2 |
|---|---|---|---|---|---|---|
| A | Gaussian, no DRAG (baseline) | 1.75e-03 | 1.01e-05 | 24.10 MHz | 0.00° | −0.00° |
| B | Gaussian + DRAG (standard baseline) | 7.46e-06 | 4.27e-09 | 24.10 MHz | +7.76° | −3.88° |
| C | optimised 3D, unguarded | 9.21e-05 | 9.18e-05 | 45.47 MHz | −7.72° | +3.86° |
| D | optimised 3D + fluence cap | 6.81e-06 | 6.22e-06 | 24.23 MHz | +145.20° | −72.60° |
| **E** | **optimised 3D, exact-gate arm** | **3.46e-08** | 3.37e-08 | 25.28 MHz | +173.66° | −86.83° |
| **F** | **optimised 3D, robust arm** | **5.39e-07** | 5.13e-07 | 42.81 MHz | −149.99° | +75.00° |

**X(π) and Y(π) share this table exactly.** Y(π) is not an independent solve: it is
the X(π) waveform played with the drive phase advanced by `φ_axis = π/2`, which is an
exact frame change. Every figure of merit — infidelity, leakage, peak amplitude,
`φ_vz`, the whole robustness grid — is invariant under it, and we verified that
numerically rather than assuming it (the exported Y waveform matches
`rotate(X, π/2)` to 3×10⁻¹⁷).

**Which to run.** Take **B** first — it is a textbook Gaussian+DRAG and exists to
confirm that conventions, scaling and readout all line up before anything exotic is
tried. Then **E** for the best nominal number, and **F** if your amplitude
calibration drifts (grids below).

For scale: the T1/T2 floor for a 50 ns gate on this device (T1 = 35 µs,
T2echo = 41.9 µs) is **6.36e-04**, so B/D/E/F all sit far below the decoherence
limit — you should be measuring the decoherence floor, not the pulse.

### Robustness, 1−F̄ over (amplitude error ε, drive detuning)

**arm E** — sharp optimum, best if you are well calibrated

| ε \ δ | −0.2 MHz | −0.1 | 0 | +0.1 | +0.2 |
|---|---|---|---|---|---|
| −5% | 4.29e-03 | 4.04e-03 | 3.80e-03 | 3.57e-03 | 3.34e-03 |
| −2% | 8.23e-04 | 7.09e-04 | 6.06e-04 | 5.15e-04 | 4.36e-04 |
| **0** | 2.51e-05 | 6.31e-06 | **3.46e-08** | 6.77e-06 | 2.70e-05 |
| +2% | 4.34e-04 | 5.13e-04 | 6.06e-04 | 7.12e-04 | 8.34e-04 |
| +5% | 3.29e-03 | 3.52e-03 | 3.76e-03 | 4.02e-03 | 4.30e-03 |

**arm F** — flat optimum, stays below 4e-5 across the whole grid

| ε \ δ | −0.2 MHz | −0.1 | 0 | +0.1 | +0.2 |
|---|---|---|---|---|---|
| −5% | 2.29e-05 | 1.85e-05 | 2.00e-05 | 2.72e-05 | 3.99e-05 |
| −2% | 8.66e-06 | 2.94e-06 | 1.63e-06 | 4.50e-06 | 1.13e-05 |
| **0** | 7.28e-06 | 2.17e-06 | **5.39e-07** | 2.19e-06 | 6.93e-06 |
| +2% | 4.95e-06 | 1.50e-06 | 6.98e-07 | 2.38e-06 | 6.37e-06 |
| +5% | 1.19e-05 | 1.26e-05 | 1.50e-05 | 1.88e-05 | 2.39e-05 |

---

## Conventions

Everything below is repeated in every CSV header.

**Hamiltonian** (rotating frame, RWA, three levels):

```
H(t) = Ω_x(t)·(a + a†)/2  +  Ω_y(t)·i(a† − a)/2  +  Δ·|2⟩⟨2|
a|1⟩ = |0⟩,   a|2⟩ = √2|1⟩,   Δ = 2π·α,  α/2π = −0.19848 GHz
```

**Target gates**: `X(π) = exp(−iπσ_x/2) = −iX` and `Y(π) = exp(−iπσ_y/2) = −iY`.
Global phase is irrelevant.

**Units**: `Omega_*_rad_per_ns` is **angular** frequency in rad/ns. The MHz columns
are `Ω/2π` in MHz, i.e. what you read off a Rabi-chevron fit. Conversion:
`Ω[rad/ns] = 2π × f[MHz] × 10⁻³`.

**Normalised columns**: `I_norm = Ω_x/Ω_max`, `Q_norm = Ω_y/Ω_max`, with
`Ω_max/2π = 50 MHz` — the design amplitude ceiling, **not** the peak of the
particular waveform. Peaks range 24–45 MHz, so `|I_norm| ≤ 1` always with room.

**Time grid**: `t_ns` are **cell midpoints**, played sample-and-hold.
`t₀ = dt/2 = 0.2083 ns`, `dt = 0.4166667 ns` (2.4 GSa/s), total `T = 50.000 ns`.

**Drive frequency**: **on resonance with the bare f01. No offset needed.** The
optimisation ran against the exact three-level propagator, so the AC-Stark shift
from |2⟩ is already absorbed into the pulse shape. Verified — the residual optimal
detuning is:

| arm | optimal amplitude rescale | optimal drive detuning |
|---|---|---|
| D | ×0.99998 | +0.0 kHz |
| E | ×1.000000 | −1.3 kHz |
| F | ×1.0116 | +2.1 kHz |

(For contrast, a composite BB1 π-pulse we audited on the same device needed
−4830 kHz of Stark compensation. These do not.)

---

## ⚠️ Please check your IQ phase sense before the first run

These are genuinely 3D pulses (Ω_y is large, not just a small DRAG correction), and
**the sign of Q matters enormously**:

| arm | as delivered | with Q negated |
|---|---|---|
| B | 7.46e-06 | 6.99e-03 |
| D | 6.24e-06 | 4.00e-02 |
| E | 3.76e-08 | 4.54e-02 |
| F | 5.14e-07 | 7.19e-02 |

This is invisible in any two-level model — flipping Q there is an exact symmetry. It
only shows up on the real three-level transmon, where the anharmonicity sign fixes
the handedness.

**Cheap test**: arm B is a plain Gaussian + DRAG with the quadrature already at the
standard sign for α < 0. If your own DRAG calibration on this qubit converges to the
*opposite* sign of Q, negate the `Q_norm` / `rabi_y_MHz` / `Omega_y_rad_per_ns`
columns everywhere here (and negate `φ_vz` too).

---

## Verifying the files yourself

`verify_waveform.py` needs only numpy and scipy, reads nothing but the CSV (target
gate and `φ_vz` included — they come out of the header), and re-derives the gate:

```
$ python verify_waveform.py Ypi/waveforms/Ypi_novera_E_optimized_ad_exact_vzfree.csv

=== Ypi/waveforms/Ypi_novera_E_optimized_ad_exact_vzfree.csv   [target Y(pi)]
  120 samples, dt = 0.416667 ns (2.4000 GSa/s), T = 50.0000 ns, peak = 25.2790 MHz
  realised gate  W = Rz(-66.2180 deg) . Rx(179.9962 deg) . Rz(-246.2163 deg)
  leakage 1 - tr(B'B)/2                  = 3.3704e-08
  1 - Fbar  vs Y(pi), NO virtual Z      = 3.4575e-08
  1 - Fbar  vs Y(pi), best virtual Z    = 3.4575e-08   at phi_vz = +0.0000 deg
  header declares NO virtual Z is needed -> the first number above is the answer
```

The canonical (non-`vzfree`) file of the same arm gives `1 − F̄ = 6.65e-01` without
the virtual Z and `3.76e-08` with it — that contrast is the whole point.

Run it over everything at once:

```
python verify_waveform.py Xpi/waveforms/*.csv Ypi/waveforms/*.csv
```

---

## What the simulation does and does not include

**Included**: exact three-level transmon propagator, leakage to |2⟩, the AC-Stark
shift, the 2.4 GSa/s sample-and-hold discretisation (quoted numbers are for the
played 120-point grid, not an idealised continuous pulse), amplitude-error and
detuning sweeps, and a Lindblad T1/T2 floor quoted separately.

**Not included**: AWG bandwidth/skew, IQ mixer imbalance and carrier leakage, cable
dispersion, readout error, TLS. Expect the measured number above the simulated one;
the *ranking* between arms is what we would ask you to test first.

---

## Not in this package

X(π/2) and Y(π/2) exist only for our default device model (α/2π = −200 MHz), not yet
re-solved for Novera. Ask and we will produce them the same way — note they will
need the real virtual Z, since the `ψ = −φ_vz/2` shortcut is π-gate-only.

---

Questions on conventions, or a measured number that disagrees with the tables above
by more than ~10×: please send the raw waveform you actually played (post-AWG
capture if you can) and we will re-simulate exactly what the chip saw.
