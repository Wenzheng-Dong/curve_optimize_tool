# Single-qubit π-gate waveforms for the Novera transmon — **1 GSa/s** delivery package

X(π) and Y(π), 50 ns, **1.0 GSa/s → 50 samples, dt = 1 ns**. Regenerated 2026-08-10
from the same solved arms as the 2.4 GSa/s package (run set
`20260809_Xpi_novera_arm{C,D,E,F}_accepted`), by
`proposals/Novera_work/make_delivery.py --sample-rate 1.0`.

This supersedes the 2.4 GSa/s package for playback on a 1 GSa/s AWG. Everything
below is the same physics; only the playback grid changed.

---

## What changed, and what did not

**Not a resampling of the 120-point files.** Each arm is a set of sine-basis
coefficients defining a *continuous* pulse. We re-evaluated that continuous curve on
the 50-point midpoint grid, so no interpolation error was introduced. The design
curve is untouched — same solve, same coefficients.

**Re-measured, not re-scaled.** Every number quoted in the CSV headers and here —
infidelity, leakage, peak amplitude, `φ_vz`, the whole robustness grid — was
re-simulated on the 50-sample waveform that the file actually contains. `φ_vz` is
fitted on those exact samples, so `verify_waveform.py` reproduces the header value.

**Bottom line: every usable arm stays far below the decoherence floor.** (Arm A is the
deliberately un-corrected no-DRAG baseline and sits above it, at both sample rates —
that is what it is there to show.)

| arm | 1−F̄ @2.4 GSa/s | 1−F̄ @1 GSa/s | worst over noise grid @2.4 | worst @1 GSa/s | peak Rabi | φ_vz | ψ = −φ_vz/2 |
|---|---|---|---|---|---|---|---|
| A | 1.74e-03 | 1.74e-03 | 6.77e-03 | 6.76e-03 | 24.06 MHz | −0.000° | +0.000° |
| B | 7.45e-06 | 7.44e-06 | 4.64e-03 | 4.64e-03 | 24.06 MHz | +7.762° | −3.881° |
| C | 6.14e-05 | 5.37e-05 | 9.02e-03 | 9.09e-03 | 45.05 MHz | −7.416° | +3.708° |
| D | 4.16e-06 | 5.15e-06 | 3.95e-03 | 3.93e-03 | 24.23 MHz | +145.825° | −72.912° |
| **E** | **2.32e-08** | **6.21e-08** | 4.30e-03 | 4.32e-03 | 25.20 MHz | +173.644° | −86.822° |
| **F** | **3.42e-07** | **4.70e-05** | 3.86e-05 | **5.53e-05** | 42.80 MHz | −150.132° | +75.066° |

> Both columns use the **same** convention, so they are directly comparable: `φ_vz`
> re-fitted on the played samples at that sample rate. The 2.4 GSa/s package's own
> README quotes slightly different figures for the same waveforms (F 5.39e-07,
> C 9.21e-05, E 3.46e-08) because it inherited `φ_vz` from the finer design grid
> instead of re-fitting. The waveforms are identical — only the frame the number is
> measured against differs. Nothing here says the coarser grid *improved* any arm.

The T1/T2 floor for a 50 ns gate on this device (T1 = 35 µs, T2echo = 41.9 µs) is
**6.36e-04**. At 1 GSa/s, at nominal amplitude and detuning, B/C/D/E/F clear that
floor by 85× / 12× / 124× / 10240× / 14× respectively, and the worst arm-F point on
the *whole* noise grid is still **11× below** it. You should still be measuring
decoherence, not the pulse.

As at 2.4 GSa/s, B/D/E are amplitude-sensitive: at the ±5 % corners of the grid they
do rise above the floor (to ~4e-03). That is unchanged by the sample rate and is
exactly why arm F exists.

**A, B, C, D, E are essentially unaffected** by the coarser grid (A/B are smooth
Gaussians; C/D/E move by well under an order of magnitude and E stays at 6e-8).

### ⚠️ Arm F: play it at the delivered amplitude. Do NOT calibrate its amplitude down.

Arm F is the one arm whose *nominal* number moves a lot: 3.4e-07 → 4.7e-05. Read this
before you tune it, because the obvious tuning is the wrong move.

**Arm F's selling point is its ceiling, not its floor**, and the ceiling barely moved:

| arm F, 1 GSa/s vs 2.4 GSa/s | @2.4 GSa/s | @1 GSa/s | change |
|---|---|---|---|
| floor (nominal, ε = δ = 0) | 3.4e-07 | 4.7e-05 | ×138 worse |
| **ceiling (worst over the whole ±5 % / ±0.2 MHz grid)** | **3.9e-05** | **5.5e-05** | **×1.4 worse** |

The floor is arm E's job. Arm F exists to bound the *worst* case, and that bound is
essentially unchanged. At 1 GSa/s arm F is still the **only** arm that stays under the
decoherence floor across the entire grid:

| arm, at 1 GSa/s | nominal | worst over grid | vs 6.36e-04 floor |
|---|---|---|---|
| B Gaussian + DRAG | 7.44e-06 | 4.64e-03 | 0.1× — above floor |
| E optimised, exact-gate | 6.21e-08 | 4.32e-03 | 0.1× — above floor |
| **F robust, as delivered** | 4.70e-05 | **5.53e-05** | **11.5× below** ✅ |

So arm F's worst case is still **84× better than Gaussian + DRAG's**. Nothing about
its purpose was lost by dropping to 1 GSa/s.

**Why arm F and not the others.** Not bandwidth — every optimised arm has the same
spectral width (99.9 % of the drive power inside 200 MHz, against a 500 MHz Nyquist at
1 GSa/s), so nothing is being aliased. Two things combine: arm F's sample-and-hold
perturbation on the propagator is the largest of any arm (‖ΔU‖ = 1.7e-02 vs 1.7e-03
for E), driven mainly by its 1.7× higher peak amplitude; and that perturbation happens
to point almost exactly along the *amplitude* direction — the continuous pulse at
+6 % amplitude gives 4.4e-05, against the measured 4.7e-05 at 1 GSa/s. Arm F is the
amplitude-robust arm, so it absorbs that perturbation for 4.7e-05. The same
perturbation on arm E would cost 5.4e-03. **Arm F's robustness is what contained this,
not what failed.**

**The tempting fix, and why not to take it.** Because the perturbation looks like a
+6 % amplitude offset, rescaling the drive to ≈ ×0.935 does recover the nominal
number. It is not free — it trades away exactly the property you chose arm F for:

| arm F operating point, 1 GSa/s | nominal | worst over grid | vs floor | vs B's worst |
|---|---|---|---|---|
| **×1.000, 0 kHz (as delivered)** | 4.70e-05 | **5.53e-05** | **11.5× below** | **84× better** |
| ×0.935, +100 kHz | 1.08e-06 | 2.52e-04 | 2.5× below | 18× better |

Nominal improves 40×, **but the worst case gets 4.6× worse** and the grid stops being
flat. The reason is straightforward: arm F's flat basin was designed around ×1.000, so
moving the operating point to 0.935 makes the ±5 % window span 0.89–0.98, which reaches
outside that basin. Chasing arm F's nominal number turns it into an arm-E-shaped
trade — and if that is what you want, just run arm E, which does it better (6.2e-08).

**Therefore:**

- **Run arm F at the amplitude in the file.** If your Rabi calibration wants to pull it
  ~6 % low, that is the calibration optimising the nominal point at the expense of the
  robustness you picked this arm for. Pin the amplitude instead.
- Its flat-basin centre being at the delivered amplitude is the whole design; do not
  re-centre it by hand.
- Use **arm E** when you are well calibrated and want the best number, **arm F** when
  you want the guarantee. That division of labour is unchanged from 2.4 GSa/s.

### Robustness, 1−F̄ over (amplitude error ε, drive detuning) — at 1 GSa/s

**arm F** — flat, stays below 5.6e-05 everywhere on the grid

| ε \ δ | −0.2 MHz | −0.1 | 0 | +0.1 | +0.2 |
|---|---|---|---|---|---|
| −5% | 3.15e-05 | 1.76e-05 | 9.29e-06 | 6.90e-06 | 1.07e-05 |
| −2% | 4.60e-05 | 3.90e-05 | 3.61e-05 | 3.75e-05 | 4.34e-05 |
| **0** | 5.24e-05 | 4.81e-05 | **4.70e-05** | 4.94e-05 | 5.53e-05 |
| +2% | 4.66e-05 | 4.37e-05 | 4.33e-05 | 4.54e-05 | 5.03e-05 |
| +5% | 2.36e-05 | 2.05e-05 | 1.89e-05 | 1.88e-05 | 2.03e-05 |

**arm E** — sharp optimum, best if you are well calibrated. Grid is unchanged from
2.4 GSa/s (its sensitivity is amplitude, not sampling).

| ε \ δ | −0.2 MHz | −0.1 | 0 | +0.1 | +0.2 |
|---|---|---|---|---|---|
| −5% | 3.33e-03 | 3.55e-03 | 3.78e-03 | 4.02e-03 | 4.27e-03 |
| −2% | 4.30e-04 | 5.08e-04 | 5.98e-04 | 7.00e-04 | 8.13e-04 |
| **0** | 2.87e-05 | 7.62e-06 | **6.21e-08** | 5.52e-06 | 2.35e-05 |
| +2% | 8.44e-04 | 7.22e-04 | 6.15e-04 | 5.21e-04 | 4.42e-04 |
| +5% | 4.32e-03 | 4.04e-03 | 3.78e-03 | 3.54e-03 | 3.31e-03 |

**Which to run.** Unchanged advice: take **B** first (textbook Gaussian + DRAG, there
to confirm conventions, scaling and readout line up), then **E** for the best nominal
number, then **F** if your amplitude calibration drifts — **running F at the delivered
amplitude, not at the amplitude your nominal-point calibration prefers** (see the arm-F
section above).

---

## ⚠️ The one thing that must not be missed

**Playing a waveform file alone does not give you the target gate.**

The optimiser solves for the qubit-block polar decomposition

```
W  =  R_z(φ_vz) · U_target
```

so every pulse leaves a residual frame rotation `φ_vz` that has to be applied as a
**virtual Z (frame update) after the pulse**. `φ_vz` is *not* small — for arm E it is
+173.6°. Forgetting it turns a 6×10⁻⁸ gate into a 0.66 error, i.e. an essentially
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

Note the `ψ` values differ slightly from the 2.4 GSa/s package (e.g. arm F
+74.997° → +75.066°), because `φ_vz` is the frame realised by *these* samples.
Use the values in these files, not the old ones.

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

**X(π) and Y(π) share every figure of merit exactly.** Y(π) is not an independent
solve: it is the X(π) waveform played with the drive phase advanced by
`φ_axis = π/2`, an exact frame change.

---

## Sample count — please confirm this suits your AWG

50 ns at 1 GSa/s is **exactly 50 samples**. Some AWGs require the waveform length to
be a multiple of 4/8/16, or have a minimum length. If 50 is awkward for yours, tell
us the granularity constraint — we would rather re-solve at a compatible gate
duration than have you zero-pad or truncate, since either changes the gate.

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
`t₀ = dt/2 = 0.5 ns`, `dt = 1.0 ns` (1 GSa/s), total `T = 50.000 ns`.

**Drive frequency**: **on resonance with the bare f01. No offset needed.** The
optimisation ran against the exact three-level propagator, so the AC-Stark shift
from |2⟩ is already absorbed into the pulse shape. Re-checked on the 50-sample
grid — the residual optimal drive detuning is:

| arm | amplitude minimising *nominal* | detuning minimising *nominal* | 1−F̄ there | 1−F̄ as delivered |
|---|---|---|---|---|
| B | ×1.000 | 0 kHz | 7.44e-06 | 7.44e-06 |
| D | ×1.000 | 0 kHz | 5.15e-06 | 5.15e-06 |
| E | ×1.000 | 0 kHz | 6.21e-08 | 6.21e-08 |
| F | ×0.935 | +100 kHz | 1.08e-06 | 4.70e-05 |

From a converged 2-D scan (amplitude ×0.90–1.06 in steps of 0.005, detuning ±300 kHz
in steps of 25 kHz). **B, D and E are already at their optimum as delivered** — no
amplitude or frequency adjustment needed, exactly as at 2.4 GSa/s. No arm needs a Stark
offset. (For contrast, a composite BB1 π-pulse we audited on the same device needed
−4830 kHz of Stark compensation. These do not.)

> ⚠️ **Arm F's row is a description, not a recommendation.** It is the sampling effect
> discussed in the arm-F section above, not a Stark shift, and moving to it costs 4.5×
> on the worst case over the noise grid. Run arm F at ×1.000 / 0 kHz as delivered. This
> table minimises the *nominal* point only; for arm F that is the wrong objective.

---

## ⚠️ Please check your IQ phase sense before the first run

These are genuinely 3D pulses (Ω_y is large, not just a small DRAG correction), and
**the sign of Q matters enormously**. Re-measured on the 50-sample grid:

| arm | as delivered | with Q negated |
|---|---|---|
| B | 7.44e-06 | 6.97e-03 |
| D | 5.15e-06 | 2.65e-02 |
| E | 6.21e-08 | 3.04e-02 |
| F | 4.70e-05 | 6.67e-02 |

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
$ python verify_waveform.py Xpi/waveforms/Xpi_novera_F_optimized_ad_robust_vzfree.csv

=== Xpi/waveforms/Xpi_novera_F_optimized_ad_robust_vzfree.csv   [target X(pi)]
  50 samples, dt = 1.000000 ns (1.0000 GSa/s), T = 50.0000 ns, peak = 42.7959 MHz
  realised gate  W = Rz(+153.5335 deg) . Rx(179.0534 deg) . Rz(+153.5335 deg)
  leakage 1 - tr(B'B)/2                  = 2.3149e-06
  1 - Fbar  vs X(pi), NO virtual Z      = 4.7810e-05
  1 - Fbar  vs X(pi), best virtual Z    = 4.7810e-05   at phi_vz = +0.0000 deg
  header declares NO virtual Z is needed -> the first number above is the answer
```

Run it over everything at once:

```
python verify_waveform.py Xpi/waveforms/*.csv Ypi/waveforms/*.csv
```

**One expected small discrepancy.** `verify_waveform.py` quotes the sub-unitary
average gate fidelity `(|tr(U†B)|² + tr(B†B))/6`, which folds the leakage in, while
the header's "three-level coherent 1 − F̄" is the coherent number with leakage
reported separately on the next line. The two differ by roughly the leakage — e.g.
arm F, 4.70e-05 (header) vs 4.78e-05 (verifier). Both propagators agree to 3e-15;
this is a choice of figure of merit, not a disagreement about the physics.

---

## What the simulation does and does not include

**Included**: exact three-level transmon propagator, leakage to |2⟩, the AC-Stark
shift, the **1 GSa/s sample-and-hold discretisation** (quoted numbers are for the
played 50-point grid, not an idealised continuous pulse), amplitude-error and
detuning sweeps, and a Lindblad T1/T2 floor quoted separately.

**Not included**: AWG bandwidth/skew, IQ mixer imbalance and carrier leakage, cable
dispersion, readout error, TLS. Expect the measured number above the simulated one;
the *ranking* between arms is what we would ask you to test first.

Worth noting at this sample rate: the pulse's highest design harmonic is 200 MHz
against a 500 MHz Nyquist limit, so the waveform is well inside band — but your
AWG's reconstruction filter now sits much closer to the signal than it did at
2.4 GSa/s. If you can capture the played waveform, send it and we will re-simulate
exactly what the chip saw.

---

## Not in this package

X(π/2) and Y(π/2) exist only for our default device model (α/2π = −200 MHz), not yet
re-solved for Novera. Ask and we will produce them the same way — note they will
need the real virtual Z, since the `ψ = −φ_vz/2` shortcut is π-gate-only.

---

Questions on conventions, or a measured number that disagrees with the tables above
by more than ~10×: please send the raw waveform you actually played (post-AWG
capture if you can) and we will re-simulate exactly what the chip saw.
