# Proposal — Novera chip: optimized X(π) and Y(π), plus a 1qb-DB bench

**Deliverable.** Three notebooks and one importable waveform export, for the *measured*
Novera device parameters, built on the machinery the four X/Y proposals already share
(`proposals/_gatelib`). Nothing about the method is new: only the device changes.

Everything below that is a number carries where it came from. Everything that is a
judgement call is marked ★ and says what the alternative was.

## Folder layout (reorganised 2026-08-09)

```
README.md                     this file
run_novera.py                 the N2 solver driver -- ★ must stay at this depth
novera_solve_summary.json       (it resolves REPO_ROOT as HERE.parent.parent)
gates/
  Novera_Xpi/  Novera_Ypi/    armlib + manifest.json + waveforms/  -- the canonical record.
                              ★ folder names are bound in _gatelib/gatespec.py's `folder=`
                              field; gate_dir() finds them by name anywhere under
                              proposals/, so they may move but must not be renamed.
notebooks/                    the three notebooks. They resolve their gate folder as
                              Path.cwd().parent/"gates"/<name>, so run them with the
                              kernel cwd at notebooks/ (they assert if it is not).
delivery/                     the experimentalist-facing package -- see delivery/README.md.
                              Derived from gates/*/waveforms/, never edited by hand.
audits/                       audits of waveforms that came from elsewhere.
```

---

## 0. What is reused, and what actually changes

| | status |
|---|---|
| target gate, arms, recipes (`armspec.ARM_RECIPES`) | **reused verbatim** — a recipe is a *method*, held fixed across devices on purpose |
| optimizer, budget, propagator, DRAG/Stark readout | reused, untouched |
| noise hypothesis (δ_z, ε → the budget weights `w_i`) | **unchanged** (§1) — so the objective function is literally the same one |
| device (α, T1, T2) | **changed** — the only physical input that moves |
| `_gatelib` device binding | ⚠️ **must be fixed first** (§2) |

The solved coefficients from `1qb_Gaussian-drag-optimization_Xpi/` are **not** reusable as
final answers — α moved, so the Hamiltonian moved. They *are* reusable as warm starts,
which is what §4 exploits.

---

## 1. The device — `configs/device_novera.json`

Only the fields that differ from `curve_opt.device`'s documented defaults are written to
the override file; everything else falls back, and an unknown field name raises.

| field | value | source |
|---|---|---|
| `anharmonicity` | **−0.19848** GHz | measured (−198.48 MHz) |
| `t1` | **35 000** ns | measured (35 µs) |
| `t2_echo` | **41 900** ns | measured (41.9 µs) |
| `rabi_max` | 0.050 GHz (default) | not re-measured; sets η and the peak ceiling |
| `sample_rate` | 2.4 GS/s (default) | ⇒ `N_play = 120` at T = 50 ns |
| `gate_time` T | 50 ns (default) | §1.1 |
| `n_modes` M | **20** (default) | §1.1 |
| `static_detuning` δ_z | 1e-4 GHz (default) | §1.2 |
| `control_error` ε | 0.02 (default) | design-time noise assumption, not a test value |

Derived: η = **0.2519**, `fenchel_min_time` = 20 ns, F₀ = θ²/T = π²/50, decoherence floor
= **6.359e-4** (default device: 4.167e-4, so **1.53×**).

★ **T1 and T2 do not enter the optimization.** `t1`, `t2_echo`, `gamma1`, `gamma_phi` and
`decoherence_floor` appear nowhere in `budget.py`, `optimize.py`, `parametrization.py`,
`propagate.py`, `gate.py` or `basis.py` — verified by grep, and consistent with F07's
measurement that the decoherence increment is waveform-independent (4.177e-4 vs 4.238e-4
for two very different inputs). They shift the reported total and nothing else. This is
why the X(π) recipes transfer at all.

### 1.1 ★ M stays at 20, and n\* no longer matches exactly

`n* = 2T|α| = 19.848`, so `Device.resonant_harmonic_matched` is **False** and constructing
the device warns. This is reported, not fixed, and M is **not** retuned. Reason:

```
harmonic spacing 1/(2T) = 10 MHz      spectral resolution 1/T = 20 MHz

  M=18 : top harmonic 0.180 GHz,  −18.48 MHz from resonance,  |n*−M| = 1.848
  M=19 : top harmonic 0.190 GHz,   −8.48 MHz,                 |n*−M| = 0.848
★ M=20 : top harmonic 0.200 GHz,   +1.52 MHz,                 |n*−M| = 0.152
  M=22 : top harmonic 0.220 GHz,  +21.52 MHz,                 |n*−M| = 2.152
```

20 is the closest integer to n\*, and the residual 1.52 MHz is **7.6% of one spectral
resolution element** — at T = 50 ns the top harmonic and the leakage resonance are not
spectrally distinguishable. M = 18 or 22 would each move a full resolution element away
and give up the direct knob C4 uses to cancel the other harmonics' tails. Any C4 claim
from this device states the 0.76% mismatch.

### 1.2 ★ δ_z stays at 100 kHz, and it is a *calibration residual*, not the T2\* value

With both T2\* = 25 µs and T2_echo = 41.9 µs measured, the quasi-static part can be
separated: `exp(−1) = exp(−T2*/T2echo)·exp(−σ²T2*²/2)` gives **σ_δ/2π ≈ 5.7 kHz**, 17×
below the 100 kHz default. δ_z **does** enter the optimization (w₁, w₂ ∝ ⟨δ_z²⟩, and
`ad_robust`'s ∂²_δ term is weighted by it), so this matters: 17× in δ_z is ~300× in weight.

The default is kept for the main run, on the reading that **100 kHz is the drive-frequency
calibration residual an experiment is actually left with**, which is the quantity the
budget is conditioned on — not the chip's intrinsic low-frequency noise. If the intrinsic
limit is wanted, run one extra arm at δ_z = 5.7 kHz as a contrast; it is a device-override
file, not a code change.

---

## 2. ⚠️ Prerequisite — `_gatelib` binds the device at import time

`configs/device_*.json` + `recorder.load_device()` works for `src/curve_opt`, but **it does
not reach the proposal layer**: `_gatelib` captures `DEV = device.DEFAULT_DEVICE` as a
module global (`gatespec.py:76`, `armcore.py:59`) and reads it from ~34 call sites
(`gatelib.py` 28, `armcore.py` 5, `gatespec.py` 1), plus a module-level `T = DEV.gate_time`
that `armcore.Input.broadcast` depends on. There is currently **no way to run a proposal
against a non-default device.**

So step one is a refactor, not a solve: thread a `Device` through `GateLab` and
`armcore.Input` instead of reading the module global.

⛔ This touches the code path every existing gate uses, so the F08 discipline applies in
full: `python _dev_logs/F08_golden.py before` → refactor → `after` + `compare`, criterion
`rtol=0 atol=0` bit-for-bit (267 828 numbers; `proposals/` and `_runs/` are gitignored, so
`git diff` cannot see this and the golden script is the only check).

The Novera gates also need their own `SPECS` keys (`Xpi_novera`, `Ypi_novera`) with folders
under this directory, so their `configs/runs.json` entries do not collide with the
default-device X(π)/Y(π) already registered.

---

## 3. The arms

Baselines A/B are constructed, C→D→E→F are solved, all through `GateLab.solve_arm` with
`ARM_RECIPES` unchanged. Two deltas from the X(π) registration:

- **`e0_cap = "arm_B"`** for arm F, not the literal `7.2e-6`. That number *meant* "arm B's
  zero-noise 1−F̄ on this device"; α moved, so B's level moved, and `armspec.py` already
  says new gates should measure it rather than inherit it.
- **warm start** — see §4.

`cap_mult` needs no rescaling: θ = π and T = 50 ns are unchanged, so F₀ = θ²/T is unchanged
and "cap = 2F₀" keeps its meaning.

**Y(π) is not a second optimization.** `solve_mode="rotate_from_x"`: the three-level
Hamiltonian is covariant under `V(φ) = exp(iφN)`, both noise channels survive the
conjugation, so the Y waveform is the X waveform with the drive phase advanced by π/2 and
every reported number is identical point for point. The Y notebook spends its compute on
`python check_covariance.py Ypi_novera` instead of rediscovering the same optimum.

---

## 4. Execution plan — fast path, fallback, and a stop budget

Measured wall clock for the X(π) chain on the default device (`_runs/*/final.json`):
C 529 s · D 4787 s (X(π/2), 20000 iter) · E 3703 s · F 3107 s ≈ **3.4 h cold**.

### 4.1 ★ Fast path — warm-start each arm from its default-device counterpart

α moved 0.76% and M is identical, so the default-device solution is a very good starting
point. Solve each of D/E/F warm-started from the *same arm* of default-device X(π) rather
than climbing C→D→E→F from Gaussian+DRAG. Expected 10–30 min for F instead of 52.

This is **off-recipe** (`ARM_RECIPES["F"].warm = "E"`), so each run records its provenance
literally — `warm start = 20260808_F06f_adrobust_construction @ alpha=-0.200` — and never
claims it walked the registered chain.

F08b's "you cannot reach F without D and E" does **not** apply here: that was a cold start
from planar arm B for a *different gate* (θ: π → π/2). Same gate, 0.76% device shift, is a
different situation.

### 4.2 Fallback

If a warm-started arm fails to become feasible, fall back to that arm's registered recipe
chain. Cost of the failed attempt is the stop budget below, not an hour.

### 4.3 ★ Stop budget + rewind to the last feasible checkpoint

`maxiter` already stops a run early; what it does **not** do is guarantee the point you
stop on is usable. The optimizer's path is genuinely infeasible for long stretches — from
the recorded arm F run:

```
 iter   max|res_gate|   max|res_c1|     (c1 tolerance is 1e-9)
  100     6.57e-06       7.13e-07       violating, 700×
  300     2.57e-07       4.55e-07       violating
  500     9.46e-07       3.71e-07       violating
  800     1.03e-08       3.47e-18   ←   first clean checkpoint
 5900     1.54e-11       1.39e-17
```

and arm E r2 reaches `res_gate = 1.10e-1` at iteration 100. A c1-violating `a` **cannot be
broadcast at all** — the DRAG/Stark readout refuses it, because its Ω_y endpoint does not
vanish and Λ becomes basis-dependent. F08b lost a whole 6000-iteration run to exactly this.

So the stop is specified in three parts, all **outside** the solver (no change to
`src/curve_opt/optimize.py`):

1. **Budget in wall clock, not iterations.** Per-iteration cost varies 3× across arms
   (C 0.169 s, D 0.239 s, E 0.82 s, F 0.518 s), so an iteration count means different
   things for different arms. Fast-path budgets: C 600 s · D 900 s · E 900 s · F 900 s
   (≈ 55 min worst case for the whole chain).
2. **Rewind.** On stop, scan checkpoints backwards (they are written every 100 iterations
   and already carry `res_gate` and `res_c1_ends`) for the last one with
   `max|res_c1| ≤ 1e-9` and `max|res_gate| ≤` whatever `solve_arm`'s existing gate check
   uses. Register **that** checkpoint as the arm. Pure post-processing, zero extra compute.
   If no checkpoint is feasible, the arm is **not registered** and the run is reported as
   a failure — not silently downgraded.
3. **Re-score before accepting.** ★ F06c already tested "just take an early checkpoint":
   F06-min's iteration-300 point tied with baseline B (0.67% apart) with all 16/49 winning
   points on the ε ≥ 0 half-plane. Early stopping costs real quality. So a rewound
   checkpoint must clear the 7×7 (δ_z, ε) grid against arm B before it is accepted — that
   costs seconds, not hours, and it is a figure the notebook produces anyway.

Every run is recorded with `status='budget_stop'` and the chosen checkpoint's iteration.
It is never reported as `converged`; the project red line stands — a claim needs a
CONVERGED run plus a fine-grid re-evaluation.

---

## 5. The three notebooks

Structure follows `1qb_Gaussian-drag-optimization_Xpi/` but compressed: no re-derivation
of anything `_gatelib` already documents.

1. **X(π)** — played waveforms with the ±Ω_max ceiling; Table 1 (per arm: C1–C4, zero-noise
   1−F̄, leakage, fluence in units of F₀, peak as % of Ω_max, Φ_vz, sampling loss,
   + decoherence floor and total); the 7×7 (δ_z, ε) heatmap and its 1-D slice; the
   winner call under a stated criterion (§7).
2. **Y(π)** — the rotated waveforms, and `check_covariance.py Ypi_novera` as the result.
   Deliberately short: if it needs more than a page, the symmetry argument has broken and
   that is the finding.
3. **1qb-DB** — the XX cycle sequence on the optimized X(π), per `proposals/1qb-DB`.
   Report `envelope_loss` (the ε = 0 reference), `signal`, `n@5%`, the axial-closure table
   for **both** channels, and the drive-frequency scan on a 41-point grid (the reference is
   a fringe pattern, not a bump). ★ At T1 = 35 µs the default n = 0…60 (6 µs) is T1/5.8
   rather than T1/10, so the envelope floor is higher than in the X(π) study; report it
   explicitly rather than quietly rescaling the sequence length.

   ★ **Figures are part of the deliverable, not decoration** — all four from
   `proposals/1qb-DB/README.md` §3, and `plot_db_traces` above all:

   | figure | why it is required |
   |---|---|
   | **`plot_db_traces`** — $P_0(n)$ per arm | ★ **the headline.** "Does the optimized gate oscillate less when the drive amplitude is off?" is the question the whole bench exists to answer, and it is a *picture*, not a table row. The grey dashed line is the ε = 0 reference run; the coloured curves are read **against** it, and on a transmon that reference is not flat. |
   | `plot_played_waveforms` | fixed colours per quadrature — they are different physical channels, not arm identity |
   | `plot_error_curves`, **both** `noise="amplitude"` and `"dephasing"` | the second is not optional: §7 of the 1qb-DB README exists because the Z channel, not the injected ε, dominated arms D/E on the default device |
   | `plot_signal_vs_epsilon` | log axis, **keep the negative ε** — C3 is even in ε to leading order, so any ±ε asymmetry reads out the odd higher-order channel |

   The prose conclusion must answer the headline question **in numbers**: how much slower
   arm F oscillates than arm B at the injected ε, with the saturation cycle and the $P_0$
   swing over the sequence. If that advantage is smaller on Novera than on the default
   device, say so plainly.

---

## 6. Export contract

`Novera_work/gates/<gate>/waveforms/<arm>.json` + `<arm>_120pt.csv`, written through
`recorder` (only module allowed to touch the filesystem).

★ **Φ_vz ships with the waveform.** The solved object is a three-level polar decomposition
`W = R_z(Φ_vz)·U_target`. Exporting (Ω_x, Ω_y) alone hands the user a pulse that is **not**
the target gate. Φ_vz is a required field, and the JSON states the convention in words.

| field | |
|---|---|
| `omega_x`, `omega_y` | rad/ns, on the N_play = 120 AWG grid (`sample_rate × T`) |
| `phi_vz` | rad — the virtual Z the caller must apply |
| `theta`, `phi_axis`, `gate_time`, `sample_rate` | |
| `device` | the full `Device.to_manifest()` blob |
| `run_id`, `status`, `warm_start_provenance` | traceability back to `_runs/` |

Downstream (qeam) landing point is `WaveformPulse(samples, dt)`; note that qeam's pulse
layer is ns/MHz and its α is a **positive** amplitude, opposite in sign to the convention
here. Conversion is out of scope for this folder and is called out so nobody assumes it
was done.

---

## 7. Acceptance

1. `configs/device_novera.json` loads and reproduces §1's derived table (η = 0.2519,
   n\* = 19.848, floor = 6.359e-4), and an unknown field raises.
2. F08 golden regression is bit-for-bit after the §2 refactor (`rtol=0 atol=0`).
3. Every registered arm passes the c1-endpoint and gate-residual feasibility check —
   no arm is registered from an infeasible checkpoint.
4. **The winner is whatever the runs say**, under three criteria reported side by side:
   zero-noise coherent 1−F̄, winning points on the 7×7 grid vs arm B, and axial closure in
   the DB bench. They can disagree — in the default-device study arm C lost the single-gate
   comparison and won the DB one. Arm F winning is the *expectation*, not the spec.
5. `check_covariance.py Ypi_novera` passes at the documented tolerances.

---

## 8. What is not claimed

- **C3 is a diagnostic here, not a claim.** `propagate.tantrix_area` uses the bare-Ω form,
  which is not the tantrix area when τ ≠ 0, so every 3D arm's reported C3 is known to be
  biased. Arm F's objective (`ad_robust`) does not depend on C3, so this does not affect
  the deliverable; the fix is tracked separately as F09d and deliberately not bundled here.
- **Arm F has never formally converged** — three rounds on the default device all stopped
  at `maxiter`, and §4.3 makes early stopping the norm rather than the exception. Physical
  convergence (optimality) is reported; formal convergence is not asserted.
- **The decoherence floor dominates the total.** At 6.359e-4 it is ~4 orders above any
  optimized arm's coherent error, so "total 1−F̄" is not a discriminating headline on this
  chip. The comparison that means something is the **coherent** error and the ε-response.
- **Ω_max was not re-measured** for this chip; η = 0.2519 inherits the default 50 MHz.
- Hardware amplitude calibration (rad/ns → DAC) is out of scope.
