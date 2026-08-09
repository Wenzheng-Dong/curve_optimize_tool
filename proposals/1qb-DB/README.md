# Proposal — 1qb-DB: deterministic benchmarking of the optimized $X(\pi)$

**Deliverable:** one Jupyter notebook. Take the waveforms the single-gate proposal
already produced, run the 1qb-DB protocol on all of them under one three-level
transmon error model, and report what a laboratory would actually see: the survival
probability $P_0(n)$ after $n$ repeated $XX$ cycles, at $\epsilon=\pm3\%$ injected
amplitude error.

This is the *measurability* half of the project. The single-gate proposal reports
$1-\bar F$ down to $5\times10^{-7}$; nothing in a laboratory reads that off one gate.
1qb-DB amplifies the *coherent* part of the error by $2n$, which is exactly the part the
budget's $C_3$ term claims to remove — so it is also the sharpest available test of that
claim. Every configuration it runs is one an instrument can actually produce; nothing is
ever switched off in the model to make a number look clean (§2.4).

Migrated from `pulse-shape-Novera/proposals/1qb-DB` (protocol, error model and readout
definitions from `1qb-DB-demo/db_transmon.py`, git `59fb616`). Sections 1–3 below say
what was kept, what was changed, and why.

---

## 0. What the sequence measures, and what it does not

One DB cycle is the gate pair $XX$; an ideal pair returns $|0\rangle$ to itself, so
$P_0(n)=1$ for all $n$. Write one imperfect gate as $G=R_x(\pi)E$ with
$E=\exp(-i\epsilon\, \mathbf R\cdot\boldsymbol\sigma/2)$ the first-order error rotation
and $\mathbf R=\mathbf R(T)$ the closure vector of the amplitude error curve (§2). Then

$$
G^2=R_x(2\pi)\exp\!\big[-i\epsilon(\mathbf R+\mathbf R')\cdot\boldsymbol\sigma/2\big]+O(\epsilon^2),
\qquad \mathbf R'=2(\hat n\cdot\mathbf R)\hat n-\mathbf R ,
$$

so $\mathbf R+\mathbf R'=2(\hat n\cdot\mathbf R)\hat n$.

★ **Only the component of the closure along the rotation axis survives the pair —
doubled. The transverse part is echoed away.** The first-order prediction is therefore

$$
\theta_{\rm cycle}=2\epsilon\,|\hat n\cdot\mathbf R(T)|,\qquad P_0(n)=\cos^2(n\theta_{\rm cycle}/2),
$$

and for the Gaussian, where $\mathbf R(T)=\pi\hat x$, this reduces to Novera's
$P_0=\cos^2(n\pi\epsilon)$.

The consequence is a scoping statement, not a footnote: **1qb-DB is not a general
robustness meter.** A pulse can have a badly open error curve and still give a flat
trace, and the converse. It measures one projection, amplified — and that is what makes
it cheap and unambiguous. The full $(\delta_z,\epsilon)$ picture stays in
`1qb_Gaussian-drag-optimization/`.

### ★ The axial projection applies to *every* channel, not just the injected one

The echo argument above never used the fact that $\mathbf R$ came from the amplitude
error. It applies verbatim to the quasi-static $Z$ channel, whose first-order error
curve is the design curve $\mathbf r(t)$ itself:

$$
\theta_{\rm cycle}^{(z)}=2\delta_z\,|\hat n\cdot\mathbf r(T)|,\qquad
\theta_{\rm cycle}^{(\epsilon)}=2\epsilon\,|\hat n\cdot\mathbf R(T)| .
$$

Novera could ignore this — its pulses are planar, their $\mathbf r(T)$ is transverse,
and it measured the static detuning four orders below the control error. **That does not
carry over here**, and assuming it would have been the mistake this proposal was one run
away from making. A single-gate infidelity weights the *full* closure isotropically
($C_1=w_1|\mathbf r(T)|^2$ with $w_1=\langle\delta_z^2\rangle/6$, and
$\delta_z/2\pi=0.1$ MHz is small), so an optimizer can leave the $Z$ curve wide open for
almost nothing. The sequence disagrees: it amplifies the axial part of both channels by
$2n$, and a term worth $5\times10^{-5}$ per gate is worth a full swing of $P_0$ after
120 of them.

`axial_table` prints both channels' axial closure side by side with their ratio — a
design-side prediction, computed from the waveform, no simulation involved.
`detuning_scan` then measures the same statement on the device, using the drive
frequency as the knob (§2.4). Read them **before** the DB traces: they say which channel
a given arm's trace is actually reporting on.

---

## 1. Fixed setup

Everything from `src/curve_opt/device.py`; nothing hardcoded here or in the notebook.

| Item | Value |
|---|---|
| Target | $X(\pi)$, winding branch $\theta=\pi$ (`_gatelib`, so `Xpi2`/`Ypi`/`Ypi2` are a one-line change) |
| $T$ | 50 ns — **the same for every arm** |
| $\Delta=\alpha$ | $-1.2566$ rad/ns |
| $\Omega_{\max}$ | $0.31416$ rad/ns, $\eta=0.25$ |
| Played grid | $N_{\rm play}=120$ (2.4 GS/s × 50 ns) — the AWG waveform, not a continuum |
| Cycles | $n=0,\ldots,60$ ⇒ 120 gates ⇒ 6 µs, one tenth of $T_1$ |
| Injected error | $\epsilon=\pm3\%$ (Novera's value, so the two projects' traces compare directly) |
| $T_1$, $T_2^{\rm echo}$ | 60 µs each |
| Static detuning | $\delta_z/2\pi=0.1$ MHz |

★ **Design $\epsilon$ vs. test $\epsilon$ are different quantities.** The design value
`device.control_error = 2%` is a noise-environment assumption entering $w_3$; it shaped
arms C/D/E and is frozen. The test value is the sweep axis, applied in the simulator to
an already-fixed waveform. Re-optimizing per test value would measure something else.

### Arms

Loaded through `proposals/_gatelib`, so the numbers are term-for-term comparable with
the single-gate proposal's Table 1 rather than merely similar.

| Arm | Construction | single-gate $1-\bar F$ |
|---|---|---|
| **A** Gaussian | `planar`, no DRAG | 1.72e-3 |
| **B** Gaussian + DRAG | `planar_drag` — the industry baseline | 7.23e-6 |
| **C** Optimized (3D) | `solve_budget`, run `20260808_F06min…` | 1.81e-4 |
| **D** Opt + fluence cap | run `20260810_F06b…` | 9.01e-6 |
| **E** Opt AD-exact | run `20260810_F06e…_r2` | 4.86e-7 |

Arm A is kept deliberately: a bare Gaussian is not an X gate on a transmon, and
watching it saturate the trace within a few cycles is the calibration-free statement of
what DRAG buys. Arm C is kept for the same reason in the other direction — it is the
arm whose surrogate fell six decades while the truth climbed, and the DB trace is an
independent way to see that.

---

## 2. Four changes from Novera

1. **No gauge fixing.** Novera's waveforms are dimensionless, so it must choose a
   gauge; it chooses equal peak drive, which hands each waveform a different $T_g$
   (22 ns Gaussian, 100 ns composite) and therefore a different decoherence envelope —
   its robust pulse pays a 4× worse envelope loss for its flatter oscillation. Every arm
   here is at $T=50$ ns by construction, so **the envelope is common and the comparison
   isolates the control error with nothing to subtract.** This is the main reason the
   migration is worth doing rather than citing Novera's numbers.

2. **Calibration is virtual-Z only by default.** Novera's waveforms are two-level
   designs replayed on a transmon and need a Rabi + Ramsey calibration before they are X
   gates at all. Arms B–E here are already exact three-level $X(\pi)$ up to the solved
   $\Phi_{\rm vz}$, so the default protocol (`CAL_VZ`) turns only the one free knob.
   The full three-knob laboratory protocol (`CAL_LAB`: amplitude, drive frequency,
   virtual Z, both bounded) is available and used as a **check** — on B–E it must come
   back as a no-op, and `check_calibration_is_free` asserts that. Arm A is expected to
   fail it; that failure is the point.

   ★ **The protocol rule** (`1qb_Gaussian-drag-optimization/README.md` §5.3): calibrate
   **once at $\delta_z=\epsilon=0$ and freeze**. Re-fitting per noise point hands each
   arm a free recalibration and flatters whichever drifts most in phase. The injected
   $\epsilon$ multiplies the *calibrated* amplitude — it is the residual an experiment
   is left with, which is what DB measures.

3. **The per-cycle angle is measured, not inferred.** Novera's headline finding is that
   the two-level closure under-predicts the transmon per-gate error by two orders of
   magnitude for every waveform designed to be robust, because $\epsilon$ also rescales
   the Stark shift $\Omega^2/4|\Delta|$ and the drive-frequency knob cancels that only at
   $\epsilon=0$; the residual is first order in $\epsilon$ however well the curve closes.
   Arms C/D/E were optimized against the **three-level** model and arm E against the
   AD-exact expected infidelity, so this project has a specific reason to expect a
   smaller gap — but "expect" is the operative word. `angle_table` prints predicted and
   measured side by side and their ratio. **A large ratio is a result, not a bug.**

4. ★ **No unphysical configuration anywhere in the reported protocol.** Novera runs each
   sequence twice — once `full`, once `coherent` with $T_1/T_2$ and $\delta_z$ deleted
   from the Hamiltonian — and reports the second as `oscillation`, "the control-error
   signal alone". This proposal carried that over in its first draft and has **removed
   it**. No instrument has a knob that switches $T_1$ off, so a number read off such a
   run is not a measurement; putting it in a results table beside ones that are invites
   quoting a robustness figure no experiment can reproduce.

   Nothing is lost. The two jobs `coherent` was doing are both done with real operations:

   * *separating the signal from the envelope* → **subtract**. An experiment runs the same
     sequence once without the deliberate amplitude error and differences the two traces.
     That is what every entry of `db_readout` now is, and it is what §3's Table 5 reports.
   * *attributing a collapse to a channel* → **turn a knob**. The drive frequency is one.
     Detuning it on purpose adds to the unknown static offset, so the resulting curve is
     the arm's $Z$-channel sensitivity measured the way an experiment would measure it —
     `detuning_scan`, §3's Table 6 and Fig. 4.

   The `dissipation=False` argument survives in `gate_channel` because two of the
   *validation* checks need it (a coherent limit is a statement about numerics, not about
   a device), but no reported quantity uses it.

---

## 3. What the notebook produces

**Table 1 — calibration** (`print_calibration`). Per arm: scale, drive-frequency
correction, virtual Z, rotation error before/after, coherent $1-\bar F$, $P_2$, and
`frame_phase - phi_vz` (must be $\approx0$: the DB frame phase *equals* the single-gate
proposal's frozen $\Phi_{\rm vz}$ — the ladder rotation's qubit block is $R_z(-f)$, so
the sign flips twice. That identity is a convention tripwire; the residual is the
played-grid vs. design-grid sampling loss).

**Table 2 — error-curve geometry** (`print_geometry_table`). Length, $|\mathbf R(T)|$,
$|\hat n\cdot\mathbf R(T)|$, $|\vec A|$ of the **amplitude** error curve on the played
waveform. Geometry alone ranks the arms before any trace is simulated. The dephasing
curve is available from the same function and is reported once as a reminder that the
two channels are different questions.

**Table 3 — axial closure of both channels** (`print_axial_table`). $|\mathbf R_{\rm amp}(T)|$,
$|\hat n\cdot\mathbf R_{\rm amp}(T)|$ and the per-cycle angle it implies, next to the same
three for the $Z$ channel, plus their ratio. §0's scoping statement, as numbers.

**Table 4 — predicted vs. measured per-cycle angle** (`print_gate_table`). The
amplitude channel's logic in six columns; see §2.3.

**Table 5 — DB readout** (`print_db_table`). Two independent readings rather than one
$\Delta P_0$, because decoherence pulls $P_0$ down whether or not the control is robust:

- `envelope_loss` — $1-P_0(N)$ of the $\epsilon=0$ **reference run**: the floor the
  signal is seen against. Common to every arm here, unlike Novera;
- `signal` / `max_signal` — $|P_0(N,\pm\epsilon)-P_0(N,0)|$, and the same difference
  maximized over the sequence: the discriminator;
- `n@5%` — the first cycle at which that difference reaches 5%. Amplitudes saturate once
  the accumulated error passes $\pi/2$; this column keeps ordering the arms afterwards.
  `-` means the sequence never resolved the error at this length, which for the good arms
  is the expected outcome and is itself the result;
- plus $\Delta P_0$ (peak-to-peak of the raw trace) and $P_2(N)$.

**Table 6 — drive-frequency scan** (`print_detuning_scan`). Which channel each trace is
actually reporting, measured with the one knob an experiment has — see §2.4. Per arm:
$P_0$ at the nominal frequency, how far the drive may be detuned before the reference
falls below $1/2$, and how far the $\epsilon$ discriminator moves across the scan.

**Figures.**

1. **Played waveforms**, markers on, with the $\pm\Omega_{\max}$ ceiling. The two
   quadratures carry **fixed** colours rather than the arm colour — they are different
   physical channels ($\Omega_x$ the drive, $\Omega_y$ the DRAG/Stark correction, and
   $\Omega_y\equiv0$ on the `planar` layer) and must read the same way in every panel.
   Arm identity is in the title; the arm colours are for Figs. 2–5, where arms share an
   axis.
2. **Error curves** in 3D and the three plane projections, for *both* channels
   (`noise="amplitude"` and `noise="dephasing"` — given §7, the second is not optional).
   Each curve starts at the origin (○) and ends at $\mathbf R(T)$ (■), so the gap
   between the markers *is* the closure.
3. **$P_0(n)$** per arm, full model. The **grey dashed line is the $\epsilon=0$ reference
   run** — the same sequence without the deliberate amplitude error, which is something
   an experiment simply does. It is what the coloured curves are read against, never a
   result on its own; on a transmon it is not flat, and the vertical gap to the coloured
   curves is the part that is about $\epsilon$. Every entry in `db_readout` is a
   difference against it, because the raw $P_0$ would credit an arm for decay it cannot
   control.
4. **Drive-frequency scan**: $P_0(N)$ of the reference (left) and the $\epsilon$ signal
   (right) vs. applied detuning. ★ The left panel is a **fringe pattern, not a bump** — a
   120-gate sequence is an interferometer on the $Z$ channel, running through a full
   fringe every $\pi/(2n|\hat n\cdot\mathbf r(T)|)\approx0.35$ MHz for arms D and E. The
   grid is 41 points for that reason; the 7-point grid this started with drew straight
   lines through the fringes, which reads as a smooth curve and is not one.
5. **Signal vs. injected $\epsilon$** on a log axis — ★ keep the negative values, since
   $C_3$ is even in $\epsilon$ to leading order and any $\pm\epsilon$ asymmetry reads out
   the odd higher-order channel.

---

## 4. Validation

Right first, then accurate. `run_all_checks` prints all four.

1. **Coherent limit.** With the dissipator off the $9\times9$ channel must equal
   $U^*\otimes U$ of the coherent gate, to $\sim10^{-12}$. Two genuinely separate
   implementations — `curve_opt.gate.three_level_propagator` (jax, closed-form cell
   exponentials) against a `scipy.linalg.expm` chain on a QuTiP-assembled Liouvillian —
   so this pins the vectorization convention, the frame rotation and the Hamiltonian
   sign conventions against the rest of the repository at once.
2. **Matrix-exponential backends** (`check_expm_backends`). `gate_channel` takes all
   $N$ cell exponentials in one vmapped, jitted jax call because the per-cell
   `scipy.linalg.expm` loop is ~30× slower and made the drive-frequency scan the
   dominant cost of the notebook. That is a change of numerical implementation, so it
   is checked like one: same generators, same multiplication order, two different
   exponential routines. Agrees to $3\times10^{-15}$.
3. **Sequence composition** (`check_sequence_composition`). `db_sequence` claims $n$
   cycles is $(\Lambda^2)^n$. The independent path integrates the $2n$-gate
   *phase-advanced* waveform as one Liouvillian product — same discretization, so only
   the composition and the frame bookkeeping are under test. Passes at $10^{-15}$;
   anything above $10^{-12}$ is a bug with nothing else to blame it on. From
   $(FU)^m=F^m\prod_k F^{-k}UF^{k}$ with $F^{-k}U(\Omega)F^k=U(R_{+kf}\Omega)$, gate $k$
   is played with its quadratures advanced by $+kf$; the leftover $F^m$ is diagonal and
   cannot move a population. ★ This check earned its place immediately: the first draft
   had the phase-advance sign backwards, and it failed at $10^{-3}$ for every arm with a
   non-zero virtual Z while arm A, whose $f$ is exactly zero, passed at $10^{-15}$.
4. **Independent integrator** (`check_independent_integrator`). The same concatenated
   sequence through `qutip.mesolve` — cubic-spline interpolation and adaptive steps
   against this module's piecewise-constant midpoint cells. They differ at $O(dt^2)$;
   the observed $2\times10^{-7}$ (arm A) to $7\times10^{-5}$ (arm C) tracks each arm's
   sampling loss. The tolerance is loose on purpose and the **values are reported**, not
   merely thresholded.
5. **Grid convergence.** Re-broadcast every arm at $N=120,240,600$ and re-run the
   readout. $N=120$ is what the hardware plays and is therefore what gets reported; the
   finer grids say how far that is from the continuum the design was solved in.
6. **Calibration headroom** (`check_calibration_is_free`). How much the three-knob
   laboratory protocol can still find on each arm. ★ Read `infidelity_lab` against
   `infidelity_vz_only` before concluding anything from either — see §7.
7. **First-order amplitude response** (`check_closure_prediction`). ★ The residual at
   $\epsilon=0$ is not zero, so comparing a small-$\epsilon$ angle against the closure
   directly measures the residual, not the channel — the first draft did exactly that and
   reported ratios of 10–20 that were entirely artefact. The check uses a symmetric
   difference of the error *vectors*, $[\mathbf v(+\epsilon)-\mathbf v(-\epsilon)]/2\epsilon$,
   which cancels every $\epsilon$-independent and even-order term, and compares it to
   $2(\hat n\cdot\mathbf R(T))\hat n$ **as a vector** — direction included. Ratio $\ne1$
   is the size of the three-level Stark-rescaling correction the two-level closure cannot
   see, so it is reported rather than asserted.

Plus the project red lines, unchanged: `jax_enable_x64` on, midpoint quadrature,
`B @ A` propagator order, device constants only from `device.py`.

---

## 5. Layout and cost

```
proposals/1qb-DB/
  README.md      # this specification
  dblib.py       # arms, error curves, channel, calibration, sequence, readout, checks, figures
  notebook.ipynb # orchestration + four tables + four figures
```

No `waveforms/` directory: arms are rebuilt from `_gatelib` (coefficients in
`_runs/`), which is what makes the grid-convergence check possible at all. The
on-disk CSVs in `1qb_Gaussian-drag-optimization/waveforms/` remain the export contract
for anyone outside this repository.

Run with the **Python (curve)** kernel from this directory. Cost is dominated by
$5\ \text{arms}\times6\ \text{configs}\times120$ exponentials of a $9\times9$ matrix —
seconds, not minutes. `check_sequence_vs_reintegration` and `check_grid_convergence` are
the only slow steps and are opt-in in the notebook.

Headless: `conda run -n curve python proposals/1qb-DB/dblib.py`.

---

## 6. Status and known issues

- Scaffold — **written and smoke-tested end to end; no production run yet.** Every
  table and figure below exists in code; the numbers quoted in §7 come from the smoke
  test at $N_{\rm play}=120$ and have **not** been re-checked on a finer grid, so they
  are indicative, not quotable.
- Hardware-unit conversion and a real experiment — out of scope here; the waveforms
  already carry physical units and the AWG rate, so the remaining step is the
  device-specific amplitude calibration.

### ⚠ Arm E provenance mismatch — needs a decision before any production run

`_gatelib/gatespec.py` points `SPECS["Xpi"].run_ids["E"]` at
`20260810_F06e_adexact_construction`, while the X($\pi$) proposal's own `armlib.py`
(and its published `manifest.json`, notebook and dev-log numbers) use the converged
**round 2** run `20260810_F06e_adexact_construction_r2`. Both directories exist in
`_runs/`. `_gatelib/README.md` states that `SPECS["Xpi"]` mirrors `armlib`'s three
run ids, so this is a pre-existing inconsistency in `_gatelib`, not something this
folder introduced — but since 1qb-DB loads its arms through `_gatelib`, **it is
currently benchmarking round 1 of arm E.** Not fixed here: it is outside this
proposal's scope and touching it silently would change published numbers in another
folder.

---

## 7. Preliminary readings from the smoke test

Indicative only (see §6). Reported because they change what the first production run
should be looking for.

**The $Z$ channel, not the amplitude error, dominates the trace for arms D and E.**
Axial closures at $\epsilon=3\%$, $\delta_z/2\pi=0.1$ MHz:

| arm | $\lvert\hat n\cdot\mathbf R_{\rm amp}\rvert$ | $\theta^{(\epsilon)}_{\rm cycle}$ | $\lvert\hat n\cdot\mathbf r(T)\rvert$ [ns] | $\theta^{(z)}_{\rm cycle}$ | ratio |
|---|---:|---:|---:|---:|---:|
| A | 3.142 | 1.9e-1 | 0.00 | 0 | 0.00 |
| B | 3.130 | 1.9e-1 | 0.15 | 1.9e-4 | 0.00 |
| C | 0.559 | 3.4e-2 | 1.84 | 2.3e-3 | 0.07 |
| D | 1.227 | 7.4e-2 | 21.03 | 2.6e-2 | 0.36 |
| E | 0.644 | 3.9e-2 | 23.53 | 3.0e-2 | 0.76 |

and the drive-frequency scan confirms it **without switching anything off**. $P_0(60)$
of the $\epsilon=0$ reference, and how far the drive may be detuned before it falls
below $1/2$:

| arm | $P_0$ at nominal | detuning to $P_0<1/2$ | signal at nominal | signal min → max over the scan |
|---|---:|---:|---:|---:|
| A | 0.867 | > ±0.5 MHz | 0.965 | 0.958 → 0.969 |
| B | 0.869 | 0.449 MHz | 0.967 | 0.933 → 0.971 |
| C | 0.943 | 0.450 MHz | 0.273 | 0.144 → 0.819 |
| D | 0.454 | **already below at nominal** | 0.796 | 0.722 → 0.934 |
| E | 0.360 | **already below at nominal** | 0.676 | 0.636 → 0.912 |

A and B tolerate the full ±0.5 MHz because their $\mathbf r(T)$ is purely transverse
($17.2\,\hat y$) and the $X$ train echoes it completely. D and E have already lost more
than half of $P_0$ at the nominal frequency, with no detuning applied at all.

★ And the reference is a **fringe pattern, not a bump**: a 120-gate sequence is an
interferometer on the $Z$ channel, running a full fringe every
$\pi/(2n|\hat n\cdot\mathbf r(T)|)\approx0.35$ MHz for D and E. The first version of
this scan used a 7-point grid and aliased those fringes into what looked like a smooth
curve. Read Fig. 4, not a table of samples.

Three consequences for the first run:

1. **The DB experiment does not isolate the amplitude error for D and E.** Their
   `signal` column swings by 30% across the drive-frequency scan, so the number at any
   one working point is not a clean $\epsilon$ reading. Report the detuning-resolved
   figure, not a single point.
2. **The axial projection is a design target nobody has optimized against.** $C_1$ and
   $C_3$ weight the full closure isotropically, so the optimizer spent D and E's budget
   opening $\mathbf r(T)$ along $\hat x$ — free per gate, decisive over 120. If a
   DB-friendly gate is wanted, that is a new constraint
   ($\lvert\hat n\cdot\mathbf r(T)\rvert$ and $\lvert\hat n\cdot\mathbf R(T)\rvert$
   bounded), not a new weight.
3. **Arm C, the arm that lost the single-gate comparison, currently wins the DB one**
   (`n@5%` = 25 vs. 3 for the Gaussian; D and E reach 5% at 5 and 6). Its design curve
   is nearly closed in both channels. Whether that survives a fine-grid re-check is the
   first thing to settle.

**A lab amplitude calibration is worth $2000\times$ to the DRAG baseline.** Check 5
compares the two calibration protocols at $\epsilon=0$:

| arm | virtual-Z only | three-knob lab | gain | scale turned | freq. turned |
|---|---:|---:|---:|---:|---:|
| A | 1.72e-3 | 5.90e-6 | 291× | +0.22% | +0.93 MHz |
| B | 7.23e-6 | **3.35e-9** | 2159× | −0.21% | +1.0 kHz |
| C | 1.73e-4 | 1.72e-4 | 1.0× | +7.6e-5 | −13 kHz |
| D | 8.49e-6 | 8.49e-6 | 1.0× | +6.5e-5 | +0.7 kHz |
| E | 7.43e-8 | 6.70e-8 | 1.1× | +1.0e-4 | +1.0 kHz |

C/D/E gain nothing, which is the intended confirmation that they are already exact.
Arm B gains three orders from a $0.2\%$ amplitude retune — which is precisely what a
laboratory DRAG calibration does, and which the single-gate proposal's virtual-Z-only
protocol never granted it. **Under the calibration a real experiment would run, the
Gaussian+DRAG baseline's nominal infidelity ($3.3\times10^{-9}$) beats arm E's
($6.7\times10^{-8}$).**

This does not touch the robustness claim — DB measures the response to $\epsilon$, not
the nominal residual, and arm B's axial closure is still $5\times$ arm E's. But it does
mean the nominal-infidelity headline in `1qb_Gaussian-drag-optimization/` is comparing
a calibrated optimized arm against an *un*calibrated baseline. Which protocol the
headline should use is a decision about what is being claimed; it is flagged here, not
resolved here.

Still open, as originally posed: whether 60 cycles is the right length. With the $Z$
channel included every arm saturates well inside it; at $\delta_z=0$ arms D and E may
not, and the honest move then is to lengthen the sequence until decoherence — not the
control error — is the limit, and report that crossover as the figure of merit.
