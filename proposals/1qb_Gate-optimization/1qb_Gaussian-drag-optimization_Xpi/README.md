# Proposal — Single-qubit X(π): Gaussian vs. Gaussian+DRAG vs. Optimized

**Deliverable:** one Jupyter notebook. Build three control waveforms for the same
gate, evaluate every budget cost and the *true* simulated fidelity for each, sweep
the two noise channels, produce one table and four figures. Waveforms are exported
to disk so anyone can re-import them without re-running the optimizer.

Plan: build it, run it, then decide what to do about the results. This document
specifies only what is needed to get a correct first run.

---

## 0. What we already know

The optimized arm (arm C) has been run once before, in
`_dev_logs/F06min_beat_gaussian_drag.md`, and re-measured independently with QuTiP
in `_dev_logs/F07_end_to_end.md`:

| | predicted (surrogate) | measured three-level $1-\bar F$ |
|---|---|---|
| Gaussian + DRAG | 6.78e-4 | **6.60e-4** |
| Optimized | 4.19e-10 | **1.64e-3** |

The surrogate fell six orders of magnitude; the real gate got 2.5× worse. Mechanism:
plan §2.5b waives the term $\delta\tau \sim \epsilon\kappa^2/\Delta$ as one order
down in $\eta$ — but once $\mathcal{A}_{\mathbf T}$ is driven to zero, the waived
term is the whole answer (measured 2.2e-3 at $\epsilon=2\%$, scaling as $\epsilon^2$
with a coefficient $4\times10^8$ larger than $C_3$).

So: expect arm C to lose. Run it anyway and report it honestly — this notebook's
job is to make *where and why* it loses visible. Fixing it is the next conversation.

---

## 1. Fixed setup (identical for every arm)

Everything comes from `device.py`; do not hardcode any of it in the notebook.

| Item | Value |
|---|---|
| Target | $X(\pi)$, winding branch $\theta=\pi$ |
| $T$ | 50 ns |
| $\Delta = \alpha$ | $-1.2566$ rad/ns |
| $\Omega_{\max}$ | $0.31416$ rad/ns, so $\eta = \Omega_{\max}/\lvert\Delta\rvert = 0.25$ (a device constant) |
| $M$ | 20 |
| Decoherence floor | $(\Gamma_1+\Gamma_\phi)T/3 = 4.1667\times10^{-4}$ |
| AWG | 2.4 GS/s ⇒ 120 samples per gate |
| Design noise moments | $\delta_z/2\pi = 10^{-4}$ GHz, $\epsilon = 2\%$ |

First pass (Table 1) runs with both noise channels **off in the simulator**
($\delta=\epsilon=0$). The design-time $\epsilon=2\%$ inside the budget weights
stays on — see §5.1.

---

## 2. The arms

| Arm | Layer | Construction |
|---|---|---|
| **A** Gaussian | `planar` | Truncated Gaussian, $\sigma=T/6$, amplitude set so the *truncated numerical* area equals $\pi$; least-squares fit to the $M=20$ sine basis, `c1`-projected. $\Omega_y\equiv0$. |
| **B** Gaussian+DRAG | `planar_drag` | Same $\mathbf a$ as A; the DRAG/Stark readout map supplies $\Omega_y$ and the Stark phase. The industry baseline — first column of every table. |
| **C** Optimized | `general` | `optimize.solve_budget`, warm start = B's $\mathbf a$ with $\mathbf c=0$, $\Phi_0=0$. |
| **D** Opt + fluence cap | `general` | Same as C plus the F06b hard cap $\int\kappa^2dt\le2F_0$ (surrogate validity guard). Loaded from the recorded cap-2$F_0$ run (maxiter checkpoint). |
| **E** Opt AD-exact | `general` | F06e `objective_mode="ad_exact"`: true-model expected infidelity via AD; $C_1$–$C_4$ demoted to report quantities. Cap 4$F_0$ as guard. Converged (r2). |
| **F** Opt AD-robust | `general` | F06f `objective_mode="ad_robust"`: AD sensitivities alone, zero-noise infidelity as a hard fuse $e_0\le7.2\times10^{-6}$; evaluated on the 120-pt AWG grid; no fluence cap, $\rho=0.8$. **Working point, non-claim** (r1 maxiter checkpoint, optimality 9.8e-9; see `_dev_logs/F06f_robust_objective.md` §5). |

Arms A and B: built by `_gatelib/gatelib.py::GateLab.build_gaussian_drag` from
`_gatelib/armspec.py::BASELINE_RECIPES` ($\sigma=T/6$, 8000-point fit); A is the same
construction with the DRAG stage off. (Before F08c this construction lived in
`_dev_logs/F05_budget_validation.py`; the primitives moved to `_gatelib/armcore.py`
with every number unchanged.)

Arm C settings — all from `device.py` defaults, **nothing tuned**:
`gate_level="three_level"` throughout (a `two_level` warm start leaves the real
three-level gate unconstrained — F05b: coherent $1-\bar F$ 1.37e-2 vs 3.59e-8),
`hessian_mode="default"` (F04: Gauss-Newton does not win here), hard constraints =
gate residual (vector-valued, polar decomposition) + peak epigraph + R guard
($\max\lvert\tau\rvert \le 0.5\lvert\Delta\rvert/2$) + `c1` endpoints.

★ On "how do we pick the cost weights": we don't. $w_1=\langle\delta_z^2\rangle/6$,
$w_2=\langle\delta_z^4\rangle/6$, $w_3=2\epsilon^2/3$, $w_4=1/3$ are conditional
weights fixed by the assumed noise moments (`budget.weights_from_device`); changing
them *is* changing the noise assumption. The instinct behind the question — keep
$\eta\ll1$ so unmodelled higher-order terms stay small — is right but cannot act
through weights, since $\eta=0.25$ is a device constant. It would have to act
through a hard guard (e.g. a fluence cap). Not in this notebook; discuss after the run.

---

## 3. What we measure

Two numbers per arm, never conflated:

1. **Predicted** — the weighted budget $\sum_i w_i C_i$.
2. **True** — $1-\bar F$ from a three-level simulation of the *played* waveform,
   with leakage population reported separately.

★ **Evaluation-object rule** (project red line): $C_1,C_2,C_3$ on the **design
curve**; gate residual, $C_4$, peak on the **broadcast (played) waveform**.

★ **$C_4$ trap for arm A**: `budget.budget_terms` applies the DRAG/Stark readout map
unconditionally, so on arm A it would report the DRAG-corrected leakage of an
undragged waveform. For arm A compute $C_4$ directly on the broadcast waveform, as
`armcore.py::Input.predicted_terms` does. On B/C the two paths agree
bit-for-bit.

Decoherence is waveform-independent (F07: 4.177e-4 vs 4.238e-4 across two very
different waveforms), so it enters as the same additive constant everywhere and
cannot change the ranking. Give it its own column; do not fold it in.

**Table 1** (noiseless, ✅ per column):

| Arm | $C_1$ | $C_2$ | $C_3$ | $C_4$ | predicted total | decoh. floor | **true $1-\bar F$ (jax)** | **true (QuTiP)** | leakage | peak/$\Omega_{\max}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| A | | | | | | 4.1667e-4 | | | | |
| B | | | | | | 4.1667e-4 | | | | 48.2% |
| C | | | | | | 4.1667e-4 | | | | 93.1% |

The two "true" columns are independent numerical paths — jax
(`gate.three_level_propagator`, piecewise-constant `expm`) and QuTiP (`qutip_sim`,
adaptive ODE, does not import `curve_opt.gate`). They must agree to ~1e-5 relative
on a fine grid; disagreement is a bug, not a result. F07 references at $N=4000$:
arm B 7.2317e-6, arm C 1.8086e-4. **Print the predicted/true ratio per row** — a row
where the surrogate sits $10^6$ below the truth is the headline, not a footnote.

---

## 4. Fig. 1 — the control waveforms

The first thing to look at, and the cheapest.

* Three panels (one per arm) sharing axes: $\Omega_x(t)$ and $\Omega_y(t)$ in
  rad/ns vs. $t$ in ns over $[0,T]$, with $\pm\Omega_{\max}$ drawn as a dashed
  ceiling so peak utilisation is visible by eye.
* Arm A has $\Omega_y\equiv0$ (plot it anyway, as a flat line — it makes the DRAG
  correction in B legible as a difference).
* Overlay all three $\Omega_x$ on one extra panel for direct shape comparison.
* Plot the **played** waveform (what the AWG would emit), at the 120-point sampling,
  markers on — the sampling itself is part of what we are comparing.

Add a second row of panels for the design-curve quantities of arm C, $\kappa(t)$ and
$\tau(t)$, with the R-guard bound $\rho\lvert\Delta\rvert/2$ drawn in — this is where
the optimizer's torsion, the thing arms A/B do not have at all, becomes visible.

---

## 5. Noise sweep

### 5.1 Design $\epsilon$ vs. test $\epsilon$ — different quantities

* **Design $\epsilon$** = `device.control_error` = 2%: a noise-environment
  assumption entering $w_3$, so it shapes arm C's waveform. Held fixed at 2%.
* **Test $\epsilon$** = the sweep axis: an actual amplitude error
  $(\Omega_x,\Omega_y)\to(1+\epsilon)(\Omega_x,\Omega_y)$ applied in the simulator
  to an already-fixed waveform.

Sweeping the test axis with the design value fixed measures robustness. Re-optimizing
per test value would measure something else.

### 5.2 Grids

* Test $\epsilon \in \{-5,-3,-1,0,1,3,5\}\%$. ★ Keep the negative values: $C_3$ is
  even in $\epsilon$ to leading order, so **any asymmetry between $\pm\epsilon$ reads
  out the odd higher-order channel** — that asymmetry is a result.
* Static detuning $\delta_z/2\pi \in \{0,\pm0.1,\pm0.3,\pm0.5\}$ MHz (the design
  assumption is 0.1 MHz rms, so this reaches $5\sigma$). Confirm the range against
  the Novera device data before the production run.

Both channels are quasi-static, so a deterministic grid is the right treatment — no
Monte-Carlo needed.

### 5.3 ★ Calibration protocol — the easiest place to cheat by accident

Calibrate the virtual-Z angle $\Phi_{\rm vz}$ **once per arm at $\delta=\epsilon=0$
and freeze it** for the whole sweep — that is what a real experiment does. Re-fitting
at every noise point hands each arm a free recalibration and flatters whichever arm
drifts most in phase. Frozen values go in the headline figures; the refit variant may
appear as a clearly labelled sensitivity check. (`qutip_sim.fit_phi_vz` refits;
freezing = pass the nominal value to `channel_infidelity`.)

### 5.4 Fig. 2 — heatmaps

One heatmap per arm over $(\delta_z,\epsilon)$, colour = true $1-\bar F$, **shared
log colour scale** across arms. Plus one difference map,
$\log_{10}[(1-\bar F)_C/(1-\bar F)_B]$, so "better than baseline" is the sign of
the colour.

### 5.5 Fig. 3 — 1-D slice

$1-\bar F$ vs. $\epsilon$ at $\delta_z=0$, all three arms on one log axis, with each
arm's predicted $C_3$-only curve overlaid dashed. For arm C the gap between dashed
and solid is ~$4\times10^8$; this figure is where that is legible.

---

## 6. Fig. 4 — the divergence plot

Checkpoint arm C every 100 iterations (`checkpoint_every=100`, recorder already
supports it) and store per checkpoint: the surrogate total and each $C_i$; the
**true** three-level $1-\bar F$ of the played waveform at that iterate; and
diagnostics (gate residual, peak utilisation, $\max\lvert\tau\rvert$ / guard bound,
$\int\kappa^2dt$).

Plotted on one log axis, the surrogate falls through six decades while the truth
turns around and climbs. The turning point marks where the first-order framework
leaves its own domain of validity, and whichever diagnostic correlates with the turn
is the direct input to fixing it later.

Recompute the true $1-\bar F$ offline from the RunRecord after the solve (~30
checkpoints × one propagator each), not inside the objective.

---

## 7. Discipline and on-disk contract

Non-negotiable (`AGENTS.md` / `CLAUDE.md`): `jax_enable_x64` on; midpoint quadrature;
`B @ A` propagator order; a quoted run must be **CONVERGED** plus fine-grid re-checked
(4×–25×); every optimization goes through the recorder into `_runs/<run_id>/`
(`schema="budget"`) and **the notebook plots only from RunRecord**; the gate stays a
hard vector-valued constraint, never a scalar $1-\bar F=0$; temp scripts in the
scratchpad, not the repo root.

The notebook orchestrates and displays; anything expensive or reusable (arm
construction, cost evaluation, sweep driver) lives in an importable module, so the
notebook re-executes top-to-bottom in minutes off persisted runs.

```
proposals/1qb_Gaussian-drag-optimization/
  README.md                       # this specification
  armlib.py                       # arms, costs, sweep, trajectory, export
  notebook.ipynb                  # orchestration + the four figures + Table 1
  waveforms/
    A_gaussian_120pt.csv          A_gaussian.json        # coeffs a, c, Phi_0, phi_vz, layer, run_id
    B_gaussian_drag_120pt.csv     B_gaussian_drag.json
    C_optimized_120pt.csv         C_optimized.json
  manifest.json                   # arm -> {csv, json, run_id, git commit, sampling loss}
```

Run it with the **Python (curve)** kernel, from this directory. Whole notebook is
~40 s: arm C is loaded from its recorded run (`RESOLVE_ARM_C = True` re-solves it in
~9 minutes instead).

CSV schema is F07's, unchanged: `time_ns, Omega_x_rad_per_ns, Omega_y_rad_per_ns`,
120 rows, midpoint sampling (`qutip_sim.export_waveform_csv` / `load_waveform_csv`).
Report each arm's sampling loss (120-point vs. fine-grid $1-\bar F$; F07 measured
0.03% for B, 2.4% for C) — state the number, set no threshold.

---

## 8. Existing assets — reuse, don't rebuild

| Need | Where |
|---|---|
| Device parameters (single source of truth) | `src/curve_opt/device.py` |
| Design curve, DRAG/Stark readout, three layers | `src/curve_opt/parametrization.py` |
| $C_1$–$C_4$, weights from noise moments | `src/curve_opt/budget.py` |
| Three-level propagator, polar-decomposition gate residual | `src/curve_opt/gate.py` |
| Budget optimizer, `BudgetProblem`, fine-grid re-evaluation | `src/curve_opt/optimize.py` |
| Independent QuTiP simulation, Lindblad, CSV export | `src/curve_opt/qutip_sim.py` |
| Run persistence and offline reload | `src/curve_opt/recorder.py` |
| Arm A/B construction, the `Input` abstraction | `proposals/_gatelib/armcore.py`, `armspec.py` |
| What each arm is (objective, caps, guards, grids) | `proposals/_gatelib/armspec.py::ARM_RECIPES` |
| Which recorded run backs which arm | `configs/runs.json` |
| Arm C reference construction and its measured numbers | `_dev_logs/F06min_construction.py`, `F06min_validation_results.json` |
| End-to-end QuTiP comparison + reference waveforms | `_dev_logs/F07_end_to_end.py`, `F07_waveforms/` |

Reading order: `AGENTS.md` → `_plan_full_cost.md` (v5.1) →
`_dev_logs/F06min_beat_gaussian_drag.md` → `_dev_logs/F07_end_to_end.md`.
