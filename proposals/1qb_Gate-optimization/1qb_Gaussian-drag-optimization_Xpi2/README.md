# Proposal — Single-qubit $X(\pi/2)$: Gaussian vs. Gaussian+DRAG vs. Optimized

The $X(\pi)$ study repeated at half the rotation angle. **Read
`../1qb_Gaussian-drag-optimization/README.md` first** — the method, the arms, the
evaluation-object rule, the calibration protocol and the on-disk contract are all
unchanged and are not restated here. This file records only what is different, and why
this is worth a separate run rather than a rescaling of the $X(\pi)$ answer.

Status: **architecture only.** No solve has been run; `waveforms/` and `manifest.json`
are produced by the notebook's export cell once arms C/D/E exist.

---

## 1. What changes

Exactly two things:

1. **$\theta = \pi/2$.** `optimize.BudgetProblem` already takes an arbitrary `theta`
   against `propagate.target_x`, so no `src/` change is needed.
2. **The fluence floor $F_0 = \theta^2/T = 0.049348$ rad²/ns**, not the hard-coded
   $\pi^2/T$. The F06b/F06e caps are quoted as multiples of $F_0$, and the multiple is
   what carries the meaning. Using $\pi^2/T$ here would silently turn "cap $=2F_0$" into
   "cap $=8F_0$" and switch the validity guard off.

Everything else is held fixed *on purpose*: $T=50$ ns, $M=20$, $\Omega_{\max}$, the R
guard, the decoherence floor, and the budget weights $w_1\ldots w_4$ — which are
conditional on the assumed noise moments, so retuning them would be changing the noise
model rather than the gate.

## 2. Why this is not a rescaling of $X(\pi)$

A $\pi/2$ Gaussian carries half the area, so **its peak sits at 24.1 % of
$\Omega_{\max}$ instead of 48 %** (measured, arms A and B). The peak epigraph is nowhere
near active at the warm start, and the optimizer has roughly twice the amplitude
headroom it had for $X(\pi)$ — the same direction that drove arm C out of the surrogate's
domain of validity in the first place.

So the question this folder answers is not "does the ranking survive at $\pi/2$" but:

* **Which guard binds?** With the peak bound slack, the fluence cap should be doing all
  the work. `run_construction.py` prints `peak_frac` and `fluence / F0` at the end of
  every solve; that is where to look.
* **Does the surrogate/truth gap shrink?** The F06d screening located the gap in the
  waived $\delta\tau\sim\epsilon\kappa^2/\Delta$ term. If that diagnosis is right, the gap
  should fall roughly with $\kappa^2$, i.e. ~4× at half the angle. If it does not, the
  diagnosis is incomplete and that is a result.
* **Is arm E still worth its complexity?** The AD-exact objective was introduced because
  the design-curve surrogate misstated the sensitivities. At smaller $\kappa$ the
  surrogate should be closer to honest, so arm E's margin over arm D should narrow. How
  much it narrows measures how much of E's win was $\eta$-driven.

Zero-noise reference points already measured, for orientation:

| arm | peak/$\Omega_{\max}$ | true $1-\bar F$ | fluence/$F_0$ |
|---|---|---|---|
| A Gaussian | 24.1 % | 8.454e-05 | 1.712 |
| B Gaussian+DRAG | 24.1 % | 1.137e-07 | 1.712 |

Note arm B already sits at 1.1e-07, i.e. **two decades below the $X(\pi)$ baseline's
7.2e-06**, and three decades below the decoherence floor 4.167e-04 that every arm carries
additively. Any coherent-error improvement the optimized arms buy here is invisible in a
total-error sense; the honest framing of this run is a *coherent-error* comparison, and
the report should say so rather than quoting totals that the floor dominates.

## 3. Running it

Two ways in, **one recipe**. `_gatelib/gatelib.py::ARM_RECIPES` holds what each arm is
(warm start, cap multiple, `objective_mode`, `maxiter`); both entry points call
`GateLab.solve_arm`, so a notebook solve and a script solve are the same run.

**In the notebook** — the cell right after the arms cell:

```python
SOLVE = ["C"]        # or ["C", "D", "E"] to chain the whole lineage in one session
DRY_RUN = False
```

Chaining works because `E` warm-starts from `D`: the cell passes the in-memory arm, so
you skip the register-and-restart round trip.

**From the shell** — same thing, better for the long ones:

```bash
conda activate curve
cd proposals/1qb_Gaussian-drag-optimization_Xpi2

python run_construction.py C --dry-run   # prints the manifest, solves nothing
python run_construction.py C             # F06-min recipe, no cap        (~10 min)
python run_construction.py D             # + hard fluence cap 2 F0       (long: maxiter 20000)
python run_construction.py E             # AD-exact objective, cap 4 F0  (short, warm from D)
```

Arm D is the one to run from the shell rather than a notebook cell — a kernel holding a
20000-iteration solve is a liability.

Either way the run is **recorded to `_runs/<run_id>/`**. That is not ceremony: a solve
that lives only in kernel memory cannot be plotted from a RunRecord, re-checked on a
finer grid, or quoted, and recording means you pay for it once. Paste the printed
`run_id` into `../_gatelib/gatespec.py`'s `SPECS["Xpi2"].run_ids` and every later
kernel loads it instead of re-solving — including `_Ypi2`'s, which reads the same entry.

Arm D warm-starts from B and arm E from D, matching the $X(\pi)$ lineage
(`_dev_logs/F06b_construction.py`, `F06e_construction.py`). Arm C is deliberately left
unguarded — it is the control that shows whether the $\pi/2$ gate reproduces the
divergence at all.

★ If a solve stops early (`maxiter`, or an interrupted kernel), `solve_arm` prints a
loud warning that the arm violates the **`c1` endpoint equality constraint** and returns
without characterizing it. That arm cannot be broadcast at all — the DRAG/Stark readout
layer refuses it by design, since a `c1`-violating `a` gives a basis-dependent
$\Omega_y$ endpoint. The run is still on disk; re-run with a larger `maxiter`. Without
that check the failure surfaces much later as a `ValueError` deep inside
`parametrization.check_c1`, which reads like a plumbing bug rather than "the solve is
not finished".

## 4. Acceptance

Unchanged project red lines apply (`AGENTS.md`, `CLAUDE.md`): `jax_enable_x64` on,
midpoint quadrature, `B @ A` propagator order, the gate as a hard vector-valued
constraint, every run through the recorder, the notebook plotting only from RunRecord.

Additionally, before any number from here is quoted:

* the solve must report **CONVERGED**, or the maxiter caveat must be stated with the
  tail drift, as arm D's $X(\pi)$ run did;
* the jax/QuTiP cross-check in Table 1 must agree to ≲1e-3 relative — disagreement is a
  bug, not a result;
* the sampling loss (120-point vs. fine-grid $1-\bar F$) is reported, not thresholded.

## 5. Open question, left open

$T=50$ ns is kept for comparability with the $X(\pi)$ study, but a $\pi/2$ gate has no
physical reason to take as long as a $\pi$ gate. The decoherence floor is $\propto T$
while the coherent terms are not, so the $T$ that minimises *total* error is a different
optimization — worth its own run, not a footnote here. Flagged, not resolved.
