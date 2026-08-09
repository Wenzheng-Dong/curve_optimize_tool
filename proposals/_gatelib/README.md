# `_gatelib` — the shared machinery behind the four single-qubit proposals

The $X(\pi)$ study in `1qb_Gaussian-drag-optimization/` turned out to be almost entirely
gate-independent: five arms, one Table 1, four figures, one export contract. Only two
things actually knew which gate was being built — the rotation angle $\theta$ and the
target matrix. This package lifts those two out, so the other three gates are a
configuration entry rather than a copy of `armlib.py`.

```
_gatelib/
  gatespec.py          GateSpec + the four registered gates + the frame-rotation helper
                       + the configs/runs.json registry loader/writer
  armspec.py           ArmRecipe / BaselineRecipe -- what each arm *is*, all six of them
  armcore.py           Input, fit_phi_vz, c1_project_and_rescale, fidelity_jax
  gatelib.py           GateLab: arms, solve_arm, Table 1, noise sweep, trace, export
  check_covariance.py  CLI: verify a rotated gate really is its source gate's image
```

## Where each kind of parameter lives (F08)

| what | where | why there |
|---|---|---|
| chip: $\alpha$, $\Omega_{\max}$, $T_1/T_2$, sample rate | `curve_opt.device` (`hardware` group) | derived quantities + validation |
| noise hypothesis: $\delta_z$, $\epsilon$ | `curve_opt.device` (`noise` group) | sets the budget weights $w_i$ |
| design choices: $T$, $M$ | `curve_opt.device` (`design` group) | coupled by $n^\*=2T|\alpha|=M$ |
| target gate: $\theta$, $\phi_{\rm axis}$ | `gatespec.SPECS` | one entry per gate |
| what an arm is | `armspec.ARM_RECIPES` / `BASELINE_RECIPES` | each number carries its argument |
| which run backs which arm | `configs/runs.json` | pure data, written by scripts |

Rule of thumb: **Python holds defaults and the reasoning behind them; JSON holds only
overrides and registries.** A device override file (`configs/device_*.json`) may name
just the fields it changes; `curve_opt.recorder.load_device` reads it and everything
else falls back to the documented default. Unknown field names raise rather than
silently do nothing.

Each proposal folder holds a three-line `armlib.py` that binds one `GateSpec` into a
`GateLab` and publishes its methods, so the notebooks keep the flat
`al.characterize(...)` / `al.ARM_ORDER` call style.

## One recipe, two entry points

`armspec.ARM_RECIPES` holds warm start, fluence-cap multiple, `e0_cap`, R-guard `rho`,
solver grids, `hessian_mode`, `objective_mode` and `maxiter` for arms **C/D/E/F**. Both `run_construction.py` and the notebook's in-place solve cell go through
`GateLab.solve_arm`, so there is no second copy that could drift — a notebook solve and a
script solve are the same run. Use the notebook to solve and inspect in one place; use
the shell for the long ones (arm D runs to `maxiter=20000`).

`solve_arm` always records to `_runs/<run_id>/`. The project requires every optimization
on disk and every plot read back from a RunRecord, and recording also means a solve is
paid for once: the finished run registers itself in `configs/runs.json` (or call
`register_solved`), and later kernels load it instead of re-solving. Chaining within a session is still supported — pass the previous arm as
`warm_arm=` — the result is persisted either way.

It also checks the **`c1` endpoint equality constraint** on the returned arm and warns
loudly if the solve stopped before closing it. Such an arm cannot be broadcast at all
(the DRAG/Stark readout refuses a `c1`-violating `a`, since its $\Omega_y$ endpoint
would be basis dependent); without the check that surfaces much later as a `ValueError`
deep inside `parametrization.check_c1`, which reads like a plumbing bug rather than "the
solve is not finished".

## The four gates

| key | gate | $\theta$ | axis | how the arms are produced |
|---|---|---|---|---|
| `Xpi` | $X(\pi)$ | $\pi$ | $x$ | already solved — `_runs/20260808…`, `20260810_F06b…`, `20260810_F06e…` |
| `Xpi2` | $X(\pi/2)$ | $\pi/2$ | $x$ | **own solves** — `run_construction.py {C,D,E}` |
| `Ypi` | $Y(\pi)$ | $\pi$ | $y$ | frame rotation of `Xpi` |
| `Ypi2` | $Y(\pi/2)$ | $\pi/2$ | $y$ | frame rotation of `Xpi2` |

## Why the Y gates are not new optimization problems

The three-level model is

$$H(t) = \Delta\,|2\rangle\langle 2| + \tfrac12\big[\Omega_x(t)X_3 + \Omega_y(t)Y_3\big] + \text{noise}.$$

Let $N=\mathrm{diag}(0,1,2)$ and $V(\varphi)=e^{i\varphi N}$. Then

$$V H(\Omega_x,\Omega_y) V^\dagger = H(\Omega_x\cos\varphi-\Omega_y\sin\varphi,\ \Omega_x\sin\varphi+\Omega_y\cos\varphi),$$

since $|2\rangle\langle2|$ commutes with $N$ and $VX_3V^\dagger$ mixes the quadratures as a
planar rotation. **Both noise channels survive the conjugation**: the quasi-static
$\delta_z$ enters through a diagonal operator proportional to $N$, and the amplitude
error $\epsilon$ is a scalar multiplying $\Omega$. Therefore, at fixed $T$ and device:

* the optimal waveform for a rotation about the equatorial axis at azimuth $\varphi$ is
  the optimal waveform for $R_x$ with the drive phase advanced by $\varphi$;
* every design-curve cost $C_1\ldots C_4$, the true $1-\bar F$, the leakage population and
  the whole $(\delta_z,\epsilon)$ sweep are **numerically identical, point for point**.

Re-solving would burn optimizer time to rediscover the same optimum. So the Y folders
run with `solve_mode="rotate_from_x"` and spend their compute on *checking* the claim
instead of assuming it:

```
python check_covariance.py Ypi      # -> PASS: all costs agree to 0.0, infidelities to ~1e-15
python check_covariance.py Ypi2
```

Measured on all five $Y(\pi)$ arms: $C_1,C_2,C_3$ agree to exactly `0.0`, $C_4$ to
$\le 2\times10^{-11}$ relative, and every $1-\bar F$ — zero-noise and across the full
$7\times7$ sweep — to $\le 5\times10^{-15}$ absolute.

**This check is also the tripwire.** The day the model grows an IQ imbalance, a
quadrature-dependent crosstalk, or a noise channel that does not commute with $N$, the
symmetry breaks, the check fails, and the Y folders must switch to
`solve_mode="independent"`. That path additionally needs a `propagate.target_axis(theta,
phi)` in `src/` and an axis-aware `BudgetProblem` — not written, deliberately, since
nothing currently needs it.

## What changes with $\theta$, and what does not

| quantity | scales with $\theta$? | why |
|---|---|---|
| budget weights $w_1\ldots w_4$ | **no** | conditional on the assumed noise moments; rescaling them changes the noise model, not the gate |
| peak bound $\Omega_{\max}$ | **no** | device constant |
| R guard $\max\lvert\tau\rvert\le\rho\lvert\Delta\rvert/2$ | **no** | device-level validity bound |
| $T$, $M$, $\eta=\Omega_{\max}/\lvert\Delta\rvert$ | **no** | device constants |
| decoherence floor | **no** | $\propto T$, waveform independent |
| **fluence floor $F_0$** | **yes: $F_0=\theta^2/T$** | square-pulse lower bound on $\int\kappa^2dt$ *for this angle* |

That last row is the one thing a naive copy would get wrong. The F06b/F06e caps are
quoted as multiples of $F_0$ ("cap $=2F_0$"), and the multiple is what carries the
physical meaning — how far above the cheapest possible pulse the achieved $\eta$ is
allowed to go. Keeping the hard-coded $\pi^2/T$ would turn "cap $=2F_0$" into
"cap $=8F_0$" for a $\pi/2$ gate and switch the validity guard off without saying so.

## Relationship to the existing $X(\pi)$ folder

`1qb_Gaussian-drag-optimization/armlib.py` used to stand alone with its own 475-line
copy of this machinery and its own `run_id` table — and the two tables had already
drifted apart once (arm E pointed at round 1 in one place and round 2 in the other).
F08a migrated it: all four folders are now the same three-line shim. The migration was
verified by re-deriving every reported number for all four gates before and after and
requiring bit-for-bit equality (`_dev_logs/F08_golden.py`, 267 828 numbers), plus
checking that each arm recipe rebuilds the manifest of the run it was actually solved
with (`_dev_logs/F08_recipe_fidelity.py`).

Gate-independent reuse now comes from `armcore.py` — `Input` (with the arm-A $C_4$
evaluation-object trap already handled), `c1_project_and_rescale`, `fit_phi_vz`,
`rz_matrix` and `fidelity_jax`. These were written for
`_dev_logs/F05_budget_validation.py` and imported from there until F08c, which made a
frozen dev-log artefact a runtime dependency of the proposals; the dev log now imports
them back, so there is one implementation and the numbers are unchanged. What F05
hard-codes at module scope — `THETA`, `U_TARGET` — is re-implemented in `gatelib` as
gate-parametric methods so the $X(\pi)$ numbers stay term-for-term comparable with the
dev logs.
