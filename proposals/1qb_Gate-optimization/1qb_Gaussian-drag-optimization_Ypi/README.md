# Proposal — Single-qubit $Y(\pi)$

$Y(\pi)$ is the $X(\pi)$ gate with the drive phase advanced by $\pi/2$. In this model
that is an **exact** frame rotation — noise channels included — so this folder runs no
optimization. It produces the $Y(\pi)$ waveforms and verifies the symmetry the claim
rests on.

**Read `../1qb_Gaussian-drag-optimization/README.md` first**; the method is that folder's,
unchanged. The derivation of the covariance is in `../_gatelib/README.md` and in
`gatespec.py`'s module docstring.

---

## 1. The claim, in one line

With $N=\mathrm{diag}(0,1,2)$ and $V(\varphi)=e^{i\varphi N}$, conjugating the three-level
Hamiltonian by $V(\pi/2)$ rotates the drive quadratures and leaves the detuning term, the
quasi-static $\delta_z$ coupling (proportional to $N$) and the amplitude error (a scalar)
untouched. So each $Y(\pi)$ arm is the corresponding $X(\pi)$ arm with

$$(\Omega_x + i\Omega_y) \longrightarrow e^{i\pi/2}(\Omega_x + i\Omega_y),\qquad
U_{\rm target} \longrightarrow R_z(\tfrac\pi2)\,R_x(\pi)\,R_z(-\tfrac\pi2) = R_y(\pi),$$

and every reported quantity is identical.

Implementation: `gatelib.RotatedInput` wraps the source arm and rotates only
`broadcast()`. The design curve $(\mathbf a,\mathbf c,\Phi_0)$ is untouched, so
$\kappa$, $\tau$ and $C_1,C_2,C_3$ are *literally* the $X(\pi)$ arm's; $C_4\sim|\Lambda|^2$
and the peak $|\Omega|$ are rotation invariant. **Fig. 1 is the only figure that should
look different** — the drive has moved from $\Omega_x$ into $\Omega_y$.

## 2. Why we check instead of assert

```bash
conda activate curve
python ../_gatelib/check_covariance.py Ypi
```

Measured across all five arms (A, B, C, D, E):

| quantity | agreement |
|---|---|
| $C_1, C_2, C_3$ | exactly `0.0` relative |
| $C_4$ | $\le 2\times10^{-11}$ relative |
| zero-noise $1-\bar F$ | $\le 5\times10^{-15}$ absolute |
| full $7\times7$ $(\delta_z,\epsilon)$ sweep | $\le 4\times10^{-15}$ absolute |

★ Both sides are swept at the **same frozen $\Phi_{\rm vz}$**, on purpose. The rotation
maps the calibration problem onto itself exactly (fidelity is invariant under
simultaneous conjugation, so the two gates share the optimal $\Phi_{\rm vz}$), but each
side reaches it through its own bounded scalar minimization, which lands ~3e-8 rad apart.
Letting each keep its own fit inflated arm D's sweep residual to 9e-10 — three decades
above everything else on the page — for a reason that has nothing to do with the
rotation. The refit spread is reported separately instead of being folded into the
verdict.

**This check is the tripwire, not a formality.** The day the model grows an IQ imbalance,
a quadrature-dependent crosstalk, or a noise channel that does not commute with $N$, the
symmetry breaks and this check fails. At that point set
`SPECS["Ypi"].solve_mode = "independent"` — which additionally needs a
`propagate.target_axis(theta, phi)` in `src/` and an axis-aware `BudgetProblem`,
deliberately not written since nothing currently needs it.

## 3. Running it

```bash
conda activate curve
cd proposals/1qb_Gaussian-drag-optimization_Ypi
jupyter lab notebook.ipynb        # kernel: Python (curve)
```

No optimizer time. The notebook builds the arms from the $X(\pi)$ runs registered in
`../_gatelib/gatespec.py`, reproduces Table 1 and Figs. 1–4, runs the covariance check in
§5, and exports the 120-point CSVs (F07 schema) to `waveforms/`.

## 4. How to report this

The waveforms are a real deliverable — an AWG needs the rotated quadratures, and this
folder is where they come from. The *numbers* are not an independent confirmation of
anything: they are the $X(\pi)$ run, relabelled. Reporting them as a second data point
would be double-counting. Say "inherited by exact frame rotation, verified to 1e-15", and
point at §5.
