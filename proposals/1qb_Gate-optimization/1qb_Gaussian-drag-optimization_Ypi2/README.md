# Proposal — Single-qubit $Y(\pi/2)$

$Y(\pi/2)$ is the $X(\pi/2)$ gate with the drive phase advanced by $\pi/2$ — an **exact**
frame rotation of the three-level model, noise channels included. No optimization runs
here.

This folder is the composition of the other two changes: the new rotation angle comes
from `../1qb_Gaussian-drag-optimization_Xpi2`, the axis rotation from the same mechanism
as `../1qb_Gaussian-drag-optimization_Ypi`. Read **`../1qb_Gaussian-drag-optimization/README.md`**
for the method, **`_Xpi2/README.md`** for what the smaller angle changes, and
**`../_gatelib/README.md`** for the covariance derivation.

---

## 1. Dependency

Arms A and B (Gaussian, Gaussian+DRAG) work immediately — they are constructed
analytically at $\theta=\pi/2$ and then rotated.

Arms C, D, E appear only once the $X(\pi/2)$ solves have landed and their `run_id`s are
registered in `../_gatelib/gatespec.py` under `SPECS["Xpi2"].run_ids`. This folder reads
them through that entry; there is nothing to run here to make them appear. `ARM_ORDER` is
decided at import time from which run directories exist, so restart the kernel after
registering a run.

## 2. The rotation

Each arm is the corresponding $X(\pi/2)$ arm with

$$(\Omega_x + i\Omega_y) \longrightarrow e^{i\pi/2}(\Omega_x + i\Omega_y),\qquad
U_{\rm target} = R_y(\pi/2) = \tfrac{1}{\sqrt2}\begin{pmatrix}1 & -1\\ 1 & 1\end{pmatrix}.$$

`gatelib.RotatedInput` rotates only `broadcast()`; the design curve is untouched, so
$\kappa$, $\tau$ and $C_1,C_2,C_3$ are literally the $X(\pi/2)$ arm's, and $C_4$ and the
peak are rotation invariant. **Fig. 1 is the only figure that should look different.**

## 3. The check

```bash
conda activate curve
python ../_gatelib/check_covariance.py Ypi2
```

Measured on arms A and B (the only ones available before the $X(\pi/2)$ solves):
$C_1,C_2,C_3$ agree to exactly `0.0`, $C_4$ to $\le 8\times10^{-13}$ relative, and the
infidelities — zero-noise and across the full $7\times7$ sweep — to $\le 6\times10^{-15}$
absolute.

★ Note why the verdict uses a **mixed** criterion, $|{\rm diff}| \le 10^{-13} +
10^{-8}|{\rm ref}|$, rather than a relative one. Arm B here has $1-\bar F = 1.1\times10^{-7}$
— two decades below the $X(\pi)$ baseline — so the ~5e-15 absolute wobble from
`fit_phi_vz`'s scalar minimization reads as 5e-8 *relative*. A pure relative tolerance
would have to be loosened until it stopped testing anything. The absolute term carries
the small-infidelity arms; the relative term carries the rest.

Rerun the check after each new arm lands. It is the tripwire: if the model ever stops
being covariant (IQ imbalance, quadrature-dependent crosstalk, a noise channel not
commuting with $N$), this is what tells you, and the folder must switch to
`solve_mode="independent"`.

## 4. Running it

```bash
conda activate curve
cd proposals/1qb_Gaussian-drag-optimization_Ypi2
jupyter lab notebook.ipynb        # kernel: Python (curve)
```

No optimizer time. Exports the 120-point CSVs (F07 schema) to `waveforms/`.

## 5. How to report this

The rotated waveforms are the deliverable. The numbers are inherited from the $X(\pi/2)$
run and must be reported as such — quoting them as an independent result would be
double-counting the same solve twice over (once for the angle, once for the axis).
