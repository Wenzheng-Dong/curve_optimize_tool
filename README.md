# curve-opt-tool

Constrained pulse design for noise-robust single- and two-qubit gates, working
directly at the control level.

The control fields are expanded in a sine series on `[0, T]`,

```
Omega_x(t) = sum_{n=1..M} a_n sin(n pi t / T)
Omega_y(t) = sum_{n=1..M} b_n sin(n pi t / T)
```

and the series coefficients are the optimization variables. The gate condition,
the space-curve closure (first-order Z-noise cancellation) and the vanishing
projected area (second order) are imposed as **hard equality constraints**, not
weighted penalties. The objective trades pulse energy against peak amplitude,
with the non-smooth peak term handled through an epigraph reformulation.

A second question the tool is built to answer: **how much does the ansatz — the
initial curve — actually influence the optimization trajectory and the optimal
control it lands on?**

## Status

Early development. Planning is complete; the numerical foundations
(planar and 3D feasibility, quadrature order, propagator ordering, Gauss-Newton
speedup) are validated. Per-step status is tracked in a development log that is
kept out of this repository.

All results are currently **proxy-layer** statements: they concern the geometric
robustness conditions in the quasi-static limit, not a full physical-model
infidelity.

## Environment

```bash
conda env create -f environment.yml   # creates the `curve` env
conda activate curve
pytest
```

Float64 is enabled process-wide when `curve_opt` is imported; JAX defaults to
float32, which cannot represent the residuals this project claims.

## Layout

```
src/curve_opt/
    basis.py       coefficients <-> waveforms, gate row, analytic bounds
    geometry.py    planar chain Omega -> theta -> r -> closure / area
    propagate.py   SU(2) propagator and the 3D constraints
    metrics.py     individual, scale-invariant cost terms
    ansatz.py      curve intake, projection, step-0 error report
    optimize.py    trust-constr wrapper (hard constraints, Gauss-Newton)
    recorder.py    run-record I/O -- the only module that touches the disk
    plotting.py    figures, from stored run records only
tests/             pytest, including the numerical-discipline gates
results/           curated presentation material, re-plotted from run records
```
