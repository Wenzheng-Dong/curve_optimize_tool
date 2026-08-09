#!/usr/bin/env python3
"""Stand-alone check of a delivered waveform CSV. Requires only numpy + scipy.

    python verify_waveform.py Xpi/waveforms/Xpi_novera_F_optimized_ad_robust.csv
    python verify_waveform.py Xpi/waveforms/*.csv Ypi/waveforms/*.csv

Re-simulates the three-level transmon from the CSV alone (nothing else from this
package is used), reports the gate that is actually realised, and says whether a
virtual Z is still required. The target gate and phi_vz are read out of the CSV's
own `#` header, so the check is genuinely end-to-end.

Conventions -- identical to the CSV headers:
    H(t) = Omega_x(t) (a + a^dag)/2 + Omega_y(t) i(a^dag - a)/2 + Delta |2><2|
    a|1> = |0>, a|2> = sqrt(2)|1>,  Delta = 2*pi*alpha,  Omega in rad/ns
    samples are cell midpoints played sample-and-hold
"""
import sys
import numpy as np
from scipy.linalg import expm

ALPHA_GHZ = -0.19848                      # Novera anharmonicity, alpha/2pi
DELTA = 2 * np.pi * ALPHA_GHZ             # rad/ns

_A = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
DX = 0.5 * (_A + _A.conj().T)
DY = 0.5j * (_A.conj().T - _A)
I3 = np.eye(3, dtype=complex)
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)

TARGETS = {                                # exp(-i pi (n.sigma)/2)
    "X(pi)": -1j * SX,
    "Y(pi)": -1j * SY,
}


def rz(phi):
    return np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)])


def avg_gate_fidelity(B, target):
    """Average gate fidelity of a (possibly sub-unitary) 2x2 block against target."""
    tr = np.trace(target.conj().T @ B)
    return (abs(tr) ** 2 + np.trace(B.conj().T @ B).real) / 6.0


def load(path):
    cols, rows, meta = None, [], []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("#"):
            meta.append(line[1:].strip())
        elif cols is None:
            cols = line.split(",")
        elif line.strip():
            rows.append([float(v) for v in line.split(",")])
    return dict(zip(cols, np.array(rows).T)), meta


def target_from_meta(meta):
    for line in meta:
        for name in TARGETS:
            if line.startswith(name + " gate for"):
                return name
    raise ValueError("could not find the target gate in the CSV header")


def phi_vz_from_meta(meta):
    for line in meta:
        if line.startswith("phi_vz ="):
            return float(line.split("=")[1].split("rad")[0])
    return None


def polar_unitary(B):
    M = B.conj().T @ B
    w, v = np.linalg.eigh(M)
    w = np.clip(w, 1e-300, None)
    return B @ (v @ np.diag(1 / np.sqrt(w)) @ v.conj().T)


def zxz(W):
    """W ~ Rz(a) Rx(theta) Rz(b) up to global phase; angles in degrees."""
    V = W / np.sqrt(np.linalg.det(W))
    th = 2 * np.arctan2(abs(V[0, 1]), abs(V[0, 0]))
    apb = -2 * np.angle(V[0, 0]) if abs(V[0, 0]) > 1e-12 else 0.0
    amb = -2 * np.angle(V[0, 1] / (-1j)) if abs(V[0, 1]) > 1e-12 else 0.0
    return np.degrees((apb + amb) / 2), np.degrees(th), np.degrees((apb - amb) / 2)


def check(path):
    d, meta = load(path)
    name = target_from_meta(meta)
    UT = TARGETS[name]
    t = d["t_ns"]
    ox, oy = d["Omega_x_rad_per_ns"], d["Omega_y_rad_per_ns"]
    dt = float(np.diff(t)[0])
    U = I3.copy()
    H0 = np.diag([0, 0, DELTA]).astype(complex)
    for k in range(len(t)):
        U = expm(-1j * (H0 + ox[k] * DX + oy[k] * DY) * dt) @ U
    B = U[:2, :2]
    a, th, b = zxz(polar_unitary(B))
    leak = 1 - np.trace(B.conj().T @ B).real / 2

    grid = np.linspace(-np.pi, np.pi, 20001)
    f = np.array([avg_gate_fidelity(B, rz(p) @ UT) for p in grid])
    phi_best = grid[f.argmax()]
    phi_doc = phi_vz_from_meta(meta)

    print(f"\n=== {path}   [target {name}]")
    print(f"  {len(t)} samples, dt = {dt:.6f} ns ({1/dt:.4f} GSa/s), T = {t[-1]+dt/2:.4f} ns, "
          f"peak = {np.hypot(ox, oy).max()/(2*np.pi)*1e3:.4f} MHz")
    print(f"  realised gate  W = Rz({a:+.4f} deg) . Rx({th:.4f} deg) . Rz({b:+.4f} deg)")
    print(f"  leakage 1 - tr(B'B)/2                  = {leak:.4e}")
    print(f"  1 - Fbar  vs {name}, NO virtual Z      = {1 - avg_gate_fidelity(B, UT):.4e}")
    print(f"  1 - Fbar  vs {name}, best virtual Z    = {1 - f.max():.4e}   "
          f"at phi_vz = {np.degrees(phi_best):+.4f} deg")
    if phi_doc is not None:
        print(f"  phi_vz documented in the header        = {np.degrees(phi_doc):+.4f} deg   "
              f"(agreement: {abs(np.degrees(phi_best - phi_doc)):.4f} deg, grid step is 0.018 deg)")
    else:
        print("  header declares NO virtual Z is needed -> the first number above is the answer")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        check(p)
