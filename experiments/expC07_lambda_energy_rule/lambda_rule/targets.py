"""Held-out targets for expC10, generated from one seed and never inspected before the run."""
from __future__ import annotations

import numpy as np

SEED = 20260908


def make_targets():
    rng = np.random.default_rng(SEED)
    T = []
    # F1: random trigonometric polynomials, band edge exact
    for K in (2, 8, 32):
        for d in range(2):
            a = rng.uniform(0.2, 1.0, K)
            phi = rng.uniform(0, 2 * np.pi, K)
            ks = np.arange(1, K + 1)
            T.append({"name": f"trig_K{K}_{d}", "family": "F1", "omega": float(K * np.pi),
                      "params": {"a": a.tolist(), "phi": phi.tolist(), "K": K}})
    # F2: random rational functions, band edge = 95% energy of the closed-form transform
    for d in range(6):
        J = int(rng.integers(1, 4))
        c = rng.uniform(-1, 1, J)
        p = rng.uniform(-0.8, 0.8, J)
        s = np.exp(rng.uniform(np.log(0.05), np.log(0.5), J))
        T.append({"name": f"rat_J{J}_{d}", "family": "F2", "omega": None,
                  "params": {"c": c.tolist(), "p": p.tolist(), "s": s.tolist(), "J": J}})
    for t in T:
        if t["omega"] is None:
            t["omega"] = band_edge_rational(t["params"])
    return T


def evaluate(t, x):
    x = np.asarray(x, dtype=np.float64)
    P = t["params"]
    if t["family"] == "F1":
        ks = np.arange(1, P["K"] + 1)
        return np.sum(np.array(P["a"])[:, None] * np.sin(ks[:, None] * np.pi * x[None, :] + np.array(P["phi"])[:, None]), axis=0)
    out = np.zeros_like(x)
    for cj, pj, sj in zip(P["c"], P["p"], P["s"]):
        out += cj * sj / ((x - pj) ** 2 + sj ** 2)
    return out


def spectrum_sq_rational(P, w):
    """|f_hat(w)|^2 for f = sum c_j s_j / ((x-p_j)^2 + s_j^2); term transform c_j pi e^{-s_j|w|} e^{-i w p_j}."""
    w = np.asarray(w, dtype=float)
    F = np.zeros_like(w, dtype=complex)
    for cj, pj, sj in zip(P["c"], P["p"], P["s"]):
        F += cj * np.pi * np.exp(-sj * np.abs(w)) * np.exp(-1j * w * pj)
    return np.abs(F) ** 2


def band_edge_rational(P, frac=0.95):
    w = np.linspace(0, 2000.0, 400001)
    S = spectrum_sq_rational(P, w)
    cum = np.cumsum(S)
    cum /= cum[-1]
    return float(w[np.searchsorted(cum, frac)])


if __name__ == "__main__":
    for t in make_targets():
        print(f"{t['name']:12s} family {t['family']} omega/pi = {t['omega']/np.pi:8.3f}")
