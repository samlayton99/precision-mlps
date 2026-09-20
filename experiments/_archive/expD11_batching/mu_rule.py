"""What should the damping mu actually be set FROM?

For the readout block the Gauss-Newton step is exact, so mu is not a linearity
dial.  The claim under test is that mu is a VARIANCE dial: it should equal the
total uncertainty in the thing being fitted, which has three additive sources

    observation noise      y is noisy
    geometry staleness     Phi moves under Adam between solve and use
    batch sampling         the solve sees b rows, not the pool

If that framing is right, each source should have a power law mu* ~ (source)^p
with p = 2, i.e. mu* tracks a VARIANCE, and the sources should combine
additively.  Then the update rule is a variance budget, not a heuristic.

M1  mu* vs observation noise sigma
M2  mu* vs geometry drift eta  (centers perturbed between solve and use)
M3  additivity: does mu*(sigma, eta) = mu*(sigma) + mu*(eta)?
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import core11 as c
from src.construction.qi_mpmath import default_halo

OUT = c.RESULTS / "mu_rule.jsonl"
FIG = c.FIGS
T0 = time.time()
TARGET = lambda z: np.sin(4.0 * z)
MUS = np.concatenate([[0.0], np.logspace(-30, 2, 65)])


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def geom(N, lam=0.30):
    h = 2.0 / N
    return h, lam / h, -1.0 + h * np.arange(-int(default_halo(N, lambda_star=lam)),
                                            N + int(default_halo(N, lambda_star=lam)) + 1)


def build(x, gamma, cen):
    return np.hstack([np.tanh(gamma * (x[:, None] - cen[None, :])),
                      np.ones((x.size, 1))])


def cell(N, sigma, eta, seed=0, row_mult=6):
    """Solve on (Phi, y+noise); evaluate against the DRIFTED geometry."""
    h, gamma, cen = geom(N)
    rng = np.random.default_rng(1234 + seed)
    xt = np.linspace(-1, 1, row_mult * cen.size)
    xe = np.linspace(-1, 1, 4001)
    A = build(xt, gamma, cen)
    y = TARGET(xt)
    if sigma > 0:
        e = rng.standard_normal(xt.size)
        y = y + e * sigma * np.linalg.norm(y) / np.linalg.norm(e)
    # geometry drift: centers move by eta*h between the solve and its use
    cen2 = cen + eta * h * rng.standard_normal(cen.size)
    Ae2 = build(xe, gamma, cen2)
    ye = TARGET(xe)
    ny = float(np.linalg.norm(ye))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Uy = U.T @ y
    out = []
    for mu in MUS:
        f = s / (s ** 2 + mu) if mu > 0 else np.where(s > 1e-15 * s[0], 1 / np.where(s > 1e-15 * s[0], s, 1), 0)
        w = Vt.T @ (f * Uy)
        out.append(float(np.linalg.norm(Ae2 @ w - ye) / ny))
    out = np.array(out)
    j = int(np.argmin(out))
    return dict(N=N, d=A.shape[1], sigma=sigma, eta=eta, seed=seed,
                mu_star=float(MUS[j]), err_star=float(out[j]),
                err_mu0=float(out[0]), sigma1=float(s[0]),
                curve=[float(v) for v in out])


def main():
    if OUT.exists():
        OUT.unlink()
    rows = []
    N = 256
    for sig in (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
        for sd in range(3):
            r = cell(N, sig, 0.0, seed=sd)
            r["panel"] = "M1"
            rows.append(r); c.append(OUT, r)
        log(f"M1 sigma={sig:.0e} mu*={rows[-1]['mu_star']:.2e} err={rows[-1]['err_star']:.2e}")
    for eta in (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
        for sd in range(3):
            r = cell(N, 0.0, eta, seed=sd)
            r["panel"] = "M2"
            rows.append(r); c.append(OUT, r)
        log(f"M2 eta={eta:.0e} mu*={rows[-1]['mu_star']:.2e} err={rows[-1]['err_star']:.2e}")
    for sig in (1e-8, 1e-4):
        for eta in (1e-8, 1e-4):
            for sd in range(3):
                r = cell(N, sig, eta, seed=sd)
                r["panel"] = "M3"
                rows.append(r); c.append(OUT, r)
            log(f"M3 sigma={sig:.0e} eta={eta:.0e} mu*={rows[-1]['mu_star']:.2e}")
    figure(rows)
    log("done")


def gm(v):
    v = np.array([x for x in v if x > 0])
    return float(np.exp(np.log(v).mean())) if v.size else 0.0


def figure(rows):
    fig, axs = plt.subplots(1, 3, figsize=(16.0, 5.4))
    fig.subplots_adjust(left=0.06, right=0.975, bottom=0.13, top=0.70, wspace=0.34)

    for k, (pan, xk, xlab, col) in enumerate(
            [("M1", "sigma", r"observation noise  $\sigma_{rel}$", "C0"),
             ("M2", "eta", r"geometry drift  $\eta$   (centers moved by $\eta h$)", "C1")]):
        ax = axs[k]
        sub = [q for q in rows if q["panel"] == pan]
        xv = sorted({q[xk] for q in sub if q[xk] > 0})
        mus = [gm([q["mu_star"] for q in sub if q[xk] == x]) for x in xv]
        ax.loglog(xv, mus, "o-", color=col, ms=7, lw=2, label=r"measured $\mu^*$")
        xa = np.array(xv)
        sc = mus[-1] / xa[-1] ** 2
        ax.loglog(xa, sc * xa ** 2, "k--", lw=1.6, label=r"$\propto x^2$ (a variance)")
        ax.loglog(xa, (mus[-1] / xa[-1]) * xa, ":", color="grey", lw=1.6,
                  label=r"$\propto x^1$ (not a variance)")
        p = np.polyfit(np.log10(xa), np.log10(np.maximum(mus, 1e-300)), 1)[0]
        ax.set_title(f"{pan}  optimal damping vs "
                     f"{'noise' if pan=='M1' else 'drift'}\nfitted slope = {p:.2f}",
                     fontsize=10)
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"$\mu^*$  (damping minimising eval error)")
        ax.grid(alpha=.3, which="both")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.16), ncol=3,
                  fontsize=7.5, frameon=False)

    ax = axs[2]
    m3 = [q for q in rows if q["panel"] == "M3"]
    m1 = [q for q in rows if q["panel"] == "M1"]
    m2 = [q for q in rows if q["panel"] == "M2"]
    lbl, pred, meas = [], [], []
    for sig in sorted({q["sigma"] for q in m3}):
        for eta in sorted({q["eta"] for q in m3}):
            a = gm([q["mu_star"] for q in m1 if q["sigma"] == sig])
            b = gm([q["mu_star"] for q in m2 if q["eta"] == eta])
            mm = gm([q["mu_star"] for q in m3
                     if q["sigma"] == sig and q["eta"] == eta])
            if mm > 0:
                lbl.append(f"$\\sigma$={sig:.0e}\n$\\eta$={eta:.0e}")
                pred.append(a + b)
                meas.append(mm)
    xs = np.arange(len(lbl))
    ax.bar(xs - 0.2, pred, 0.4, color="C7", label=r"predicted  $\mu^*(\sigma)+\mu^*(\eta)$")
    ax.bar(xs + 0.2, meas, 0.4, color="C2", label=r"measured  $\mu^*(\sigma,\eta)$")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(lbl, fontsize=7.5)
    ax.set_ylabel(r"$\mu^*$")
    ax.set_title("M3  do the two sources add?\n(variance budget test)", fontsize=10)
    ax.grid(alpha=.3, axis="y", which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.16), ncol=2,
              fontsize=7.5, frameon=False)

    fig.text(0.5, 0.975, "M   What should $\\mu$ be set from?  Testing the variance-budget rule",
             ha="center", va="top", fontsize=12, weight="bold")
    fig.text(0.5, 0.935,
             "If $\\mu$ is a variance dial, the optimal damping should scale as the SQUARE of each uncertainty source (dashed reference), and the sources should ADD.\n"
             "QI-1D N=256, exact damped solves over a 65-point $\\mu$ grid, 3 seeds, geometric mean. Error is always measured against the drifted geometry on clean targets.",
             ha="center", va="top", fontsize=8.8, linespacing=1.5)
    fig.savefig(FIG / "M_mu_rule.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
