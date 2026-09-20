"""Damped Gauss-Newton (Levenberg-Marquardt) on the linear parameter block.

Sam's proposal: instead of hunting for an iterative solver that reaches the
floor, damp the normal equations,

    delta = argmin ||J delta - r||^2 + mu ||delta||^2,

and control mu.  mu -> 0 is the full least-squares step; mu -> infinity is
gradient descent with learning rate 1/mu.

NOTE ON NOTATION: this project already uses lambda for the QI bandwidth
lambda* = gamma*h.  The damping is called mu here, everywhere, to keep the two
apart.

Three panels, each isolating one variable:
  E1  what damping does to the required stored state c*
  E2  accuracy vs iteration budget T at fixed mu (is the step gradient-priced?)
  E3  repeated damped steps on fresh batches, sweeping mu and b
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
from window_law import lsqr_window
import disproof as D

OUT = c.RESULTS / "damping.jsonl"
FIG = c.FIGS
T0 = time.time()
I = (lambda v: v)


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def setup(N, lam=0.30):
    A, y, Ae, ye = D.qi_system(N, lam)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = int((s > 1e-15 * s[0]).sum())
    return dict(A=A, y=y, Ae=Ae, ye=ye, ny=float(np.linalg.norm(ye)),
                U=U, s=s, Vt=Vt, r=r, d=A.shape[1], n=A.shape[0], N=N)


def err(p, w):
    return float(np.linalg.norm(p["Ae"] @ w - p["ye"]) / p["ny"])


def damped_exact(p, mu):
    s = p["s"]
    return p["Vt"].T @ ((s / (s ** 2 + mu)) * (p["U"].T @ p["y"]))


def aug(A, b, mu, d):
    if mu <= 0:
        return A, b
    return np.vstack([A, np.sqrt(mu) * np.eye(d)]), np.concatenate([b, np.zeros(d)])


# ------------------------------------------------------------------- E1 ---
def data_e1(Ns=(256, 512), js=(8, 16, 32, 64, 128, 200, None),
            cws=(0, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512)):
    rows = []
    for N in Ns:
        p = setup(N)
        s, r, d = p["s"], p["r"], p["d"]
        for j in js:
            jj = r - 1 if j is None else j
            if jj >= r:
                jj = r - 1
            mu = float(s[jj] ** 2)
            rmu = int((s > np.sqrt(mu)).sum())
            eex = err(p, damped_exact(p, mu))
            Aa, ya = aug(p["A"], p["y"], mu, d)
            cstar = None
            for cw in cws:
                if cw > 1.4 * r:
                    break
                x, _ = lsqr_window(Aa, ya, I, d, int(2 * r), cw)
                if err(p, x) <= 3 * eex:
                    cstar = cw
                    break
            rows.append(dict(panel="E1", N=N, d=d, r=r, mu=mu, r_mu=rmu,
                             damped_opt=eex, cstar=cstar,
                             kappa_mu=float(np.sqrt((s[0] ** 2 + mu)
                                                    / (s[r - 1] ** 2 + mu)))))
            log(f"E1 N={N} r_mu={rmu:4d} mu={mu:.1e} opt={eex:.2e} c*={cstar}")
    return rows


# ------------------------------------------------------------------- E2 ---
def data_e2(N=256, js=(16, 64, 128, 200, None),
            budgets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)):
    p = setup(N)
    s, r, d = p["s"], p["r"], p["d"]
    rows = []
    for j in js:
        jj = r - 1 if j is None else j
        mu = float(s[jj] ** 2)
        rmu = int((s > np.sqrt(mu)).sum())
        eex = err(p, damped_exact(p, mu))
        Aa, ya = aug(p["A"], p["y"], mu, d)
        for T in budgets:
            x, _ = lsqr_window(Aa, ya, I, d, T, 0)
            rows.append(dict(panel="E2", N=N, d=d, r=r, mu=mu, r_mu=rmu,
                             T=T, eval=err(p, x), damped_opt=eex))
        log(f"E2 r_mu={rmu} mu={mu:.1e}")
    return rows


# ------------------------------------------------------------------- E3 ---
def data_e3(N=256, mus=(1e2, 1e0, 1e-2, 1e-4, 1e-8, 0.0),
            bmults=(1.0, 0.25, 1 / 16, 1 / 64), T=8, steps=400):
    p = setup(N)
    A, y, d, n = p["A"], p["y"], p["d"], p["n"]
    rows = []
    for bm in bmults:
        b = max(int(bm * d), 2)
        for mu in mus:
            rng = np.random.default_rng(7)
            w = np.zeros(d)
            best = np.inf
            for _ in range(steps):
                S = rng.choice(n, min(b, n), replace=False)
                Ab = A[S]
                Aa, ya = aug(Ab, y[S] - Ab @ w, mu, d)
                w = w + lsqr_window(Aa, ya, I, d, T, 0)[0]
                best = min(best, err(p, w))
            rows.append(dict(panel="E3", N=N, d=d, b=b, bmult=bm, mu=mu,
                             T=T, steps=steps, eval=best))
        log(f"E3 b={b} (b/d={bm:.4g}) done")
    return rows


# --------------------------------------------------------------- figure ---
def figure(rows):
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.9))

    ax = axs[0]
    e1 = [q for q in rows if q["panel"] == "E1"]
    for i, N in enumerate(sorted({q["N"] for q in e1})):
        sub = sorted([q for q in e1 if q["N"] == N], key=lambda q: q["r_mu"])
        ax.plot([q["damped_opt"] for q in sub],
                [q["cstar"] for q in sub], "o-", color=f"C{i}", ms=6, lw=1.9,
                label=f"QI-1D N={N} (r={sub[0]['r']})")
        ax.axhline(sub[0]["r"], color=f"C{i}", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([0, 1, 10, 100, 500])
    ax.set_yticklabels(["0", "1", "10", "100", "500"])
    ax.set_xlabel("accuracy of the damped optimum  (rel $L_2$)")
    ax.set_ylabel("$c^*$ : stored vectors needed to reach it")
    ax.set_title("E1  damping removes the memory requirement\n"
                 "until you ask for the floor  (dotted = $r$)", fontsize=9.5)
    ax.invert_xaxis()
    ax.grid(alpha=.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.16), ncol=1,
              fontsize=8, frameon=False)

    ax = axs[1]
    e2 = [q for q in rows if q["panel"] == "E2"]
    rms = sorted({q["r_mu"] for q in e2})
    for i, rm in enumerate(rms):
        sub = sorted([q for q in e2 if q["r_mu"] == rm], key=lambda q: q["T"])
        ax.loglog([q["T"] for q in sub], [max(q["eval"], 1e-17) for q in sub],
                  "o-", color=f"C{i}", ms=4.5, lw=1.7,
                  label=f"$r_\\mu$={rm}")
    Ts = np.array([2.0, 512.0])
    ax.loglog(Ts, 9.5e-2 * (Ts / 2) ** -1.6, "k--", lw=1.4,
              label="$T^{-1.6}$ (measured rate)")
    ax.set_xlabel("iteration budget $T$   (1 iter = one fwd-bwd)")
    ax.set_ylabel("eval rel $L_2$")
    ax.set_title("E2  accuracy is ALGEBRAIC in the iteration budget\n"
                 "(curves coincide until damping binds)", fontsize=9.5)
    ax.grid(alpha=.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.16), ncol=3,
              fontsize=7.5, frameon=False)

    ax = axs[2]
    e3 = [q for q in rows if q["panel"] == "E3"]
    mus = sorted({q["mu"] for q in e3})
    for i, mu in enumerate(mus):
        sub = sorted([q for q in e3 if q["mu"] == mu], key=lambda q: -q["bmult"])
        ax.loglog([q["bmult"] for q in sub],
                  [max(q["eval"], 1e-17) for q in sub], "o-", color=f"C{i}",
                  ms=5, lw=1.7,
                  label=(r"$\mu$=0 (undamped)" if mu == 0 else f"$\\mu$={mu:.0e}"))
    ax.set_xlabel("batch fraction $b/d$")
    ax.set_ylabel("best eval rel $L_2$ over 400 steps")
    ax.set_title("E3  repeated damped steps on FRESH batches\n"
                 "$O(d)$ state, $T$=8 per step", fontsize=9.5)
    ax.invert_xaxis()
    ax.grid(alpha=.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.16), ncol=3,
              fontsize=7.5, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.80])
    fig.text(0.5, 0.985, "E   Damped Gauss-Newton on the linear block: what the damping $\\mu$ actually buys",
             ha="center", va="top", fontsize=11.5, weight="bold")
    fig.text(0.5, 0.945,
             "Damping collapses every singular value below $\\sqrt{\\mu}$ into one cluster, so the damped operator has few distinct scales. By the expD11 spectral law that should remove the\n"
             "stored-state requirement -- and it does (E1: $c^*=0$ everywhere except at the floor). The price is in E2: without stored orthogonality, accuracy improves only algebraically in iterations.",
             ha="center", va="top", fontsize=8.7, linespacing=1.5)
    fig.savefig(FIG / "E_damping.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    rows = []
    for fn in (data_e1, data_e2, data_e3):
        rs = fn()
        for q in rs:
            c.append(OUT, q)
        rows.extend(rs)
    figure(rows)
    log(f"figure -> {FIG/'E_damping.png'}")
