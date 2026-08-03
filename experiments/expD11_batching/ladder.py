"""Can a damping SCHEDULE reach the floor at bounded peak state?

expD11 wall: reaching the fp64 floor needs c ~ r stored orthogonalization
vectors, i.e. Theta(d*r) state.
results/checkpoint_D_optimizers/expD09_2nd_order_regime/DAMPED_GAUSS_NEWTON.md: at damping mu the requirement vanishes (c* = 0) but the
accuracy is capped at the damped optimum.

The idea under test: peel the spectrum in chunks.  Stage k damps at mu_k,
solves with a SMALL FIXED window c, then DISCARDS its stored vectors before
stage k+1 lowers mu.  If each stage only has to resolve the directions between
sqrt(mu_k) and sqrt(mu_{k+1}), peak state is Theta(d*c) with c fixed, and the
number of stages is r/c.  That would clear requirement 2.

Against it: restarting a Krylov process re-converges directions that rounding
has already corrupted.  In favour: the measured restart penalty scales with
kappa of the operator being restarted, and each damped stage has small
kappa_mu.  Genuinely uncertain -- hence the experiment.

Panels
  L1  warm start        after a damped phase, what c does the terminal solve need?
  L2  the mu-ladder     fixed c, K descending stages, state discarded between
  L3  diagonal damping  scalar mu*I vs Marquardt mu*diag(J'J) vs a spread D
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

OUT = c.RESULTS / "ladder.jsonl"
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


def aug_scalar(A, b, mu, d):
    if mu <= 0:
        return A, b
    return np.vstack([A, np.sqrt(mu) * np.eye(d)]), np.concatenate([b, np.zeros(d)])


def aug_diag(A, b, dvec, d):
    return (np.vstack([A, np.diag(np.sqrt(dvec))]),
            np.concatenate([b, np.zeros(d)]))


# ------------------------------------------------------------------- L1 ---
def data_l1(Ns=(256, 512), fracs=(0.0, 0.25, 0.5, 0.75, 0.9),
            cws=(0, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768)):
    """After a damped phase resolving the top r_mu directions, how much stored
    state does the terminal undamped solve need?  Warm start from w_mu."""
    rows = []
    for N in Ns:
        p = setup(N)
        s, r, d = p["s"], p["r"], p["d"]
        floor = err(p, p["Vt"].T @ ((p["U"].T @ p["y"]) / np.where(s > 1e-15 * s[0], s, np.inf)))
        for fr in fracs:
            j = max(int(fr * r) - 1, 0)
            mu = float(s[j] ** 2) if fr > 0 else None
            if mu is None:
                w0 = np.zeros(d)
                rmu = 0
            else:
                w0 = p["Vt"].T @ ((s / (s ** 2 + mu)) * (p["U"].T @ p["y"]))
                rmu = int((s > np.sqrt(mu)).sum())
            e0 = err(p, w0)
            cstar = None
            for cw in cws:
                if cw > 1.5 * r:
                    break
                x, _ = lsqr_window(p["A"], p["y"] - p["A"] @ w0, I, d,
                                   int(2 * r), cw)
                if err(p, w0 + x) <= 3 * floor:
                    cstar = cw
                    break
            rows.append(dict(panel="L1", N=N, d=d, r=r, frac=fr, r_mu=rmu,
                             warm_err=e0, floor=floor, cstar=cstar,
                             predicted=r - rmu))
            log(f"L1 N={N} r_mu={rmu:4d} warm={e0:.2e} c*={cstar} "
                f"(naive prediction r-r_mu={r-rmu})")
    return rows


# ------------------------------------------------------------------- L2 ---
def data_l2(Ns=(256, 512), cs=(0, 16, 32, 64), Ks=(1, 2, 4, 8, 16), T=None):
    """K descending damping stages, fixed window c, state discarded between
    stages.  Peak state is d*c regardless of K."""
    rows = []
    for N in Ns:
        p = setup(N)
        s, r, d = p["s"], p["r"], p["d"]
        floor = err(p, p["Vt"].T @ ((p["U"].T @ p["y"]) / np.where(s > 1e-15 * s[0], s, np.inf)))
        budget = T or int(2 * r)
        for cw in cs:
            for K in Ks:
                js = np.unique(np.clip(
                    np.round(np.linspace(r / K, r, K)).astype(int) - 1, 0, r - 1))
                w = np.zeros(d)
                traj = []
                for k, j in enumerate(js):
                    mu = float(s[j] ** 2) if k < len(js) - 1 else 0.0
                    Aa, ya = aug_scalar(p["A"], p["y"] - p["A"] @ w, mu, d)
                    dlt, _ = lsqr_window(Aa, ya, I, d,
                                         max(budget // len(js), 8), cw)
                    w = w + dlt
                    traj.append(err(p, w))
                rows.append(dict(panel="L2", N=N, d=d, r=r, cwin=cw, K=int(len(js)),
                                 eval=float(min(traj)), final=float(traj[-1]),
                                 floor=floor, traj=[float(t) for t in traj],
                                 peak_state=int(d * max(cw, 1))))
                log(f"L2 N={N} c={cw:4d} K={len(js):3d} best={min(traj):.2e} "
                    f"(floor {floor:.1e})")
    return rows


# ------------------------------------------------------------------- L3 ---
def data_l3(N=256, spreads=(0, 2, 4, 8), seed=0,
            cws=(0, 8, 16, 32, 64, 128, 192, 256)):
    """Per-coordinate damping. Same geometric-mean damping in every variant;
    only the SPREAD of the diagonal changes.  Plus Marquardt's diag(J'J)."""
    p = setup(N)
    s, r, d = p["s"], p["r"], p["d"]
    A, y = p["A"], p["y"]
    floor = err(p, p["Vt"].T @ ((p["U"].T @ y) / np.where(s > 1e-15 * s[0], s, np.inf)))
    mu0 = float(s[min(128, r - 1)] ** 2)
    rng = np.random.default_rng(seed)
    variants = []
    for sp in spreads:
        dv = np.full(d, mu0) if sp == 0 else mu0 * 10.0 ** (
            rng.uniform(-sp / 2, sp / 2, d))
        variants.append((f"spread $10^{{{sp}}}$" if sp else r"scalar $\mu I$", dv))
    cn = (A ** 2).sum(0)
    variants.append(("Marquardt diag($J^TJ$)",
                     mu0 * cn / float(np.exp(np.mean(np.log(cn))))))
    rows = []
    for nm, dv in variants:
        Aa, ya = aug_diag(A, y, dv, d)
        sv = np.linalg.svd(Aa, compute_uv=False)
        kap = float(sv[0] / sv[-1])
        cstar, best = None, np.inf
        accs = {}
        for cw in cws:
            if cw > 1.5 * r:
                break
            x, _ = lsqr_window(Aa, ya, I, d, int(2 * r), cw)
            e = err(p, x)
            accs[cw] = e
            best = min(best, e)
            if cstar is None and e <= 3 * best and cw > 0 and e < 1.2 * accs[0]:
                pass
        # c* = smallest window reaching within 3x of this variant's own best
        for cw in sorted(accs):
            if accs[cw] <= 3 * best:
                cstar = cw
                break
        rows.append(dict(panel="L3", N=N, d=d, r=r, variant=nm, kappa=kap,
                         spread=float(dv.max() / dv.min()), cstar=cstar,
                         best=float(best), floor=floor,
                         dmin=float(dv.min()), dmax=float(dv.max()),
                         sigma1=float(s[0]), mu0=mu0,
                         kappa_pred=float(s[0] / np.sqrt(dv.min())),
                         accs={str(k): float(v) for k, v in accs.items()}))
        log(f"L3 {nm:<24} kappa={kap:.2e} spread={dv.max()/dv.min():.1e} "
            f"best={best:.2e} c*={cstar}")
    return rows


# --------------------------------------------------------------- figure ---
def figure(rows):
    fig, axs = plt.subplots(1, 3, figsize=(16.2, 5.8))
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.135, top=0.70,
                        wspace=0.42)

    ax = axs[0]
    l1 = [q for q in rows if q["panel"] == "L1"]
    for i, N in enumerate(sorted({q["N"] for q in l1})):
        sub = sorted([q for q in l1 if q["N"] == N], key=lambda q: q["r_mu"])
        ax.plot([q["r_mu"] for q in sub],
                [q["cstar"] if q["cstar"] is not None else np.nan for q in sub],
                "o-", color=f"C{i}", ms=6, lw=1.9, label=f"measured, N={N}")
        ax.plot([q["r_mu"] for q in sub], [q["predicted"] for q in sub],
                "--", color=f"C{i}", lw=1.4, alpha=.75,
                label=f"if only the tail mattered ($r-r_\\mu$), N={N}")
    ax.set_xlabel("$r_\\mu$ : directions already resolved by the damped phase")
    ax.set_ylabel("$c^*$ for the terminal solve")
    ax.set_title("L1  a warm start does NOT shrink the\nterminal solve's memory",
                 fontsize=9.5)
    ax.grid(alpha=.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.14), ncol=1,
              fontsize=7.5, frameon=False)

    ax = axs[1]
    l2 = [q for q in rows if q["panel"] == "L2"]
    N0 = sorted({q["N"] for q in l2})[0]
    sub0 = [q for q in l2 if q["N"] == N0]
    for i, cw in enumerate(sorted({q["cwin"] for q in sub0})):
        pts = sorted([(q["K"], q["eval"]) for q in sub0 if q["cwin"] == cw])
        ax.loglog([p[0] for p in pts], [max(p[1], 1e-17) for p in pts], "o-",
                  color=f"C{i}", ms=5.5, lw=1.8, label=f"fixed $c$={cw}")
    ax.axhline(sub0[0]["floor"], color="k", ls=":", lw=1.5, label="fp64 floor")
    ax.set_xlabel("$K$ : number of descending damping stages")
    ax.set_ylabel("best eval rel $L_2$")
    ax.set_title(f"L2  the $\\mu$-ladder at bounded peak state\n(QI-1D N={N0})",
                 fontsize=9.5)
    ax.set_ylim(1e-16, 1e-2)
    ax.grid(alpha=.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.14), ncol=3,
              fontsize=7.5, frameon=False)

    ax = axs[2]
    l3 = [q for q in rows if q["panel"] == "L3"]
    sc = [q for q in l3 if "spread" in q["variant"] or "scalar" in q["variant"]]
    sc = sorted(sc, key=lambda q: q["spread"])
    mq = [q for q in l3 if "Marquardt" in q["variant"]]
    sp = [max(q["spread"], 1.0) for q in sc]
    ax.loglog(sp, [q["best"] for q in sc], "o-", color="C0", ms=6.5, lw=1.9,
              label="best accuracy reached")
    if mq:
        ax.loglog([1.0], [mq[0]["best"]], "*", color="C2", ms=16,
                  label="Marquardt diag($J^TJ$)")
    ax.set_xlabel(r"damping spread  $\max_i \mu_i / \min_i \mu_i$")
    ax.set_ylabel("best eval rel $L_2$")
    ax.grid(alpha=.3, which="both")
    ax2 = ax.twinx()
    ax2.loglog(sp, [q["kappa"] for q in sc], "s--", color="C3", ms=5, lw=1.6,
               label=r"$\kappa$ of damped operator")
    ax2.loglog(sp, [q["kappa_pred"] for q in sc], ":", color="k", lw=1.6,
               label=r"$\sigma_1/\sqrt{\min_i d_i}$  (predicted)")
    ax2.set_ylabel(r"$\kappa$ of the damped operator", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax.set_title("L3  per-coordinate damping: the SMALLEST\ndamping sets $\\kappa$ "
                 "(all variants keep $c^*=0$)", fontsize=9.5)
    h1, la1 = ax.get_legend_handles_labels()
    h2, la2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, la1 + la2, loc="lower center",
              bbox_to_anchor=(0.5, 1.14), ncol=2, fontsize=7.5, frameon=False)

    fig.text(0.5, 0.985,
             "L   Can a damping schedule reach the floor at bounded memory, and what does per-coordinate damping cost?",
             ha="center", va="top", fontsize=12, weight="bold")
    fig.text(0.5, 0.945,
             "L1 and L2 both say no: neither warm-starting from a damped solution nor splitting the descent into stages reduces what the terminal solve must store.\n"
             "L3 answers the per-coordinate question: a spread diagonal is HARDER, and the damage is governed by the least-damped (most-trusted) coordinate alone.",
             ha="center", va="top", fontsize=8.8, linespacing=1.5)
    fig.savefig(FIG / "L_ladder.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    rows = []
    for fn in (data_l1, data_l2, data_l3):
        rs = fn()
        for q in rs:
            c.append(OUT, q)
        rows.extend(rs)
    figure(rows)
    log(f"figure -> {FIG/'L_ladder.png'}")
