"""expD11 -- why no O(d*k) iterative solver reaches the fp64 floor on QI systems.

Every panel here isolates ONE variable on a FROZEN operator with a FIXED
accumulated buffer, so batching, drift and sampling are out of the picture and
cannot be blamed.

D1  the two regimes      accuracy vs reorthogonalization window c, with and
                         without a block-Jacobi preconditioner
D2  the spectral law     accuracy vs c on synthetic spectra that differ ONLY in
                         how many distinct scales they contain
D3  the scaling verdict  fixed c across N at three bandwidths lambda, plus how
                         the QI spectral gaps close as N grows

Writes disproof.jsonl and three PNGs.
"""
from __future__ import annotations

import json
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
from src.construction.qi_mpmath import default_halo

OUT = c.RESULTS / "disproof.jsonl"
FIG = c.FIGS
T0 = time.time()
TARGET = lambda z: np.sin(4.0 * z)


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def bjac(Aa, d, k=64, rcond=1e-13):
    fac = []
    for t in range(0, d, k):
        C = np.arange(t, min(t + k, d))
        _, s, Vt = np.linalg.svd(Aa[:, C], full_matrices=False)
        kp = s > rcond * max(s[0], 1e-300)
        fac.append((C, Vt.T, np.where(kp, 1.0 / np.where(kp, s, 1.0), 0.0)))

    def Mh(v):
        o = np.zeros(d)
        for C, V, ih in fac:
            o[C] = V @ (ih * (V.T @ v[C]))
        return o
    return Mh


def qi_system(N, lam, row_mult=6):
    """QI-1D feature matrix on the construction's own geometry."""
    h = 2.0 / N
    gamma = lam / h
    halo = int(default_halo(N, lambda_star=lam))
    cen = -1.0 + h * np.arange(-halo, N + halo + 1)
    W = cen.size
    xt = np.linspace(-1, 1, row_mult * W)
    xe = np.linspace(-1, 1, 4001)
    A = np.hstack([np.tanh(gamma * (xt[:, None] - cen[None, :])),
                   np.ones((xt.size, 1))])
    Ae = np.hstack([np.tanh(gamma * (xe[:, None] - cen[None, :])),
                    np.ones((xe.size, 1))])
    return A, TARGET(xt), Ae, TARGET(xe)


def solve_err(A, y, Ae, ye, Mh, d, r, cw):
    cv = np.inf if cw == "full" else int(cw)
    x, _ = lsqr_window(A, y, Mh, d, int(2 * r), cv)
    return float(np.linalg.norm(Ae @ Mh(x) - ye) / np.linalg.norm(ye))


# ------------------------------------------------------------------ data ---
def data_d1(Ns=(128, 256, 384, 512), lam=0.30,
            cws=(0, 16, 32, 64, 96, 128, 192, 256, 384, 512, "full")):
    rows = []
    for N in Ns:
        A, y, Ae, ye = qi_system(N, lam)
        d = A.shape[1]
        s = np.linalg.svd(A, compute_uv=False)
        r = int((s > 1e-15 * s[0]).sum())
        log(f"D1 N={N} d={d} r={r}")
        for pname, Mh in (("none", (lambda v: v)), ("bjac k=64", bjac(A, d, 64))):
            for cw in cws:
                e = solve_err(A, y, Ae, ye, Mh, d, r, cw)
                rows.append(dict(panel="D1", N=N, d=d, r=r, lam=lam,
                                 precond=pname, cwin=str(cw), eval=e))
        log(f"   done N={N}")
    return rows


def data_d2(d=400, r=240, n=1600,
            cws=(0, 4, 8, 16, 32, 64, 128, 192, 240, "full")):
    rng = np.random.default_rng(3)
    U = np.linalg.qr(rng.standard_normal((n, r)))[0]
    V = np.linalg.qr(rng.standard_normal((d, r)))[0]
    Ue = np.linalg.qr(rng.standard_normal((800, r)))[0]
    g = rng.standard_normal(r)
    specs = {"2 levels": np.where(np.arange(r) < r // 2, 1.0, 1e-14),
             "4 levels": 10.0 ** (-14 * (np.arange(r) // (r // 4)) / 3),
             "16 levels": 10.0 ** (-14 * (np.arange(r) // (r // 16)) / 15),
             "64 levels": 10.0 ** (-14 * (np.arange(r) // (r // 64)) / 63),
             "gapless (r levels)": np.logspace(0, -14, r)}
    rows = []
    I = (lambda v: v)
    for name, s in specs.items():
        A, Ae = (U * s) @ V.T, (Ue * s) @ V.T
        coef = s * g
        y, ye = U @ coef, Ue @ coef
        for cw in cws:
            e = solve_err(A, y, Ae, ye, I, d, r, cw)
            rows.append(dict(panel="D2", spectrum=name, d=d, r=r,
                             cwin=str(cw), eval=e,
                             nlevels=int(np.unique(np.round(np.log10(s), 6)).size)))
        log(f"D2 {name}")
    return rows


def data_d3(lams=(0.05, 0.10, 0.25), Ns=(96, 192, 384, 768),
            cws=(0, 16, 32, 64)):
    rows = []
    I = (lambda v: v)
    for lam in lams:
        for N in Ns:
            A, y, Ae, ye = qi_system(N, lam)
            d = A.shape[1]
            s = np.linalg.svd(A, compute_uv=False)
            r = int((s > 1e-15 * s[0]).sum())
            gap = float(np.median(s[:r - 1] / s[1:r]))
            for cw in list(cws) + [r]:
                e = solve_err(A, y, Ae, ye, I, d, r, cw)
                rows.append(dict(panel="D3", lam=lam, N=N, d=d, r=r,
                                 medgap=gap, cwin=str(cw),
                                 is_r=(cw == r), eval=e))
            log(f"D3 lam={lam} N={N} d={d} r={r} medgap={gap:.3f}")
    return rows


# --------------------------------------------------------------- figures ---
def _finish(fig, axs, title, sub, handles_ax, ncol, top=0.80, lgy=0.845):
    """Shared top matter: suptitle, one figure-level legend, reserved margin.
    Legends never sit inside an axes and never collide with a subplot title."""
    fig.tight_layout(rect=[0, 0, 1, top])
    h, l = handles_ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, lgy), ncol=ncol,
               fontsize=8.5, frameon=False)
    fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=10.5,
             weight="bold")
    fig.text(0.5, 0.935, sub, ha="center", va="top", fontsize=8.6, linespacing=1.5)


def _dticks(ax, ds):
    """Linear d axis with one label per measured width -- a log axis draws
    minor-tick labels that collide at these spacings."""
    ax.set_xscale("linear")
    ax.set_xticks(ds)
    ax.set_xticklabels([str(v) for v in ds], fontsize=8)
    ax.set_xlim(min(ds) - 0.06 * (max(ds) - min(ds)),
                max(ds) + 0.06 * (max(ds) - min(ds)))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def fig_d1(rows):
    rs = [q for q in rows if q["panel"] == "D1"]
    Ns = sorted({q["N"] for q in rs})
    fig, axs = plt.subplots(1, len(Ns), figsize=(3.5 * len(Ns), 4.6),
                            sharey=True)
    axs = np.atleast_1d(axs)
    for ax, N in zip(axs, Ns):
        sub = [q for q in rs if q["N"] == N]
        r = sub[0]["r"]
        for pn, col, mk, lab in (("none", "C0", "o", "no preconditioner"),
                                 ("bjac k=64", "C3", "s",
                                  "block-Jacobi preconditioner, k=64")):
            pts = sorted([(np.inf if q["cwin"] == "full" else int(q["cwin"]),
                           q["eval"]) for q in sub if q["precond"] == pn])
            xs = [min(p[0], 1.3 * r) for p in pts]
            ax.semilogy(xs, [max(p[1], 1e-17) for p in pts], mk + "-",
                        color=col, ms=4.5, lw=1.8, label=lab)
        ax.axvline(r, color="k", ls="--", lw=1.2)
        ax.annotate(f"$c=r={r}$", xy=(r, 2e-4), fontsize=8, ha="right",
                    va="top", rotation=90, xytext=(-3, 0),
                    textcoords="offset points")
        ax.set_title(f"QI-1D  N={N}\n$d$={sub[0]['d']},  $r$={r}", fontsize=9.5)
        ax.set_xlabel("window $c$  (vectors stored)")
        ax.grid(alpha=.3, which="both")
        ax.set_ylim(1e-16, 1e-3)
    axs[0].set_ylabel("eval rel $L_2$")
    _finish(fig, axs,
            "D1   On a frozen operator the ONLY thing that reaches the fp64 floor is stored orthogonality -- and it needs $c \\approx r$",
            "Each panel: plain LSQR plus a sliding window of $c$ stored Golub-Kahan vectors, on a fixed accumulated buffer (no batching, no drift).\n"
            "Read: blue reaches $10^{-15}$, but only once $c$ crosses the dashed $c=r$ line. Red is FLAT at $\\sim10^{-6}$ -- the preconditioner caps accuracy and no window rescues it.",
            axs[0], 2)
    fig.savefig(FIG / "D1_two_regimes.png", dpi=140)
    plt.close(fig)


def fig_d2(rows):
    rs = [q for q in rows if q["panel"] == "D2"]
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    names = ["2 levels", "4 levels", "16 levels", "64 levels",
             "gapless (r levels)"]
    r = rs[0]["r"]
    for i, nm in enumerate(names):
        pts = sorted([(np.inf if q["cwin"] == "full" else int(q["cwin"]),
                       q["eval"]) for q in rs if q["spectrum"] == nm])
        xs = [min(p[0], 1.2 * r) for p in pts]
        ax.semilogy(xs, [max(p[1], 1e-17) for p in pts], "o-", color=f"C{i}",
                    ms=5, lw=1.9, label=nm)
    ax.axvline(r, color="k", ls="--", lw=1.2)
    ax.annotate(f"$c=r={r}$", xy=(r, 2e-4), fontsize=8.5, ha="right", va="top",
                rotation=90, xytext=(-3, 0), textcoords="offset points")
    ax.set_xlabel("window $c$  (vectors stored)")
    ax.set_ylabel("eval rel $L_2$")
    ax.set_ylim(1e-16, 1e-3)
    ax.grid(alpha=.3, which="both")
    _finish(fig, [ax],
            "D2   The state you must store is set by the SPECTRUM, not by $d$ or $\\kappa$",
            "Every curve has the same $d=400$, $r=240$, $\\kappa=10^{14}$ and the same target energy profile. The ONLY difference is how many distinct singular-value scales the matrix has.\n"
            "Read: 2- and 4-level spectra reach $10^{-15}$ at $c=0$, i.e. with zero stored state. A gapless spectrum needs the full $c=r$. QI feature matrices are the gapless case.",
            ax, 5, top=0.79, lgy=0.835)
    fig.savefig(FIG / "D2_spectral_law.png", dpi=140)
    plt.close(fig)


def fig_d3(rows):
    rs = [q for q in rows if q["panel"] == "D3"]
    lams = sorted({q["lam"] for q in rs})
    fig, axs = plt.subplots(1, len(lams) + 1, figsize=(3.9 * (len(lams) + 1), 4.8))
    for ax, lam in zip(axs[:-1], lams):
        sub = [q for q in rs if q["lam"] == lam]
        ds = sorted({q["d"] for q in sub})
        for i, cw in enumerate(["0", "16", "32", "64"]):
            pts = sorted([(q["d"], q["eval"]) for q in sub
                          if q["cwin"] == cw and not q["is_r"]])
            if pts:
                ax.plot([p[0] for p in pts], [max(p[1], 1e-17) for p in pts],
                        "o-", color=f"C{i}", ms=5, lw=1.8,
                        label=f"fixed $c$={cw}")
        pts = sorted([(q["d"], q["eval"]) for q in sub if q["is_r"]])
        ax.plot([p[0] for p in pts], [max(p[1], 1e-17) for p in pts],
                "^--", color="k", ms=6, lw=1.8, label="$c=r$  (state grows with $d$)")
        ax.set_yscale("log")
        _dticks(ax, ds)
        ax.set_title(f"$\\lambda$ = {lam}", fontsize=10)
        ax.set_xlabel("$d$  (columns)")
        ax.set_ylim(1e-16, 1e-4)
        ax.grid(alpha=.3, which="both")
    axs[0].set_ylabel("eval rel $L_2$")
    ax = axs[-1]
    allds = sorted({q["d"] for q in rs})
    for i, lam in enumerate(lams):
        pts = sorted({(q["d"], q["medgap"]) for q in rs if q["lam"] == lam})
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=f"C{i}",
                ms=5, lw=1.8)
        ax.annotate(f"$\\lambda$={lam}", xy=pts[0], fontsize=8.5,
                    color=f"C{i}", xytext=(4, 4), textcoords="offset points")
    ax.axhline(1.0, color="k", ls=":", lw=1.4)
    ax.annotate("no gaps left", xy=(allds[0], 1.0), fontsize=8, va="bottom",
                xytext=(2, 3), textcoords="offset points")
    _dticks(ax, allds[::2])
    ax.set_xlabel("$d$  (columns)")
    ax.set_ylabel("median $\\sigma_i/\\sigma_{i+1}$")
    ax.set_title("why: the spectrum goes gapless", fontsize=10)
    ax.grid(alpha=.3, which="both")
    _finish(fig, axs,
            "D3   No FIXED amount of stored state survives width scaling, at any bandwidth $\\lambda$",
            "Left three panels: accuracy vs width when the stored state $c$ is held fixed. Right panel: the median ratio between consecutive singular values.\n"
            "Read: every coloured fixed-$c$ line RISES with $d$ -- constant memory loses accuracy as the problem grows. Only the black $c=r$ line holds the floor, and its state is $\\Theta(d\\cdot r)=\\Theta(d^2)$.",
            axs[0], 5, top=0.79, lgy=0.835)
    fig.savefig(FIG / "D3_scaling_verdict.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    rows = []
    for fn in (data_d1, data_d2, data_d3):
        rs = fn()
        for q in rs:
            c.append(OUT, q)
        rows.extend(rs)
    fig_d1(rows); fig_d2(rows); fig_d3(rows)
    log(f"figures -> {FIG}")
