"""Experiment I figures -- the iterations/k/accuracy tradeoff, and its law.

Row 1  iso-ACCURACY lines: x = k, y = iterations to FIRST reach tau, one
       line per tau, one panel per target, at fixed d.
Row 2  iso-d lines at tau = 1e-12: x = k, y = iterations, one line per d.
Row 3  trajectories (x = iteration, y = disagreement, one line per k) --
       the view that shows WHY the iso-lines look the way they do.

A separate panel fits and reports the exponents of

    t  ~  C * tau^(-a) * k^(-b) * d^(c)

by least squares in logs, over the cells that were not truncated by the cap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import common_val as cv  # noqa: E402

TAUS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
FNS = ["sine", "runge", "abs_cubed"]


def _fit(rows):
    """log t = log C - a log tau - b log k + c log d, on untruncated hits."""
    X, Y = [], []
    for r in rows:
        for tau in TAUS:
            t = r.get(f"t_{tau:.0e}", -1)
            if t and t > 0:
                X.append([1.0, np.log10(1.0 / tau), np.log10(r["k"]),
                          np.log10(r["d"])])
                Y.append(np.log10(t))
    if len(Y) < 10:
        return None
    X = np.asarray(X); Y = np.asarray(Y)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ coef
    r2 = 1 - np.sum((Y - pred) ** 2) / np.sum((Y - Y.mean()) ** 2)
    return {"logC": coef[0], "a": coef[1], "b": -coef[2], "c": coef[3],
            "r2": float(r2), "n": len(Y)}


def render():
    rows = cv.load_rows("I_isolines")
    if not rows:
        print("  no I data")
        return
    ks = sorted({r["k"] for r in rows})
    ds = sorted({r["d"] for r in rows})
    d_mid = ds[len(ds) // 2]

    tcm = plt.get_cmap("viridis")
    tcol = {t: tcm(x) for t, x in zip(TAUS, np.linspace(0.08, 0.9, len(TAUS)))}
    dcm = plt.get_cmap("plasma")
    dcol = {d: dcm(x) for d, x in zip(ds, np.linspace(0.05, 0.85, len(ds)))}
    kcm = plt.get_cmap("cividis")
    kcol = {k: kcm(x) for k, x in zip(ks, np.linspace(0.05, 0.9, len(ks)))}

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 14))

    # ---------------- row 1: iso-accuracy ----------------
    for j, fn in enumerate(FNS):
        ax = axes[0, j]
        for tau in TAUS:
            pts = [(r["k"], r[f"t_{tau:.0e}"]) for r in rows
                   if r["fn"] == fn and r["d"] == d_mid
                   and r.get(f"t_{tau:.0e}", -1) > 0]
            if pts:
                pts.sort()
                ax.loglog([p[0] for p in pts], [p[1] for p in pts], "-o",
                          color=tcol[tau], ms=5, lw=1.8,
                          label=f"$\\tau=10^{{{int(np.log10(tau))}}}$")
        ax.set_xlabel("block size $k$")
        ax.set_ylabel("iterations to FIRST reach $\\tau$")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"{fn}  ($d={d_mid}$)", fontsize=11)
        if j == 0:
            ax.legend(fontsize=7, ncol=2, title="iso-accuracy",
                      title_fontsize=7)

    # ---------------- row 2: iso-d at tau = 1e-12 ----------------
    for j, fn in enumerate(FNS):
        ax = axes[1, j]
        for d in ds:
            pts = [(r["k"], r["t_1e-12"]) for r in rows
                   if r["fn"] == fn and r["d"] == d and r.get("t_1e-12", -1) > 0]
            if pts:
                pts.sort()
                ax.loglog([p[0] for p in pts], [p[1] for p in pts], "-o",
                          color=dcol[d], ms=5, lw=1.8, label=f"$d={d}$")
        kk = np.array(ks, float)
        ax.loglog(kk, 3e5 * (kk / kk[0]) ** -2.0, "k--", lw=1.2,
                  label=r"$\propto k^{-2}$")
        ax.set_xlabel("block size $k$")
        ax.set_ylabel(r"iterations to first reach $10^{-12}$")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"{fn}: iso-$d$ at $\\tau=10^{{-12}}$", fontsize=11)
        if j == 0:
            ax.legend(fontsize=7, ncol=2)

    # ---------------- row 3: trajectories ----------------
    # pick a d that actually HAS stored trajectories (only some N were kept)
    d_traj = sorted({r["d"] for r in rows if "traj_it" in r})
    d_traj = d_traj[len(d_traj) // 2] if d_traj else d_mid
    for j, fn in enumerate(FNS):
        ax = axes[2, j]
        got = False
        for k in ks:
            r = next((r for r in rows if r["fn"] == fn and r["d"] == d_traj
                      and r["k"] == k and "traj_it" in r), None)
            if r is None:
                continue
            got = True
            ax.loglog(r["traj_it"], np.maximum(r["traj_dis"], 1e-17),
                      color=kcol[k], lw=1.5, label=f"$k={k}$")
        for tau in (1e-4, 1e-8, 1e-12):
            ax.axhline(tau, color="0.6", ls=":", lw=1.0)
        ax.set_ylim(1e-16, 1e1)
        ax.set_xlabel("LSQR iteration")
        ax.set_ylabel(r"$\|A_{ev}(w-w_{svd})\|/\|y_{ev}\|$")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"{fn}: trajectories at $d={d_traj}$"
                     + ("" if got else "  (no trajectory stored)"), fontsize=11)
        if j == 0 and got:
            ax.legend(fontsize=7, ncol=2)

    fit = _fit(rows)
    sub = ""
    if fit:
        sub = (f"fitted over {fit['n']} (cell, $\\tau$) pairs:   "
               f"$t \\approx 10^{{{fit['logC']:.2f}}}\\,"
               f"\\tau^{{-{fit['a']:.2f}}}\\,k^{{-{fit['b']:.2f}}}\\,"
               f"d^{{{fit['c']:.2f}}}$   ($R^2={fit['r2']:.3f}$)")
        print("  FIT:", sub)
    fig.suptitle("I -- the accuracy / block-size / iteration tradeoff.\n" + sub,
                 y=1.0, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    cv.FIGS.mkdir(parents=True, exist_ok=True)
    p = cv.FIGS / "I_isolines.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")
    return fit


def render_law():
    """Second figure: the three THEORY tests, stated in plots_iso docstring.

    T1  t vs 1/tau against the tau^{-1/2} prediction (Krylov cost of resolving
        down to sigma/sigma_1 = tau, since kappa_eff(tau) = 1/tau).
    T2  t vs k/d -- if whitening flattens a WINDOW of k adjacent directions,
        k enters only through k/d and the curves must COLLAPSE across d.
        A pure power law in k would NOT collapse.
    T3  the k-exponent fitted separately at each d. If the k/d model is right,
        the fitted b must DRIFT with d rather than being a constant.
    """
    rows = cv.load_rows("I_isolines")
    if not rows:
        return
    ds = sorted({r["d"] for r in rows})
    dcm = plt.get_cmap("plasma")
    dcol = {d: dcm(x) for d, x in zip(ds, np.linspace(0.05, 0.85, len(ds)))}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # T1 -- tau scaling
    ax = axes[0]
    for d in ds:
        rs = [r for r in rows if r["fn"] == "sine" and r["d"] == d]
        for r in rs:
            if r["k"] not in (32, 64):
                continue
            xs = [1 / t for t in TAUS if r.get(f"t_{t:.0e}", -1) > 0]
            ys = [r[f"t_{t:.0e}"] for t in TAUS if r.get(f"t_{t:.0e}", -1) > 0]
            if len(xs) >= 3:
                ax.loglog(xs, ys, "-o", color=dcol[d], ms=3.5, lw=1.1,
                          alpha=0.85)
    xr = np.array([1e2, 1e12])
    ax.loglog(xr, 3 * xr ** 0.5, "k--", lw=1.6, label=r"theory $\tau^{-1/2}$")
    ax.loglog(xr, 30 * xr ** (1 / 3), color="tab:red", ls="-.", lw=1.6,
              label=r"measured $\approx\tau^{-1/3}$")
    ax.set_xlabel(r"$1/\tau$")
    ax.set_ylabel("iterations to first reach $\\tau$")
    ax.set_title("T1: accuracy scaling (sine, $k=32,64$)\n"
                 "power law, NOT logarithmic", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # T2 -- the k/d collapse test
    ax = axes[1]
    for d in ds:
        pts = sorted((r["k"] / r["d"], r["t_1e-12"]) for r in rows
                     if r["fn"] == "sine" and r["d"] == d
                     and r.get("t_1e-12", -1) > 0)
        if len(pts) >= 3:
            ax.semilogy([p[0] for p in pts], [p[1] for p in pts], "-o",
                        color=dcol[d], ms=5, lw=1.7, label=f"$d={d}$")
    ax.set_xlabel("$k/d$")
    ax.set_ylabel(r"iterations to reach $10^{-12}$")
    ax.set_title("T2: do the curves COLLAPSE in $k/d$?\n"
                 "collapse => $k$ enters only as $k/d$", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    # T3 -- does the fitted k-exponent drift with d?
    ax = axes[2]
    for fn, mk in (("sine", "o"), ("runge", "s"), ("abs_cubed", "^")):
        xs, ys = [], []
        for d in ds:
            pts = [(r["k"], r["t_1e-12"]) for r in rows
                   if r["fn"] == fn and r["d"] == d and r.get("t_1e-12", -1) > 0]
            if len(pts) < 4:
                continue
            X = np.log10([p[0] for p in pts])
            Y = np.log10([p[1] for p in pts])
            b = -np.polyfit(X, Y, 1)[0]
            xs.append(d); ys.append(b)
        if xs:
            ax.plot(xs, ys, "-" + mk, ms=6, lw=1.6, label=fn)
    ax.set_xlabel("$d$")
    ax.set_ylabel(r"fitted exponent $b$ in $t\propto k^{-b}$")
    ax.set_title("T3: is $b$ a constant?\n"
                 "drift => a single power law in $k$ is wrong", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("I(law) -- testing the predicted form  "
                 r"$t\approx C\,\tau^{-1/2}d^{\gamma}e^{-\beta k/d}$  "
                 "against the pure power law", y=1.03, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = cv.FIGS / "I2_law_tests.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


if __name__ == "__main__":
    render()
    render_law()
