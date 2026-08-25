"""Figures for lead_B.  Loads the saved JSON where the run was expensive and
recomputes the cheap arms inline."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import scipy.linalg as sla                                 # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb, core11 as C, e3_invchol as E3, e5_local_global as E5  # noqa: E402
from scipy.sparse.linalg import lsqr, lsmr                  # noqa: E402

FIG = lb.OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CL = {"QI-1D N=128": "#1f77b4", "QI-1D N=256": "#2c8fbf",
      "QI-2D radon_tensor N=576": "#d62728", "twin-unstruct d=512": "#111111",
      "twin-cluster d=512": "#7f7f7f"}


def leg(ax, ncol=2, **kw):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=ncol,
              fontsize=7, frameon=False, borderaxespad=0, **kw)


# ---------------------------------------------------------------- F1 -------
def f1():
    """Gate 1 for every O(d k) approximation of the ideal preconditioner."""
    recs = json.loads((lb.OUT / "e3_invchol.json").read_text())
    r9 = json.loads((lb.OUT / "e9_refine.json").read_text())
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    ks = [4, 8, 16, 32, 64, 128]
    for ax, fam, ttl in zip(axes[:3], ["band", "blk", "ichol"],
                            [r"band$_k(R^{-1})$", r"blockdiag$_k(R^{-1})$",
                             r"band$_k(R)^{-1}$"]):
        for r in recs:
            v = [r.get("%s_%d" % (fam, k)) for k in ks]
            xs = [k for k, y in zip(ks, v) if y]
            ys = [y for y in v if y]
            if not ys:
                continue
            ax.plot(xs, ys, "o-", ms=3.5, lw=1.2,
                    color=CL.get(r["cell"], "#999"), label=r["cell"])
            ax.axhline(r["kappa"], color=CL.get(r["cell"], "#999"), lw=0.6,
                       ls=":", alpha=0.6)
        ax.axhspan(1, 1e2, color="#2ca02c", alpha=0.12)
        ax.set_yscale("log"); ax.set_xscale("log", base=2)
        ax.set_xlabel("bandwidth / block size $k$")
        ax.set_title(ttl, fontsize=9, pad=8)
        ax.set_ylim(1, 1e16); ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel(r"$\kappa(A M)$  (dd product)")
    h, l = axes[0].get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for a, b in zip(h, l):
        if b not in seen:
            seen.add(b); hh.append(a); ll.append(b)
    fig.legend(hh, ll, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5,
               fontsize=7.5, frameon=False)
    axes[0].text(0.04, 0.06, "dotted = unpreconditioned $\\kappa(A)$\n"
                 "green band = usable ($\\kappa\\leq 10^2$)\nfull $R^{-1}$ gives "
                 "$\\kappa\\approx1$", transform=axes[0].transAxes, fontsize=6.5,
                 va="bottom")
    ax = axes[3]
    for key, c in [(k, CL.get(k[4:], "#999")) for k in r9 if k.startswith("tol_")]:
        o = np.array(r9[key])
        x = np.where(o[:, 0] == 0, 1e-17, o[:, 0])
        ax.plot(x, o[:, 1], "o-", ms=4, lw=1.4, color=c, label=key[4:])
        kap = 3.07e14 if "QI" in key else 1.38e16
        ax.plot(x, np.maximum(1.0, x * kap), ":", lw=1.0, color=c,
                label=r"$\delta\,\kappa(A)$")
    ax.set_xscale("log"); ax.set_xlabel(r"additive perturbation $\delta$ of $R^{-1}$")
    ax.set_title(r"why no truncation survives: $\kappa(AM)\approx\delta\,\kappa(A)$",
                 fontsize=8, pad=8)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIG / "F1_invchol_gate1.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- F2 -------
def f2():
    """kappa_loc(k): the window-visibility law and the structure detector."""
    recs = json.loads((lb.OUT / "e5_local_global.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    ks = [16, 32, 64, 128]
    for r in recs:
        col = CL.get(r["cell"], "#999")
        y = [r["k%d" % k]["loc_med"] for k in ks]
        axes[0].plot(ks, y, "o-", ms=4, lw=1.3, color=col, label=r["cell"])
        axes[0].axhline(r["kappa"], color=col, ls=":", lw=0.6, alpha=0.6)
        axes[1].plot(ks, [v / r["kappa"] for v in y], "o-", ms=4, lw=1.3,
                     color=col, label=r["cell"])
        ya = [r["k%d" % k]["loc_med_after"] for k in ks]
        axes[0].plot(ks, ya, "s--", ms=3, lw=0.9, color=col, alpha=0.55)
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("window size $k$"); ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel(r"$\kappa_{\rm loc}(k)$  (median over random windows)")
    axes[0].set_title("solid: on $A$   dashed: after one block-whitening stage\n"
                      "dotted: global $\\kappa(A)$", fontsize=7.5, pad=40)
    axes[1].set_ylabel(r"$\kappa_{\rm loc}(k)\,/\,\kappa(A)$")
    axes[1].set_title("fraction of the conditioning a $k$-column window can see\n"
                      "(=1 means block methods can finish the job)", fontsize=7.5,
                      pad=40)
    axes[1].axhline(1.0, color="#2ca02c", lw=1.0, ls="-", alpha=0.7)
    leg(axes[0], ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    fig.savefig(FIG / "F2_kappa_loc.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- F3 -------
def f3():
    """Restarted refinement does not contract; and the Tikhonov path does
    reach the floor but costs sqrt(kappa) per decade to track."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3))
    p = C.prob_qi1d(128); A, y = p["A"], p["y"]; ny = np.linalg.norm(y)
    for m, c in zip([25, 50, 100, 200], ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]):
        w = np.zeros(p["d"]); tr = []
        for j in range(14):
            w = w + lsmr(lb.opA(A), y - A @ w, atol=0, btol=0, conlim=0,
                         maxiter=m)[0]
            tr.append(np.linalg.norm(y - A @ w) / ny)
        axes[0].plot(np.arange(1, 15) * m, tr, "o-", ms=3, lw=1.2, color=c,
                     label="restart every %d it" % m)
    un = [np.linalg.norm(y - A @ lsmr(lb.opA(A), y, atol=0, btol=0, conlim=0,
                                      maxiter=it)[0]) / ny
          for it in [25, 50, 100, 200, 400, 800, 1600, 2800]]
    axes[0].plot([25, 50, 100, 200, 400, 800, 1600, 2800], un, "k^--", ms=4,
                 lw=1.2, label="never restarted")
    axes[0].axhline(p["floor_pool"], color="#2ca02c", lw=1.2, label="pool floor")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("total LSMR iterations")
    axes[0].set_ylabel(r"$\|Aw-y\|/\|y\|$")
    axes[0].set_title("QI-1D $N{=}128$: refinement rounds do not contract",
                      fontsize=8, pad=40)
    leg(axes[0], ncol=2)

    U, s, Vt = np.linalg.svd(A, full_matrices=False); Uty = U.T @ y
    s1 = s[0]
    ex = np.logspace(0, -17, 40)
    res = [np.linalg.norm(y - A @ (Vt.T @ (s * Uty / (s ** 2 + (e * s1) ** 2)))) / ny
           for e in ex]
    axes[1].plot(ex, res, "-", color="#8c564b", lw=1.6,
                 label=r"exact Tikhonov path $w_\mu$")
    axes[1].axhline(p["floor_pool"], color="#2ca02c", lw=1.2, label="pool floor")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].invert_xaxis(); axes[1].set_xlabel(r"$\mu/\sigma_1$")
    axes[1].set_ylabel(r"$\|Aw_\mu-y\|/\|y\|$")
    axes[1].set_title("the path itself reaches the floor", fontsize=8, pad=40)
    leg(axes[1], ncol=1)

    from e2_continuation import stacked
    its, xs = [], []
    for a_ in range(0, -13, -2):
        mua, mub = s1 * 10.0 ** a_, s1 * 10.0 ** (a_ - 1)
        wa = Vt.T @ (s * Uty / (s ** 2 + mua ** 2))
        rhs = np.concatenate([y - A @ wa, -mub * wa])
        r = lsqr(stacked(A, mub), rhs, atol=1e-13, btol=1e-13, conlim=0,
                 iter_lim=6000)
        its.append(int(r[2])); xs.append(10.0 ** a_)
    axes[2].plot(xs, its, "o-", color="#8c564b", lw=1.4, ms=4,
                 label="LSQR its for one decade of $\\mu$")
    axes[2].plot(xs, [0.6 * np.sqrt(1 / x) for x in xs], "k:", lw=1.0,
                 label=r"$\propto\sqrt{\sigma_1/\mu}$")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].invert_xaxis(); axes[2].set_xlabel(r"$\mu/\sigma_1$ at stage entry")
    axes[2].set_ylabel("inner iterations (cap 6000)")
    axes[2].set_title("but tracking it costs $\\sqrt{\\kappa}$", fontsize=8, pad=40)
    leg(axes[2], ncol=1)
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.76))
    fig.savefig(FIG / "F3_restart_and_continuation.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- F4 -------
def f4():
    """fp64 vs double-double at O(d k) state, long budget.  Where the block
    preconditioner clusters the spectrum, fp64 hits a hard floor and dd walks
    through it; where it does not, dd buys nothing."""
    e1 = json.loads((lb.OUT / "e1_precision.json").read_text())
    e6 = json.loads((lb.OUT / "e6_best_odk.json").read_text())
    cells = []
    for r in e6:
        if r["cell"] not in cells:
            cells.append(r["cell"])
    fig, axes = plt.subplots(1, len(cells), figsize=(2.9 * len(cells), 3.7),
                             sharey=True)
    for ax, cell in zip(np.atleast_1d(axes), cells):
        rr = [r for r in e6 if r["cell"] == cell]
        pr = rr[0]["probe"]
        u = [r for r in e1 if r["cell"] == cell]
        if u:
            ax.plot(u[0]["probe"], [v[1] for v in u[0]["fp64"]], "-",
                    color="#bbbbbb", lw=1.2, label="fp64 LSQR, no precond")
            ax.plot(u[0]["probe"], [v[1] for v in u[0]["dd"]], "--",
                    color="#bbbbbb", lw=1.2, label="dd LSQR, no precond")
        for r, c in zip(rr, ["#1f77b4", "#d62728"]):
            ax.plot(pr, r["fp64"], "-", color=c, lw=1.6,
                    label="fp64 + block-Jacobi $k$=%d" % r["k"])
            ax.plot(pr, r["dd"], "--", color=c, lw=1.6,
                    label="dd + block-Jacobi $k$=%d" % r["k"])
        ax.axhline(rr[0]["floor"], color="#2ca02c", lw=1.4, label="pool floor")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-16, 3)
        ax.set_xlabel("LSQR iterations"); ax.grid(alpha=0.25, lw=0.4)
        ax.text(0.5, 1.01, cell, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=7.5)
        ax.text(0.03, 0.03, r"$\kappa_{\rm loc}/\kappa=%.0e$"
                % (rr[-1]["kappa_loc"] / rr[-1]["kappa"]),
                transform=ax.transAxes, fontsize=6.5, va="bottom")
    np.atleast_1d(axes)[0].set_ylabel("eval rel $L_2$")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(FIG / "F4_precision.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- F5 -------
def f5():
    """Requirements 5 and 7 for the O(d k) candidates."""
    try:
        recs = json.loads((lb.OUT / "e8_reqs.json").read_text())
        r9 = json.loads((lb.OUT / "e9_refine.json").read_text())
    except FileNotFoundError:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6))
    for key, c in zip([k for k in r9 if k.startswith("drift2_")],
                      ["#1f77b4", "#111111"]):
        o = np.array(r9[key])
        x = np.where(o[:, 0] == 0, 3e-14, o[:, 0])
        nm = key[7:]
        axes[0].plot(x, o[:, 1], "-", ms=3, color="#2ca02c", lw=1.6, alpha=0.8,
                     label=nm + " drifted-problem floor" if c == "#111111"
                     else None)
        axes[0].plot(x, o[:, 2], "o-", ms=4, color=c, lw=1.4,
                     label=nm + " block-Jacobi (steering)")
        axes[0].plot(x, o[:, 3], "s--", ms=4, color=c, lw=1.2, alpha=0.8,
                     label=nm + " block-QR (baked in)")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel(r"operator drift $\eta$ (leftmost point = 0)")
    axes[0].set_ylabel("eval rel $L_2$")
    axes[0].set_title("requirement 7: staleness.  Baked-in whitening TRACKS\n"
                      "the moving floor; it is not poisoned by it.",
                      fontsize=7.5, pad=54)
    leg(axes[0], ncol=1)
    for key, c in zip([k for k in recs if k.startswith("batch_")],
                      ["#1f77b4", "#111111"]):
        o = np.array(recs[key])
        axes[1].plot(o[:, 0], o[:, 1], "o-", ms=4, color=c, lw=1.4,
                     label=key[6:] + " achieved")
        axes[1].plot(o[:, 0], o[:, 2], "^:", ms=4, color=c, lw=1.1,
                     label=key[6:] + " one-batch floor")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("batch rows $b$")
    axes[1].set_ylabel("eval rel $L_2$")
    axes[1].set_title("requirement 5: fresh batch every restart\n(core11.type_a,"
                      " the repo's tuned streaming solver)", fontsize=7.5, pad=54)
    leg(axes[1], ncol=1)
    c = np.array(recs["cost"])
    for i, (nm, col) in enumerate([("block-Jacobi $d k$", "#1f77b4"),
                                   ("block-QR $d k$", "#ff7f0e"),
                                   ("sketch SVD $d r$", "#d62728")], start=1):
        e = np.polyfit(np.log(c[:, 0]), np.log(np.maximum(c[:, i], 1e-6)), 1)[0]
        axes[2].plot(c[:, 0], c[:, i], "o-", ms=4, color=col, lw=1.4,
                     label="%s  fit $d^{%.2f}$" % (nm, e))
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("$d$"); axes[2].set_ylabel("setup seconds")
    axes[2].set_title("requirement 3: measured setup exponent\n(timing, not a clean"
                      " flop count at these $d$)", fontsize=7.5, pad=54)
    leg(axes[2], ncol=1)
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.70))
    fig.savefig(FIG / "F5_requirements.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    for fn in (f1, f2, f3, f4, f5):
        try:
            fn(); print("ok", fn.__name__)
        except Exception as e:
            print("FAIL", fn.__name__, type(e).__name__, e)
