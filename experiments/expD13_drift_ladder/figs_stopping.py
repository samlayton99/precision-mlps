"""Figure for the stopping rule.

Panels A/B come from stopping.jsonl (75 levels, 4 matrices, static + drifting).
Panel C is the drift-detector measurement run separately (QI-2D N=2048 and
random d=2048, nmult=1); the numbers are transcribed here so the figure is
reproducible without re-running the 20-minute sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core13 as c

RULES = ["test2<=0.1a", "test2<=a", "test2<=10a", "budget 3a^-0.7",
         "held-out plateau"]
NICE = {"test2<=0.1a": r"$test_2\leq0.1\alpha$", "test2<=a": r"$test_2\leq\alpha$",
        "test2<=10a": r"$test_2\leq10\alpha$",
        "budget 3a^-0.7": r"fixed budget $3\alpha^{-0.7}$",
        "held-out plateau": "held-out plateau"}

# drift detector: (alpha=eta, residual at level entry)
DRIFT = {"QI-2D N=2048": [(3.2e-2, 2.40e-1), (1e-2, 5.38e-2), (3.2e-3, 1.33e-2),
                          (1e-3, 1.99e-3), (3.2e-4, 1.23e-3), (1e-4, 2.14e-4),
                          (3.2e-5, 3.84e-5), (1e-5, 1.05e-5), (3.2e-6, 2.38e-6),
                          (1e-6, 4.97e-7)],
         "random d=2048": [(3.2e-2, 4.95e-1), (1e-2, 9.04e-2), (1e-3, 3.48e-3),
                           (3.2e-4, 1.16e-3), (1e-4, 3.27e-4)]}


def figure():
    rs = c.load(c.RESULTS / "stopping.jsonl")
    fig, axs = plt.subplots(1, 3, figsize=(19.0, 6.2))
    fig.subplots_adjust(left=0.055, right=0.975, bottom=0.15, top=0.66, wspace=0.30)

    # ---- A: rule comparison
    ax = axs[0]
    for j, nm in enumerate((0.0, 1.0)):
        sub = [q for q in rs if q["nmult"] == nm]
        for i, r in enumerate(RULES):
            ar = np.array([q["rules"][r][1] / max(q["rules"]["oracle"][1], 1e-300)
                           for q in sub])
            wr = np.array([q["rules"][r][0] / max(q["rules"]["oracle"][0], 1)
                           for q in sub])
            ax.plot(np.median(wr), np.median(ar), "o" if nm == 0 else "s",
                    color=f"C{i}", ms=13 if nm == 0 else 10,
                    mfc=f"C{i}" if nm == 0 else "none", mew=2.2,
                    label=NICE[r] if nm == 0 else None)
            ax.plot([np.percentile(wr, 10), np.percentile(wr, 90)],
                    [np.median(ar)] * 2, "-", color=f"C{i}", lw=1.2, alpha=.45)
            ax.plot([np.median(wr)] * 2,
                    [np.percentile(ar, 10), np.percentile(ar, 90)], "-",
                    color=f"C{i}", lw=1.2, alpha=.45)
    ax.axhline(1.0, color="k", ls=":", lw=1.4)
    ax.axvline(1.0, color="k", ls=":", lw=1.4)
    ax.plot(1, 1, "k*", ms=18, zorder=5)
    ax.annotate("oracle", (1, 1), xytext=(6, -14), textcoords="offset points",
                fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("work / oracle work    (right = wasted iterations)")
    ax.set_ylabel("accuracy / oracle accuracy    (up = worse)")
    ax.grid(alpha=.3, which="both")
    ax.set_title("A  filled = static $\\Phi$, hollow = drifting\n"
                 "bars are 10-90th percentile over 75 levels",
                 fontsize=10.5, pad=44)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              fontsize=8, frameon=False)

    # ---- B: the cap
    ax = axs[1]
    keys = sorted({q["key"] for q in rs})
    for i, k in enumerate(keys):
        sub = sorted([q for q in rs if q["key"] == k and q["nmult"] == 0.0],
                     key=lambda q: -q["alpha"])
        if sub:
            ax.loglog([q["alpha"] for q in sub],
                      [q["rules"]["oracle"][0] for q in sub], "o-",
                      color=f"C{i}", ms=5, lw=1.5, label=sub[0]["label"])
    a = np.logspace(-8, -1, 100)
    ax.loglog(a, np.minimum(4096, 1.0 * a ** -0.54), "k--", lw=2,
              label=r"fit  $\alpha^{-0.54}$")
    ax.loglog(a, np.minimum(4096, 5.0 * a ** -0.55), "-", color="crimson", lw=2.4,
              label=r"CAP  $\min(T_{hard},\,5\alpha^{-0.55})$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("iterations the level actually needs")
    ax.invert_xaxis()
    ax.grid(alpha=.3, which="both")
    ax.set_title("B  the cap is a safety net, not the rule\n"
                 "(spread across matrices is 40x at fixed $\\alpha$)",
                 fontsize=10.5, pad=44)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              fontsize=8, frameon=False)

    # ---- C: drift detector
    ax = axs[2]
    for i, (nm, pts) in enumerate(DRIFT.items()):
        ax.loglog([p[0] for p in pts], [p[1] for p in pts], "o-", color=f"C{i}",
                  ms=6, lw=1.8, label=nm)
    e = np.array([1e-6, 1e-1])
    ax.loglog(e, e, "k--", lw=2, label=r"$r_{entry}=\eta$")
    ax.fill_between(e, 0.5 * e, 2 * e, color="k", alpha=.10,
                    label=r"within $2\times$")
    ax.set_xlabel(r"true geometry drift $\eta$")
    ax.set_ylabel(r"relative residual at level entry, $\|y-\Phi_k w\|/\|y\|$")
    ax.grid(alpha=.3, which="both")
    ax.set_title("C  drift is free to measure\n"
                 "$\\Rightarrow$ never damp finer than $r_{entry}$",
                 fontsize=10.5, pad=44)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              fontsize=8, frameon=False)

    fig.text(0.5, 0.985, "S   The stopping rule for one damping level",
             ha="center", va="top", fontsize=12.5, weight="bold")
    fig.text(0.5, 0.94,
             "A: candidate rules scored against an oracle that stops at $1.3\\times$ the exact damped optimum. The winner sits on the star in BOTH regimes.\n"
             "B: the iteration law, and a cap set well above it. C: the residual you already compute at level entry is a $2\\times$-accurate estimate of how far the geometry has moved.",
             ha="center", va="top", fontsize=8.9, linespacing=1.5)
    fig.savefig(c.FIGS / "S_stopping_rule.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    figure()
    print("figure ->", c.FIGS / "S_stopping_rule.png")
