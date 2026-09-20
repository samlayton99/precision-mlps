"""Experiment H figure -- the (d, k, iterations) law.

Replaces B / E / F / G, all four of which were deleted for confounds:
  B  fixed iteration budget          -> k looked coupled
  E  ratio-to-floor (saturates at 1) -> could not resolve below the floor
  F  best-over-trajectory + saturating metric -> everything read exactly 1.0
  G  budget scaled as 1/k, starving small k -> k looked coupled again

H runs every cell with NO early stopping to a generous per-cell cap, takes
the minimum over the trajectory, and reads every metric at that single
iterate. Targets are limited-smoothness (|x| is C^0, |x|^3 is C^2) so the
approximation floor keeps falling with d and the metric never saturates.
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

KCOL = {16: "#7b3294", 32: "tab:red", 64: "tab:orange",
        128: "tab:blue", 256: "tab:green"}


def render():
    rows = cv.load_rows("H_law")
    if not rows:
        print("  no H data")
        return
    fns = sorted({r["fn"] for r in rows})
    ks = sorted({r["k"] for r in rows})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))

    # ---- (0,0) and (0,1): accuracy vs d at fixed k, per target -----------
    for j, fn in enumerate(fns):
        ax = axes[0, j]
        for k in ks:
            rs = sorted([r for r in rows if r["fn"] == fn and r["k"] == k],
                        key=lambda r: r["d"])
            if not rs:
                continue
            ok = [r for r in rs if not r["not_converged"]]
            nc = [r for r in rs if r["not_converged"]]
            ax.loglog([r["d"] for r in rs], [max(r["dis"], 1e-17) for r in rs],
                      "-", color=KCOL[k], lw=1.8, label=f"$k={k}$")
            if ok:
                ax.plot([r["d"] for r in ok], [max(r["dis"], 1e-17) for r in ok],
                        "o", color=KCOL[k], ms=6)
            if nc:
                ax.plot([r["d"] for r in nc], [max(r["dis"], 1e-17) for r in nc],
                        "x", color=KCOL[k], ms=9, mew=2.2)
        fl = sorted({(r["d"], r["floor"]) for r in rows if r["fn"] == fn})
        ax.loglog([f[0] for f in fl], [f[1] for f in fl], "k--", lw=1.6,
                  label="approximation floor")
        ax.set_xlabel("$d$  (readout parameters)")
        ax.set_ylabel(r"$\|A_{ev}(w-w_{svd})\|/\|y_{ev}\|$  at the minimum")
        ax.set_ylim(1e-16, 1e-2)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"{fn}: solver error vs the DIRECT solve\n"
                     "flat in $d$ at fixed $k$ = MEMORY uncoupled "
                     r"($\times$ = hit cap, still improving)", fontsize=10)
        ax.legend(fontsize=8, ncol=2, loc="lower left")

    # ---- (1,0): iterations to the minimum vs d --------------------------
    ax = axes[1, 0]
    for k in ks:
        rs = sorted([r for r in rows if r["fn"] == "abs_cubed" and r["k"] == k
                     and not r["not_converged"]], key=lambda r: r["d"])
        if len(rs) >= 2:
            ax.loglog([r["d"] for r in rs], [r["it_min"] for r in rs], "-o",
                      color=KCOL[k], ms=6, lw=1.8, label=f"$k={k}$")
    ds = np.array(sorted({r["d"] for r in rows}), float)
    ax.loglog(ds, 200 * (ds / ds[0]) ** 1.6, "k--", lw=1.3,
              label=r"$\propto d^{1.6}$")
    ax.set_xlabel("$d$")
    ax.set_ylabel("iterations to the minimum")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("COMPUTE vs $d$ (abs_cubed): this is what is coupled",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=2)

    # ---- (1,1): iterations vs k, the memory/compute exchange rate -------
    ax = axes[1, 1]
    dcm = plt.get_cmap("plasma")
    dsu = sorted({r["d"] for r in rows})
    dcol = {d: dcm(t) for d, t in zip(dsu, np.linspace(0.05, 0.82, len(dsu)))}
    for d in dsu:
        rs = sorted([r for r in rows if r["fn"] == "abs_cubed" and r["d"] == d
                     and not r["not_converged"]], key=lambda r: r["k"])
        if len(rs) >= 2:
            ax.loglog([r["k"] for r in rs], [r["it_min"] for r in rs], "-o",
                      color=dcol[d], ms=6, lw=1.8, label=f"$d={d}$")
    kk = np.array(ks, float)
    ax.loglog(kk, 3e5 * (kk / kk[0]) ** -2.5, "k--", lw=1.3,
              label=r"$\propto k^{-2.5}$")
    ax.set_xlabel("block size $k$")
    ax.set_ylabel("iterations to the minimum")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("the exchange rate: buying memory buys iterations back",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("H -- the $(d, k, \\mathrm{iterations})$ law, no early "
                 "stopping, all metrics at one iterate.\nTargets have limited "
                 "smoothness so the floor keeps falling with $d$ and never "
                 "saturates the metric.", y=1.0, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    cv.FIGS.mkdir(parents=True, exist_ok=True)
    p = cv.FIGS / "H_dk_iteration_law.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


if __name__ == "__main__":
    render()
