"""Init-slice plot for VarPro + Adam->GN (our data only), in the expD02
init_slice style: one subplot per target, x = N (width W = N + 2*halo + 1),
y = eval relative L2, both log; lines = the geometry-init ladder; a dashed fp64
floor reference. Reads results/.../expD04_varpro/varpro_gn_summary.csv.

This is the analogue of expD02's init_slice_adam, but the "regime" is
VarPro + Adam->Gauss-Newton (readout projected out) instead of Adam-both.

Usage: python experiments/expD04_varpro/plot_init_slice.py
"""

import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD04_varpro"
SUMMARY_CSV = RESULTS_DIR / "varpro_gn_summary.csv"
OUT = RESULTS_DIR / "figures" / "init_slice_varpro_gn.png"

# target display order (registry order); only those present are drawn
TARGET_ORDER = ["sine", "cosine", "sine_8pi", "runge", "tanh_steep",
                "sine_mixture", "exp", "poly5", "abs_cubed"]
# init ladder: key -> (label, color), drawn in this order
INITS = [
    ("xavier",          r"Xavier (random centers + $\gamma$)",      "#d62728"),
    ("center_spread",   r"grid centers, random $\gamma$",           "#ff7f0e"),
    ("scale_corrected", r"grid centers, correct-scale $\gamma$",    "#1f77b4"),
    ("qi",              r"QI geometry (grid + exact $\gamma$)",     "#2ca02c"),
]
FLOOR = 5e-14


def median(vals):
    vals = sorted(v for v in vals if v == v)
    return vals[len(vals) // 2] if vals else float("nan")


def main():
    if not SUMMARY_CSV.exists():
        print(f"No summary at {SUMMARY_CSV}"); sys.exit(1)
    rows = list(csv.DictReader(open(SUMMARY_CSV)))
    targets = [t for t in TARGET_ORDER if any(r["target"] == t for r in rows)]
    Ns = sorted({int(r["resolution"]) for r in rows})
    present_inits = [(k, lab, c) for k, lab, c in INITS
                     if any(r["geom_init"] == k for r in rows)]

    ncol = 3
    nrow = math.ceil(len(targets) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 4.2 * nrow), squeeze=False)

    def val(target, init, N):
        return median([float(r["vpgn_best_eval_rel_l2"]) for r in rows
                       if r["target"] == target and r["geom_init"] == init
                       and int(r["resolution"]) == N])

    for idx, target in enumerate(targets):
        ax = axes[idx // ncol][idx % ncol]
        for init, label, color in present_inits:
            ys = [val(target, init, N) for N in Ns]
            ax.loglog(Ns, ys, "-o", color=color, ms=5, label=label)
        ax.axhline(FLOOR, ls="--", color="k", lw=1.3, label="fp64 floor")
        ax.set_title(target, fontsize=12)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel(r"$N$   (width $W = N + 2\,$halo$+1$)")
        ax.set_ylabel(r"eval relative $L_2$")
        ax.set_xticks(Ns)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())

    for j in range(len(targets), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.suptitle("expD04 init slice -- VarPro + Adam→Gauss-Newton (readout projected out)\n"
                 r"eval relative $L_2$ vs width, by geometry initialization",
                 fontsize=13, y=0.99, va="top")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95),
               ncol=len(labels), fontsize=9, frameon=True, borderaxespad=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}  ({len(targets)} targets, inits={[k for k,_,_ in present_inits]})")


if __name__ == "__main__":
    main()
