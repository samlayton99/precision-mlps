"""Overlay our optimizer results onto expD02's init_slice figures.

Objective: compare Adam to standard second-order optimizers, in the two regimes
the expD02 init-slice plots use. Each regime plot shows ONLY methods that match
that regime, so the comparison is apples-to-apples:

  init_slice_adam_both_overlay        -- regime: init + pure gradient descent,
      both layers, NO least squares. Our lines: free QI geom + pure Adam,
      full-net Adam->GN, full-net LBFGS.

  init_slice_adam_first_lstsq_overlay -- regime: gradient descent on the
      geometry + a least-squares readout on top. Our lines: VarPro (Adam->GN +
      lstsq readout), LBFGS + lstsq readout.

expD02 lines are float32 / lambda*=0.25; ours are fp64 / lambda=0.30. Shared
x-axis is N (widths align); actual W differs slightly. All our lines start from
the free QI geometry (qi_geom_random_readout -- readout NOT handed over).

Usage: python experiments/expD04_varpro/overlay_init_slice_adam.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
D02 = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD02_adam_geometry"
D04 = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD04_varpro"
CUBE = D02 / "cube.json"
FIGDIR = D04 / "figures"

# expD02 inits: (key, label, color) -- faithful to plot_slices.py
INITS = [
    ("xavier",        r"Xavier ($\gamma\approx0.1$)",  "#9467bd"),
    ("scaled_xavier", "scaled Xavier",                 "#1f77b4"),
    ("const_weight",  "const weight, covered centers", "#ff7f0e"),
    ("qi",            "QI init",                       "#2ca02c"),
]
BASELINE_LABEL = "QI + lstsq (construction floor)"

# our result files -> (path, value column, optional row filter)
SOURCES = {
    "free_adam":  (D04 / "free_qi_adam_summary.csv",    "adam_best_eval_rel_l2",   lambda r: True),
    "fullnet_gn": (D04 / "adam_gn_fullnet_summary.csv", "adamgn_best_eval_rel_l2", lambda r: True),
    "varpro_gn":  (D04 / "varpro_gn_summary.csv",       "vpgn_best_eval_rel_l2",   lambda r: r["geom_init"] == "qi"),
    "lbfgs_both": (D04 / "second_order_lbfgs_summary.csv",    "best_eval_rel_l2",
                   lambda r: r["regime"] == "both"),
    "lbfgs_lstsq": (D04 / "second_order_lbfgs_summary.csv",   "best_eval_rel_l2",
                    lambda r: r["regime"] == "lstsq"),
    "ssb_both": (D04 / "second_order_ssbroyden_summary.csv",  "best_eval_rel_l2",
                 lambda r: r["regime"] == "both"),
    "ssb_lstsq": (D04 / "second_order_ssbroyden_summary.csv", "best_eval_rel_l2",
                  lambda r: r["regime"] == "lstsq"),
}

# which of our lines belong on which regime figure
OUR_LINES = {
    "adam_both": [
        ("free_adam",  "free QI geom + pure Adam (GD) [fp64]",        "#8c564b", "s", 7,  2.2),
        ("lbfgs_both", "LBFGS, full net [fp64]",                      "#e377c2", "P", 9,  2.2),
        ("ssb_both",   "SSBroyden, full net [fp64]",                  "#bcbd22", "D", 7,  2.2),
        ("fullnet_gn", "Adam→GN, full net [fp64]",                    "#17becf", "v", 8,  2.4),
    ],
    "adam_first_lstsq": [
        ("lbfgs_lstsq", "LBFGS + lstsq readout [fp64]",              "#e377c2", "P", 9, 2.2),
        ("ssb_lstsq",   "SSBroyden + lstsq readout [fp64]",         "#bcbd22", "D", 7, 2.2),
        ("varpro_gn",   "Adam→GN + lstsq readout [fp64] ← ours",    "#d62728", "*", 13, 2.6),
    ],
}

REGIMES = [
    ("adam_both",        "init + pure gradient descent (both layers, no lstsq)",
     "init_slice_adam_both_overlay"),
    ("adam_first_lstsq", "gradient descent on geometry + least-squares readout",
     "init_slice_adam_first_lstsq_overlay"),
]


def cube_val(rows, init, fn, N, key):
    return next(r[key] for r in rows if r["init"] == init and r["function"] == fn and r["N"] == N)


def median(vals):
    vals = sorted(v for v in vals if v == v)
    return vals[len(vals) // 2] if vals else float("nan")


def load_sources():
    out = {}
    for name, (path, col, filt) in SOURCES.items():
        if not path.exists():
            out[name] = {}
            continue
        agg = {}
        for r in csv.DictReader(open(path)):
            if not filt(r):
                continue
            agg.setdefault((r["target"], int(r["resolution"])), []).append(float(r[col]))
        out[name] = agg
    return out


def main():
    if not CUBE.exists():
        print(f"No cube at {CUBE}"); sys.exit(1)
    data = json.load(open(CUBE))
    rows, widths = data["rows"], data["config"]["widths_n"]
    functions = data["config"]["functions"]
    src = load_sources()

    for regime_key, regime_label, out_stem in REGIMES:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
        for ax, fn in zip(axes.ravel(), functions):
            for init, label, color in INITS:
                ys = [cube_val(rows, init, fn, N, regime_key) for N in widths]
                ax.loglog(widths, ys, "-o", color=color, ms=5, label=f"{label} [expD02 Adam]")
            base = [cube_val(rows, "xavier", fn, N, "qi_lstsq") for N in widths]
            ax.loglog(widths, base, "--", color="k", lw=1.6, label=BASELINE_LABEL)
            for name, label, color, marker, ms, lw in OUR_LINES[regime_key]:
                agg = src.get(name, {})
                if not any(k[0] == fn for k in agg):
                    continue
                ys = [median(agg.get((fn, N), [])) for N in widths]
                ax.loglog(widths, ys, f"-{marker}", color=color, ms=ms, lw=lw,
                          label=label, zorder=5)
            ax.set_title(fn, fontsize=12)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_xlabel(r"$N$   (width $W = N + 2\,$halo$+1$)")
            ax.set_ylabel(r"eval relative $L_2$")
            ax.set_xticks(widths)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())

        seen, handles, labels = set(), [], []
        for ax in axes.ravel():
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen.add(l); handles.append(h); labels.append(l)
        fig.tight_layout(rect=[0, 0, 1, 0.86])
        fig.suptitle(
            f"init slice -- {regime_label}\n"
            "Adam vs second-order optimizers from the free QI geometry.  "
            "expD02 lines = float32, $\\lambda^*{=}0.25$;  ours = fp64, $\\lambda{=}0.30$ "
            "(shared axis $N$; $W$ differs slightly)",
            fontsize=13, y=0.99, va="top")
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
                   ncol=3, fontsize=9.5, frameon=True, borderaxespad=0)
        FIGDIR.mkdir(parents=True, exist_ok=True)
        out = FIGDIR / f"{out_stem}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
