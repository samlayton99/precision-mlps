"""Overlay our VarPro + Adam->Gauss-Newton (QI geometry) onto expD02's real
init_slice_adam_both figure.

Reproduces expD02's initialization slice for the adam_both regime (4 inits +
QI+lstsq construction floor, read from cube.json) and overlays ONE new line:
our VarPro+Adam->GN with QI geometry (readout projected out), read from
varpro_gn_summary.csv. The point: our method sits at the fp64 floor while every
Adam-both init stalls.

Caveat printed on the figure: expD02 is float32 / lambda*=0.25; ours is
fp64 / lambda=0.30. Shared x-axis is N (widths align); actual W differs slightly.

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
OURS = D04 / "varpro_gn_summary.csv"
FREE_ADAM = D04 / "free_qi_adam_summary.csv"       # free QI init + pure Adam (subset of funcs)
FULLNET_GN = D04 / "adam_gn_fullnet_summary.csv"   # full-net Adam->GN, NO lstsq (subset of funcs)
OUT = D04 / "figures" / "init_slice_adam_overlay.png"

# expD02 inits: (key, label, color)  -- kept faithful to plot_slices.py
INITS = [
    ("xavier",        r"Xavier ($\gamma\approx0.1$)",  "#9467bd"),
    ("scaled_xavier", "scaled Xavier",                 "#1f77b4"),
    ("const_weight",  "const weight, covered centers", "#ff7f0e"),
    ("qi",            "QI init",                       "#2ca02c"),
]
# the two expD02 regimes to overlay onto: (cube key, human label, out-file stem)
REGIMES = [
    ("adam_both",        "Adam-both (fully trained)",            "init_slice_adam_both_overlay"),
    ("adam_first_lstsq", "first-layer Adam + lstsq readout",     "init_slice_adam_first_lstsq_overlay"),
]
BASELINE_LABEL = "QI + lstsq (construction floor)"
OURS_LABEL = "VarPro + Adam→GN, QI geom (lstsq readout) [fp64]  ← ours"
OURS_COLOR = "#d62728"
FREE_ADAM_LABEL = "free QI geom + pure Adam (GD) [fp64]"
FREE_ADAM_COLOR = "#8c564b"
FULLNET_LABEL = "Adam→GN, full net, QI geom (NO lstsq) [fp64]"
FULLNET_COLOR = "#17becf"


def cube_val(rows, init, fn, N, key):
    return next(r[key] for r in rows if r["init"] == init and r["function"] == fn and r["N"] == N)


def median(vals):
    vals = sorted(v for v in vals if v == v)
    return vals[len(vals) // 2] if vals else float("nan")


def main():
    if not CUBE.exists():
        print(f"No cube at {CUBE}"); sys.exit(1)
    if not OURS.exists():
        print(f"No VarPro summary at {OURS}"); sys.exit(1)
    data = json.load(open(CUBE))
    rows, widths = data["rows"], data["config"]["widths_n"]
    functions = data["config"]["functions"]
    ours = list(csv.DictReader(open(OURS)))
    free_adam = list(csv.DictReader(open(FREE_ADAM))) if FREE_ADAM.exists() else []
    fullnet = list(csv.DictReader(open(FULLNET_GN))) if FULLNET_GN.exists() else []

    def our_qi(fn, N):
        return median([float(r["vpgn_best_eval_rel_l2"]) for r in ours
                       if r["target"] == fn and r["geom_init"] == "qi"
                       and int(r["resolution"]) == N])

    def free_qi_adam(fn, N):
        return median([float(r["adam_best_eval_rel_l2"]) for r in free_adam
                       if r["target"] == fn and int(r["resolution"]) == N])

    def fullnet_adamgn(fn, N):
        return median([float(r["adamgn_best_eval_rel_l2"]) for r in fullnet
                       if r["target"] == fn and int(r["resolution"]) == N])

    for regime_key, regime_label, out_stem in REGIMES:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
        for ax, fn in zip(axes.ravel(), functions):
            for init, label, color in INITS:
                ys = [cube_val(rows, init, fn, N, regime_key) for N in widths]
                ax.loglog(widths, ys, "-o", color=color, ms=5, label=f"{label} [{regime_label}]")
            base = [cube_val(rows, "xavier", fn, N, "qi_lstsq") for N in widths]
            ax.loglog(widths, base, "--", color="k", lw=1.6, label=BASELINE_LABEL)
            # free QI geometry + pure Adam (GD): only where we have it (runge, sine_mixture)
            if any(r["target"] == fn for r in free_adam):
                fy = [free_qi_adam(fn, N) for N in widths]
                ax.loglog(widths, fy, "-s", color=FREE_ADAM_COLOR, ms=7, lw=2.2,
                          label=FREE_ADAM_LABEL, zorder=5)
            # full-net Adam->GN, NO lstsq: only where we have it (runge, sine_mixture)
            if any(r["target"] == fn for r in fullnet):
                gy = [fullnet_adamgn(fn, N) for N in widths]
                ax.loglog(widths, gy, "-v", color=FULLNET_COLOR, ms=8, lw=2.4,
                          label=FULLNET_LABEL, zorder=5)
            # our overlay -- bold with star markers so it reads as the hero line
            ours_y = [our_qi(fn, N) for N in widths]
            ax.loglog(widths, ours_y, "-*", color=OURS_COLOR, ms=13, lw=2.6,
                      label=OURS_LABEL, zorder=6)
            ax.set_title(fn, fontsize=12)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_xlabel(r"$N$   (width $W = N + 2\,$halo$+1$)")
            ax.set_ylabel(r"eval relative $L_2$")
            ax.set_xticks(widths)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())

        # collect legend entries across all panels (free-Adam line is only on some)
        seen, handles, labels = set(), [], []
        for ax in axes.ravel():
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen.add(l); handles.append(h); labels.append(l)
        fig.tight_layout(rect=[0, 0, 1, 0.86])
        fig.suptitle(
            f"init slice ({regime_label}): VarPro + Adam→Gauss-Newton vs expD02 inits\n"
            "expD02 lines = float32, $\\lambda^*{=}0.25$;  our overlays = fp64, $\\lambda{=}0.30$  "
            "(shared axis $N$; $W$ differs slightly)",
            fontsize=13, y=0.99, va="top")
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
                   ncol=3, fontsize=9.5, frameon=True, borderaxespad=0)
        out = OUT.parent / f"{out_stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
