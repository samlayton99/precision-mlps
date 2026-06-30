"""expD02 barrier view -- one at-a-glance composite figure per function.

Collapses the 24-panel slice plots into a per-function story that maps directly
onto the checkpoint-D bullets. For each of the 6 functions, one PNG with:

  TOP  -- three init x regime heatmaps (N = 32, 128, 512), color = log10 rel L2
          (cell text = the value). Rows are inits worst->best (xavier ->
          scaled_xavier -> const_weight -> qi); columns are regimes. The jump
          from column 0 (adam_both, learned readout) to column 1
          (adam_first_lstsq) IS the readout barrier; the right column (qi_lstsq)
          is the init-independent floor (constant down each column = sanity).
  BOTTOM -- the scaling companion: rel L2 vs N (loglog), the few treatment lines
          that carry the narrative -- the floor rides to eps while gamma-only and
          the Adam ceiling plateau.

Usage:
    python experiments/expD02_adam_geometry/plot_barrier_view.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD02_adam_geometry"
CUBE_PATH = RESULTS_DIR / "cube.json"
OUT_DIR = RESULTS_DIR / "barrier_view"

# rows worst->best (top->bottom); columns left->right as in the cube.
INITS = ["xavier", "scaled_xavier", "const_weight", "qi"]
INIT_LABEL = {"xavier": "xavier\n(gamma~0.1)", "scaled_xavier": "scaled_xavier\n(gamma* fixed,\nrandom centers)",
              "const_weight": "const_weight\n(gamma*, covering\ncenters)", "qi": "qi\n(ideal geometry)"}
REGIMES = ["adam_both", "adam_first_lstsq", "frozen_init_lstsq", "qi_lstsq"]
REGIME_LABEL = {"adam_both": "adam_both\n(learned readout)",
                "adam_first_lstsq": "adam geom\n+ lstsq",
                "frozen_init_lstsq": "frozen init\n+ lstsq",
                "qi_lstsq": "qi geom\n+ lstsq (floor)"}
HEAT_WIDTHS = [32, 128, 512]
VMIN, VMAX = -14.0, 0.0


def cube_lookup(rows):
    """(init, function, N, regime) -> value."""
    out = {}
    for r in rows:
        for rg in REGIMES:
            out[(r["init"], r["function"], r["N"], rg)] = r[rg]
    return out


def draw_heatmap(ax, lut, fn, N, show_ylabels):
    """One init x regime heatmap for (fn, N). log10 color, value-annotated."""
    M = np.array([[np.log10(max(lut[(i, fn, N, rg)], 1e-16)) for rg in REGIMES]
                  for i in INITS])
    im = ax.imshow(M, cmap="viridis_r", vmin=VMIN, vmax=VMAX, aspect="auto")
    ax.set_xticks(range(len(REGIMES)))
    ax.set_xticklabels([REGIME_LABEL[rg] for rg in REGIMES], fontsize=7, rotation=0)
    if show_ylabels:
        ax.set_yticks(range(len(INITS)))
        ax.set_yticklabels([INIT_LABEL[i] for i in INITS], fontsize=7)
    else:
        ax.set_yticks([])
    ax.set_title(f"N = {N}", fontsize=10)
    for r in range(len(INITS)):
        for c in range(len(REGIMES)):
            val = M[r, c]
            ax.text(c, r, f"{val:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if val < (VMIN + VMAX) / 2 else "black")
    # readout barrier separator between col 0 and col 1
    ax.axvline(0.5, color="#d62728", lw=2.0)
    return im


def draw_scaling(ax, lut, fn, widths):
    """rel L2 vs N: the narrative lines for one function."""
    lines = [
        (("qi", "qi_lstsq"),            "ideal geom + lstsq (floor)", "#2ca02c", "-o"),
        (("scaled_xavier", "frozen_init_lstsq"), "gamma-only geom + lstsq", "#1f77b4", "-s"),
        (("xavier", "adam_first_lstsq"), "Adam geom (from xavier) + lstsq", "#ff7f0e", "-^"),
    ]
    for (init, rg), label, color, style in lines:
        ys = [lut[(init, fn, N, rg)] for N in widths]
        ax.loglog(widths, ys, style, color=color, ms=5, label=label)
    # Adam ceiling: best adam_both over all inits (the end-to-end gradient ceiling)
    ys = [min(lut[(i, fn, N, "adam_both")] for i in INITS) for N in widths]
    ax.loglog(widths, ys, "-D", color="#d62728", ms=5, label="best adam_both (gradient ceiling)")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlabel(r"$N$   (width $W = N + 2\,$halo$+1$)")
    ax.set_ylabel(r"relative $L_2$")
    ax.set_xticks(widths)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              fontsize=8, borderaxespad=0, frameon=False)


def make_figure(lut, fn, widths):
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.55, wspace=0.08)
    fig.suptitle(f"expD02 barrier view -- {fn}\n"
                 r"log$_{10}$ relative $L_2$; rows = init worst$\to$best, "
                 r"cols = training regime; red line = readout barrier "
                 r"(learned readout $\to$ lstsq)", fontsize=12, y=0.99)
    im = None
    for j, N in enumerate(HEAT_WIDTHS):
        ax = fig.add_subplot(gs[0, j])
        im = draw_heatmap(ax, lut, fn, N, show_ylabels=(j == 0))
    cbar = fig.colorbar(im, ax=fig.axes[:len(HEAT_WIDTHS)], shrink=0.8,
                        pad=0.015, location="right")
    cbar.set_label(r"log$_{10}$ rel $L_2$  (darker = better)", fontsize=8)
    ax_line = fig.add_subplot(gs[1, :])
    draw_scaling(ax_line, lut, fn, widths)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"barrier_{fn}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    if not CUBE_PATH.exists():
        print(f"No cube at {CUBE_PATH}; run build_cube.py first.")
        sys.exit(1)
    data = json.load(open(CUBE_PATH))
    lut = cube_lookup(data["rows"])
    widths = data["config"]["widths_n"]
    for fn in data["config"]["functions"]:
        out = make_figure(lut, fn, widths)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
