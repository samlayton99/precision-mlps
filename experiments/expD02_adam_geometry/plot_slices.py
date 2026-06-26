"""Plot the expD02 cube two ways (reads results/.../cube.json; makes no data).

training_slice/      -- one PNG per INITIALIZATION; lines = the 4 training regimes.
                        (training_slice_<init>.png; the xavier PNG is the original
                        adam_geometry.png with 6 functions instead of 4.)
initialization_slice/-- one PNG per TRAINING REGIME (the 3 that vary by init);
                        lines = the 4 initializations, plus the QI-construction
                        (QI init + lstsq) baseline overlaid. No 4th PNG: regime 4
                        (QI + lstsq) is init-independent, so it is the baseline line.

Each PNG is a 2x3 grid (one subplot per function), x = N (width W = N+2*halo+1),
y = eval relative L2, both log. Legends are placed above the axes.

Usage: python experiments/expD02_adam_geometry/plot_slices.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD02_adam_geometry"
CUBE_PATH = RESULTS_DIR / "cube.json"

# regimes: (column key, label, color)
REGIMES = [
    ("adam_both",         "fully trained (Adam, both layers)", "#d62728"),
    ("adam_first_lstsq",  "first layer Adam + lstsq readout",  "#ff7f0e"),
    ("frozen_init_lstsq", "frozen init geometry + lstsq",      "#1f77b4"),
    ("qi_lstsq",          "QI geometry + lstsq (floor)",       "#2ca02c"),
]
# the 3 regimes that vary by init (regime 4 = qi_lstsq is the baseline, not a panel)
INIT_SLICE_REGIMES = [
    ("adam_both",         "fully trained (Adam, both layers)"),
    ("adam_first_lstsq",  "first layer Adam + lstsq readout"),
    ("frozen_init_lstsq", "frozen init geometry + lstsq"),
]
# inits: (key, label, color)
INITS = [
    ("xavier",        r"Xavier ($\gamma\approx0.1$)",  "#9467bd"),
    ("scaled_xavier", "scaled Xavier (raw centers)",   "#1f77b4"),
    ("const_weight",  "const weight, covered centers", "#ff7f0e"),
    ("qi",            "QI init",                       "#2ca02c"),
]
BASELINE_LABEL = "QI + lstsq (construction)"

INIT_TITLE = {
    "xavier": r"Xavier init ($\gamma\approx 0.1$)",
    "scaled_xavier": r"scaled-Xavier init (raw $-b/w$ centers, $\mathrm{mean}|w|=\gamma^*$)",
    "const_weight": r"const-weight init ($w=\gamma^*$, Xavier centers covering the span)",
    "qi": r"QI init ($w=\gamma^*$, $b=-\gamma^* c_{\rm unif}$) + QI readout",
}


def _val(rows, init, fn, N, key):
    return next(r[key] for r in rows if r["init"] == init and r["function"] == fn and r["N"] == N)


def _setup_axes(ax, fn, widths):
    ax.set_title(fn, fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlabel(r"$N$   (width $W = N + 2\,$halo$+1$ neurons)")
    ax.set_ylabel(r"relative $L_2$")
    ax.set_xticks(widths)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())


def _finish_fig(fig, axes, title, ncol):
    """Pack the subplots with tight_layout, reserving the top band for the title
    and a single shared legend (above the axes, not overlapping them)."""
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.suptitle(title, fontsize=13, y=0.99, va="top")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=ncol, fontsize=9, frameon=True, borderaxespad=0)


def plot_training_slice(data, out_dir):
    """One PNG per init; lines = the 4 training regimes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, widths = data["rows"], data["config"]["widths_n"]
    functions = data["config"]["functions"]
    for init, _, _ in INITS:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
        for ax, fn in zip(axes.ravel(), functions):
            for key, label, color in REGIMES:
                ys = [_val(rows, init, fn, N, key) for N in widths]
                ax.loglog(widths, ys, "-o", color=color, ms=5, label=label)
            _setup_axes(ax, fn, widths)
        title = (f"expD02 training slice -- {INIT_TITLE[init]}\n"
                 r"relative $L_2$ vs width, by training regime ($\lambda^*=0.25$, float32 Adam)")
        _finish_fig(fig, axes, title, ncol=4)
        out = out_dir / f"training_slice_{init}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


def plot_initialization_slice(data, out_dir):
    """One PNG per training regime (the 3 that vary by init); lines = the 4 inits,
    plus the QI-construction (QI init + lstsq) baseline overlaid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, widths = data["rows"], data["config"]["widths_n"]
    functions = data["config"]["functions"]
    for regime_key, regime_label in INIT_SLICE_REGIMES:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
        for ax, fn in zip(axes.ravel(), functions):
            for init, label, color in INITS:
                ys = [_val(rows, init, fn, N, regime_key) for N in widths]
                ax.loglog(widths, ys, "-o", color=color, ms=5, label=label)
            # QI-construction baseline (QI init + lstsq); init-independent.
            base = [_val(rows, "xavier", fn, N, "qi_lstsq") for N in widths]
            ax.loglog(widths, base, "--", color="k", lw=1.6, label=BASELINE_LABEL)
            _setup_axes(ax, fn, widths)
        title = (f"expD02 initialization slice -- regime: {regime_label}\n"
                 r"relative $L_2$ vs width, by initialization ($\lambda^*=0.25$, float32 Adam)")
        _finish_fig(fig, axes, title, ncol=5)
        out = out_dir / f"init_slice_{regime_key}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    if not CUBE_PATH.exists():
        print(f"No cube at {CUBE_PATH}. Run build_cube.py first.")
        sys.exit(1)
    with open(CUBE_PATH) as f:
        data = json.load(f)
    plot_training_slice(data, RESULTS_DIR / "training_slice")
    plot_initialization_slice(data, RESULTS_DIR / "initialization_slice")
