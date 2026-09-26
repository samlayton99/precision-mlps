"""One compact two-panel note figure, using saved arrays only."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism"
DEST = SOURCE / "compact_note"


def run():
    ratio_data = json.loads((SOURCE / "direct_ratio_interval/data.json").read_text())
    time_data = json.loads((SOURCE / "note_interval_figures/data.json").read_text())
    DEST.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14,
                         "axes.titlesize": 14, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "legend.fontsize": 13})
    fig, axes = plt.subplots(1, 2, figsize=(11., 3.8))
    colors = plt.colormaps["viridis"]([.25, .72])
    gammas = np.array([r["gamma"] for r in ratio_data["records"]])
    for rank, color in zip([26, 33], colors):
        rows = [next(item for item in row["ranks"] if item["rank"] == rank)
                for row in ratio_data["records"]]
        low, actual, high = [np.array([row[key] for row in rows])
                             for key in ["lower", "actual", "upper"]]
        axes[0].fill_between(gammas, low, high, color=color, alpha=.15)
        for values, style in [(low, "--"), (actual, "-"), (high, ":")]:
            axes[0].plot(gammas, values, style, color=color, lw=1.8)
    rows = time_data["records"]
    g = np.array([r["gamma"] for r in rows])
    low, actual, high = [np.array([r["counts"][key] for r in rows])
                         for key in ["necessary", "actual", "sufficient"]]
    axes[1].fill_between(g, low, high, color="0.5", alpha=.18)
    for values, style in [(low, "--"), (actual, "-"), (high, ":")]:
        axes[1].plot(g, values, style, color="0.15", lw=1.8)
    axes[1].scatter(g, actual, color="0.15", s=25, zorder=3)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([4, 8, 16, 32, 64], ["4", "8", "16", "32", "64"])
        ax.set_xlim(3.85, 67.)
        ax.set_xlabel(r"Slope $\gamma$")
        ax.grid(alpha=.18)
    axes[0].set_ylim(1e-16, 3e-3)
    axes[0].set_title("Finite eigenvalue ratios", pad=11)
    axes[0].set_ylabel(r"$\rho_i=\lambda_i/\lambda_1$")
    axes[1].set_ylim(1e4, 1e13)
    axes[1].set_title("Steps to 1% residual", pad=11)
    axes[1].set_ylabel("Steps")
    handles = [Line2D([0], [0], color=color, lw=1.8, label=f"Rank {rank}")
               for rank, color in zip([26, 33], colors)]
    handles += [Line2D([0], [0], color="0.15", ls=style, lw=1.8, label=label)
                for style, label in [("-", "Actual"), ("--", "Lower bound"), (":", "Upper bound")]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=5, frameon=False, columnspacing=1.35)
    fig.subplots_adjust(left=.10, right=.985, bottom=.195, top=.76, wspace=.30)
    fig.savefig(DEST / "gamma_ratios_and_steps.png", dpi=240)
    plt.close(fig)
    (DEST / "MANIFEST.md").write_text(
        "# Compact gamma note figure\n\n"
        "`gamma_ratios_and_steps.png` is the only figure produced. It replots saved arrays; no eigensolve or training is run.\n\n"
        "- Left: `../direct_ratio_interval/data.json`, ranks 26 and 33 over the saved gamma sweep; original two-sided ratio intervals.\n"
        "- Right: `../note_interval_figures/data.json`, gamma 4, 8, 16, 32, 64; necessary, actual spectral, and sufficient counts to 1% relative training residual with eta=0.5/lambda_1.\n\n"
        "The left panel uses the original selected-rank evaluations; the right panel uses the full residual calculation with its checked numerical guard and conservative unresolved-mass treatment documented in `../note_interval_figures/MANIFEST.md`. Neither panel is an interval-certified numerical calculation.\n")
    print(DEST / "gamma_ratios_and_steps.png")


if __name__ == "__main__":
    run()
