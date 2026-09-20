"""Plot loss and applied-update ratios from saved 10k trajectories only."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/checkpoint_D_optimizers/expD31_split_adam/long_run"
TARGETS = {"sine": "Sine", "sine_mixture": "Mixed sine", "runge": "Runge",
           "gaussian_envelope": "Gaussian envelope"}
MUS = (50, 100, 250, 500)
SCHEDULES = ("constant", "cosine")


def main():
    cases = {}
    for schedule in SCHEDULES:
        for target in TARGETS:
            for mu in MUS:
                path = RESULTS / "data" / f"{target}__{mu}__{schedule}.npz"
                with np.load(path) as data:
                    c = {k: data[k] for k in ("step", "L", "F", "G", "F_step_norm", "G_step_norm")}
                np.testing.assert_array_equal(c["step"], np.arange(10001))
                np.testing.assert_allclose(c["F"] + c["G"], c["L"], rtol=1e-13, atol=1e-15)
                for k in ("F", "G", "F_step_norm", "G_step_norm"):
                    assert np.all(np.isfinite(c[k]) & (c[k] > 0)), (path, k)
                c["loss_ratio"] = c["F"] / c["G"]
                # The final state has a proposed direction but no applied update.
                c["update_ratio"] = c["F_step_norm"][:-1] / c["G_step_norm"][:-1]
                cases[schedule, target, mu] = c

    limits = {}
    for key in ("loss_ratio", "update_ratio"):
        lo = min(float(c[key].min()) for c in cases.values())
        hi = max(float(c[key].max()) for c in cases.values())
        limits[key] = (10.0 ** np.floor(np.log10(min(lo, 1))),
                       10.0 ** np.ceil(np.log10(max(hi, 1))))
    colors = dict(zip(MUS, plt.cm.viridis(np.linspace(.05, .9, len(MUS)))))
    handles = [Line2D([], [], color=colors[mu], lw=2, label=f"μ = {mu}") for mu in MUS]
    handles.append(Line2D([], [], color=".3", ls="--", lw=1.2, label="Ratio = 1: equal sizes"))
    for schedule in SCHEDULES:
        fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), dpi=180, sharex=True, sharey="row")
        for col, (target, label) in enumerate(TARGETS.items()):
            axes[0, col].set_title(label, fontsize=16, pad=14)
            for mu in MUS:
                c = cases[schedule, target, mu]
                axes[0, col].plot(c["step"], c["loss_ratio"], color=colors[mu], lw=1.15)
                axes[1, col].plot(c["step"][:-1], c["update_ratio"], color=colors[mu], lw=1.15)
            for row, key in enumerate(("loss_ratio", "update_ratio")):
                ax = axes[row, col]
                ax.axhline(1, color=".3", ls="--", lw=1.2)
                ax.set_yscale("log")
                ax.set_ylim(*limits[key])
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
                ax.set_xlim(0, 10000)
                ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
                ax.grid(alpha=.17)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(labelsize=10, labelleft=True)
            axes[1, col].set_xlabel("Training updates", fontsize=12)
        axes[0, 0].set_ylabel(r"$F_\tau\,/\,G_\tau$" + "\nLoss ratio", fontsize=16, labelpad=14)
        axes[1, 0].set_ylabel(r"$\|\eta_t\mu u_F\|_2\,/\,\|\eta_tu_G\|_2$"
                              + "\nUpdate-norm ratio", fontsize=15, labelpad=14)
        rate = ("Constant learning rate: 0.002" if schedule == "constant" else
                "Cosine decay over all 10,000 updates: 0.002 → 0.000002")
        fig.suptitle(r"Xavier · original VarPro $J_*$ split · balance of F and G" + "\n" + rate,
                     fontsize=19, y=.985)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .895),
                   ncol=5, frameon=False, fontsize=12)
        fig.subplots_adjust(left=.095, right=.985, top=.775, bottom=.18, hspace=.29, wspace=.26)
        fig.text(.5, .066,
                 "Top: numerical refitted loss Fτ divided by the remaining loss Gτ = L − Fτ. No μ weighting in this row.\n"
                 "Bottom: magnitudes of the two geometry-update contributions, after separate Adam histories and outside μ scaling.\n"
                 "Above 1: F contribution larger; below 1: G contribution larger. Norm ratios do not measure direction or cancellation.\n"
                 "Saved trajectories only; no training repeated. All 10,000 applied updates shown; corresponding axes match across both figures.",
                 ha="center", va="center", fontsize=11, linespacing=1.5)
        path = RESULTS / "figures" / f"balance_{schedule}.png"
        fig.savefig(path)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
