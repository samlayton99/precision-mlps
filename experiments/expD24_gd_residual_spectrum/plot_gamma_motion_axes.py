"""Replot the frozen-control figure's gamma motion without rerunning training."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test"
TARGETS = (
    ("sine", "Sine"),
    ("sine_mixture", "Mixed sine"),
    ("runge", "Runge"),
    ("gaussian_envelope", "Gaussian envelope\n(earlier whole-line case)"),
)
GAMMAS = (1, 4, 16, 64)


def main():
    with np.load(RESULTS / "data/controls.npz", allow_pickle=False) as saved:
        curves = {
            (target, gamma, field): saved[f"{target}_{gamma}__{field}"]
            for target, _ in TARGETS
            for gamma in GAMMAS
            for field in ("net_motion", "path_motion")
        }
    steps = np.arange(len(next(iter(curves.values()))))
    for values in curves.values():
        assert values.shape == steps.shape
        assert np.all(np.isfinite(values)) and np.all(values >= 0)
        assert np.all(values[:2] == 0) and np.all(values[2:] > 0)

    upper = 1.08 * max(values.max() for values in curves.values())
    lower = 10 ** np.floor(np.log10(min(values[2:].min() for values in curves.values())))
    colors = plt.cm.viridis(np.linspace(.1, .9, len(GAMMAS)))
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 14), dpi=170, sharex="col", sharey="col")
    for row, (target, label) in enumerate(TARGETS):
        for gamma, color in zip(GAMMAS, colors):
            for field, style in (("net_motion", "-"), ("path_motion", "--")):
                values = curves[target, gamma, field]
                axes[row, 0].plot(steps, values, style, color=color, lw=1.8)
                # Strict logarithmic axes cannot display the two zero-motion states.
                positive = (steps > 0) & (values > 0)
                axes[row, 1].plot(steps[positive], values[positive], style, color=color, lw=1.8)
        axes[row, 0].set(xlim=(0, steps[-1]), ylim=(0, upper),
                         ylabel=f"{label}\nMean absolute γ movement")
        axes[row, 0].ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
        axes[row, 1].set(xscale="log", yscale="log", xlim=(2, steps[-1]), ylim=(lower, upper),
                         ylabel="Mean absolute γ movement")
        for ax in axes[row]:
            ax.grid(alpha=.22, which="major")
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("GD step")
    axes[0, 0].set_title("Linear–linear: absolute size of the movement", fontsize=12, pad=20)
    axes[0, 1].set_title("Log–log: early motion and relative growth", fontsize=12, pad=20)
    fig.suptitle("Gamma movement during joint GD — the same trajectories on two scales", fontsize=17, y=.983)
    fig.legend([Line2D([], [], color=color, lw=2.5) for color in colors],
               [f"Initial γ = {gamma}" for gamma in GAMMAS],
               loc="upper center", bbox_to_anchor=(.5, .958), ncol=4, frameon=False)
    fig.legend([Line2D([], [], color="black", lw=1.8, ls=style) for style in ("-", "--")],
               ["Net displacement from initialization", "Total travel (sum of absolute step changes)"],
               loc="upper center", bbox_to_anchor=(.5, .935), ncol=2, frameon=False)
    fig.subplots_adjust(top=.872, bottom=.095, left=.11, right=.975, hspace=.44, wspace=.27)
    fig.text(.5, .028,
             "Mean over all 177 neurons, including halo; γ = |a|. Shared axis limits within each column.\n"
             "Steps 0 and 1 have zero movement and appear only on linear axes. Zero readout delays geometry motion until step 2.\n"
             "Saved 2,000-step runs, GD rate 0.002. First three functions: [−1, 1]; Gaussian: earlier whole-line objective. No retraining.",
             ha="center", va="center", fontsize=9.5, linespacing=1.6)
    path = RESULTS / "gamma_motion_axes.png"
    fig.savefig(path)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
