"""Plot readout-only training with the saved QI hidden geometry held fixed."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "expd41-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext
import numpy as np


TARGETS = ("sine", "runge", "sine_mixture")
TARGET_NAMES = {"sine": "Sine", "runge": "Runge", "sine_mixture": "Sine mixture"}
ACTIVATION_NAMES = {
    "tanh": "Tanh · standard",
    "notch": "Integrated notch · negative control",
    "local": "Localized variational · optimized candidate",
    "sinc": "Integrated sinc · idealized reference",
}
ACTIVATION_SHORT = {
    "tanh": "Tanh", "notch": "Integrated\nnotch",
    "local": "Localized\nvariational", "sinc": "Integrated\nsinc",
}
COLORS = {"zero_readout": "#bc3038", "xavier_readout": "#286fb0", "least_squares": "#252b33"}
DISPLAY_MIN = 1e-16


def _read_json(path):
    return json.loads(Path(path).read_text())


def _all_activations(config):
    return tuple(config["activations"]) + tuple(config.get("supplementary_activations", ()))


def _load(out_path):
    config = _read_json(out_path / "config.json")
    bandwidth = _read_json(out_path / "bandwidth.json")
    activations = _all_activations(config)
    if len(config["activations"]) != 3:
        raise ValueError("The main frozen-geometry comparison requires exactly three activations")
    if len(set(activations)) != len(activations):
        raise ValueError("Activation lists must be unique and disjoint")
    seeds = tuple(config.get("xavier_seeds", (0, 1, 2)))
    if 0 not in seeds:
        raise ValueError("Xavier readout seed 0 is required for the trajectory figures")
    horizon = int(config["steps"])
    runs = {}
    for path in sorted((out_path / "runs").glob("*.json")):
        run = _read_json(path)
        if run.get("complete") is not True:
            raise ValueError(f"Run is not marked complete: {path}")
        key = (run["activation"], run["target"], run["initialization"], int(run["seed"]))
        if key in runs:
            raise ValueError(f"Duplicate frozen-geometry run: {key}")
        history = run["history"]
        steps = np.array([row["step"] for row in history], dtype=int)
        if not len(history) or steps[0] != 0 or steps[-1] != horizon:
            raise ValueError(f"Expected complete step-0 to step-{horizon} history: {path}")
        if np.any(np.diff(steps) <= 0):
            raise ValueError(f"History steps must increase strictly: {path}")
        for field in ("actual_rel_l2", "floor_rel_l2"):
            values = np.array([row[field] for row in history], dtype=float)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Invalid {field} values: {path}")
        runs[key] = run
    for activation in activations:
        for target in TARGETS:
            for initialization, selected_seeds in (("zero_readout", (0,)), ("xavier_readout", seeds)):
                for seed in selected_seeds:
                    key = (activation, target, initialization, int(seed))
                    if key not in runs:
                        raise ValueError(f"Missing completed frozen-geometry run: {key}")
            reference = runs[(activation, target, "zero_readout", 0)]["history"][0]["floor_rel_l2"]
            for initialization, selected_seeds in (("zero_readout", (0,)), ("xavier_readout", seeds)):
                for seed in selected_seeds:
                    key = (activation, target, initialization, int(seed))
                    errors = np.array([row["floor_rel_l2"] for row in runs[key]["history"]])
                    if not np.allclose(errors, reference, rtol=1e-10, atol=0):
                        raise ValueError(f"Least-squares reference is not shared and constant at fixed geometry: {key}")
    return config, bandwidth, runs


def _display(values):
    return np.maximum(np.asarray(values, dtype=float), DISPLAY_MIN)


def _upper_limit(runs):
    values = [row[field] for run in runs.values() for row in run["history"]
              for field in ("actual_rel_l2", "floor_rel_l2")]
    return 10.0 ** np.ceil(np.log10(max(1.0, max(values)) * 1.2))


def _log_axis(ax, upper):
    ax.set_yscale("log")
    ax.set_ylim(DISPLAY_MIN, upper)
    top_power = int(np.ceil(np.log10(upper)))
    stride = max(2, int(np.ceil((top_power + 16) / 9)))
    ax.yaxis.set_major_locator(FixedLocator(10.0 ** np.arange(-16, top_power + 1, stride)))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(axis="y", which="major", color="#d9dde2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _step_label(value, _):
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def _save(fig, directory, name):
    for extension in ("png", "pdf"):
        path = directory / f"{name}.{extension}"
        fig.savefig(path, dpi=190, facecolor="white")
        print(f"Saved {path}", flush=True)
    plt.close(fig)


def _trajectory_legend():
    return [
        Line2D([], [], color=COLORS["zero_readout"], lw=2, label="Zero readout · QI geometry"),
        Line2D([], [], color=COLORS["xavier_readout"], lw=2, label="Xavier readout · QI geometry · seed 0"),
        Line2D([], [], color=COLORS["least_squares"], lw=1.6, ls="--", label="Shared least-squares readout"),
    ]


def _setup_caption(config):
    return (f"{config['interior_centers']} interior centers + {config['halo_per_side']} halo centers per side · "
            f"FP64 training / LS · {config['n_train']:,} training / {config['n_eval']:,} evaluation points")


def _plot_trajectory(ax, activation, target, config, runs, upper, *, early=False):
    horizon = int(config["steps"])
    reference = runs[(activation, target, "zero_readout", 0)]["history"][0]["floor_rel_l2"]
    ax.axhline(max(reference, DISPLAY_MIN), color=COLORS["least_squares"], ls="--", lw=1.6, zorder=2)
    for initialization in ("zero_readout", "xavier_readout"):
        history = runs[(activation, target, initialization, 0)]["history"]
        steps = np.array([row["step"] for row in history])
        values = _display([row["actual_rel_l2"] for row in history])
        ax.plot(steps, values, color=COLORS[initialization], lw=1.8, zorder=3)
        ax.plot(steps[0], values[0], marker="o", color=COLORS[initialization], mfc="white",
                ms=5.5, mew=1.2, clip_on=False, zorder=5)
    _log_axis(ax, upper)
    ax.set_xlim(0, horizon)
    if early:
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.75)
        ticks = sorted(set((0, 1, 10, 100, 1000, horizon)))
    else:
        ticks = np.linspace(0, horizon, 6)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(_step_label))


def _trajectories(directory, config, bandwidth, runs, upper, *, early=False):
    fig, axes = plt.subplots(3, 3, figsize=(17.2, 12.0), sharex=True, sharey=True)
    for row, target in enumerate(TARGETS):
        for col, activation in enumerate(config["activations"]):
            ax = axes[row, col]
            _plot_trajectory(ax, activation, target, config, runs, upper, early=early)
            if row == 0:
                chosen = bandwidth["selected"][activation]["lambda"]
                ax.set_title(ACTIVATION_NAMES[activation] + "\n" + rf"Selected $\lambda={chosen:g}$",
                             fontsize=12.5, pad=14, linespacing=1.5)
            if col == 0:
                ax.set_ylabel(f"{TARGET_NAMES[target]}\nRelative L2 error", fontsize=12, labelpad=9)
            if row == 2:
                ax.set_xlabel("Adam step" + (" (symmetric log scale)" if early else ""), fontsize=11)
    fig.suptitle("Readout-only training with fixed QI hidden geometry", fontsize=21, y=0.976)
    fig.text(0.5, 0.944,
             f"Only output weights and bias train · {int(config['steps']):,} Adam steps · "
             + ("expanded early steps" if early else "linear step axis"), ha="center", fontsize=12)
    fig.legend(handles=_trajectory_legend(), loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.081, right=0.985, bottom=0.15, top=0.82, wspace=0.12, hspace=0.21)
    fig.text(0.5, 0.095,
             "Red starts with zero output weights and bias. Blue randomizes only the output weights. Both use the same fixed QI hidden geometry.",
             ha="center", fontsize=10.5)
    fig.text(0.5, 0.073,
             "The dashed line is the numerical least-squares reference for that fixed geometry; it is not a training update.",
             ha="center", fontsize=10.5)
    fig.text(0.5, 0.04, _setup_caption(config) + "\n"
             r"Open circles mark step 0. Errors below $10^{-16}$ are displayed at $10^{-16}$.",
             ha="center", va="center", fontsize=10, linespacing=1.65)
    _save(fig, directory, "frozen_early" if early else "frozen_trajectories")


def _endpoints(directory, config, runs, upper):
    activations = tuple(config["activations"])
    seeds = tuple(config.get("xavier_seeds", (0, 1, 2)))
    group_x = np.arange(len(activations))
    width = 0.23
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 7.0), sharey=True)
    for col, target in enumerate(TARGETS):
        ax = axes[col]
        series = [
            np.array([runs[(activation, target, "zero_readout", 0)]["history"][-1]["actual_rel_l2"]
                      for activation in activations]),
            np.array([[runs[(activation, target, "xavier_readout", int(seed))]["history"][-1]["actual_rel_l2"]
                       for seed in seeds] for activation in activations]),
            np.array([runs[(activation, target, "zero_readout", 0)]["history"][-1]["floor_rel_l2"]
                      for activation in activations]),
        ]
        for index, (initialization, values) in enumerate(zip(("zero_readout", "xavier_readout", "least_squares"), series)):
            position = group_x + (index - 1) * width
            endpoints = _display(np.median(values, axis=1) if values.ndim == 2 else values)
            reference = initialization == "least_squares"
            color = COLORS[initialization]
            ax.bar(position, endpoints - DISPLAY_MIN, bottom=DISPLAY_MIN, width=width * 0.9,
                   color=to_rgba(color, 0.18 if reference else 0.85), edgecolor=color,
                   hatch="///" if reference else None, linewidth=1.1, zorder=3)
            ax.scatter(position, endpoints, marker="_", s=85, color=color,
                       linewidths=1.5, clip_on=False, zorder=5)
            if values.ndim == 2:
                for seed_index, jitter in enumerate(np.linspace(-0.04, 0.04, len(seeds))):
                    ax.scatter(position + jitter, _display(values[:, seed_index]), s=24, marker="o",
                               facecolors="white", edgecolors=color, linewidths=1.1, clip_on=False, zorder=6)
        _log_axis(ax, upper)
        ax.set_xticks(group_x, [ACTIVATION_SHORT[activation] for activation in activations], fontsize=11)
        ax.set_xlim(-0.55, len(activations) - 0.45)
        ax.set_title(TARGET_NAMES[target], fontsize=14, pad=12)
        if col == 0:
            ax.set_ylabel("Final relative L2 error", fontsize=12)
    handles = [
        Patch(facecolor=to_rgba(COLORS["zero_readout"], 0.85), edgecolor=COLORS["zero_readout"], label="Zero readout · QI geometry"),
        Patch(facecolor=to_rgba(COLORS["xavier_readout"], 0.85), edgecolor=COLORS["xavier_readout"],
              label=f"Xavier readout · median of {len(seeds)} seeds"),
        Patch(facecolor=to_rgba(COLORS["least_squares"], 0.18), edgecolor=COLORS["least_squares"],
              hatch="///", label="Shared least-squares readout"),
        Line2D([], [], ls="none", marker="o", mfc="white", mec=COLORS["xavier_readout"], label="Individual Xavier readout seeds"),
    ]
    fig.suptitle(f"Readout-only error at the final step ({int(config['steps']):,})", fontsize=21, y=0.975)
    fig.text(0.5, 0.92, "Every bar uses the same saved QI hidden geometry for its activation and target", ha="center", fontsize=12)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.885), ncol=4,
               frameon=False, fontsize=10.5)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.23, top=0.76, wspace=0.13)
    fig.text(0.5, 0.11,
             "Zero readout sets output weights and bias to zero. Xavier randomizes only output weights; both use fixed QI hidden geometry.",
             ha="center", fontsize=10.5)
    fig.text(0.5, 0.069,
             r"Bars start at $10^{-16}$ on the logarithmic axis; smaller errors are displayed at that limit. "
             "Bar tops encode error.", ha="center", fontsize=10.5)
    _save(fig, directory, "frozen_endpoints")


def _reference(directory, config, bandwidth, runs, upper, activation):
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 7.1), sharex=True, sharey=True)
    for col, target in enumerate(TARGETS):
        ax = axes[col]
        _plot_trajectory(ax, activation, target, config, runs, upper)
        ax.set_title(TARGET_NAMES[target], fontsize=14, pad=12)
        ax.set_xlabel("Adam step", fontsize=11)
        if col == 0:
            ax.set_ylabel("Relative L2 error", fontsize=12)
    chosen = bandwidth["selected"][activation]["lambda"]
    fig.suptitle(ACTIVATION_NAMES[activation] + " · fixed hidden geometry", fontsize=20, y=0.975)
    fig.text(0.5, 0.921,
             rf"Selected $\lambda={chosen:g}$ · "
             f"readout-only training · {int(config['steps']):,} Adam steps · same error limits as the main comparison",
             ha="center", fontsize=11.5)
    fig.legend(handles=_trajectory_legend(), loc="upper center", bbox_to_anchor=(0.5, 0.89),
               ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.245, top=0.77, wspace=0.13)
    fig.text(0.5, 0.136,
             "Red starts with zero output weights and bias. Blue randomizes only output weights; both keep the saved QI hidden geometry fixed.",
             ha="center", fontsize=10.5)
    fig.text(0.5, 0.103,
             "The dashed line is the shared numerical least-squares reference; it is not a training update.",
             ha="center", fontsize=10.5)
    fig.text(0.5, 0.053, _setup_caption(config) + "\n"
             r"Open circles mark step 0. Errors below $10^{-16}$ are displayed at $10^{-16}$.",
             ha="center", va="center", fontsize=10, linespacing=1.65)
    _save(fig, directory, f"frozen_{activation}_reference")


def render(out):
    """Require completed frozen runs and save the requested PNG/PDF figure pairs."""
    out = Path(out)
    config, bandwidth, runs = _load(out)
    directory = out / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    upper = _upper_limit(runs)
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.titleweight": "regular", "hatch.linewidth": 0.65}):
        _trajectories(directory, config, bandwidth, runs, upper)
        _trajectories(directory, config, bandwidth, runs, upper, early=True)
        _endpoints(directory, config, runs, upper)
        for activation in config.get("supplementary_activations", ()):
            _reference(directory, config, bandwidth, runs, upper, activation)
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("results/checkpoint_D_optimizers/expD41_activation_lens/frozen_geometry"))
    args = parser.parse_args()
    render(args.out)
