"""Render the activation comparison from completed, saved experiment data."""
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
ACTIVATION_NAMES = {
    "tanh": "Tanh · standard",
    "notch": "Integrated notch · negative control",
    "local": "Localized variational · optimized candidate",
    "sinc": "Integrated sinc · idealized reference",
}
ACTIVATION_SHORT = {"tanh": "Tanh", "notch": "Integrated\nnotch", "local": "Localized\nvariational", "sinc": "Integrated\nsinc"}
TARGET_NAMES = {"sine": "Sine", "runge": "Runge", "sine_mixture": "Sine mixture"}
COLORS = {"qi": "#bc3038", "xavier": "#286fb0"}
TARGET_COLORS = {"sine": "#2378a6", "runge": "#bd5c1c", "sine_mixture": "#7753a3"}
ACTIVATION_COLORS = {"tanh": "#286fb0", "notch": "#bc3038", "local": "#29865a", "sinc": "#7753a3"}
DISPLAY_MIN = 1e-16
SERIES = (
    ("qi", "actual_rel_l2", "Constructed QI · actual readout", "-"),
    ("qi", "floor_rel_l2", "Constructed QI · least-squares readout", "--"),
    ("xavier", "actual_rel_l2", "Xavier · actual readout", "-"),
    ("xavier", "floor_rel_l2", "Xavier · least-squares readout", "--"),
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _activations(config: dict, *, include_supplementary=False) -> tuple[str, ...]:
    main = tuple(config["activations"])
    if include_supplementary:
        return main + tuple(config.get("supplementary_activations", ()))
    return main


def _load(out_path: Path):
    config = _read_json(out_path / "config.json")
    bandwidth = _read_json(out_path / "bandwidth.json")
    activations = _activations(config, include_supplementary=True)
    if len(_activations(config)) != 3:
        raise ValueError("The main comparison requires exactly three activations")
    if len(set(activations)) != len(activations):
        raise ValueError("Main and supplementary activation lists must be disjoint and unique")
    runs = {}
    steps = int(config["steps"])
    for path in sorted((out_path / "runs").glob("*.json")):
        run = _read_json(path)
        if run.get("complete") is not True:
            raise ValueError(f"Run is not marked complete: {path}")
        key = (run["activation"], run["target"], run["initialization"], int(run["seed"]))
        if key in runs:
            raise ValueError(f"Duplicate run {key}: {path}")
        history = run["history"]
        recorded_steps = np.array([row["step"] for row in history], dtype=int)
        if not len(history) or recorded_steps[0] != 0 or recorded_steps[-1] != steps:
            raise ValueError(f"Expected a complete step-0 to step-{steps} history: {path}")
        if np.any(np.diff(recorded_steps) <= 0):
            raise ValueError(f"History steps must increase strictly: {path}")
        for field in ("actual_rel_l2", "floor_rel_l2"):
            values = np.array([row[field] for row in history], dtype=float)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Invalid {field} values: {path}")
        runs[key] = run
    seeds = tuple(config.get("xavier_seeds", (0, 1, 2)))
    if 0 not in seeds:
        raise ValueError("Seed 0 is required for the representative trajectory panels")
    for activation in activations:
        for target in TARGETS:
            for initialization, selected_seeds in (("qi", (0,)), ("xavier", seeds)):
                for seed in selected_seeds:
                    key = (activation, target, initialization, int(seed))
                    if key not in runs:
                        raise ValueError(f"Missing completed run: {key}")
    for row in bandwidth["rows"]:
        for field in ("construction_rel_l2", "floor_rel_l2"):
            if not np.isfinite(row[field]) or row[field] < 0:
                raise ValueError(f"Invalid bandwidth {field}: {row}")
    return config, bandwidth, runs


def _display(values):
    return np.maximum(np.asarray(values, dtype=float), DISPLAY_MIN)


def _upper_limit(values) -> float:
    maximum = max(1.0, float(np.max(values)))
    return 10.0 ** np.ceil(np.log10(maximum * 1.2))


def _log_axis(ax, upper: float):
    ax.set_yscale("log")
    ax.set_ylim(DISPLAY_MIN, upper)
    top_power = int(np.ceil(np.log10(upper)))
    stride = max(2, int(np.ceil((top_power + 16) / 9)))
    powers = np.arange(-16, top_power + 1, stride)
    ax.yaxis.set_major_locator(FixedLocator(10.0 ** powers))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(axis="y", which="major", color="#d9dde2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _step_label(value, _):
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def _save(fig, directory: Path, name: str):
    for extension in ("png", "pdf"):
        path = directory / f"{name}.{extension}"
        fig.savefig(path, dpi=190, facecolor="white")
        print(f"Saved {path}", flush=True)
    plt.close(fig)


def _method_legend():
    return [
        Line2D([], [], color=COLORS[initialization], lw=2, linestyle=style, label=label)
        for initialization, _, label, style in SERIES
    ]


def _setup_caption(config: dict) -> str:
    centers = config["interior_centers"]
    halo = config["halo_per_side"]
    return (
        f"{centers} interior centers + {halo} halo centers per side · FP64 training / LS · "
        f"{config['n_train']:,} training / {config['n_eval']:,} evaluation points"
    )


def _trajectories(directory, config, bandwidth, runs, upper, *, early=False):
    horizon = int(config["steps"])
    fig, axes = plt.subplots(3, 3, figsize=(17.2, 12.0), sharex=True, sharey=True)
    for row, target in enumerate(TARGETS):
        for col, activation in enumerate(_activations(config)):
            ax = axes[row, col]
            # Draw the dashed diagnostics last so coincident actual/LS curves remain visible.
            for initialization, field, _, style in SERIES:
                history = runs[(activation, target, initialization, 0)]["history"]
                step = np.array([item["step"] for item in history])
                values = _display([item[field] for item in history])
                actual = field == "actual_rel_l2"
                ax.plot(step, values, color=COLORS[initialization], ls=style,
                        lw=1.8 if actual else 1.4, alpha=0.95,
                        zorder=3 if actual else 4)
                ax.plot(step[0], values[0], marker="o" if actual else "x", ms=5.5,
                        color=COLORS[initialization], mfc="white", mew=1.2,
                        clip_on=False, zorder=6)
            _log_axis(ax, upper)
            ax.set_xlim(0, horizon)
            if early:
                ax.set_xscale("symlog", linthresh=1.0, linscale=0.75)
                ticks = [0, 1, 10, 100, 1000, horizon]
            else:
                ticks = np.linspace(0, horizon, 6)
            ax.xaxis.set_major_locator(FixedLocator(sorted(set(ticks))))
            ax.xaxis.set_major_formatter(FuncFormatter(_step_label))
            if row == 0:
                chosen = bandwidth["selected"][activation]["lambda"]
                ax.set_title(f"{ACTIVATION_NAMES[activation]}\n" + rf"Selected $\lambda={chosen:g}$",
                             fontsize=12.5, pad=14, linespacing=1.5)
            if col == 0:
                ax.set_ylabel(f"{TARGET_NAMES[target]}\nRelative L2 error", fontsize=12, labelpad=9)
            if row == 2:
                ax.set_xlabel("Adam step" + (" (symmetric log scale)" if early else ""), fontsize=11)
    fig.suptitle("Constructed QI readout versus full Xavier initialization", fontsize=21, y=0.976)
    fig.text(0.5, 0.944,
             f"All parameters trainable · {horizon:,} Adam steps · seed 0 for both initializations"
             + (" · expanded early steps" if early else " · linear step axis"),
             ha="center", fontsize=12)
    fig.legend(handles=_method_legend(), loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=4, frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.081, right=0.985, bottom=0.135, top=0.82, wspace=0.12, hspace=0.21)
    fig.text(0.5, 0.076,
             "Dashed curves refit only the readout by least squares at the current hidden-layer geometry; "
             "the refit is diagnostic, not a training update.", ha="center", fontsize=10.5)
    fig.text(0.5, 0.048,
             _setup_caption(config) + "\n"
             r"Open circles / crosses mark step 0. Errors below $10^{-16}$ are displayed at $10^{-16}$.",
             ha="center", va="center", fontsize=10, linespacing=1.65)
    _save(fig, directory, "trajectories_early" if early else "trajectories")


def _reference_trajectories(directory, config, bandwidth, runs, upper, activation):
    horizon = int(config["steps"])
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.9), sharex=True, sharey=True)
    for col, target in enumerate(TARGETS):
        ax = axes[col]
        for initialization, field, _, style in SERIES:
            history = runs[(activation, target, initialization, 0)]["history"]
            step = np.array([item["step"] for item in history])
            values = _display([item[field] for item in history])
            actual = field == "actual_rel_l2"
            ax.plot(step, values, color=COLORS[initialization], ls=style,
                    lw=1.8 if actual else 1.4, alpha=0.95, zorder=3 if actual else 4)
            ax.plot(step[0], values[0], marker="o" if actual else "x", ms=5.5,
                    color=COLORS[initialization], mfc="white", mew=1.2,
                    clip_on=False, zorder=6)
        _log_axis(ax, upper)
        ax.set_xlim(0, horizon)
        ax.xaxis.set_major_locator(FixedLocator(np.linspace(0, horizon, 6)))
        ax.xaxis.set_major_formatter(FuncFormatter(_step_label))
        ax.set_title(TARGET_NAMES[target], fontsize=14, pad=12)
        ax.set_xlabel("Adam step", fontsize=11)
        if col == 0:
            ax.set_ylabel("Relative L2 error", fontsize=12)
    chosen = bandwidth["selected"][activation]["lambda"]
    fig.suptitle(ACTIVATION_NAMES[activation] + " · supplementary reference", fontsize=20, y=0.975)
    fig.text(0.5, 0.919,
             rf"Selected $\lambda={chosen:g}$ · "
             f"all parameters trainable · {horizon:,} Adam steps · seed 0 · same error limits as the main comparison",
             ha="center", fontsize=11.5)
    fig.legend(handles=_method_legend(), loc="upper center", bbox_to_anchor=(0.5, 0.885),
               ncol=4, frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.215, top=0.76, wspace=0.13)
    fig.text(0.5, 0.108,
             "Dashed curves refit only the readout by least squares at the current hidden-layer geometry; "
             "the refit is diagnostic, not a training update.", ha="center", fontsize=10.5)
    fig.text(0.5, 0.056,
             _setup_caption(config) + "\n"
             r"Open circles / crosses mark step 0. Errors below $10^{-16}$ are displayed at $10^{-16}$.",
             ha="center", va="center", fontsize=10, linespacing=1.65)
    _save(fig, directory, f"{activation}_reference")


def _endpoint_bars(directory, config, runs, upper):
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.9), sharey=True)
    seeds = tuple(config.get("xavier_seeds", (0, 1, 2)))
    width = 0.18
    activations = _activations(config)
    group_x = np.arange(len(activations))
    for col, target in enumerate(TARGETS):
        ax = axes[col]
        for series_index, (initialization, field, _, _) in enumerate(SERIES):
            position = group_x + (series_index - 1.5) * width
            selected_seeds = (0,) if initialization == "qi" else seeds
            endpoints = np.array([
                [runs[(activation, target, initialization, int(seed))]["history"][-1][field]
                 for seed in selected_seeds]
                for activation in activations
            ])
            medians = _display(np.median(endpoints, axis=1))
            is_floor = field == "floor_rel_l2"
            color = COLORS[initialization]
            ax.bar(position, medians - DISPLAY_MIN, bottom=DISPLAY_MIN, width=width * 0.9,
                   color=to_rgba(color, 0.24 if is_floor else 0.85), edgecolor=color,
                   hatch="///" if is_floor else None, linewidth=1.1, zorder=3)
            # Caps keep endpoints at the plotting minimum visible even for zero-height bars.
            ax.scatter(position, medians, marker="_", s=85, linewidths=1.5,
                       color=color, zorder=5, clip_on=False)
            if initialization == "xavier":
                jitter = np.linspace(-0.035, 0.035, len(selected_seeds))
                for seed_index in range(len(selected_seeds)):
                    ax.scatter(position + jitter[seed_index], _display(endpoints[:, seed_index]),
                               s=23, facecolors="white", edgecolors=color, linewidths=1.1,
                               marker="o", zorder=6, clip_on=False)
        _log_axis(ax, upper)
        ax.set_xticks(group_x, [ACTIVATION_SHORT[activation] for activation in activations], fontsize=11)
        ax.set_xlim(-0.55, len(activations) - 0.45)
        ax.set_title(TARGET_NAMES[target], fontsize=14, pad=12)
        if col == 0:
            ax.set_ylabel("Final relative L2 error", fontsize=12)
    handles = [
        Patch(facecolor=to_rgba(COLORS[initialization], 0.24 if field == "floor_rel_l2" else 0.85),
              edgecolor=COLORS[initialization], hatch="///" if field == "floor_rel_l2" else None,
              label=label)
        for initialization, field, label, _ in SERIES
    ]
    handles.append(Line2D([], [], ls="none", marker="o", mfc="white", mec=COLORS["xavier"],
                          label="Individual Xavier seeds"))
    fig.suptitle(f"Error at the final step ({int(config['steps']):,})", fontsize=21, y=0.97)
    fig.text(0.5, 0.915,
             f"One constructed QI start · Xavier bars show the median of {len(seeds)} seeds · all activations use the same axes",
             ha="center", fontsize=12)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.88), ncol=5,
               frameon=False, fontsize=10.5)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.76, wspace=0.13)
    fig.text(0.5, 0.105,
             "Hatched bars: least-squares refit of the readout at the final geometry. "
             "Unhatched bars: the trained network at that same step.", ha="center", fontsize=10.5)
    fig.text(0.5, 0.062,
             r"Bars start at $10^{-16}$ on the logarithmic axis; values below this display limit are clipped. "
             "Bar tops, not bar areas, encode error.", ha="center", fontsize=10.5)
    _save(fig, directory, "endpoint_bars")


def _bandwidth_sweep(directory, config, bandwidth):
    activations = _activations(config, include_supplementary=True)
    ncols = 2 if len(activations) > 3 else len(activations)
    nrows = int(np.ceil(len(activations) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17.2, 11.8 if nrows > 1 else 7.0),
                             sharey=True, squeeze=False)
    rows = bandwidth["rows"]
    upper = _upper_limit([row[field] for row in rows
                          for field in ("construction_rel_l2", "floor_rel_l2")])
    for index, activation in enumerate(activations):
        ax = axes.flat[index]
        chosen = float(bandwidth["selected"][activation]["lambda"])
        reference = float(config["lambda_references"][activation])
        all_lambdas = []
        for target in TARGETS:
            subset = sorted((row for row in rows if row["activation"] == activation and row["target"] == target),
                            key=lambda item: item["lambda"])
            if not subset:
                raise ValueError(f"No bandwidth rows for {(activation, target)}")
            lambdas = np.array([row["lambda"] for row in subset])
            all_lambdas.extend(lambdas)
            color = TARGET_COLORS[target]
            for field, style in (("construction_rel_l2", "-"), ("floor_rel_l2", "--")):
                values = _display([row[field] for row in subset])
                ax.plot(lambdas, values, color=color, ls=style, lw=1.8, marker=".", ms=4)
                selected_index = np.flatnonzero(np.isclose(lambdas, chosen, rtol=1e-10, atol=1e-12))
                if selected_index.size != 1:
                    raise ValueError(f"Selected lambda missing or duplicated in bandwidth rows: {(activation, target)}")
                ax.scatter(lambdas[selected_index], values[selected_index], s=36, marker="o",
                           facecolors="white", edgecolors=color, linewidths=1.4, zorder=5, clip_on=False)
        ax.axvline(reference, color="#6d737b", lw=1.1, ls=":", zorder=1)
        ax.axvline(chosen, color="#30343b", lw=1.1, ls="-.", zorder=2)
        if activation == "local" and "local_scaled_gram" in bandwidth:
            bound = config["local_design"]["minimum_gram_ratio"]
            infeasible = [float(value) for value, gram in bandwidth["local_scaled_gram"].items()
                          if gram["minimum_over_energy"] < bound]
            ax.plot(infeasible, [0.965] * len(infeasible), linestyle="none", marker="x",
                    color="#777e87", ms=6, mew=1.3, transform=ax.get_xaxis_transform(), zorder=6)
        _log_axis(ax, upper)
        ax.set_xscale("log")
        ax.set_xlim(min(all_lambdas) / 1.06, max(all_lambdas) * 1.06)
        ticks = [value for value in (0.1, 0.25, 0.5, 1.0, 2.0) if min(all_lambdas) <= value <= max(all_lambdas)]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.set_xlabel(r"Bandwidth $\lambda=\gamma h$", fontsize=12)
        ax.set_title(ACTIVATION_NAMES[activation] + "\n" +
                     rf"Selected $\lambda={chosen:g}$ · reference $\lambda={reference:g}$",
                     fontsize=12.5, pad=13, linespacing=1.5)
        if index % ncols == 0:
            ax.set_ylabel("Relative L2 error", fontsize=12)
    for index in range(len(activations), axes.size):
        axes.flat[index].set_visible(False)
    handles = [Line2D([], [], color=TARGET_COLORS[target], lw=2, label=TARGET_NAMES[target]) for target in TARGETS]
    handles.extend([
        Line2D([], [], color="#343a43", lw=1.8, label="QI construction"),
        Line2D([], [], color="#343a43", lw=1.8, ls="--", label="Least-squares readout"),
        Line2D([], [], color="#343a43", lw=1.1, ls="-.", marker="o", mfc="white", label="Selected bandwidth"),
        Line2D([], [], color="#6d737b", lw=1.1, ls=":", label="Reference bandwidth"),
    ])
    if "local_scaled_gram" in bandwidth:
        handles.append(Line2D([], [], color="#777e87", ls="none", marker="x", ms=6,
                              label="Local: violates Gram bound"))
    fig.suptitle("Bandwidth selection before training", fontsize=21, y=0.975)
    fig.text(0.5, 0.944 if nrows > 1 else 0.923,
             "Float64 screening; selected tanh construction refined at 30 digits",
             ha="center", fontsize=12)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.915 if nrows > 1 else 0.892), ncol=4,
               frameon=False, fontsize=10.5)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13 if nrows > 1 else 0.185,
                        top=0.81 if nrows > 1 else 0.69, wspace=0.12, hspace=0.40)
    fig.text(0.5, 0.076 if nrows > 1 else 0.115,
             "One bandwidth per activation minimizes worst-target construction error. "
             "Local bandwidth selection retains the design’s derivative-Gram bound.",
             ha="center", fontsize=10.5)
    fig.text(0.5, 0.052 if nrows > 1 else 0.077,
             "Solid curves use the explicit QI readout. Dashed curves refit the readout at the same initial geometry; "
             "they do not determine the selected bandwidth.", ha="center", fontsize=10.5)
    fig.text(0.5, 0.027 if nrows > 1 else 0.039,
             r"Values below $10^{-16}$ are displayed at $10^{-16}$. "
             "Reference bandwidths are comparison markers, not additional optimization results.",
             ha="center", fontsize=10.5)
    _save(fig, directory, "bandwidth_sweep")


def _cutoff_sensitivity(directory, config, validation):
    """Compare saved final-geometry LS refits without rerunning any experiment."""
    activations = _activations(config, include_supplementary=True)
    seeds = tuple(config.get("xavier_seeds", (0, 1, 2)))
    cutoffs = np.array([1e-14, 1e-13, 1e-12])
    default = float(config["readout_rcond"])
    indexed = {}
    for row in validation["rows"]:
        target, activation, initialization, seed_text = row["run"].split("__")
        key = (activation, target, initialization, int(seed_text.removeprefix("s")))
        if key in indexed:
            raise ValueError(f"Duplicate validation run: {key}")
        values = {float(cutoff): result["relative_l2"] for cutoff, result in row["cutoffs"].items()}
        array = np.array([values[cutoff] for cutoff in cutoffs], dtype=float)
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f"Invalid cutoff validation values: {key}")
        indexed[key] = array
    curves = {}
    for activation in activations:
        for target in TARGETS:
            curves[(activation, target, "qi")] = indexed[(activation, target, "qi", 0)]
            curves[(activation, target, "xavier")] = np.median(
                [indexed[(activation, target, "xavier", int(seed))] for seed in seeds], axis=0)
    all_values = _display(np.concatenate(list(curves.values())))
    lower = max(DISPLAY_MIN, 10.0 ** np.floor(np.log10(all_values.min() / 1.2)))
    upper = 10.0 ** np.ceil(np.log10(all_values.max() * 1.2))
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 7.3), sharey=True)
    for col, target in enumerate(TARGETS):
        ax = axes[col]
        for activation in activations:
            for initialization, style, marker in (("qi", "-", "o"), ("xavier", "--", "D")):
                ax.plot(cutoffs, _display(curves[(activation, target, initialization)]),
                        color=ACTIVATION_COLORS[activation], ls=style, marker=marker,
                        mfc="white", ms=5, lw=1.8, zorder=3)
        ax.axvline(default, color="#777e87", lw=1.1, ls=":", zorder=1)
        _log_axis(ax, upper)
        ax.set_ylim(lower, upper)
        ax.set_xscale("log")
        ax.set_xlim(cutoffs[0] / 1.2, cutoffs[-1] * 1.2)
        ax.xaxis.set_major_locator(FixedLocator(cutoffs))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.set_xlabel("Relative singular-value cutoff (rcond)", fontsize=11)
        ax.set_title(TARGET_NAMES[target], fontsize=14, pad=12)
        if col == 0:
            ax.set_ylabel("Final dense-grid LS relative L2 error", fontsize=12)
    handles = [Line2D([], [], color=ACTIVATION_COLORS[activation], lw=2,
                      label=ACTIVATION_SHORT[activation].replace("\n", " "))
               for activation in activations]
    handles.extend([
        Line2D([], [], color="#343a43", ls="-", marker="o", mfc="white", lw=1.8, label="QI"),
        Line2D([], [], color="#343a43", ls="--", marker="D", mfc="white", lw=1.8,
               label=f"Xavier median ({len(seeds)} seeds)"),
        Line2D([], [], color="#777e87", ls=":", lw=1.1, label="Default cutoff"),
    ])
    fig.suptitle("Sensitivity of the final least-squares readout to its cutoff", fontsize=21, y=0.975)
    fig.text(0.5, 0.919,
             f"Final geometry after {int(config['steps']):,} Adam steps · "
             f"{int(validation['grid_size']):,}-point evaluation · default rcond = {default:g}",
             ha="center", fontsize=12)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.887), ncol=4,
               frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.225, top=0.71, wspace=0.13)
    fig.text(0.5, 0.116,
             "Each curve refits the readout at the same saved final geometry; only the singular-value cutoff changes.",
             ha="center", fontsize=10.5)
    fig.text(0.5, 0.076,
             "These are numerical least-squares diagnostics, not universal or certified approximation floors.",
             ha="center", fontsize=10.5)
    if np.any(np.concatenate(list(curves.values())) < DISPLAY_MIN):
        fig.text(0.5, 0.039, r"Errors below $10^{-16}$ are displayed at $10^{-16}$.",
                 ha="center", fontsize=10.5)
    _save(fig, directory, "cutoff_sensitivity")


def render(out_path: Path):
    """Render main comparisons and supplementary references into ``out_path/figures``."""
    out_path = Path(out_path)
    config, bandwidth, runs = _load(out_path)
    directory = out_path / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    upper = _upper_limit([row[field] for run in runs.values() for row in run["history"]
                          for field in ("actual_rel_l2", "floor_rel_l2")])
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.titleweight": "regular", "hatch.linewidth": 0.65}):
        _trajectories(directory, config, bandwidth, runs, upper)
        _trajectories(directory, config, bandwidth, runs, upper, early=True)
        _endpoint_bars(directory, config, runs, upper)
        _bandwidth_sweep(directory, config, bandwidth)
        for activation in config.get("supplementary_activations", ()):
            _reference_trajectories(directory, config, bandwidth, runs, upper, activation)
        validation_path = out_path / "validation.json"
        if validation_path.exists():
            _cutoff_sensitivity(directory, config, _read_json(validation_path))
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("results/checkpoint_D_optimizers/expD41_activation_lens"))
    args = parser.parse_args()
    render(args.out)
