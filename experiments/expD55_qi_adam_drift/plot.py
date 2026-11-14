"""Adam trajectories with a log-to-linear step axis and explicit display smoothing."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/precisionmlps-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC11_true_precision_law import paper_style
from experiments.expD55_qi_adam_drift.run import OUT

COLORS = {1e-2: "#0072B2", 1e-3: "#E69F00", 1e-4: "#D55E00", 1e-5: "#8B5FBF", 1e-6: "#009E73", 1e-7: "#CC79A7"}
TITLES = [
    r"a) $x^2$",
    r"b) $\sin(2\pi x)$",
    r"c) $\sin(2\pi x)+\frac{1}{2}\sin(6\pi x)+\frac{1}{4}\sin(14\pi x)$",
]
SMOOTHING = {
    "method": "Centered arithmetic mean of log10(relative L2), exponentiated (geometric mean of errors)",
    "window_span_steps": 100,
    "sample_offsets_steps": [-50, -25, 0, 25, 50],
    "early_steps": "Steps 0 through 100 shown raw to preserve initialization and onset",
    "edges": "Truncate the window at the final step; no extrapolation",
    "statistics": "All numerical summaries use raw errors, not displayed smoothed values",
}

AXIS_TRANSITION = 1000.
AXIS_END = 10000.
LOG_PANEL_FRACTION = .25
LOG_SLOPE = 1. / np.log(10.)
AXIS_JOIN = np.log10(AXIS_TRANSITION) + LOG_SLOPE


def step_axis_forward(values):
    """Map steps 0..1000 into the first quarter, then 1000..10000 linearly."""
    x = np.asarray(values, dtype=float)
    early = np.where(x <= 1., LOG_SLOPE * x,
                     np.log10(np.maximum(x, 1.)) + LOG_SLOPE)
    return np.where(x <= AXIS_TRANSITION, LOG_PANEL_FRACTION * early / AXIS_JOIN,
                    LOG_PANEL_FRACTION + (1. - LOG_PANEL_FRACTION) *
                    (x - AXIS_TRANSITION) / (AXIS_END - AXIS_TRANSITION))


def step_axis_inverse(values):
    z = np.asarray(values, dtype=float)
    early_z = z * AXIS_JOIN / LOG_PANEL_FRACTION
    early = np.where(early_z <= LOG_SLOPE, early_z / LOG_SLOPE,
                     10. ** (np.minimum(early_z, AXIS_JOIN) - LOG_SLOPE))
    return np.where(z <= LOG_PANEL_FRACTION, early,
                    AXIS_TRANSITION + (z - LOG_PANEL_FRACTION) *
                    (AXIS_END - AXIS_TRANSITION) / (1. - LOG_PANEL_FRACTION))


def smooth_error(steps, errors, window_steps=100):
    """Time-spaced log-error smoothing, preserving the first 100 raw updates."""
    if window_steps <= 0 or window_steps % 50:
        raise ValueError("Smoothing span must be a positive multiple of 50 steps")
    x, y = np.asarray(steps, dtype=float), np.asarray(errors, dtype=float)
    if len(x) != len(y) or len(x) == 0 or np.any(np.diff(x) <= 0):
        raise ValueError("Require equal nonempty arrays and strictly increasing steps")
    if np.any(~np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("Log smoothing requires finite positive errors")
    z = np.log10(y)
    result = y.copy()
    for i, step in enumerate(x):
        if step <= 100:
            continue
        locations = step + np.arange(-window_steps / 2, window_steps / 2 + 1, 25)
        locations = locations[(locations >= x[0]) & (locations <= x[-1])]
        result[i] = 10 ** np.mean(np.interp(locations, x, z))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=OUT / "warmup1000")
    parser.add_argument("--max-step", type=int, help="Plot only this prefix; retain the complete saved history")
    parser.add_argument("--xscale", choices=("linear", "hybrid"))
    args = parser.parse_args()
    output = args.results.resolve()
    config = json.loads((output / "config.json").read_text())
    standard = config.get("initialization") == "standard"
    xscale = args.xscale or ("linear" if standard else "hybrid")
    plot_steps = args.max_step if args.max_step is not None else config["steps"]
    if not 0 < plot_steps <= config["steps"]:
        raise ValueError("--max-step must be positive and no larger than the saved run")
    if xscale == "hybrid" and plot_steps != AXIS_END:
        raise ValueError("The hybrid axis is defined for the 10,000-step figure")
    view_suffix = "" if plot_steps == config["steps"] else f"_{plot_steps // 1000}k" if plot_steps % 1000 == 0 else f"_{plot_steps}"
    reference_output = OUT / "warmup1000" if standard else output
    reference_config = json.loads((reference_output / "config.json").read_text())
    for key in ("targets", "learning_rates", "warmup_steps", "train_points",
                "eval_points", "adam", "dtype", "seed", "threads"):
        assert config[key] == reference_config[key], f"Unmatched QI comparison: {key}"
    data, sources, summaries, displayed = {}, [], [], {}
    for name in config["targets"]:
        data[name] = {}
        sources.append(reference_output / name / "initial.json")
        for lr in config["learning_rates"]:
            path = output / name / f"lr_{lr:.0e}.json"
            record = json.loads(path.read_text())
            h = record["history"]
            assert h[0]["step"] == 0 and h[-1]["step"] == config["steps"]
            sources.append(path)
            summaries.append(record["summary"])
            h = [row for row in h if row["step"] <= plot_steps]
            x = np.asarray([p["step"] for p in h])
            raw = np.asarray([p["eval_rel_l2"] for p in h])
            smoothed = smooth_error(x, raw)
            data[name][lr] = (x, raw, smoothed)
            displayed[f"{name}/lr_{lr:.0e}"] = {
                "step": x.tolist(), "raw_rel_l2": raw.tolist(),
                "displayed_rel_l2": smoothed.tolist(),
            }
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
    for smooth in (True, False):
        fig, axes = plt.subplots(1, 3)
        for ax, name, title in zip(axes, config["targets"], TITLES):
            ax.set_title(title)
            ax.set_yscale("log")
            ax.set_ylim(3e-17, 10)
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.,), numticks=100))
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda y, _: rf"$10^{{{int(round(np.log10(y)))}}}$"
                if int(round(np.log10(y))) % 4 == 0 else ""))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2., 5.), numticks=100))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("Adam step")
            if ax is axes[0]:
                ax.set_ylabel(r"Relative $L^2$ error")
            if xscale == "hybrid":
                ax.set_xscale("function", functions=(step_axis_forward, step_axis_inverse))
                ax.set_xticks([0, 10, 100, 1000, 2000, 4000, 6000, 8000, 10000],
                              ["0", "10", "100", "1k", "2k", "4k", "6k", "8k", "10k"])
                ax.xaxis.set_minor_locator(FixedLocator([1.]))
            else:
                ax.set_xscale("linear")
                ticks = np.linspace(0, plot_steps, 6)
                ax.set_xticks(ticks, [f"{v / 1000:g}k" if v >= 1000 else f"{v:g}" for v in ticks])
                ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(0, plot_steps)
            ax.axvspan(0, config["warmup_steps"], color="#E6E6E6", alpha=.34, lw=0)
            ax.axvspan(config["warmup_steps"], config["warmup_steps"],
                       edgecolor="#888888", lw=.65, linestyle=":")
            initial_info = json.loads((output / name / "initial.json").read_text())
            reference_info = json.loads((reference_output / name / "initial.json").read_text())
            assert initial_info["width"] == reference_info["width"]
            baseline = reference_info["eval_rel_l2"]
            for lr in config["learning_rates"]:
                x, raw, smoothed = data[name][lr]
                assert raw[0] == initial_info["eval_rel_l2"]
                exponent = int(round(np.log10(lr)))
                ax.plot(x, smoothed if smooth else raw, color=COLORS[lr], marker=None,
                        label=rf"$\eta = 10^{{{exponent}}}$", zorder=3)
            ax.axhline(baseline, color="#444444", linestyle="--",
                       label="QI reference" if standard else "QI initialization", zorder=2)
            ax.grid(axis="y", which="major")
            ax.set_axisbelow(True)
            ax.legend()
            ax.annotate(f"QI: {baseline:.1e}", xy=(.97, baseline),
                        xycoords=("axes fraction", "data"), xytext=(0, 6),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=6.3, color="#444444")
        paper_style.apply(fig, axes)
        handles, labels = axes[0].get_legend_handles_labels()
        for ax in axes:
            ax.get_legend().remove()
            ax.set_title(ax.get_title(), fontsize=8.5, fontweight="bold", y=1.17, pad=0)
            # Open up the crowded log labels without changing ticks or font size.
            for value, label in zip(ax.get_xticks(), ax.get_xticklabels()):
                if xscale == "hybrid" and value == 10:
                    label.set_horizontalalignment("right")
                elif xscale == "hybrid" and value in (1000, 2000):
                    label.set_horizontalalignment("left")
            assert all(line.get_marker() in (None, "", "None") for line in ax.lines)
            assert ax.get_xscale() == ("function" if xscale == "hybrid" else "linear")
        fig.subplots_adjust(bottom=.17, top=.80)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .81),
                   ncol=len(config["learning_rates"])+1, frameon=False, fontsize=7,
                   handlelength=1.6, handletextpad=.45, columnspacing=1.4)
        fig.canvas.draw()
        for ax in axes:
            # Check rendered geometry and the paper's actual point sizes.
            pixel_x = ax.transData.transform((AXIS_TRANSITION, 1.))[0]
            expected_fraction = LOG_PANEL_FRACTION if xscale == "hybrid" else AXIS_TRANSITION / plot_steps
            np.testing.assert_allclose((pixel_x - ax.bbox.x0) / ax.bbox.width, expected_fraction)
            assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == 8.
            assert all(t.get_fontsize() == 7. for t in [*ax.get_xticklabels(), *ax.get_yticklabels()])
            boxes = [t.get_window_extent(fig.canvas.get_renderer()) for t in ax.get_xticklabels()]
            assert all(right.x0 - left.x1 > 3. * fig.dpi / 72.
                       for left, right in zip(boxes, boxes[1:]))
        prefix = "standard_adam" if standard else "qi_adam_drift"
        stem = prefix + view_suffix + ("" if smooth else "_raw")
        fig.savefig(output / f"{stem}.png", dpi=600)
        fig.savefig(output / f"{stem}.pdf")
        plt.close(fig)
    with (output / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output / f"displayed_curves{view_suffix}.json").write_text(json.dumps(displayed, indent=2) + "\n")
    (output / f"figure_provenance{view_suffix}.json").write_text(json.dumps({
        "style_file": str(Path(paper_style.__file__).relative_to(ROOT)),
        "style_sha256": hashlib.sha256(Path(paper_style.__file__).read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_figure": "results/checkpoint_C_geometry/expC11_true_precision_law/figures/all_targets_chirp_precision_law_true_600dpi.png",
        "data_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "smoothing": SMOOTHING,
        "initialization": config.get("initialization", "qi"),
        "qi_reference_directory": str(reference_output.relative_to(ROOT)),
        "markers": "None",
        "x_axis": "linear" if xscale == "linear" else "First 25%: steps 0..1000 (linear 0..1, logarithmic 1..1000). Remaining 75%: steps 1000..10000 linearly. Continuous at joins.",
        "plotted_steps": plot_steps,
        "saved_steps": config["steps"],
        "axis_label_fontsize_pt": 8,
        "tick_label_fontsize_pt": 7,
        "tick_label_alignment": "centered" if xscale == "linear" else "10 right-aligned; 1k and 2k left-aligned; other labels centered; at least 3 pt between labels",
        "log_section_width_fraction": LOG_PANEL_FRACTION if xscale == "hybrid" else 0.,
        "legend_position": "Above graphs, below function titles",
        "shading": f"Linear learning-rate warmup, steps 0 through {config['warmup_steps']}",
    }, indent=2) + "\n")
    print(output / (("standard_adam" if standard else "qi_adam_drift") + view_suffix + ".png"))


if __name__ == "__main__":
    main()
