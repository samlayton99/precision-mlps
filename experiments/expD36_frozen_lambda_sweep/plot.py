"""Four requested 3-by-4 views, drawn only from saved data."""
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "precisionmlps-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, FuncFormatter
import numpy as np

LABELS = {
    "sine": ("Sine", r"$\sqrt{2}\sin(2\pi x)$"),
    "quadratic": ("Quadratic", r"$\sqrt{5}\,x^2$"),
    "mixed": ("Mixed sine", r"$[\sin(2\pi x)+0.1\sin(20\pi x)]/\sqrt{0.505}$"),
    "runge": ("Runge", r"$1/(1+25x^2)$"),
}


def plot_all(output, baseline_output=None):
    output = Path(output)
    meta = json.loads((output / "data/metadata.json").read_text())
    with np.load(output / "data/trajectory.npz") as data:
        steps, errors = data["steps"], data["eval_rel_l2"]
    with np.load(output / "data/reference.npz") as data:
        lambdas, floor = data["lambdas"], data["eval_rel_l2"]
    horizon = int(steps[-1])
    baseline_errors = None
    if baseline_output is not None:
        baseline_output = Path(baseline_output)
        with np.load(baseline_output / "data/trajectory.npz") as data:
            raw_steps, raw_errors = data["steps"], data["eval_rel_l2"]
        baseline_errors = raw_errors[raw_steps <= horizon]
        np.testing.assert_array_equal(steps, raw_steps[raw_steps <= horizon])
        with np.load(baseline_output / "data/reference.npz") as data:
            np.testing.assert_array_equal(lambdas, data["lambdas"])
            np.testing.assert_array_equal(floor, data["eval_rel_l2"])
    requested = meta["config"]["snapshot_steps"]
    views = [("early", requested[0]), ("middle", requested[1]), ("final", horizon), ("best", None)]
    colors = plt.get_cmap("viridis")(np.linspace(.08, .92, len(lambdas)))
    method_colors = ["#c65c18", "#1766a3"]
    combined = errors if baseline_errors is None else np.concatenate((errors, baseline_errors))
    pos = combined[combined > 0]
    trajectory_min = 10**np.floor(np.log10(pos.min()))
    upper = max(1.25, 10**np.ceil(np.log10(pos.max())) if pos.max() > 1.25 else 1.25)
    top_min = min(1e-16, 10**np.floor(np.log10(floor[floor > 0].min())))
    best_index = np.argmin(errors, axis=0)
    best_error = np.take_along_axis(errors, best_index[None], axis=0)[0]
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, selected in views:
        if selected is not None and selected not in steps:
            continue
        if selected is None:
            sliced = best_error
            heading = f"Best recorded relative L2 through {horizon:,} steps"
            selected_text = "Dots mark each curve's best recorded evaluation; selections can occur at different steps."
        else:
            sliced = errors[np.flatnonzero(steps == selected)[0]]
            heading = f"Readout fitting at step {selected:,}"
            selected_text = f"Vertical dashed lines mark step {selected:,}; the top row uses that same step for every run."
        nrows = 3 if baseline_errors is None else 4
        fig, axes = plt.subplots(nrows, 4, figsize=(23, 13.5 if nrows == 3 else 17.5), dpi=160)
        for col, target in enumerate(meta["config"]["targets"]):
            ax = axes[0, col]
            for method in range(2):
                ax.plot(lambdas, sliced[method, :, col], "o-", color=method_colors[method], lw=2, ms=4)
            ax.plot(lambdas, floor[:, col], "o:", color=".18", lw=2, ms=3)
            ax.axvline(.25, color=".75", lw=1, zorder=0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(top_min, upper)
            ax.set_xlim(lambdas[0]/1.25, lambdas[-1]*1.25)
            ax.set_xlabel(r"Scale $\lambda=\gamma h$", fontsize=12)
            title, equation = LABELS[target]
            ax.set_title(title + "\n" + equation, fontsize=13, pad=13)
            for method in range(2):
                ax = axes[method+1, col]
                for k, color in enumerate(colors):
                    ax.plot(steps, errors[:, method, k, col], color=color, lw=1.35, alpha=.95)
                    if selected is None:
                        idx = best_index[method, k, col]
                        ax.scatter(steps[idx], errors[idx, method, k, col], color=color,
                                   s=44, edgecolors="black", linewidths=.7, zorder=5, clip_on=False)
                if selected is not None:
                    ax.axvline(selected, color=".15", ls="--", lw=1.4, zorder=6)
                ax.set_yscale("log")
                ax.set_ylim(trajectory_min*.85, upper)
                ax.set_xlim(0, horizon*1.015)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k" if x else "0"))
                ax.set_xlabel("Gradient step", fontsize=12)
            if baseline_errors is not None:
                raw_slice = (baseline_errors.min(axis=0) if selected is None
                             else baseline_errors[np.flatnonzero(steps == selected)[0]])
                ax = axes[3, col]
                for method, color in enumerate(method_colors):
                    ax.plot(lambdas, raw_slice[method, :, col], "x--", color=color,
                            lw=1.6, ms=5, alpha=.75)
                    ax.plot(lambdas, sliced[method, :, col], "o-", color=color, lw=2, ms=4)
                ax.axvline(.25, color=".8", lw=.8, zorder=0)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_ylim(trajectory_min*.85, upper)
                ax.set_xlim(lambdas[0]/1.25, lambdas[-1]*1.25)
                ax.set_xlabel(r"Scale $\lambda=\gamma h$", fontsize=12)
                ax.set_title("Same-budget comparison" if selected is not None else "Best of each at equal budget",
                             fontsize=11, pad=9)
            for row in range(nrows):
                axes[row, col].grid(which="major", alpha=.2)
                axes[row, col].tick_params(labelsize=10)
                axes[row, col].yaxis.set_major_locator(LogLocator(base=10, numticks=7))
                if col:
                    axes[row, col].tick_params(labelleft=False)
        axes[0, 0].set_ylabel("Selected performance\nRelative L2 error", fontsize=13)
        axes[1, 0].set_ylabel("GD trajectories\nRelative L2 error", fontsize=13)
        axes[2, 0].set_ylabel("Adam trajectories\nRelative L2 error", fontsize=13)
        if baseline_errors is not None:
            axes[3, 0].set_ylabel("Ablation vs raw baseline\nRelative L2 error", fontsize=13)
        display_name = meta.get("display_name", "Frozen geometry")
        fig.suptitle(display_name + " · " + heading, fontsize=21, y=.99)
        methods = [Line2D([], [], color=c, lw=2, marker="o", label=l)
                   for c, l in zip(method_colors, ("GD", "Adam"))]
        methods.append(Line2D([], [], color=".18", ls=":", lw=2, label="Numerical least-squares reference"))
        if baseline_errors is not None:
            methods += [Line2D([], [], color=".35", ls="-", marker="o", lw=2, label="Row 4: ablation"),
                        Line2D([], [], color=".35", ls="--", marker="x", lw=1.6, label="Row 4: raw baseline")]
        fig.legend(handles=methods, loc="upper center", bbox_to_anchor=(.5, .958), ncol=3,
                   frameon=False, fontsize=12)
        handles = [Line2D([], [], color=c, lw=2, label=rf"$\lambda={v:.3g}$")
                   for c, v in zip(colors, lambdas)]
        fig.legend(handles=handles, title="Trajectory colors (same in both optimizer rows)",
                   loc="upper center", bbox_to_anchor=(.5, .916), ncol=min(len(lambdas), 10), frameon=False,
                   fontsize=11, title_fontsize=10)
        fig.subplots_adjust(left=.077, right=.985, top=.797 if nrows == 3 else .825,
                            bottom=.13, hspace=.40, wspace=.16)
        cfg = meta["config"]
        fig.text(.5, .072, selected_text, ha="center", fontsize=12)
        fig.text(.5, .039,
                 f"N = {cfg['n']} · {meta['neurons_including_halo']} neurons including {cfg['halo_per_side']} halos per side · zero readout · {cfg['coordinates']} coordinates · FP64\n"
                 f"{cfg['n_train']:,} training / {cfg['n_eval']:,} evaluation midpoints on [−1, 1] · GD η = 1/σmax(B)² · Adam η = 0.001, β = (0.9, 0.999), ε = 10⁻⁸ · constant rates\n"
                 f"All errors use the independent evaluation grid. Curves sample every step through 100, then every 20. Least squares uses relative SVD cutoff {cfg['readout_rcond']:g}."
                 + ("\nRow 4 reuses the original baseline. GD keeps the same normalized-rate rule; Adam keeps the same native settings."
                    if baseline_errors is not None else ""),
                 ha="center", va="center", fontsize=10, linespacing=1.5)
        path = figure_dir / f"{name}.png"
        fig.savefig(path, dpi=160, facecolor="white")
        plt.close(fig)
        print(f"Saved {path}", flush=True)
