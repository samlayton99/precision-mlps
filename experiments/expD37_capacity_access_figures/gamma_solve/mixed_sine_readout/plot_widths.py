"""Replot saved width sweeps with identical axes and no figure footers."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator, NullFormatter
import numpy as np


def load(out, width):
    data = out / "data" / f"N{width}"
    cfg = json.loads((data / "config.json").read_text())
    curves = {}
    for method in ["gd", "adam"]:
        with np.load(data / f"{method}.npz") as f:
            curves[method] = dict(f)
    summaries = {method: json.loads((data / f"{method}_summary.json").read_text())
                 for method in ["gd", "adam"]}
    with np.load(data / "problem.npz") as f:
        residual = f["phi"] @ curves["adam"]["coefficients"] - f["y"][None, :, None]
        final = np.linalg.norm(residual, axis=(1, 2)) / np.linalg.norm(f["y"])
    error_difference = float(np.max(abs(final - curves["adam"]["errors"][-1])))
    assert error_difference < 1e-11
    for method in curves:
        assert np.isfinite(curves[method]["errors"]).all()
        assert np.max(abs(curves[method]["errors"][0] - 1)) < 1e-12
    (data / "plot_checks.json").write_text(json.dumps({
        "final_adam_error_difference_in_original_sample_coordinates": error_difference,
        "all_recorded_errors_finite": True,
        "zero_readout_initial_relative_error_is_one": True}, indent=2) + "\n")
    return cfg, curves, summaries


def style(ax, small=False):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=(1, 1e9), ylim=(.1, 110))
    ax.axhline(1, color=".4", ls=":", lw=.9)
    ax.text(1.6, 1.15, "1%", fontsize=8, color=".4")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xticks([1, 1e3, 1e6, 1e9] if small else [1, 1e2, 1e4, 1e6, 1e8])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(labelsize=8 if small else 9)
    ax.grid(alpha=.13)


def draw(ax, curves, method, gamma_index, color):
    steps_key = "long_steps" if method == "gd" else "steps"
    error_key = "long_errors" if method == "gd" else "errors"
    steps = curves[method][steps_key]
    keep = steps >= 1
    ax.plot(steps[keep], 100*curves[method][error_key][keep, gamma_index],
            color=color, lw=1.5 if method == "gd" else .8)


def main(out, layout):
    widths = [1024, 256]
    data = {width: load(out, width) for width in widths}
    gammas = data[1024][0]["gammas"]
    assert data[256][0]["gammas"] == gammas
    colors = ["#2e5fa5", "#cf8e17", "#218f91", "#d44d46"]
    target = r"$f(x)=\sin(2\pi x)+\frac{1}{2}\sin(6\pi x)+\frac{1}{4}\sin(10\pi x)$"
    if layout in ["2x2", "both"]:
        fig, axes = plt.subplots(2, 2, figsize=(8.8, 6), sharex=True, sharey=True)
        for row, width in enumerate(widths):
            for col, method in enumerate(["gd", "adam"]):
                ax = axes[row, col]
                for j, color in enumerate(colors): draw(ax, data[width][1], method, j, color)
                style(ax)
                if row == 1: ax.set_xlabel("Gradient updates", fontsize=10)
            axes[row, 0].set_ylabel(rf"$N={width}$" + "\nRelative L2 error (%)", fontsize=10)
        axes[0, 0].set_title("GD — actual tanh spectrum", fontsize=11)
        axes[0, 1].set_title("Adam — 20,000 steps", fontsize=11)
        handles = [Line2D([0], [0], color=c, lw=2, label=rf"$\gamma={g:g}$")
                   for c, g in zip(colors, gammas)]
        fig.suptitle(target, y=.99, fontsize=12)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .94),
                   ncol=4, frameon=False, fontsize=10)
        fig.subplots_adjust(left=.11, right=.98, bottom=.095, top=.82, hspace=.20, wspace=.10)
        fig.savefig(out / "gd_vs_adam_widths.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    if layout in ["4x4", "both"]:
        fig, axes = plt.subplots(4, 4, figsize=(10, 7.8), sharex=True, sharey=True)
        for row, (width, method) in enumerate([(1024, "gd"), (1024, "adam"), (256, "gd"), (256, "adam")]):
            for col, (gamma, color) in enumerate(zip(gammas, colors)):
                ax = axes[row, col]
                draw(ax, data[width][1], method, col, color)
                style(ax, small=True)
                if row == 0: ax.set_title(rf"$\gamma={gamma:g}$", fontsize=11, color=color)
                if row == 3: ax.set_xlabel("Gradient updates", fontsize=9)
            axes[row, 0].set_ylabel(rf"$N={width}$" + f" · {'GD' if method == 'gd' else 'Adam'}\nRelative L2 error (%)", fontsize=9)
        fig.suptitle(target, y=.99, fontsize=12)
        for ax in axes[-1]:
            labels = ax.get_xticklabels()
            labels[0].set_ha("left")
            labels[-1].set_ha("right")
        fig.subplots_adjust(left=.10, right=.985, bottom=.07, top=.915, hspace=.17, wspace=.10)
        fig.savefig(out / "gd_vs_adam_widths_by_gamma.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    rows = []
    for width in widths:
        cfg, _, summary = data[width]
        for g, a in zip(summary["gd"], summary["adam"]["rows"]):
            ahit = f"{a['first_hit_one_percent']:,}" if a["first_hit_one_percent"] is not None else ">20,000"
            rows.append(f"| {width} | {g['gamma']:g} | {g['first_hit_one_percent']:,} | {ahit} | {100*a['final_error']:.4f}% |")
    body = """# Mixed-sine width comparison

Status: complete; September 23, 2026.

## TL;DR

Repeat the saved mixed-sine experiment at N=1024 and N=256. GD is computed spectrally through one billion steps; Adam executes 20,000 updates. Every panel uses the same limits: steps 1–10^9 and relative L2 error 0.1%–110%.

## Question

Compare GD and Adam across the two requested widths with identical plot scales.

## Experiment design

The target is $\\sin(2\\pi x)+\\tfrac12\\sin(6\\pi x)+\\tfrac14\\sin(10\\pi x)$ on $[-1,1]$. The same 8,193 endpoint-inclusive samples and gammas 8,12,16,64 are used at both widths. Centers have spacing $h=2/N$, N+1 interior centers, and $\\lceil\\sqrt N\\rceil$ halo centers on each side. Thus N=256 uses 289 tanh features and N=1024 uses 1,089, each with an additional bias. The width change follows the existing halo-count rule, so the physical halo extent also changes.

Geometry is frozen and raw readout parameters start at zero. The objective is half mean squared error. GD uses the SVD of the actual sample-normalized tanh feature matrix and step $0.5/\\lambda_{max}(K)$, evaluating the eigenvalue power formula without long GD training. Adam uses learning rate 0.001, betas (0.9,0.999), epsilon 1e-8, no schedule and no weight decay. Four independent gammas run in a batched tensor. Everything uses FP64. Error is training relative L2, displayed in percent. Adam curves end at 20,000 even though every horizontal axis extends to one billion. Curves below 0.1% are outside the displayed range. No smoothing or best-so-far transformation is applied.

**Code and data.** `mixed_sine_readout/run.py` generates the width-specific problems and reuses the preceding spectral and Adam implementations. `mixed_sine_readout/plot_widths.py` generates these figures. Configuration, inputs, trajectories, optimizer state, and checks are in `data/N256/` and `data/N1024/`.

## Results

| N | Gamma | GD first step ≤1% | Adam first step ≤1% | Adam error at 20,000 |
|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + "\n\n### Figures\n\n"
    if layout in ["2x2", "both"]:
        body += "- [Width by optimizer](gd_vs_adam_widths.png): top row N=1024, bottom row N=256; GD left, Adam right. Four gamma lines per panel. All axes have the same limits.\n"
    if layout in ["4x4", "both"]:
        body += "- [Width and optimizer by gamma](gd_vs_adam_widths_by_gamma.png): columns are gamma 8,12,16,64; rows are N=1024 GD, N=1024 Adam, N=256 GD, N=256 Adam. All axes have the same limits.\n"
    body += """
## Verification and limits

The previously verified GD and Adam implementations are reused. Saved trajectories are checked for finite values and unit initial relative error. Final Adam errors are independently recomputed from saved coefficients and the original sample matrix; discrepancies are in each width's `plot_checks.json`. First crossings do not imply Adam remains below the threshold afterward. The spectral calculation describes exact-arithmetic GD evaluated in FP64; no claim is made about accumulation of rounding over billions of executed updates.

## Conclusions

The figures provide the requested comparison at common scales. No new optimizer tuning was performed.

## Open questions

None added for this plot request.
"""
    (out / "width_comparison_results.md").write_text(body)
    print("\n".join(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layout", choices=["2x2", "4x4", "both"], default="both")
    args = parser.parse_args()
    main(args.out, args.layout)
