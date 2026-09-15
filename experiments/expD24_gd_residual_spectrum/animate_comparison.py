"""Render the three paired 4x4 comparisons as log-spectrum GIFs.

Run: .venv/bin/python experiments/expD24_gd_residual_spectrum/animate_comparison.py
Replays only trajectories missing intermediate states, verifies their existing
checkpoints exactly, and expands the existing single data file. --render-only
redraws already expanded data. No temporal interpolation is used.
Add --render-only --signed-log to write signed-log copies plus the four-gamma
overview into the experiment's log_axes/ folder, using the existing data files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.expD24_gd_residual_spectrum import readout_comparison as comparison
from experiments.expD24_gd_residual_spectrum import run as original
from experiments.expD24_gd_residual_spectrum import whole_line as whole
from experiments.expD24_gd_residual_spectrum.signed_log import (
    FLOOR, signed_log, set_signed_log_axis,
)
from experiments.expD24_gd_residual_spectrum.spectrum import (
    frame_steps, frame_durations, write_gif_frame,
)


def check_replay(previous, expanded):
    """Recording additional frames must not alter a single training update."""
    np.testing.assert_array_equal(expanded["loss"], previous["loss"])
    indices = [expanded["snapshot_steps"].index(step) for step in previous["snapshot_steps"]]
    for key in previous["parameters"]:
        np.testing.assert_array_equal(expanded["parameters"][key][indices], previous["parameters"][key])


def expand_states(cases, config, path):
    desired = frame_steps(config["steps"])
    saved_original, _, _, _ = original.load_data(original.RESULTS / "data.npz")
    saved_whole, _ = whole.load_data(original.RESULTS / "whole_line/data.npz")
    for i, previous in enumerate(cases):
        if previous["snapshot_steps"] == desired:
            continue
        target, arm, method = (previous[key] for key in ("target", "arm", "method"))
        print(f"[{i+1}/{len(cases)}] Fill animation states: {target}, {arm}, {method}", flush=True)
        problem = comparison.Problem(target, config)
        expanded = (comparison.reused_gd(problem, arm, saved_original, saved_whole,
                                         displayed_steps=desired) if method == "gd" else None)
        if expanded is None:
            expanded = comparison.train(problem, arm, method, displayed_steps=desired)
        check_replay(previous, expanded)
        print("Existing loss history and model checkpoints match exactly.", flush=True)
        expanded["views"] = comparison.diagnose(expanded, problem)
        cases[i] = expanded
        comparison.save_cases(cases, config, path)
    return cases


def render(cases, config, output, seconds=16.0, preview_dir=None, signed_residual=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MultipleLocator, NullFormatter
    from PIL import Image

    steps = frame_steps(config["steps"])
    assert all(case["snapshot_steps"] == steps for case in cases)
    lookup = {(case["target"], case["arm"], mode): view
              for case in cases for mode, view in case["views"].items()}
    k = comparison.display_frequency_grid(config)
    floor = FLOOR if signed_residual else 1e-14
    spatial_display = signed_log if signed_residual else np.asarray
    limits = {}
    for target in config["targets"]:
        views = [view for (name, _, _), view in lookup.items() if name == target]
        maximum = max(abs(view["spectrum"]).max() for view in views)
        limits[target] = (1.08 * max(abs(view["residual"]).max() for view in views),
                          10.0 ** np.ceil(np.log10(maximum)))
    titles = {"gd": "Ordinary gradient descent",
              "varpro": "Variable projection · readout solved every step",
              "gd_refit": "Ordinary GD · readout refitted only for evaluation"}
    cmap = plt.get_cmap("viridis")
    norm = Normalize(0, np.log1p(config["steps"]))
    durations = frame_durations(len(steps), seconds)
    for mode, heading in titles.items():
        fig, axes = plt.subplots(4, 4, figsize=(18, 13.8), dpi=125)
        artists = []
        for row, target in enumerate(config["targets"]):
            half = 2 if target == "gaussian_envelope" else 1
            x = np.linspace(-half, half, 2049)
            for j, arm in enumerate(config["arms"]):
                left, right = axes[row, 2*j:2*j+2]
                view = lookup[target, arm, mode]
                left.plot(x, spatial_display(view["residual"][0]), color="#c2c7cd", lw=0.8)
                right.plot(k, np.maximum(abs(view["spectrum"][0]), floor), color="#c2c7cd", lw=0.8)
                spatial, = left.plot(x, spatial_display(view["residual"][0]), color=cmap(0), lw=1.5)
                spectral, = right.plot(k, np.maximum(abs(view["spectrum"][0]), floor), color=cmap(0), lw=1.4)
                left.set(xlim=(-half, half), ylim=(-limits[target][0], limits[target][0]))
                if signed_residual:
                    set_signed_log_axis(left, signed_log(limits[target][0]))
                left.axhline(0, color="#b2b8be", lw=0.5, zorder=0)
                right.set(xlim=(0, config["max_mode"]), ylim=(floor, limits[target][1]), yscale="log",
                          xticks=np.arange(0, config["max_mode"] + 1, 8),
                          yticks=10.0 ** np.arange(int(np.log10(floor)), int(np.log10(limits[target][1])) + 1, 4))
                right.xaxis.set_minor_locator(MultipleLocator(2))
                right.yaxis.set_minor_formatter(NullFormatter())
                right.grid(axis="x", which="minor", alpha=0.09)
                annotations = [ax.text(0.5, 1.04, "", ha="center", transform=ax.transAxes, fontsize=10)
                               for ax in (left, right)]
                if j == 0:
                    domain = "on ℝ" if target == "gaussian_envelope" else "on [−1, 1]"
                    residual_label = "Signed-log residual" if signed_residual else "Residual"
                    left.set_ylabel(comparison.LABELS[target] + f"\n{domain}\n{residual_label}", fontsize=11)
                else:
                    left.set_ylabel("Signed-log residual" if signed_residual else "Residual", fontsize=11)
                right.set_ylabel("Fourier magnitude", fontsize=11)
                for ax in (left, right):
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.grid(which="major", alpha=0.18)
                    ax.tick_params(labelsize=9, labelbottom=row == 3)
                artists.append((view, spatial, spectral, annotations))
        for ax, label in zip(axes[0], ("Xavier · residual", "Xavier · spectrum", "QI · residual", "QI · spectrum")):
            ax.set_title(label, fontsize=14, pad=33)
        for col in (0, 2):
            axes[-1, col].set_xlabel("x (Gaussian: central region)", fontsize=10)
        for col in (1, 3):
            axes[-1, col].set_xlabel(r"Physical frequency $k=\omega/\pi$", fontsize=11)
        title = fig.suptitle("", fontsize=20, y=0.986)
        fig.text(0.5, 0.951,
                 f"N = {config['resolution']} · halo {config['halo']} per side · GD rate {config['learning_rate']:g} · faint gray: step 0",
                 ha="center", fontsize=11)
        # The full color strip also supplies all later viridis colors to the
        # first frame's GIF palette, preventing color changes from quantizing badly.
        color_ax = fig.add_axes([0.26, 0.908, 0.48, 0.012])
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_ax,
                               orientation="horizontal")
        tick_steps = [0, 2, 10, 50, 200, 800, config["steps"]]
        colorbar.set_ticks(np.log1p(tick_steps), labels=[f"{s:,}" for s in tick_steps])
        colorbar.ax.tick_params(labelsize=9, length=2)
        colorbar.set_label("GD step · more frames near initialization", fontsize=10, labelpad=2)
        scale_note = ("Signed-log residuals as in Checkpoint A: |residual| ≤ 10⁻¹⁶ maps to the center line. Fourier magnitudes: log scale, floor 10⁻¹⁶.\n"
                      "Limits are fixed and matched across the three animations. Sine / mixed sine / Runge use [−1, 1]; Gaussian uses the whole line.\n"
                      if signed_residual else
                      "Signed residuals: linear scale. Fourier magnitudes: log scale. Limits are fixed and matched across the three animations.\n"
                      "Magnitudes below 10⁻¹⁴ meet the display floor. Sine / mixed sine / Runge use [−1, 1]; Gaussian uses the whole line.\n")
        fig.text(0.5, 0.021, scale_note +
                 "Relative L₂ above each panel is measured for the displayed model. Readout SVD cutoff: 10⁻¹³.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=0.085 if signed_residual else 0.065,
                            right=0.985, top=0.80, bottom=0.105,
                            hspace=0.37, wspace=0.29)
        destination = output / f"{mode}.gif"
        partial = destination.with_suffix(".gif.partial")
        palette = None
        print(f"Rendering {mode}: {len(steps)} frames, {seconds:g} seconds", flush=True)
        with partial.open("wb") as stream:
            for index, step in enumerate(steps):
                title.set_text(f"{heading} · step {step:,} / {config['steps']:,}")
                color = cmap(norm(np.log1p(step)))
                for view, spatial, spectral, annotations in artists:
                    spatial.set_ydata(spatial_display(view["residual"][index]))
                    spectral.set_ydata(np.maximum(abs(view["spectrum"][index]), floor))
                    spatial.set_color(color)
                    spectral.set_color(color)
                    for annotation in annotations:
                        annotation.set_text(f"Relative L₂ = {view['relative_l2'][index]:.3g}")
                fig.canvas.draw()
                rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
                palette = write_gif_frame(stream, rgb, palette, int(durations[index]))
                if preview_dir is not None and index in (0, len(steps)//2, len(steps)-1):
                    rgb.save(preview_dir / f"{mode}_{step}.png")
            stream.write(b";")
        partial.replace(destination)
        plt.close(fig)
        print(f"Saved {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=comparison.RESULTS)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--seconds", type=float, default=16.0)
    parser.add_argument("--signed-log", action="store_true",
                        help="Write separate log_axes/ copies with Checkpoint A's signed-log residuals and a 1e-16 floor.")
    args = parser.parse_args()
    path = args.output / "data.npz"
    cases, config = comparison.load_cases(path)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    if not args.render_only:
        cases = expand_states(cases, config, path)
    import tempfile
    preview_dir = Path(tempfile.mkdtemp(prefix="expD24-comparison-previews-"))
    output = args.output.parent / "log_axes" if args.signed_log else args.output
    output.mkdir(parents=True, exist_ok=True)
    render(cases, config, output, args.seconds, preview_dir, signed_residual=args.signed_log)
    if args.signed_log:
        whole_cases, whole_config = whole.load_data(args.output.parent / "whole_line/data.npz")
        whole.plot_overview(whole_cases, whole_config, output, signed_residual=True, filename="whole_line.png")
    print(f"Preview frames: {preview_dir}", flush=True)


if __name__ == "__main__":
    main()
