"""Reproduce frequency-learning, spectral-coverage, and interior-check figures.

Run: .venv/bin/python experiments/expD24_gd_residual_spectrum/analyze.py
Uses the saved model states; does not repeat training or modify the GIF.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.integrate import cumulative_trapezoid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "precisionmlps-d24-mpl"))

from experiments.expD24_gd_residual_spectrum.run import RESULTS, TARGET_LABELS, load_data
from experiments.expD24_gd_residual_spectrum.spectrum import FiniteIntervalTransform
from src.data.targets import get_target

NAMES = {"xavier": "Standard Xavier", "scaled_xavier": "Scaled Xavier",
         "qi_zero": "QI geometry · zero readout"}
COLORS = ("#2166ac", "#dc7927", "#2b8c5a")


def residual(case, x, frame=-1):
    parameters = case["frame_parameters"]
    a, b, c, d = (parameters[key][frame] for key in ("a", "b", "c", "d"))
    return (np.tanh(np.asarray(x)[..., None] * a + b) @ c + d[0]
            - get_target(case["target"]).fn_numpy(np.asarray(x)))


def band_energies(case, config, modes, frame_steps):
    """Contributions to relative L2 squared, for the unchanged windowed residual.

    k = omega/pi on [-1,1], so integral_0^infinity |E(pi*k)|^2 dk equals
    integral_-1^1 |e(x)|^2 dx. The high band is the Parseval complement and
    includes frequencies outside the saved spectrum's displayed range.
    """
    n = config["n_eval"]
    x = -1 + (np.arange(n) + 0.5) * 2 / n
    target_energy = 2 * np.mean(get_target(case["target"]).fn_numpy(x) ** 2)
    power = abs(case["spectra"]) ** 2
    low_mask = modes <= 4
    middle_mask = (modes >= 4) & (modes <= 10)
    assert modes[low_mask][-1] == 4 and modes[middle_mask][-1] == 10
    low = np.trapezoid(power[:, low_mask], modes[low_mask], axis=-1) / target_energy
    middle = np.trapezoid(power[:, middle_mask], modes[middle_mask], axis=-1) / target_energy
    total = case["relative_l2"][frame_steps] ** 2
    high = total - low - middle
    assert np.all(high >= 0), "Band quadrature exceeds total residual energy."
    return low, middle, high, total


def plot_frequency_learning(cases, config, modes, frame_steps, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(case["target"], case["arm"]): case for case in cases}
    selected = [lookup["sine_mixture", arm] for arm in config["arms"]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=160, sharex="row", sharey="row")
    x = np.linspace(-1, 1, 2001)
    final = [residual(case, x) for case in selected]
    spatial_limit = 1.05 * max(np.max(abs(values)) for values in final)
    for j, case in enumerate(selected):
        ax = axes[0, j]
        ax.plot(x, final[j], color="#333c48", lw=1.2)
        ax.axhline(0, color="#858b91", lw=0.6)
        ax.set(xlim=(-1, 1), ylim=(-spatial_limit, spatial_limit), xlabel="x")
        ax.set_title(NAMES[case["arm"]], fontsize=13, pad=30)
        ax.text(0.5, 1.035, f"Final relative L₂ = {case['relative_l2'][-1]:.3f}",
                ha="center", transform=ax.transAxes, fontsize=10)
        low, middle, high, total = band_energies(case, config, modes, frame_steps)
        ax = axes[1, j]
        for values, color in zip((low, middle, high), COLORS):
            ax.plot(frame_steps, values, color=color, lw=2)
        ax.plot(frame_steps, total, color="#333c48", lw=1, ls="--")
        ax.set(xlim=(0, config["steps"]), ylim=(1e-5, 30), yscale="log", xlabel="GD step")
        for ax in axes[:, j]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.15)
    axes[0, 0].set_ylabel(f"Residual in real space\nat step {config['steps']:,}", fontsize=12)
    axes[1, 0].set_ylabel("Contribution to squared\nrelative L₂ error", fontsize=12)
    handles = [plt.Line2D([], [], color=color, lw=2) for color in COLORS]
    handles.append(plt.Line2D([], [], color="#333c48", ls="--", lw=1))
    fig.legend(handles, ["Low: 0 ≤ k ≤ 4", "Middle: 4 < k ≤ 10", "High: k > 10",
                         "Total squared relative L₂"], loc="upper center",
               bbox_to_anchor=(0.5, 0.455), ncol=4, frameon=False, fontsize=11)
    fig.suptitle("Where the mixed-frequency error remains", fontsize=19, y=0.985)
    fig.text(0.5, 0.935, "Target: sin(2πx) + 0.5 sin(6πx) + 0.25 sin(14πx) · k = ω/π · unchanged saved training runs",
             ha="center", fontsize=11)
    fig.text(0.5, 0.018, "Bands integrate squared magnitude from the original, untapered finite-interval transform.\n"
             "High-frequency energy includes the undisplayed tail, using Parseval; band boundaries separate the three target frequencies.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.805, bottom=0.105, wspace=0.13, hspace=0.48)
    fig.savefig(output / "frequency_learning.png")
    plt.close(fig)


def measure_coverage(cases, max_mode=512, n_eval=32768):
    """Remaining energy beyond each cutoff, using a denser spatial grid.

    Integrate the continuous transform on a 1/16 frequency grid. Parseval
    supplies the full energy, including the tail beyond max_mode; no assumption
    of zero spectral energy at the last evaluated frequency is made.
    """
    x = -1 + (np.arange(n_eval) + 0.5) * 2 / n_eval
    transform = FiniteIntervalTransform(x, max_mode=max_mode, points=max_mode * 16 + 1)
    outside = {}
    for case in cases:
        r = residual(case, x)
        total = 2 * np.mean(r * r)
        integrated = cumulative_trapezoid(abs(transform(r)) ** 2, transform.modes, initial=0)
        fraction = 1 - integrated / total
        assert fraction.min() > -1e-6, "Fourier quadrature exceeds the spatial energy."
        outside[case["target"], case["arm"]] = fraction
    return transform.modes, outside


def plot_coverage(config, cutoffs, outside, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=160, sharex=True, sharey=True)
    keep = cutoffs >= 16
    for ax, target in zip(axes.flat, config["targets"]):
        for arm, color in zip(config["arms"], COLORS):
            ax.loglog(cutoffs[keep], np.maximum(100 * outside[target, arm][keep], 1e-5),
                      color=color, lw=2, label=NAMES[arm])
        ax.axvline(config["max_mode"], color="#777777", ls="--", lw=1)
        ax.set_title(TARGET_LABELS[target], fontsize=12)
        ax.set(xlim=(16, cutoffs[-1]), ylim=(1e-5, 100), xticks=[16, 32, 64, 128, 256, 512])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(which="major", alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Residual energy beyond cutoff (%)", fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel("Frequency cutoff k = ω/π", fontsize=11)
    fig.suptitle(f"How much residual energy is outside the frequency window? · step {config['steps']:,}", fontsize=15, y=0.985)
    fig.text(0.5, 0.938, f"Dashed line: current GIF cutoff, k = {config['max_mode']:g} · lower curves mean more of the energy is included",
             ha="center", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.79, bottom=0.09, hspace=0.24, wspace=0.12)
    fig.savefig(output / "spectral_coverage.png")
    plt.close(fig)


def plot_halo_and_interior(cases, config, output):
    """Evaluate the saved QI models on a smaller interval, without retraining."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = {case["target"]: case for case in cases if case["arm"] == "qi_zero"}
    reference = selected[config["targets"][0]]["frame_parameters"]
    centers = -reference["b"][0] / reference["a"][0]
    n = 32768
    full_x = -1 + (np.arange(n) + 0.5) * 2 / n
    inner_x = -0.8 + (np.arange(n) + 0.5) * 1.6 / n
    fig = plt.figure(figsize=(12, 9), dpi=160)
    grid = fig.add_gridspec(3, 2, height_ratios=(0.9, 2, 2))
    ax = fig.add_subplot(grid[0, :])
    halo = abs(centers) > 1
    ax.scatter(centers[~halo], np.full(sum(~halo), 0.4), marker="|", s=100, color="#555d66")
    ax.scatter(centers[halo], np.full(sum(halo), 0.4), marker="|", s=100, color="#dc7927")
    ax.plot([-1, 1], [0.05, 0.05], color="#2166ac", lw=5)
    ax.plot([-0.8, 0.8], [-0.3, -0.3], color="#2b8c5a", lw=5)
    ax.text(0, 0.62, "Initial QI neuron centers", ha="center", fontsize=11)
    ax.text(-1.19, 0.62, "24 halo", ha="center", fontsize=10, color="#ad5b12")
    ax.text(1.19, 0.62, "24 halo", ha="center", fontsize=10, color="#ad5b12")
    ax.text(1.05, 0.05, "Training interval", va="center", fontsize=10, color="#2166ac")
    ax.text(0.85, -0.3, "Interior check", va="center", fontsize=10, color="#2b8c5a")
    ax.set(xlim=(-1.5, 1.5), ylim=(-0.6, 0.9), yticks=[],
           xticks=[-1.375, -1, -0.8, 0, 0.8, 1, 1.375])
    ax.set_xticklabels(["−1.375", "−1", "−0.8", "0", "0.8", "1", "1.375"], fontsize=9)
    ax.spines[["left", "right", "top"]].set_visible(False)
    results = {}
    for index, target in enumerate(config["targets"]):
        case = selected[target]
        function = get_target(target).fn_numpy
        full_r, inner_r = residual(case, full_x), residual(case, inner_x)
        full_energy, inner_energy = 2 * np.mean(full_r ** 2), 1.6 * np.mean(inner_r ** 2)
        full_relative = np.linalg.norm(full_r) / np.linalg.norm(function(full_x))
        inner_relative = np.linalg.norm(inner_r) / np.linalg.norm(function(inner_x))
        fraction = inner_energy / full_energy
        assert 0 <= fraction <= 1 + 1e-6
        results[target] = {"full_relative_l2": float(full_relative),
                           "interior_relative_l2": float(inner_relative),
                           "error_energy_fraction_inside": float(fraction)}
        ax = fig.add_subplot(grid[1 + index // 2, index % 2])
        x = np.linspace(-1, 1, 2001)
        ax.plot(x, residual(case, x), color="#2166ac", lw=1.4)
        ax.axvspan(-0.8, 0.8, color="#2b8c5a", alpha=0.10)
        for boundary in (-0.8, 0.8):
            ax.axvline(boundary, color="#2b8c5a", ls="--", lw=0.8)
        ax.axhline(0, color="#777777", lw=0.6)
        ax.set(xlim=(-1, 1), xlabel="x", ylabel="Actual residual")
        ax.set_title(f"{TARGET_LABELS[target]} · {100 * fraction:.2f}% of error energy inside [−0.8, 0.8]",
                     fontsize=11, pad=29)
        ax.text(0.5, 1.035, f"Relative L₂: full {full_relative:.4f} · interior {inner_relative:.4f}",
                ha="center", transform=ax.transAxes, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.suptitle(f"QI halo and an interior-only check · saved models at step {config['steps']:,}", fontsize=17, y=0.985)
    fig.text(0.5, 0.946, "The halo adds centers outside [−1, 1]; training samples remain inside [−1, 1]. No training was repeated.",
             ha="center", va="top", fontsize=10)
    fig.text(0.5, 0.022, "Green shading: the proposed interior interval. Relative L₂ is normalized by the target norm on each respective interval.\n"
             "Each residual panel uses its own vertical scale. Cropping a Fourier integral would create new cutoff boundaries at ±0.8.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.885, bottom=0.1, hspace=0.7, wspace=0.2)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "halo_and_interior.png")
    plt.close(fig)
    return results


def window_spectrum(case, half_width, max_k=32, points=2049, n_eval=32768):
    """Use physical k = omega/pi, independent of the observation interval.

    FiniteIntervalTransform counts cycles across its own interval. Convert k
    to those cycles before evaluating, so cropping never shifts the frequency
    axis. Magnitudes are unnormalized integrals, as in the original GIF.
    """
    x = -half_width + (np.arange(n_eval) + 0.5) * 2 * half_width / n_eval
    length = n_eval * float(x[1] - x[0])
    transform = FiniteIntervalTransform(x, max_mode=max_k * length / 2, points=points)
    r = residual(case, x)
    target = get_target(case["target"]).fn_numpy(x)
    return {"x": x, "residual": r, "k": transform.omega / np.pi,
            "spectrum": transform(r),
            "relative_l2": float(np.linalg.norm(r) / np.linalg.norm(target))}


def plot_interior_spectra(cases, config, output, log_scale=True):
    """Four target rows: saved QI residual, full spectrum, interior spectrum."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, LogLocator, NullFormatter

    selected = {case["target"]: case for case in cases if case["arm"] == "qi_zero"}
    max_k = config["max_mode"]
    diagnostics = {
        target: [window_spectrum(selected[target], half_width, max_k=max_k,
                                 points=config["frequency_points"])
                 for half_width in (1.0, 0.8)]
        for target in config["targets"]
    }
    floor = 1e-6
    maximum = max(
        abs(window["spectrum"]).max()
        for windows in diagnostics.values() for window in windows)
    ceiling = 10.0 ** np.ceil(np.log10(maximum)) if log_scale else 1.08 * maximum
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), dpi=160)
    full_color, inner_color = COLORS[0], COLORS[2]
    for row, target in enumerate(config["targets"]):
        case = selected[target]
        full, inner = diagnostics[target]
        np.testing.assert_allclose(full["k"], inner["k"], rtol=2e-14, atol=1e-14)
        ax = axes[row, 0]
        x = np.linspace(-1, 1, 4097)
        r = residual(case, x)
        limit = 1.08 * np.max(abs(r))
        ax.axvspan(-0.8, 0.8, color=inner_color, alpha=0.10)
        ax.plot(x, r, color=full_color, lw=1.4)
        for boundary in (-0.8, 0.8):
            ax.axvline(boundary, color=inner_color, ls="--", lw=0.8)
        ax.axhline(0, color="#858b91", lw=0.6)
        ax.set(xlim=(-1, 1), ylim=(-limit, limit), xticks=[-1, -0.5, 0, 0.5, 1])
        ax.set_ylabel(TARGET_LABELS[target] + "\n" + r"$e(x)=\hat f(x)-f(x)$", fontsize=12)
        inner_energy = 1.6 * np.mean(inner["residual"] ** 2)
        full_energy = 2 * np.mean(full["residual"] ** 2)
        ax.text(0.5, 1.04, f"Interior contains {100 * inner_energy / full_energy:.2f}% of error energy",
                transform=ax.transAxes, ha="center", fontsize=10)
        for column, window, color in ((1, full, full_color), (2, inner, inner_color)):
            ax = axes[row, column]
            # Clip only the displayed magnitude; retain the original complex data.
            magnitude = abs(window["spectrum"])
            ax.plot(window["k"], np.maximum(magnitude, floor) if log_scale else magnitude,
                    color=color, lw=1.25)
            ax.set(xlim=(0, max_k), ylim=(floor if log_scale else 0, ceiling),
                   yscale="log" if log_scale else "linear",
                   xticks=np.arange(0, max_k + 1, 4))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            if log_scale:
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
                ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylabel(r"$|\widehat e_I(\omega)|$", fontsize=12)
            ax.text(0.5, 1.04, f"Relative L₂ on this interval = {window['relative_l2']:.4f}",
                    transform=ax.transAxes, ha="center", fontsize=10, color=color)
            ax.grid(axis="x", which="minor", color="#e9edf2", lw=0.35)
        for ax in axes[row]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(which="major", color="#d5dce5", lw=0.55)
            ax.tick_params(labelsize=10)
            if row < 3:
                ax.tick_params(labelbottom=False)
    for ax, title in zip(axes[0], ("Residual in function space", "Spectrum over [−1, 1]",
                                   "Spectrum over [−0.8, 0.8]")):
        ax.set_title(title, fontsize=14, pad=39)
    axes[-1, 0].set_xlabel("x", fontsize=12)
    for ax in axes[-1, 1:]:
        ax.set_xlabel(r"Physical frequency $k=\omega/\pi$", fontsize=12)
    fig.suptitle(f"Full interval versus interior · QI at GD step {config['steps']:,}",
                 fontsize=20, y=0.987)
    fig.text(0.5, 0.947, "Same saved models in all columns · green shading marks [−0.8, 0.8] · identical axes for all eight spectra",
             ha="center", fontsize=11)
    scale_note = "spectral display floor is 10⁻⁶" if log_scale else "all spectral magnitudes use linear axes"
    fig.text(0.5, 0.023,
             r"Spectrum: $\widehat e_I(\omega)=\int_I e(x)e^{-i\omega x}\,dx$ · hard interval cutoff, no taper or length normalization."
             f"\nRelative L₂ uses the target norm on each interval. Residual y-scales vary by row; {scale_note}.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.845, bottom=0.095,
                        hspace=0.38, wspace=0.24)
    output.mkdir(parents=True, exist_ok=True)
    filename = "interior_spectrum.png" if log_scale else "interior_spectrum_linear.png"
    fig.savefig(output / filename)
    plt.close(fig)
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RESULTS / "data.npz")
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases, config, modes, frame_steps = load_data(args.data)
    plot_frequency_learning(cases, config, modes, frame_steps, args.output)
    cutoffs, outside = measure_coverage(cases)
    plot_coverage(config, cutoffs, outside, args.output)
    plot_halo_and_interior(cases, config, args.output / "validation")
    plot_interior_spectra(cases, config, args.output)
    plot_interior_spectra(cases, config, args.output, log_scale=False)
    for target in config["targets"]:
        for arm in config["arms"]:
            print(target, arm, {cutoff: round(float(100 * outside[target, arm][cutoff * 16]), 6)
                                for cutoff in (32, 64, 128, 256, 512)})
    print(f"Saved {args.output / 'frequency_learning.png'} and {args.output / 'spectral_coverage.png'}")


if __name__ == "__main__":
    main()
