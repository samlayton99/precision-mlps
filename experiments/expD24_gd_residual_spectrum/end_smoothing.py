"""Gaussian smoothing confined to end strips of the saved residuals.

Run: .venv/bin/python experiments/expD24_gd_residual_spectrum/end_smoothing.py
This changes the diagnostic residual, not the trained model or its loss.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.expD24_gd_residual_spectrum.analyze import (
    RESULTS, TARGET_LABELS, load_data, residual,
)
from experiments.expD24_gd_residual_spectrum.spectrum import FiniteIntervalTransform


def smooth_ends(x, values, half_width, sigma=0.02, strip=0.12):
    """Blend Gaussian convolution into a zero-extended residual near its ends.

    Interior |x| <= half_width-strip is preserved exactly. The blend reaches
    one 3*sigma before either endpoint, making the original cutoff disappear.
    Outside the interval the convolution tails are retained, not cut off again.
    The C-infinity blend introduces no new jump at the start of the end strip.
    """
    dx = float(x[1] - x[0])
    if strip <= 3 * sigma:
        raise ValueError("End strip must be wider than three Gaussian sigmas.")
    radius = int(np.ceil(8 * sigma / dx))
    offsets = np.arange(-radius, radius + 1) * dx
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    convolution = fftconvolve(values, kernel, mode="same")
    t = (abs(x) - (half_width - strip)) / (strip - 3 * sigma)
    blend = np.zeros_like(t)
    blend[t >= 1] = 1
    middle = (t > 0) & (t < 1)
    blend[middle] = expit(1 / (1 - t[middle]) - 1 / t[middle])
    return (1 - blend) * values + blend * convolution


def smoothing_diagnostic(case, half_width, sigma=0.02, strip=0.12,
                         dx=0.0001, max_k=32, points=2049):
    # All requested boundaries are cell edges; padding contains the full kernel.
    extent = half_width + 10 * sigma
    n = int(round(2 * extent / dx))
    x = -extent + (np.arange(n) + 0.5) * dx
    raw = np.zeros(n)
    inside = abs(x) < half_width
    raw[inside] = residual(case, x[inside])
    smoothed = smooth_ends(x, raw, half_width, sigma, strip)
    length = n * float(x[1] - x[0])
    transform = FiniteIntervalTransform(x, max_mode=max_k * length / 2, points=points)
    total = float(dx * np.sum(raw ** 2))
    return {"x": x, "raw": raw, "smoothed": smoothed, "k": transform.omega / np.pi,
            "raw_spectrum": transform(raw), "smoothed_spectrum": transform(smoothed),
            "relative_change": float(np.sqrt(dx * np.sum((smoothed - raw) ** 2) / total)),
            "energy_ratio": float(np.sum(smoothed ** 2) / np.sum(raw ** 2))}


def plot_smoothing(cases, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = {case["target"]: case for case in cases if case["arm"] == "qi_zero"}
    checks = {target: [smoothing_diagnostic(selected[target], a,
                                          max_k=config["max_mode"],
                                          points=config["frequency_points"])
                       for a in (1.0, 0.8)] for target in config["targets"]}
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), dpi=160)
    gray, orange = "#8793a0", "#c96119"
    for row, target in enumerate(config["targets"]):
        full, inner = checks[target]
        ax = axes[row, 0]
        ax.plot(full["x"], full["raw"], color=gray, lw=1.4)
        ax.plot(full["x"], full["smoothed"], color=orange, lw=1.4)
        for boundary in (-1, 1):
            ax.axvline(boundary, color="#777777", ls="--", lw=0.7)
        for lo, hi in ((-1.2, -0.88), (0.88, 1.2)):
            ax.axvspan(lo, hi, color=orange, alpha=0.06)
        ax.axhline(0, color="#aaaaaa", lw=0.6)
        limit = 1.08 * max(abs(full["raw"]).max(), abs(full["smoothed"]).max())
        ax.set(xlim=(-1.2, 1.2), ylim=(-limit, limit))
        ax.set_ylabel(TARGET_LABELS[target] + "\nResidual", fontsize=12)
        ax.text(0.5, 1.04, "Interior [−0.88, 0.88] unchanged", ha="center",
                transform=ax.transAxes, fontsize=10)
        ceiling = 1.08 * max(abs(d[key]).max() for d in (full, inner)
                             for key in ("raw_spectrum", "smoothed_spectrum"))
        for column, diagnostic in ((1, full), (2, inner)):
            ax = axes[row, column]
            ax.plot(diagnostic["k"], abs(diagnostic["raw_spectrum"]), color=gray, lw=1.3)
            ax.plot(diagnostic["k"], abs(diagnostic["smoothed_spectrum"]), color=orange, lw=1.5)
            ax.set(xlim=(0, config["max_mode"]), ylim=(0, ceiling),
                   xticks=np.arange(0, config["max_mode"] + 1, 4))
            ax.set_ylabel("Fourier magnitude", fontsize=11)
            ax.text(0.5, 1.04,
                    f"Modified residual retains {100 * diagnostic['energy_ratio']:.1f}% of original energy",
                    ha="center", transform=ax.transAxes, fontsize=9.5)
        for ax in axes[row]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.18)
            ax.tick_params(labelsize=10, labelbottom=row == 3)
    for ax, title in zip(axes[0], ("Residual and smoothed ends", "Starting window [−1, 1]",
                                   "Starting window [−0.8, 0.8]")):
        ax.set_title(title, fontsize=14, pad=36)
    axes[-1, 0].set_xlabel("x", fontsize=12)
    for ax in axes[-1, 1:]:
        ax.set_xlabel(r"Physical frequency $k=\omega/\pi$", fontsize=12)
    fig.suptitle(f"A little Gaussian smoothing at the ends · QI at step {config['steps']:,}",
                 fontsize=19, y=0.989)
    handles = [plt.Line2D([], [], color=color, lw=2) for color in (gray, orange)]
    fig.legend(handles, ["Original residual, zero outside the window", "Only the ends smoothed; Gaussian tails retained"],
               loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=2, frameon=False, fontsize=11)
    fig.text(0.5, 0.02,
             "Gaussian σ = 0.02; blend confined to the last 0.12 units at each end. Interior values are preserved exactly.\n"
             "All axes are linear. Spectral limits match within each row; row scales differ. Smoothing modifies the diagnostic, not the trained error.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.837, bottom=0.09, hspace=0.38, wspace=0.24)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "end_smoothing_linear.png")
    plt.close(fig)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RESULTS / "data.npz")
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    cases, config, _, _ = load_data(args.data)
    checks = plot_smoothing(cases, config, args.output)
    for target, windows in checks.items():
        print(target, [{k: round(w[k], 6) for k in ("relative_change", "energy_ratio")}
                       for w in windows])
    print(args.output / "end_smoothing_linear.png")


if __name__ == "__main__":
    main()
