"""Boundary controls for the Fourier diagnostic, with closed-form references.

Run: .venv/bin/python experiments/expD24_gd_residual_spectrum/verify_spectrum.py
Outputs go in results/.../expD24_gd_residual_spectrum/validation/.
Training data, the original GIF, and the objective remain unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.integrate import quad_vec

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "precisionmlps-d24-mpl"))

from experiments.expD24_gd_residual_spectrum.analyze import RESULTS, load_data, residual
from experiments.expD24_gd_residual_spectrum.spectrum import FiniteIntervalTransform
from src.data.targets import get_target


def residual_derivative(case, x, frame=-1):
    parameters = case["frame_parameters"]
    a, b, c = (parameters[key][frame] for key in ("a", "b", "c"))
    features = np.tanh(np.asarray(x)[..., None] * a + b)
    return ((1 - features * features) * a) @ c - get_target(case["target"]).deriv_numpy(np.asarray(x))


def boundary_expansion(case, omega):
    """Two integration-by-parts boundary terms, not a fitted spectral curve.

    E = B0 + B1 + (i*omega)^(-2) integral e''(x) exp(-i*omega*x) dx.
    Thus B0+B1 is only a high-frequency approximation; its complex discrepancy
    is reported rather than interpreted as a fraction of additive energy.
    """
    omega = np.asarray(omega)
    assert np.all(omega != 0)
    endpoints = np.array([-1.0, 1.0])
    values = residual(case, endpoints)
    slopes = residual_derivative(case, endpoints)
    left, right = np.exp(1j * omega), np.exp(-1j * omega)
    return ((values[0] * left - values[1] * right) / (1j * omega)
            + (slopes[0] * left - slopes[1] * right) / (1j * omega) ** 2)


def plot_controls(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 8192
    x = -1 + (np.arange(n) + 0.5) * 2 / n
    transform = FiniteIntervalTransform(x, max_mode=32, points=2049)
    k, omega = transform.modes, transform.omega
    sigma = 0.12
    controls = [
        ("Flat error, cutoff at ±1", lambda x: (abs(x) < 1).astype(float),
         2 * np.sinc(k), 1.0),
        ("Same flat error, cutoff at ±0.5", lambda x: (abs(x) < 0.5).astype(float),
         np.sinc(k / 2), 0.5),
        ("Smoothly decaying Gaussian", lambda x: np.exp(-x * x / (2 * sigma * sigma)),
         np.sqrt(2 * np.pi) * sigma * np.exp(-0.5 * (sigma * omega) ** 2), None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=160, sharex="row", sharey="row")
    xx = np.linspace(-1.25, 1.25, 5001)
    for j, (title, function, exact, half_width) in enumerate(controls):
        numerical = transform(function(x))
        max_error = np.max(abs(numerical - exact))
        assert max_error < (6e-7 if half_width is not None else 1e-13)
        # Adaptive integration at noninteger frequencies is independent of the
        # transform algorithm and of the closed-form reference.
        test_indices = np.array([16, 32, 80, 160, 304, 480, 800])
        test_omega = omega[test_indices]
        limit = 1.0 if half_width is None else half_width
        integrand = ((lambda t: np.exp(-t * t / (2 * sigma * sigma)))
                     if half_width is None else (lambda t: 1.0))
        direct, _ = quad_vec(lambda t: integrand(t) * np.exp(-1j * test_omega * t),
                             -limit, limit, epsabs=1e-13, epsrel=1e-13)
        np.testing.assert_allclose(direct, exact[test_indices], atol=1e-13)
        print(title, "max transform vs closed-form discrepancy:", float(max_error))
        ax = axes[0, j]
        ax.plot(xx, function(xx), color="#333c48", lw=1.7)
        if half_width is not None:
            for edge in (-half_width, half_width):
                ax.axvline(edge, color="#a37a3d", ls="--", lw=0.8)
        ax.set(title=title, xlim=(-1.25, 1.25), ylim=(-0.05, 1.12), xlabel="x")
        ax = axes[1, j]
        ax.semilogy(k, np.maximum(abs(numerical), 1e-10), color="#2166ac", lw=2,
                    label="Experiment's transform code")
        ax.semilogy(k, np.maximum(abs(exact), 1e-10), color="#dc7927", lw=1.2, ls="--",
                    label="Exact mathematical formula")
        ax.plot(k[test_indices], np.maximum(abs(direct), 1e-10), ".", color="#333c48",
                markersize=5, label="Independent direct integral")
        ax.set(xlim=(0, 16), ylim=(1e-10, 3), xlabel="Frequency k = ω/π")
        for ax in axes[:, j]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.15)
    axes[0, 0].set_ylabel("Residual supplied to the transform", fontsize=11)
    axes[1, 0].set_ylabel("Fourier magnitude", fontsize=11)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=3,
               frameon=False, fontsize=10)
    fig.suptitle("Do the bumps come from the transform code or the cutoff?", fontsize=17, y=0.985)
    fig.text(0.5, 0.935, "The two flat errors have no interior oscillations. Moving their boundaries moves the spectral zeros.",
             ha="center", va="top", fontsize=11)
    fig.text(0.5, 0.015, "Gaussian: σ = 0.12; its values at ±1 are below 10⁻¹⁵. All three use the same numerical transform.\n"
             "Zero Fourier values are placed at the display floor for the logarithmic axis.", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.12, hspace=0.32, wspace=0.12)
    fig.savefig(output / "fourier_controls.png")
    plt.close(fig)


def plot_endpoint_prediction(case, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 32768
    x = -1 + (np.arange(n) + 0.5) * 2 / n
    transform = FiniteIntervalTransform(x, max_mode=64, points=4097)
    r = residual(case, x)
    outside = abs(x) > 0.9
    edge_fraction = np.sum(r[outside] ** 2) / np.sum(r * r)
    k = transform.modes[1:]
    actual = transform(r)[1:]
    predicted = boundary_expansion(case, transform.omega[1:])
    for lo, hi in ((16, 32), (32, 64)):
        mask = (k >= lo) & (k <= hi)
        numerator = np.trapezoid(abs(actual[mask] - predicted[mask]) ** 2, k[mask])
        denominator = np.trapezoid(abs(actual[mask]) ** 2, k[mask])
        print(f"QI sine endpoint approximation, k={lo}..{hi}, relative complex-spectrum discrepancy:",
              float(np.sqrt(numerator / denominator)))
    print("QI sine residual energy in outer 10% of interval:", float(edge_fraction))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    xx = np.linspace(-1.15, 1.15, 4601)
    extended = np.where(abs(xx) <= 1, residual(case, xx), 0)
    ax = axes[0]
    ax.plot(xx, extended, color="#2166ac", lw=1.7)
    for left, right in ((-1, -0.9), (0.9, 1)):
        ax.axvspan(left, right, color="#dc7927", alpha=0.17)
    for edge in (-1, 1):
        ax.axvline(edge, color="#777777", ls="--", lw=0.8)
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set(xlim=(-1.15, 1.15), xlabel="x", ylabel="Residual", title="The actual residual and its imposed zero extension")
    ax = axes[1]
    ax.semilogy(k, abs(actual), color="#2166ac", lw=1.7, label="Computed residual spectrum")
    ax.semilogy(k, abs(predicted), color="#dc7927", lw=1.2, ls="--", label="Prediction from endpoint values and slopes")
    ax.set(xlim=(16, 64), ylim=(1e-7, 0.03), xlabel="Frequency k = ω/π", ylabel="Fourier magnitude",
           title="Upper-frequency bumps predicted from four numbers")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.87), ncol=2, frameon=False, fontsize=10)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.suptitle(f"Boundary check on the trained network · QI sine · step {config['steps']:,}", fontsize=16, y=0.985)
    fig.text(0.5, 0.93, "The orange spectral prediction uses only e(−1), e(1), e′(−1), e′(1); no spectral coefficients are fitted.",
             ha="center", va="top", fontsize=10)
    fig.text(0.5, 0.022, f"Shaded regions occupy 10% of the interval and contain {100 * edge_fraction:.1f}% of the remaining squared error.\n"
             "The endpoint expansion is a high-frequency approximation, not an additive partition of spectral energy.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.76, bottom=0.16, wspace=0.2)
    fig.savefig(output / "endpoint_prediction.png")
    plt.close(fig)


def main():
    output = RESULTS / "validation"
    output.mkdir(parents=True, exist_ok=True)
    plot_controls(output)
    cases, config, _, _ = load_data(RESULTS / "data.npz")
    selected = next(case for case in cases if case["target"] == "sine" and case["arm"] == "qi_zero")
    plot_endpoint_prediction(selected, config, output)
    print(f"Saved validation figures in {output}")


if __name__ == "__main__":
    main()
