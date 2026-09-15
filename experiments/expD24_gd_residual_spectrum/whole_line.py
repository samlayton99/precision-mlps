"""Gaussian-envelope target and a tanh network with cancelling tails.

Run: .venv/bin/python experiments/expD24_gd_residual_spectrum/whole_line.py
Use --plot-only to reproduce figures from whole_line/data.npz.

f(x) = sum_j c_j [tanh(a_j*x+b_j) - tanh(x)], c = v-mean(v), a_j > 0.
The reference subtraction makes tail cancellation explicit even in floating
point. In exact arithmetic sum(c)=0, so it is the zero-sum-readout tanh model.
There is no free output bias. Raw a,b,v are trained jointly by plain GD.
The scale derivative with a center held fixed is still c_j*(x-z_j)*sech^2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import erfc, roots_legendre
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.expD24_gd_residual_spectrum.analyze import RESULTS
from experiments.expD24_gd_residual_spectrum.run import uniform_geometry
from experiments.expD24_gd_residual_spectrum.spectrum import (
    frame_steps, frame_durations, write_gif_frame, snapshot_steps, snapshot_legend,
)

HERE = Path(__file__).resolve().parent


def quadrature(config, refinement=1):
    core, extent = config["quadrature_core"], config["quadrature_extent"]
    central = config["quadrature_central_nodes"] * refinement
    tail = config["quadrature_tail_nodes"] * refinement
    xs, ws = [], []
    for lo, hi, n in ((-extent, -core, tail), (-core, core, central), (core, extent, tail)):
        roots, weights = roots_legendre(n)
        xs.append((lo + hi) / 2 + (hi - lo) / 2 * roots)
        ws.append((hi - lo) / 2 * weights)
    return np.concatenate(xs), np.concatenate(ws)


def target(x, config):
    sine_sum = sum(amplitude * np.sin(np.pi * mode * x)
                   for mode, amplitude in zip(config["target_modes"], config["target_amplitudes"]))
    return np.exp(-0.5 * (x / config["envelope_sigma"]) ** 2) * sine_sum


def target_spectrum(omega, config):
    sigma = config["envelope_sigma"]
    gaussian = lambda w: np.sqrt(2 * np.pi) * sigma * np.exp(-0.5 * (sigma * w) ** 2)
    return sum(amplitude * (gaussian(omega - np.pi * mode) - gaussian(omega + np.pi * mode)) / (2j)
               for mode, amplitude in zip(config["target_modes"], config["target_amplitudes"]))


def target_energy(config):
    omega = np.pi * np.asarray(config["target_modes"])
    amplitudes = np.asarray(config["target_amplitudes"])
    sigma = config["envelope_sigma"]
    gram = np.sqrt(np.pi) * sigma / 2 * (
        np.exp(-sigma**2 * (omega[:, None] - omega) ** 2 / 4)
        - np.exp(-sigma**2 * (omega[:, None] + omega) ** 2 / 4))
    return float(amplitudes @ gram @ amplitudes)


def predict(parameters, x):
    a, b, v = (parameters[key] for key in ("a", "b", "v"))
    c = v - v.mean()
    return (np.tanh(np.asarray(x)[..., None] * a + b)
            - np.tanh(np.asarray(x))[..., None]) @ c


def sech2_spectrum(xi):
    """F(xi)=pi*xi/sinh(pi*xi/2), evaluated without overflow or a 0/0."""
    s = np.pi * abs(np.asarray(xi)) / 2
    result = np.empty_like(s)
    small = s < 1e-4
    result[small] = 2 - s[small] ** 2 / 3 + 7 * s[small] ** 4 / 180
    result[~small] = 4 * s[~small] * np.exp(-s[~small]) / (-np.expm1(-2 * s[~small]))
    return result


def model_spectrum(parameters, omega):
    """Exact whole-line transform of the model; no FFT window is involved."""
    a, b, v = (parameters[key] for key in ("a", "b", "v"))
    if np.any(a <= 0):
        raise ValueError("This cancelling-tail construction requires positive slopes.")
    c, z = v - v.mean(), -b / a
    omega = np.asarray(omega)
    result = np.empty(omega.shape, dtype=np.complex128)
    nonzero = omega != 0
    w = omega[nonzero]
    features = (sech2_spectrum(w[:, None] / a) * np.exp(-1j * w[:, None] * z)
                - sech2_spectrum(w)[:, None]) / (1j * w[:, None])
    result[nonzero] = features @ c
    result[~nonzero] = -2 * np.dot(c, z)
    return result


def residual_tail_bounds(parameters, config):
    """Analytic L1 and L2 upper bounds outside the training quadrature extent."""
    a, b, v = (parameters[key] for key in ("a", "b", "v"))
    c, z = v - v.mean(), -b / a
    extent = config["quadrature_extent"]
    if np.any(a <= 0) or np.any(abs(z) >= extent):
        raise ValueError("Slopes/centers left the domain of the tail bound.")
    distances = np.stack((extent - z, extent + z))
    l1 = np.sum(abs(c) / a * np.exp(-2 * a * distances))
    l2 = np.sum(abs(c) / np.sqrt(a) * np.sqrt(np.sum(np.exp(-4 * a * distances), axis=0)))
    # The subtracted reference's effective readout is -sum(c).
    l1 += 2 * abs(c.sum()) * np.exp(-2 * extent)
    l2 += np.sqrt(2) * abs(c.sum()) * np.exp(-2 * extent)
    sigma, amplitude = config["envelope_sigma"], sum(abs(a) for a in config["target_amplitudes"])
    l1 += amplitude * sigma * np.sqrt(2 * np.pi) * erfc(extent / (np.sqrt(2) * sigma))
    l2 += amplitude * np.sqrt(sigma * np.sqrt(np.pi) * erfc(extent / sigma))
    return float(l1), float(l2)


def train(config, gamma):
    centers, _, _ = uniform_geometry(config["resolution"], config["halo"])
    params = {
        "a": torch.nn.Parameter(torch.full((len(centers),), gamma, dtype=torch.float64)),
        "b": torch.nn.Parameter(torch.tensor(-gamma * centers)),
        "v": torch.nn.Parameter(torch.zeros(len(centers), dtype=torch.float64)),
    }
    x_np, w_np = quadrature(config)
    x, weights = torch.tensor(x_np), torch.tensor(w_np)
    y = torch.tensor(target(x_np, config))
    reference = torch.tanh(x)[:, None]
    optimizer = torch.optim.SGD(list(params.values()), lr=config["learning_rate"])
    steps = frame_steps(config["steps"])
    indices = {step: index for index, step in enumerate(steps)}
    frames = {key: np.empty((len(steps), len(centers))) for key in params}
    k = np.linspace(0, config["max_mode"], config["frequency_points"])
    spectra = np.empty((len(steps), len(k)), dtype=np.complex128)
    losses = np.empty(config["steps"] + 1)
    tails = np.empty((len(steps), 2))
    exact_target = target_spectrum(np.pi * k, config)
    for step in range(config["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        c = params["v"] - params["v"].mean()
        prediction = (torch.tanh(x[:, None] * params["a"] + params["b"]) - reference) @ c
        loss = 0.5 * torch.sum(weights * (prediction - y) ** 2)
        losses[step] = loss.detach().item()
        if not np.isfinite(losses[step]) or params["a"].detach().min().item() <= 0:
            raise FloatingPointError(f"Invalid state at gamma={gamma}, step={step}")
        if step in indices:
            index = indices[step]
            state = {key: value.detach().numpy().copy() for key, value in params.items()}
            for key in frames:
                frames[key][index] = state[key]
            spectra[index] = model_spectrum(state, np.pi * k) - exact_target
            tails[index] = residual_tail_bounds(state, config)
        if step % 500 == 0:
            print(f"gamma={gamma:g}, step={step}, relative L2={np.sqrt(2 * losses[step] / target_energy(config)):.6f}", flush=True)
        if step < config["steps"]:
            loss.backward()
            optimizer.step()
    return {"initial_gamma": gamma, "loss": losses, "relative_l2": np.sqrt(2 * losses / target_energy(config)),
            "parameters": frames, "spectra": spectra, "tail_bounds": tails}


def save_data(cases, config, path):
    payload = {"config_json": json.dumps(config)}
    for key in ("loss", "relative_l2", "spectra", "tail_bounds"):
        payload[key] = np.stack([case[key] for case in cases])
    for key in ("a", "b", "v"):
        payload[key] = np.stack([case["parameters"][key] for case in cases])
    np.savez_compressed(path, **payload)


def load_data(path):
    with np.load(path, allow_pickle=False) as data:
        config = json.loads(data["config_json"].item())
        arrays = {key: data[key] for key in ("loss", "relative_l2", "spectra", "tail_bounds", "a", "b", "v")}
    cases = [{"initial_gamma": gamma,
              **{key: arrays[key][i] for key in ("loss", "relative_l2", "spectra", "tail_bounds")},
              "parameters": {key: arrays[key][i] for key in ("a", "b", "v")}}
             for i, gamma in enumerate(config["initial_gammas"])]
    return cases, config


def validate(cases, config):
    x, weights = quadrature(config, refinement=2)
    y = target(x, config)
    np.testing.assert_allclose(weights @ (y * y), target_energy(config), rtol=1e-10)
    # gamma=64 has a small but resolved tail beyond k=128.
    k = np.linspace(0, 512, 32769)
    for case in cases:
        # Check all displayed states with independent, denser spatial quadrature.
        discrepancies = []
        for index, step in enumerate(frame_steps(config["steps"])):
            state = {key: values[index] for key, values in case["parameters"].items()}
            r = predict(state, x) - y
            energy = float(weights @ (r * r))
            discrepancies.append(abs(energy - 2 * case["loss"][step]) / energy)
        state = {key: values[-1] for key, values in case["parameters"].items()}
        exact = model_spectrum(state, np.pi * k) - target_spectrum(np.pi * k, config)
        spectral_energy = np.trapezoid(abs(exact) ** 2, k)
        expected = 2 * case["loss"][-1]
        np.testing.assert_allclose(spectral_energy, expected, rtol=1e-8, atol=1e-12)
        assert max(discrepancies) < 1e-8
        assert case["tail_bounds"][:, 1].max() < 1e-12
        print(f"gamma={case['initial_gamma']:g}: max refined-quadrature discrepancy={max(discrepancies):.3g}; "
              f"Parseval relative discrepancy={abs(spectral_energy / expected - 1):.3g}; "
              f"max omitted-tail L2 bound={case['tail_bounds'][:, 1].max():.3g}", flush=True)


def plot_overview(cases, config, output, signed_residual=False, filename="overview.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.expD24_gd_residual_spectrum.signed_log import (
        FLOOR, signed_log, set_signed_log_axis,
    )

    x = np.linspace(-2, 2, 4097)
    y = target(x, config)
    k = np.linspace(0, config["max_mode"], config["frequency_points"])
    steps = frame_steps(config["steps"])
    chosen = snapshot_steps(config["steps"], available=steps)
    rows = len(cases)
    fig, axes = plt.subplots(rows, 3, figsize=(16, 3 * rows + 2.2), dpi=160,
                             sharex="col", sharey="col", squeeze=False)
    colors = snapshot_legend(fig, chosen, y=0.917)
    spectral_max = max(abs(case["spectra"]).max() for case in cases) * 1.08
    spatial_display = signed_log if signed_residual else np.asarray
    residual_max = 0.0
    for row, case in enumerate(cases):
        ax = axes[row, 0]
        for step, color in zip(chosen, colors):
            state = {key: values[steps.index(step)] for key, values in case["parameters"].items()}
            displayed_residual = spatial_display(predict(state, x) - y)
            residual_max = max(residual_max, float(abs(displayed_residual).max()))
            ax.plot(x, displayed_residual, color=color, lw=1.3)
        ax.set(xlim=(-2, 2), ylim=(-1.05 * max(abs(y)), 1.05 * max(abs(y))))
        residual_label = "Signed-log residual" if signed_residual else "Residual"
        ax.set_ylabel(f"Initial γ = {case['initial_gamma']:g}\n{residual_label}", fontsize=13)
        if signed_residual:
            set_signed_log_axis(ax, residual_max)
            ax.axhline(0, color="#b2b8be", lw=0.5, zorder=0)
        ax.text(0.5, 1.035, f"Final relative L₂ = {case['relative_l2'][-1]:.4f}",
                transform=ax.transAxes, ha="center", fontsize=11)
        ax = axes[row, 1]
        for step, color in zip(chosen, colors):
            magnitude = abs(case["spectra"][steps.index(step)])
            ax.plot(k, np.maximum(magnitude, FLOOR) if signed_residual else magnitude, color=color, lw=1.5,
                    label=f"Step {step:,}")
        ax.set(xlim=(0, config["max_mode"]), ylim=(FLOOR if signed_residual else 0, spectral_max))
        if signed_residual:
            ax.set(yscale="log", ylim=(FLOOR, spectral_max), yticks=10.0 ** np.arange(-16, 1, 4))
        ax.set_ylabel("Fourier magnitude", fontsize=11)
        ax = axes[row, 2]
        ax.plot(np.arange(config["steps"] + 1), case["relative_l2"], color="#8d959d", lw=1)
        ax.scatter(chosen, case["relative_l2"][chosen], c=colors, s=26, zorder=3)
        ax.set(xlim=(0, config["steps"]), ylim=(0, 1.04))
        ax.set_ylabel("Relative L₂ on ℝ", fontsize=11)
        for ax in axes[row]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.18)
    for ax, title in zip(axes[0], ("Actual residual in function space", "Analytic whole-line spectrum", "Training error")):
        ax.set_title(title, fontsize=14, pad=33)
    axes[-1, 0].set_xlabel("x (central region shown)", fontsize=11)
    axes[-1, 1].set_xlabel(r"Physical frequency $k=\omega/\pi$", fontsize=11)
    axes[-1, 2].set_xlabel("GD step", fontsize=11)
    fig.suptitle("A residual that decays on the whole line", fontsize=21, y=0.987)
    fig.text(0.5, 0.956,
             r"Target: $e^{-x^2/(2\cdot 0.4^2)}[\sin(2\pi x)+0.5\sin(6\pi x)+0.25\sin(14\pi x)]$",
             ha="center", fontsize=13)
    scales = "signed-log residuals · log spectrum" if signed_residual else "all axes linear"
    fig.text(0.5, 0.934, "Same uniform centers and zero readouts · only the initial scale differs · " + scales,
             ha="center", fontsize=11)
    scale_note = "Checkpoint A signed-log display: |residual| ≤ 10⁻¹⁶ maps to the center line; Fourier magnitudes use a 10⁻¹⁶ display floor.\n" if signed_residual else ""
    fig.text(0.5, 0.014 if signed_residual else 0.02, scale_note +
             "The network's tails cancel; the spectrum uses its analytic transform on ℝ, with no imposed cutoff or smoothing.\n"
             f"N = {config['resolution']}; halo {config['halo']} per side; GD rate {config['learning_rate']:g}; loss = ½ ∫ℝ e² dx. "
             "Readouts sum to zero; no output bias.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=0.075 if signed_residual else 0.07, right=0.985,
                        top=0.815, bottom=0.09 if signed_residual else 0.075,
                        hspace=0.33, wspace=0.26)
    fig.savefig(output / filename)
    plt.close(fig)


def animate(cases, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    x = np.linspace(-2, 2, 2049)
    y = target(x, config)
    k = np.linspace(0, config["max_mode"], config["frequency_points"])
    steps = frame_steps(config["steps"])
    fig, axes = plt.subplots(len(cases), 2, figsize=(12, 3 * len(cases) + 2), dpi=120,
                             sharex="col", sharey="col", squeeze=False)
    lines, labels = [], []
    for row, case in enumerate(cases):
        axes[row, 0].plot(x, -y, color="#c6cbd0", lw=0.9)
        spatial, = axes[row, 0].plot(x, -y, color="#246ca6", lw=1.4)
        axes[row, 0].set(xlim=(-2, 2), ylim=(-1.05 * max(abs(y)), 1.05 * max(abs(y))))
        axes[row, 0].set_ylabel(f"Initial γ = {case['initial_gamma']:g}\nResidual", fontsize=12)
        axes[row, 1].plot(k, abs(case["spectra"][0]), color="#c6cbd0", lw=0.9)
        spectral, = axes[row, 1].plot(k, abs(case["spectra"][0]), color="#246ca6", lw=1.4)
        axes[row, 1].set(xlim=(0, config["max_mode"]),
                         ylim=(0, 1.08 * max(abs(c["spectra"]).max() for c in cases)))
        axes[row, 1].set_ylabel("Fourier magnitude", fontsize=12)
        label = axes[row, 0].text(0.5, 1.04, "", transform=axes[row, 0].transAxes, ha="center", fontsize=11)
        lines.append((spatial, spectral))
        labels.append(label)
        for ax in axes[row]:
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_title("Actual residual · central region", pad=37, fontsize=14)
    axes[0, 1].set_title("Analytic whole-line spectrum", pad=37, fontsize=14)
    axes[-1, 0].set_xlabel("x", fontsize=12)
    axes[-1, 1].set_xlabel(r"Physical frequency $k=\omega/\pi$", fontsize=12)
    title = fig.suptitle("", fontsize=19, y=0.982)
    fig.text(0.5, 0.932, "Gaussian-envelope mixed sine · cancelling tanh tails · gray curves show the initial residual",
             ha="center", fontsize=10)
    fig.text(0.5, 0.025, "All axes are linear and fixed through time. No residual taper, smoothing, or Fourier observation window.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.07, wspace=0.18, hspace=0.29)
    palette = None
    durations = frame_durations(len(steps), config["animation_seconds"])
    with (output / "spectrum.gif").open("wb") as stream:
        for index, step in enumerate(steps):
            title.set_text(f"Whole-line residual · GD step {step:,} / {config['steps']:,}")
            for case, (spatial, spectral), label in zip(cases, lines, labels):
                state = {key: values[index] for key, values in case["parameters"].items()}
                spatial.set_ydata(predict(state, x) - y)
                spectral.set_ydata(abs(case["spectra"][index]))
                label.set_text(f"Relative L₂ = {case['relative_l2'][step]:.4f}")
            fig.canvas.draw()
            rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
            palette = write_gif_frame(stream, rgb, palette, int(durations[index]))
        stream.write(b";")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "whole_line.yaml")
    parser.add_argument("--output", type=Path, default=RESULTS / "whole_line")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        cases, config = load_data(args.output / "data.npz")
    else:
        config = yaml.safe_load(args.config.read_text())
        torch.set_num_threads(config["threads"])
        torch.use_deterministic_algorithms(True)
        # Reuse identical saved trajectories when expanding the scale ladder.
        reusable = {}
        if (args.output / "data.npz").exists():
            previous, old_config = load_data(args.output / "data.npz")
            relevant = lambda cfg: {key: value for key, value in cfg.items() if key != "initial_gammas"}
            if relevant(old_config) == relevant(config):
                reusable = {case["initial_gamma"]: case for case in previous}
        cases = [reusable[gamma] if gamma in reusable else train(config, gamma)
                 for gamma in config["initial_gammas"]]
        save_data(cases, config, args.output / "data.npz")
    validate(cases, config)
    plot_overview(cases, config, args.output)
    if not args.no_gif:
        animate(cases, config, args.output)
    print(f"Saved results in {args.output}", flush=True)


if __name__ == "__main__":
    main()
