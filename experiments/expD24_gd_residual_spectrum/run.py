"""expD24: two figures and one compressed data file in a flat results folder.

Run: .venv/bin/python experiments/expD24_gd_residual_spectrum/run.py
Use --plot-only to redraw from data.npz, or --data-only to skip rendering.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.schema import ModelConfig
from src.data.targets import get_target
from src.models.mlp import QIMlp

# Reuse the exact C05 Xavier RNG convention, without importing its plotting
# drivers or invoking any of its readout solves. Halo size follows Sam's new
# explicit setting, rather than C05's older default_halo rule.
C05_PATH = ROOT / "experiments/expC05_geometry_interpolation/common.py"
_spec = importlib.util.spec_from_file_location("expd24_c05_common", C05_PATH)
assert _spec is not None and _spec.loader is not None
c05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c05)

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results/checkpoint_D_optimizers/expD24_gd_residual_spectrum"
ARMS = ("xavier", "scaled_xavier", "qi_zero")
ARM_LABELS = {
    "xavier": "Xavier\nrandom readout",
    "scaled_xavier": "Scaled Xavier · centers preserved\nrandom readout",
    "qi_zero": "QI geometry · λ = 0.25\nzero readout",
}
TARGET_LABELS = {
    "sine": "Sine",
    "sine_mixture": "Mixed frequencies",
    "runge": "Runge",
    "abs_cubed": r"$|x|^3$",
}
COLORS = ("#2878b5", "#e68a2e", "#31966c")


def midpoint_grid(n: int, domain: list[float]) -> torch.Tensor:
    lo, hi = domain
    return (lo + (torch.arange(n, dtype=torch.float64) + 0.5) * ((hi - lo) / n))[:, None]


def uniform_geometry(resolution: int, halo: int = 24):
    if resolution < 1 or halo < 0:
        raise ValueError("Resolution must be positive and halo nonnegative.")
    h = 2.0 / resolution
    centers = -1.0 + np.arange(-halo, resolution + halo + 1, dtype=np.float64) * h
    return centers, h, halo


def initial_arrays(resolution: int, seed: int, arm: str, lambda_star: float = 0.25,
                   halo: int = 24):
    centers, h, halo = uniform_geometry(resolution, halo)
    width = len(centers)
    a, b = c05.xavier_draw(width, seed, resolution)
    # Separate deterministic readout stream; target-independent and paired
    # exactly between Xavier and scaled Xavier.
    rng = np.random.default_rng([seed, resolution, 24])
    bound = math.sqrt(6.0 / (width + 1))
    c = rng.uniform(-bound, bound, width)
    scale = 1.0
    if arm == "scaled_xavier":
        scale = (lambda_star / h) / np.mean(np.abs(a))
        original_centers = -b / a
        original_signs = np.sign(a)
        a, b = a * scale, b * scale
        np.testing.assert_allclose(-b / a, original_centers, rtol=1e-13, atol=1e-12)
        np.testing.assert_array_equal(np.sign(a), original_signs)
        np.testing.assert_allclose(np.mean(np.abs(a)) * h, lambda_star, rtol=1e-14)
    elif arm == "qi_zero":
        a = np.full(width, lambda_star / h)
        b = -a * centers
        c = np.zeros(width)
    elif arm != "xavier":
        raise ValueError(f"Unknown arm {arm}")
    arrays = {"a": a, "b": b, "c": c, "d": np.zeros(1)}
    metadata = {"resolution": resolution, "width": width, "halo": int(halo),
                "h": h, "scale_factor": float(scale), "seed": seed, "arm": arm}
    return arrays, metadata


def model_from_arrays(arrays: dict[str, np.ndarray]) -> QIMlp:
    model = QIMlp(ModelConfig(width=len(arrays["a"]), layer_type="standard"))
    with torch.no_grad():
        for name, parameter in named_arrays(model).items():
            parameter.copy_(torch.as_tensor(arrays[name]).reshape_as(parameter))
    return model


def named_arrays(model: QIMlp) -> dict[str, torch.Tensor]:
    return {"a": model.inner_layer.linear.weight,
            "b": model.inner_layer.linear.bias,
            "c": model.readout.weight,
            "d": model.readout.bias}


def snapshot(model: QIMlp) -> dict[str, np.ndarray]:
    return {key: p.detach().numpy().reshape(-1).copy() for key, p in named_arrays(model).items()}


def train_case(config: dict, target_name: str, arm: str, transform, displayed_steps):
    resolution = config["resolutions"][0]
    arrays, metadata = initial_arrays(resolution, config["seed"], arm,
                                      config["lambda_star"], config["halo"])
    model = model_from_arrays(arrays)
    x_train = midpoint_grid(config["n_train"], config["domain"])
    x_eval = midpoint_grid(config["n_eval"], config["domain"])
    target = get_target(target_name)
    y_train, y_eval = target.fn(x_train), target.fn(x_eval)
    y_norm = torch.linalg.vector_norm(y_eval)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                momentum=0, weight_decay=0)
    count = config["steps"] + 1
    losses, relative_l2 = np.empty(count), np.empty(count)
    spectra = np.empty((len(displayed_steps), len(transform.modes)), dtype=np.complex128)
    frame_parameters = {key: np.empty((len(displayed_steps), value.size))
                        for key, value in arrays.items()}
    frame_indices = {step: index for index, step in enumerate(displayed_steps)}
    for step in range(count):
        optimizer.zero_grad(set_to_none=True)
        residual = model(x_train) - y_train
        loss = 0.5 * residual.square().mean()
        with torch.no_grad():
            eval_residual = model(x_eval) - y_eval
            losses[step] = loss.item()
            relative_l2[step] = (torch.linalg.vector_norm(eval_residual) / y_norm).item()
            # Evaluate at every update; store spectra and parameters only for
            # displayed frames to keep the single data file compact.
            transformed = transform(eval_residual.numpy().ravel())
            if not np.isfinite(transformed).all() or not np.isfinite(losses[step]):
                raise FloatingPointError(f"Nonfinite result at {target_name}/{arm}, step {step}")
            if step in frame_indices:
                index = frame_indices[step]
                spectra[index] = transformed
                for key, value in snapshot(model).items():
                    frame_parameters[key][index] = value
        if step < config["steps"]:
            loss.backward()
            optimizer.step()
    return {"target": target_name, "arm": arm, "metadata": metadata,
            "loss": losses, "relative_l2": relative_l2, "spectra": spectra,
            "frame_parameters": frame_parameters,
            "final_parameters": snapshot(model)}


def save_data(cases, config, modes, displayed_steps, path):
    """Array axes: target, initialization, step/frame, then frequency/neuron.

    Losses cover every update. Complex spectra and model parameters cover the
    selected animation frames, so these frames can also be analyzed or redrawn
    on another evaluation/frequency grid without repeating training.
    """
    lookup = {(case["target"], case["arm"]): case for case in cases}
    ordered = [[lookup[target, arm] for arm in config["arms"]]
               for target in config["targets"]]
    payload = {
        "config_json": json.dumps(config),
        "metadata_json": json.dumps([[case["metadata"] for case in row] for row in ordered]),
        "steps": np.arange(config["steps"] + 1),
        "frame_steps": np.asarray(displayed_steps),
        "modes": modes,
    }
    for key in ("loss", "relative_l2", "spectra"):
        payload[key] = np.asarray([[case[key] for case in row] for row in ordered])
    for key in ("a", "b", "c", "d"):
        payload[key] = np.asarray([[case["frame_parameters"][key] for case in row]
                                   for row in ordered])
    np.savez_compressed(path, **payload)


def load_data(path):
    with np.load(path, allow_pickle=False) as data:
        config = json.loads(data["config_json"].item())
        metadata = json.loads(data["metadata_json"].item())
        # Materialize once; indexing an NpzFile repeatedly decompresses it.
        arrays = {key: data[key] for key in ("loss", "relative_l2", "spectra", "a", "b", "c", "d")}
        cases = []
        for i, target in enumerate(config["targets"]):
            for j, arm in enumerate(config["arms"]):
                parameters = {key: arrays[key][i, j] for key in ("a", "b", "c", "d")}
                cases.append({"target": target, "arm": arm, "metadata": metadata[i][j],
                              **{key: arrays[key][i, j] for key in ("loss", "relative_l2", "spectra")},
                              "frame_parameters": parameters,
                              "final_parameters": {key: value[-1] for key, value in parameters.items()}})
        return cases, config, data["modes"], data["frame_steps"].tolist()


def plot_loss(cases, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(case["target"], case["arm"]): case for case in cases}
    minimum = min(case["loss"].min() for case in cases)
    maximum = max(case["loss"].max() for case in cases)
    limits = (10.0 ** math.floor(math.log10(minimum)),
              10.0 ** math.ceil(math.log10(maximum)))
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True, sharey=True)
    for i, target in enumerate(config["targets"]):
        for j, arm in enumerate(config["arms"]):
            ax = axes[i, j]
            ax.plot(np.arange(config["steps"] + 1), lookup[target, arm]["loss"],
                    color=COLORS[0], linewidth=2)
            ax.set_yscale("log")
            ax.set_ylim(*limits)
            ax.set_xlim(0, config["steps"])
            ax.grid(which="major", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(ARM_LABELS[arm], fontsize=12, pad=15)
            if j == 0:
                ax.set_ylabel(TARGET_LABELS[target] + "\nTraining half-MSE", fontsize=12)
            if i == 3:
                ax.set_xlabel("GD step")
    fig.suptitle(f"Training loss · N = {config['resolutions'][0]} · {config['steps']:,} GD steps", fontsize=19, y=0.985)
    fig.text(0.5, 0.945, f"Learning rate {config['learning_rate']:g} · halo {config['halo']} per side · all parameters trained · fp64", ha="center", fontsize=11)
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.065, top=0.84, hspace=0.2, wspace=0.12)
    fig.savefig(output / "loss.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--output", type=Path, default=RESULTS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--data-only", action="store_true", help="Train and save data.npz without rendering.")
    mode.add_argument("--plot-only", action="store_true", help="Render the saved data.npz without training.")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.plot_only:
        cases, config, modes, displayed_steps = load_data(output / "data.npz")
    else:
        config = yaml.safe_load(args.config.read_text())
    assert len(config["resolutions"]) == 1 and len(config["targets"]) == 4
    assert config["arms"] == list(ARMS) and config["domain"] == [-1.0, 1.0]
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "precisionmlps-d24-mpl"))
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    # Import only the experiment's plotting/math helper, never another runner.
    spec = importlib.util.spec_from_file_location("expd24_spectrum", HERE / "spectrum.py")
    spectrum = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spectrum)
    if not args.plot_only:
        displayed_steps = spectrum.frame_steps(config["steps"])
        x_eval = midpoint_grid(config["n_eval"], config["domain"]).numpy().ravel()
        transform = spectrum.FiniteIntervalTransform(x_eval, config["max_mode"], config["frequency_points"])
        modes = transform.modes
        cases = []
        for target in config["targets"]:
            for arm in config["arms"]:
                print(f"[{len(cases)+1}/12] {target}, {arm}, {config['steps']} steps", flush=True)
                cases.append(train_case(config, target, arm, transform, displayed_steps))
        save_data(cases, config, modes, displayed_steps, output / "data.npz")
        print(f"Saved data: {output / 'data.npz'}", flush=True)
    if args.data_only:
        return
    plot_loss(cases, config, output)
    previews = Path(tempfile.mkdtemp(prefix="expD24-preview-"))
    spectrum.animate(cases, config, modes, displayed_steps, output,
                     ARM_LABELS, TARGET_LABELS, previews)
    print(f"Preview images in session scratch: {previews}", flush=True)
    print(f"Results: {output / 'spectrum.gif'} and {output / 'loss.png'}", flush=True)


if __name__ == "__main__":
    main()
