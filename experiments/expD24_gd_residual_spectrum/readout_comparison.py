"""Paired GD, variable-projection GD, and GD with diagnostic readout refits.

The Gaussian row uses whole-line Fourier quadrature and cancelling tanh tails.
Other rows use the original half-MSE on [-1,1]. All readout solves use the
repo's relative SVD cutoff 1e-13; the geometry step holds that solution fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scipy.linalg as sla
from scipy.special import roots_legendre
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.expD24_gd_residual_spectrum import run as original
from experiments.expD24_gd_residual_spectrum import whole_line as whole
from experiments.expD24_gd_residual_spectrum.spectrum import (
    FiniteIntervalTransform, snapshot_steps, snapshot_legend, frame_steps,
)
from src.data.targets import get_target

HERE = Path(__file__).resolve().parent
RESULTS = original.RESULTS / "readout_comparison"
LABELS = {**original.TARGET_LABELS, "gaussian_envelope": "Gaussian envelope"}
RCOND = 1e-13


def frequency_quadrature(order=32):
    """Resolve soft Xavier bandwidths and target bands, then integrate to infinity."""
    edges = [0, 1e-6, 1e-5, 1e-4, 1e-3, .003, .01, .03, .1, .3,
             1, 2, 4, 6, 8, 10, 12, 14, 16, *np.arange(20, 129, 4)]
    # Keep high-band panels narrow too: least squares may combine features
    # into oscillatory spectra even where the original GD spectrum was tiny.
    roots, weights = roots_legendre(order)
    nodes, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        nodes.append((lo + hi) / 2 + (hi - lo) / 2 * roots)
        ws.append((hi - lo) / 2 * weights)
    # k = 128 + 128*t/(1-t), t in (0,1), so no high-frequency cutoff.
    roots, weights = roots_legendre(2 * order)
    t = (roots + 1) / 2
    nodes.append(128 + 128 * t / (1 - t))
    ws.append(64 * weights / (1 - t) ** 2)
    return np.concatenate(nodes), np.concatenate(ws)


def whole_features(a, b, k):
    """Fourier design with real readouts and exact cancellation of both tails.

    Signed Xavier slopes are allowed. The signed readouts are projected onto
    sum(sign(a)*c)=0. Within a fixed sign pattern this is a smooth tanh model.
    Subtract the common zero-frequency singularity before dividing by omega.
    """
    omega = np.pi * k[:, None]
    s = np.pi * omega / (2 * abs(a))
    F = 4 * s * torch.exp(-s) / (-torch.expm1(-2 * s))
    delta = torch.where(s < 1e-3, -s**2 / 3 + 7 * s**4 / 180, F - 2)
    numerator = F * torch.expm1(1j * omega * b / a) + delta
    base = numerator / (1j * omega)
    return (base - base.mean(dim=1, keepdim=True)) * torch.sign(a)


def whole_spatial_features(a, b, x):
    signs = np.sign(a)
    features = np.tanh(x[:, None] * abs(a) + signs * b) - np.tanh(x)[:, None]
    return (features - features.mean(axis=1, keepdims=True)) * signs


def real_system(features, y, weights):
    A = features * np.sqrt(weights)[:, None]
    rhs = y * np.sqrt(weights)
    if np.iscomplexobj(A):
        A = np.concatenate((A.real, A.imag))
        rhs = np.concatenate((rhs.real, rhs.imag))
    return np.asarray(A, order="F"), np.asarray(rhs)


def solve_readout(features, y, weights):
    A, rhs = real_system(features, y, weights)
    U, singular_values, Vh = sla.svd(A, full_matrices=False, check_finite=False,
                                    lapack_driver="gesdd")
    keep = singular_values > RCOND * singular_values[0]
    coefficients = U[:, keep].T @ rhs
    v = Vh[keep].T @ (coefficients / singular_values[keep])
    projected = U[:, keep] @ coefficients - rhs
    r = A @ v - rhs
    stationarity = np.linalg.norm(A.T @ r) / (singular_values[0] * np.linalg.norm(rhs))
    normal_projection = np.linalg.norm(A.T @ projected) / (singular_values[0] * np.linalg.norm(rhs))
    if np.iscomplexobj(features):
        n = len(y)
        optimal_residual = (projected[:n] + 1j * projected[n:]) / np.sqrt(weights)
    else:
        optimal_residual = projected / np.sqrt(weights)
    return v, {"rank": int(sum(keep)), "stationarity": float(stationarity),
               "projection_stationarity": float(normal_projection),
               "readout_norm": float(np.linalg.norm(v)), "loss": float(0.5 * (projected @ projected)),
               "actual_loss": float(0.5 * (r @ r)), "optimal_residual": optimal_residual}


class Problem:
    def __init__(self, target, config, refinement=1):
        self.target, self.config = target, config
        self.whole = target == "gaussian_envelope"
        if self.whole:
            self.x, self.weights = frequency_quadrature(config["frequency_quadrature_order"] * refinement)
            self.y = whole.target_spectrum(np.pi * self.x, config)
        else:
            n = config["n_train"] * refinement
            self.x = -1 + (np.arange(n) + 0.5) * 2 / n
            self.weights = np.full(n, 1 / n)
            self.y = get_target(target).fn_numpy(self.x)
        self.tx = torch.tensor(self.x)
        self.tw = torch.tensor(self.weights)
        self.ty = torch.tensor(self.y)
        self.target_energy = float(np.sum(self.weights * abs(self.y)**2))

    def features(self, a, b):
        if self.whole:
            return whole_features(a, b, self.tx)
        values = torch.tanh(self.tx[:, None] * a + b)
        return torch.cat((values, torch.ones((len(self.x), 1), dtype=torch.float64)), dim=1)

    def initial(self, arm):
        arrays, _ = original.initial_arrays(self.config["resolution"], self.config["seed"], arm,
                                             halo=self.config["halo"])
        v = arrays["c"] if self.whole else np.r_[arrays["c"], arrays["d"]]
        return {"a": arrays["a"], "b": arrays["b"], "v": v}

    def objective(self, features, v, optimal_residual=None):
        r = features @ (v.to(features.dtype)) - self.ty
        if optimal_residual is not None:
            # Evaluate the envelope gradient with U U^T y-y. Forming A*v-y
            # can lose many digits when the least-squares readout is enormous.
            # The detached correction retains the physical geometry Jacobian.
            r = r + (torch.tensor(optimal_residual) - r).detach()
        # Avoid complex abs' undefined/underflowed phase at tiny residuals.
        power = r.real.square() + r.imag.square() if torch.is_complex(r) else r.square()
        return 0.5 * torch.sum(self.tw * power)


def train(problem, arm, method, steps=None, displayed_steps=None):
    config = problem.config
    total = config["steps"] if steps is None else steps
    chosen = (list(displayed_steps) if displayed_steps is not None
              else snapshot_steps(total, available=frame_steps(total)))
    initial = problem.initial(arm)
    params = {key: torch.nn.Parameter(torch.tensor(value)) for key, value in initial.items()}
    optimized = [params["a"], params["b"]] if method == "varpro" else list(params.values())
    optimizer = torch.optim.SGD(optimized, lr=config["learning_rate"])
    losses = np.empty(total + 1)
    ranks = np.full(total + 1, -1, dtype=int)
    stationarity = np.full(total + 1, np.nan)
    norms = np.empty(total + 1)
    frames = {key: np.empty((len(chosen), value.size)) for key, value in initial.items()}
    for step in range(total + 1):
        optimizer.zero_grad(set_to_none=True)
        features = problem.features(params["a"], params["b"])
        optimal_residual = None
        if method == "varpro":
            v, info = solve_readout(features.detach().numpy(), problem.y, problem.weights)
            with torch.no_grad():
                params["v"].copy_(torch.tensor(v))
            ranks[step], stationarity[step] = info["rank"], info["projection_stationarity"]
            optimal_residual = info["optimal_residual"]
        loss = problem.objective(features, params["v"], optimal_residual)
        losses[step] = loss.detach().item()
        norms[step] = torch.linalg.vector_norm(params["v"]).detach().item()
        if not np.isfinite(losses[step]):
            raise FloatingPointError(f"Nonfinite {problem.target}/{arm}/{method} at step {step}")
        if step in chosen:
            index = chosen.index(step)
            for key in params:
                frames[key][index] = params[key].detach().numpy()
        if step % 500 == 0:
            print(problem.target, arm, method, step,
                  f"relL2={np.sqrt(2 * losses[step] / problem.target_energy):.6g}", flush=True)
        if step < total:
            loss.backward()
            optimizer.step()
    return {"target": problem.target, "arm": arm, "method": method,
            "snapshot_steps": chosen, "parameters": frames, "loss": losses,
            "rank": ranks, "stationarity": stationarity, "readout_norm": norms}


def reused_gd(problem, arm, saved_original, saved_whole, displayed_steps=None):
    """Use the existing, unchanged GD trajectory wherever it already exists."""
    config = problem.config
    chosen = (list(displayed_steps) if displayed_steps is not None
              else snapshot_steps(config["steps"], available=frame_steps(config["steps"])))
    old_steps = frame_steps(config["steps"])
    indices = [old_steps.index(step) for step in chosen]
    if problem.whole and arm == "qi_zero":
        previous = next(case for case in saved_whole if case["initial_gamma"] == 16)
        parameters = {key: values[indices].copy() for key, values in previous["parameters"].items()}
    elif not problem.whole:
        previous = next(case for case in saved_original
                        if case["target"] == problem.target and case["arm"] == arm)
        old = previous["frame_parameters"]
        parameters = {"a": old["a"][indices].copy(), "b": old["b"][indices].copy(),
                      "v": np.concatenate((old["c"][indices], old["d"][indices]), axis=1)}
    else:
        return None
    n = config["steps"] + 1
    norms = np.full(n, np.nan)
    norms[chosen] = np.linalg.norm(parameters["v"], axis=1)
    return {"target": problem.target, "arm": arm, "method": "gd",
            "snapshot_steps": chosen, "parameters": parameters, "loss": previous["loss"].copy(),
            "rank": np.full(n, -1), "stationarity": np.full(n, np.nan), "readout_norm": norms}


def display_frequency_grid(config):
    # Xavier's broad whole-line tails have content far below k=1. Preserve it
    # even though the fixed, linear display also extends through the target bands.
    return np.unique(np.r_[np.linspace(0, config["max_mode"], config["frequency_points"]),
                           np.geomspace(1e-6, 1, 401)])


def diagnose(case, problem):
    """Refit copies of saved GD states; never write a refit into the trajectory."""
    config = problem.config
    k = display_frequency_grid(config)
    x_plot = np.linspace(-2 if problem.whole else -1, 2 if problem.whole else 1, 2049)
    n = config["n_eval"]
    x_eval = -1 + (np.arange(n) + .5) * 2 / n
    full_transform = FiniteIntervalTransform(x_eval, config["max_mode"], config["frequency_points"])
    extra_k = np.geomspace(1e-6, 1, 401)
    extra_phase = (full_transform.dx * np.exp(-1j * np.pi * extra_k[:, None] * x_eval)
                   if not problem.whole else None)
    fine = Problem(problem.target, config, refinement=2) if problem.whole else None
    modes = ["gd", "gd_refit"] if case["method"] == "gd" else ["varpro"]
    result = {mode: {"residual": [], "spectrum": [], "relative_l2": [],
                     "rank": [], "stationarity": [], "readout_norm": [],
                     "refit_v": [], "quadrature_difference": []} for mode in modes}
    for index in range(len(case["snapshot_steps"])):
        arrays = {key: value[index].copy() for key, value in case["parameters"].items()}
        before = {key: value.copy() for key, value in arrays.items()}
        with torch.no_grad():
            A = problem.features(torch.tensor(arrays["a"]), torch.tensor(arrays["b"])).numpy()
        if case["method"] == "gd":
            refit, info = solve_readout(A, problem.y, problem.weights)
        else:
            refit = arrays["v"]
            _, info = solve_readout(A, problem.y, problem.weights)
        if problem.whole:
            spatial = whole_spatial_features(arrays["a"], arrays["b"], x_plot)
            target_plot = whole.target(x_plot, config)
            with torch.no_grad():
                dense = whole_features(torch.tensor(arrays["a"]), torch.tensor(arrays["b"]),
                                       torch.tensor(k[1:])).numpy()
                fine_A = fine.features(torch.tensor(arrays["a"]), torch.tensor(arrays["b"])).numpy()
            signs = np.sign(arrays["a"])
            z = -arrays["b"] / arrays["a"]
            dc = -2 * (z - z.mean()) * signs
            spectral = np.concatenate((dc[None], dense))
        else:
            spatial = np.column_stack((np.tanh(x_plot[:, None] * arrays["a"] + arrays["b"]), np.ones(len(x_plot))))
            eval_A = np.column_stack((np.tanh(x_eval[:, None] * arrays["a"] + arrays["b"]), np.ones(n)))
            target_plot = get_target(problem.target).fn_numpy(x_plot)
            target_eval = get_target(problem.target).fn_numpy(x_eval)
        for mode in modes:
            v = arrays["v"] if mode == "gd" else refit
            r = spatial @ v - target_plot
            if problem.whole:
                spectrum = spectral @ v - whole.target_spectrum(np.pi * k, config)
                eval_r = fine_A @ v - fine.y
                energy = float(np.sum(fine.weights * abs(eval_r)**2))
                relative = np.sqrt(energy / whole.target_energy(config))
                coarse = float(np.sum(problem.weights * abs(A @ v - problem.y)**2))
                difference = abs(energy - coarse) / whole.target_energy(config)
                assert difference < 1e-5, f"Whole-line quadrature unresolved: {difference}"
            else:
                eval_r = eval_A @ v - target_eval
                relative = np.linalg.norm(eval_r) / np.linalg.norm(target_eval)
                # Evaluate the extra low-frequency grid directly, then combine
                # with the existing dense transform without interpolating it.
                regular = full_transform(eval_r)
                extra = extra_phase @ eval_r
                spectrum = np.empty(len(k), dtype=np.complex128)
                spectrum[np.searchsorted(k, full_transform.modes)] = regular
                spectrum[np.searchsorted(k, extra_k)] = extra
                difference = np.nan
            values = result[mode]
            values["residual"].append(r)
            values["spectrum"].append(spectrum)
            values["relative_l2"].append(relative)
            values["rank"].append(info["rank"] if mode != "gd" else -1)
            values["stationarity"].append(info["stationarity"] if mode != "gd" else np.nan)
            values["readout_norm"].append(np.linalg.norm(v))
            values["refit_v"].append(v.copy())
            values["quadrature_difference"].append(difference)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], before[key])
    return {mode: {key: np.asarray(value) for key, value in values.items()}
            for mode, values in result.items()}


def save_cases(cases, config, path):
    payload = {"config_json": json.dumps(config), "rcond": RCOND}
    metadata = []
    for i, case in enumerate(cases):
        metadata.append({key: case[key] for key in ("target", "arm", "method", "snapshot_steps")})
        for key in ("loss", "rank", "stationarity", "readout_norm"):
            payload[f"case{i}_{key}"] = case[key]
        for key, values in case["parameters"].items():
            payload[f"case{i}_parameter_{key}"] = values
        for mode, view in case.get("views", {}).items():
            for key, values in view.items():
                payload[f"case{i}_view_{mode}_{key}"] = values
    payload["metadata_json"] = json.dumps(metadata)
    partial = path.with_suffix(".npz.partial")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    partial.replace(path)


def load_cases(path):
    cases = []
    with np.load(path, allow_pickle=False) as data:
        config = json.loads(data["config_json"].item())
        for i, metadata in enumerate(json.loads(data["metadata_json"].item())):
            case = {**metadata, "parameters": {}, "views": {}}
            for key in ("loss", "rank", "stationarity", "readout_norm"):
                case[key] = data[f"case{i}_{key}"]
            for key in ("a", "b", "v"):
                case["parameters"][key] = data[f"case{i}_parameter_{key}"]
            for mode in ("gd", "gd_refit", "varpro"):
                prefix = f"case{i}_view_{mode}_"
                view = {key[len(prefix):]: data[key] for key in data.files if key.startswith(prefix)}
                if view:
                    case["views"][mode] = view
            cases.append(case)
    return cases, config


def plot_comparison(cases, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(case["target"], case["arm"], mode): (case, view)
              for case in cases for mode, view in case["views"].items()}
    saved_steps = cases[0]["snapshot_steps"]
    chosen = snapshot_steps(config["steps"], available=saved_steps)
    indices = [saved_steps.index(step) for step in chosen]
    k = display_frequency_grid(config)
    titles = {"gd": "Ordinary gradient descent", "varpro": "Variable projection · readout solved before every geometry step",
              "gd_refit": "Ordinary GD · least-squares readout used only for evaluation"}
    # Preserve absolute comparisons across initializations AND across the three
    # figures. Tiny refitted curves really do lie near zero on these axes.
    limits = {}
    for target in config["targets"]:
        views = [view for (name, _, _), (_, view) in lookup.items() if name == target]
        limits[target] = (1.08 * max(abs(view["residual"]).max() for view in views),
                          1.08 * max(abs(view["spectrum"]).max() for view in views))
    for mode, title in titles.items():
        fig, axes = plt.subplots(4, 4, figsize=(18, 14), dpi=160)
        snapshot_legend(fig, chosen, y=0.942)
        colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(chosen)))
        for row, target in enumerate(config["targets"]):
            half = 2 if target == "gaussian_envelope" else 1
            x = np.linspace(-half, half, 2049)
            for arm_index, arm in enumerate(config["arms"]):
                case, view = lookup[target, arm, mode]
                left, right = axes[row, 2 * arm_index:2 * arm_index + 2]
                for index, color in zip(indices, colors):
                    left.plot(x, view["residual"][index], color=color, lw=1.25)
                    right.plot(k, abs(view["spectrum"][index]), color=color, lw=1.25)
                left.set(xlim=(-half, half), ylim=(-limits[target][0], limits[target][0]))
                right.set(xlim=(0, config["max_mode"]), ylim=(0, limits[target][1]),
                          xticks=np.arange(0, config["max_mode"] + 1, 8))
                if target == "gaussian_envelope":
                    # The initial Xavier residual has a large peak near k=.001.
                    # Retain the complete, matched main axes and expose the
                    # target bands in a shared linear zoom instead of clipping
                    # the low-frequency peak out of the displayed spectrum.
                    zoom = right.inset_axes([0.25, 0.30, 0.70, 0.57])
                    for index, color in zip(indices, colors):
                        zoom.plot(k, abs(view["spectrum"][index]), color=color, lw=1)
                    zoom.set(xlim=(0, 18), ylim=(0, 0.55), xticks=[2, 6, 14], yticks=[0, 0.25, 0.5])
                    zoom.set_title("Target bands · linear zoom", fontsize=8, pad=3)
                    zoom.tick_params(labelsize=7, pad=1)
                    zoom.grid(alpha=0.15)
                    for spine in zoom.spines.values():
                        spine.set_color("#9da3aa")
                text = f"Relative L₂: {view['relative_l2'][0]:.3g} → {view['relative_l2'][-1]:.3g}"
                if mode != "gd":
                    text += f" · final rank {view['rank'][-1]}"
                for ax in (left, right):
                    ax.text(0.5, 1.04, text, ha="center", transform=ax.transAxes, fontsize=9)
                if arm_index == 0:
                    domain = "on ℝ" if target == "gaussian_envelope" else "on [−1, 1]"
                    left.set_ylabel(LABELS[target] + f"\n{domain}\nResidual", fontsize=11)
                else:
                    left.set_ylabel("Residual", fontsize=10)
                right.set_ylabel("Fourier magnitude", fontsize=10)
                for ax in (left, right):
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.grid(alpha=0.17)
                    ax.tick_params(labelsize=9, labelbottom=row == 3)
            for column in (0, 2):
                axes[-1, column].set_xlabel("x (Gaussian row: central region)", fontsize=10)
            for column in (1, 3):
                axes[-1, column].set_xlabel(r"Physical frequency $k=\omega/\pi$", fontsize=11)
        for ax, label in zip(axes[0], ("Xavier · residual", "Xavier · spectrum", "QI · residual", "QI · spectrum")):
            ax.set_title(label, fontsize=14, pad=34)
        fig.suptitle(title, fontsize=20, y=0.987)
        fig.text(0.5, 0.960,
                 f"N = {config['resolution']} · halo {config['halo']} per side · {config['steps']:,} steps · GD rate {config['learning_rate']:g} · linear axes",
                 ha="center", fontsize=11)
        fig.text(0.5, 0.021,
                 "Axis limits match within each function across both initializations and all three figures. Spectra show the actual evaluated residual.\n"
                 "Sine, mixed sine, Runge: original finite-interval MLP. Gaussian: cancelling tails and whole-line loss/transform. "
                 "Readout SVD cutoff: 10⁻¹³.\n"
                 "Gaussian insets enlarge the target bands; the full axes retain Xavier's large initial peak near zero frequency.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=0.065, right=0.985, top=0.832, bottom=0.105,
                            hspace=0.37, wspace=0.30)
        fig.savefig(output / f"{mode}.png")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "readout_comparison.yaml")
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "data.npz"
    if args.plot_only:
        cases, config = load_cases(path)
    else:
        config = yaml.safe_load(args.config.read_text())
        torch.set_num_threads(config["threads"])
        torch.use_deterministic_algorithms(True)
        saved_original, _, _, _ = original.load_data(original.RESULTS / "data.npz")
        saved_whole, _ = whole.load_data(original.RESULTS / "whole_line/data.npz")
        cases = []
        if path.exists():
            previous, old_config = load_cases(path)
            if old_config == config:
                cases = previous
        done = {(case["target"], case["arm"], case["method"]) for case in cases}
        for target in config["targets"]:
            problem = Problem(target, config)
            for arm in config["arms"]:
                for method in ("gd", "varpro"):
                    if (target, arm, method) in done:
                        continue
                    case = reused_gd(problem, arm, saved_original, saved_whole) if method == "gd" else None
                    if case is None:
                        case = train(problem, arm, method)
                    case["views"] = diagnose(case, problem)
                    cases.append(case)
                    save_cases(cases, config, path)
                    for mode, view in case["views"].items():
                        print("Evaluated", target, arm, mode, "final relative L2", view["relative_l2"][-1], flush=True)
    plot_comparison(cases, config, args.output)
    print(f"Saved the three comparison figures in {args.output}", flush=True)


if __name__ == "__main__":
    main()
