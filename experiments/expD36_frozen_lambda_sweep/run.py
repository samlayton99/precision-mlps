"""Frozen tanh dictionaries, ordinary GD/Adam, reusable FP64 snapshots.

Training evaluates B.T @ (B @ c - y) directly, never rounded normal equations.
The four targets and two optimizers share a matrix multiply, but each column
has independent coefficients/moments. Geometry and all rates remain fixed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import scipy.linalg as sla
import torch
from threadpoolctl import threadpool_limits
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep"
TARGET_EQUATIONS = {
    "sine": "sqrt(2)*sin(2*pi*x)",
    "quadratic": "sqrt(5)*x**2",
    "mixed": "(sin(2*pi*x)+0.1*sin(20*pi*x))/sqrt(0.505)",
    "runge": "1/(1+25*x**2)",
}


def configuration():
    return yaml.safe_load((HERE / "config.yaml").read_text())


def fingerprint(cfg):
    # A larger horizon is a continuation, not a change to the constant-rate run.
    immutable = {k: v for k, v in cfg.items()
                 if k not in {"steps", "snapshot_steps", "torch_threads"}}
    return hashlib.sha256(json.dumps(immutable, sort_keys=True).encode()).hexdigest()


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def save_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n")
    os.replace(temporary, path)


def target_values(x, names):
    functions = {
        "sine": lambda: math.sqrt(2) * torch.sin(2 * math.pi * x),
        "quadratic": lambda: math.sqrt(5) * x.square(),
        "mixed": lambda: (torch.sin(2 * math.pi * x)
                           + .1 * torch.sin(20 * math.pi * x)) / math.sqrt(.505),
        "runge": lambda: 1 / (1 + 25 * x.square()),
    }
    return torch.stack([functions[name]() for name in names], dim=-1)


def midpoint_grid(size):
    return -1 + (2 / size) * (torch.arange(size, dtype=torch.float64) + .5)


@dataclass
class Problem:
    cfg: dict
    centers: torch.Tensor
    lambdas: np.ndarray
    gammas: np.ndarray
    x_train: torch.Tensor
    x_eval: torch.Tensor
    A: torch.Tensor
    E: torch.Tensor
    B: torch.Tensor
    BT: torch.Tensor
    y_train: torch.Tensor
    y_eval: torch.Tensor
    y_train_pair: torch.Tensor
    y_eval_pair: torch.Tensor
    y_normalized_pair: torch.Tensor
    gd_rates: torch.Tensor
    reference: dict
    readout_map: torch.Tensor | None = None
    mapped_A: torch.Tensor | None = None
    mapped_E: torch.Tensor | None = None


def make_problem(cfg):
    torch.set_default_dtype(torch.float64)
    if cfg["coordinates"] != "raw" or cfg["initial_readout"] != "zero":
        raise ValueError("This first-phase runner implements raw, zero-start readouts only")
    if cfg["schedule"] != "constant":
        raise ValueError("Continuation requires the implemented constant schedule")
    h = 2 / cfg["n"]
    centers = -1 + h * torch.arange(-cfg["halo_per_side"],
                                    cfg["n"] + cfg["halo_per_side"] + 1)
    gamma_x = cfg["xavier_gain"] * math.sqrt(2 / (len(centers) + 1))
    if "lambda_values" in cfg:
        # An explicit subset permits new cases without retraining the sweep.
        lambdas = np.asarray(cfg["lambda_values"], dtype=float)
    else:
        lambdas = np.geomspace(h * gamma_x, cfg["lambda_max"], cfg["lambda_count"])
        if len(lambdas) < 3 or not lambdas[0] < cfg["lambda_reference"] < lambdas[-1]:
            raise ValueError("Reference lambda must replace an interior point")
        nearest = 1 + np.argmin(np.abs(np.log(lambdas[1:-1] / cfg["lambda_reference"])))
        lambdas[nearest] = cfg["lambda_reference"]
        lambdas = np.r_[lambdas, cfg.get("additional_lambdas", [])]
    if not len(lambdas) or not np.isfinite(lambdas).all() or np.any(lambdas <= 0):
        raise ValueError("Lambdas must be finite and positive")
    lambdas = np.unique(lambdas)
    gammas = lambdas / h
    x_train, x_eval = midpoint_grid(cfg["n_train"]), midpoint_grid(cfg["n_eval"])

    def features(x):
        hidden = torch.tanh(torch.from_numpy(gammas)[:, None, None]
                            * (x[None, :, None] - centers[None, None, :]))
        return torch.cat((torch.ones((len(lambdas), len(x), 1)), hidden), dim=2)

    A, E = features(x_train), features(x_eval)
    y_train, y_eval = target_values(x_train, cfg["targets"]), target_values(x_eval, cfg["targets"])
    B = A / math.sqrt(len(x_train))
    y_norm = y_train / math.sqrt(len(x_train))
    reference = {key: [] for key in ("coefficients", "singular_values", "rank",
                                     "train_rel_l2", "eval_rel_l2", "normal_residual")}
    rates = []
    for i in range(len(lambdas)):
        u, s, vt = sla.svd(B[i].numpy(), full_matrices=False,
                          check_finite=False, lapack_driver="gesdd")
        keep = s > cfg["readout_rcond"] * s[0]
        c = torch.from_numpy(vt[keep].T @ ((u[:, keep].T @ y_norm.numpy()) / s[keep, None]))
        r_train, r_eval = A[i] @ c - y_train, E[i] @ c - y_eval
        reference["coefficients"].append(c.numpy())
        reference["singular_values"].append(s)
        reference["rank"].append(int(keep.sum()))
        reference["train_rel_l2"].append((torch.linalg.vector_norm(r_train, dim=0)
                                          / torch.linalg.vector_norm(y_train, dim=0)).numpy())
        reference["eval_rel_l2"].append((torch.linalg.vector_norm(r_eval, dim=0)
                                         / torch.linalg.vector_norm(y_eval, dim=0)).numpy())
        reference["normal_residual"].append((torch.linalg.vector_norm(B[i].T @ (r_train / math.sqrt(len(x_train))), dim=0)).numpy())
        rates.append(cfg["gd_rate_multiplier"] / s[0]**2)
    reference = {key: np.asarray(value) for key, value in reference.items()}
    return Problem(cfg, centers, lambdas, gammas, x_train, x_eval, A, E, B,
                   B.transpose(1, 2).contiguous(), y_train, y_eval,
                   torch.cat((y_train, y_train), dim=1),
                   torch.cat((y_eval, y_eval), dim=1),
                   torch.cat((y_norm, y_norm), dim=1),
                   torch.tensor(rates)[:, None, None], reference)


def zero_state(p):
    k, d, f = len(p.lambdas), p.B.shape[2], len(p.cfg["targets"])
    return {"c": torch.zeros(k, d, 2 * f), "adam_m": torch.zeros(k, d, f),
            "adam_v": torch.zeros(k, d, f), "step": 0}


@torch.no_grad()
def advance(p, state, count):
    c, m, v = state["c"], state["adam_m"], state["adam_v"]
    f = len(p.cfg["targets"])
    beta1, beta2, eps = (p.cfg[k] for k in ("adam_beta1", "adam_beta2", "adam_eps"))
    for step in range(state["step"] + 1, state["step"] + count + 1):
        residual = torch.bmm(p.B, c) - p.y_normalized_pair
        gradient = torch.bmm(p.BT, residual)
        c[:, :, :f].sub_(p.gd_rates * gradient[:, :, :f])
        g = gradient[:, :, f:]
        m.mul_(beta1).add_(g, alpha=1-beta1)
        v.mul_(beta2).addcmul_(g, g, value=1-beta2)
        denom = v.sqrt().div_(math.sqrt(1-beta2**step)).add_(eps)
        c[:, :, f:].addcdiv_(m, denom, value=-p.cfg["adam_lr"]/(1-beta1**step))
    state["step"] += count


@torch.no_grad()
def evaluate(p, state):
    def error(matrix, y):
        residual = torch.bmm(matrix, state["c"]) - y
        return (torch.linalg.vector_norm(residual, dim=1)
                / torch.linalg.vector_norm(y, dim=0)[None, :])
    k, f = len(p.lambdas), len(p.cfg["targets"])
    # Public metric axes: optimizer, lambda, function.
    convert = lambda a: a.numpy().reshape(k, 2, f).transpose(1, 0, 2)
    train_matrix = p.A if p.mapped_A is None else p.mapped_A
    eval_matrix = p.E if p.mapped_E is None else p.mapped_E
    return convert(error(train_matrix, p.y_train_pair)), convert(error(eval_matrix, p.y_eval_pair))


def physical_coefficients(p, state):
    """Checkpoint c is the optimizer variable; saved snapshots are physical."""
    return state["c"] if p.readout_map is None else p.readout_map @ state["c"]


def evaluation_steps(cfg, start, stop):
    points = set(range(0, min(stop, cfg["dense_eval_until"]) + 1))
    points.update(range(0, stop + 1, cfg["eval_every"]))
    points.update(s for s in cfg["snapshot_steps"] if s <= stop)
    points.add(stop)
    return sorted(s for s in points if s > start)


def fresh_history(p, state):
    train, evaluation = evaluate(p, state)
    return {"steps": [0], "train_rel_l2": [train], "eval_rel_l2": [evaluation],
            "coefficient_steps": [0], "coefficient_snapshots": [physical_coefficients(p, state).numpy().copy()]}


def checkpoint(path, p, state, history):
    error = np.asarray(history["eval_rel_l2"])
    save_npz(path, config_sha256=fingerprint(p.cfg), step=state["step"],
             c=state["c"].numpy(), adam_m=state["adam_m"].numpy(), adam_v=state["adam_v"].numpy(),
             physical_c=physical_coefficients(p, state).numpy(),
             **{key: np.asarray(value) for key, value in history.items()},
             best_recorded_index=np.argmin(error, axis=0))


def resume(path, p):
    if not path.exists():
        state = zero_state(p)
        return state, fresh_history(p, state)
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["config_sha256"]) != fingerprint(p.cfg):
            raise ValueError("Configuration changed: use a new output directory")
        state = {key: torch.from_numpy(saved[key].copy()) for key in ("c", "adam_m", "adam_v")}
        state["step"] = int(saved["step"])
        history = {key: list(saved[key]) for key in ("steps", "train_rel_l2", "eval_rel_l2",
                                                    "coefficient_steps", "coefficient_snapshots")}
    return state, history


def record_problem(p, output):
    map_arrays = {} if p.readout_map is None else {"readout_map": p.readout_map.numpy()}
    save_npz(output / "data/reference.npz", **p.reference, lambdas=p.lambdas, gammas=p.gammas,
             centers=p.centers.numpy(), x_train=p.x_train.numpy(), x_eval=p.x_eval.numpy(),
             y_train=p.y_train.numpy(), y_eval=p.y_eval.numpy(), gd_rates=p.gd_rates.numpy().ravel(), **map_arrays)
    metadata = {"config": p.cfg, "config_sha256": fingerprint(p.cfg),
                "targets": {k: TARGET_EQUATIONS[k] for k in p.cfg["targets"]},
                "h": 2/p.cfg["n"], "neurons_including_halo": len(p.centers),
                "readout_parameters_including_bias": p.B.shape[2],
                "lambdas": p.lambdas.tolist(), "gammas": p.gammas.tolist(),
                "gd_rates": p.gd_rates.numpy().ravel().tolist(),
                "training_loss": "0.5 * mean((prediction-target)**2)",
                "metric": "norm(prediction-target)/norm(target) on independent evaluation midpoints",
                "geometry": "uniform centers, identical positive slopes, all hidden parameters frozen",
                "grid": "base geometric grid with exact reference lambda, plus explicitly appended values; explicit lambda_values overrides the grid",
                "initial_gamma": "PR Xavier slope RMS: tanh gain * sqrt(2/(W+1))",
                "method": "direct FP64 residual and transpose products; no Gram matrix, momentum, decay, or line search",
                "metric_axes": ["snapshot", "optimizer: GD/Adam", "lambda", "function"],
                "coefficient_axes": ["snapshot", "lambda", "parameter: bias first", "GD functions then Adam functions"],
                "coefficient_storage": "coefficient_snapshots and physical_c are physical readouts; checkpoint c and Adam moments are optimizer coordinates",
                "reference": "truncated SVD of training-normalized physical features; same cutoff for all cases",
                "best": "minimum over recorded evaluation snapshots, separately for every optimizer/lambda/function",
                "torch_version": torch.__version__, "numpy_version": np.__version__}
    save_json(output / "data/metadata.json", metadata)


def train(p, output, horizon):
    path = output / "data/trajectory.npz"
    state, history = resume(path, p)
    if state["step"] > horizon:
        raise ValueError("Requested horizon precedes saved state")
    started = time.perf_counter()
    next_save = (state["step"]//p.cfg["checkpoint_every"] + 1)*p.cfg["checkpoint_every"]
    for step in evaluation_steps(p.cfg, state["step"], horizon):
        advance(p, state, step-state["step"])
        train_error, eval_error = evaluate(p, state)
        if not np.isfinite(eval_error).all() or not torch.isfinite(state["c"]).all():
            raise FloatingPointError(f"Nonfinite state at step {step}; prior checkpoint preserved")
        history["steps"].append(step)
        history["train_rel_l2"].append(train_error)
        history["eval_rel_l2"].append(eval_error)
        if step % p.cfg["coefficients_every"] == 0 or step == horizon:
            history["coefficient_steps"].append(step)
            history["coefficient_snapshots"].append(physical_coefficients(p, state).numpy().copy())
        if step >= next_save or step == horizon:
            checkpoint(path, p, state, history)
            elapsed = time.perf_counter()-started
            save_json(output / "data/status.json", {"step": step, "requested_horizon": horizon,
                      "complete": step == horizon, "invocation_seconds": elapsed})
            runs = 2*len(p.lambdas)*len(p.cfg["targets"])
            print(f"Saved all {runs} runs at step {step:,}; elapsed {elapsed:.1f}s", flush=True)
            next_save += p.cfg["checkpoint_every"]
    return state, history


def benchmark(p, count):
    state = zero_state(p)
    advance(p, state, 10)
    started = time.perf_counter()
    advance(p, state, count)
    step_seconds = (time.perf_counter()-started)/count
    started = time.perf_counter()
    for _ in range(5):
        evaluate(p, state)
    evaluation_seconds = (time.perf_counter()-started)/5
    n_evals = len(evaluation_steps(p.cfg, 0, p.cfg["steps"]))
    result = {"seconds_per_full_64_run_step": step_seconds,
              "seconds_per_evaluation": evaluation_seconds,
              "estimated_training_and_evaluation_minutes":
                  (step_seconds*p.cfg["steps"] + evaluation_seconds*n_evals)/60}
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int)
    parser.add_argument("--benchmark", type=int, default=0)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    cfg = configuration()
    if args.steps is not None:
        cfg["steps"] = args.steps
    torch.set_num_threads(cfg["torch_threads"])
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=cfg["torch_threads"]):
        if not args.plot_only:
            saved_path = args.output / "data/trajectory.npz"
            if saved_path.exists():
                with np.load(saved_path, allow_pickle=False) as saved:
                    if str(saved["config_sha256"]) != fingerprint(cfg):
                        raise ValueError("Configuration changed: use a new output directory")
            p = make_problem(cfg)
            if args.benchmark:
                benchmark(p, args.benchmark)
                return
            record_problem(p, args.output)
            train(p, args.output, cfg["steps"])
        from plot import plot_all
        plot_all(args.output)


if __name__ == "__main__":
    main()
