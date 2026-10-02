"""Constructive QI versus Xavier, ordinary Adam, and observational readout solves.

The candidate activations are hypotheses, not a preassigned empirical ranking.
Every diagnostic uses the current geometry and never injects solved readouts.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/expd41-mpl")

import numpy as np
import scipy
from scipy import linalg
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expD41_activation_lens.activations import activation_np as base_activation_np, activation_torch as base_activation_torch, derivative_np as base_derivative_np
from src.data.targets import get_target
from src.training.train_loop import train_step

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/checkpoint_D_optimizers/expD41_activation_lens"
_local_cache = None


def local_activation():
    global _local_cache
    if _local_cache is None:
        from experiments.expD41_activation_lens.local_activation import LocalActivation, load_design
        _local_cache = LocalActivation(load_design(OUT / "local_design.json"))
    return _local_cache


def activation_np(name, z):
    return local_activation().numpy(z) if name == "local" else base_activation_np(name, z)


def derivative_np(name, z):
    return local_activation().derivative_numpy(z) if name == "local" else base_derivative_np(name, z)


def activation_torch(name, z):
    return local_activation().torch(z) if name == "local" else base_activation_torch(name, z)


def all_activations(cfg):
    return cfg["activations"] + cfg.get("supplementary_activations", [])


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def grid(n):
    """Midpoint quadrature on [-1,1]; nested powers of two are disjoint."""
    return -1 + (np.arange(n, dtype=np.float64) + 0.5) * (2 / n)


def relative(pred, y):
    return float(np.linalg.norm(pred - y) / np.linalg.norm(y))


def geometry(cfg, lam):
    n, halo = cfg["interior_centers"], cfg["halo_per_side"]
    h = 2 / (n - 1)
    centers = -1 + h * np.arange(-halo, n + halo, dtype=np.float64)
    return centers, h, lam / h


def cardinal_stencil(activation, lam, cfg):
    """Target-independent finite cardinal filter, extending the repo constructor.

    T_rc = lambda sigma'(lambda(r-c)); T q = h delta_0.
    A fixed, explicit cutoff handles the new kernels' singular/weak directions.
    This is a derivative-kernel construction, not a fit to function values.
    """
    h = 2 / (cfg["interior_centers"] - 1)
    half = cfg["cardinal_half_width"]
    k = np.arange(-half, half + 1)
    matrix = lam * derivative_np(activation, lam * (k[:, None] - k[None, :]))
    rhs = np.zeros(k.size)
    rhs[half] = h
    q, _, rank, singular = linalg.lstsq(matrix, rhs, cond=cfg["cardinal_rcond"], lapack_driver="gelsd")
    return q, {
        "cardinal_rank": int(rank),
        "cardinal_residual": float(np.linalg.norm(matrix @ q - rhs) / h),
        "cardinal_norm": float(np.linalg.norm(q)),
        "cardinal_condition_retained": float(singular[0] / singular[rank - 1]),
    }


def construct(activation, target, lam, cfg, stencil=None, polish=False):
    """Derivative/cardinal readout plus boundary anchor; no function-value LS."""
    centers, h, gamma = geometry(cfg, lam)
    if activation == "tanh" and polish and cfg.get("selected_tanh_constructor_precision") == "mpmath":
        from src.construction.qi_mpmath import construct_qi
        target_fn = get_target(target)
        qi = construct_qi(target_fn.fn_numpy, target_fn.deriv_numpy,
            N=cfg["interior_centers"] - 1, halo=cfg["halo_per_side"],
            lambda_star=lam, Kc=cfg["cardinal_half_width"], precision="mpmath",
            mp_dps=cfg["selected_tanh_constructor_digits"])
        return (np.full(centers.size, qi.gamma), -qi.gamma * qi.centers, qi.a_coeffs, qi.c0), {
            "lambda": float(lam), "gamma": gamma, "h": h, "width": int(centers.size),
            "readout_construction": "repository derivative cardinal convolution; mpmath selected-point construction",
            "constructor_digits": cfg["selected_tanh_constructor_digits"],
            "cardinal_residual": qi.toeplitz_residual}
    q, info = cardinal_stencil(activation, lam, cfg) if stencil is None else stencil
    k = np.arange(-cfg["cardinal_half_width"], cfg["cardinal_half_width"] + 1)
    derivatives = get_target(target).deriv_numpy(centers[:, None] - h * k[None, :])
    v = derivatives @ q
    w = np.full(centers.size, gamma)
    b = -gamma * centers
    at_left = activation_np(activation, -w + b)
    bias = float(get_target(target).fn_numpy(np.array(-1.0))) - math.fsum((v * at_left).tolist())
    return (w, b, v, bias), {"lambda": float(lam), "gamma": gamma, "h": h,
        "width": int(centers.size), "readout_construction": "derivative cardinal convolution; left-boundary anchor",
        **info}


def features(activation, w, b, x):
    return activation_np(activation, x[:, None] * w[None, :] + b[None, :])


def readout_solve(phi, y, rcond):
    """Residualize a free bias before TSVD so constants remain unpenalized."""
    means, ymean = phi.mean(axis=0), y.mean()
    v, _, rank, singular = linalg.lstsq(phi - means, y - ymean, cond=rcond, lapack_driver="gelsd")
    bias = float(ymean - means @ v)
    return v, bias, {"rank": int(rank), "singular_max": float(singular[0]),
        "singular_min_retained": float(singular[rank - 1]) if rank else 0.0}


class Network(nn.Module):
    def __init__(self, activation, params):
        super().__init__()
        self.activation = activation
        w, b, v, bias = params
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float64))
        self.v = nn.Parameter(torch.tensor(v, dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float64))

    def forward(self, x):
        return activation_torch(self.activation, x[:, None] * self.w + self.b) @ self.v + self.bias

    def arrays(self):
        return tuple(p.detach().cpu().numpy().copy() for p in (self.w, self.b, self.v, self.bias))


def xavier(width, seed, cfg):
    """Canonical gain-one Glorot uniform weights; all biases zero."""
    rng = torch.Generator().manual_seed(seed)
    bound = cfg["xavier_gain"] * math.sqrt(6 / (width + 1))
    w = torch.empty(width, dtype=torch.float64).uniform_(-bound, bound, generator=rng).numpy()
    v = torch.empty(width, dtype=torch.float64).uniform_(-bound, bound, generator=rng).numpy()
    return w, np.zeros(width), v, 0.0


def evaluate(activation, params, target, cfg, xeval=None):
    w, b, v, bias = params
    xt = grid(cfg["n_train"])
    xe = grid(cfg["n_eval"]) if xeval is None else xeval
    fn = get_target(target).fn_numpy
    yt, ye = fn(xt), fn(xe)
    phi = features(activation, w, b, xt)
    solved, sbias, info = readout_solve(phi, yt, cfg["readout_rcond"])
    phie = features(activation, w, b, xe)
    return {"actual_rel_l2": relative(phie @ v + bias, ye),
        "floor_rel_l2": relative(phie @ solved + sbias, ye),
        "train_actual_rel_l2": relative(phi @ v + bias, yt),
        "train_floor_rel_l2": relative(phi @ solved + sbias, yt),
        "readout_norm": float(np.linalg.norm(v)), "floor_readout_norm": float(np.linalg.norm(solved)),
        "mean_abs_gamma": float(np.mean(np.abs(w))), **info}, (solved, sbias)


def sweep(cfg, out):
    rows = []
    local_gram = {}
    for activation in all_activations(cfg):
        for lam in cfg["lambda_candidates"]:
            if activation == "local":
                from experiments.expD41_activation_lens.local_design import audit_scaled_gram
                local_gram[str(lam)] = audit_scaled_gram(local_activation().design, scale=lam)
            stencil = cardinal_stencil(activation, lam, cfg)
            for target in cfg["targets"]:
                params, info = construct(activation, target, lam, cfg, stencil)
                metrics, _ = evaluate(activation, params, target, cfg, grid(cfg["n_validation"]))
                rows.append({"activation": activation, "target": target, "lambda": lam,
                    "construction_rel_l2": metrics["actual_rel_l2"], "floor_rel_l2": metrics["floor_rel_l2"],
                    "construction_readout_norm": metrics["readout_norm"], **info})
            errors = [row["construction_rel_l2"] for row in rows[-len(cfg["targets"]):]]
            print(f"bandwidth {activation:5s} lambda={lam:g} construction={errors}", flush=True)
            save_json(out / "bandwidth_partial.json", {"rows": rows})
    selected = {}
    for activation in all_activations(cfg):
        eligible_lambdas = [lam for lam in cfg["lambda_candidates"] if activation != "local"
            or local_gram[str(lam)]["minimum_over_energy"] >= cfg["local_design"]["minimum_gram_ratio"]]
        if not eligible_lambdas:
            raise ValueError("No local bandwidth preserves the declared derivative Gram bound")
        scores = [(max(r["construction_rel_l2"] for r in rows if r["activation"] == activation and r["lambda"] == lam), lam)
                  for lam in eligible_lambdas]
        score, lam = min(scores)
        selected[activation] = {"lambda": lam, "score": score, "criterion": cfg["lambda_selection"]}
    result = {"rows": rows, "selected": selected, "local_scaled_gram": local_gram,
        "references": cfg["lambda_references"],
        "reference_caveat": "notch Gaussian-envelope reference is not a notch-specific bandwidth theorem",
        "selection_grid_size": cfg["n_validation"]}
    save_json(out / "bandwidth.json", result)
    print("selected " + json.dumps(selected), flush=True)
    return result


def recording_steps(total):
    return sorted(set(range(21)) | set(np.rint(np.geomspace(21, total, 85)).astype(int)) | {total})


def learning_rate(step, cfg):
    if step <= cfg["warmup_steps"]:
        return cfg["learning_rate"] * step / cfg["warmup_steps"]
    fraction = (step - cfg["warmup_steps"]) / (cfg["steps"] - cfg["warmup_steps"])
    return cfg["learning_rate_end"] + 0.5 * (cfg["learning_rate"] - cfg["learning_rate_end"]) * (1 + math.cos(math.pi * fraction))


def mse(model, x, y):
    return torch.mean((model(x) - y) ** 2)


def run_one(job):
    cfg, out_str, activation, target, initialization, seed, lam = job
    torch.set_num_threads(cfg["threads_per_run"])
    torch.manual_seed(seed)
    out = Path(out_str)
    name = f"{target}__{activation}__{initialization}__s{seed}"
    path = out / "runs" / f"{name}.json"
    signature = {"config": cfg, "activation": activation, "target": target,
        "initialization": initialization, "seed": seed, "lambda": lam,
        "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (HERE / "run.py", HERE / "activations.py")}}
    if activation == "local":
        signature["local_sources"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (HERE / "local_activation.py", OUT / "local_design.json")}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("fingerprint") != fingerprint:
            raise ValueError(f"Existing run has different config or source; preserve it and use a new output directory: {path}")
        if (previous.get("complete") and previous["history"][-1]["step"] == cfg["steps"]
                and path.with_suffix(".npz").exists() and path.with_suffix(".pt").exists()):
            return name + " already complete"
    began = time.perf_counter()
    if initialization == "qi":
        params, init_info = construct(activation, target, lam, cfg, polish=True)
    else:
        width = cfg["interior_centers"] + 2 * cfg["halo_per_side"]
        params = xavier(width, seed, cfg)
        init_info = {"seed": seed, "width": width, "gain": cfg["xavier_gain"], "biases": "zero"}
    model = Network(activation, params)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["adam_betas"]),
                           eps=cfg["adam_epsilon"], weight_decay=cfg["weight_decay"])
    x = torch.tensor(grid(cfg["n_train"]), dtype=torch.float64)
    y = torch.tensor(get_target(target).fn_numpy(x.numpy()), dtype=torch.float64)
    history, saved = [], {key: [] for key in ("step", "w", "b", "v", "bias", "solved_v", "solved_bias")}
    record_at = set(recording_steps(cfg["steps"]))
    result = {"activation": activation, "target": target, "initialization": initialization,
              "seed": seed, "initialization_info": init_info, "history": history, "complete": False,
              "fingerprint": fingerprint, "signature": signature}
    for step in range(cfg["steps"] + 1):
        if step:
            lr = learning_rate(step, cfg)
            for group in opt.param_groups:
                group["lr"] = lr
            loss = train_step(model, opt, x, y, mse)
            if not math.isfinite(loss):
                raise FloatingPointError(f"{name} nonfinite loss at step {step}")
        if step in record_at:
            params = model.arrays()
            metrics, solved = evaluate(activation, params, target, cfg)
            history.append({"step": step, "lr": learning_rate(max(1, step), cfg), **metrics})
            for key, value in zip(("w", "b", "v", "bias", "solved_v", "solved_bias"), (*params, *solved)):
                saved[key].append(value)
            saved["step"].append(step)
            result["wall_seconds"] = time.perf_counter() - began
            save_json(path, result)
            if step == 0 or step >= cfg["steps"] or step in (100, 1000):
                print(f"{name} step={step} actual={metrics['actual_rel_l2']:.3e} LS={metrics['floor_rel_l2']:.3e}", flush=True)
    np.savez_compressed(path.with_suffix(".npz"), **{k: np.asarray(v) for k, v in saved.items()})
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": cfg["steps"],
                "config": cfg, "activation": activation, "target": target, "initialization": initialization, "seed": seed},
               path.with_suffix(".pt"))
    result["complete"] = True
    result["wall_seconds"] = time.perf_counter() - began
    save_json(path, result)
    return f"{name}: {history[-1]['actual_rel_l2']:.3e}; LS={history[-1]['floor_rel_l2']:.3e}; {result['wall_seconds']:.1f}s"


def train(cfg, out):
    bandwidth = json.loads((out / "bandwidth.json").read_text())
    jobs = []
    for target in cfg["targets"]:
        for activation in all_activations(cfg):
            lam = bandwidth["selected"][activation]["lambda"]
            for initialization, seeds in (("qi", [0]), ("xavier", cfg["xavier_seeds"])):
                for seed in seeds:
                    jobs.append((cfg, str(out), activation, target, initialization, seed, lam))
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            print("complete " + future.result(), flush=True)


def validate(cfg, out):
    """Reconstruct saved endpoint readouts on a denser, disjoint grid."""
    results = []
    for path in sorted((out / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        if not run.get("complete"):
            raise ValueError(f"Incomplete run: {path}")
        arrays = np.load(path.with_suffix(".npz"))
        params = tuple(arrays[key][-1] for key in ("w", "b", "v", "bias"))
        xe = grid(cfg["n_verify"])
        ye = get_target(run["target"]).fn_numpy(xe)
        phi = features(run["activation"], *params[:2], xe)
        actual = relative(phi @ params[2] + params[3], ye)
        floor = relative(phi @ arrays["solved_v"][-1] + arrays["solved_bias"][-1], ye)
        row = {"run": path.stem, "actual_rel_l2_dense": actual, "floor_rel_l2_dense": floor,
            "actual_grid_ratio": actual / run["history"][-1]["actual_rel_l2"],
            "floor_grid_ratio": floor / run["history"][-1]["floor_rel_l2"], "cutoffs": {}}
        xt = grid(cfg["n_train"])
        yt = get_target(run["target"]).fn_numpy(xt)
        phit = features(run["activation"], *params[:2], xt)
        for cutoff in (1e-12, 1e-13, 1e-14):
            v, bias, info = readout_solve(phit, yt, cutoff)
            row["cutoffs"][str(cutoff)] = {"relative_l2": relative(phi @ v + bias, ye), **info}
        results.append(row)
    save_json(out / "validation.json", {"grid_size": cfg["n_verify"], "rows": results})
    print(f"Validated {len(results)} saved endpoints on {cfg['n_verify']} independent points", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sweep", "train", "validate", "plot", "all"])
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    torch.set_num_threads(cfg["threads_per_run"])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "runs").mkdir(exist_ok=True)
    if args.stage in ("sweep", "all"):
        save_json(args.out / "config.json", cfg)
        save_json(args.out / "environment.json", {"python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "executable": sys.executable,
            "sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")}})
        sweep(cfg, args.out)
    if args.stage in ("train", "all"):
        train(cfg, args.out)
    if args.stage in ("validate", "all"):
        validate(cfg, args.out)
    if args.stage in ("plot", "all"):
        from experiments.expD41_activation_lens.plot import render
        render(args.out)


if __name__ == "__main__":
    main()
