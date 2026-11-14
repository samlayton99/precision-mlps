"""QI or standard initialization followed by Adam; rates share initial weights."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import roots_legendre

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.construction import construct_qi
from src.data.targets import get_target
from src.training.train_loop import train_step

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/checkpoint_D_optimizers/expD55_qi_adam_drift"


def target(name):
    if name == "square":
        return (lambda x: x * x), (lambda x: 2 * x)
    t = get_target(name)
    return t.fn_numpy, t.deriv_numpy


def make_standard_model(width):
    """Unmodified torch.nn.Linear reset_parameters, in float64 from creation."""
    return torch.nn.Sequential(
        torch.nn.Linear(1, width, dtype=torch.float64),
        torch.nn.Tanh(),
        torch.nn.Linear(width, 1, dtype=torch.float64),
    )


def make_model(qi):
    net = make_standard_model(len(qi.centers))
    with torch.no_grad():
        net[0].weight.fill_(qi.gamma)
        net[0].bias.copy_(torch.from_numpy(-qi.gamma * qi.centers))
        net[2].weight.copy_(torch.from_numpy(qi.a_coeffs[None, :]))
        net[2].bias.fill_(qi.c0)
    return net


def lr_at(step, peak, warmup):
    """step is the one-based index of the update about to be taken."""
    return peak * min(step / warmup, 1.0)


def evaluation_steps(steps):
    return sorted({0, steps, *range(1, min(100, steps) + 1),
                   *range(125, steps + 1, 25)})


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float64).reshape(-1, 1)


def rel_l2(net, x, y, weights=None):
    with torch.no_grad():
        residual = net(x) - y
        if weights is None:
            return float(torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(y))
        return float(torch.sqrt((weights * residual.square()).sum() /
                                (weights * y.square()).sum()))


def mse(net, x, y):
    return (net(x) - y).square().mean()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def run_target(name, config, output):
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    output.mkdir(parents=True, exist_ok=True)
    f, df = target(name)
    # Midpoint rules are disjoint: the eval rule has twice as many points.
    xtr = tensor(-1 + (np.arange(config["train_points"]) + .5) * 2 / config["train_points"])
    xev = tensor(-1 + (np.arange(config["eval_points"]) + .5) * 2 / config["eval_points"])
    ytr, yev = tensor(f(xtr.numpy())), tensor(f(xev.numpy()))
    # Independent, nonuniform population-L2 cross-check, only at endpoints.
    xg, wg = roots_legendre(config["eval_points"])
    xg, yg, wg = tensor(xg), tensor(f(xg)), tensor(wg)
    initialization = config.get("initialization", "qi")
    q = None
    if initialization == "qi":
        q = construct_qi(f, df, **config["qi"])
        width = len(q.centers)
        initial = make_model(q)
    elif initialization == "standard":
        width = config["width"]
        initial = make_standard_model(width)
    else:
        raise ValueError(f"Unknown initialization: {initialization}")
    baseline = rel_l2(initial, xev, yev)
    if initialization == "qi" and baseline > config["initial_error_ceiling"]:
        raise RuntimeError(f"{name}: QI error {baseline} exceeds the prespecified ceiling")
    torch.save(initial.state_dict(), output / "initial.pt")
    grids = dict(train_x=xtr.numpy(), train_y=ytr.numpy(), eval_x=xev.numpy(), eval_y=yev.numpy())
    np.savez(output / "dataset.npz", **grids)
    if q is not None:
        np.savez(output / "qi.npz", centers=q.centers, a_coeffs=q.a_coeffs,
                 c0=q.c0, gamma=q.gamma, halo=q.halo, **grids)
    initial_grad_loss = mse(initial, xtr, ytr)
    initial_grad_loss.backward()
    grad_norm = float(torch.sqrt(sum(p.grad.square().sum() for p in initial.parameters())))
    initial.zero_grad(set_to_none=True)
    initial_info = {
        "target": name, "initialization": initialization, "width": width,
        "parameter_count": sum(p.numel() for p in initial.parameters()),
        "eval_rel_l2": baseline, "train_rel_l2": rel_l2(initial, xtr, ytr),
        "gauss_rel_l2": rel_l2(initial, xg, yg, wg), "gradient_l2": grad_norm,
    }
    if q is not None:
        initial_info.update(toeplitz_residual=q.toeplitz_residual, halo=q.halo)
    write_json(output / "initial.json", initial_info)
    print(f"{name}: {initialization} width={width}, initial relative L2={baseline:.3e}", flush=True)
    log_steps = set(evaluation_steps(config["steps"]))
    summaries = []
    for peak in config["learning_rates"]:
        start = time.monotonic()
        net = copy.deepcopy(initial)
        opt = torch.optim.Adam(net.parameters(), lr=peak, **config["adam"])
        run_id = f"lr_{peak:.0e}"
        history = [{"step": 0, "lr": 0.0, "eval_rel_l2": baseline,
                    "train_rel_l2": initial_info["train_rel_l2"]}]
        losses, lrs = [], []
        assert all(torch.equal(p, p0) for p, p0 in zip(net.parameters(), initial.parameters()))
        for step in range(1, config["steps"] + 1):
            current_lr = lr_at(step, peak, config["warmup_steps"])
            for group in opt.param_groups:
                group["lr"] = current_lr
            loss = train_step(net, opt, xtr, ytr, mse)
            if not np.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {name}, {peak}, step {step}")
            losses.append(loss)  # before this update: exact error at step-1
            lrs.append(current_lr)
            if step in log_steps:
                history.append({"step": step, "lr": current_lr,
                                "eval_rel_l2": rel_l2(net, xev, yev),
                                "train_rel_l2": rel_l2(net, xtr, ytr)})
            checkpoint_every = config.get("checkpoint_every", 0)
            if checkpoint_every and (step % checkpoint_every == 0 or step == config["steps"]):
                torch.save({"step": step, "target": name, "peak_lr": peak, "config": config,
                            "model": net.state_dict(), "optimizer": opt.state_dict(),
                            "torch_rng_state": torch.get_rng_state(), "history": history,
                            "pre_update_mse": torch.tensor(losses, dtype=torch.float64),
                            "learning_rate": torch.tensor(lrs, dtype=torch.float64)},
                           output / f"{run_id}_step_{step:06d}.pt")
            if step % config.get("progress_every", 2000) == 0:
                print(f"{name} lr={peak:.0e}: step={step}, rel L2={history[-1]['eval_rel_l2']:.3e}, "
                      f"elapsed={time.monotonic()-start:.1f}s", flush=True)
        final = history[-1]["eval_rel_l2"]
        summary = {"target": name, "learning_rate": peak, "steps": config["steps"],
                   "initial_rel_l2": baseline, "final_rel_l2": final,
                   "final_gauss_rel_l2": rel_l2(net, xg, yg, wg),
                   "final_to_initial_ratio": final / baseline,
                   "maximum_logged_rel_l2": max(h["eval_rel_l2"] for h in history),
                   "elapsed_seconds": time.monotonic() - start}
        write_json(output / f"{run_id}.json", {"summary": summary, "history": history})
        np.savez(output / f"{run_id}_all_steps.npz", pre_update_mse=losses, learning_rate=lrs,
                 step=np.arange(config["steps"]))
        torch.save(net.state_dict(), output / f"{run_id}_final.pt")
        summaries.append(summary)
    write_json(output / "summary.json", summaries)


def run_job(job):
    run_target(*job)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["square", "sine", "sine_mixture"])
    p.add_argument("--output", type=Path)
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if args.output is None:
        args.output = OUT / ("standard_init_warmup1000" if config.get("initialization") == "standard" else "warmup1000")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "config.json", config)
    write_json(args.output / "provenance.json", {
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_worktree_has_preexisting_changes": True,
    })
    jobs = [(name, config, args.output / name)
            for name in ([args.target] if args.target else config["targets"])]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            list(pool.map(run_job, jobs))
    else:
        for job in jobs:
            run_job(job)


if __name__ == "__main__":
    main()
