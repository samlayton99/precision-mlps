#!/usr/bin/env python3
"""Width sweep for CD-RGE with target-independent feature whitening.

This experiment intentionally contains no least-squares coefficient solve.
The only target-dependent parameter update is a batched, two-sided random
finite-difference update following Chaubard's CD-RGE/1SPSA rule.  Whitening is
constructed from the initialized feature matrix alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WIDTHS = (8, 16, 32, 64, 128, 256, 512, 1024)
GAMMAS = (0.5, 1.0, 2.0, 4.0, 6.0, 7.5, 8.0, 8.5, 9.0, 10.0,
          12.0, 16.0, 32.0, 64.0, 128.0, 256.0)


def geometry(width: int, gamma: float, device: torch.device):
    # Preserve the earlier R=4 construction once width permits it, while
    # retaining at least five interior intervals in the requested width-8 run.
    halo = min(4, max(1, (width - 6) // 2))
    intervals = width - 2 * halo - 1
    h = 2.0 / intervals
    centers = -1.0 + torch.arange(
        -halo, intervals + halo + 1, dtype=torch.float64, device=device
    ) * h
    assert centers.numel() == width
    return centers, h, gamma, halo


def design(x: torch.Tensor, centers: torch.Tensor, gamma: float):
    phi = torch.tanh(gamma * (x[:, None] - centers[None, :]))
    phi = phi - torch.tanh(gamma * (-1.0 - centers))[None, :]
    return torch.column_stack((torch.ones_like(x), phi))


def whitening_transform(a: torch.Tensor, method: str, block_size: int,
                        rcond: float):
    """Return T such that optimization uses a @ T; never inspect targets."""
    n, d = a.shape
    root_n = math.sqrt(n)
    transform = torch.zeros((d, d), dtype=a.dtype, device=a.device)
    blocks = [(0, d)] if method == "full" else [
        (start, min(start + block_size, d)) for start in range(0, d, block_size)
    ]
    retained = 0
    columns = []
    for start, end in blocks:
        _, singular, vh = torch.linalg.svd(a[:, start:end], full_matrices=False)
        keep = singular > rcond * singular[0]
        local = vh[keep].T @ torch.diag(root_n / singular[keep])
        embedded = torch.zeros((d, local.shape[1]), dtype=a.dtype, device=a.device)
        embedded[start:end] = local
        columns.append(embedded)
        retained += local.shape[1]
    return torch.column_stack(columns), retained


def cdrge(loss_batch, initial: torch.Tensor, *, steps: int, npert: int, eps0: float,
          halve_every: int, beta1: float, seed: int, device: torch.device,
          log_every: int = 64):
    """Batched faithful CD-RGE update; loss_batch returns queried scalar losses."""
    gen = torch.Generator(device=device).manual_seed(seed)
    z = initial.detach().clone()
    dimension = z.numel()
    momentum = torch.zeros_like(z)
    best_z = z.clone()
    best_loss = float(loss_batch(z[None])[0])
    trace = [{"step": 0, "train_relative_mse": best_loss, "epsilon": eps0}]
    evaluations = 1
    for step in range(1, steps + 1):
        stage = (step - 1) // halve_every if halve_every > 0 else 0
        eps = eps0 * (0.5 ** stage)
        probes = torch.randint(
            0, 2, (npert, dimension), generator=gen, device=device,
            dtype=torch.int8,
        ).to(torch.float64).mul_(2).sub_(1)
        query = torch.cat((z[None] + eps * probes, z[None] - eps * probes), dim=0)
        queried = loss_batch(query)
        evaluations += 2 * npert
        difference = queried[:npert] - queried[npert:]
        # Upstream buffer contains -eps times the RGE gradient estimate.
        buffer = -(difference[:, None] * probes).sum(dim=0) / (2.0 * npert)
        momentum.mul_(beta1).add_(buffer, alpha=1.0 - beta1)
        z.add_(momentum, alpha=1.0)  # lr / epsilon = 1 throughout
        if step % log_every == 0 or step == steps:
            value = float(loss_batch(z[None])[0])
            evaluations += 1
            trace.append({"step": step, "train_relative_mse": value, "epsilon": eps})
            if math.isfinite(value) and value < best_loss:
                best_loss, best_z = value, z.clone()
            if not math.isfinite(value):
                break
    return best_z, best_loss, evaluations, trace


def run_cell(width: int, initial_gamma: float, method: str, steps: int, seed: int,
             args, device: torch.device):
    started = time.time()
    centers, h, gamma, halo = geometry(width, initial_gamma, device)
    n = max(129, 4 * width + 1)
    train_x = torch.linspace(-1.0, 1.0, n, dtype=torch.float64, device=device)
    val_x = -1.0 + 2.0 * (torch.arange(n, dtype=torch.float64, device=device) + 0.5) / n
    train_y = torch.sin(2.0 * math.pi * train_x)
    val_y = torch.sin(2.0 * math.pi * val_x)
    raw_train = design(train_x, centers, gamma)
    transform, rank = whitening_transform(
        raw_train, method, args.block_size, args.rcond
    )
    white_train = raw_train @ transform
    scale = train_y.square().mean().clamp_min(torch.finfo(torch.float64).tiny)

    initial = torch.zeros(rank, dtype=torch.float64, device=device)

    def predict_batch(x, params):
        return (design(x, centers, gamma) @ transform @ params.T).T

    def loss_batch(params):
        pred = predict_batch(train_x, params)
        return (pred - train_y[None, :]).square().mean(dim=1) / scale

    optimized, train_loss, evaluations, trace = cdrge(
        loss_batch, initial, steps=steps, npert=args.npert, eps0=args.epsilon,
        halve_every=args.halve_every, beta1=args.beta1,
        seed=seed, device=device,
    )
    val_pred = predict_batch(val_x, optimized[None])[0]
    val_l2 = float(torch.linalg.vector_norm(val_pred - val_y) /
                   torch.linalg.vector_norm(val_y))
    return {
        "width": width, "initial_lambda": gamma * h, "initial_gamma": gamma,
        "final_gamma": gamma,
        "final_lambda": gamma * h,
        "h": h,
        "halo": halo, "method": method, "block_size": args.block_size,
        "whiten_rcond": args.rcond, "retained_rank": rank,
        "train_count": n, "steps": steps, "npert": args.npert,
        "function_evaluations": evaluations, "best_train_relative_mse": train_loss,
        "validation_relative_l2": val_l2, "seed": seed,
        "elapsed_seconds": time.time() - started, "trace": trace,
        "optimized_parameters": optimized.detach().cpu().tolist(),
        "raw_output_coefficients": (
            transform @ optimized
        ).detach().cpu().tolist(),
    }


def evaluate_test(record, args, device):
    width, initial_gamma = record["width"], record["initial_gamma"]
    centers, _, gamma, _ = geometry(width, initial_gamma, device)
    n = record["train_count"]
    train_x = torch.linspace(-1.0, 1.0, n, dtype=torch.float64, device=device)
    raw_train = design(train_x, centers, gamma)
    transform, _ = whitening_transform(raw_train, record["method"],
                                        args.block_size, args.rcond)
    optimized = torch.tensor(record["optimized_parameters"], dtype=torch.float64, device=device)
    coord = optimized
    final_gamma = record["final_gamma"]
    test_n = 16385
    test_x = -1.0 + 2.0 * (torch.arange(test_n, dtype=torch.float64, device=device) + 0.371) / test_n
    test_y = torch.sin(2.0 * math.pi * test_x)
    pred = design(test_x, centers, final_gamma) @ transform @ coord
    record["test_count"] = test_n
    record["test_relative_l2"] = float(
        torch.linalg.vector_norm(pred - test_y) / torch.linalg.vector_norm(test_y)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", default=",".join(map(str, WIDTHS)))
    parser.add_argument("--gammas", default=",".join(map(str, GAMMAS)))
    parser.add_argument("--methods", default="full,partial")
    parser.add_argument("--selection-steps", type=int, default=1024)
    parser.add_argument("--final-steps", type=int, default=8192)
    parser.add_argument("--npert", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--halve-every", type=int, default=1024)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--rcond", type=float, default=0.0,
                        help="Relative SVD cutoff. Headline default keeps every numerical direction.")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    widths = [int(x) for x in args.widths.split(",")]
    gammas = [float(x) for x in args.gammas.split(",")]
    methods = args.methods.split(",")
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float64)
    RESULTS.mkdir(parents=True, exist_ok=True)

    trials, selected = [], []
    for method in methods:
        for width in widths:
            candidates = []
            for index, gamma in enumerate(gammas):
                row = run_cell(width, gamma, method, args.selection_steps,
                               args.seed + 1000 * width + index, args, device)
                row["phase"] = "validation_selection"
                trials.append(row)
                candidates.append(row)
                print(method, width, gamma, "validation", row["validation_relative_l2"], flush=True)
            chosen = min(candidates, key=lambda x: x["validation_relative_l2"])
            final = run_cell(width, chosen["initial_gamma"], method, args.final_steps,
                             args.seed + 100000 + width, args, device)
            final["phase"] = "selected_final"
            final["selection_validation_relative_l2"] = chosen["validation_relative_l2"]
            evaluate_test(final, args, device)
            selected.append(final)
            print("SELECTED", method, width, final["final_lambda"],
                  "test", final["test_relative_l2"], flush=True)

    payload = {
        "title": "ZO w/ Whitening",
        "target": "sin(2*pi*x) on [-1, 1]",
        "model": "one-hidden-layer fixed-tanh-feature MLP with affine readout",
        "protocol": vars(args),
        "audit": {
            "target_dependent_least_squares": False,
            "autograd_or_backpropagation": False,
            "target_used_to_construct_whitening": False,
            "target_dependent_optimizer": "CD-RGE two-sided random finite differences only",
            "gamma_selection": "gamma is swept directly and selected by midpoint-grid validation after a fixed 1024-step budget; lambda=gamma*h is derived only for reporting",
            "test_role": "evaluated only for the selected lambda after an independent final run",
            "full_whitening": "one global target-independent SVD of the feature matrix",
            "partial_whitening": "independent target-independent SVDs of contiguous feature blocks",
        },
        "trials": trials,
        "selected": selected,
    }
    (RESULTS / "zo_whitening_width_sweep.json").write_text(json.dumps(payload, indent=2) + "\n")
    fields = ["method", "width", "initial_lambda", "initial_gamma",
              "final_lambda", "final_gamma", "h", "retained_rank",
              "train_count", "steps", "npert", "function_evaluations",
              "best_train_relative_mse", "validation_relative_l2",
              "test_relative_l2", "elapsed_seconds"]
    with (RESULTS / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row[key] for key in fields})


if __name__ == "__main__":
    main()
