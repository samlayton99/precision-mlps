"""expF04/all20_2layers_uniform -- two-layer MLP with uniform ridge centers.

This is the all20_2layers protocol with one scientific change: both QI initializer
calls use uniform_centers=True. Outputs are isolated in all20_2layers_uniform.

Run:
    uv run python experiments/expF04_qi_init_real_data/all20_2layers_uniform/run.py
    uv run python experiments/expF04_qi_init_real_data/all20_2layers_uniform/run.py \
        --no-wandb --width 512
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ALL20 = PARENT / "all20"
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(ALL20))
from model import SimpleMLP2, qi_ridge_init_layer_  # noqa: E402
import datasets as DS  # noqa: E402

OUT_DIR = (
    REPO_ROOT
    / "results"
    / "checkpoint_F_applications"
    / "expF04_qi_init_real_data"
    / "all20_2layers_uniform"
)
CACHE = REPO_ROOT / "data" / "cache_all20"
WIDTH, EPOCHS, BATCH, LR, SEEDS = 256, 50, 128, 1e-3, [0, 1, 2]
ACTS = ["tanh", "gelu"]
INITS = ["baseline", "qi1", "qi2"]
WANDB_PROJECT = "precisionMLPs-expF04"
CFG_COLORS = {
    "tanh-baseline": "tab:blue",
    "tanh-qi1": "tab:green",
    "tanh-qi2": "tab:cyan",
    "gelu-baseline": "tab:orange",
    "gelu-qi1": "tab:red",
    "gelu-qi2": "tab:purple",
}


def apply_init(model, scheme, xtr, p):
    if scheme == "baseline":
        return
    qi_ridge_init_layer_(
        model.fc1, xtr, centers_per_dir=p, uniform_centers=True
    )
    if scheme == "qi2":
        sample = xtr[:4096]
        h1 = model.hidden1(sample)
        qi_ridge_init_layer_(
            model.fc2, h1, centers_per_dir=p, uniform_centers=True
        )


def evaluate(model, x, y, kind, bs=8192):
    model.eval()
    loss, correct, n = 0.0, 0, x.shape[0]
    with torch.no_grad():
        for i in range(0, n, bs):
            out = model(x[i : i + bs])
            if kind == "classif":
                loss += F.cross_entropy(
                    out, y[i : i + bs], reduction="sum"
                ).item()
                correct += (out.argmax(1) == y[i : i + bs]).sum().item()
            else:
                loss += F.mse_loss(
                    out.squeeze(-1), y[i : i + bs], reduction="sum"
                ).item()
    return loss / n, (correct / n if kind == "classif" else None)


def run_config(job):
    torch.set_num_threads(job["threads"])
    name, act, scheme, seed = (
        job["name"],
        job["act"],
        job["scheme"],
        job["seed"],
    )
    d = torch.load(CACHE / f"{name}.pt", weights_only=True)
    xtr, ytr, xte, yte = d["xtr"], d["ytr"], d["xte"], d["yte"]
    kind, d_in, d_out = d["kind"], d["d_in"], d["d_out"]

    width, p = job["width"], job["P"]
    torch.manual_seed(seed)
    model = SimpleMLP2(d_in, width, d_out, activation=act)
    apply_init(model, scheme, xtr, p)

    wb = None
    if job["use_wandb"]:
        import wandb

        wb = wandb.init(
            project=WANDB_PROJECT,
            reinit=True,
            group=f"2lu-{name}",
            job_type=f"{act}-{scheme}",
            name=f"2lu-{name}-{act}-{scheme}-N{width}-s{seed}",
            tags=["all20_2layers_uniform", name, act, scheme, f"N{width}"],
            settings=wandb.Settings(silent=True),
            config={
                "subexp": "all20_2layers_uniform",
                "task": name,
                "act": act,
                "scheme": scheme,
                "seed": seed,
                "kind": kind,
                "width": width,
                "uniform_centers": True,
            },
        )

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = xtr.shape[0]
    best_eval, best_acc = float("inf"), None
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad(set_to_none=True)
            out = model(xtr[idx])
            loss = (
                F.cross_entropy(out, ytr[idx])
                if kind == "classif"
                else F.mse_loss(out.squeeze(-1), ytr[idx])
            )
            loss.backward()
            opt.step()
        ev, acc = evaluate(model, xte, yte, kind)
        if ev < best_eval:
            best_eval, best_acc = ev, acc
        if wb is not None:
            log = {
                "epoch": epoch,
                "eval_loss": ev,
                "best_eval_loss": best_eval,
            }
            if acc is not None:
                log["eval_acc"] = acc
            wb.log(log)
    if wb is not None:
        wb.summary.update(
            {"best_eval_loss": best_eval, "final_eval_loss": ev}
        )
        wb.finish()
    return {
        "name": name,
        "act": act,
        "scheme": scheme,
        "seed": seed,
        "kind": kind,
        "best_eval": best_eval,
        "final_eval": ev,
        "best_acc": best_acc,
    }


def cfg_key(act, scheme):
    return f"{act}-{scheme}"


def make_figures(rows, names, width):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    suffix = f"_N{width}"

    def aggregate(name, act, scheme, key="best_eval"):
        values = [
            row[key]
            for row in rows
            if row["name"] == name
            and row["act"] == act
            and row["scheme"] == scheme
        ]
        return float(np.mean(values)) if values else None

    order = [name for name in DS.TASK_ORDER if name in names]

    fig, ax = plt.subplots(figsize=(max(11, 0.95 * len(order)), 5.5))
    x = np.arange(len(order))
    bar_width = 0.13
    configurations = [(act, scheme) for act in ACTS for scheme in INITS]
    for j, (act, scheme) in enumerate(configurations):
        values = [aggregate(name, act, scheme) for name in order]
        label = cfg_key(act, scheme)
        ax.bar(
            x + (j - 2.5) * bar_width,
            values,
            bar_width,
            label=label,
            color=CFG_COLORS[label],
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DS.PRETTY.get(name, name) for name in order], rotation=45, ha="right"
    )
    ax.set_ylabel("best eval loss (over training)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=6,
        frameon=False,
    )
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.set_title(
        f"expF04/all20_2layers_uniform -- uniform centers, N={width}", y=1.06
    )
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bars_path = OUT_DIR / "figures" / f"all20_2lu_bars{suffix}.png"
    bars_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(bars_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {bars_path}")

    baseline = {
        name: aggregate(name, "tanh", "baseline") for name in order
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means = []
    for act in ACTS:
        for scheme in INITS:
            ratios = [
                aggregate(name, act, scheme) / baseline[name]
                for name in order
                if baseline[name]
            ]
            means.append(
                (cfg_key(act, scheme), float(np.exp(np.mean(np.log(ratios)))))
            )
    means.sort(key=lambda item: item[1])
    labels = [item[0] for item in means]
    values = [item[1] for item in means]
    ax.barh(
        np.arange(len(labels)), values, color=[CFG_COLORS[label] for label in labels]
    )
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(1.0, color="k", lw=1)
    ax.set_xlabel(
        "geo-mean best-eval-loss ratio vs tanh-baseline (lower = better)"
    )
    ax.set_title(
        f"expF04/all20_2layers_uniform -- ranking, N={width}, "
        f"across {len(order)} tasks"
    )
    ax.grid(True, axis="x", alpha=0.3)
    for i, value in enumerate(values):
        ax.text(value, i, f" {value:.3f}", va="center")
    fig.tight_layout()
    ranking_path = OUT_DIR / "figures" / f"all20_2lu_ranking{suffix}.png"
    fig.savefig(ranking_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {ranking_path}")
    return means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--width", type=int, default=WIDTH)
    args = parser.parse_args()
    width = args.width
    p = int(math.isqrt(width))
    use_wandb = not args.no_wandb
    threads = max(1, (os.cpu_count() or 4) // args.workers)
    if use_wandb:
        os.environ.setdefault("WANDB_SILENT", "true")

    if not CACHE.exists() or not any(CACHE.glob("*.pt")):
        print(
            "ERROR: all20 cache missing. Run all20/run.py first "
            "(it builds data/cache_all20)."
        )
        sys.exit(1)
    names = (
        args.datasets.split(",")
        if args.datasets
        else [path.stem for path in sorted(CACHE.glob("*.pt"))]
    )
    names = [name for name in DS.TASK_ORDER if name in names]
    print(f"{len(names)} datasets: {names}")

    jobs = [
        {
            "name": name,
            "act": act,
            "scheme": scheme,
            "seed": seed,
            "width": width,
            "P": p,
            "use_wandb": use_wandb,
            "threads": threads,
        }
        for name in names
        for act in ACTS
        for scheme in INITS
        for seed in SEEDS
    ]
    print(
        f"expF04/all20_2layers_uniform | {len(jobs)} runs | "
        f"2 hidden layers N={width} | uniform centers | "
        f"configs={[cfg_key(act, scheme) for act in ACTS for scheme in INITS]} | "
        f"seeds={SEEDS} | wandb={use_wandb}\n"
    )

    rows, start, done = [], time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_config, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            rows.append(result)
            done += 1
            tail = (
                f" acc={result['best_acc']:.3f}"
                if result["best_acc"] is not None
                else ""
            )
            print(
                f"  [{done}/{len(jobs)}] {result['name']:18s} "
                f"{result['act']:4s} {result['scheme']:8s} "
                f"s{result['seed']} best_eval={result['best_eval']:.4e}{tail}",
                flush=True,
            )
    print(f"done in {time.time() - start:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / f"all20_2lu_rows_N{width}.json"
    rows_path.write_text(json.dumps(rows))
    means = make_figures(rows, names, width)
    print(
        f"\nOVERALL RANKING N={width} "
        "(geo-mean best-eval ratio vs tanh-baseline):"
    )
    for label, value in means:
        print(f"  {label:16s} {value:.3f}")
    print(f"  saved {rows_path}")


if __name__ == "__main__":
    main()
