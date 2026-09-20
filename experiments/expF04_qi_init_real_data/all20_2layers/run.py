"""expF04/all20_2layers -- QI ridge init on a TWO-hidden-layer MLP, 6 configs.

Architecture: d_in -> 256 -> 256 -> d_out (fc1, fc2 hidden; fc3 readout).
Six configs = {tanh, gelu} x {baseline, qi1, qi2}:
  baseline  all three layers default init
  qi1       fc1 = scaled ridge QI init;      fc2, fc3 default
  qi2       fc1 AND fc2 = scaled ridge init;  fc3 default
            (fc2 uses fresh random directions, tiled over the post-fc1 hidden activations)

QI init = the best scheme from ridge_real (scaled_psqrt: sqrt(N) bundles, per-ridge
gamma, random projection-sampled centers).

Adam setup replicates the checkpoint-D init runs (expD05): plain Adam, lr=1e-3, default
betas, CONSTANT lr (no warmup, no schedule). expD05 was full-batch; that is infeasible
here (60k-image datasets x many steps x 288 runs), so we minibatch and, like expD05,
report the BEST eval seen over training (not just the final epoch).

Run:  uv run python experiments/expF04_qi_init_real_data/all20_2layers/run.py
      --no-wandb / --datasets a,b / --workers N
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
from model import SimpleMLP2, qi_ridge_init_layer_   # noqa: E402
import datasets as DS                                 # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF04_qi_init_real_data" / "all20_2layers"
CACHE = REPO_ROOT / "data" / "cache_all20"          # reuse the all20 cache
WIDTH, EPOCHS, BATCH, LR, SEEDS = 256, 50, 128, 1e-3, [0, 1, 2]
ACTS = ["tanh", "gelu"]
INITS = ["baseline", "qi1", "qi2"]
P = int(math.isqrt(WIDTH))                            # sqrt(N) centers per ridge bundle
MAX_TRAIN = 30000
WANDB_PROJECT = "precisionMLPs-expF04"
CFG_COLORS = {
    "tanh-baseline": "tab:blue", "tanh-qi1": "tab:green", "tanh-qi2": "tab:cyan",
    "gelu-baseline": "tab:orange", "gelu-qi1": "tab:red", "gelu-qi2": "tab:purple",
}


def apply_init(model, scheme, xtr, p):
    if scheme == "baseline":
        return
    qi_ridge_init_layer_(model.fc1, xtr, centers_per_dir=p, uniform_centers=False)
    if scheme == "qi2":
        sample = xtr[:4096]
        h1 = model.hidden1(sample)                   # post-fc1 activations
        qi_ridge_init_layer_(model.fc2, h1, centers_per_dir=p, uniform_centers=False)


def evaluate(model, x, y, kind, bs=8192):
    model.eval()
    loss, correct, n = 0.0, 0, x.shape[0]
    with torch.no_grad():
        for i in range(0, n, bs):
            out = model(x[i:i + bs])
            if kind == "classif":
                loss += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
                correct += (out.argmax(1) == y[i:i + bs]).sum().item()
            else:
                loss += F.mse_loss(out.squeeze(-1), y[i:i + bs], reduction="sum").item()
    return loss / n, (correct / n if kind == "classif" else None)


def run_config(job):
    torch.set_num_threads(job["threads"])
    name, act, scheme, seed = job["name"], job["act"], job["scheme"], job["seed"]
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
        wb = wandb.init(project=WANDB_PROJECT, reinit=True, group=f"2l-{name}",
                        job_type=f"{act}-{scheme}", name=f"2l-{name}-{act}-{scheme}-s{seed}",
                        tags=["all20_2layers", name, act, scheme],
                        settings=wandb.Settings(silent=True),
                        config={"subexp": "all20_2layers", "task": name, "act": act,
                                "scheme": scheme, "seed": seed, "kind": kind, "width": width})

    opt = torch.optim.Adam(model.parameters(), lr=LR)   # expD05: plain Adam, constant lr
    n = xtr.shape[0]
    best_eval, best_acc, best_epoch = float("inf"), None, -1
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            out = model(xtr[idx])
            loss = (F.cross_entropy(out, ytr[idx]) if kind == "classif"
                    else F.mse_loss(out.squeeze(-1), ytr[idx]))
            loss.backward(); opt.step()
        ev, acc = evaluate(model, xte, yte, kind)
        if ev < best_eval:                              # expD05: track best over training
            best_eval, best_acc, best_epoch = ev, acc, epoch
        if wb is not None:
            log = {"epoch": epoch, "eval_loss": ev, "best_eval_loss": best_eval}
            if acc is not None:
                log["eval_acc"] = acc
            wb.log(log)
    if wb is not None:
        wb.summary.update({"best_eval_loss": best_eval, "final_eval_loss": ev})
        wb.finish()
    return {"name": name, "act": act, "scheme": scheme, "seed": seed, "kind": kind,
            "best_eval": best_eval, "final_eval": ev, "best_acc": best_acc}


def cfg_key(act, scheme):
    return f"{act}-{scheme}"


def make_figures(rows, names, width):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfgs = [cfg_key(a, s) for a in ACTS for s in INITS]
    sfx = f"_N{width}"

    def agg(name, act, scheme, key="best_eval"):
        v = [r[key] for r in rows if r["name"] == name and r["act"] == act and r["scheme"] == scheme]
        return float(np.mean(v)) if v else None

    order = [n for n in DS.TASK_ORDER if n in names]

    # 1. per-task grouped bars: 6 configs, best eval loss
    fig, ax = plt.subplots(figsize=(max(11, 0.95 * len(order)), 5.5))
    x = np.arange(len(order)); w = 0.13
    for j, (a, s) in enumerate([(a, s) for a in ACTS for s in INITS]):
        vals = [agg(n, a, s) for n in order]
        ax.bar(x + (j - 2.5) * w, vals, w, label=cfg_key(a, s), color=CFG_COLORS[cfg_key(a, s)])
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([DS.PRETTY.get(n, n) for n in order], rotation=45, ha="right")
    ax.set_ylabel("best eval loss (over training)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=6, frameon=False)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.set_title(f"expF04/all20_2layers -- 6 configs per task (2 hidden layers, N={width})", y=1.06)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p1 = OUT_DIR / "figures" / f"all20_2l_bars{sfx}.png"
    p1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p1, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p1}")

    # 2. overall ranking: geo-mean loss ratio vs tanh-baseline, per config
    base = {n: agg(n, "tanh", "baseline") for n in order}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means = []
    for a in ACTS:
        for s in INITS:
            ratios = [agg(n, a, s) / base[n] for n in order if base[n]]
            means.append((cfg_key(a, s), float(np.exp(np.mean(np.log(ratios))))))
    means.sort(key=lambda kv: kv[1])
    labels = [m[0] for m in means]; vals = [m[1] for m in means]
    ax.barh(np.arange(len(labels)), vals, color=[CFG_COLORS[l] for l in labels])
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    ax.invert_yaxis(); ax.axvline(1.0, color="k", lw=1)
    ax.set_xlabel("geo-mean best-eval-loss ratio vs tanh-baseline (lower = better)")
    ax.set_title(f"expF04/all20_2layers -- config ranking, N={width}, across {len(order)} tasks")
    ax.grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center")
    fig.tight_layout()
    p2 = OUT_DIR / "figures" / f"all20_2l_ranking{sfx}.png"
    fig.savefig(p2, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p2}")
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--width", type=int, default=WIDTH)
    args = ap.parse_args()
    width = args.width
    p = int(math.isqrt(width))
    use_wandb = not args.no_wandb
    threads = max(1, (os.cpu_count() or 4) // args.workers)
    if use_wandb:
        os.environ.setdefault("WANDB_SILENT", "true")

    if not CACHE.exists() or not any(CACHE.glob("*.pt")):
        print("ERROR: all20 cache missing. Run all20/run.py first (it builds data/cache_all20).")
        sys.exit(1)
    names = args.datasets.split(",") if args.datasets else [
        p.stem for p in sorted(CACHE.glob("*.pt"))]
    names = [n for n in DS.TASK_ORDER if n in names]
    print(f"{len(names)} datasets: {names}")

    jobs = [{"name": n, "act": a, "scheme": s, "seed": seed, "width": width, "P": p,
             "use_wandb": use_wandb, "threads": threads}
            for n in names for a in ACTS for s in INITS for seed in SEEDS]
    print(f"expF04/all20_2layers | {len(jobs)} runs | 2 hidden layers N={width} "
          f"| configs={[cfg_key(a,s) for a in ACTS for s in INITS]} | seeds={SEEDS} | wandb={use_wandb}\n")

    rows, t0, done = [], time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_config, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result(); rows.append(r); done += 1
            tail = f" acc={r['best_acc']:.3f}" if r["best_acc"] is not None else ""
            print(f"  [{done}/{len(jobs)}] {r['name']:18s} {r['act']:4s} {r['scheme']:8s} "
                  f"s{r['seed']} best_eval={r['best_eval']:.4e}{tail}", flush=True)
    print(f"done in {time.time()-t0:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"all20_2l_rows_N{width}.json").write_text(json.dumps(rows))
    means = make_figures(rows, names, width)
    print(f"\nOVERALL RANKING N={width} (geo-mean best-eval ratio vs tanh-baseline):")
    for label, v in means:
        print(f"  {label:16s} {v:.3f}")
    print(f"  saved {OUT_DIR}/all20_2l_rows_N{width}.json")


if __name__ == "__main__":
    main()
