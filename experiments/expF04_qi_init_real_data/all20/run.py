"""expF04/all20 -- best QI init (scaled_psqrt) vs baseline across every available task.

Fixed config: width N=256, tanh, plain Adam both layers, 50 epochs, 3 seeds. Two arms:
  baseline      PyTorch default init
  scaled_psqrt  sqrt(N) ridge bundles, per-ridge gamma, random projection-sampled centers
                (the most consistent winner from ridge_real)

Runs on whatever datasets.available() reports (skips un-fetched / unparseable). Metric:
final eval loss (cross-entropy / MSE) + accuracy for classification. Emits a win/loss
summary bar chart (log2 loss ratio per task) + per-task grouped bars.

Run:  uv run python experiments/expF04_qi_init_real_data/all20/run.py
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
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(HERE))
from model import SimpleMLP, qi_ridge_init_   # noqa: E402
import datasets as DS                          # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF04_qi_init_real_data" / "all20"
CACHE = REPO_ROOT / "data" / "cache_all20"
WIDTH, EPOCHS, BATCH, LR, SEEDS = 256, 50, 128, 1e-3, [0, 1, 2]
MAX_TRAIN = 30000          # cap big datasets for bounded runtime
WANDB_PROJECT = "precisionMLPs-expF04"


def prepare_cache(names, data_dir):
    CACHE.mkdir(parents=True, exist_ok=True)
    ready = []
    for name in names:
        p = CACHE / f"{name}.pt"
        if not p.exists():
            t = DS.load(name, data_dir)
            xtr, ytr = t.x_train, t.y_train
            if xtr.shape[0] > MAX_TRAIN:
                rng = np.random.default_rng(0)
                idx = rng.permutation(xtr.shape[0])[:MAX_TRAIN]
                xtr, ytr = xtr[idx], ytr[idx]
            torch.save({"name": name, "kind": t.kind, "d_in": t.d_in, "d_out": t.d_out,
                        "xtr": torch.tensor(xtr), "ytr": torch.tensor(ytr),
                        "xte": torch.tensor(t.x_test), "yte": torch.tensor(t.y_test)}, p)
        ready.append(name)
    return ready


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
    name, arm, seed = job["name"], job["arm"], job["seed"]
    d = torch.load(CACHE / f"{name}.pt", weights_only=True)
    xtr, ytr, xte, yte = d["xtr"], d["ytr"], d["xte"], d["yte"]
    kind, d_in, d_out = d["kind"], d["d_in"], d["d_out"]

    torch.manual_seed(seed)
    model = SimpleMLP(d_in, WIDTH, d_out, activation="tanh")
    if arm == "scaled_psqrt":
        qi_ridge_init_(model, xtr, centers_per_dir=int(math.isqrt(WIDTH)), uniform_centers=False)

    wb = None
    if job["use_wandb"]:
        import wandb
        wb = wandb.init(project=WANDB_PROJECT, reinit=True, group=f"all20-{name}", job_type=arm,
                        name=f"all20-{name}-{arm}-s{seed}", tags=["all20", name, arm],
                        settings=wandb.Settings(silent=True),
                        config={"subexp": "all20", "task": name, "arm": arm, "seed": seed,
                                "kind": kind, "width": WIDTH, "activation": "tanh"})

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = xtr.shape[0]
    curve = {"eval": [], "eval_acc": []}
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
        curve["eval"].append(ev); curve["eval_acc"].append(acc)
        if wb is not None:
            log = {"epoch": epoch, "eval_loss": ev}
            if acc is not None:
                log["eval_acc"] = acc
            wb.log(log)
    if wb is not None:
        wb.summary.update({"final_eval_loss": curve["eval"][-1]})
        wb.finish()
    return {"name": name, "arm": arm, "seed": seed, "kind": kind,
            "eval": curve["eval"][-1], "acc": curve["eval_acc"][-1]}


def make_figures(rows, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def agg(name, arm, key):
        vals = [r[key] for r in rows if r["name"] == name and r["arm"] == arm and r[key] is not None]
        return float(np.mean(vals)) if vals else None

    # win/loss: log2( scaled / baseline ) eval loss, one bar per task
    order = [n for n in DS.TASK_ORDER if n in names]
    ratios, labels, kinds = [], [], []
    for n in order:
        b, s = agg(n, "baseline", "eval"), agg(n, "scaled_psqrt", "eval")
        if b and s:
            ratios.append(math.log2(s / b)); labels.append(DS.PRETTY.get(n, n))
            kinds.append([r["kind"] for r in rows if r["name"] == n][0])

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(labels))))
    colors = ["tab:green" if x < 0 else "tab:red" for x in ratios]
    ypos = np.arange(len(labels))
    ax.barh(ypos, ratios, color=colors)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{l}  ({k[:5]})" for l, k in zip(labels, kinds)])
    ax.axvline(0, color="k", lw=1)
    ax.invert_yaxis()
    ax.set_xlabel("log2( scaled_psqrt loss / baseline loss )   <-- init better | baseline better -->")
    ax.set_title(f"expF04/all20 -- QI ridge init vs baseline (N=256, tanh, {len(labels)} tasks, "
                 f"mean of {len(SEEDS)} seeds)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "figures" / "all20_winloss.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}")

    # grouped final-loss bars
    fig, ax = plt.subplots(figsize=(max(9, 0.7 * len(order)), 5))
    x = np.arange(len(order))
    b = [agg(n, "baseline", "eval") for n in order]
    s = [agg(n, "scaled_psqrt", "eval") for n in order]
    ax.bar(x - 0.2, b, 0.4, label="baseline", color="tab:blue")
    ax.bar(x + 0.2, s, 0.4, label="scaled_psqrt", color="tab:brown")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([DS.PRETTY.get(n, n) for n in order], rotation=45, ha="right")
    ax.set_ylabel("final eval loss"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    p2 = OUT_DIR / "figures" / "all20_bars.png"
    fig.savefig(p2, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p2}")
    return ratios, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--datasets", default=None)
    args = ap.parse_args()
    use_wandb = not args.no_wandb
    threads = max(1, (os.cpu_count() or 4) // args.workers)
    if use_wandb:
        os.environ.setdefault("WANDB_SILENT", "true")

    data_dir = REPO_ROOT / "data"
    print("scanning available datasets...")
    names = args.datasets.split(",") if args.datasets else DS.available(data_dir)
    print(f"\n{len(names)} datasets available: {names}")
    names = prepare_cache(names, data_dir)

    jobs = [{"name": n, "arm": a, "seed": s, "use_wandb": use_wandb, "threads": threads}
            for n in names for a in ("baseline", "scaled_psqrt") for s in SEEDS]
    print(f"expF04/all20 | {len(jobs)} runs | N={WIDTH} tanh | seeds={SEEDS} | wandb={use_wandb}\n")

    rows, t0, done = [], time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_config, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result(); rows.append(r); done += 1
            tail = f" acc={r['acc']:.3f}" if r["acc"] is not None else ""
            print(f"  [{done}/{len(jobs)}] {r['name']:18s} {r['arm']:13s} s{r['seed']} "
                  f"eval={r['eval']:.4e}{tail}", flush=True)
    print(f"done in {time.time()-t0:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "all20_rows.json").write_text(json.dumps(rows))
    make_figures(rows, names)
    print(f"  saved {OUT_DIR/'all20_rows.json'}")


if __name__ == "__main__":
    main()
