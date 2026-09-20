"""expF04 -- does the QI geometry, used as an initialization, help a real MLP?

For each of 4 real tasks (2 classif, 2 regr) x {large, medium} width x seeds, train a
single-hidden-layer GELU MLP with Adam under two inits:
  - qi:       weight rows = gamma * unit-direction, centers swept over the projection
              range, bias = -gamma*c_k  (the repo's gamma*(x-c) lifted to high-d)
  - baseline: default PyTorch init
Log train/eval loss per epoch to wandb (live) and emit two summary figures (one per
width): 2x2 task grids, qi vs baseline (2 colors), train solid / eval dashed.

The 48 (task,arm,width,seed) configs are independent -> run them across a process pool
(each worker pinned to a few threads). A one-time on-disk tensor cache lets workers load
normalized data in ms instead of re-parsing.

Run (after fetch_data.py + `wandb login`):
    uv run python experiments/expF04_qi_init_real_data/run.py            # parallel, online wandb
    uv run python experiments/expF04_qi_init_real_data/run.py --serial   # one at a time
    uv run python experiments/expF04_qi_init_real_data/run.py --smoke    # tiny synthetic, no wandb
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import data as D          # noqa: E402
from model import SimpleMLP, qi_init_   # noqa: E402

ARMS = ["qi", "baseline"]
COLORS = {"qi": "tab:red", "baseline": "tab:blue"}
OUT_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF04_qi_init_real_data"
CACHE_DIR = REPO_ROOT / "data" / "cache"


# ----------------------------- data cache -----------------------------

def _cache_path(name: str, prefix: str = "") -> Path:
    # smoke runs use a "smoke_" prefix so they can NEVER clobber the real data cache
    return CACHE_DIR / f"{prefix}{name}.pt"


def build_cache(names, data_dir, rebuild=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        p = _cache_path(name)
        if p.exists() and not rebuild:
            continue
        t = D.load_task(name, data_dir)
        torch.save({"name": t.name, "kind": t.kind, "d_in": t.d_in, "d_out": t.d_out,
                    "x_train": t.x_train, "y_train": t.y_train,
                    "x_test": t.x_test, "y_test": t.y_test}, p)
        print(f"  cached {name} -> {p.name}")


def load_cached(name: str, prefix: str = "") -> D.Task:
    return D.Task(**torch.load(_cache_path(name, prefix), weights_only=True))


# ----------------------------- training -----------------------------

def evaluate(model, x, y, kind, bs=8192):
    model.eval()
    tot_loss, tot_correct, n = 0.0, 0, x.shape[0]
    with torch.no_grad():
        for i in range(0, n, bs):
            xb, yb = x[i:i + bs], y[i:i + bs]
            out = model(xb)
            if kind == "classif":
                tot_loss += F.cross_entropy(out, yb, reduction="sum").item()
                tot_correct += (out.argmax(1) == yb).sum().item()
            else:
                tot_loss += F.mse_loss(out.squeeze(-1), yb, reduction="sum").item()
    return tot_loss / n, (tot_correct / n if kind == "classif" else None)


def train_one(task: D.Task, width: int, arm: str, seed: int, cfg: dict, device, wb=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimpleMLP(task.d_in, width, task.d_out).to(device)   # identical draw per arm
    xtr = task.x_train.to(device); ytr = task.y_train.to(device)
    xte = task.x_test.to(device); yte = task.y_test.to(device)

    diag = {}
    if arm == "qi":
        diag = qi_init_(model, xtr, lam=cfg["lambda_star"])

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    n, bs = xtr.shape[0], cfg["batch_size"]
    curve = {"train": [], "eval": [], "eval_acc": []}

    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=device)
        run_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = xtr[idx], ytr[idx]
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = (F.cross_entropy(out, yb) if task.kind == "classif"
                    else F.mse_loss(out.squeeze(-1), yb))
            loss.backward(); opt.step()
            run_loss += loss.item() * xb.shape[0]
        tr_loss = run_loss / n
        ev_loss, ev_acc = evaluate(model, xte, yte, task.kind)
        curve["train"].append(tr_loss)
        curve["eval"].append(ev_loss)
        curve["eval_acc"].append(ev_acc)
        if wb is not None:
            log = {"epoch": epoch, "train_loss": tr_loss, "eval_loss": ev_loss}
            if ev_acc is not None:
                log["eval_acc"] = ev_acc
            wb.log(log)
    curve["diag"] = diag
    return curve


# ----------------------------- worker -----------------------------

def run_config(job: dict):
    """One (task, arm, width, seed) config. Top-level + picklable for the process pool."""
    torch.set_num_threads(job["threads"])
    cfg = job["cfg"]
    name, arm, width, seed, label = job["name"], job["arm"], job["width"], job["seed"], job["label"]
    task = load_cached(name, cfg.get("cache_prefix", ""))
    device = torch.device("cpu")

    wb = None
    if job["use_wandb"]:
        import wandb
        wb = wandb.init(
            project=cfg["wandb_project"], reinit=True, group=name, job_type=arm,
            name=f"{name}-{arm}-N{width}-s{seed}", tags=[label, arm, name],
            settings=wandb.Settings(silent=True),
            config={"task": name, "arm": arm, "width": width, "seed": seed,
                    "kind": task.kind, "width_label": label,
                    **{k: cfg[k] for k in ("lr", "epochs", "batch_size", "lambda_star")}},
        )
    curve = train_one(task, width, arm, seed, cfg, device, wb)
    if wb is not None:
        final = {"final_eval_loss": curve["eval"][-1]}
        if curve["eval_acc"][-1] is not None:
            final["final_eval_acc"] = curve["eval_acc"][-1]
        wb.summary.update(final)
        wb.finish()
    return {"name": name, "arm": arm, "width": width, "seed": seed, "label": label,
            "curve": curve}


# ----------------------------- plotting -----------------------------

def _mean_curve(per_seed, key):
    return np.mean([c[key] for c in per_seed], axis=0)


def make_figure(results, tasks, width, label, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flat
    for i, name in enumerate(tasks):
        ax = axes[i]
        kind = D.TASK_META[name][2]
        for arm in ARMS:
            per_seed = results[(name, arm, width)]
            ep = np.arange(1, len(per_seed[0]["train"]) + 1)
            ax.semilogy(ep, _mean_curve(per_seed, "train"), color=COLORS[arm], ls="-", lw=1.8)
            ax.semilogy(ep, _mean_curve(per_seed, "eval"), color=COLORS[arm], ls="--", lw=1.8)
        loss_name = "cross-entropy" if kind == "classif" else "MSE"
        ax.set_title(f"{D.PRETTY[name]}  ({loss_name})")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.grid(True, which="both", alpha=0.25)
    for j in range(len(tasks), 4):
        axes[j].axis("off")

    handles = [
        Line2D([0], [0], color=COLORS["qi"], ls="-", lw=2, label="QI-init  train"),
        Line2D([0], [0], color=COLORS["qi"], ls="--", lw=2, label="QI-init  eval"),
        Line2D([0], [0], color=COLORS["baseline"], ls="-", lw=2, label="baseline  train"),
        Line2D([0], [0], color=COLORS["baseline"], ls="--", lw=2, label="baseline  eval"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=4, frameon=False)
    fig.suptitle(f"expF04  QI-init vs baseline  --  {label} width (N={width})",
                 y=1.02, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ----------------------------- main -----------------------------

def load_cfg(args):
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    if args.tasks:
        cfg["tasks"] = args.tasks.split(",")
    cfg["cache_prefix"] = ""
    if args.smoke:
        cfg.update(epochs=2, seeds=[0], widths={"medium": 32}, cache_prefix="smoke_")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run, no data/wandb")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--serial", action="store_true", help="disable the process pool")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args)
    use_wandb = not (args.no_wandb or args.smoke)
    workers = 1 if (args.serial or args.smoke) else args.workers
    threads = max(1, (os.cpu_count() or 4) // workers)
    if use_wandb:
        os.environ.setdefault("WANDB_SILENT", "true")
    print(f"expF04 | tasks={cfg['tasks']} | widths={cfg['widths']} | seeds={cfg['seeds']} "
          f"| epochs={cfg['epochs']} | wandb={use_wandb} | workers={workers} x {threads} threads")

    # build job list
    jobs = [{"name": name, "arm": arm, "width": width, "seed": seed, "label": label,
             "cfg": cfg, "use_wandb": use_wandb, "threads": threads}
            for label, width in cfg["widths"].items()
            for name in cfg["tasks"] for arm in ARMS for seed in cfg["seeds"]]

    # prepare data (cache for real runs; smoke writes synthetic cache)
    if args.smoke:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for name in cfg["tasks"]:
            t = D.smoke_task(name)
            torch.save({"name": t.name, "kind": t.kind, "d_in": t.d_in, "d_out": t.d_out,
                        "x_train": t.x_train, "y_train": t.y_train,
                        "x_test": t.x_test, "y_test": t.y_test},
                       _cache_path(name, cfg["cache_prefix"]))
    else:
        print("preparing data cache...")
        build_cache(cfg["tasks"], REPO_ROOT / "data", rebuild=args.rebuild_cache)

    results = {}
    t0 = time.time()
    done = 0

    def _record(r):
        nonlocal done
        done += 1
        results.setdefault((r["name"], r["arm"], r["width"]), []).append(r["curve"])
        acc = r["curve"]["eval_acc"][-1]
        tail = f" acc={acc:.3f}" if acc is not None else ""
        print(f"  [{done}/{len(jobs)}] {r['label']:6s} N={r['width']:<3d} {r['name']:18s} "
              f"{r['arm']:8s} s{r['seed']} eval_loss={r['curve']['eval'][-1]:.4e}{tail}", flush=True)

    if workers == 1:
        for job in jobs:
            _record(run_config(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_config, job) for job in jobs]
            for fut in as_completed(futs):
                _record(fut.result())
    print(f"training done in {time.time()-t0:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, width in cfg["widths"].items():
        make_figure(results, cfg["tasks"], width, label,
                    OUT_DIR / "figures" / f"expF04_loss_{label}.png")

    dump = {f"{n}|{a}|{w}": [{"train": c["train"], "eval": c["eval"],
                              "eval_acc": c["eval_acc"], "diag": c.get("diag", {})}
                             for c in v]
            for (n, a, w), v in results.items()}
    (OUT_DIR / "curves.json").write_text(json.dumps(dump))
    print(f"  saved {OUT_DIR/'curves.json'}")


if __name__ == "__main__":
    main()
