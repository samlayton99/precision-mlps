"""expF04/ridge_real -- ridge-bundle QI init vs flat lift vs baseline on the 4 real tasks.

Same protocol as the parent expF04 run (plain Adam, both layers, 50 epochs) with the
init-arm axis extended:

  baseline      PyTorch default
  flat          parent expF04 arm (M=N dirs x 1 center, gamma=0.125*N/A)
  ridge_p4      M x P ridge bundles, P=4
  ridge_psqrt   P=sqrt(N)
  ridge_pn4     P=N/4
  scaled_psqrt  ridge_psqrt directions/gamma, random centers (D02 win-#3 analog)

Requires the parent data cache (built automatically from ../data if missing).

Run:  uv run python experiments/expF04_qi_init_real_data/ridge_real/run.py
      (--smoke: tiny synthetic pipeline check, no wandb)
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
import yaml

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(PARENT))
import data as D                                        # noqa: E402
from model import SimpleMLP, qi_init_, qi_ridge_init_   # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF04_qi_init_real_data" / "ridge_real"
CACHE_DIR = REPO_ROOT / "data" / "cache"

ARM_COLORS = {
    "baseline": "tab:blue", "flat": "tab:red", "ridge_p4": "tab:orange",
    "ridge_psqrt": "tab:green", "ridge_pn4": "tab:purple", "scaled_psqrt": "tab:brown",
}


def resolve_p(arm: str, width: int) -> int | None:
    if arm in ("baseline", "flat"):
        return None
    tag = arm.split("_p", 1)[1]
    return int(math.isqrt(width)) if tag == "sqrt" else (width // 4 if tag == "n4" else int(tag))


def apply_init(model, arm, width, x_train, lam):
    if arm == "baseline":
        return {}
    if arm == "flat":
        return qi_init_(model, x_train, lam=lam)
    return qi_ridge_init_(model, x_train, lam=lam, centers_per_dir=resolve_p(arm, width),
                          uniform_centers=not arm.startswith("scaled"))


def _cache_path(name: str, prefix: str = "") -> Path:
    # smoke runs use a "smoke_" prefix so they can NEVER clobber the real data cache
    return CACHE_DIR / f"{prefix}{name}.pt"


def load_cached(name: str, prefix: str = "") -> D.Task:
    return D.Task(**torch.load(_cache_path(name, prefix), weights_only=True))


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


def run_config(job: dict):
    torch.set_num_threads(job["threads"])
    cfg = job["cfg"]
    name, arm, width, seed, label = job["name"], job["arm"], job["width"], job["seed"], job["label"]
    act = job["act"]
    task = load_cached(name, cfg.get("cache_prefix", ""))

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SimpleMLP(task.d_in, width, task.d_out, activation=act)
    xtr, ytr = task.x_train, task.y_train
    xte, yte = task.x_test, task.y_test
    diag = apply_init(model, arm, width, xtr, cfg["lambda_star"])

    wb = None
    if job["use_wandb"]:
        import wandb
        wb = wandb.init(
            project=cfg["wandb_project"], reinit=True, group=f"real-{name}", job_type=arm,
            name=f"real-{name}-{act}-{arm}-N{width}-s{seed}",
            tags=["ridge_real", label, arm, name, act], settings=wandb.Settings(silent=True),
            config={"subexp": "ridge_real", "task": name, "arm": arm, "width": width,
                    "seed": seed, "kind": task.kind, "width_label": label, "activation": act,
                    **{k: cfg[k] for k in ("lr", "epochs", "batch_size", "lambda_star")}},
        )

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    n, bs = xtr.shape[0], cfg["batch_size"]
    curve = {"train": [], "eval": [], "eval_acc": []}
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n)
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
        ev_loss, ev_acc = evaluate(model, xte, yte, task.kind)
        curve["train"].append(run_loss / n)
        curve["eval"].append(ev_loss)
        curve["eval_acc"].append(ev_acc)
        if wb is not None:
            log = {"epoch": epoch, "train_loss": curve["train"][-1], "eval_loss": ev_loss}
            if ev_acc is not None:
                log["eval_acc"] = ev_acc
            wb.log(log)
    if wb is not None:
        final = {"final_eval_loss": curve["eval"][-1]}
        if curve["eval_acc"][-1] is not None:
            final["final_eval_acc"] = curve["eval_acc"][-1]
        wb.summary.update(final)
        wb.finish()
    return {"name": name, "arm": arm, "width": width, "seed": seed, "label": label,
            "act": act, "curve": curve, "diag": diag}


# ----------------------------- figures -----------------------------

def make_curve_figure(results, cfg, width, label, act, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.flat
    for i, name in enumerate(cfg["tasks"]):
        ax = axes[i]
        kind = D.TASK_META[name][2]
        for arm in cfg["arms"]:
            per_seed = results.get((name, arm, width, act))
            if not per_seed:
                continue
            ep = np.arange(1, len(per_seed[0]["train"]) + 1)
            tr = np.mean([c["train"] for c in per_seed], axis=0)
            ev = np.mean([c["eval"] for c in per_seed], axis=0)
            ax.semilogy(ep, tr, color=ARM_COLORS[arm], ls="-", lw=0.9, alpha=0.5)
            ax.semilogy(ep, ev, color=ARM_COLORS[arm], ls="--", lw=1.8)
        loss_name = "cross-entropy" if kind == "classif" else "MSE"
        ax.set_title(f"{D.PRETTY[name]}  ({loss_name})")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.grid(True, which="both", alpha=0.25)
    handles = [Line2D([0], [0], color=ARM_COLORS[a], ls="--", lw=1.8, label=a)
               for a in cfg["arms"]]
    handles.append(Line2D([0], [0], color="gray", ls="-", lw=0.9, alpha=0.6, label="(thin solid = train)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, frameon=False)
    fig.suptitle(f"expF04/ridge_real  --  {act.upper()}, {label} width (N={width})  eval dashed, train thin solid",
                 y=1.05, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def make_bar_figure(results, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = [(label, width, act) for label, width in cfg["widths"].items()
            for act in cfg["activations"]]
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.6 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (label, width, act) in enumerate(rows):
        for c, name in enumerate(cfg["tasks"]):
            ax = axes[r, c]
            vals, errs, cols = [], [], []
            for arm in cfg["arms"]:
                per_seed = results.get((name, arm, width, act), [])
                fin = [cur["eval"][-1] for cur in per_seed]
                vals.append(np.mean(fin) if fin else np.nan)
                errs.append(np.std(fin) if fin else 0.0)
                cols.append(ARM_COLORS[arm])
            x = np.arange(len(cfg["arms"]))
            ax.bar(x, vals, yerr=errs, color=cols, capsize=3)
            ax.set_yscale("log")
            ax.set_xticks([])
            ax.set_title(f"{D.PRETTY[name]}  N={width} {act}", fontsize=10)
            ax.grid(True, axis="y", which="both", alpha=0.25)
        axes[r, 0].set_ylabel("final eval loss")
    handles = [Patch(color=ARM_COLORS[a], label=a) for a in cfg["arms"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(cfg["arms"]), frameon=False)
    fig.suptitle("expF04/ridge_real -- final eval loss by init arm (mean +/- std over seeds)",
                 y=1.08, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    cfg["cache_prefix"] = ""
    if args.smoke:
        cfg.update(epochs=2, seeds=[0], widths={"medium": 64}, cache_prefix="smoke_",
                   activations=["gelu", "tanh"])
    use_wandb = not (args.no_wandb or args.smoke)
    workers = 1 if args.smoke else args.workers
    threads = max(1, (os.cpu_count() or 4) // workers)
    if use_wandb:
        os.environ.setdefault("WANDB_SILENT", "true")

    # data cache (reuses the parent's; smoke fabricates)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in cfg["tasks"]:
        if args.smoke:
            t = D.smoke_task(name)
            torch.save({"name": t.name, "kind": t.kind, "d_in": t.d_in, "d_out": t.d_out,
                        "x_train": t.x_train, "y_train": t.y_train,
                        "x_test": t.x_test, "y_test": t.y_test},
                       _cache_path(name, cfg["cache_prefix"]))
        elif not _cache_path(name).exists():
            t = D.load_task(name, REPO_ROOT / "data")
            torch.save({"name": t.name, "kind": t.kind, "d_in": t.d_in, "d_out": t.d_out,
                        "x_train": t.x_train, "y_train": t.y_train,
                        "x_test": t.x_test, "y_test": t.y_test}, _cache_path(name))
            print(f"  cached {name}")

    jobs = [{"cfg": cfg, "name": name, "arm": arm, "width": width, "seed": seed,
             "label": label, "act": act, "use_wandb": use_wandb, "threads": threads}
            for label, width in cfg["widths"].items()
            for act in cfg["activations"]
            for name in cfg["tasks"] for arm in cfg["arms"] for seed in cfg["seeds"]]
    print(f"expF04/ridge_real | {len(jobs)} runs | arms={cfg['arms']} | widths={cfg['widths']} "
          f"| acts={cfg['activations']} | seeds={cfg['seeds']} | wandb={use_wandb} "
          f"| workers={workers}x{threads}t")

    results = {}
    t0 = time.time()
    done = 0

    def _rec(r):
        nonlocal done
        done += 1
        results.setdefault((r["name"], r["arm"], r["width"], r["act"]), []).append(r["curve"])
        acc = r["curve"]["eval_acc"][-1]
        tail = f" acc={acc:.3f}" if acc is not None else ""
        print(f"  [{done}/{len(jobs)}] {r['label']:6s} N={r['width']:<3d} {r['act']:4s} "
              f"{r['name']:18s} {r['arm']:13s} s{r['seed']} "
              f"eval={r['curve']['eval'][-1]:.4e}{tail}", flush=True)

    if workers == 1:
        for job in jobs:
            _rec(run_config(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_config, job) for job in jobs]
            for fut in as_completed(futs):
                _rec(fut.result())
    print(f"training done in {time.time()-t0:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, width in cfg["widths"].items():
        for act in cfg["activations"]:
            make_curve_figure(results, cfg, width, label, act,
                              OUT_DIR / "figures" / f"ridge_real_loss_{label}_{act}.png")
    make_bar_figure(results, cfg, OUT_DIR / "figures" / "ridge_real_final_bars.png")

    dump = {f"{n}|{a}|{w}|{ac}": [{"train": c["train"], "eval": c["eval"], "eval_acc": c["eval_acc"]}
                                  for c in v]
            for (n, a, w, ac), v in results.items()}
    (OUT_DIR / "curves.json").write_text(json.dumps(dump))
    print(f"  saved {OUT_DIR/'curves.json'}")


if __name__ == "__main__":
    main()
