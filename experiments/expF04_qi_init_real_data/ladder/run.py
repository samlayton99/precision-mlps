"""expF04/ladder -- does the D02 green line (right init + plain Adam) survive dimension?

Smooth synthetic targets at d in {1,2,4,8,16}; single-hidden-layer GELU MLP, width 256;
every arm trains BOTH layers with plain Adam (cosine lr decay). Arms differ only in the
first-layer init:

  baseline      PyTorch default
  flat          expF04 lift: M=N directions x 1 center, gamma=0.125*N/A (degenerate corner)
  ridge_p4      M x P ridge bundles, P=4     (many soft-ish ridges)
  ridge_psqrt   P=sqrt(N)=16                 (balanced)
  ridge_pn4     P=N/4=64                     (few sharp densely-packed ridges)
  scaled_psqrt  ridge_psqrt directions/gamma, RANDOM centers (D02 win-#3 analog)
  ridge_full    d=1 only: M=1, P=N -- the exact repo construction (green-line anchor)

Metric: final eval rel L2 = ||f_hat - f|| / ||f|| on held-out uniform samples.
Figures: per-target rel-L2 vs dimension (one line per arm), eval + train versions.

Run:  uv run python experiments/expF04_qi_init_real_data/ladder/run.py
      (--smoke for a 2-epoch pipeline check without wandb)
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
sys.path.insert(0, str(HERE))
from model import SimpleMLP, qi_init_, qi_ridge_init_   # noqa: E402
import targets as T                                     # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF04_qi_init_real_data" / "ladder"

ARM_COLORS = {
    "baseline": "tab:blue", "flat": "tab:red", "ridge_p4": "tab:orange",
    "ridge_psqrt": "tab:green", "ridge_pn4": "tab:purple",
    "scaled_psqrt": "tab:brown", "ridge_full": "tab:cyan",
}


def resolve_arm(arm: str, width: int) -> dict:
    """Map arm name -> init spec."""
    if arm == "baseline":
        return {"kind": "baseline"}
    if arm == "flat":
        return {"kind": "flat"}
    if arm == "ridge_full":
        return {"kind": "ridge", "P": width, "uniform": True}
    if arm.startswith("ridge_p") or arm.startswith("scaled_p"):
        tag = arm.split("_p", 1)[1]
        P = int(math.isqrt(width)) if tag == "sqrt" else (width // 4 if tag == "n4" else int(tag))
        return {"kind": "ridge", "P": P, "uniform": arm.startswith("ridge")}
    raise KeyError(arm)


def apply_init(model, spec, x_train, lam):
    if spec["kind"] == "baseline":
        return {}
    if spec["kind"] == "flat":
        return qi_init_(model, x_train, lam=lam)
    return qi_ridge_init_(model, x_train, lam=lam,
                          centers_per_dir=spec["P"], uniform_centers=spec["uniform"])


def rel_l2(model, x, y) -> float:
    with torch.no_grad():
        r = model(x).squeeze(-1) - y
        return (r.norm() / y.norm().clamp_min(1e-30)).item()


def run_config(job: dict):
    torch.set_num_threads(job["threads"])
    cfg, arm, target, d, seed = job["cfg"], job["arm"], job["target"], job["d"], job["seed"]
    width = cfg["width"]

    xtr, ytr, xte, yte = T.make_dataset(target, d, cfg["n_train"], cfg["n_eval"], seed)
    torch.manual_seed(seed)
    model = SimpleMLP(d, width, 1)
    diag = apply_init(model, resolve_arm(arm, width), xtr, cfg["lambda_star"])

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    n, bs = xtr.shape[0], cfg["batch_size"]

    wb = None
    if job["use_wandb"]:
        import wandb
        wb = wandb.init(
            project=cfg["wandb_project"], reinit=True, group=f"ladder-{target}",
            job_type=arm, name=f"ladder-{target}-d{d}-{arm}-s{seed}",
            tags=["ladder", target, arm, f"d{d}"], settings=wandb.Settings(silent=True),
            config={"subexp": "ladder", "target": target, "d": d, "arm": arm,
                    "seed": seed, "width": width,
                    **{k: cfg[k] for k in ("lr", "epochs", "batch_size", "lambda_star", "n_train")}},
        )

    curve = {"train": [], "eval_rel_l2": []}
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n)
        run_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xtr[idx]).squeeze(-1), ytr[idx])
            loss.backward(); opt.step()
            run_loss += loss.item() * idx.shape[0]
        sched.step()
        model.eval()
        curve["train"].append(run_loss / n)
        curve["eval_rel_l2"].append(rel_l2(model, xte, yte))
        if wb is not None and (epoch % 5 == 4 or epoch == cfg["epochs"] - 1):
            wb.log({"epoch": epoch, "train_mse": curve["train"][-1],
                    "eval_rel_l2": curve["eval_rel_l2"][-1]})

    final = {"final_eval_rel_l2": curve["eval_rel_l2"][-1],
             "final_train_rel_l2": rel_l2(model, xtr, ytr)}
    if wb is not None:
        wb.summary.update(final)
        wb.finish()
    return {"target": target, "d": d, "arm": arm, "seed": seed,
            **final, "diag": diag, "curve": curve}


# ----------------------------- figures -----------------------------

def make_figures(rows, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    arms = cfg["arms"] + ["ridge_full"]
    for metric, fname, title in [
        ("final_eval_rel_l2", "ladder_eval_relL2.png", "eval rel $L_2$ vs dimension"),
        ("final_train_rel_l2", "ladder_train_relL2.png", "train rel $L_2$ vs dimension (optimization floor)"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
        for ax, target in zip(axes, cfg["targets"]):
            for arm in arms:
                pts = {}
                for r in rows:
                    if r["target"] == target and r["arm"] == arm:
                        pts.setdefault(r["d"], []).append(r[metric])
                if not pts:
                    continue
                ds = sorted(pts)
                med = [float(np.median(pts[dd])) for dd in ds]
                lo = [float(np.min(pts[dd])) for dd in ds]
                hi = [float(np.max(pts[dd])) for dd in ds]
                ax.plot(ds, med, "o-", color=ARM_COLORS[arm], lw=1.8, ms=4)
                ax.fill_between(ds, lo, hi, color=ARM_COLORS[arm], alpha=0.15)
            ax.set_xscale("log", base=2); ax.set_yscale("log")
            ax.set_xticks(cfg["dims"]); ax.set_xticklabels(cfg["dims"])
            ax.set_xlabel("input dimension d")
            ax.set_title(T.PRETTY[target])
            ax.grid(True, which="both", alpha=0.25)
        axes[0].set_ylabel("rel $L_2$")
        handles = [Line2D([0], [0], color=ARM_COLORS[a], marker="o", lw=1.8, ms=4, label=a)
                   for a in arms]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=len(arms), frameon=False)
        fig.suptitle(f"expF04/ladder  --  {title}  (N={cfg['width']}, median over seeds, min-max band)",
                     y=1.12, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        path = OUT_DIR / "figures" / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    if args.smoke:
        cfg.update(dims=[1, 4], seeds=[0], epochs=2, n_train=1024, n_eval=512)
    use_wandb = not (args.no_wandb or args.smoke)
    workers = 1 if args.smoke else args.workers
    threads = max(1, (os.cpu_count() or 4) // workers)
    if use_wandb:
        os.environ.setdefault("WANDB_SILENT", "true")

    jobs = []
    for target in cfg["targets"]:
        for d in cfg["dims"]:
            arms = cfg["arms"] + (["ridge_full"] if d == 1 else [])
            for arm in arms:
                for seed in cfg["seeds"]:
                    jobs.append({"cfg": cfg, "target": target, "d": d, "arm": arm,
                                 "seed": seed, "use_wandb": use_wandb, "threads": threads})
    print(f"expF04/ladder | {len(jobs)} runs | dims={cfg['dims']} targets={cfg['targets']} "
          f"| arms={cfg['arms']}+ridge_full@d1 | seeds={cfg['seeds']} | epochs={cfg['epochs']} "
          f"| wandb={use_wandb} | workers={workers}x{threads}t")

    rows = []
    t0 = time.time()

    def _rec(r):
        rows.append(r)
        print(f"  [{len(rows)}/{len(jobs)}] {r['target']:11s} d={r['d']:<3d} {r['arm']:13s} "
              f"s{r['seed']} eval={r['final_eval_rel_l2']:.3e} train={r['final_train_rel_l2']:.3e}",
              flush=True)

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
    (OUT_DIR / "ladder_rows.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "curve"} for r in rows]))
    make_figures(rows, cfg)
    print(f"  saved {OUT_DIR/'ladder_rows.json'}")


if __name__ == "__main__":
    main()
