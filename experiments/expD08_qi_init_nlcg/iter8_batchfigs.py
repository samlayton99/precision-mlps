"""iteration_8 standing batching figures (spec: iteration_8/
fable_thoughts.md rev 4, Phase 3). Auto-tether always, never fixed.

Figure 1 -- SOFT WIN, constant information volume: one fixed pool of
8012 points on [-1,1]; random batches each step at fractions
{1, 3/4, 1/2, 1/4, 1/16, 1/64} of the pool; NO accumulation. Best eval
rel L2 vs batch fraction, pool lstsq floor dashed, archived iter7-style
per-batch slim as the gray "before". Win = smooth decay, not a cliff.

Figure 2 -- STRONG WIN, interpolation bar chart: per target, three bars
+ floor line: (i) full batch n=2003, (ii) pool 8000, one random
2003-batch per step (no accumulation), (iii) pool 8000, carried exact
accumulation. Win = (ii) approaching (i)/(iii).

Run:  uv run python experiments/expD08_qi_init_nlcg/iter8_batchfigs.py
      [--figure 1|2|all] [--workers 6] [--iters 1500] [--render-only]
Outputs under results/.../iteration_8/batching/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run  # noqa: E402
import batching  # noqa: E402

OUT = run.OUT_DIR / "batching"
FNS = run.TARGETS3
POOL_SOFT = 8012
FRACS = [("1", 1), ("3/4", None), ("1/2", 2), ("1/4", 4),
         ("1/16", 16), ("1/64", 64)]      # None -> custom 3/4 divisor
POOL_STRONG = 8000
B_STRONG = 2003


def soft_case(job):
    div = job["div"]
    n_tr = POOL_SOFT
    bsize = int(round(n_tr * 0.75)) if div is None else -(-n_tr // div)
    r = batching.batch_case({
        "fn": job["fn"], "N": 256, "blabel": job["blabel"],
        "div": (1 if bsize >= n_tr else 2), "bsize": bsize,
        "opt": "iter9", "iters": job["iters"], "threads": job["threads"],
        "n_train": n_tr, "save_snaps": False})
    return {"fn": r["fn"], "blabel": job["blabel"], "bsize": r["bsize"],
            "best_rel": r["best_rel"], "floor": r["floor"],
            "rels": r["rels"], "wall_s": r["wall_s"]}


def strong_case(job):
    arm = job["arm"]
    if arm == "full2003":
        n_tr, div, bs = 2003, 1, 2003
    elif arm == "pool_random":
        n_tr, div, bs = POOL_STRONG, 4, B_STRONG   # random 2003 per step
    else:  # pool_accum: exact aggregation == the full-batch step on the
        #   pool, computed by definition (carried mode is mathematically
        #   identical) -- run the current core at div=1 so the AUTO tether is used,
        #   per Sam's rule (never the fixed tether in batching figures)
        n_tr, div, bs = POOL_STRONG, 1, POOL_STRONG
    r = batching.batch_case({
        "fn": job["fn"], "N": 256, "blabel": arm, "div": div,
        "bsize": bs, "opt": "iter9", "iters": job["iters"],
        "threads": job["threads"], "n_train": n_tr, "save_snaps": False})
    return {"fn": r["fn"], "arm": arm, "best_rel": r["best_rel"],
            "floor": r["floor"], "rels": r["rels"], "wall_s": r["wall_s"]}


def render_soft(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # archived iter7-style per-batch slim rows (nohalo, 10000-pool) as
    # the gray "before" -- floor-relative shapes comparable, but n and
    # pool differ; stated on the figure.
    gray = {}
    try:
        arch = json.loads((run.OUT_DIR.parent / "batching" / "slim"
                           / "batching_rows.json").read_text())
        for r in arch:
            if r["N"] == 256:
                gray.setdefault(r["fn"], {})[r["blabel"]] = r
    except FileNotFoundError:
        pass

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    xs = np.arange(len(FRACS))
    for k, fn in enumerate(FNS):
        ax = axes.flat[k]
        rs = {r["blabel"]: r for r in rows if r["fn"] == fn}
        ys = [rs[lb]["best_rel"] if lb in rs else np.nan
              for lb, _ in FRACS]
        ax.semilogy(xs, ys, "o-", color="#1f77b4", lw=1.8, ms=5,
                    label=f"iteration_{run.ITERATION} (per-batch, no accumulation)")
        if fn in gray:
            glabels = {"B": 0, "B/4": 3, "B/16": 4, "B/64": 5}
            gx = [glabels[lb] for lb in gray[fn] if lb in glabels]
            gy = [gray[fn][lb]["best_rel"] for lb in gray[fn]
                  if lb in glabels]
            o = np.argsort(gx)
            ax.semilogy(np.array(gx)[o], np.array(gy)[o], "s--",
                        color="#999999", lw=1.2, ms=4,
                        label="iter7-style per-batch slim "
                              "(archived, 10000-pool)")
        if rs:
            ax.axhline(next(iter(rs.values()))["floor"], color="k",
                       ls=":", lw=1.2, label="pool lstsq floor")
        ax.set_xticks(xs)
        ax.set_xticklabels([lb for lb, _ in FRACS])
        ax.set_ylim(1e-16, 1e0)
        ax.set_title(fn, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        if k >= 2:
            ax.set_xlabel("batch fraction of the 8012-point pool")
        if k % 2 == 0:
            ax.set_ylabel("best eval rel L2")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9)
    fig.suptitle("soft win: batch size vs error at constant information "
                 "volume (N=256, auto-tether, random batches, "
                 "no accumulation)", y=0.92, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = OUT / "softwin_batchsize_vs_error.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140)
    print("saved", p)


def render_strong(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = ["full2003", "pool_random", "pool_accum"]
    labels = ["full batch\nn=2003", "8000 pool\nrandom 2003/step",
              "8000 pool\ncarried accumulation"]
    colors = ["#555555", "#1f77b4", "#2ca02c"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    for k, fn in enumerate(FNS):
        ax = axes[k]
        rs = {r["arm"]: r for r in rows if r["fn"] == fn}
        for i, arm in enumerate(arms):
            if arm in rs:
                ax.bar(i, max(rs[arm]["best_rel"], 1e-16), color=colors[i],
                       width=0.62)
                ax.text(i, max(rs[arm]["best_rel"], 1e-16) * 1.6,
                        f"{rs[arm]['best_rel']:.0e}", ha="center",
                        fontsize=7)
        if "full2003" in rs:
            ax.axhline(rs["full2003"]["floor"], color="k", ls=":", lw=1.2)
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 1e0)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(fn, fontsize=10)
        ax.grid(alpha=0.3, axis="y", which="both")
    axes[0].set_ylabel("best eval rel L2")
    fig.suptitle("strong win: interpolation-regime bar chart (N=256, "
                 "auto-tether; dotted = n=2003 lstsq floor; win = middle "
                 "bar approaching its neighbors)", y=1.02, fontsize=11)
    fig.tight_layout()
    p = OUT / "strongwin_interpolation_bars.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", default="all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()
    threads = max(1, (os.cpu_count() or 4) // args.workers)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.render_only:
        render_soft(json.loads((OUT / "soft_rows.json").read_text()))
        render_strong(json.loads((OUT / "strong_rows.json").read_text()))
        return

    if args.figure in ("1", "all"):
        jobs = [{"fn": f, "blabel": lb, "div": dv, "iters": args.iters,
                 "threads": threads} for f in FNS for lb, dv in FRACS]
        rows, t0 = [], time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(soft_case, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                rows.append(r)
                print(f"[soft {i}/{len(jobs)}] {r['fn']:9s} "
                      f"frac={r['blabel']:4s} b={r['bsize']} "
                      f"best={r['best_rel']:.2e} floor={r['floor']:.1e} "
                      f"({r['wall_s']}s)", flush=True)
        (OUT / "soft_rows.json").write_text(json.dumps(rows))
        render_soft(rows)
        print(f"soft win done in {time.time()-t0:.0f}s")

    if args.figure in ("2", "all"):
        jobs = [{"fn": f, "arm": a, "iters": args.iters,
                 "threads": threads}
                for f in FNS
                for a in ("full2003", "pool_random", "pool_accum")]
        rows, t0 = [], time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(strong_case, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                rows.append(r)
                print(f"[strong {i}/{len(jobs)}] {r['fn']:9s} "
                      f"{r['arm']:12s} best={r['best_rel']:.2e} "
                      f"floor={r['floor']:.1e} ({r['wall_s']}s)",
                      flush=True)
        (OUT / "strong_rows.json").write_text(json.dumps(rows))
        render_strong(rows)
        print(f"strong win done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
