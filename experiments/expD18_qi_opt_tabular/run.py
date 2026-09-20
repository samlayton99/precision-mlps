"""expD18 -- the win-condition experiment: {Adam, QI optimizer} x
{QI-inspired init, standard init} on real tabular regression.

Per PROGRAM_FRAMING.md section 7 item 3. Six regression tasks from the
expF04/all20 cache; init = expF04's best variant (scaled_psqrt: sqrt(N) ridge
bundles, per-ridge gamma, random projection-sampled centers) vs PyTorch
default; optimizer = plain Adam vs the minimal assembled QI optimizer
(qi_opt.py). Scoring reads the three PROGRAM_FRAMING section 4.3 comparisons
off two lstsq probes (pre-train and post-train geometry, solved on train,
scored on test).

Run:  uv run --extra dev python experiments/expD18_qi_opt_tabular/run.py
      [--tasks a,b] [--seeds 0,1,2] [--steps 4000] [--workers 8] [--figures-only]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

torch.set_default_dtype(torch.float64)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qi_opt = _load("d18_qi_opt", HERE / "qi_opt.py")
f04_model = _load("d18_f04_model", REPO / "experiments" / "expF04_qi_init_real_data" / "model.py")

OUT = REPO / "results" / "checkpoint_D_optimizers" / "expD18_qi_opt_tabular"
DATA_OUT = OUT / "data"
FIGS = OUT / "figures"
CACHE = REPO / "data" / "cache_all20"

TASKS = ["superconductivity", "sarcos", "airfoil", "parkinsons", "bike_sharing", "beijing_pm25"]
WIDTH = 256
HOLDOUT_EVERY = 10

INIT_ARMS = ["std", "qi"]
OPT_ARMS = ["adam", "qiopt"]
COLORS = {"adam": "tab:blue", "qiopt": "tab:red"}
STYLES = {"std": "--", "qi": "-"}


def load_task(name):
    d = torch.load(CACHE / f"{name}.pt", weights_only=True)
    assert d["kind"] == "regr", name
    xtr = d["xtr"].double()
    ytr = d["ytr"].double().ravel()
    xte = d["xte"].double()
    yte = d["yte"].double().ravel()
    ho = torch.zeros(xtr.shape[0], dtype=torch.bool)
    ho[HOLDOUT_EVERY // 2::HOLDOUT_EVERY] = True
    return (xtr[~ho], ytr[~ho], xtr[ho], ytr[ho], xte, yte, int(d["d_in"]))


def run_cell(job):
    torch.set_num_threads(job["threads"])
    torch.set_default_dtype(torch.float64)
    task, init, opt, seed, steps = job["task"], job["init"], job["opt"], job["seed"], job["steps"]
    x_fit, y_fit, x_ho, y_ho, x_te, y_te, d_in = load_task(task)

    torch.manual_seed(seed)
    model = f04_model.SimpleMLP(d_in, WIDTH, 1, activation="tanh").double()
    if init == "qi":
        f04_model.qi_ridge_init_(model, x_fit, centers_per_dir=int(math.isqrt(WIDTH)),
                                 uniform_centers=False)

    t0 = time.time()
    pre_probe, pre_tag = qi_opt.probe(model, x_fit, y_fit, x_ho, y_ho, x_te, y_te)
    res = qi_opt.train_arm(model, opt, x_fit, y_fit, x_ho, y_ho, x_te, y_te,
                           steps=steps, seed=seed)
    post_probe, post_tag = qi_opt.probe(model, x_fit, y_fit, x_ho, y_ho, x_te, y_te)
    dt = time.time() - t0

    return {"task": task, "init": init, "opt": opt, "seed": seed,
            "pre_probe": pre_probe, "pre_tag": pre_tag,
            "post_probe": post_probe, "post_tag": post_tag,
            "own_final": res["own_final"],
            "own_final_prefinisher": res["own_final_prefinisher"],
            "finisher_tag": res.get("finisher_tag"),
            "passes": res["passes"], "steps": steps, "wall_s": dt,
            "solve_log": res["solve_log"], "hist": res["hist"],
            "probes": res["probes"]}


def median_curve(rows, key="test_rel"):
    steps = rows[0]["hist"]["step"]
    vals = np.array([r["hist"][key] for r in rows])
    return steps, np.median(vals, axis=0)


def make_figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in TASKS if any(r["task"] == t for r in rows)]

    # ---- figure 1: curves ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True)
    ymin, ymax = np.inf, 0.0
    for r in rows:
        v = np.array(r["hist"]["test_rel"])
        ymin, ymax = min(ymin, v.min()), max(ymax, v.max())
    PROBE_MARKERS = {"qi": "x", "std": "+"}  # marker style distinguishes init
    for ax, task in zip(axes.ravel(), tasks):
        for opt in OPT_ARMS:
            for init in INIT_ARMS:
                sub = [r for r in rows if r["task"] == task and r["opt"] == opt and r["init"] == init]
                if not sub:
                    continue
                s, v = median_curve(sub)
                ax.plot(s, v, color=COLORS[opt], linestyle=STYLES[init],
                        label=f"{opt} / {init} init")
                # non-invasive terminal-solve probes: what the validated
                # finisher WOULD give at this step (computed on copies)
                if sub[0].get("probes"):
                    ps = sub[0]["probes"]["step"]
                    pv = np.median(np.array([r["probes"]["test_rel"] for r in sub]), axis=0)
                    ax.plot(ps, pv, linestyle="none", marker=PROBE_MARKERS[init],
                            color=COLORS[opt], markersize=7,
                            markeredgewidth=1.6, alpha=0.85,
                            label=f"{opt} / {init}: would-be terminal solve")
        ax.set_yscale("log")
        ax.set_ylim(ymin * 0.5, ymax * 2)
        ax.set_title(task)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("eval rel $L_2$ (test)")
    axes[1, 0].set_ylabel("eval rel $L_2$ (test)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               fontsize=8)
    fig.suptitle("expD18: {Adam, QI optimizer} x {std, QI-inspired init} on tabular regression\n"
                 "lines = training trajectory; x/+ = would-be validated terminal solve every 250 steps "
                 "(median over seeds)", y=0.885)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIGS / "expD18_curves.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: the three-comparison probes ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True)
    arms = [(o, i) for o in OPT_ARMS for i in INIT_ARMS]
    xpos = np.arange(len(arms))
    for ax, task in zip(axes.ravel(), tasks):
        for xi, (opt, init) in enumerate(arms):
            sub = [r for r in rows if r["task"] == task and r["opt"] == opt and r["init"] == init]
            if not sub:
                continue
            pre = np.median([r["pre_probe"] for r in sub])
            post = np.median([r["post_probe"] for r in sub])
            own = np.median([r["own_final"] for r in sub])
            c = COLORS[opt]
            ax.plot([xi, xi], [pre, post], color=c, alpha=0.4, lw=1.5, zorder=1)
            ax.scatter([xi], [pre], facecolors="none", edgecolors=c, s=70, zorder=2,
                       label="pre-train probe" if (xi == 0 and task == tasks[0]) else None)
            ax.scatter([xi], [post], color=c, s=70, zorder=3,
                       label="post-train probe" if (xi == 0 and task == tasks[0]) else None)
            ax.scatter([xi], [own], color=c, marker="x", s=80, zorder=4,
                       label="own final readout" if (xi == 0 and task == tasks[0]) else None)
        ax.set_yscale("log")
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{o}\n{i} init" for o, i in arms], fontsize=8)
        ax.set_title(task)
        ax.grid(alpha=0.3, axis="y")
    axes[0, 0].set_ylabel("test rel $L_2$")
    axes[1, 0].set_ylabel("test rel $L_2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.suptitle("expD18 probes: geometry score = post vs pre (circles); "
                 "readout score = x vs filled circle (median over seeds)", y=0.92)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGS / "expD18_probes.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    rows_path = DATA_OUT / "rows.jsonl"

    if not args.figures_only:
        jobs = [{"task": t, "init": i, "opt": o, "seed": s,
                 "steps": args.steps, "threads": args.threads}
                for t in args.tasks.split(",")
                for i in INIT_ARMS for o in OPT_ARMS
                for s in [int(x) for x in args.seeds.split(",")]]
        rows = []
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_cell, j): j for j in jobs}
            for k, fut in enumerate(as_completed(futs)):
                r = fut.result()
                rows.append(r)
                print(f"[{k+1}/{len(jobs)}] {r['task']:18s} {r['init']:3s} {r['opt']:5s} "
                      f"s{r['seed']} own={r['own_final']:.3e} pre={r['pre_probe']:.3e} "
                      f"post={r['post_probe']:.3e} ({r['wall_s']:.0f}s)", flush=True)
        with open(rows_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"total {time.time()-t0:.0f}s -> {rows_path}")
    else:
        rows = [json.loads(l) for l in open(rows_path)]

    make_figures(rows)
    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()
