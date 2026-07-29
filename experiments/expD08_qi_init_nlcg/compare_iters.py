"""Compare iteration_5 (fixed tether T=1e6), iteration_6 (auto-tether, rel-trip,
ratchet 1e2), iteration_7 (auto-tether, noise-trip, ratchet 1e4) on the full
30-run grid. Outputs to results/.../iteration_7/:
  compare_scaling_2x3.png     best_rel vs N per target, all three + floor
  compare_trajectories.png    rels vs iteration, N in {256,512}, iter6 vs iter7

Run: uv run python experiments/expD08_qi_init_nlcg/compare_iters.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "results/checkpoint_D_optimizers/expD08_qi_init_nlcg"
OUT = BASE / "iteration_9"

ITERS = {
    "iteration_5": ("fixed tether T=1e6 (iter5)", "#999999"),
    "iteration_6": ("auto-tether v4 (iter6)", "#d62728"),
    "iteration_7": ("auto-tether noise-trip (iter7)", "#1f77b4"),
    "iteration_8": ("continuous controller (iter8)", "#2ca02c"),
    "iteration_9": ("no-B + ceiling + recycling (iter9)", "#9467bd"),
}
FNS = ["sine", "sine_8pi", "sine_mixture", "runge", "exp", "abs_cubed"]
NS = [32, 64, 128, 256, 512]


def load():
    data = {}
    for it in ITERS:
        rows = json.loads((BASE / it / "expD08_rows.json").read_text())
        data[it] = {(r["fn"], r["N"]): r for r in rows}
    return data


def scaling(data):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    for k, fn in enumerate(FNS):
        ax = axes.flat[k]
        floors = [data["iteration_7"][(fn, n)]["floor"] for n in NS]
        ax.plot(NS, floors, "k--", lw=1.2, label="lstsq floor")
        for it, (lab, col) in ITERS.items():
            ys = [data[it][(fn, n)]["best_rel"] for n in NS]
            ax.plot(NS, ys, "o-", color=col, lw=1.6, ms=4, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 1e0)
        ax.set_title(fn)
        ax.grid(alpha=0.3, which="both")
        if k >= 3:
            ax.set_xlabel("width N")
        if k % 3 == 0:
            ax.set_ylabel("best eval rel L2")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=4)
    fig.suptitle("iteration_5 / 6 / 7 / 8 / 9 on the 30-run grid", y=0.92)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = OUT / "compare_scaling_2x3.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("saved", p)


def trajectories(data):
    ns = [256, 512]
    fig, axes = plt.subplots(len(ns), len(FNS), figsize=(22, 7),
                             sharex=True, sharey=True)
    for i, n in enumerate(ns):
        for j, fn in enumerate(FNS):
            ax = axes[i, j]
            for it in ["iteration_7", "iteration_9"]:
                lab, col = ITERS[it]
                r = data[it][(fn, n)]
                rels = np.array(r["rels"], dtype=float)
                xs = np.linspace(0, 3000, len(rels))
                ax.plot(xs, rels, color=col, lw=1.0, label=lab)
            ax.axhline(data["iteration_7"][(fn, n)]["floor"], color="k",
                       ls=":", lw=1.0, label="floor")
            ax.set_yscale("log")
            ax.set_ylim(1e-16, 1e0)
            ax.grid(alpha=0.3, which="both")
            if i == 0:
                ax.set_title(fn, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"N={n}\neval rel L2")
            if i == len(ns) - 1:
                ax.set_xlabel("iteration")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = OUT / "compare_trajectories.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("saved", p)


def summary(data):
    print("\nmedian / per-fn log10(best_rel) gaps, iter9 minus others:")
    for other in ["iteration_5", "iteration_7", "iteration_8"]:
        gaps = []
        for fn in FNS:
            for n in NS:
                a = data["iteration_9"][(fn, n)]["best_rel"]
                b = data[other][(fn, n)]["best_rel"]
                gaps.append(math.log10(max(a, 1e-300) / max(b, 1e-300)))
        gaps = np.array(gaps)
        print(f"  vs {other}: median {np.median(gaps):+.2f}  "
              f"mean {gaps.mean():+.2f}  wins {(gaps < -0.1).sum()}  "
              f"losses {(gaps > 0.1).sum()}  ties {(np.abs(gaps) <= 0.1).sum()}")
    print("\niter9 cells at/below 3x floor:")
    hit = sum(1 for fn in FNS for n in NS
              if data["iteration_9"][(fn, n)]["best_rel"]
              <= 3 * data["iteration_9"][(fn, n)]["floor"])
    print(f"  {hit}/30")


if __name__ == "__main__":
    d = load()
    scaling(d)
    trajectories(d)
    summary(d)
