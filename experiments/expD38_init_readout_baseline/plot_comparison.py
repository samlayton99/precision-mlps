"""Sam's requested view: logarithmic steps and MSE, y ceiling exactly 1."""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/precision_d38_matplotlib")
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.expD38_init_readout_baseline.run import OUT, TITLES, config, recipe_config, select, valid_data_identity


def plot(seeds):
    cfg = config()
    recipes = select(cfg)
    runs = [json.loads(p.read_text()) for p in (OUT / "data/compare").glob("*.json")]
    runs = [r for r in runs if r.get("complete") and valid_data_identity(r["identity"], cfg) and r["identity"]["seed"] in seeds
            and recipe_config(r["identity"]["config"]) == recipe_config(cfg)
            and r["identity"]["steps"] == cfg["comparison_steps"]
            and r["identity"]["width"] == recipes[r["identity"]["task"]]["width"]
            and r["identity"]["lr"] == recipes[r["identity"]["task"]]["lr"]]
    expected = {(task, scheme, seed) for task in cfg["tasks"] for scheme in ("standard", "qi") for seed in seeds}
    actual = {(r["identity"]["task"], r["identity"]["scheme"], r["identity"]["seed"]) for r in runs}
    if actual != expected:
        raise ValueError(f"Requested seeds incomplete: {len(actual)}/{len(expected)}")
    colors = {"standard": "#1f77b4", "qi": "#e58320"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    seed_text = f"seed {seeds[0]}" if len(seeds) == 1 else f"mean of {len(seeds)} seeds"
    for split in ("train", "val", "test"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
        for ax, task in zip(axes.flat, cfg["tasks"]):
            task_runs = [r for r in runs if r["identity"]["task"] == task]
            for scheme in ("standard", "qi"):
                subset = [r for r in task_runs if r["identity"]["scheme"] == scheme]
                for field, style in (("learned", "-"), ("solved", ":")):
                    traces = [[q for q in r["trace"] if field in q and q["step"] > 0] for r in subset]
                    x = np.array([q["step"] for q in traces[0]])
                    y = np.array([[q[field][split] for q in trace] for trace in traces])
                    ax.loglog(x, y.mean(0), color=colors[scheme], ls=style, lw=2)
                    if len(seeds) > 1:
                        ax.fill_between(x, y.min(0), y.max(0), color=colors[scheme], alpha=.10)
                    # Mark clipped sampled means rather than silently removing them.
                    clipped = y.mean(0) > 1
                    ax.plot(x[clipped], np.full(clipped.sum(), .96), marker="^", ls="", ms=4,
                            color=colors[scheme], clip_on=True)
            ax.axhline(task_runs[0]["affine"][split], color="#666666", ls=":", lw=1.8)
            ax.set_xlim(1, cfg["comparison_steps"])
            # Fixed limits within each figure make the tasks easy to compare.
            ax.set_ylim(1e-4 if split == "train" else 1e-3, 1)
            ax.set_title(f"{TITLES[task]}\nwidth {recipes[task]['width']} + {recipes[task]['width']}", fontsize=12)
            ax.set_xlabel("Gradient step")
            ax.set_ylabel(f"{split.capitalize()} MSE (standardized target)")
            ax.grid(alpha=.15, which="both")
        handles = [Line2D([], [], color=colors[s], ls=st, lw=2, label=label)
                   for s, st, label in [("standard","-","Standard · trained head"),("qi","-","QI · trained head"),
                                        ("standard",":","Standard · LS head"),("qi",":","QI · LS head")]]
        handles.append(Line2D([], [], color="#666666", ls=":", label="Linear regression"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.94), ncol=5, frameon=False, fontsize=9)
        fig.suptitle(f"Initialization comparison · {split} · {seed_text} · ordinary Adam", y=.995, fontsize=15)
        note = "Step 0 omitted. Triangles mark sampled means above the MSE ceiling of 1."
        if len(seeds) > 1:
            note += " Shading: seed range, not a confidence interval."
        fig.text(.5,.012,note, ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0,.035,1,.88), h_pad=2.2)
        path = OUT / "figures" / f"comparison_{split}_loglog_capped_seeds{'-'.join(map(str, seeds))}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    args = parser.parse_args()
    plot(args.seeds)
