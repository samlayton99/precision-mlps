"""Compare the 2-layer sweep at N=256 vs N=512 (expF04/all20_2layers).

Reads all20_2l_rows_N256.json and all20_2l_rows_N512.json. For every config and width,
computes the geo-mean best-eval-loss ratio against a SINGLE common reference -- the
256 tanh-baseline per task -- so 256 and 512 land on the same scale and the wider-
baseline bar itself shows whether width alone helps. Panels: overall / regression(MSE)
/ classification(CE).

Run:  uv run python experiments/expF04_qi_init_real_data/all20_2layers/compare_widths.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "all20"))
import datasets as DS   # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF04_qi_init_real_data" / "all20_2layers"
ACTS = ["tanh", "gelu"]
INITS = ["baseline", "qi1", "qi2"]
CFGS = [f"{a}-{s}" for a in ACTS for s in INITS]
WIDTHS = [256, 512]
WCOLOR = {256: "tab:blue", 512: "tab:orange"}


def load(width):
    p = OUT_DIR / f"all20_2l_rows_N{width}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run run.py --width {width} first")
    return json.load(open(p))


def mean_best(rows, name, act, scheme):
    v = [r["best_eval"] for r in rows if r["name"] == name and r["act"] == act and r["scheme"] == scheme]
    return float(np.mean(v)) if v else None


def gm(x):
    x = [v for v in x if v is not None and v > 0]
    return float(np.exp(np.mean(np.log(x)))) if x else np.nan


def main():
    data = {w: load(w) for w in WIDTHS}
    names = [n for n in DS.TASK_ORDER if any(r["name"] == n for r in data[256])]
    kind = {n: [r["kind"] for r in data[256] if r["name"] == n][0] for n in names}
    ref = {n: mean_best(data[256], n, "tanh", "baseline") for n in names}   # common denominator

    def panel_vals(subset):
        # {config: {width: geo-mean ratio vs 256 tanh-baseline}}
        out = {}
        for a in ACTS:
            for s in INITS:
                cfg = f"{a}-{s}"
                out[cfg] = {}
                for w in WIDTHS:
                    ratios = [mean_best(data[w], n, a, s) / ref[n] for n in subset if ref[n]]
                    out[cfg][w] = gm(ratios)
        return out

    subsets = [("overall", names),
               ("regression (MSE)", [n for n in names if kind[n] == "regr"]),
               ("classification (CE)", [n for n in names if kind[n] == "classif"])]

    # ---- print table ----
    print("geo-mean best-eval ratio vs 256 tanh-baseline (lower = better):\n")
    for title, sub in subsets:
        pv = panel_vals(sub)
        print(f"[{title}]  ({len(sub)} tasks)")
        print(f"  {'config':16s} {'N=256':>8s} {'N=512':>8s}  {'512/256':>8s}")
        for cfg in sorted(CFGS, key=lambda c: pv[c][512]):
            r256, r512 = pv[cfg][256], pv[cfg][512]
            print(f"  {cfg:16s} {r256:>8.3f} {r512:>8.3f}  {r512/r256:>8.3f}")
        print()

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    x = np.arange(len(CFGS))
    bw = 0.38
    for ax, (title, sub) in zip(axes, subsets):
        pv = panel_vals(sub)
        for k, w in enumerate(WIDTHS):
            vals = [pv[c][w] for c in CFGS]
            ax.bar(x + (k - 0.5) * bw, vals, bw, color=WCOLOR[w], label=f"N={w}")
        ax.axhline(1.0, color="k", lw=1, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(CFGS, rotation=45, ha="right")
        ax.set_title(f"{title}  ({len(sub)} tasks)")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("geo-mean best-eval ratio\n(vs 256 tanh-baseline; lower = better)")
    fig.legend(handles=[Patch(color=WCOLOR[w], label=f"N={w}") for w in WIDTHS],
               loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.suptitle("expF04/all20_2layers -- width 256 vs 512, all 6 configs "
                 "(2 hidden layers, common 256 tanh-baseline reference)", y=1.07)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUT_DIR / "figures" / "all20_2l_compare_256_512.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")


if __name__ == "__main__":
    main()
