"""expD21 analysis: consistency ranking of the five normalization variants.

Judgement is CONSISTENCY, not best-in-class. A variant that is 5x better on some
cells and 10x worse on others is worse than one uniformly 1.5x better, so the
summary reports mean rank, WORST-CASE rank, rank variance, regret count, worst
single regret, and the spread of the ratio-to-baseline -- not just a mean.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _load("expD21_run", HERE / "run.py")

OUT_DIR = R.OUT_DIR
DATA_DIR = R.DATA_DIR
FIG_DIR = R.FIG_DIR

VARIANTS = R.VARIANTS
ACTS = R.ACTS
SHORT = {"baseline": "baseline", "rms_nocenter": "rms (no ctr)",
         "rms_center": "rms + center", "batchnorm_noaffine": "BN (no aff)",
         "layernorm_noaffine": "LN (no aff)"}
COLORS = {"baseline": "#444444", "rms_nocenter": "#1f77b4",
          "rms_center": "#d62728", "batchnorm_noaffine": "#2ca02c",
          "layernorm_noaffine": "#9467bd"}


def load_rows():
    rows = []
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        for line in open(f):
            rows.append(json.loads(line))
    return rows


def cells(rows):
    """(activation, class, problem) -> variant -> list over seeds."""
    out = {}
    for r in rows:
        key = (r["activation"], r["class"], r["problem"])
        out.setdefault(key, {}).setdefault(r["variant"], []).append(r)
    return out


def med(rs, k):
    v = [r[k] for r in rs if r.get(k) is not None and math.isfinite(r[k])]
    return float(np.median(v)) if v else float("nan")


# ----------------------------- consistency -----------------------------

def consistency(rows):
    C = cells(rows)
    keys = sorted(k for k in C if len(C[k]) == len(VARIANTS))
    ranks = {v: [] for v in VARIANTS}
    ratios = {v: [] for v in VARIANTS}
    per_cell = {}
    for k in keys:
        vals = {v: med(C[k][v], "final_err") for v in VARIANTS}
        base = vals["baseline"]
        order = sorted(VARIANTS, key=lambda v: vals[v])
        for i, v in enumerate(order):
            ranks[v].append(i + 1)
        for v in VARIANTS:
            ratios[v].append(base / vals[v] if vals[v] > 0 else np.nan)
        per_cell[k] = (vals, {v: base / vals[v] for v in VARIANTS})
    return keys, ranks, ratios, per_cell


def summarize(ranks, ratios, tag=""):
    print(f"\n{'='*94}")
    print(f"CONSISTENCY SUMMARY {tag}   (ratio = baseline_err / variant_err; >1 means better)")
    print(f"{'variant':16s} {'mean rank':>9s} {'worst rank':>10s} {'rank var':>9s} "
          f"{'med ratio':>10s} {'min ratio':>10s} {'max ratio':>10s} "
          f"{'ratio spread':>12s} {'regret':>7s} {'worst regret':>13s}")
    rowsout = {}
    for v in VARIANTS:
        rk = np.array(ranks[v], dtype=float)
        rt = np.array([x for x in ratios[v] if np.isfinite(x)], dtype=float)
        regret = int((rt < 1.0).sum())
        worst = float(rt.min()) if rt.size else np.nan
        spread = float(rt.max() / rt.min()) if rt.size and rt.min() > 0 else np.nan
        rowsout[v] = dict(mean_rank=float(rk.mean()), worst_rank=float(rk.max()),
                          rank_var=float(rk.var()), med_ratio=float(np.median(rt)),
                          min_ratio=worst, max_ratio=float(rt.max()),
                          spread=spread, regret=regret, n=len(rt))
        print(f"{SHORT[v]:16s} {rk.mean():9.2f} {int(rk.max()):10d} {rk.var():9.2f} "
              f"{np.median(rt):10.3f} {worst:10.3f} {rt.max():10.3f} "
              f"{spread:12.1f} {regret:3d}/{len(rt):<3d} {1/worst:12.2f}x")
    return rowsout


def per_activation(rows):
    out = {}
    for act in ACTS:
        sub = [r for r in rows if r["activation"] == act]
        keys, ranks, ratios, _ = consistency(sub)
        out[act] = summarize(ranks, ratios, tag=f"-- {act} only ({len(keys)} cells)")
    return out


# ----------------------------- stability over training -----------------------------

def ranking_stability(rows):
    """Does the variant ranking at 1/3 and 2/3 of training match the end?"""
    C = cells(rows)
    keys = sorted(k for k in C if len(C[k]) == len(VARIANTS))
    fracs = [1 / 3, 2 / 3, 1.0]
    orders = []
    for fr in fracs:
        rk = {v: [] for v in VARIANTS}
        for k in keys:
            vals = {}
            for v in VARIANTS:
                es = [r["evals"] for r in C[k][v]]
                pick = []
                for e in es:
                    e = [p for p in e if p[1] is not None]
                    if not e:
                        continue
                    tgt = fr * e[-1][0]
                    pick.append(min(e, key=lambda p: abs(p[0] - tgt))[1])
                vals[v] = float(np.median(pick)) if pick else np.nan
            order = sorted(VARIANTS, key=lambda v: vals[v])
            for i, v in enumerate(order):
                rk[v].append(i + 1)
        orders.append({v: float(np.mean(rk[v])) for v in VARIANTS})
    print(f"\n{'='*94}")
    print("RANKING STABILITY over training (mean rank at 1/3, 2/3, end of run)")
    print(f"{'variant':16s} {'@1/3':>8s} {'@2/3':>8s} {'@end':>8s}  {'stable?':>8s}")
    for v in VARIANTS:
        a, b, c = orders[0][v], orders[1][v], orders[2][v]
        print(f"{SHORT[v]:16s} {a:8.2f} {b:8.2f} {c:8.2f}  "
              f"{'yes' if abs(b - c) <= 0.5 else 'NO':>8s}")
    return orders


# ----------------------------- other measurements -----------------------------

def secondary(rows):
    C = cells(rows)
    print(f"\n{'='*94}")
    print("SECONDARY MEASUREMENTS (median over seeds and problems, per activation x class)")
    print(f"{'act':5s} {'class':13s} {'variant':16s} {'colspread':>10s} {'dead%':>7s} "
          f"{'rel drift':>10s} {'abs drift':>10s} {'geom dmg':>10s} {'params':>7s}")
    for act in ACTS:
        for cls in R.PROBLEMS:
            for v in VARIANTS:
                rs = [r for r in rows if r["activation"] == act
                      and r["class"] == cls and r["variant"] == v]
                if not rs:
                    continue
                dead = [R.dead_stats(r).get("dead_frac", np.nan) for r in rs]
                dmg = [r["post_probe"] / r["pre_probe"] for r in rs
                       if r["pre_probe"] > 0]
                print(f"{act:5s} {cls:13s} {SHORT[v]:16s} "
                      f"{med(rs,'colnorm_init') if False else np.median([r['colnorm_init']['max_over_min_live'] for r in rs]):10.1e} "
                      f"{100*np.nanmedian(dead):7.1f} {med(rs,'rel_drift_end'):10.2e} "
                      f"{med(rs,'abs_drift_end'):10.2e} {np.median(dmg):10.1e} "
                      f"{int(med(rs,'n_params')):7d}")


def pinn_accuracy(rows):
    print(f"\n{'='*94}")
    print("INVERSE PROBLEM ABSOLUTE ACCURACY (median over seeds; W=512)")
    print(f"{'act':5s} {'problem':12s} {'variant':16s} {'true':>8s} {'recovered':>13s} "
          f"{'abs err':>11s} {'decimals':>9s} {'field rel L2':>13s}")
    for act in ACTS:
        for prob in R.PROBLEMS["pinn_inverse"]:
            for v in VARIANTS:
                rs = [r for r in rows if r["activation"] == act
                      and r["problem"] == prob and r["variant"] == v]
                if not rs:
                    continue
                pt = rs[0]["param_true"]
                pf = float(np.median([r["param_final"] for r in rs]))
                ae = abs(pf - pt)
                dec = -math.log10(ae / abs(pt)) if ae > 0 else 99.0
                print(f"{act:5s} {prob:12s} {SHORT[v]:16s} {pt:8.4g} {pf:13.8g} "
                      f"{ae:11.2e} {dec:9.2f} {med(rs,'final_err'):13.2e}")


# ----------------------------- figures -----------------------------

def fig_ranks(rows):
    C = cells(rows)
    keys = sorted((k for k in C if len(C[k]) == len(VARIANTS)),
                  key=lambda k: (k[0], k[1], k[2]))
    M = np.full((len(VARIANTS), len(keys)), np.nan)
    for j, k in enumerate(keys):
        base = med(C[k]["baseline"], "final_err")
        for i, v in enumerate(VARIANTS):
            e = med(C[k][v], "final_err")
            if e > 0:
                M[i, j] = base / e

    fig, ax = plt.subplots(figsize=(1.05 * len(keys) + 4.5, 3.6))
    vmax = max(3.0, np.nanmax(M))
    vmin = min(1 / 3.0, np.nanmin(M[M > 0]))
    lim = max(vmax, 1 / vmin)
    im = ax.imshow(M, cmap="RdYlGn", norm=LogNorm(vmin=1 / lim, vmax=lim),
                   aspect="auto")
    ax.set_yticks(range(len(VARIANTS)))
    ax.set_yticklabels([SHORT[v] for v in VARIANTS])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k[0]}\n{k[2]}" for k in keys], fontsize=7.5,
                       rotation=90)
    for i in range(len(VARIANTS)):
        for j in range(len(keys)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        fontsize=6.4,
                        color="black" if 0.5 < M[i, j] < 4 else "white")
    nact = sum(1 for k in keys if k[0] == ACTS[0])
    ax.axvline(nact - 0.5, color="black", lw=2)
    cb = fig.colorbar(im, ax=ax, pad=0.012)
    cb.set_label("baseline err / variant err   (green = better)", fontsize=8)
    ax.set_title("expD21: consistency of five normalizations, identical parameter counts\n"
                 "a uniformly good variant shows as a uniformly green ROW "
                 f"(left of the bar = {ACTS[0]}, right = {ACTS[1]})", fontsize=9.5)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / "expD21_ranks.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


def fig_curves(rows):
    C = cells(rows)
    classes = list(R.PROBLEMS)
    nrow = len(classes) * len(ACTS)
    fig, axes = plt.subplots(nrow, 3, figsize=(13.5, 2.65 * nrow),
                             squeeze=False)
    for ri, (act, cls) in enumerate([(a, c) for a in ACTS for c in classes]):
        for ci, prob in enumerate(R.PROBLEMS[cls]):
            ax = axes[ri][ci]
            key = (act, cls, prob)
            if key not in C:
                ax.axis("off")
                continue
            for v in VARIANTS:
                rs = C[key].get(v, [])
                if not rs:
                    continue
                grid = [p[0] for p in rs[0]["evals"]]
                stack = []
                for r in rs:
                    d = {p[0]: p[1] for p in r["evals"] if p[1] is not None}
                    stack.append([d.get(g, np.nan) for g in grid])
                y = np.nanmedian(np.array(stack, dtype=float), axis=0)
                ax.plot(grid, y, color=COLORS[v], lw=1.4, label=SHORT[v])
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both", lw=0.4)
            ax.set_title(f"{act} / {prob}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"{cls}\neval rel $L_2$", fontsize=8.5)
            if ri == nrow - 1:
                ax.set_xlabel("iteration", fontsize=8.5)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               ncol=len(VARIANTS), fontsize=9, frameon=False)
    fig.suptitle("expD21: training curves, median over 3 data-realization seeds",
                 y=0.982, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.962))
    p = FIG_DIR / "expD21_curves.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} runs from {DATA_DIR}")
    # Gate: within a (class, activation) cell every variant must have the SAME
    # trainable parameter count. Counts DO differ between activations, because
    # the standard halo rule depends on lambda* (tanh 0.25 -> halo 70, gelu
    # 0.707 -> halo 51); every variant comparison is within-activation, so that
    # is not a confound. expD19's BN/LN arms carried 2W extra parameters; these
    # do not.
    ok = True
    print("PARAMETER GATE (must be one value per class x activation):")
    seen = {}
    for r in rows:
        seen.setdefault((r["class"], r["activation"]), set()).add(r["n_params"])
    for k in sorted(seen):
        v = sorted(seen[k])
        ok &= len(v) == 1
        print(f"  {k[0]:13s} {k[1]:5s} {v}  {'OK' if len(v) == 1 else 'MISMATCH'}")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL -- variants not at equal capacity'}")
    keys, ranks, ratios, _ = consistency(rows)
    summarize(ranks, ratios, tag=f"-- all cells ({len(keys)})")
    per_activation(rows)
    ranking_stability(rows)
    secondary(rows)
    pinn_accuracy(rows)
    fig_ranks(rows)
    fig_curves(rows)


if __name__ == "__main__":
    main()
