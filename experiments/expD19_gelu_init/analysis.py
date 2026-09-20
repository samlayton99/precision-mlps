"""expD19 analysis: figures and the arm-ranking tables.

    uv run --extra dev python experiments/expD19_gelu_init/analysis.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("expD19_run", Path(__file__).resolve().parent / "run.py")
DATA_DIR, FIG_DIR = R.DATA_DIR, R.FIG_DIR

ARM_COLOR = {
    "baseline": "#444444",
    "halo8": "#1f77b4",
    "static_colnorm": "#2ca02c",
    "recommended": "#d62728",
    "batchnorm": "#9467bd",
    "layernorm": "#ff7f0e",
}
ARM_LABEL = {
    "baseline": "baseline (expD17)",
    "halo8": "halo 8 / small collar",
    "static_colnorm": "static col-norm (reparam)",
    "recommended": "recommended (halo8+colnorm+linear)",
    "batchnorm": "BatchNorm",
    "layernorm": "LayerNorm",
}
CLASS_LABEL = {"interp1d": "1-D interp", "interp2d": "2-D interp",
               "pinn_inverse": "2-D inverse PINN"}


def load_rows():
    rows = []
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    return rows


def key(r):
    return (r["class"], r["problem"], r["width"], r["activation"], r["arm"])


# ----------------------------- figure 1 -----------------------------

def fig_arms(rows):
    """Final eval rel L2 by arm: rows = (class, width), cols = problem."""
    cells = []
    for cls in R.WIDTHS:
        for w in R.WIDTHS[cls]:
            cells.append((cls, w))
    nrow, ncol = len(cells), 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.6 * nrow), squeeze=False)
    arms = R.GELU_ARMS
    xs = np.arange(len(arms))
    for i, (cls, w) in enumerate(cells):
        vals_row = []
        for j, prob in enumerate(R.PROBLEMS[cls]):
            ax = axes[i][j]
            hs = []
            for k, arm in enumerate(arms):
                r = next((x for x in rows if key(x) == (cls, prob, w, "gelu", arm)), None)
                v = r["final_err"] if r and np.isfinite(r["final_err"]) else np.nan
                hs.append(v)
                if np.isfinite(v):
                    vals_row.append(v)
                ax.bar(k, v if np.isfinite(v) else 0, color=ARM_COLOR[arm],
                       label=ARM_LABEL[arm] if (i == 0 and j == 0) else None)
                if not np.isfinite(v):
                    ax.text(k, 1, "diverged", rotation=90, ha="center",
                            va="bottom", fontsize=7, color="red")
            # tanh reference lines where the cross-check ran
            for arm, ls in (("baseline", ":"), ("recommended", "--")):
                t = next((x for x in rows if key(x) == (cls, prob, w, "tanh", arm)), None)
                if t and np.isfinite(t["final_err"]):
                    ax.axhline(t["final_err"], color="black", ls=ls, lw=1.0,
                               label=f"tanh {arm}" if (i == 0 and j == 0) else None)
                    vals_row.append(t["final_err"])
            ax.set_yscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels([a.replace("_", "\n") for a in arms],
                               fontsize=6.5, rotation=0)
            ax.grid(True, axis="y", alpha=0.3, which="both")
            ax.set_title(f"{prob}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{CLASS_LABEL[cls]} W={w}\neval rel $L_2$", fontsize=8)
        if vals_row:
            lo, hi = min(vals_row), max(vals_row)
            for j in range(ncol):
                axes[i][j].set_ylim(lo * 0.3, hi * 4)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.005), frameon=False)
    fig.suptitle("expD19: final eval error by init arm (GELU bars; black lines = tanh reference)",
                 fontsize=11, y=0.955)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "expD19_arms.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ----------------------------- figure 2 -----------------------------

def fig_colnorm(rows):
    """Feature-column norms by region and arm, at init, one panel per class."""
    W_PICK = {c: R.WIDTHS[c][-1] for c in R.WIDTHS}     # widest cell that ran
    PROB = {"interp1d": "sine", "interp2d": "sine2d", "pinn_inverse": "bratu_lam"}
    regions = [("span", "interior (sign-spanning)"),
               ("pos", "left halo / positive preact"),
               ("neg", "right halo / negative preact")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for i, cls in enumerate(R.WIDTHS):
        ax = axes[i]
        w, prob = W_PICK[cls], PROB[cls]
        for k, arm in enumerate(R.GELU_ARMS):
            r = next((x for x in rows if key(x) == (cls, prob, w, "gelu", arm)), None)
            if r is None:
                continue
            ci = r["colnorm_init"]
            for m, (tag, _) in enumerate(regions):
                v = ci.get(f"mean_{tag}", np.nan)
                if v is None or not np.isfinite(v):
                    continue
                ax.scatter(k + (m - 1) * 0.22, max(v, 1e-18), s=70,
                           marker=["o", "^", "v"][m], color=ARM_COLOR[arm],
                           edgecolor="black", linewidth=0.4, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(range(len(R.GELU_ARMS)))
        ax.set_xticklabels([a.replace("_", "\n") for a in R.GELU_ARMS],
                           fontsize=7)
        ax.grid(True, axis="y", alpha=0.3, which="both")
        ax.set_title(f"{CLASS_LABEL[cls]}  {prob}  W={w}", fontsize=9)
        if i == 0:
            ax.set_ylabel("RMS feature-column norm at init")
    handles = [plt.Line2D([], [], marker=m, ls="", color="grey",
                          markeredgecolor="black", label=lab)
               for m, (_, lab) in zip(["o", "^", "v"], regions)]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle("expD19: the scale pathology and which arms remove it "
                 "(GELU; flat across markers = balanced feature matrix)",
                 fontsize=11, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = FIG_DIR / "expD19_colnorm.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ----------------------------- figure 3 -----------------------------

def fig_pinn(rows):
    """Inverse-PINN parameter recovery per arm: the divergence test."""
    probs = R.PROBLEMS["pinn_inverse"]
    widths = R.WIDTHS["pinn_inverse"]
    fig, axes = plt.subplots(len(widths), 3, figsize=(13, 3.6 * len(widths)),
                             squeeze=False)
    for i, w in enumerate(widths):
        for j, prob in enumerate(probs):
            ax = axes[i][j]
            for arm in R.GELU_ARMS:
                r = next((x for x in rows if key(x) == ("pinn_inverse", prob, w,
                                                        "gelu", arm)), None)
                if r is None or "param_traj" not in r:
                    continue
                tr = np.asarray(r["param_traj"], dtype=float)
                pt = r["param_true"]
                rel = np.abs(tr[:, 1] - pt) / abs(pt)
                ax.semilogy(tr[:, 0], np.maximum(rel, 1e-6), lw=1.3,
                            color=ARM_COLOR[arm],
                            label=ARM_LABEL[arm] if (i == 0 and j == 0) else None)
            ax.axhline(0.1, color="black", ls=":", lw=0.9)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_ylim(1e-4, 30)
            ax.set_title(f"{prob}  W={w}", fontsize=9)
            if j == 0:
                ax.set_ylabel("relative error in the\nrecovered PDE parameter", fontsize=8)
            if i == len(widths) - 1:
                ax.set_xlabel("iteration")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("expD19: inverse-PINN parameter recovery by arm "
                 "(dotted = 10% error; GELU)", fontsize=11, y=0.86)
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    out = FIG_DIR / "expD19_pinn_recovery.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ----------------------------- tables -----------------------------

def fmt(v, w=10, p=2):
    """`None` means the cell was not run; a non-finite value means it blew up."""
    if v is None:
        return f"{'not run':>{w}}"
    if isinstance(v, float) and not np.isfinite(v):
        return f"{'DIVERGED':>{w}}"
    return f"{v:{w}.{p}e}"


def tables(rows):
    out = {}
    print("\n" + "=" * 118)
    print("A. FINAL EVAL REL L2 by arm (GELU), and the two tanh reference arms")
    print("=" * 118)
    hdr = f"{'cell':<34}" + "".join(f"{a[:13]:>14}" for a in R.GELU_ARMS)
    print(hdr)
    for cls in R.WIDTHS:
        for w in R.WIDTHS[cls]:
            for prob in R.PROBLEMS[cls]:
                line = f"{cls[:9]+'/'+prob+'/w'+str(w):<34}"
                for arm in R.GELU_ARMS:
                    r = next((x for x in rows if key(x) == (cls, prob, w, "gelu", arm)), None)
                    line += fmt(r["final_err"] if r else None, 14)
                print(line)
    print()

    print("=" * 118)
    print("B. GEOMETRY DAMAGE: post-train lstsq probe / pre-train probe "
          "(>1 = training damaged the geometry)")
    print("=" * 118)
    print(f"{'cell':<34}" + "".join(f"{a[:13]:>14}" for a in R.GELU_ARMS))
    for cls in ["interp1d", "interp2d"]:
        for w in R.WIDTHS[cls]:
            for prob in R.PROBLEMS[cls]:
                line = f"{cls[:9]+'/'+prob+'/w'+str(w):<34}"
                for arm in R.GELU_ARMS:
                    r = next((x for x in rows if key(x) == (cls, prob, w, "gelu", arm)), None)
                    if r is None or not np.isfinite(r["post_probe"]):
                        line += f"{'--':>14}"
                    else:
                        line += f"{r['post_probe']/r['pre_probe']:14.2e}"
                print(line)
    print()

    print("=" * 118)
    print("C. DEAD NEURONS (update < 1e-10 over the first 30 snapshots), GELU")
    print("=" * 118)
    print(f"{'cell':<34}" + "".join(f"{a[:13]:>14}" for a in R.GELU_ARMS))
    for cls in R.WIDTHS:
        for w in R.WIDTHS[cls]:
            prob = R.PROBLEMS[cls][0]
            line = f"{cls[:9]+'/'+prob+'/w'+str(w):<34}"
            for arm in R.GELU_ARMS:
                r = next((x for x in rows if key(x) == (cls, prob, w, "gelu", arm)), None)
                d = R.dead_stats(r) if r else {}
                line += f"{100*d['dead_frac']:13.1f}%" if d else f"{'--':>14}"
            print(line)
    print()

    print("=" * 118)
    print("D. FEATURE-COLUMN NORM SPREAD at init (max/min over live columns), GELU")
    print("=" * 118)
    print(f"{'cell':<34}" + "".join(f"{a[:13]:>14}" for a in R.GELU_ARMS))
    for cls in R.WIDTHS:
        w = R.WIDTHS[cls][-1]
        prob = R.PROBLEMS[cls][0]
        line = f"{cls[:9]+'/'+prob+'/w'+str(w):<34}"
        for arm in R.GELU_ARMS:
            r = next((x for x in rows if key(x) == (cls, prob, w, "gelu", arm)), None)
            line += fmt(r["colnorm_init"]["max_over_min_live"] if r else None, 14)
        print(line)
    print()

    print("=" * 118)
    print("E. INVERSE-PINN PARAMETER RECOVERY, relative error % (GELU)")
    print("=" * 118)
    print(f"{'cell':<34}" + "".join(f"{a[:13]:>14}" for a in R.GELU_ARMS))
    for w in R.WIDTHS["pinn_inverse"]:
        for prob in R.PROBLEMS["pinn_inverse"]:
            line = f"{'pinn/'+prob+'/w'+str(w):<34}"
            for arm in R.GELU_ARMS:
                r = next((x for x in rows if key(x) == ("pinn_inverse", prob, w,
                                                        "gelu", arm)), None)
                if r is None or "param_final" not in r:
                    line += f"{'--':>14}"
                else:
                    rel = 100 * abs(r["param_final"] - r["param_true"]) / abs(r["param_true"])
                    line += f"{rel:13.1f}%"
            print(line)
    print()

    print("=" * 118)
    print("F. tanh CROSS-CHECK: baseline vs recommended (does the fix harm tanh?)")
    print("=" * 118)
    print(f"{'cell':<34}{'tanh base':>14}{'tanh recomm':>14}{'ratio':>10}"
          f"{'gelu base':>14}{'gelu recomm':>14}{'ratio':>10}")
    for cls in R.TANH_CLASSES:
        for w in R.WIDTHS[cls]:
            for prob in R.PROBLEMS[cls]:
                line = f"{cls[:9]+'/'+prob+'/w'+str(w):<34}"
                vals = []
                for act in ("tanh", "gelu"):
                    a = next((x for x in rows if key(x) == (cls, prob, w, act, "baseline")), None)
                    b = next((x for x in rows if key(x) == (cls, prob, w, act, "recommended")), None)
                    va = a["final_err"] if a else None
                    vb = b["final_err"] if b else None
                    line += fmt(va, 14) + fmt(vb, 14)
                    line += (f"{va/vb:10.2f}" if (va and vb and np.isfinite(va)
                                                  and np.isfinite(vb)) else f"{'--':>10}")
                print(line)
    print()

    print("=" * 118)
    print("G. PARAMETER COUNT by arm (1-D sine W=256 cell)")
    print("=" * 118)
    for arm in R.GELU_ARMS:
        r = next((x for x in rows if key(x) == ("interp1d", "sine", 256, "gelu", arm)), None)
        if r:
            print(f"  {ARM_LABEL[arm]:<44} W={r['extra']['W']:5d}  "
                  f"trainable params = {r['n_params']:6d}")
    return out


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} runs from {DATA_DIR}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tables(rows)
    fig_arms(rows)
    fig_colnorm(rows)
    fig_pinn(rows)


if __name__ == "__main__":
    main()
