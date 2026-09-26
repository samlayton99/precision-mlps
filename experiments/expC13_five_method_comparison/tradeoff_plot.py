"""Isoquants, expansion paths and per-digit demands for QUILLS and ChebNet (neurons N <= 1024, p <= 53).

E_m(N, p): validation-grid relative L2 error of the validation-best network of method m with at most N
hidden neurons, at p bits (p = 9, 11, ..., 51 in the p-bit format; p = 53 native binary64 with numpy/scipy).
Kink for accuracy eps: N* = smallest N at which some p gives E <= eps; p* = smallest p with E(N*, p) <= eps.

    .venv/bin/python experiments/expC13_five_method_comparison/tradeoff_plot.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC13_five_method_comparison import common as C  # noqa: E402
from experiments.expC13_five_method_comparison.tradeoff_data import N_GRID, P_GRID  # noqa: E402

DATA = C.OUT / "data"
TARGETS = ["exp", "sine", "runge", "chirp"]
PS = P_GRID + [53]
STYLE = {"quills": ("QUILLS", "#0f7f86", 2.6), "chebnet": ("ChebNet", "#e07b00", 1.8),
         "mhaskar": ("Mhaskar", "#7b3fa0", 1.3), "staircase": ("Staircase", "#8c564b", 1.3),
         "costarelli": ("Costarelli-Spigler", "#2a9d3a", 1.3)}
OTHERS = ["mhaskar", "staircase", "costarelli"]
ISO = [1e-4, 1e-8, 1e-12]
DIGITS = [a / 2 for a in range(2, 27)]          # 1, 1.5, ..., 13 digits


def candidates():
    """(target, method, p) -> list of (neurons, validation error)."""
    out = {}
    cand = DATA / "candidates"
    for t in TARGETS:
        for p in P_GRID:
            q = [(r["width"], r["rel_l2"]) for r in json.loads((cand / f"{t}_quills_p{p}.json").read_text())
                 if r.get("status") == "ok" and r["width"] != 64]          # W = 64 is rerun with halo 16
            c = [(r["neurons"], r["rel_l2"]) for r in json.loads((cand / f"{t}_chebnet_neurons_p{p}.json").read_text())
                 if r.get("status") == "ok"]
            out[(t, "quills", p)] = q
            out[(t, "chebnet", p)] = c
        for m in OTHERS:                                            # as swept, no extra tuning; p = 53 is the sweep's
            for p in PS:
                out[(t, m, p)] = [(r["neurons"], r["rel_l2"]) for r in
                                  json.loads((cand / f"{t}_{m}_p{p}.json").read_text()) if r.get("status") == "ok"]
    for s in (DATA / "tradeoff_rows.jsonl").read_text().splitlines():
        r = json.loads(s)
        if r.get("status") != "ok" or not np.isfinite(r["val_rel_l2"]):
            continue
        out.setdefault((r["target"], r["method"], r["p"]), []).append((r["neurons"], r["val_rel_l2"]))
    return out


def error_table(cands, target, method):
    """E[i, j] for N_GRID[i], PS[j]: best validation error with at most N neurons."""
    E = np.full((len(N_GRID), len(PS)), np.inf)
    for j, p in enumerate(PS):
        rows = cands.get((target, method, p), [])
        for i, N in enumerate(N_GRID):
            ok = [e for n, e in rows if n <= N and np.isfinite(e)]
            if ok:
                E[i, j] = min(ok)
    return E


def kink(E, eps):
    feas = E <= eps
    rows = [i for i in range(len(N_GRID)) if feas[i].any()]
    if not rows:
        return None
    i = rows[0]
    j = int(np.flatnonzero(feas[i])[0])
    return N_GRID[i], PS[j]


def isoquant(E, eps):
    """(N, smallest p with E <= eps) for each N where eps is reachable."""
    pts = []
    for i, N in enumerate(N_GRID):
        js = np.flatnonzero(E[i] <= eps)
        if js.size:
            pts.append((N, PS[js[0]]))
    return pts


def _legend(fig, handles, labels, ncol, y=0.995):
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, fontsize=10)


def figure_isoquants(tables):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    h = {}
    for ax, t in zip(axes.flat, TARGETS):
        for m, (label, color, lw) in STYLE.items():
            E = tables[(t, m)]
            for eps in ISO:
                pts = isoquant(E, eps)
                if not pts:
                    continue
                xs = [n for n, _ in pts] + [N_GRID[-1]]
                ys = [p for _, p in pts] + [pts[-1][1]]
                (h["iso"],) = ax.step(xs, ys, where="post", color=color, lw=lw * 0.6, alpha=0.9)
                k = kink(E, eps)
                ax.plot(*k, "o", color=color, ms=8, mec="white", mew=1, zorder=5)
                ax.annotate(f"$10^{{{int(np.log10(eps))}}}$", k, textcoords="offset points", xytext=(6, -12),
                            fontsize=8, color=color)
            path = [kink(E, 10.0 ** -a) for a in DIGITS]
            path = [k for k in path if k]
            (h[m],) = ax.plot([n for n, _ in path], [p for _, p in path], color=color, lw=1, ls=":", marker=".", ms=5)
        h["kink"] = plt.Line2D([], [], color="0.3", marker="o", ls="none", ms=8)
        ax.set_xscale("log", base=2)
        ax.set_xlim(28, 1170)
        ax.set_xticks([32, 64, 128, 256, 512, 1024], ["32", "64", "128", "256", "512", "1024"])
        ax.set_ylim(7, 55)
        ax.set_yticks([9, 17, 25, 33, 41, 49, 53], ["9", "17", "25", "33", "41", "49", "53\n(fp64)"])
        ax.set_title(C.TITLES[t], fontsize=12)
        ax.grid(True, color="0.92", lw=0.6)
    for ax in axes[-1]:
        ax.set_xlabel("hidden neurons $N$", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("bits of precision $p$", fontsize=11)
    iso = plt.Line2D([], [], color="0.3", lw=1.2)
    handles = [plt.Line2D([], [], color=STYLE["quills"][1], lw=3), plt.Line2D([], [], color=STYLE["chebnet"][1], lw=3),
               iso, h["kink"], plt.Line2D([], [], color="0.3", lw=1, ls=":", marker=".")]
    labels = ["QUILLS", "ChebNet", "isoquant ($10^{-4}, 10^{-8}, 10^{-12}$)", "kink", "expansion path"]
    _legend(fig, handles, labels, ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def figure_demands(tables):
    """y: target error eps; x: neurons at the kink (top) or bits at the kink (bottom)."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharey=True)
    for j, t in enumerate(TARGETS):
        for m, (label, color, lw) in STYLE.items():
            E = tables[(t, m)]
            ks = [(10.0 ** -a, kink(E, 10.0 ** -a)) for a in DIGITS]
            ks = [(eps, k) for eps, k in ks if k]
            if not ks:
                continue
            z = 5 if m == "quills" else 4
            axes[0, j].plot([k[0] for _, k in ks], [eps for eps, _ in ks], color=color, lw=lw, marker="o", ms=3.5,
                            label=label, zorder=z)
            axes[1, j].plot([k[1] for _, k in ks], [eps for eps, _ in ks], color=color, lw=lw, marker="o", ms=3.5,
                            label=label, zorder=z)
        pr = np.arange(9, 54)
        axes[1, j].plot(pr, 2.0 ** -pr, color="0.6", lw=1, ls=":", label="unit roundoff $2^{-p}$")
        axes[0, j].set_xscale("log", base=2)
        axes[0, j].set_xlim(28, 1170)
        axes[0, j].set_xticks([32, 64, 128, 256, 512, 1024], ["32", "64", "128", "256", "512", "1024"])
        axes[1, j].set_xlim(7, 55)
        axes[1, j].set_xticks([9, 17, 25, 33, 41, 49, 53], ["9", "17", "25", "33", "41", "49", "53\n(fp64)"])
        axes[0, j].set_title(C.TITLES[t], fontsize=12)
        axes[0, j].set_xlabel("neurons at the kink, $N^*$")
        axes[1, j].set_xlabel("bits at the kink, $p^*$")
        for i in range(2):
            axes[i, j].set_yscale("log")
            axes[i, j].set_ylim(1e-14, 1)
            axes[i, j].grid(True, color="0.92", lw=0.6)
    axes[0, 0].set_ylabel("target error $\\varepsilon$")
    axes[1, 0].set_ylabel("target error $\\varepsilon$")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    _legend(fig, handles, labels, ncol=6)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main():
    cands = candidates()
    tables = {(t, m): error_table(cands, t, m) for t in TARGETS for m in STYLE}
    summary = {}
    for t in TARGETS:
        for m in STYLE:
            summary[f"{t}/{m}"] = {a: kink(tables[(t, m)], 10.0 ** -a) for a in DIGITS}
            print(t, m, {a: k for a, k in summary[f"{t}/{m}"].items() if k})
    (DATA / "tradeoff_kinks.json").write_text(json.dumps(summary, indent=1) + "\n")
    out = C.OUT / "figures"
    for name, fig in (("tradeoff_demands", figure_demands(tables)),):
        fig.savefig(out / f"{name}.png", dpi=160)
        plt.close(fig)
    print("figures:", out)


if __name__ == "__main__":
    main()
