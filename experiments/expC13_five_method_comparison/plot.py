"""Figures for expC13 from results/.../data/summary.jsonl and the candidate tables.

    .venv/bin/python experiments/expC13_five_method_comparison/plot.py
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

DATA = C.OUT / "data"
FIG = C.OUT / "figures"
TARGETS = ["exp", "sine", "runge", "chirp"]
STYLE = {  # method: (label, color, linestyle, linewidth, zorder)
    "quills": ("QUILLS", "#0f7f86", "-", 2.4, 6),
    "quills_w1024": ("QUILLS, W = 1024 only", "#0f7f86", ":", 1.2, 5),
    "chebnet": ("ChebNet", "#e07b00", "-", 1.6, 5),
    "chebnet_paper": ("ChebNet, no normalization", "#e07b00", ":", 1.2, 4),
    "chebnet_neurons": ("ChebNet, 1024-neuron budget", "#f2b766", "-.", 1.2, 3),
    "mhaskar": ("Mhaskar, incl. maximal-order variant", "#7b3fa0", "-", 1.6, 5),
    "mhaskar_appendix": ("Mhaskar (1996), Lemma 3.2", "#7b3fa0", "--", 1.1, 4),
    "staircase": ("Staircase, tuned", "#8c564b", "-", 1.6, 4),
    "staircase_classical": ("Staircase, classical", "#8c564b", "--", 1.1, 3),
    "costarelli": ("Costarelli-Spigler, variants", "#2a9d3a", "-", 1.6, 4),
    "costarelli_appendix": ("Costarelli-Spigler, appendix", "#2a9d3a", "--", 1.1, 3),
}


def rows():
    out = {}
    for s in (DATA / "summary.jsonl").read_text().splitlines():
        r = json.loads(s)
        if r.get("status") == "ok":
            out[(r["target"], r["method"], r["p"])] = r
    return out


def series(R, target, method, key):
    ps = sorted(p for (t, m, p) in R if t == target and m == method)
    return np.array(ps), np.array([R[(target, method, p)][key] for p in ps], dtype=float)


def _legend_top(fig, handles, labels, ncol, y=1.0):
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, fontsize=9)


def precision_figure(R):
    pmax = max(p for (_, _, p) in R)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.4), sharex=True, sharey=True)
    handles = {}
    for ax, target in zip(axes.flat, TARGETS):
        p_ref = np.arange(8, pmax + 1)
        ax.plot(p_ref, 2.0 ** -p_ref, color="0.55", lw=1, ls=(0, (1, 2)), zorder=1)
        handles["u"] = ax.lines[-1]
        if pmax > 53:
            ax.axvspan(53.5, pmax + 2, color="0.94", zorder=0)
        for method, (label, color, ls, lw, z) in STYLE.items():
            p, e = series(R, target, method, "rel_l2")
            if p.size == 0:
                continue
            (h,) = ax.plot(p, np.maximum(e, 1e-300), color=color, ls=ls, lw=lw, zorder=z)
            ext = p > 53                                    # sparse points: mark where data exist
            ax.plot(p[ext], np.maximum(e[ext], 1e-300), color=color, ls="none", marker="o", ms=2.5, zorder=z)
            handles[method] = h
        ax.set_yscale("log")
        ax.set_title(C.TITLES[target], fontsize=11)
        ax.grid(True, which="major", color="0.9", lw=0.6)
        ax.set_xlim(7, pmax + 1)
        ax.set_ylim(2.0 ** -(pmax + 8), 10)
    for ax in axes[-1]:
        ax.set_xlabel("significand bits $p$ (every construction and inference operation)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"relative $L^2$ error (8001-point grid)")
    order = [m for m in STYLE if m in handles] + ["u"]
    labels = [STYLE[m][0] if m in STYLE else r"unit roundoff $2^{-p}$" for m in order]
    _legend_top(fig, [handles[m] for m in order], labels, ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    if pmax > 53:
        axes[0, 0].text(54.5, 1, "MPFR records", fontsize=8, color="0.4", va="top")
    return fig


def diagnostics_figure(R):
    pmax = max(p for (_, _, p) in R)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), sharex=True)
    handles = {}
    top = max(np.log10(r["max_abs"]) for r in R.values()) + 2           # one scale for all panels
    for j, target in enumerate(TARGETS):
        for method, (label, color, ls, lw, z) in STYLE.items():
            p, size = series(R, target, method, "params")
            if p.size == 0:
                continue
            (h,) = axes[0, j].plot(p, size, color=color, ls=ls, lw=lw, zorder=z)
            handles[method] = h
            p, big = series(R, target, method, "max_abs")
            axes[1, j].plot(p, np.log10(big), color=color, ls=ls, lw=lw, zorder=z)
        axes[0, j].axhline(C.PARAM_BUDGET, color="0.5", lw=0.8, ls=":")
        axes[0, j].set_title(C.TITLES[target], fontsize=11)
        axes[0, j].set_yscale("log")
        axes[0, j].set_ylim(5, 10000)
        axes[1, j].set_ylim(-1, top)
        axes[1, j].set_xlabel("significand bits $p$")
        for i in range(2):
            axes[i, j].grid(True, color="0.9", lw=0.6)
            axes[i, j].set_xlim(7, pmax + 1)
    axes[0, 0].set_ylabel("nonzero parameters of the selected network")
    axes[1, 0].set_ylabel(r"$\log_{10}$ largest |weight or bias|")
    order = [m for m in STYLE if m in handles]
    _legend_top(fig, [handles[m] for m in order], [STYLE[m][0] for m in order], ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def size_figure(p=53):
    """Validation error of the best candidate with at most a given number of parameters, at one p."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    handles = {}
    for ax, target in zip(axes, TARGETS):
        for method, (label, color, ls, lw, z) in STYLE.items():
            path = DATA / "candidates" / f"{target}_{method}_p{p}.json"
            if not path.exists():
                continue
            tab = [r for r in json.loads(path.read_text()) if r.get("status") == "ok"]
            if not tab:
                continue
            tab.sort(key=lambda r: r["params"])
            size = np.array([r["params"] for r in tab])
            env = np.minimum.accumulate(np.array([r["rel_l2"] for r in tab]))
            if len(tab) == 1:                               # a single candidate (QUILLS): a marker
                (h,) = ax.plot(size, env, color=color, marker="o", ms=7, ls="none", zorder=z)
            else:
                (h,) = ax.step(size, env, where="post", color=color, ls=ls, lw=lw, zorder=z)
            handles[method] = h
        ax.axvline(C.PARAM_BUDGET, color="0.5", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(8, 10000)
        ax.set_ylim(1e-17, 10)
        ax.set_title(C.TITLES[target], fontsize=11)
        ax.set_xlabel("nonzero parameters (at most)")
        ax.grid(True, color="0.9", lw=0.6)
    axes[0].set_ylabel(f"best validation relative $L^2$ error, p = {p}")
    order = [m for m in STYLE if m in handles]
    _legend_top(fig, [handles[m] for m in order], [STYLE[m][0] for m in order], ncol=4, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig



SIMPLE = {  # one line per method: its best configuration within the 3073-parameter budget
    "quills": ("QUILLS", "#0f7f86"),
    "chebnet": ("ChebNet", "#e07b00"),
    "mhaskar": ("Mhaskar", "#7b3fa0"),
    "staircase": ("Staircase", "#8c564b"),
    "costarelli": ("Costarelli-Spigler", "#2a9d3a"),
}


def simple_precision(R):
    """Error against bits, five methods, the reporting grid, the parameter budget."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), sharex=True, sharey=True)
    handles = {}
    for ax, target in zip(axes.flat, TARGETS):
        pr = np.arange(8, 129)
        (handles["u"],) = ax.plot(pr, 2.0 ** -pr, color="0.6", lw=1, ls=":")
        ax.axvline(53, color="0.85", lw=1, zorder=0)
        for method, (label, color) in SIMPLE.items():
            p, e = series(R, target, method, "rel_l2")
            (handles[method],) = ax.plot(p, e, color=color, lw=2.6 if method == "quills" else 1.8,
                                         marker="o", ms=2.5, zorder=5 if method == "quills" else 4)
        ax.set_yscale("log")
        ax.set_ylim(1e-40, 10)
        ax.set_xlim(6, 130)
        ax.set_title(C.TITLES[target], fontsize=12)
        ax.grid(True, color="0.92", lw=0.6)
    for ax in axes[-1]:
        ax.set_xlabel("bits of precision $p$ (every operation)", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("relative error", fontsize=11)
    order = list(SIMPLE) + ["u"]
    labels = [SIMPLE[m][0] if m in SIMPLE else "unit roundoff $2^{-p}$" for m in order]
    _legend_top(fig, [handles[m] for m in order], labels, ncol=6, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def simple_size():
    """Error against nonzero parameters at p = 53 (same metric and selection rule as simple_precision;
    the point at the budget is the precision figure's p = 53 value)."""
    D = json.loads((DATA / "size_curves_p53.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), sharex=True, sharey=True)
    handles = {}
    for ax, target in zip(axes.flat, TARGETS):
        ax.axvline(C.PARAM_BUDGET, color="0.5", lw=1, ls=":")
        ax.text(C.PARAM_BUDGET * 1.05, 3, "budget", fontsize=9, color="0.4")
        for method, (label, color) in SIMPLE.items():
            pts = D[target][method]
            x = np.array([pt["cap"] for pt in pts], dtype=float)      # the size allowed
            y = np.array([pt["rel_l2"] for pt in pts])
            inside = x <= C.PARAM_BUDGET
            (handles[method],) = ax.plot(x[inside], y[inside], color=color, lw=2.6 if method == "quills" else 1.8,
                                         marker="o", ms=4, zorder=5 if method == "quills" else 4)
            if (~inside).any():                      # ChebNet past the budget
                xs = np.concatenate([x[inside][-1:], x[~inside]])
                ys = np.concatenate([y[inside][-1:], y[~inside]])
                (handles["beyond"],) = ax.plot(xs, ys, color=color, lw=1.4, ls="--", marker="o", ms=3,
                                               mfc="white", zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10, 1e4)
        ax.set_ylim(1e-16, 10)
        ax.set_title(C.TITLES[target], fontsize=12)
        ax.grid(True, color="0.92", lw=0.6)
    for ax in axes[-1]:
        ax.set_xlabel("size allowed (nonzero weights and biases)", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("relative error at $p = 53$", fontsize=11)
    order = list(SIMPLE) + ["beyond"]
    labels = [SIMPLE[m][0] if m in SIMPLE else "ChebNet past the budget" for m in order]
    _legend_top(fig, [handles[m] for m in order], labels, ncol=6, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "detail").mkdir(exist_ok=True)
    R = rows()
    figs = [("precision", simple_precision(R)), ("size_p53", simple_size()),
            ("detail/precision_all_variants", precision_figure(R)), ("detail/selected_networks", diagnostics_figure(R)),
            ("detail/validation_error_vs_size_p53", size_figure(53))]
    for name, fig in figs:
        fig.savefig(FIG / f"{name}.png", dpi=160)
        plt.close(fig)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
