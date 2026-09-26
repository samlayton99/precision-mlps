"""Tap-out width (the rate) and floor (the intercept) of every method at each precision; N <= 1024, p <= 53.

E(N, p): validation error of the validation-best network with at most N hidden neurons (ChebNet: neurons
summed over all its hidden layers), at p bits (p = 9, 11, ..., 51 in the p-bit format; p = 53 native binary64).
  floor        E*(p) = E(1024, p), the best error available within the width cap.
  tap-out      N*(p) = smallest N with E(N, p) <= FACTOR * E*(p): the width at which the error arrives at its floor.
  width-limited  no tap-out within the cap: two more bits improve E(1024, p) by less than P_GAIN (precision is
               not what binds) and the last half-doubling of width (724 -> 1024) still improves it by N_GAIN or more.
  no tap-out is reported where E*(p) > USEFUL (the method does not approximate the target at that precision).
The two more bits at p = 53 are not available (p <= 53); p = 53 takes the verdict of the 51 -> 53 comparison.

    .venv/bin/python experiments/expC13_five_method_comparison/tapout_plot.py
Writes data/tapout.json, figures/tapout_summary.png, figures/tapout_curves.png.
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
from experiments.expC13_five_method_comparison.tradeoff_plot import PS, TARGETS, candidates, error_table  # noqa: E402
from experiments.expC13_five_method_comparison.tradeoff_data import N_GRID  # noqa: E402

FACTOR, P_GAIN, N_GAIN, USEFUL = 10.0, 2.0, 1.5, 0.1
METHODS = {"quills": ("QUILLS", "#0f7f86", 2.4, "o"), "chebnet": ("ChebNet", "#e07b00", 2.0, "s"),
           "mhaskar": ("Mhaskar", "#7b3fa0", 1.4, "^"), "staircase": ("Staircase", "#8c564b", 1.4, "v"),
           "costarelli": ("Costarelli-Spigler", "#2a9d3a", 1.4, "D")}
CURVE_PS = [21, 33, 45, 53]
TOP = 1210                                                   # band above N = 1024: width-limited
BAND_ROW = {"staircase": 1065, "costarelli": 1120, "mhaskar": 1145}
P_TICKS = ([9, 17, 25, 33, 41, 49, 53], ["9", "17", "25", "33", "41", "49", "53\n(fp64)"])
N_TICKS = ([32, 64, 128, 256, 512, 1024], ["32", "64", "128", "256", "512", "1024"])


def tapout(E):
    """Per precision: status ('tap' | 'width' | 'none'), tap-out width N* (None unless 'tap'), floor E*."""
    NG = np.array(N_GRID)
    i724 = N_GRID.index(724)
    best = E[-1]
    rows = []
    for j, p in enumerate(PS):
        k = min(j, len(PS) - 2)                            # p = 53 uses the 51 -> 53 comparison
        precision_binds = best[k] >= P_GAIN * best[k + 1]
        if not np.isfinite(best[j]) or best[j] > USEFUL:
            status, n = "none", None
        elif not precision_binds and E[i724, j] >= N_GAIN * best[j]:
            status, n = "width", None
        else:
            status, n = "tap", int(NG[np.argmax(E[:, j] <= FACTOR * best[j])])
        rows.append({"p": p, "status": status, "tapout": n, "floor": float(best[j]),
                     "error_at_tapout": float(E[N_GRID.index(n), j]) if n else None})
    return rows


def _panel_p(ax):
    ax.set_xlim(7, 55)
    ax.set_xticks(*P_TICKS)
    ax.grid(True, color="0.92", lw=0.6)


def figure_summary(res):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6), sharex=True)
    for j, t in enumerate(TARGETS):
        top, bot = axes[0, j], axes[1, j]
        top.axhspan(1024, TOP, color="0.93", lw=0)
        top.axhspan(0, 32, color="0.93", lw=0)
        top.text(8, 1182, "width-limited: still improving at N = 1024", fontsize=7.5, color="0.35", va="center")
        for m, (label, color, lw, mk) in METHODS.items():
            rows = res[(t, m)]
            ps = np.array([r["p"] for r in rows])
            tap = np.array([r["tapout"] if r["status"] == "tap" else np.nan for r in rows], float)
            z = 5 if m == "quills" else 4
            top.plot(ps, tap, color=color, lw=lw, marker=mk, ms=4, label=label, zorder=z)
            wl = [r["status"] == "width" for r in rows]
            top.plot(ps[wl], np.full(sum(wl), BAND_ROW.get(m, 1060)), ls="none", marker=mk, ms=4.5, mfc="white",
                     mec=color, mew=1.2, zorder=z)
            floor = np.array([r["floor"] for r in rows])
            solid = np.array([r["status"] == "tap" for r in rows])
            bot.plot(ps, floor, color=color, lw=lw, zorder=z)
            bot.plot(ps[solid], floor[solid], ls="none", marker=mk, ms=4, color=color, zorder=z)
            bot.plot(ps[~solid], floor[~solid], ls="none", marker=mk, ms=4.5, mfc="white", mec=color, mew=1.2,
                     zorder=z)
        pr = np.arange(9, 54)
        bot.plot(pr, 2.0 ** -pr, color="0.55", lw=1, ls=":", label="unit roundoff $2^{-p}$")
        top.set_ylim(0, TOP)
        top.set_yticks([0, 128, 256, 384, 512, 640, 768, 896, 1024])
        bot.set_yscale("log")
        bot.set_ylim(1e-17, 10)
        top.set_title(C.TITLES[t], fontsize=12)
        bot.set_xlabel("bits of precision $p$", fontsize=11)
        _panel_p(top)
        _panel_p(bot)
    axes[0, 0].set_ylabel("tap-out width $N^*$ (hidden neurons; shaded: $\\leq 32$)", fontsize=11)
    axes[1, 0].set_ylabel("floor: best error with $N \\leq 1024$", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    handles += h2 + [plt.Line2D([], [], color="0.3", ls="none", marker="o", mfc="white", mew=1.2)]
    labels += l2 + ["open marker: width-limited or no useful fit (error is not a floor)"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def figure_curves(tables, res):
    fig, axes = plt.subplots(len(CURVE_PS), 4, figsize=(17, 3.4 * len(CURVE_PS)), sharex=True, sharey=True)
    NG = np.array(N_GRID)
    for i, p in enumerate(CURVE_PS):
        jp = PS.index(p)
        for j, t in enumerate(TARGETS):
            ax = axes[i, j]
            for m, (label, color, lw, mk) in METHODS.items():
                z = 5 if m == "quills" else 4
                ax.plot(NG, tables[(t, m)][:, jp], color=color, lw=lw, marker=mk, ms=3, label=label, zorder=z)
                r = res[(t, m)][jp]
                if r["status"] == "tap":
                    ax.plot(r["tapout"], r["error_at_tapout"], "o", ms=11, mfc="none", mec=color, mew=2, zorder=6)
            ax.axhline(2.0 ** -p, color="0.55", lw=1, ls=":", label="unit roundoff $2^{-p}$")
            ax.set_xscale("log", base=2)
            ax.set_xlim(28, 1170)
            ax.set_xticks(*N_TICKS)
            ax.set_yscale("log")
            ax.set_ylim(1e-17, 10)
            ax.grid(True, color="0.92", lw=0.6)
            ax.set_title(f"{C.TITLES[t]}, p = {p}" + (" (fp64)" if p == 53 else ""), fontsize=11)
            if i == len(CURVE_PS) - 1:
                ax.set_xlabel("hidden neurons $N$", fontsize=11)
            if j == 0:
                ax.set_ylabel("best error with $\\leq N$ neurons", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="0.3", ls="none", marker="o", ms=11, mfc="none", mew=2))
    labels.append("tap-out $N^*$")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.998), ncol=7, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def main():
    cands = candidates()
    tables = {(t, m): error_table(cands, t, m) for t in TARGETS for m in METHODS}
    res = {k: tapout(E) for k, E in tables.items()}
    (C.OUT / "data" / "tapout.json").write_text(json.dumps({f"{t}/{m}": v for (t, m), v in res.items()}, indent=1)
                                                 + "\n")
    for t in TARGETS:
        for m in ("quills", "chebnet"):
            print(t, m, " ".join(f"{r['p']}:{r['tapout'] if r['status'] == 'tap' else r['status']}"
                                 f"/{np.log2(r['floor'] * 2.0 ** r['p']):.1f}b" for r in res[(t, m)]))
    out = C.OUT / "figures"
    for name, fig in (("tapout_summary", figure_summary(res)), ("tapout_curves", figure_curves(tables, res))):
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)
    print("figures:", out)


if __name__ == "__main__":
    main()
