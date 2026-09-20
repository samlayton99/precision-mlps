"""Figures for expD14 iteration 0.

  T2_ablation.png    eval rel L2 vs iteration, one line per ablation arm,
                     2x2 over (init x target), floor of the initial geometry
                     dashed.
  T2_signals.png     the two measured control signals of the proposal arm --
                     alpha (the damping the drift gauge asks for) and rho (the
                     energy the gain allocator gives L) -- against iteration,
                     with the error curve above them.
  T3_solver.png      the O(d) LSQR at several tau against the reference direct
                     solve: trajectories, and best error vs passes spent.

Run: uv run --extra dev python experiments/expD14_lobotomy/iteration_0/figs0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core0 as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CELLS = [("qi", "sine"), ("qi", "runge"), ("rand", "sine"), ("rand", "runge")]
TITLE = {"qi": "QI init", "rand": "random init"}
COL = {"adam": "0.5", "lobo": "tab:red", "clip": "tab:blue",
       "adamstep": "tab:green", "nodamp": "tab:purple", "alpha1": "tab:brown",
       "rho1": "tab:orange", "rho0": "tab:cyan", "resetmom": "tab:pink",
       "adamL": "olive"}
LW = {"lobo": 2.6, "adam": 2.0}


def _grid(nrow, ncol, size=(6.0, 4.2)):
    fig, ax = plt.subplots(nrow, ncol, figsize=(size[0] * ncol, size[1] * nrow),
                           squeeze=False)
    return fig, ax


def fig_ablation(recs, path):
    arms = [a for a in COL if any(r["arm"] == a for r in recs)]
    fig, ax = _grid(2, 2)
    for k, (init, fn) in enumerate(CELLS):
        a = ax[k // 2][k % 2]
        sel = [r for r in recs if r["init"] == init and r["fn"] == fn]
        if not sel:
            continue
        f0 = sel[0]["floor0"]
        for arm in arms:
            r = next((x for x in sel if x["arm"] == arm), None)
            if r is None:
                continue
            h = r["hist"]
            a.semilogy(h["it"], np.maximum(h["rel"], 1e-17), color=COL[arm],
                       lw=LW.get(arm, 1.3), label=arm, alpha=0.9)
        a.axhline(f0, ls="--", color="k", lw=1.2)
        a.text(0.99, f0, " floor of the initial geometry", transform=a.get_yaxis_transform(),
               ha="right", va="bottom", fontsize=8)
        a.set_ylim(1e-16, 3)
        a.set_xlabel("iteration")
        a.set_ylabel(r"eval rel $L_2$")
        a.set_title(f"{TITLE[init]} -- {fn}")
        a.grid(alpha=0.25, which="both")
    h, lab = ax[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               ncol=min(len(lab), 6), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  -> {path}")


def fig_signals(recs, path, arm="lobo"):
    fig, ax = _grid(3, 2, size=(6.0, 3.0))
    for c, (init, fn) in enumerate([("qi", "sine"), ("rand", "sine")]):
        r = next((x for x in recs if x["init"] == init and x["fn"] == fn
                  and x["arm"] == arm), None)
        if r is None:
            continue
        h = r["hist"]
        a0, a1, a2 = ax[0][c], ax[1][c], ax[2][c]
        a0.semilogy(h["it"], np.maximum(h["rel"], 1e-17), color="tab:red", lw=2)
        a0.axhline(r["floor0"], ls="--", color="k", lw=1.1,
                   label="floor of the initial geometry")
        a0.set_ylabel(r"eval rel $L_2$")
        a0.set_ylim(1e-16, 3)
        a0.set_title(f"{TITLE[init]} -- {fn}", pad=26)
        a0.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1,
                  frameon=False, fontsize=8)

        a1.semilogy(h["it"], np.maximum(h["alpha"], 1e-16), color="tab:blue",
                    lw=1.6, label=r"$\alpha$ asked for by the drift gauge")
        a1.semilogy(h["it"], np.maximum(np.array(h["rel"]), 1e-17), color="0.7",
                    lw=1.0, ls=":", label=r"eval rel $L_2$ (for reference)")
        a1.set_ylabel(r"$\alpha=\sqrt{\mu}/\sigma_1$")
        a1.set_ylim(1e-16, 3)
        a1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                  frameon=False, fontsize=8)

        a2.plot(h["it"], h["rho"], color="tab:orange", lw=1.5,
                label=r"$\rho$ -- share of the step energy given to $L$")
        a2.set_ylim(-0.03, 1.03)
        a2.set_ylabel(r"$\rho$")
        a2.set_xlabel("iteration")
        a2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1,
                  frameon=False, fontsize=8)
        for a in (a0, a1, a2):
            a.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  -> {path}")


def fig_solver(recs, path):
    arms = ["direct"] + [f"lsqr{t}" for t in (1, 3, 10, 30, 100)]
    cols = dict(zip(arms, ["k", "tab:blue", "tab:green", "tab:orange",
                           "tab:red", "tab:purple"]))
    fig, ax = _grid(2, 2)
    for k, (init, fn) in enumerate(CELLS):
        a = ax[k // 2][k % 2]
        sel = [r for r in recs if r["init"] == init and r["fn"] == fn]
        if not sel:
            continue
        for arm in arms:
            r = next((x for x in sel if x["arm"] == arm), None)
            if r is None:
                continue
            h = r["hist"]
            a.semilogy(h["it"], np.maximum(h["rel"], 1e-17), color=cols[arm],
                       lw=2.2 if arm == "direct" else 1.3,
                       ls="--" if arm == "direct" else "-", label=arm)
        a.axhline(sel[0]["floor0"], ls=":", color="k", lw=1.2)
        a.set_ylim(1e-16, 3)
        a.set_xlabel("iteration")
        a.set_ylabel(r"eval rel $L_2$")
        a.set_title(f"{TITLE[init]} -- {fn}")
        a.grid(alpha=0.25, which="both")
    h, lab = ax[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               ncol=min(len(lab), 6), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  -> {path}")


def fig_t2b(recs, path):
    """The step-3 tension, in one picture: what each arm REACHES, and what the
    geometry it left behind is WORTH (= the error after one terminal solve)."""
    arms = ["adam", "nodamp", "drift", "drift_gain"]
    cols = ["0.5", "tab:purple", "tab:red", "tab:green"]
    targets = sorted({r["fn"] for r in recs},
                     key=lambda t: ["sine", "sine_8pi", "runge", "exp"].index(t))
    fig, ax = _grid(2, 2, size=(6.4, 3.8))
    for i, init in enumerate(("qi", "rand")):
        for j, (key, what) in enumerate(
                [("best_rel", "reached during the run"),
                 ("floor_final", "after one terminal exact solve")]):
            a = ax[i][j]
            w = 0.8 / len(arms)
            for k, (arm, c) in enumerate(zip(arms, cols)):
                vals, xs = [], []
                for t, fn in enumerate(targets):
                    r = next((x for x in recs if x["init"] == init
                              and x["fn"] == fn and x["arm"] == arm), None)
                    if r is None:
                        continue
                    vals.append(max(r[key], 1e-17))
                    xs.append(t + (k - (len(arms) - 1) / 2) * w)
                a.bar(xs, vals, width=w, color=c, label=arm)
            for t, fn in enumerate(targets):
                r = next((x for x in recs if x["init"] == init
                          and x["fn"] == fn), None)
                if r:
                    a.plot([t - 0.45, t + 0.45], [r["floor0"]] * 2, "k--", lw=1.2)
            a.set_yscale("log")
            a.set_ylim(1e-17, 5)
            a.set_xticks(range(len(targets)))
            a.set_xticklabels(targets, fontsize=9)
            a.set_ylabel(r"eval rel $L_2$")
            a.set_title(f"{TITLE[init]} -- {what}")
            a.grid(alpha=0.25, axis="y", which="both")
    h, lab = ax[0][0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="k", ls="--"))
    lab.append("floor of the initial geometry")
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               ncol=5, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  -> {path}")


def fig_grid(recs, path):
    """T4: error vs width, one panel per target, arms overlaid, both inits."""
    targets = ["sine", "sine_8pi", "runge", "sine_mixture", "exp", "abs_cubed"]
    arms = ["adam", "lobo", "lobo_soft"]
    cols = {"adam": "0.5", "lobo": "tab:red", "lobo_soft": "tab:green"}
    for init in ("qi", "rand"):
        fig, ax = _grid(2, 3, size=(4.6, 3.6))
        for t, fn in enumerate(targets):
            a = ax[t // 3][t % 3]
            for arm in arms:
                sel = sorted([r for r in recs if r["init"] == init
                              and r["fn"] == fn and r["arm"] == arm],
                             key=lambda r: r["N"])
                if not sel:
                    continue
                Ns = [r["N"] for r in sel]
                a.loglog(Ns, [max(r["best_rel"], 1e-17) for r in sel], "o-",
                         color=cols[arm], label=arm, lw=1.8)
                a.loglog(Ns, [max(r["floor_final"], 1e-17) for r in sel], "s:",
                         color=cols[arm], label=f"{arm} + finisher", lw=1.2,
                         alpha=0.8, ms=4)
            sel = sorted([r for r in recs if r["init"] == init and r["fn"] == fn
                          and r["arm"] == "adam"], key=lambda r: r["N"])
            if sel:
                a.loglog([r["N"] for r in sel], [r["floor0"] for r in sel],
                         "k--", lw=1.2, label="floor of the initial geometry")
            a.set_ylim(1e-17, 5)
            a.set_xlabel("width $N$")
            a.set_ylabel(r"eval rel $L_2$")
            a.set_title(fn)
            a.grid(alpha=0.25, which="both")
        h, lab = ax[0][0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 1.035),
                   ncol=4, frameon=False, fontsize=9)
        fig.suptitle(f"{TITLE[init]} -- $L$ discovered by the certificate",
                     y=1.115, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        p = path.with_name(path.stem + f"_{init}.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        print(f"  -> {p}")


def fig_undersolve(recs, path):
    """T3's other half: the shallower the solve, the better the geometry it
    leaves. x is the per-step solver budget tau (direct plotted at the right
    as the exact-solve limit); y is the error a terminal exact solve would
    give on the geometry each arm finished with."""
    taus = [1, 3, 10, 30, 100]
    fig, ax = _grid(1, 2, size=(6.4, 4.2))
    for j, init in enumerate(("qi", "rand")):
        a = ax[0][j]
        for fn, c in (("sine", "tab:blue"), ("runge", "tab:red")):
            sel = [r for r in recs if r["init"] == init and r["fn"] == fn]
            if not sel:
                continue
            xs, ys = [], []
            for t in taus:
                r = next((x for x in sel if x["arm"] == f"lsqr{t}"), None)
                if r:
                    xs.append(t)
                    ys.append(max(r["floor_final"], 1e-17))
            a.loglog(xs, ys, "o-", color=c, lw=1.8,
                     label=fn + r": lsqr $\tau$")
            d = next((x for x in sel if x["arm"] == "direct"), None)
            if d:
                a.loglog([300], [max(d["floor_final"], 1e-17)], "*", color=c,
                         ms=16, label=f"{fn}: exact solve")
            a.axhline(sel[0]["floor0"], ls="--", color=c, lw=1.0, alpha=0.6)
        a.set_xlabel("solver iterations per step "
                     r"$\tau$   (star = exact solve)")
        a.set_ylabel(r"eval rel $L_2$ after a terminal exact solve")
        a.set_title(f"{TITLE[init]} -- what the geometry left behind is worth")
        a.set_ylim(1e-17, 5)
        a.grid(alpha=0.25, which="both")
        a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2,
                 frameon=False, fontsize=9)
    fig.text(0.5, -0.02, "dashed lines: the floor of the initial geometry, "
             "per target -- a curve above its own dashed line means the "
             "geometry got worse", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  -> {path}")


def main():
    C.FIGS.mkdir(parents=True, exist_ok=True)
    t2 = C.load(C.RESULTS / "t2_assembly.jsonl")
    if t2:
        fig_ablation(t2, C.FIGS / "T2_ablation.png")
        fig_signals(t2, C.FIGS / "T2_signals.png")
    t2b = C.load(C.RESULTS / "t2b_alpha.jsonl")
    if t2b:
        fig_t2b(t2b, C.FIGS / "T2b_tension.png")
    t3 = C.load(C.RESULTS / "t3_solver.jsonl")
    if t3:
        fig_solver(t3, C.FIGS / "T3_solver.png")
        fig_undersolve(t3, C.FIGS / "T3b_undersolving.png")
    t4 = C.load(C.RESULTS / "t4_grid.jsonl")
    if t4:
        fig_grid(t4, C.FIGS / "T4_grid.png")


if __name__ == "__main__":
    main()
