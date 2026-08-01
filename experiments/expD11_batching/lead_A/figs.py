"""Figures for lead_A.  Legends always OUTSIDE the axes, above them."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core11, wlsqr

FIGS = HERE.parents[2] / "results" / "checkpoint_D_optimizers" / "expD11_batching" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
CS = [32, 64, 128, 256]
COL = {"qi1d": "#1f77b4", "qi2d": "#2ca02c", "twinU": "#d62728"}
MK = {206: "o", 270: "s", 462: "^", 922: "D", 145: "o", 320: "s", 571: "^",
      1021: "D", 256: "o", 512: "s", 1024: "^", 2048: "D"}


def load(f):
    p = HERE / f
    return [json.loads(l) for l in open(p)] if p.exists() else []


# ---------------------------------------------------------------- fig 1 ----
def fig_frontier():
    R = load("frontier.jsonl")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for fam in ("qi1d", "qi2d", "twinU"):
        rows = [r for r in R if r["fam"] == fam and r["c"] < 1e5]
        for d in sorted(set(r["d"] for r in rows)):
            g = sorted([r for r in rows if r["d"] == d], key=lambda r: r["c"])
            xr = [r["c"] / r["rank"] for r in g]
            yv = [r["best"] for r in g]
            ax[0].plot(xr, yv, MK.get(d, "o") + "-", color=COL[fam], ms=5, lw=1.2,
                       alpha=.85, label=f"{fam} d={d}")
            ax[1].plot([r["c"] for r in g], yv, MK.get(d, "o") + "-",
                       color=COL[fam], ms=5, lw=1.2, alpha=.85)
        for r in [q for q in R if q["fam"] == fam and q["c"] > 1e5]:
            ax[0].plot(r["best_it"] / r["rank"], r["best"], "*", color=COL[fam],
                       ms=12, mew=0)
            ax[1].plot(r["best_it"], r["best"], "*", color=COL[fam], ms=12, mew=0)
    for a, xl in zip(ax, ["window / numerical rank,  c / r",
                          "window size c  (state = c*d floats)"]):
        a.set_yscale("log"); a.set_xlabel(xl); a.set_ylim(1e-16, 1e-3)
        a.axhspan(1e-16, 1e-14, color="0.85", zorder=0)
        a.grid(alpha=.3, which="both")
    ax[0].set_xscale("log"); ax[0].set_ylabel("best eval rel $L_2$")
    ax[0].axvline(0.8, color="k", ls=":", lw=1)
    ax[0].set_title("collapses onto $c/r$  (knee at $c/r\\approx0.8$)", fontsize=10)
    ax[1].set_xscale("log")
    ax[1].set_title("same data vs absolute $c$: no collapse", fontsize=10)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=6,
               fontsize=7.5, frameon=False)
    fig.text(0.5, 0.885, "stars = full reorthogonalization, plotted at the state it "
             "actually used ($c=$ iterations); grey band = truncated-SVD floor region",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.865))
    fig.savefig(FIGS / "leadA_F1_window_frontier.png", dpi=150)
    print("F1 done")


# ---------------------------------------------------------------- fig 2 ----
def fig_traj():
    cases = [("qi1d N=128", core11.prob_qi1d(128)),
             ("twin-unstruct d=512", core11.prob_twin(512, "unstruct")),
             ("qi2d radon d=571", core11.prob_qi2d(576))]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for a, (nm, p) in zip(ax, cases):
        it = int(2.5 * p["rank"] + 60)
        for c, col, lab in [(0, "#999999", "no reorth"),
                            (32, "#9ecae1", "c=32"), (64, "#4292c6", "c=64"),
                            (128, "#2171b5", "c=128"), (256, "#08306b", "c=256"),
                            (10**6, "#d62728", "full reorth")]:
            _, h = wlsqr.lsqr_win(p["A"], p["y"], it, c=c, mode="recent",
                                  Aev=p["A_ev"], yev=p["y_ev"],
                                  ynorm=p["ye_norm"], probe_every=5)
            xs = [q["it"] for q in h]; ys = [q["ev"] for q in h]
            a.plot(xs, ys, color=col, lw=1.6 if c in (0, 10**6) else 1.1, label=lab)
        a.axhline(p["floor_pool"], color="k", ls="--", lw=1, label="SVD floor")
        a.axvline(p["rank"], color="k", ls=":", lw=1, label="iteration = r")
        a.set_yscale("log"); a.set_xlabel("LSQR iteration")
        a.set_title(f"{nm}  (d={p['d']}, r={p['rank']})", fontsize=10)
        a.grid(alpha=.3, which="both"); a.set_ylim(1e-17, 3)
    ax[0].set_ylabel("eval rel $L_2$")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=8,
               fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIGS / "leadA_F2_trajectories.png", dpi=150)
    print("F2 done")


# ---------------------------------------------------------------- fig 3 ----
def fig_bd():
    R = load("batch_drift.jsonl")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for cs, col in [("qi1d128", "#1f77b4"), ("twinU512", "#d62728")]:
        dr = sorted([r for r in R if r["case"] == cs and r["test"] == "drift"],
                    key=lambda r: r["eta"])
        e = np.array([max(r["eta"], 1e-16) for r in dr])
        ax[0].plot(e, [r["best"] for r in dr], "o-", color=col, label=cs)
        bt = sorted([r for r in R if r["case"] == cs and r["test"] == "batch"
                     and not r["grow"]], key=lambda r: r["b"])
        ax[1].plot([r["b"] for r in bt], [r["best"] for r in bt], "o-",
                   color=col, label=f"{cs}: stochastic LSQR")
        ax[1].plot([r["b"] for r in bt], [r["floor1"] for r in bt], "s--",
                   color=col, alpha=.6, label=f"{cs}: 1-batch SVD floor")
        ax[0].axhline(dr[0]["floor"], color=col, ls=":", lw=1)
    ax[0].plot([1e-16, 1e-3], [3e-16, 3e-3], "k--", lw=1, label="slope 1 ($3\\eta$)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("per-matvec relative operator perturbation $\\eta$")
    ax[0].set_ylabel("best eval rel $L_2$")
    ax[0].set_title("req 7 drift: tracks $\\eta$ linearly, no preconditioner to go stale",
                    fontsize=9, pad=34)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("rows per matvec $b$")
    ax[1].set_title("req 5 batching: a fresh batch per matvec destroys the recurrence",
                    fontsize=9, pad=34)
    for a in ax:
        a.grid(alpha=.3, which="both")
        h, l = a.get_legend_handles_labels()
        a.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2,
                 fontsize=7.5, frameon=False, borderaxespad=0)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIGS / "leadA_F3_batch_drift.png", dpi=150)
    print("F3 done")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which: fig_frontier()
    if "2" in which: fig_traj()
    if "3" in which: fig_bd()
