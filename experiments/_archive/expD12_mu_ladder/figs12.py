"""expD12 figures. Re-rendered after every problem; never allowed to crash a run."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core12 as k

ORD = ["qi1d_256", "qi1d_1024", "qi2d_256", "qi2d_2048",
       "twin_512", "twin_2048", "twin_spec2d"]
CMAP = plt.get_cmap("viridis")


def _rows(noise=0.0):
    return [q for q in k.load(k.RESULTS / "ladder.jsonl")
            if abs(q.get("noise", 0.0) - noise) < 1e-15]


def _keys(rs):
    return [x for x in ORD if x in {q["key"] for q in rs}]


def _grid(nc, ncols=4, w=4.0, h=3.5, top=0.80):
    nr = int(np.ceil(nc / ncols))
    fig, ax = plt.subplots(nr, min(ncols, max(nc, 1)),
                           figsize=(w * min(ncols, max(nc, 1)), h * nr + 1.6),
                           squeeze=False)
    return fig, ax.ravel()


def _top(fig, title, sub, y1=0.985, y2=0.945):
    fig.text(0.5, y1, title, ha="center", va="top", fontsize=12, weight="bold")
    fig.text(0.5, y2, sub, ha="center", va="top", fontsize=8.8, linespacing=1.5)


# ------------------------------------------------------- F3 per-Phi mu ----
def f3_per_phi():
    rs = [q for q in _rows() if q["arm"] == "ladder"]
    ks = _keys(rs)
    if not ks:
        return
    fig, axs = _grid(len(ks))
    al = [q["alpha"] for q in rs if q["alpha"] > 0]
    norm = LogNorm(vmin=min(al), vmax=max(al))
    for ax, key in zip(axs, ks):
        sub = sorted([q for q in rs if q["key"] == key],
                     key=lambda q: -q["alpha"])
        for q in sub:
            t = np.array(q["traj"])
            if t.size == 0:
                continue
            ax.loglog(t[:, 0], np.maximum(t[:, 1], 1e-17),
                      color=CMAP(norm(q["alpha"])), lw=1.6)
            ax.plot([t[-1, 0]], [max(q["damped_opt"], 1e-17)], "o",
                    color=CMAP(norm(q["alpha"])), ms=3.5)
        ax.axhline(sub[0]["floor_pool"], color="crimson", ls=":", lw=1.5)
        ax.set_title(f"{sub[0]['label']}\n$d$={sub[0]['d']}, $r$={sub[0]['rank']}",
                     fontsize=9)
        ax.set_xlabel("LSQR iterations within the level")
        ax.set_ylabel("eval rel $L_2$")
        ax.set_ylim(1e-16, 3e0)
        ax.grid(alpha=.3, which="both")
    for ax in axs[len(ks):]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), ax=axs.tolist(),
                      fraction=0.015, pad=0.012)
    cb.set_label(r"$\alpha=\sqrt{\mu}/\sigma_1$   (damping level)", fontsize=8.5)
    _top(fig, "F3   One convergence curve per damping level, per $\\Phi$",
         "Each line is one $\\alpha$ level solved at $c=0$ (no stored state), warm-started from the level above; the dot marks its exact damped optimum.\n"
         "Dark = strong damping (gradient-like), bright = weak damping. Red dotted = the fp64 pool floor. Read: each level flattens onto its own plateau.")
    fig.savefig(k.FIGS / "F3_per_phi_mu_curves.png", dpi=135)
    plt.close(fig)


# --------------------------------------------------------- F4 scatter ----
def f4_scatter():
    rs = [q for q in _rows() if q["arm"] == "ladder" and q["alpha"] > 0]
    if not rs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(13.0, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.72, wspace=0.26)
    ks = _keys(rs)
    for i, key in enumerate(ks):
        sub = sorted([q for q in rs if q["key"] == key], key=lambda q: q["alpha"])
        axs[0].loglog([q["alpha"] for q in sub],
                      [max(q["reached"], 1e-17) for q in sub], "o-",
                      color=f"C{i}", ms=5, lw=1.4, label=sub[0]["label"])
        axs[1].loglog([q["alpha"] for q in sub], [q["iters"] for q in sub],
                      "o-", color=f"C{i}", ms=5, lw=1.4, label=sub[0]["label"])
    a = np.array([1e-10, 1e-1])
    axs[0].loglog(a, a, "k--", lw=1.5, label=r"accuracy $=\alpha$")
    axs[1].loglog(a, 3 * a ** -0.5, "k--", lw=1.5, label=r"$\propto\kappa_\mu^{1/2}$")
    axs[1].loglog(a, 0.02 * a ** -1.0, ":", color="grey", lw=1.5,
                  label=r"$\propto\kappa_\mu$")
    axs[0].set_ylabel("accuracy the level converges to")
    axs[1].set_ylabel("LSQR iterations to converge that level")
    for ax in axs:
        ax.set_xlabel(r"$\alpha=\sqrt{\mu}/\sigma_1$")
        ax.grid(alpha=.3, which="both")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
                  fontsize=7, frameon=False)
    _top(fig, "F4   The two scheduling laws",
         "Left: what accuracy each damping level buys. Right: what it costs. Together these set the $\\mu$ schedule and the iteration budget.",
         y1=0.985, y2=0.945)
    fig.savefig(k.FIGS / "F4_mu_vs_convergence.png", dpi=135)
    plt.close(fig)


# ------------------------------------------------------ F5 concatenated ---
def f5_concat():
    rs = _rows()
    ks = _keys(rs)
    if not ks:
        return
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.11, top=0.74)
    for i, key in enumerate(ks):
        lad = sorted([q for q in rs if q["key"] == key and q["arm"] == "ladder"],
                     key=lambda q: -q["alpha"])
        term = [q for q in rs if q["key"] == key and q["arm"] == "terminal"]
        xs, ys = [], []
        for q in lad:
            for it, e, _ in q["traj"]:
                xs.append(q["cum_before"] + it)
                ys.append(e)
        hand = (xs[-1], ys[-1]) if xs else None
        for q in term:
            for it, e, _ in q["traj"]:
                xs.append(q["cum_before"] + it)
                ys.append(e)
        ax.loglog(xs, np.maximum(ys, 1e-17), "-", color=f"C{i}", lw=1.8,
                  label=(lad or term)[0]["label"])
        if hand:
            ax.plot([hand[0]], [max(hand[1], 1e-17)], "*", color=f"C{i}",
                    ms=15, zorder=5)
        if term:
            ax.axhline(term[0]["floor_pool"], color=f"C{i}", ls=":", lw=0.9,
                       alpha=.55)
    ax.set_xlabel("cumulative LSQR iterations (whole ladder, then the terminal solve)")
    ax.set_ylabel("eval rel $L_2$")
    ax.set_ylim(1e-16, 3e0)
    ax.grid(alpha=.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4,
              fontsize=8, frameon=False)
    _top(fig, "F5   The whole run as one line, per $\\Phi$",
         "Every iteration of every damping level concatenated, then the terminal full-reorthogonalization solve. Stars mark the handoff, where $c=0$ gives out and $O(rd)$ state is allocated.\n"
         "Dotted horizontals are each matrix's fp64 pool floor. Read: a long cheap descent, a knee at the handoff, then a short drop to the floor.",
         y1=0.985, y2=0.94)
    fig.savefig(k.FIGS / "F5_full_trajectory.png", dpi=135)
    plt.close(fig)


# ---------------------------------------------------------- F1 batching --
def f1_batch():
    rs = k.load(k.RESULTS / "batch.jsonl")
    if not rs:
        return
    ks = [x for x in ORD if x in {q["key"] for q in rs}]
    fig, axs = plt.subplots(1, max(len(ks), 1), figsize=(6.2 * max(len(ks), 1), 5.2),
                            squeeze=False)
    axs = axs.ravel()
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.70, wspace=0.24)
    for ax, key in zip(axs, ks):
        sub = sorted([q for q in rs if q["key"] == key], key=lambda q: -q["bmult"])
        x = [q["bmult"] for q in sub]
        ax.loglog(x, [max(q["eval"], 1e-17) for q in sub], "o-", color="C2",
                  ms=6, lw=2, label="batched $\\mu$-ladder")
        ax.loglog(x, [max(q["floor_1batch"], 1e-17) for q in sub], "s--",
                  color="k", ms=5, lw=1.5, label="one-batch floor (soft-win bar)")
        ax.axhline(sub[0]["floor_pool"], color="crimson", ls=":", lw=1.8,
                   label="pool floor (hard-win bar)")
        ax.set_title(f"{sub[0]['label']}   $d$={sub[0]['d']}", fontsize=9.5)
        ax.set_xlabel("batch fraction $b/d$")
        ax.set_ylabel("best eval rel $L_2$")
        ax.invert_xaxis()
        ax.set_ylim(1e-16, 3e0)
        ax.grid(alpha=.3, which="both")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1,
                  fontsize=7.5, frameon=False)
    _top(fig, "F1   Batching: soft win and hard win",
         "Fresh random batch every GN step, down to $b=d/64$. SOFT win = the green line sits below the dashed one-batch floor (the solver accumulates across batches).\n"
         "HARD win = green reaches the red pool floor.")
    fig.savefig(k.FIGS / "F1_batching.png", dpi=135)
    plt.close(fig)


# ------------------------------------------------------------- F2 noise --
def f2_noise():
    all_rs = k.load(k.RESULTS / "ladder.jsonl")
    nz = sorted({q.get("noise", 0.0) for q in all_rs if q.get("noise", 0.0) > 0})
    if not nz:
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.70)
    ks = [x for x in ORD if x in {q["key"] for q in all_rs if q.get("noise", 0) > 0}]
    for i, key in enumerate(ks):
        pts = []
        for s in nz:
            sub = [q for q in all_rs if q["key"] == key
                   and abs(q.get("noise", 0.0) - s) < 1e-15]
            if sub:
                pts.append((s, min(q["reached"] for q in sub)))
        if pts:
            ax.loglog([p[0] for p in pts], [p[1] for p in pts], "o-",
                      color=f"C{i}", ms=6, lw=1.8,
                      label=[q for q in all_rs if q["key"] == key][0]["label"])
    s = np.array([1e-9, 1e-1])
    ax.loglog(s, 0.27 * s, "k--", lw=1.6, label=r"$0.27\sigma$ statistical floor")
    ax.set_xlabel(r"relative noise on $y$,  $\sigma_{rel}$")
    ax.set_ylabel("best eval rel $L_2$ reached")
    ax.grid(alpha=.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              fontsize=8, frameon=False)
    _top(fig, "F2   Noisy $y$: does the ladder stop at the statistical floor?",
         "The solve should track the dashed line -- not sit orders above it (under-solving) and not dive below it (fitting the noise).")
    fig.savefig(k.FIGS / "F2_noise.png", dpi=135)
    plt.close(fig)


def render_all():
    k.FIGS.mkdir(parents=True, exist_ok=True)
    for fn in (f3_per_phi, f4_scatter, f5_concat, f1_batch, f2_noise):
        try:
            fn()
        except Exception as ex:
            print(f"  [fig {fn.__name__}: {type(ex).__name__}: {ex}]", flush=True)


if __name__ == "__main__":
    render_all()
    print("rendered ->", k.FIGS)
