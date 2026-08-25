"""expD13 figures. Re-rendered after every cell; never allowed to crash a run."""
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
import core13 as c

ORD = ["qi1d_256", "qi1d_1024", "qi2d_256", "qi2d_2048",
       "twin_512", "twin_2048", "twin_spec2d"]
CMAP = plt.get_cmap("viridis")


def _rows(nmult=1.0, noise=0.0):
    return [q for q in c.load(c.RESULTS / "ladder.jsonl")
            if abs(q["nmult"] - nmult) < 1e-12
            and abs(q.get("noise", 0.0) - noise) < 1e-15]


def _keys(rs):
    return [x for x in ORD if x in {q["key"] for q in rs}]


def _grid(nc, ncols=4, w=4.0, h=3.6):
    nr = int(np.ceil(nc / ncols))
    fig, ax = plt.subplots(nr, min(ncols, max(nc, 1)),
                           figsize=(w * min(ncols, max(nc, 1)), h * nr + 1.8),
                           squeeze=False)
    return fig, ax.ravel()


def _top(fig, title, sub, y1=0.985, y2=0.945):
    fig.text(0.5, y1, title, ha="center", va="top", fontsize=12, weight="bold")
    fig.text(0.5, y2, sub, ha="center", va="top", fontsize=8.8, linespacing=1.5)


# --------------------------------------------------- D1 per-Phi mu curves --
def d1_per_phi():
    rs = [q for q in _rows() if q["arm"] == "ladder"]
    ks = _keys(rs)
    if not ks:
        return
    fig, axs = _grid(len(ks))
    al = [q["alpha"] for q in rs if q["alpha"] > 0]
    norm = LogNorm(vmin=min(al), vmax=max(al))
    for ax, key in zip(axs, ks):
        sub = sorted([q for q in rs if q["key"] == key], key=lambda q: -q["alpha"])
        for q in sub:
            t = np.array(q["traj"])
            if t.size == 0:
                continue
            ax.loglog(t[:, 0], np.maximum(t[:, 1], 1e-17),
                      color=CMAP(norm(q["alpha"])), lw=1.6)
            ax.plot([t[-1, 0]], [max(q["damped_opt"], 1e-17)], "o",
                    color=CMAP(norm(q["alpha"])), ms=3.5)
        ax.axhline(sub[0]["floor_true"], color="crimson", ls=":", lw=1.5)
        ax.set_title(f"{sub[0]['label']}\n$d$={sub[0]['d']}, $r$={sub[0]['rank']}",
                     fontsize=9)
        ax.set_xlabel("LSQR iterations within the level")
        ax.set_ylabel("eval rel $L_2$ (current geometry)")
        ax.set_ylim(1e-16, 3e0)
        ax.grid(alpha=.3, which="both")
    for ax in axs[len(ks):]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), ax=axs.tolist(),
                      fraction=0.015, pad=0.012)
    cb.set_label(r"$\alpha=\sqrt{\mu}/\sigma_1$  (= damping = geometry noise $\eta$)",
                 fontsize=8.5)
    _top(fig, "D1   Convergence per damping level, with the geometry DRIFTING",
         "Same as expD12's F3 but $\\Phi$ moves: at level $\\alpha$ the geometry is perturbed by $\\eta=\\alpha$ (fresh draw), so early levels solve a badly wrong $\\Phi$.\n"
         "Dark = strong damping + far geometry, bright = weak damping + nearly-true geometry. Red dotted = the floor of the TRUE geometry.")
    fig.savefig(c.FIGS / "D1_per_phi_mu_curves.png", dpi=135)
    plt.close(fig)


# --------------------------------------------------------- D2 scheduling --
def d2_scatter():
    rs = [q for q in _rows() if q["arm"] == "ladder" and q["alpha"] > 0]
    if not rs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(13.4, 5.4))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.70, wspace=0.26)
    for i, key in enumerate(_keys(rs)):
        sub = sorted([q for q in rs if q["key"] == key], key=lambda q: q["alpha"])
        axs[0].loglog([q["alpha"] for q in sub],
                      [max(q["reached"], 1e-17) for q in sub], "o-",
                      color=f"C{i}", ms=5, lw=1.4, label=sub[0]["label"])
        axs[1].loglog([q["alpha"] for q in sub], [q["iters"] for q in sub],
                      "o-", color=f"C{i}", ms=5, lw=1.4, label=sub[0]["label"])
    a = np.array([1e-12, 1e-1])
    axs[0].loglog(a, a, "k--", lw=1.5, label=r"accuracy $=\alpha=\eta$")
    axs[1].loglog(a, 3 * a ** -0.5, "k--", lw=1.5, label=r"$\propto\kappa_\mu^{1/2}$")
    axs[1].loglog(a, 0.02 * a ** -1.0, ":", color="grey", lw=1.5,
                  label=r"$\propto\kappa_\mu$")
    axs[0].set_ylabel("accuracy the level converges to")
    axs[1].set_ylabel("LSQR iterations to converge that level")
    for ax in axs:
        ax.set_xlabel(r"$\alpha$  (damping, and geometry noise $\eta$)")
        ax.grid(alpha=.3, which="both")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
                  fontsize=7, frameon=False)
    _top(fig, "D2   The two scheduling laws, under drift",
         "Do the expD12 laws survive a moving geometry? Left: accuracy vs damping. Right: iterations vs damping.")
    fig.savefig(c.FIGS / "D2_scheduling_laws.png", dpi=135)
    plt.close(fig)


# ------------------------------------------------------ D3 concatenated ----
def d3_concat():
    rs = _rows()
    ks = _keys(rs)
    if not ks:
        return
    fig, ax = plt.subplots(figsize=(12.6, 6.3))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.11, top=0.74)
    for i, key in enumerate(ks):
        lad = sorted([q for q in rs if q["key"] == key and q["arm"] == "ladder"],
                     key=lambda q: -q["alpha"])
        term = [q for q in rs if q["key"] == key and q["arm"] == "terminal"]
        xs, ys = [], []
        for q in lad:
            for it, e in q["traj"]:
                xs.append(q["cum_before"] + it); ys.append(e)
        hand = (xs[-1], ys[-1]) if xs else None
        for q in term:
            for it, e in q["traj"]:
                xs.append(q["cum_before"] + it); ys.append(e)
        ax.loglog(xs, np.maximum(ys, 1e-17), "-", color=f"C{i}", lw=1.8,
                  label=(lad or term)[0]["label"])
        if hand:
            ax.plot([hand[0]], [max(hand[1], 1e-17)], "*", color=f"C{i}",
                    ms=15, zorder=5)
        if term:
            ax.axhline(term[0]["floor_true"], color=f"C{i}", ls=":", lw=0.9,
                       alpha=.55)
    ax.set_xlabel("cumulative LSQR iterations (ladder, then the terminal solve)")
    ax.set_ylabel("eval rel $L_2$")
    ax.set_ylim(1e-16, 3e0)
    ax.grid(alpha=.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4,
              fontsize=8, frameon=False)
    _top(fig, "D3   The whole drifting run as one line, per $\\Phi$",
         "The geometry converges as the damping falls; stars mark the handoff, after which the terminal solve runs on the TRUE geometry.\n"
         "Dotted horizontals are each matrix's true-geometry floor.", y2=0.94)
    fig.savefig(c.FIGS / "D3_full_trajectory.png", dpi=135)
    plt.close(fig)


# --------------------------------------------------------- D4 matching -----
def d4_matching():
    all_rs = c.load(c.RESULTS / "ladder.jsonl")
    nms = sorted({q["nmult"] for q in all_rs if q.get("noise", 0.0) == 0.0})
    if len(nms) < 2:
        return
    ks = [x for x in ORD if x in {q["key"] for q in all_rs}]
    ks = [x for x in ks if len({q["nmult"] for q in all_rs if q["key"] == x}) > 1]
    if not ks:
        return
    fig, axs = plt.subplots(1, len(ks) + 1, figsize=(4.4 * (len(ks) + 1), 5.4))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.13, top=0.68, wspace=0.30)
    lab = {0.0: "static $\\Phi$ (control)", 0.1: "$\\eta=0.1\\alpha$",
           1.0: "$\\eta=\\alpha$ (matched)", 10.0: "$\\eta=10\\alpha$"}
    for ax, key in zip(axs[:-1], ks):
        for i, nm in enumerate(nms):
            sub = sorted([q for q in all_rs if q["key"] == key and q["nmult"] == nm
                          and q["arm"] == "ladder" and q.get("noise", 0.) == 0.],
                         key=lambda q: -q["alpha"])
            if not sub:
                continue
            ax.loglog([q["alpha"] for q in sub],
                      [max(q["err_true"], 1e-17) for q in sub], "o-",
                      color=f"C{i}", ms=5, lw=1.8, label=lab.get(nm, str(nm)))
        t = [q for q in all_rs if q["key"] == key and q["arm"] == "terminal"]
        if t:
            ax.axhline(t[0]["floor_true"], color="crimson", ls=":", lw=1.6)
        ax.set_xlabel(r"$\alpha$ (damping)")
        ax.set_ylabel("error on the TRUE geometry")
        ax.invert_xaxis()
        ax.set_ylim(1e-16, 3e0)
        ax.grid(alpha=.3, which="both")
        ax.set_title([q for q in all_rs if q["key"] == key][0]["label"], fontsize=9.5)
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=2,
                  fontsize=8, frameon=False)
    ax = axs[-1]
    for i, nm in enumerate(nms):
        xs, ys = [], []
        for key in ks:
            t = [q for q in all_rs if q["key"] == key and q["arm"] == "terminal"
                 and q["nmult"] == nm and q.get("noise", 0.) == 0.]
            if t:
                xs.append(key.replace("_", "\n")); ys.append(t[0]["reached"] / t[0]["floor_true"])
        if xs:
            ax.semilogy(range(len(xs)), ys, "o-", color=f"C{i}", ms=7, lw=1.8,
                        label=lab.get(nm, str(nm)))
    ax.axhline(1.0, color="crimson", ls=":", lw=1.6)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([k.replace("_", "\n") for k in ks], fontsize=8)
    ax.set_ylabel("final error / true floor")
    ax.grid(alpha=.3, axis="y", which="both")
    ax.set_title("does the terminal solve still reach the floor?", fontsize=9.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=2,
              fontsize=8, frameon=False)
    _top(fig, "D4   Should $\\mu$ track the geometry noise?",
         "Sweeping the matching between damping $\\alpha$ and geometry perturbation $\\eta=n\\cdot\\alpha$. If $\\mu$ is a variance dial, the matched line ($\\eta=\\alpha$) should be the efficient one:\n"
         "under-damping ($\\eta=10\\alpha$) wastes iterations solving a geometry that is about to move; over-damping ($\\eta=0.1\\alpha$) leaves accuracy on the table.", y2=0.935)
    fig.savefig(c.FIGS / "D4_noise_matching.png", dpi=135)
    plt.close(fig)


# ---------------------------------------------------------- D5 batching ----
def d5_batch():
    rs = c.load(c.RESULTS / "batch.jsonl")
    if not rs:
        return
    ks = [x for x in ORD if x in {q["key"] for q in rs}]
    fig, axs = plt.subplots(1, max(len(ks), 1), figsize=(6.4 * max(len(ks), 1), 5.4),
                            squeeze=False)
    axs = axs.ravel()
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.68, wspace=0.24)
    for ax, key in zip(axs, ks):
        sub = sorted([q for q in rs if q["key"] == key], key=lambda q: -q["bmult"])
        x = [q["bmult"] for q in sub]
        ax.loglog(x, [max(q["eval"], 1e-17) for q in sub], "o-", color="C2",
                  ms=6, lw=2, label="drifting batched $\\mu$-ladder")
        ax.loglog(x, [max(q["floor_1batch"], 1e-17) for q in sub], "s--",
                  color="k", ms=5, lw=1.5, label="one-batch floor (soft-win bar)")
        ax.axhline(sub[0]["floor_pool"], color="crimson", ls=":", lw=1.8,
                   label="true-geometry floor (hard-win bar)")
        ax.set_title(f"{sub[0]['label']}   $d$={sub[0]['d']}", fontsize=9.5)
        ax.set_xlabel("batch fraction $b/d$")
        ax.set_ylabel("best eval rel $L_2$")
        ax.invert_xaxis()
        ax.set_ylim(1e-16, 3e0)
        ax.grid(alpha=.3, which="both")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1,
                  fontsize=7.5, frameon=False)
    _top(fig, "D5   Batching under a drifting geometry",
         "Fresh random batch every GN step AND a geometry that moves with the damping. SOFT win = below the dashed one-batch floor; HARD win = reaching the red line.")
    fig.savefig(c.FIGS / "D5_batching.png", dpi=135)
    plt.close(fig)


# ------------------------------------------------------------- D6 noise ----
def d6_noise():
    all_rs = c.load(c.RESULTS / "ladder.jsonl")
    nz = sorted({q.get("noise", 0.0) for q in all_rs if q.get("noise", 0.0) > 0})
    if not nz:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
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
    _top(fig, "D6   Noisy $y$ AND a drifting geometry",
         "Two independent error sources at once. The solve should still land on the statistical floor, not below it and not orders above.")
    fig.savefig(c.FIGS / "D6_noise.png", dpi=135)
    plt.close(fig)


def render_all():
    c.FIGS.mkdir(parents=True, exist_ok=True)
    for fn in (d1_per_phi, d2_scatter, d3_concat, d4_matching, d5_batch, d6_noise):
        try:
            fn()
        except Exception as ex:
            print(f"  [fig {fn.__name__}: {type(ex).__name__}: {ex}]", flush=True)


if __name__ == "__main__":
    render_all()
    print("rendered ->", c.FIGS)
