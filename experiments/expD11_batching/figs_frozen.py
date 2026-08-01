"""Figures for the frozen-operator episode design. One question per figure.
Re-rendered after every cell; never allowed to crash the run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core11 as c

OUT = c.RESULTS / "frozen.jsonl"
FIG = c.FIGS
FAM_ORD = ["qi1d", "qi2d", "twin_cluster", "twin_unstruct"]


def _rows():
    return c.load(OUT)


def _cells(rs):
    s = {(r["family"], r["d"], r["label"], r["noise"]) for r in rs}
    return sorted(s, key=lambda t: (FAM_ORD.index(t[0]) if t[0] in FAM_ORD else 9,
                                    t[3], t[1]))


def _grid(nc, ncols=4, w=3.6, h=3.2):
    nr = int(np.ceil(nc / ncols))
    fig, ax = plt.subplots(nr, min(ncols, max(nc, 1)),
                           figsize=(w * min(ncols, max(nc, 1)), h * nr + 1.0),
                           squeeze=False)
    return fig, ax.ravel()


def _lg(ax, ncol=3, fs=7):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=ncol,
              borderaxespad=0, fontsize=fs, frameon=False)


# ------------------------------------------------------------------- H1 ---
def h1_softwin():
    """Does the episode hold its accuracy as the batch shrinks to d/64?"""
    rs = [r for r in _rows() if r["noise"] == 0.0]
    cells = [k for k in _cells(rs) if k[3] == 0.0]
    if not cells:
        return
    fig, axs = _grid(len(cells))
    for ax, key in zip(axs, cells):
        fam, d, lab, _ = key
        sub = [r for r in rs if (r["family"], r["d"]) == (fam, d)]
        fp = sub[0]["floor_pool"]
        ep = sorted([(r["bmult"], r["eval"]) for r in sub if r["arm"] == "episode"])
        if ep:
            ax.loglog([p[0] for p in ep], [max(p[1], 1e-17) for p in ep],
                      "o-", color="C2", ms=5, lw=2, label="episode (frozen)")
        st = sorted([(r["bmult"], r["eval"]) for r in sub if r["arm"] == "stream"])
        if st:
            ax.loglog([p[0] for p in st], [max(p[1], 1e-17) for p in st],
                      "^--", color="C1", ms=5, lw=1.6, label=r"streaming $\tau$=1")
        f1 = sorted([(r["bmult"], r["floor_1batch"]) for r in sub
                     if r["arm"] == "episode" and r.get("floor_1batch")])
        if f1:
            ax.loglog([p[0] for p in f1], [max(p[1], 1e-17) for p in f1],
                      "--", color="k", lw=1.4, label="one-batch floor")
        ax.axhline(fp, color="C0", ls=":", lw=1.6, label="pool floor")
        ax.set_title(f"{lab}\nd={d}", fontsize=8)
        ax.set_xlabel("batch / $d$"); ax.set_ylabel("eval rel $L_2$")
        ax.set_ylim(1e-17, 1e2); ax.grid(alpha=.3, which="both")
    for ax in axs[len(cells):]:
        ax.axis("off")
    _lg(axs[0], 4)
    fig.suptitle("H1  Batch size vs accuracy for the frozen-operator episode.  "
                 "Batch only controls how many stream steps the accumulation takes,\n"
                 "so the episode line should be FLAT.  Both solid lines must stay "
                 "below the dashed one-batch reference.", fontsize=9.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "H1_episode_softwin.png", dpi=135)
    plt.close(fig)


# ------------------------------------------------------------------- H2 ---
def h2_accum():
    """Is accuracy set by the accumulated rows rather than the batch?"""
    rs = [r for r in _rows() if r["arm"] == "accum" and r["noise"] == 0.0]
    if not rs:
        return
    cells = sorted({(r["family"], r["d"], r["label"]) for r in rs},
                   key=lambda t: (FAM_ORD.index(t[0]) if t[0] in FAM_ORD else 9, t[1]))
    fig, axs = _grid(len(cells), ncols=4)
    for ax, key in zip(axs, cells):
        sub = [r for r in rs if (r["family"], r["d"], r["label"]) == key]
        pts = sorted([(r["n_acc_mult"], r["eval"]) for r in sub])
        ax.loglog([p[0] for p in pts], [max(p[1], 1e-17) for p in pts],
                  "o-", color="C2", ms=6, lw=2, label="episode, b=d/8")
        ax.axhline(sub[0]["floor_pool"], color="C0", ls=":", lw=1.6, label="pool floor")
        ax.axvline(4, color="C3", ls="--", lw=1.2, label="$n_{acc}=4d$")
        ax.set_xlabel("$n_{acc}/d$"); ax.set_ylabel("eval rel $L_2$")
        ax.set_ylim(1e-17, 1e2); ax.grid(alpha=.3, which="both")
        ax.set_title(f"{key[2]}\nd={key[1]}", fontsize=8)
    for ax in axs[len(cells):]:
        ax.axis("off")
    _lg(axs[0], 3)
    fig.suptitle("H2  Accuracy vs accumulated rows.  This is the knob that "
                 "actually matters; the user's batch size is absorbed by the buffer.",
                 fontsize=9.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG / "H2_accumulation.png", dpi=135)
    plt.close(fig)


# ------------------------------------------------------------------- H3 ---
def h3_rounds():
    """Do refinement rounds keep helping, and where does semiconvergence bite?"""
    rs = [r for r in _rows() if r["arm"] == "episode" and r["noise"] == 0.0
          and r.get("hist") and abs(r["bmult"] - 0.125) < 1e-9]
    if not rs:
        return
    cells = sorted({(r["family"], r["d"], r["label"]) for r in rs},
                   key=lambda t: (FAM_ORD.index(t[0]) if t[0] in FAM_ORD else 9, t[1]))
    fig, axs = _grid(len(cells), ncols=4)
    for ax, key in zip(axs, cells):
        sub = [r for r in rs if (r["family"], r["d"], r["label"]) == key][0]
        h = sub["hist"]
        ax.semilogy([q["round"] for q in h], [max(q["eval"], 1e-17) for q in h],
                    "o-", color="C2", ms=5, lw=2, label="eval (oracle view)")
        ob = np.array([q["obs"] for q in h])
        ax.semilogy([q["round"] for q in h], ob / max(ob[0], 1e-300) * h[0]["eval"],
                    "s--", color="C1", ms=4, lw=1.4,
                    label=r"$\|A^\top r\|$ (observable, scaled)")
        ax.axhline(sub["floor_pool"], color="C0", ls=":", lw=1.6, label="pool floor")
        ax.set_xlabel("refinement round"); ax.set_ylabel("eval rel $L_2$")
        ax.set_ylim(1e-17, 1e2); ax.grid(alpha=.3)
        ax.set_title(f"{key[2]}\nd={key[1]}", fontsize=8)
    for ax in axs[len(cells):]:
        ax.axis("off")
    _lg(axs[0], 3)
    fig.suptitle("H3  Refinement rounds on the frozen operator.  A single long run "
                 "semiconverges; the rounds are what buy the last orders.\n"
                 "The observable must track eval, or there is no usable stopping rule.",
                 fontsize=9.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG / "H3_refinement_rounds.png", dpi=135)
    plt.close(fig)


# ------------------------------------------------------------------- H4 ---
def h4_noise():
    """Under noise, does it stop at the statistical floor?"""
    rs = [r for r in _rows() if r["arm"] == "episode" and r["noise"] > 0]
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    fams = sorted({r["family"] for r in rs})
    for i, fam in enumerate(fams):
        sub = [r for r in rs if r["family"] == fam]
        for mk, bm in (("o", 1.0), ("s", 0.125), ("^", 1 / 64)):
            pts = sorted([(r["noise"], r["eval"]) for r in sub
                          if abs(r["bmult"] - bm) < 1e-9])
            if pts:
                ax.loglog([p[0] for p in pts], [max(p[1], 1e-17) for p in pts],
                          mk, color=f"C{i}", ms=6, alpha=.85,
                          label=f"{fam} b/d={bm:g}")
    s = np.array([1e-6, 1e-2])
    ax.loglog(s, 0.27 * s, "k--", lw=1.5, label=r"$0.27\sigma$ statistical floor")
    ax.set_xlabel(r"noise $\sigma_{rel}$"); ax.set_ylabel("eval rel $L_2$")
    ax.grid(alpha=.3, which="both")
    _lg(ax, 4, 7)
    fig.suptitle("H4  Under noise the episode should sit ON the statistical floor,\n"
                 "not below it (overfitting) and not orders above it.",
                 fontsize=9.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(FIG / "H4_noise.png", dpi=135)
    plt.close(fig)


# ------------------------------------------------------------------- H5 ---
def h5_scaling():
    """Does the episode's accuracy and cost hold as d grows?"""
    rs = [r for r in _rows() if r["arm"] == "episode" and r["noise"] == 0.0
          and abs(r["bmult"] - 0.125) < 1e-9]
    if not rs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for i, fam in enumerate(FAM_ORD):
        sub = sorted([r for r in rs if r["family"] == fam], key=lambda r: r["d"])
        if not sub:
            continue
        axs[0].loglog([r["d"] for r in sub], [max(r["eval"], 1e-17) for r in sub],
                      "o-", color=f"C{i}", ms=5, label=fam)
        axs[0].loglog([r["d"] for r in sub], [max(r["floor_pool"], 1e-17) for r in sub],
                      ":", color=f"C{i}", lw=1, alpha=.7)
        axs[1].loglog([r["d"] for r in sub], [r["passes"] for r in sub],
                      "o-", color=f"C{i}", ms=5, label=fam)
    axs[0].set_ylabel("eval rel $L_2$ at b=d/8"); axs[0].set_ylim(1e-17, 1e2)
    axs[0].set_title("accuracy (dotted = each cell's pool floor)", fontsize=9)
    axs[1].set_ylabel("passes-equivalent per episode")
    axs[1].set_title("cost", fontsize=9)
    for ax in axs:
        ax.set_xlabel("d"); ax.grid(alpha=.3, which="both")
    _lg(axs[0], 4)
    fig.suptitle("H5  Width scaling of the episode design.  Accuracy should track "
                 "the floors; cost growth is the honest price.", fontsize=9.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG / "H5_scaling.png", dpi=135)
    plt.close(fig)


def render_all():
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (h1_softwin, h2_accum, h3_rounds, h4_noise, h5_scaling):
        try:
            fn()
        except Exception as ex:
            print(f"  [fig {fn.__name__}: {type(ex).__name__}: {ex}]", flush=True)


if __name__ == "__main__":
    render_all()
    print("rendered ->", FIG)
