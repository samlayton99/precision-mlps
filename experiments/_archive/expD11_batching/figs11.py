"""expD11 figures. Rendered from the JSONL after every completed cell, so the
run is watchable live. Every figure answers exactly one question; the question
is in the title and the reading instructions are in the writeup.
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

RES = c.RESULTS
FIG = c.FIGS
SWEEP = RES / "sweep.jsonl"
EXTRA = RES / "extra.jsonl"

CA, CB = "C0", "C2"
CF1, CFP = "C3", "k"
FAM_ORDER = ["qi1d", "qi2d", "twin_cluster", "twin_unstruct"]
FAM_NICE = {"qi1d": "QI 1-D", "qi2d": "QI 2-D (ckpt E)",
            "twin_cluster": "random twin, clustered",
            "twin_unstruct": "random twin, unstructured"}


def _legend(ax, ncol=3, fs=8):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=ncol,
              borderaxespad=0, fontsize=fs, frameon=False)


def _rows(tag=None, path=None):
    rs = c.load(path or SWEEP)
    return [r for r in rs if tag is None or r.get("tag") == tag]


def _cells(rs):
    """Ordered unique (family, d, label) present in the data."""
    seen = {}
    for r in rs:
        seen.setdefault((r["family"], r["d"], r["label"]), None)
    return sorted(seen, key=lambda t: (FAM_ORDER.index(t[0])
                                       if t[0] in FAM_ORDER else 9, t[1]))


def _grid(nc, ncols=4, size=(3.5, 3.0)):
    nrow = int(np.ceil(nc / ncols))
    fig, axs = plt.subplots(nrow, min(ncols, max(nc, 1)),
                            figsize=(size[0] * min(ncols, max(nc, 1)),
                                     size[1] * nrow + 0.7), squeeze=False)
    return fig, axs.ravel()


# ------------------------------------------------------------------ F1 ----
def f1_softwin():
    """Soft win: does error hold at the pool floor as the batch shrinks, and
    does it always beat what a single batch of that size can do?"""
    rs = _rows("sweep")
    cells = _cells(rs)
    if not cells:
        return
    fig, axs = _grid(len(cells))
    for ax, key in zip(axs, cells):
        fam, d, lab = key
        sub = [r for r in rs if (r["family"], r["d"]) == (fam, d)]
        fp = sub[0]["floor_pool"]
        for meth, col, mk in (("A", CA, "o"), ("B", CB, "s")):
            pts = sorted([(r["bmult"], r["eval"]) for r in sub
                          if r["method"] == meth and r["eval"] is not None])
            if pts:
                ax.loglog([p[0] for p in pts], [max(p[1], 1e-17) for p in pts],
                          mk + "-", color=col, ms=4, lw=1.3,
                          label=f"Type {meth}")
        f1 = sorted([(r["bmult"], r["floor_1batch"]) for r in sub
                     if r.get("floor_1batch") is not None])
        if f1:
            ax.loglog([p[0] for p in f1], [max(p[1], 1e-17) for p in f1],
                      "--", color=CF1, lw=1.2, label="one-batch floor")
        ax.axhline(fp, color=CFP, ls=":", lw=1.2, label="pool floor")
        ax.set_title(f"{lab}\nd={d}", fontsize=8)
        ax.set_xlabel("batch / d"); ax.set_ylabel("eval rel $L_2$")
        ax.set_ylim(1e-17, 1e2); ax.grid(alpha=.3, which="both")
    for ax in axs[len(cells):]:
        ax.axis("off")
    _legend(axs[0], ncol=4, fs=7)
    fig.suptitle("F1 soft win -- accuracy vs batch size (constant total solver "
                 "iterations). Win: Type B pinned to the pool floor at every b; "
                 "both types always below the one-batch floor.", y=0.995, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "F1_softwin_batch_vs_error.png", dpi=135)
    plt.close(fig)


# ------------------------------------------------------------------ F2 ----
def f2_strongwin():
    """Strong win: do tiny-batch runs match the full-pool run?"""
    rs = _rows("sweep")
    cells = _cells(rs)
    if not cells:
        return
    bars = [("A", 8.0, "A\nfull pool"), ("A", 0.125, "A\nb=d/8"),
            ("A", 1 / 64, "A\nb=d/64"), ("B", 0.125, "B\nb=d/8"),
            ("B", 1 / 64, "B\nb=d/64")]
    fig, axs = _grid(len(cells))
    for ax, key in zip(axs, cells):
        fam, d, lab = key
        sub = [r for r in rs if (r["family"], r["d"]) == (fam, d)]
        fp = sub[0]["floor_pool"]
        vals, labs, cols = [], [], []
        for meth, bm, nm in bars:
            hit = [r["eval"] for r in sub if r["method"] == meth
                   and abs(np.log2(max(r["bmult"], 1e-9) / bm)) < 0.01]
            vals.append(hit[0] if hit and hit[0] is not None else np.nan)
            labs.append(nm); cols.append(CA if meth == "A" else CB)
        xs = np.arange(len(vals))
        ax.bar(xs, np.maximum(np.nan_to_num(vals, nan=1e-17), 1e-17), color=cols)
        for x, v in zip(xs, vals):
            if v is not None and np.isfinite(v):
                ax.text(x, max(v, 1e-17) * 1.6, f"{v:.0e}", ha="center", fontsize=6)
        ax.axhline(fp, color=CFP, ls=":", lw=1.2, label="pool floor")
        ax.set_yscale("log"); ax.set_ylim(1e-17, 1e2)
        ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=6)
        ax.set_ylabel("eval rel $L_2$")
        ax.set_title(f"{lab}\nd={d}", fontsize=8); ax.grid(alpha=.3, axis="y")
    for ax in axs[len(cells):]:
        ax.axis("off")
    _legend(axs[0], ncol=1, fs=7)
    fig.suptitle("F2 strong win -- tiny batches vs the full pool. Win: the "
                 "Type B bars sit on the pool floor regardless of b, i.e. batch "
                 "size costs nothing once rows are accumulated.", y=0.995, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "F2_strongwin_bars.png", dpi=135)
    plt.close(fig)


# ------------------------------------------------------------------ F3 ----
def f3_accumulation():
    """Does the accumulation buffer, not the batch, set Type B's accuracy?"""
    rs = _rows("accum", EXTRA)
    if not rs:
        return
    cells = _cells(rs)
    fig, axs = _grid(len(cells), ncols=3, size=(4.0, 3.2))
    for ax, key in zip(axs, cells):
        fam, d, lab = key
        sub = [r for r in rs if (r["family"], r["d"]) == (fam, d)]
        fp = sub[0]["floor_pool"]
        for i, bm in enumerate(sorted({r["bmult"] for r in sub}, reverse=True)):
            pts = sorted([(r["acc_mult"], r["eval"]) for r in sub
                          if r["bmult"] == bm])
            ax.loglog([p[0] for p in pts], [max(p[1], 1e-17) for p in pts],
                      "o-", ms=4, lw=1.2, color=f"C{i}",
                      label=f"b=d/{int(round(1/bm))}" if bm < 1 else f"b={bm:g}d")
        ax.axhline(fp, color=CFP, ls=":", lw=1.2, label="pool floor")
        ax.axvline(4, color="C3", ls="--", lw=1.0, label="$n_{acc}=4d$ spec")
        ax.set_xlabel("$n_{acc}/d$ (rows accumulated)")
        ax.set_ylabel("eval rel $L_2$"); ax.set_ylim(1e-17, 1e2)
        ax.set_title(f"{lab}\nd={d}", fontsize=8); ax.grid(alpha=.3, which="both")
    for ax in axs[len(cells):]:
        ax.axis("off")
    _legend(axs[0], ncol=3, fs=7)
    fig.suptitle("F3 accumulation -- Type B accuracy is set by the BUFFER, not "
                 "the batch. Win: curves for different b collapse onto one line "
                 "and reach the floor at $n_{acc}\\approx4d$.", y=0.995, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "F3_accumulation.png", dpi=135)
    plt.close(fig)


# ------------------------------------------------------------------ F4 ----
def f4_cost():
    """What does each accuracy actually cost, in batch-passes?"""
    rs = _rows("sweep")
    cells = _cells(rs)
    if not cells:
        return
    fig, axs = _grid(len(cells))
    for ax, key in zip(axs, cells):
        fam, d, lab = key
        sub = [r for r in rs if (r["family"], r["d"]) == (fam, d)]
        fp = sub[0]["floor_pool"]
        for meth, col, mk in (("A", CA, "o"), ("B", CB, "s")):
            pts = [(r["passes"], r["eval"], r["bmult"]) for r in sub
                   if r["method"] == meth and r.get("passes") and r["eval"]]
            if pts:
                pts.sort()
                ax.loglog([p[0] for p in pts], [max(p[1], 1e-17) for p in pts],
                          mk, color=col, ms=5, label=f"Type {meth}", alpha=.85)
        ax.axhline(fp, color=CFP, ls=":", lw=1.2, label="pool floor")
        ax.set_xlabel("passes-equivalent (batch-sized)")
        ax.set_ylabel("eval rel $L_2$"); ax.set_ylim(1e-17, 1e2)
        ax.set_title(f"{lab}\nd={d}", fontsize=8); ax.grid(alpha=.3, which="both")
    for ax in axs[len(cells):]:
        ax.axis("off")
    _legend(axs[0], ncol=3, fs=7)
    fig.suptitle("F4 cost -- accuracy against compute actually spent (1 pass = "
                 "one O(b*d) traversal). Each point is one batch size; read the "
                 "lower-left frontier.", y=0.995, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "F4_cost_pareto.png", dpi=135)
    plt.close(fig)


# ------------------------------------------------------------------ F5 ----
def f5_noise():
    """Under noise, do both types land on the statistical floor they should?"""
    rs = _rows("noise", EXTRA)
    if not rs:
        return
    cells = sorted({(r["family"], r["d"], r["label"]) for r in rs},
                   key=lambda t: (FAM_ORDER.index(t[0]) if t[0] in FAM_ORDER else 9, t[1]))
    fig, axs = _grid(len(cells), ncols=4, size=(3.6, 3.1))
    for ax, key in zip(axs, cells):
        fam, d, lab = key
        sub = [r for r in rs if (r["family"], r["d"], r["label"]) == key]
        sig = np.array(sorted({r["noise_rel"] for r in sub}))
        for meth, col, mk in (("A", CA, "o"), ("B", CB, "s")):
            for i, bm in enumerate(sorted({r["bmult"] for r in sub}, reverse=True)):
                pts = sorted([(r["noise_rel"], r["eval"]) for r in sub
                              if r["method"] == meth and r["bmult"] == bm])
                if pts:
                    ax.loglog([p[0] for p in pts], [max(p[1], 1e-17) for p in pts],
                              mk, color=col, ms=4, alpha=0.45 + 0.18 * i,
                              label=f"{meth} b/d={bm:g}" if i < 2 else None)
        r0 = sub[0]
        ax.loglog(sig, 0.27 * sig, "k--", lw=1.2,
                  label=r"$0.27\sigma$ (pool floor)")
        ax.set_xlabel(r"$\sigma_{rel}$"); ax.set_ylabel("eval rel $L_2$")
        ax.set_title(f"{lab}\nd={d}", fontsize=8); ax.grid(alpha=.3, which="both")
    for ax in axs[len(cells):]:
        ax.axis("off")
    _legend(axs[0], ncol=3, fs=6)
    fig.suptitle("F5 noise -- achieved error vs the statistical floor. Win: "
                 "points track the dashed line across decades and never sit "
                 "below it (that would be overfitting the noise).", y=0.995, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "F5_noise_floors.png", dpi=135)
    plt.close(fig)


# ------------------------------------------------------------------ F6 ----
def f6_scaling():
    """Does any of this change as d grows to 5000?"""
    rs = _rows("sweep")
    if not rs:
        return
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for fam in FAM_ORDER:
        sub = [r for r in rs if r["family"] == fam]
        if not sub:
            continue
        ds = sorted({r["d"] for r in sub})
        for ax, (meth, bm) in zip(axs[:2], (("B", 1 / 64), ("A", 1 / 8))):
            ys, xs = [], []
            for d in ds:
                hit = [r["eval"] for r in sub if r["d"] == d and r["method"] == meth
                       and abs(np.log2(max(r["bmult"], 1e-9) / bm)) < 0.01
                       and r["eval"] is not None]
                if hit:
                    xs.append(d); ys.append(hit[0])
            if xs:
                ax.loglog(xs, np.maximum(ys, 1e-17), "o-", ms=4, label=FAM_NICE[fam])
            fl = [(r["d"], r["floor_pool"]) for r in sub if r["d"] in ds]
            fl = sorted({p[0]: p[1] for p in fl}.items())
            ax.loglog([p[0] for p in fl], [max(p[1], 1e-17) for p in fl],
                      ":", lw=1, alpha=.6)
        xs, ys = [], []
        for d in ds:
            hit = [r["passes"] for r in sub if r["d"] == d and r["method"] == "B"
                   and abs(np.log2(max(r["bmult"], 1e-9) / (1 / 8))) < 0.01
                   and r.get("passes")]
            if hit:
                xs.append(d); ys.append(hit[0])
        if xs:
            axs[2].loglog(xs, ys, "o-", ms=4, label=FAM_NICE[fam])
    for ax, t in zip(axs, ("Type B at b=d/64 (dotted = pool floors)",
                           "Type A at b=d/8 (dotted = pool floors)",
                           "Type B cost at b=d/8")):
        ax.set_xlabel("d"); ax.grid(alpha=.3, which="both"); ax.set_title(t, fontsize=9)
    axs[0].set_ylabel("eval rel $L_2$"); axs[0].set_ylim(1e-17, 1e2)
    axs[1].set_ylabel("eval rel $L_2$"); axs[1].set_ylim(1e-17, 1e2)
    axs[2].set_ylabel("passes-equivalent")
    _legend(axs[0], ncol=4, fs=7)
    fig.suptitle("F6 scaling to d=5000 -- does the batching story survive width? "
                 "Win: left panel flat on the floors; right panel's cost growth "
                 "is the honest price.", y=0.99, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "F6_d_scaling.png", dpi=135)
    plt.close(fig)


# ------------------------------------------------------------------ F7 ----
def f7_tau():
    """Why Type A must be an episodic mini-solve, not a per-step interleave."""
    rs = _rows("tau", EXTRA)
    if not rs:
        return
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.4))
    fx = [r for r in rs if r["kind"] == "fixed_op"]
    if fx:
        pts = sorted([(r["tau"], r["eval"]) for r in fx])
        axs[0].loglog([p[0] for p in pts], [max(p[1], 1e-17) for p in pts],
                      "o-", color=CA, ms=5, label="restarted every $\\tau$")
        un = [r["eval"] for r in fx if r.get("unrestarted")]
        base = [r for r in rs if r["kind"] == "unrestarted"]
        if base:
            axs[0].axhline(base[0]["eval"], color=CFP, ls=":", lw=1.3,
                           label="unrestarted (same total iters)")
    axs[0].set_xlabel(r"$\tau$ = LSMR iterations per restart")
    axs[0].set_ylabel("eval rel $L_2$")
    axs[0].set_title("restart penalty, FIXED operator, constant total iters", fontsize=9)
    axs[0].grid(alpha=.3, which="both"); _legend(axs[0], ncol=2, fs=7)

    cl = [r for r in rs if r["kind"] == "clamp"]
    if cl:
        for key, col, lab in (("plain", CF1, r"unclamped $\tau$"),
                              ("clamped", CA, r"clamped $\tau\leq b/2$"),
                              ("floor_1batch", CFP, "one-batch floor")):
            pts = sorted([(r["bmult"], r[key]) for r in cl if r.get(key)])
            if pts:
                axs[1].loglog([p[0] for p in pts], [max(p[1], 1e-17) for p in pts],
                              "o-" if key != "floor_1batch" else "--",
                              color=col, ms=4, lw=1.3, label=lab)
    axs[1].set_xlabel("batch / d"); axs[1].set_ylabel("eval rel $L_2$")
    axs[1].set_title(r"the $\tau$ clamp: catastrophe $\to$ graceful", fontsize=9)
    axs[1].grid(alpha=.3, which="both"); _legend(axs[1], ncol=3, fs=7)

    tr = [r for r in rs if r["kind"] == "trunc"]
    if tr:
        pts = sorted([(r["rcond"], r["eval"]) for r in tr])
        axs[2].loglog([p[0] for p in pts], [max(p[1], 1e-17) for p in pts],
                      "o-", color=CA, ms=5)
        axs[2].axvline(1e-12, color="C3", ls="--", lw=1.1, label="shipped $10^{-12}$")
        _legend(axs[2], ncol=1, fs=7)
    axs[2].set_xlabel("block-eigenvalue truncation rcond")
    axs[2].set_ylabel("eval rel $L_2$")
    axs[2].set_title("block-inverse truncation is load-bearing", fontsize=9)
    axs[2].grid(alpha=.3, which="both")
    fig.suptitle("F7 the three Type A design decisions, each isolated by "
                 "measurement: Krylov runway, the $\\tau$ clamp, and block "
                 "truncation.", y=0.99, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "F7_typeA_design.png", dpi=135)
    plt.close(fig)


# ------------------------------------------------------------------ F8 ----
def f8_drift_guard():
    """Type A under a moving operator; Type B's revert guard."""
    rs = _rows("drift", EXTRA)
    gd = _rows("guard", EXTRA)
    if not rs and not gd:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3))
    if rs:
        for i, key in enumerate(sorted({(r["family"], r["d"]) for r in rs})):
            sub = [r for r in rs if (r["family"], r["d"]) == key]
            pts = sorted([(max(r["eta"], 1e-10), r["eval"]) for r in sub])
            axs[0].loglog([p[0] for p in pts], [max(p[1], 1e-17) for p in pts],
                          "o-", ms=4, color=f"C{i}", label=f"{key[0]} d={key[1]}")
        et = np.array([1e-8, 1e-3])
        axs[0].loglog(et, 0.5 * et, "k--", lw=1.1, label=r"$\propto\eta$ (coupling law)")
        _legend(axs[0], ncol=2, fs=7)
    axs[0].set_xlabel(r"per-episode operator drift $\eta$ ($\eta=10^{-10}$ means 0)")
    axs[0].set_ylabel("Type A best eval rel $L_2$")
    axs[0].set_title("Type A tracks a moving operator", fontsize=9)
    axs[0].grid(alpha=.3, which="both")
    if gd:
        pts = sorted([(max(r["eta"], 1e-10), r["eval_out"], r["eval_in"],
                       r["reverted"]) for r in gd])
        axs[1].loglog([p[0] for p in pts], [max(p[1], 1e-17) for p in pts],
                      "s-", color=CB, ms=5, label="Type B output")
        axs[1].loglog([p[0] for p in pts], [max(p[2], 1e-17) for p in pts],
                      "o--", color="C7", ms=4, label="entry point $w_0$")
        for p in pts:
            if p[3]:
                axs[1].annotate("rev", (p[0], max(p[1], 1e-17)), fontsize=6,
                                textcoords="offset points", xytext=(0, 7),
                                ha="center", color="C3")
        _legend(axs[1], ncol=2, fs=7)
    axs[1].set_xlabel(r"drift present at freeze $\eta$")
    axs[1].set_ylabel("eval rel $L_2$")
    axs[1].set_title("Type B guard: output never worse than entry", fontsize=9)
    axs[1].grid(alpha=.3, which="both")
    fig.suptitle("F8 robustness -- a moving operator (left) and a premature "
                 "finale (right).", y=0.99, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG / "F8_drift_guard.png", dpi=135)
    plt.close(fig)


def render_all():
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (f1_softwin, f2_strongwin, f3_accumulation, f4_cost,
               f5_noise, f6_scaling, f7_tau, f8_drift_guard):
        try:
            fn()
        except Exception as ex:            # never let a figure kill the run
            print(f"  [fig {fn.__name__} skipped: {type(ex).__name__}: {ex}]",
                  flush=True)


if __name__ == "__main__":
    render_all()
    print("rendered ->", FIG)
