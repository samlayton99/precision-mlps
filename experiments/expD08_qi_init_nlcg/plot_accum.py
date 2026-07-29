"""Figures for the batching stage-3 (accumulation) results.

Outputs to results/.../batching/accum/:
  accum_summary_2x2.png   per-target (N=256): best eval rel L2 vs batch
                          size for every method family measured, full-batch
                          reference points and floors included
  accum_trajectories.png  best-so-far eval rel L2 vs batch-gradient evals
                          for two representative cells: adam vs unguarded
                          slim vs accum_fixed (reruns the accum cells to
                          capture trajectories; ~1 min)

Run:  uv run python experiments/expD08_qi_init_nlcg/plot_accum.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import batching  # noqa: E402

ACCUM_DIR = batching.BATCH_ROOT / "accum"
FNS = batching.FNS
LEVELS = ["B", "B/4", "B/16", "B/64"]


def _load(name):
    return json.loads((batching.BATCH_ROOT / name).read_text())


def _grid(name, opt=None):
    p = (batching.out_dir(opt) / "batching_rows.json" if opt
         else batching.BATCH_ROOT / name)
    return json.loads(p.read_text())


def render_summary(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slim = _grid(None, "slim")
    adam = _grid(None, "adam")
    abl = _load("ablations_rows.json")
    svrg = _load("svrg_rows.json")
    accum = _load("accum_rows.json")

    series = [
        ("adam", adam, None, "black", "--", "x"),
        ("slim (unguarded)", slim, None, "tab:gray", "-", "o"),
        ("floor+trust+gate", abl, ["floor", "gate", "trust"],
         "tab:orange", "-", "s"),
        ("svrg+floor+gate", svrg, ["floor", "gate", "svrg"],
         "tab:purple", "-", "^"),
        ("accum (adaptive k)", accum, ["accum"], "tab:cyan", "-", "d"),
        ("accum_fixed (exact epoch)", accum, ["accum_fixed"],
         "tab:blue", "-", "o"),
    ]
    N = 256
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    for ax, fn in zip(axes.flat, FNS):
        for label, rows, guards, color, ls, mk in series:
            pts = {}
            for r in rows:
                if r["fn"] != fn or r["N"] != N:
                    continue
                if guards is not None and r.get("guards") != guards:
                    continue
                if guards is None and r.get("guards"):
                    continue
                pts[r["blabel"]] = (r["bsize"], r["best_rel"])
            bs = [pts[lb][0] for lb in LEVELS if lb in pts]
            ys = [pts[lb][1] for lb in LEVELS if lb in pts]
            if bs:
                ax.loglog(bs, ys, ls, marker=mk, ms=5, lw=1.4,
                          color=color, label=label)
        fl = [r["floor"] for r in slim if r["fn"] == fn and r["N"] == N]
        if fl:
            ax.axhline(fl[0], color="k", ls=":", lw=1.2,
                       label="lstsq floor (full data)")
        ax.set_ylim(1e-16, 2.0)
        ax.set_title(f"{fn}   (N={N})", fontsize=11)
        ax.set_xlabel("batch size")
        ax.set_ylabel("best eval rel $L_2$")
        ax.grid(True, which="both", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False,
               fontsize=9)
    fig.suptitle("expD08 batching -- every method family, N=256, 3000 "
                 "batch-gradient evals, epoch reshuffle, n=10000",
                 y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  saved {path}")


def render_trajectories(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [("runge", 256, "B/4", 4), ("sine", 256, "B/16", 16)]
    slim = _grid(None, "slim")
    adam = _grid(None, "adam")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, (fn, N, lb, div) in zip(axes, cells):
        for label, rows, color in (("adam", adam, "black"),
                                   ("slim (unguarded)", slim, "tab:gray")):
            for r in rows:
                if (r["fn"] == fn and r["N"] == N and r["blabel"] == lb
                        and not r.get("guards")):
                    y = np.minimum.accumulate(np.array(r["rels"]))
                    x = np.arange(1, len(y) + 1)     # 1 batch eval / iter
                    ax.loglog(x, y, color=color, lw=1.3, label=label)
                    break
        # rerun accum_fixed to capture the trajectory
        r = batching.batch_case({"fn": fn, "N": N, "blabel": lb,
                                 "div": div, "iters": batching.ITERS,
                                 "threads": 6, "opt": "slim",
                                 "guards": ["accum_fixed"]})
        k = -(-batching.N_TRAIN_B // r["bsize"])
        y = np.minimum.accumulate(np.array(r["rels"]))
        x = 1.5 * k * np.arange(1, len(y) + 1)       # grad+curv sweeps
        ax.loglog(x, y, color="tab:blue", lw=1.6,
                  label="accum_fixed (exact epoch)")
        fl = [rr["floor"] for rr in slim
              if rr["fn"] == fn and rr["N"] == N][0]
        ax.axhline(fl, color="k", ls=":", lw=1.2, label="lstsq floor")
        ax.set_ylim(1e-16, 2.0)
        ax.set_title(f"{fn}  N={N}  batch {lb}", fontsize=11)
        ax.set_xlabel("batch-gradient evaluations")
        ax.set_ylabel("best-so-far eval rel $L_2$")
        ax.grid(True, which="both", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    fig.suptitle("sustained descent under batching: exact-epoch "
                 "accumulation vs per-batch stepping", y=1.10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  saved {path}")


if __name__ == "__main__":
    ACCUM_DIR.mkdir(parents=True, exist_ok=True)
    src = batching.BATCH_ROOT / "accum_rows.json"
    if src.exists():
        (ACCUM_DIR / "accum_rows.json").write_text(src.read_text())
    render_summary(ACCUM_DIR / "accum_summary_2x2.png")
    render_trajectories(ACCUM_DIR / "accum_trajectories.png")
    print(f"done; outputs under {ACCUM_DIR}")
