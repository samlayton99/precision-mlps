"""Tether-magnitude diagnostic plot -- standard per-iteration figure.

One panel per gif target (N=256): tether magnitudes S_i over training
(max and median per block, log scale, step plot at frame times) with the
eval rel-L2 trajectory overlaid on a twin axis. Vertical dotted lines =
trips. The question the figure answers: does the lock outrun the fit --
how big is S, when did it get big, and had the error already dropped?

Requires S-enriched snaps in the CURRENT iteration's snaps/ dir
(run `staircase.py --snap-s` after any grid). Output:
  results/.../iteration_<K>/tether_magnitude_N256.png

Run:  uv run python experiments/expD08_qi_init_nlcg/tether_plot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402

N = 256
FNS = run.TARGETS3
OUT = run.OUT_DIR / f"tether_magnitude_N{N}.png"
BLOCKS = [("w", "#d62728"), ("b", "#ff7f0e"), ("v", "#1f77b4")]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for k, fn in enumerate(FNS):
        ax = axes.flat[k]
        z = np.load(run.OUT_DIR / "snaps" / f"{fn}--N{N}.npz")
        if "S" not in z:
            print(f"snaps for {fn} lack S -- run staircase.py --snap-s")
            return
        steps = z["steps"].astype(int)
        S = z["S"]                       # (frames, m), m = 3W+1
        W = z["w"].shape[1]
        stats = json.loads(str(z["stats"]))
        rels = z["rels"]

        for j, (name, col) in enumerate(BLOCKS):
            Sb = S[:, j * W:(j + 1) * W]
            ax.step(steps, Sb.max(axis=1), where="post", color=col,
                    lw=1.6, label=f"max S ({name})")
            ax.step(steps, np.median(Sb, axis=1), where="post", color=col,
                    lw=1.0, ls="--", alpha=0.7, label=f"median S ({name})")
        ax.axhline(1.0, color="#2ca02c", lw=1.0, alpha=0.8)
        for t in stats.get("trips", []):
            ax.axvline(t, color="#888", ls=":", lw=0.9)
        ax.set_yscale("log")
        ax.set_ylim(0.5, max(float(S.max()) * 10, 1e2))
        ax.set_title(f"{fn}   T_final={stats.get('T_final', 1):.0e}   "
                     f"trips at {stats.get('trips', [])}", fontsize=9)
        ax.grid(alpha=0.25, which="both")
        if k % 2 == 0:
            ax.set_ylabel("tether magnitude $S_i$")
        if k >= 2:
            ax.set_xlabel("iteration")

        ax2 = ax.twinx()
        ax2.plot(np.arange(1, len(rels) + 1), rels, color="0.45", lw=1.0)
        ax2.axhline(float(z["floor"]), color="k", ls=":", lw=0.9)
        ax2.set_yscale("log")
        ax2.set_ylim(1e-16, 1e0)
        if k % 2 == 1:
            ax2.set_ylabel("eval rel L2 (gray; dotted = floor)",
                           color="0.35")
        ax2.tick_params(axis="y", labelcolor="0.35")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=6, fontsize=8)
    fig.suptitle(f"auto-tether magnitude vs learning   iteration_"
                 f"{run.ITERATION}   N={N}   (green line: S=1 = "
                 "untethered; gray dotted verticals: trips)", y=0.925,
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT, dpi=140)
    print("saved", OUT)


if __name__ == "__main__":
    main()
