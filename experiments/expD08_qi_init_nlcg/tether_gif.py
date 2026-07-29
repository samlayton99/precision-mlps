"""Tether-magnitude movie: how the auto-tether gates evolve over training.

4 rows = the 4 gif targets, 3 cols = (w | b | v) blocks; each dot is one
neuron's TETHER MAGNITUDE S_i (log scale) plotted against its current
center -b_k/w_k -- not the coefficient values. Requires S-enriched snaps
(run `staircase.py --snap-s` first).

Run:  uv run python experiments/expD08_qi_init_nlcg/tether_gif.py
Output: results/.../iteration_6/gifs/tether_magnitude_N256.gif
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402

N = 256
FNS = run.TARGETS3
OUT = run.OUT_DIR / "gifs" / "tether_magnitude_N256.gif"


def main(fps=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image

    data = {}
    for fn in FNS:
        z = np.load(run.OUT_DIR / "snaps" / f"{fn}--N{N}.npz")
        if "S" not in z:
            print(f"snaps for {fn} lack S -- run staircase.py --snap-s")
            return
        data[fn] = z
    steps = [int(s) for s in data[FNS[0]]["steps"]]
    total = steps[-1]
    W = data[FNS[0]]["w"].shape[1]
    XW = 2.0
    smax = max(float(z["S"].max()) for z in data.values()) * 10

    rbcmap = LinearSegmentedColormap.from_list("redblue", ["red", "blue"])
    ncolors = {}
    for fn in FNS:
        w0, b0 = data[fn]["w"][0], data[fn]["b"][0]
        c0 = -b0 / np.where(np.abs(w0) > 1e-12, w0, 1e-12)
        ranks = np.argsort(np.argsort(c0))
        ncolors[fn] = rbcmap(ranks / max(len(c0) - 1, 1))

    images = []
    for si, s in enumerate(steps):
        fig, axes = plt.subplots(len(FNS), 3, figsize=(12.5, 10),
                                 squeeze=False)
        for i, fn in enumerate(FNS):
            z = data[fn]
            w_, b_ = z["w"][si], z["b"][si]
            S = z["S"][si]
            c = -b_ / np.where(np.abs(w_) > 1e-12, w_, 1e-12)
            vis = np.abs(c) <= XW
            blocks = [("tether on w", S[:W]), ("tether on b", S[W:2 * W]),
                      ("tether on v", S[2 * W:3 * W])]
            for j, (ttl, Sb) in enumerate(blocks):
                ax = axes[i, j]
                ax.axvspan(-1, 1, color="#1f4e8c", alpha=0.05)
                ax.axhline(1.0, color="#2ca02c", lw=1.0, ls="--",
                           alpha=0.8)
                ax.scatter(c[vis], np.maximum(Sb[vis], 0.8), s=9,
                           c=ncolors[fn][vis], edgecolors="none", zorder=3)
                ax.set_yscale("log")
                ax.set_xlim(-XW, XW)
                ax.set_ylim(0.5, smax)
                ax.tick_params(labelsize=6)
                if i == 0:
                    ax.set_title(ttl + "  (green = untethered, S=1)",
                                 fontsize=8)
                if j == 0:
                    ax.set_ylabel(fn, fontsize=8)
                ax.text(0.02, 0.96, f"max S={Sb.max():.1e}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=6, color="#333",
                        bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))
        fig.suptitle(f"auto-tether magnitudes $S_i$   N={N}   step "
                     f"{s}/{total}   (x = current centers $-b_k/w_k$, "
                     "y = how strongly each coordinate is slowed)",
                     fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(OUT, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"saved {OUT} ({len(images)} frames, "
          f"{OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
