"""expC06 [histogram] -- weight-magnitude histogram under Xavier -> 1s interpolation.

Part 1 of expC06 (soft-neuron interpolation). The VISUALIZATION half: it animates
what happens to the inner-weight magnitude distribution as the weights interpolate
Xavier -> uniform. Part 2 (run_threshold.py) is the causal TEST -- protecting the
soft tail and watching the error. Both share expC05 weights' normalization/draw.

A companion view to expC05 `weights` mode. There, a genuine Xavier inner-weight
vector is interpolated toward the ideal QI weight (the constant "all-gamma"
vector) along an L1 octant face, holding the bandwidth budget mean|w| fixed, and
the *error* is tracked. Here we instead watch the *distribution of weight
magnitudes* deform over the interpolation -- what actually happens to w as it
moves onto the ideal.

Normalization (identical to expC05 weights): L1 / mean-abs. With the bandwidth
budget mean|w| factored out to 1 (gamma is a pure scale that shifts the axis but
not the shape, and dividing it out makes the 4 widths directly comparable),

    base_i  = |w_xav_i| / mean|w_xav|          # ~ uniform on [0, 2], mean 1
    |w_i(s)| = (1 - s) * base_i + s * 1         # linear, per element, mean held at 1

so s : 0 -> 1 carries the Xavier magnitude spread onto the 1s vector. Because
expC05's path never leaves its octant, |w_i(s)| is exactly this per-element
linear interpolation between |w_xav_i| and 1 (the ideal). s is swept LINEARLY
over 100 points (matching expC05's linspace s).

Output: a 2x2 animation (one histogram per width N in {64,128,256,512}; x-axis =
|w_i|, the absolute element value; y-axis = count) saved as MP4, plus a static
snapshot contact sheet PNG.

Usage:
    python experiments/expC06_soft_neuron_interp/run_histogram.py            # build mp4 + png
    python experiments/expC06_soft_neuron_interp/run_histogram.py --snapshots-only   # png only
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expC05_geometry_interpolation"))
from common import uniform_geometry, xavier_draw  # exact expC05 geometry + Glorot draw

RESULTS_DIR = (REPO_ROOT / "results" / "checkpoint_C_geometry"
               / "expC06_soft_neuron_interp" / "histogram")

WIDTHS = [64, 128, 256, 512]      # expC05's four widths (W = N + 2*halo + 1 is much larger)
SEED = 0                          # expC05 SEED
N_INTERP = 100                    # interpolation points (s linear in [0,1])
NBINS = 70
XMAX = 2.15                       # |w|/mean ~ uniform on [0,2]; pad for the tail
FPS = 12


def base_magnitudes():
    """Per-width L1-normalized Xavier weight magnitudes (the s=0 endpoint)."""
    out = {}
    for N in WIDTHS:
        c_uniform, _h, _halo = uniform_geometry(N)
        W = c_uniform.size
        w, _b = xavier_draw(W, SEED, N)
        a = np.abs(w)
        out[N] = (a / a.mean(), W)            # mean|.| = 1
    return out


def mags_at(base, s):
    """|w_i(s)| = (1-s)*base_i + s*1  (linear per element; mean|w| stays 1)."""
    return (1.0 - s) * base + s * 1.0


# Fixed bins / per-subplot y-limits so frames are directly comparable.
BINS = np.linspace(0.0, XMAX, NBINS + 1)
_FLAT = XMAX / NBINS / 2.0                     # bin width / domain[0,2] -> flat fraction


def _ylim(W):
    # ~10x the uniform (s=0) per-bin count; the late delta spike clips (intended).
    return 10.0 * W * (XMAX / NBINS) / 2.0


def _style_ax(ax, N, W, s):
    ax.set_xlim(0.0, XMAX)
    ax.set_ylim(0.0, _ylim(W))
    ax.set_xlabel(r"$|w_i|$  (mean$|w|=1$)")
    ax.set_ylabel("count")
    ax.set_title(f"N = {N}   (W = {W} weights)", fontsize=11)
    ax.axvline(1.0, color="0.45", ls=":", lw=1.0, zorder=1)   # the 1s-vector target


def make_animation(base, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axflat = axes.ravel()
    ss = np.linspace(0.0, 1.0, N_INTERP)

    def draw(frame):
        s = ss[frame]
        color = plt.cm.viridis(s)             # echo expC05's viridis-by-s coloring
        for ax, N in zip(axflat, WIDTHS):
            b, W = base[N]
            ax.clear()
            ax.hist(mags_at(b, s), bins=BINS, color=color,
                    edgecolor="white", linewidth=0.2, zorder=2)
            _style_ax(ax, N, W, s)
        fig.suptitle(
            rf"expC06: Xavier $\to$ $\mathbf{{1}}$ weight-magnitude histogram   "
            rf"($s={s:0.2f}$, frame {frame+1}/{N_INTERP})",
            fontsize=14)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=N_INTERP, blit=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=animation.FFMpegWriter(fps=FPS, bitrate=2400))
    plt.close(fig)
    print(f"Saved {out_path}")


def make_snapshots(base, out_path):
    """Static contact sheet: rows = snapshot s values, cols = widths."""
    s_snaps = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    nr, nc = len(s_snaps), len(WIDTHS)
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.3 * nr), squeeze=False)
    fig.suptitle(r"expC06: Xavier $\to$ $\mathbf{1}$ weight-magnitude histograms "
                 r"(rows = interpolation $s$, cols = width)", fontsize=13, y=0.995)
    for ri, s in enumerate(s_snaps):
        color = plt.cm.viridis(s)
        for ci, N in enumerate(WIDTHS):
            b, W = base[N]
            ax = axes[ri][ci]
            ax.hist(mags_at(b, s), bins=BINS, color=color,
                    edgecolor="white", linewidth=0.2, zorder=2)
            ax.set_xlim(0.0, XMAX); ax.set_ylim(0.0, _ylim(W))
            ax.axvline(1.0, color="0.45", ls=":", lw=1.0, zorder=1)
            ax.set_yticklabels([])
            if ri == 0:
                ax.set_title(f"N = {N} (W = {W})", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"s = {s:0.2f}", fontsize=10)
            if ri == nr - 1:
                ax.set_xlabel(r"$|w_i|$", fontsize=9)
            else:
                ax.set_xticklabels([])
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    base = base_magnitudes()
    make_snapshots(base, RESULTS_DIR / "weight_hist_snapshots.png")
    if "--snapshots-only" not in sys.argv:
        make_animation(base, RESULTS_DIR / "weight_hist_interp.mp4")
