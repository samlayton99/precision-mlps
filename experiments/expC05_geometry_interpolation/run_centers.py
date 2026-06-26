"""Experiment expC05 -- geometry interpolation (checkpoint C).

Interpolates a random tanh MLP toward the ideal QI geometry along two axes and
watches how the (lstsq, fp64) relative-L2 error changes, to shed light on the
two geometric barriers an optimizer faces: the ideal bandwidth (gamma/lambda)
and the ideal center layout (uniform spacing + halo).

Geometry handled exactly like the other geometry experiments (expC04, expD01): the
center POSITIONS are the primitive -- fixed, independent of the bandwidth -- and
gamma is a separate global bandwidth applied on top. This keeps the two axes
orthogonal and the comparison apples-to-apples.

  - Random anchor (t=0): the EXACT Xavier-initialized inner layer. Draw per-neuron
    weights w_i and biases b_i (both Xavier-uniform, Glorot bound sqrt(6/(1+W))) and
    take the genuine tanh centers c_i = -b_i/w_i -- the same extraction expC04 uses for
    its "trained" geometry (dead |w|<eps neurons parked, all clipped to the span).
    These positions are FIXED; they do not depend on gamma.
  - One Xavier draw per SEED FOLDER and width, shared across the four targets (the
    folder is the unit of initialization).

Two interpolation axes, each with K points -> a K x K grid:
  - uniformness t in [0, 1]: match the sorted Xavier centers one-to-one to the
    ascending uniform QI grid (grid + halo) and interpolate the POSITIONS
    c_i(t) = (1 - t) * c_xavier_i + t * c_uniform_i. Both anchors span the same domain,
    so t purely varies how uniform the centers are; t=1 is the exact QI geometry.
    gamma never enters here.
  - gamma in logspace(0.25, N, K): O(1) -> O(N), the shared bandwidth. The neuron is
    tanh(gamma*(x - c_i(t))); changing gamma does NOT move the centers (the bias is
    -gamma*c_i(t)). lambda = gamma * h = gamma * 2/N.

Per cell we build Phi once, take one truncated-SVD factorization (bias column),
and solve all four targets as multiple right-hand sides, recording relative L2 on
a dense eval grid. The (t=1, gamma~ideal) corner is the exact QI geometry and
should reach the fp64 floor.

Grid: 3 seed folders x 4 targets x 4 widths (N=64,128,256,512) x K x K.

Figure: one PNG per (seed folder, target), saved under results/.../<folder>/.
Each is a 4 x 3 grid (rows = widths). Col 1 = K x K error heatmap (x = uniformness,
y = lambda log, color = rel L2). Col 2 = error vs lambda, one line per uniformness
slice (colored by uniformness; lambda x-limits shared across rows). Col 3 = error
vs uniformness, one line per gamma slice (colored by gamma).

Usage:
    python experiments/expC05_geometry_interpolation/run_centers.py            # collect + plot
    python experiments/expC05_geometry_interpolation/run_centers.py --plot     # plot from saved data
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))   # for `common`
from common import (EXP_RESULTS, HALO_LAMBDA, N_TRAIN, N_EVAL, RCOND,
                    uniform_geometry, solve_relL2, xavier_draw, get_target)
from src.construction.center_geometry import trained_centers

RESULTS_DIR = EXP_RESULTS / "centers"

# Output dir + random-anchor mode. Defaults run the Xavier experiment into
# RESULTS_DIR; `--uniform-init` (see __main__) redirects to RESULTS_DIR/uniform_init,
# samples the random anchor UNIFORMLY over the span, and drops N=512 -- so the
# t-axis then varies uniformity only (both endpoints span the same range).
ANCHOR_MODE = "xavier"          # "xavier" (c=-b/w) | "uniform" (random over span)
OUT_DIR = RESULTS_DIR
DATA_PATH = OUT_DIR / "data.json"

# --- experiment grid ---
WIDTHS = [64, 128, 256, 512]
TARGETS = ["sine", "sine_8pi", "runge", "exp"]
SEED_FOLDERS = {"seed_1": 0, "seed_2": 1, "seed_3": 2}   # folder -> base seed (per init)
K = 21                                            # K x K interpolation grid
GAMMA_MIN = 0.25                                  # gamma_max = N per width
GLOBAL_LAMBDA_MAX = 2.0                           # lambda axis top (shared across rows)
# uniform_geometry, N_TRAIN/N_EVAL/RCOND/HALO_LAMBDA, solve_relL2 come from `common`.


def xavier_centers(W, span, seed, N):
    """The exact Xavier-init geometry: draw per-neuron weights and biases (both
    Xavier-uniform, Glorot bound sqrt(6/(1 + W))) and return the genuine tanh
    centers c = -b/w, clipped to the span with dead |w| parked -- the same
    extraction expC04 uses for its trained geometry. Gamma-independent. Seeded per
    (folder seed, width)."""
    w, b = xavier_draw(W, seed, N)
    return trained_centers(w, b, span)        # sorted, clipped to span


def uniform_anchor_centers(W, span, seed, N):
    """Random anchor sampled UNIFORMLY over the correct (span) range -- so the
    t-axis varies uniformity ONLY, since both endpoints span the same range.
    Seeded per (folder seed, width)."""
    rng = np.random.default_rng([seed, N])
    return np.sort(rng.uniform(span[0], span[1], size=W)).astype(np.float64)


def anchor_centers(W, span, seed, N):
    """The random anchor positions, per ANCHOR_MODE (gamma-independent)."""
    if ANCHOR_MODE == "uniform":
        return uniform_anchor_centers(W, span, seed, N)
    return xavier_centers(W, span, seed, N)


def collect_data():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_tr = np.linspace(-1.0, 1.0, N_TRAIN)
    X_ev = np.linspace(-1.0, 1.0, N_EVAL)
    ts = np.linspace(0.0, 1.0, K)                        # uniformness axis
    Y_tr = np.stack([get_target(t).fn_numpy(X_tr) for t in TARGETS], axis=1)   # [n_tr, 4]
    Y_ev = np.stack([get_target(t).fn_numpy(X_ev) for t in TARGETS], axis=1)   # [n_ev, 4]
    y_norms = np.linalg.norm(Y_ev, axis=0)               # [4]

    cells = []
    t0 = time.time()
    total = len(SEED_FOLDERS) * len(WIDTHS)
    done = 0
    for folder, fseed in SEED_FOLDERS.items():
        for N in WIDTHS:
            c_uniform, h, halo = uniform_geometry(N)
            W = c_uniform.size
            span = (float(c_uniform[0]), float(c_uniform[-1]))
            c_rand = anchor_centers(W, span, fseed, N)    # fixed positions, gamma-independent
            gammas = np.logspace(np.log10(GAMMA_MIN), np.log10(N), K)
            lambdas = gammas * 2.0 / N

            err = np.zeros((len(TARGETS), K, K))         # [target, t_idx, gamma_idx]
            for ti, t in enumerate(ts):
                c = (1.0 - t) * c_rand + t * c_uniform    # positions only; gamma does not enter
                for gi, g in enumerate(gammas):
                    Phi_tr = np.tanh(g * (X_tr[:, None] - c[None, :]))
                    Phi_ev = np.tanh(g * (X_ev[:, None] - c[None, :]))
                    err[:, ti, gi] = solve_relL2(Phi_tr, Phi_ev, Y_tr, Y_ev, y_norms)

            for tigt, target in enumerate(TARGETS):
                cells.append({
                    "folder": folder, "target": target, "N": N, "W": int(W),
                    "halo": int(halo), "seed": fseed,
                    "ts": ts.tolist(), "gammas": gammas.tolist(),
                    "lambdas": lambdas.tolist(), "err": err[tigt].tolist(),
                })
            done += 1
            best = {TARGETS[k]: err[k].min() for k in range(len(TARGETS))}
            print(f"[{done}/{total}] {folder} N={N:4d} (W={W}) | "
                  + " ".join(f"{k[:5]}={v:.1e}" for k, v in best.items())
                  + f" | {time.time()-t0:.0f}s")

    out = {
        "config": {
            "widths": WIDTHS, "targets": TARGETS, "seed_folders": SEED_FOLDERS,
            "K": K, "gamma_min": GAMMA_MIN, "halo_lambda": HALO_LAMBDA,
            "n_train": N_TRAIN, "n_eval": N_EVAL, "rcond": RCOND,
            "anchor_mode": ANCHOR_MODE,
            "global_lambda": [GAMMA_MIN * 2.0 / max(WIDTHS), GLOBAL_LAMBDA_MAX],
        },
        "cells": cells,
    }
    with open(DATA_PATH, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {DATA_PATH} ({time.time()-t0:.0f}s total)")
    return out


# ----------------------------- plotting -----------------------------

def _cell(cells, folder, target, N):
    return next(c for c in cells if c["folder"] == folder
               and c["target"] == target and c["N"] == N)


def plot_target(data, folder, target):
    cfg = data["config"]
    widths = cfg["widths"]
    lam_lo, lam_hi = cfg["global_lambda"]
    nrows = len(widths)
    fig, axes = plt.subplots(nrows, 3, figsize=(19, 4.6 * nrows))
    fig.suptitle(f"expC05 (centers): geometry interpolation -- {target} ({folder}) "
                 r"(random $\to$ QI geometry; lstsq, fp64, relative $L_2$)",
                 fontsize=15, y=0.999)

    for ri, N in enumerate(widths):
        c = _cell(data["cells"], folder, target, N)
        ts = np.array(c["ts"]); gammas = np.array(c["gammas"])
        lambdas = np.array(c["lambdas"]); err = np.array(c["err"])   # [t, gamma]
        emin = max(err.min(), 1e-16); emax = err.max()

        # --- col 1: K x K heatmap (x = uniformness, y = lambda log) ---
        ax = axes[ri][0]
        im = ax.pcolormesh(ts, lambdas, err.T, norm=LogNorm(vmin=emin, vmax=emax),
                           cmap="inferno", shading="auto")
        ax.set_yscale("log")
        ax.set_ylabel(f"N = {N}\n" + r"$\lambda = \gamma\,h$")
        ax.set_xlabel("uniformness $t$")
        if ri == 0:
            ax.set_title("error heatmap (rel $L_2$)")
        fig.colorbar(im, ax=ax, shrink=0.85, label="rel $L_2$")

        # --- col 2: error vs lambda, one line per uniformness slice ---
        ax = axes[ri][1]
        cmap2, norm2 = plt.cm.viridis, Normalize(0.0, 1.0)
        for ti, t in enumerate(ts):
            ax.plot(lambdas, err[ti, :], "-", lw=1.1, color=cmap2(norm2(t)))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lam_lo * 0.8, lam_hi * 1.15)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel(r"$\lambda = \gamma\,h$"); ax.set_ylabel("rel $L_2$")
        if ri == 0:
            ax.set_title("error vs $\\lambda$ (lines = uniformness slices)")
        fig.colorbar(ScalarMappable(norm=norm2, cmap=cmap2), ax=ax,
                     shrink=0.85, label="uniformness $t$")

        # --- col 3: error vs uniformness, one line per lambda slice ---
        ax = axes[ri][2]
        # color by lambda on the GLOBAL range, so this lines up with cols 1 & 2 (which are in lambda)
        cmap3, norm3 = plt.cm.plasma, LogNorm(lam_lo, lam_hi)
        for gi in range(len(gammas)):
            ax.plot(ts, err[:, gi], "-", lw=1.1, color=cmap3(norm3(lambdas[gi])))
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel("uniformness $t$"); ax.set_ylabel("rel $L_2$")
        if ri == 0:
            ax.set_title(r"error vs uniformness (lines = $\lambda$ slices)")
        fig.colorbar(ScalarMappable(norm=norm3, cmap=cmap3), ax=ax,
                     shrink=0.85, label=r"$\lambda = \gamma\,h$")

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_dir = OUT_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"interp_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_all(data):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for folder in data["config"]["seed_folders"]:
        for target in data["config"]["targets"]:
            plot_target(data, folder, target)


if __name__ == "__main__":
    # `--uniform-init`: random anchor sampled uniformly over the span, no N=512,
    # results under results/.../uniform_init/ -- so the t-axis is uniformity only.
    if "--uniform-init" in sys.argv:
        ANCHOR_MODE = "uniform"
        WIDTHS = [64, 128, 256]
        OUT_DIR = RESULTS_DIR / "uniform_init"
        DATA_PATH = OUT_DIR / "data.json"

    if "--plot" in sys.argv:
        if not DATA_PATH.exists():
            print(f"No data at {DATA_PATH}. Run without --plot first.")
            sys.exit(1)
        with open(DATA_PATH) as f:
            data = json.load(f)
    else:
        data = collect_data()
    plot_all(data)
