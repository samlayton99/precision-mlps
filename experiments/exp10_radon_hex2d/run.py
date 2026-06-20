"""Experiment 10 -- 2D ridge/Radon regime: hexagonal tangent-line geometry.

Question: extend the 1D "fixed geometry + lstsq readout" result to R^2 -> R.
Each tanh neuron is a ridge whose transition line is tangent to a circle at a
hexagonally-packed grid point (see src/construction/hex_geometry.py). With the
geometry FROZEN, we build Phi and solve the readout by least squares (the
method exp03/0B showed is as strong as the QI construction in fp64), then sweep
the bandwidth and width to ask: does this structured, training-free geometry
reach the fp64 floor on smooth 2D targets, and how does the error scale with N?

Setup (all fp64, numpy):
  - Disk radius R = 1 + halo (default 1.4); targets fit/evaluated on the unit
    disk (radius 1); neurons pack out to R.
  - Bandwidth is dual: by default we FIX lambda and sweep it (gamma = lambda/d
    backed out via the hex spacing d(N)); set SWEEP_VAR = "gamma" to fix/sweep
    gamma instead (lambda = gamma*d backed out). Both reported every cell.

Figures (all error on the eval grid):
  1. error_vs_lambda.png  -- targets (rows) x {rel L2, L_inf} (cols), one curve
     per width. The U-shape in the swept bandwidth.
  2. error_vs_width.png   -- best-over-bandwidth error vs N, one curve per
     target. The scaling-law question.
  3. conditioning.png     -- cond(Phi) and null dim vs bandwidth / width.

Usage:
    python experiments/exp10_radon_hex2d/run.py            # collect + plot
    python experiments/exp10_radon_hex2d/run.py --plot     # plot from saved data
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.hex_geometry import build_hex_geometry, build_phi_2d
from src.construction.readout import solve_readout_with_bias
from src.data.sampling2d import disk_grid, disk_uniform
from src.data.targets2d import get_target_2d

RESULTS_DIR = REPO_ROOT / "results" / "exp10_radon_hex2d"
DATA_PATH = RESULTS_DIR / "phase1_data.json"

# --- experiment grid ---
TARGETS = ["sine2d", "sine2d_hi", "gauss_bump", "runge2d", "mixed2d", "planewave"]
WIDTHS = [37, 61, 100, 127, 271, 547]      # hex numbers + two partial-ring N
HALO = 0.4
RADIUS = 1.0 + HALO                          # disk that holds the neurons
N_TRAIN = 8000                               # area-uniform disk samples
N_EVAL_SIDE = 160                            # Cartesian grid clipped to unit disk

# Bandwidth sweep: default fix lambda, sweep it; set "gamma" to swap roles.
SWEEP_VAR = "lambda"                         # "lambda" | "gamma"
LAMBDA_GRID = [0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.60, 0.90, 1.40]
GAMMA_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
RCOND = 1e-13                                # null-space / rank cutoff (rel to smax)

TARGET_COLORS = {
    "sine2d": "#1f77b4", "sine2d_hi": "#ff7f0e", "gauss_bump": "#2ca02c",
    "runge2d": "#d62728", "mixed2d": "#9467bd", "planewave": "#8c564b",
}


def sweep_values():
    return LAMBDA_GRID if SWEEP_VAR == "lambda" else GAMMA_GRID


def build_geometry(N, val):
    if SWEEP_VAR == "lambda":
        return build_hex_geometry(N, radius=RADIUS, lambda_star=val)
    return build_hex_geometry(N, radius=RADIUS, gamma=val)


def diagnostics(Phi_aug):
    """Conditioning / rank diagnostics from the augmented [Phi, 1] train matrix."""
    s = np.linalg.svd(Phi_aug, compute_uv=False)
    smax = float(s[0]); smin = float(s[-1])
    keep = s > RCOND * smax
    rank = int(keep.sum())
    return {
        "phi_cond": float(smax / smin) if smin > 0 else float("inf"),
        "phi_smin": smin, "phi_smax": smax,
        "rank": rank, "null_dim": int(Phi_aug.shape[1] - rank),
        "stable_rank": float((s ** 2).sum() / smax ** 2),
        "cols": int(Phi_aug.shape[1]),
    }


def collect_data():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X_tr = disk_uniform(N_TRAIN, radius=1.0, seed=0)
    X_ev = disk_grid(N_EVAL_SIDE, radius=1.0)
    vals = sweep_values()

    cells = []
    t0 = time.time()
    total = len(TARGETS) * len(WIDTHS) * len(vals)
    done = 0

    for tname in TARGETS:
        tgt = get_target_2d(tname)
        y_tr = tgt.fn_numpy(X_tr)
        y_ev = tgt.fn_numpy(X_ev)
        y_norm = float(np.linalg.norm(y_ev))

        for N in WIDTHS:
            best_linf = np.inf
            for val in vals:
                geo = build_geometry(N, val)
                Phi_tr = build_phi_2d(X_tr, geo["W"], geo["beta"])
                Phi_ev = build_phi_2d(X_ev, geo["W"], geo["beta"])
                v, b, info = solve_readout_with_bias(Phi_tr, y_tr, method="lstsq")

                resid = Phi_ev @ v + b - y_ev
                linf = float(np.abs(resid).max())
                rel_l2 = float(np.linalg.norm(resid) / y_norm)

                diag = diagnostics(np.hstack([Phi_tr, np.ones((N_TRAIN, 1))]))
                cells.append({
                    "target": tname, "N": N, "neurons": N,
                    "sweep_var": SWEEP_VAR, "sweep_val": float(val),
                    "lambda": float(geo["lambda_val"]), "gamma": float(geo["gamma"]),
                    "spacing_d": float(geo["spacing_d"]),
                    "linf": linf, "rel_l2": rel_l2,
                    "train_resid": float(info["residual_norm"]),
                    "max_v": float(np.abs(v).max()), "v_l2": float(np.linalg.norm(v)),
                    **diag,
                })
                best_linf = min(best_linf, linf)
                done += 1
            print(f"[{done}/{total}] {tname} N={N} | best Linf={best_linf:.2e} "
                  f"| {time.time() - t0:.0f}s")

    out = {
        "config": {
            "targets": TARGETS, "widths": WIDTHS, "halo": HALO, "radius": RADIUS,
            "n_train": N_TRAIN, "n_eval_side": N_EVAL_SIDE,
            "sweep_var": SWEEP_VAR, "lambda_grid": LAMBDA_GRID,
            "gamma_grid": GAMMA_GRID, "rcond": RCOND,
        },
        "cells": cells,
    }
    with open(DATA_PATH, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {DATA_PATH} ({time.time() - t0:.0f}s total)")
    return out


# ----------------------------- plotting -----------------------------

def _cells(data, **filt):
    return [c for c in data["cells"]
            if all(c[k] == v for k, v in filt.items())]


def plot_error_vs_bandwidth(data):
    cfg = data["config"]
    sv = cfg["sweep_var"]
    targets, widths = cfg["targets"], cfg["widths"]
    metrics = [("rel_l2", r"Relative $L_2$"), ("linf", r"$L_\infty$")]
    nrows = len(targets)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 3.4 * nrows), sharex=True)
    fig.suptitle(f"Exp10: eval error vs {sv} (frozen hex tangent geometry, "
                 f"R={cfg['radius']})", fontsize=14, y=0.997)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(widths)))

    for ri, tname in enumerate(targets):
        for ci, (mkey, mlabel) in enumerate(metrics):
            ax = axes[ri][ci]
            for wi, N in enumerate(widths):
                recs = sorted(_cells(data, target=tname, N=N),
                              key=lambda c: c["sweep_val"])
                xs = [c["sweep_val"] for c in recs]
                ys = [c[mkey] for c in recs]
                ax.loglog(xs, ys, "-o", color=cmap[wi], markersize=3, label=f"N={N}")
            ax.grid(True, alpha=0.3, which="both")
            ax.set_ylabel(f"{tname}\n{mlabel}" if ci == 0 else mlabel)
            if ri == 0:
                ax.set_title(mlabel)
            if ri == nrows - 1:
                ax.set_xlabel(sv)
    axes[0][0].legend(fontsize=8, ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out = RESULTS_DIR / "error_vs_lambda.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_error_vs_width(data):
    cfg = data["config"]
    targets, widths = cfg["targets"], cfg["widths"]
    metrics = [("rel_l2", r"Best relative $L_2$"), ("linf", r"Best $L_\infty$")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Exp10: best-over-bandwidth eval error vs width", fontsize=14)
    for ci, (mkey, mlabel) in enumerate(metrics):
        ax = axes[ci]
        for tname in targets:
            ys = [min(c[mkey] for c in _cells(data, target=tname, N=N))
                  for N in widths]
            ax.loglog(widths, ys, "-o", color=TARGET_COLORS.get(tname),
                      markersize=5, label=tname)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel("width $N$"); ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
    axes[0].legend(fontsize=9)
    plt.tight_layout()
    out = RESULTS_DIR / "error_vs_width.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_conditioning(data):
    cfg = data["config"]
    sv = cfg["sweep_var"]
    widths = cfg["widths"]
    # cond(Phi) is target-independent (geometry only); use the first target.
    t0 = cfg["targets"][0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Exp10: feature-matrix conditioning (geometry only)", fontsize=14)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(widths)))
    for wi, N in enumerate(widths):
        recs = sorted(_cells(data, target=t0, N=N), key=lambda c: c["sweep_val"])
        xs = [c["sweep_val"] for c in recs]
        axes[0].loglog(xs, [c["phi_cond"] for c in recs], "-o",
                       color=cmap[wi], markersize=3, label=f"N={N}")
        axes[1].semilogx(xs, [c["null_dim"] for c in recs], "-o",
                         color=cmap[wi], markersize=3, label=f"N={N}")
    axes[0].set_ylabel(r"cond$(\Phi)$"); axes[0].set_title(r"cond$(\Phi)$ vs " + sv)
    axes[1].set_ylabel("null dim of $[\\Phi, 1]$")
    axes[1].set_title("null dim vs " + sv)
    for ax in axes:
        ax.set_xlabel(sv); ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8)
    plt.tight_layout()
    out = RESULTS_DIR / "conditioning.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_all(data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_error_vs_bandwidth(data)
    plot_error_vs_width(data)
    plot_conditioning(data)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        if not DATA_PATH.exists():
            print(f"No data at {DATA_PATH}. Run without --plot first.")
            sys.exit(1)
        with open(DATA_PATH) as f:
            data = json.load(f)
    else:
        data = collect_data()
    plot_all(data)
