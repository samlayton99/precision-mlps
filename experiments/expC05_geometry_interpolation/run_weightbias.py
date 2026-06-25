"""Experiment 20 -- weight + bias interpolation (belongs in checkpoint 2).

Completes the exp16 / exp19 pair. exp16 fixed the ideal inner weight (gamma for
every neuron) and interpolated the center geometry; exp19 fixed the ideal geometry
(biases) and interpolated the inner weights. exp20 interpolates BOTH at once: the
genuine Xavier-initialized inner layer (w_xavier, b_xavier) is moved, along two
independent axes, onto the exact QI inner layer (weight = gamma for all neurons,
bias = -gamma * c_uniform).

Unlike the other two, gamma (= lambda) is NOT a swept axis here -- it is held
CONSTANT per (N, target). The held value is chosen by a detailed lambda sweep at the
ideal geometry (uniform centers + the 1*gamma weight vector = exact QI), taking the
argmin relative-L2 per function. The ideal geometry is seed-independent, so lambda*
depends only on (N, target) and is computed once and reused across the seed folders.

The two interpolation axes, each with K points -> a K x K grid (gamma* fixed):
  - weight uniformness s in [0, 1]:  w_i(s) = (1 - s) * w_xavier_i + s * gamma*.
        s = 0 -> the raw Xavier weights (random, some negative); s = 1 -> the constant
        1*gamma vector (every weight = gamma*).
  - bias uniformness t in [0, 1]:    b_i(t) = (1 - t) * b_xavier_i + t * (-gamma* * c_i).
        t = 0 -> the raw Xavier biases; t = 1 -> -gamma* * c_uniform, the bias vector that
        encodes the ideal uniform centers.
  (s = 0, t = 0) is the genuine Xavier init; (s = 1, t = 1) is tanh(gamma*(x - c_i)),
  the exact QI geometry, which should reach the fp64 floor. Along the t = 1 edge exp20
  reduces exactly to exp19.

The Xavier weights and biases are ordered by their Xavier center -b/w (the same neuron
ordering exp16 / exp19 use) and the i-th neuron is pinned to the i-th ascending uniform
grid center, so the t = 1 bias target -gamma * c_uniform[i] matches neuron i.

Per cell (one target) gamma* is fixed, so unlike exp16 / exp19 the design matrix differs
per target and we solve one right-hand side per (s, t) with its own truncated-SVD.

Grid: 3 seed folders x 4 targets x 4 widths (N=64,128,256,512) x K x K.

Figures: one PNG per (seed folder, target) under results/.../<folder>/, a 4 x 3 grid
(rows = widths). Col 1 = K x K error heatmap (x = weight s, y = bias t, color = rel L2).
Col 2 = error vs weight s, one line per bias-t slice (colored by t). Col 3 = error vs
bias t, one line per weight-s slice (colored by s). lambda* annotated per row. Plus
results/.../lambda_sweeps.png: the ideal-geometry U-curves and the chosen lambda* per
(N, target).

Usage:
    python experiments/expC05_geometry_interpolation/run_weightbias.py            # collect + plot
    python experiments/expC05_geometry_interpolation/run_weightbias.py --plot     # plot from saved data
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

sys.path.insert(0, str(Path(__file__).resolve().parent))   # for `common`
from common import (EXP_RESULTS, HALO_LAMBDA, N_TRAIN, N_EVAL, RCOND,
                    uniform_geometry, solve_relL2, xavier_draw, center_order, get_target)

RESULTS_DIR = EXP_RESULTS / "weightbias"
OUT_DIR = RESULTS_DIR
DATA_PATH = OUT_DIR / "data.json"

# --- experiment grid ---
WIDTHS = [64, 128, 256, 512]
TARGETS = ["sine", "sine_8pi", "runge", "exp"]
SEED_FOLDERS = {"seed_1": 0, "seed_2": 1, "seed_3": 2}   # one init per folder, shared across targets
K = 21                                            # K x K interpolation grid (weight s) x (bias t)
HALO_LAMBDA = 0.25                                # sizes the uniform grid's halo
N_TRAIN = 2003                                    # prime, overdetermined for all W (W<=923 at N=512)
N_EVAL = 4001                                     # prime, misaligned with the train grid
RCOND = 1e-13                                     # SVD truncation (relative to s_max)

# Detailed lambda sweep at the ideal geometry -> lambda* held constant per (N, target).
LAMBDA_MIN = 0.03
LAMBDA_MAX = 1.5
LAMBDA_POINTS = 81
# uniform_geometry, N_TRAIN/N_EVAL/RCOND/HALO_LAMBDA, solve_relL2 come from `common`.


def xavier_params(W, seed, N):
    """The genuine Xavier-init inner layer, ordered to match the centers / weights
    modes' neuron ordering. Draw per-neuron (w, b) (Glorot bound sqrt(6/(1+W))), order
    the neurons by their Xavier center c = -b/w (ascending, dead |w|<eps parked by bias
    sign) and return both vectors in that order. The i-th neuron is then pinned to the
    i-th ascending uniform grid center. (w[order], b[order]) is the same function as the
    raw draw (neurons are exchangeable). Seeded per (folder seed, width)."""
    w, b = xavier_draw(W, seed, N)
    order = center_order(w, b)
    return w[order], b[order]


def ideal_lambda_sweep(N, c_uniform, h, X_tr, X_ev, Y_tr, Y_ev, y_norms):
    """Detailed lambda sweep at the IDEAL geometry (exact QI: weight = gamma, centers =
    uniform). Returns (lambdas, err[target, lambda], lambda_star[target]). Seed-independent."""
    lambdas = np.logspace(np.log10(LAMBDA_MIN), np.log10(LAMBDA_MAX), LAMBDA_POINTS)
    err = np.zeros((len(TARGETS), len(lambdas)))
    for li, lam in enumerate(lambdas):
        g = lam / h
        Phi_tr = np.tanh(g * (X_tr[:, None] - c_uniform[None, :]))
        Phi_ev = np.tanh(g * (X_ev[:, None] - c_uniform[None, :]))
        err[:, li] = solve_relL2(Phi_tr, Phi_ev, Y_tr, Y_ev, y_norms)
    star_idx = np.argmin(err, axis=1)
    lambda_star = lambdas[star_idx]
    return lambdas, err, lambda_star


def collect_data():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_tr = np.linspace(-1.0, 1.0, N_TRAIN)
    X_ev = np.linspace(-1.0, 1.0, N_EVAL)
    ss = np.linspace(0.0, 1.0, K)                        # weight-uniformness axis
    ts = np.linspace(0.0, 1.0, K)                        # bias-uniformness axis
    Y_tr = np.stack([get_target(t).fn_numpy(X_tr) for t in TARGETS], axis=1)   # [n_tr, 4]
    Y_ev = np.stack([get_target(t).fn_numpy(X_ev) for t in TARGETS], axis=1)   # [n_ev, 4]
    y_norms = np.linalg.norm(Y_ev, axis=0)               # [4]

    cells = []
    ideal_sweeps = []
    t0 = time.time()
    total = len(SEED_FOLDERS) * len(WIDTHS) * len(TARGETS)
    done = 0
    for N in WIDTHS:
        c_uniform, h, halo = uniform_geometry(N)
        W = c_uniform.size

        # --- pick lambda* per target at the ideal geometry (seed-independent) ---
        lambdas, ideal_err, lambda_star = ideal_lambda_sweep(
            N, c_uniform, h, X_tr, X_ev, Y_tr, Y_ev, y_norms)
        gamma_star = lambda_star / h
        for ti, target in enumerate(TARGETS):
            interior = 0 < int(np.argmin(ideal_err[ti])) < LAMBDA_POINTS - 1
            ideal_sweeps.append({
                "N": N, "target": target, "W": int(W),
                "lambdas": lambdas.tolist(), "err": ideal_err[ti].tolist(),
                "lambda_star": float(lambda_star[ti]), "gamma_star": float(gamma_star[ti]),
                "ideal_err": float(ideal_err[ti].min()), "interior": bool(interior),
            })
        flag = "" if all(0 < int(np.argmin(ideal_err[ti])) < LAMBDA_POINTS - 1
                         for ti in range(len(TARGETS))) else "  [!] boundary lambda*"
        print(f"N={N:4d} (W={W}) lambda*: "
              + " ".join(f"{TARGETS[ti][:5]}={lambda_star[ti]:.3f}(e={ideal_err[ti].min():.1e})"
                         for ti in range(len(TARGETS))) + flag)

        # --- K x K weight/bias interpolation per (seed folder, target) ---
        for folder, fseed in SEED_FOLDERS.items():
            w_x, b_x = xavier_params(W, fseed, N)
            for ti, target in enumerate(TARGETS):
                g = gamma_star[ti]
                b_target = -g * c_uniform                 # ideal (t=1) bias
                yt = Y_tr[:, ti:ti + 1]; ye = Y_ev[:, ti:ti + 1]; yn = y_norms[ti:ti + 1]
                err = np.zeros((K, K))                    # [s_idx, t_idx]
                for si, s in enumerate(ss):
                    w = (1.0 - s) * w_x + s * g
                    Wcol = w[None, :]
                    for tj, t in enumerate(ts):
                        b = (1.0 - t) * b_x + t * b_target
                        Phi_tr = np.tanh(Wcol * X_tr[:, None] + b[None, :])
                        Phi_ev = np.tanh(Wcol * X_ev[:, None] + b[None, :])
                        err[si, tj] = solve_relL2(Phi_tr, Phi_ev, yt, ye, yn)[0]
                cells.append({
                    "folder": folder, "target": target, "N": N, "W": int(W),
                    "halo": int(halo), "seed": fseed,
                    "lambda_star": float(lambda_star[ti]), "gamma_star": float(g),
                    "ss": ss.tolist(), "ts": ts.tolist(), "err": err.tolist(),
                })
                done += 1
                print(f"  [{done}/{total}] {folder} N={N:4d} {target:8s} "
                      f"lambda*={lambda_star[ti]:.3f} | "
                      f"xavier(0,0)={err[0,0]:.1e} ideal(1,1)={err[-1,-1]:.1e} "
                      f"min={err.min():.1e} | {time.time()-t0:.0f}s")

    out = {
        "config": {
            "widths": WIDTHS, "targets": TARGETS, "seed_folders": SEED_FOLDERS,
            "K": K, "halo_lambda": HALO_LAMBDA, "n_train": N_TRAIN, "n_eval": N_EVAL,
            "rcond": RCOND, "lambda_range": [LAMBDA_MIN, LAMBDA_MAX],
            "lambda_points": LAMBDA_POINTS,
        },
        "cells": cells,
        "ideal_sweeps": ideal_sweeps,
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
    nrows = len(widths)
    fig, axes = plt.subplots(nrows, 3, figsize=(19, 4.6 * nrows))
    fig.suptitle(f"Exp20: weight + bias interpolation -- {target} ({folder}) "
                 r"(Xavier $(w,b)\to(\mathbf{1}\gamma,\,-\gamma c_{\rm unif})$; "
                 r"$\gamma$ fixed per row; lstsq, fp64, relative $L_2$)",
                 fontsize=15, y=0.999)

    for ri, N in enumerate(widths):
        c = _cell(data["cells"], folder, target, N)
        ss = np.array(c["ss"]); ts = np.array(c["ts"]); err = np.array(c["err"])   # [s, t]
        lam = c["lambda_star"]
        emin = max(err.min(), 1e-16); emax = err.max()

        # --- col 1: K x K heatmap (x = weight s, y = bias t) ---
        ax = axes[ri][0]
        im = ax.pcolormesh(ss, ts, err.T, norm=LogNorm(vmin=emin, vmax=emax),
                           cmap="inferno", shading="auto")
        ax.set_ylabel(f"N = {N},  " + rf"$\lambda^* = {lam:.3f}$" + "\nbias uniformness $t$")
        ax.set_xlabel("weight uniformness $s$")
        if ri == 0:
            ax.set_title("error heatmap (rel $L_2$)")
        fig.colorbar(im, ax=ax, shrink=0.85, label="rel $L_2$")

        # --- col 2: error vs weight s, one line per bias-t slice ---
        ax = axes[ri][1]
        cmap2, norm2 = plt.cm.viridis, Normalize(0.0, 1.0)
        for tj, t in enumerate(ts):
            ax.plot(ss, err[:, tj], "-", lw=1.1, color=cmap2(norm2(t)))
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel("weight uniformness $s$"); ax.set_ylabel("rel $L_2$")
        if ri == 0:
            ax.set_title("error vs weight $s$ (lines = bias $t$ slices)")
        fig.colorbar(ScalarMappable(norm=norm2, cmap=cmap2), ax=ax,
                     shrink=0.85, label="bias uniformness $t$")

        # --- col 3: error vs bias t, one line per weight-s slice ---
        ax = axes[ri][2]
        cmap3, norm3 = plt.cm.plasma, Normalize(0.0, 1.0)
        for si, s in enumerate(ss):
            ax.plot(ts, err[si, :], "-", lw=1.1, color=cmap3(norm3(s)))
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel("bias uniformness $t$"); ax.set_ylabel("rel $L_2$")
        if ri == 0:
            ax.set_title("error vs bias $t$ (lines = weight $s$ slices)")
        fig.colorbar(ScalarMappable(norm=norm3, cmap=cmap3), ax=ax,
                     shrink=0.85, label="weight uniformness $s$")

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_dir = OUT_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"interp_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_lambda_sweeps(data):
    """Ideal-geometry U-curves and the chosen lambda* per (N, target)."""
    sweeps = data["ideal_sweeps"]
    widths = data["config"]["widths"]
    targets = data["config"]["targets"]
    nrows = len(widths)
    fig, axes = plt.subplots(nrows, 1, figsize=(8.5, 3.4 * nrows), squeeze=False)
    fig.suptitle("Exp20: ideal-geometry $\\lambda$ sweep (exact QI; rel $L_2$) "
                 "-- vertical marks = chosen $\\lambda^*$", fontsize=13, y=0.999)
    cmap = plt.cm.tab10
    for ri, N in enumerate(widths):
        ax = axes[ri][0]
        for ti, target in enumerate(targets):
            sw = next(s for s in sweeps if s["N"] == N and s["target"] == target)
            lam = np.array(sw["lambdas"]); e = np.array(sw["err"])
            ax.plot(lam, e, "-", lw=1.3, color=cmap(ti), label=target)
            ax.axvline(sw["lambda_star"], color=cmap(ti), ls="--", lw=1.0, alpha=0.7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylabel(f"N = {N}\nrel $L_2$"); ax.set_xlabel(r"$\lambda = \gamma\,h$")
        if ri == 0:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04),
                      ncol=len(targets), borderaxespad=0, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out = OUT_DIR / "lambda_sweeps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_all(data):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_lambda_sweeps(data)
    for folder in data["config"]["seed_folders"]:
        for target in data["config"]["targets"]:
            plot_target(data, folder, target)


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
