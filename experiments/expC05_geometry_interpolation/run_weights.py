"""expC05 weights mode -- weight-pattern interpolation at fixed bandwidth scale and fixed centers.

Mirror of the centers mode. The centers mode fixes the inner weight at the ideal (gamma for
every neuron) and moves the center positions; this mode fixes the centers at the ideal uniform
QI grid AND the bandwidth SCALE at gamma, and varies only the per-neuron weight PATTERN. This
isolates "must the per-neuron bandwidths be identical?" from the bandwidth (lambda) and the
center-placement questions.

Construction (per neuron i, interpolation s in [0,1], swept bandwidth gamma):
    w_hat = w_xavier / rms(w_xavier)          # unit-RMS Xavier direction (keeps spread + signs)
    u(s)  = (1 - s) * w_hat + s * 1           # pattern: Xavier spread -> all-ones
    w(s)  = gamma * u(s) / rms(u(s))          # RMS-normalized: rms(w) = gamma for ALL s
    b(s)  = -w(s) * c_uniform                 # bias tracks weight, so center -b/w = c_i (pinned)
    preact = w(s) * (x - c_i)                 # tanh(w_i (x - c_i)), center c_i fixed

So along s only the per-neuron weight pattern (magnitude spread + sign) changes; the overall
bandwidth scale rms(w) = gamma and the centers c_i are held EXACTLY fixed (both asserted at
runtime). This removes the magnitude/center confound of the earlier (1-s)*w + s*gamma version,
where w's magnitude (hence the effective bandwidth) and the center -b/w both drifted with s.

Why RMS: a vector splits canonically into magnitude ||w||_2 and direction w/||w||_2, so "hold
the scale, vary the pattern" is "hold ||w||_2 (= sqrt(W)*RMS), rotate the direction." RMS is also
the native norm of the least-squares readout and is robust to the sign zero-crossings that make
the (log-natural) geometric mean unusable here.

Axes: weight uniformness s in [0,1] (Xavier at 0, ideal all-equal-gamma at 1); gamma in
logspace(0.25, N), lambda = rms(w)*h = gamma*h = gamma*2/N. The (s=1, gamma~N/8) corner is exact
QI and must reach the fp64 floor -- the built-in sanity check. Single seed.

Grid: 1 seed x 4 targets x 4 widths (N=64,128,256,512) x K x K. Figure per target: 4x3 grid
(rows = widths): col 1 = s x lambda error heatmap, col 2 = error vs lambda (lines = s slices),
col 3 = error vs s (lines = gamma slices).

Usage:
    python experiments/expC05_geometry_interpolation/run_weights.py            # collect + plot
    python experiments/expC05_geometry_interpolation/run_weights.py --plot     # plot from saved data
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
                    uniform_geometry, solve_relL2, xavier_draw, center_order, get_target)

RESULTS_DIR = EXP_RESULTS / "weights"
OUT_DIR = RESULTS_DIR
DATA_PATH = OUT_DIR / "data.json"

# --- experiment grid ---
WIDTHS = [64, 128, 256, 512]
TARGETS = ["sine", "sine_8pi", "runge", "exp"]
SEED_FOLDERS = {"seed_1": 0}                      # single seed (folder -> base seed)
K = 21                                            # K x K interpolation grid
GAMMA_MIN = 0.25                                  # gamma_max = N per width
GLOBAL_LAMBDA_MAX = 2.0                           # lambda axis top (shared across rows)
WDEAD = 1e-12                                     # |w| below this = a sign zero-crossing (center undefined)


def rms(v):
    return float(np.sqrt(np.mean(np.asarray(v) ** 2)))


def xavier_weights(W, seed, N):
    """Exact Xavier inner weights, ordered to match the centers mode's neuron ordering (sorted by
    Xavier center -b/w, dead |w| parked by bias sign), then pinned one-to-one to the ascending
    uniform grid. With centers pinned by construction the assignment is cosmetic, but kept for
    reproducibility / comparability with the centers mode. Seeded per (folder seed, width)."""
    w, b = xavier_draw(W, seed, N)
    return w[center_order(w, b)]


def collect_data():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_tr = np.linspace(-1.0, 1.0, N_TRAIN)
    X_ev = np.linspace(-1.0, 1.0, N_EVAL)
    ss = np.linspace(0.0, 1.0, K)                         # weight-uniformness axis
    Y_tr = np.stack([get_target(t).fn_numpy(X_tr) for t in TARGETS], axis=1)   # [n_tr, 4]
    Y_ev = np.stack([get_target(t).fn_numpy(X_ev) for t in TARGETS], axis=1)   # [n_ev, 4]
    y_norms = np.linalg.norm(Y_ev, axis=0)               # [4]

    cells = []
    t0 = time.time()
    max_center_err = 0.0          # max |(-b/w) - c_i| over live neurons, all cells (should be ~0)
    max_rms_err = 0.0             # max |rms(w) - gamma| / gamma, all cells (should be ~0)
    total = len(SEED_FOLDERS) * len(WIDTHS)
    done = 0
    for folder, fseed in SEED_FOLDERS.items():
        for N in WIDTHS:
            c_uniform, h, halo = uniform_geometry(N)      # FIXED ideal centers
            W = c_uniform.size
            w_xavier = xavier_weights(W, fseed, N)
            w_hat = w_xavier / rms(w_xavier)              # unit-RMS Xavier direction
            gammas = np.logspace(np.log10(GAMMA_MIN), np.log10(N), K)
            lambdas = gammas * 2.0 / N

            err = np.zeros((len(TARGETS), K, K))          # [target, s_idx, gamma_idx]
            for si, s in enumerate(ss):
                u = (1.0 - s) * w_hat + s * 1.0           # pattern: Xavier -> all-ones
                u_unit = u / rms(u)                       # unit RMS -> rms(w) = gamma exactly
                for gi, g in enumerate(gammas):
                    w = g * u_unit                        # rms(w) = g
                    b = -w * c_uniform                    # center -b/w = c_i, pinned
                    # invariants (cheap; assert once on the first cell, track max thereafter)
                    live = np.abs(w) > WDEAD
                    if live.any():
                        ce = float(np.max(np.abs((-b[live] / w[live]) - c_uniform[live])))
                        max_center_err = max(max_center_err, ce)
                    max_rms_err = max(max_rms_err, abs(rms(w) - g) / g)
                    Phi_tr = np.tanh(w[None, :] * X_tr[:, None] + b[None, :])
                    Phi_ev = np.tanh(w[None, :] * X_ev[:, None] + b[None, :])
                    err[:, si, gi] = solve_relL2(Phi_tr, Phi_ev, Y_tr, Y_ev, y_norms)

            for tigt, target in enumerate(TARGETS):
                cells.append({
                    "folder": folder, "target": target, "N": N, "W": int(W),
                    "halo": int(halo), "seed": fseed,
                    "ss": ss.tolist(), "gammas": gammas.tolist(),
                    "lambdas": lambdas.tolist(), "err": err[tigt].tolist(),
                })
            done += 1
            best = {TARGETS[k]: err[k].min() for k in range(len(TARGETS))}
            print(f"[{done}/{total}] {folder} N={N:4d} (W={W}) | "
                  + " ".join(f"{k[:5]}={v:.1e}" for k, v in best.items())
                  + f" | {time.time()-t0:.0f}s")

    # --- invariant + sanity checks ---
    print(f"\n[invariant] max center drift  |(-b/w) - c_i| = {max_center_err:.2e}  (want ~0)")
    print(f"[invariant] max rel rms error  |rms(w)-gamma|/gamma = {max_rms_err:.2e}  (want ~0)")
    assert max_center_err < 1e-9, "centers drifted -- bias is not tracking the weight"
    assert max_rms_err < 1e-9, "bandwidth scale not held at gamma"
    # s=1 corner (exact QI) must reach the fp64 floor on the smooth targets
    s1 = K - 1
    for target in ("sine", "exp"):
        tc = next(c for c in cells if c["target"] == target and c["N"] == 256)
        floor = min(np.array(tc["err"])[s1, :])
        print(f"[sanity] {target} N=256 s=1 best-over-gamma rel L2 = {floor:.2e}  (want <1e-12)")
        assert floor < 1e-11, f"s=1 corner did not reach the floor for {target}"

    out = {
        "config": {
            "widths": WIDTHS, "targets": TARGETS, "seed_folders": SEED_FOLDERS,
            "K": K, "gamma_min": GAMMA_MIN, "halo_lambda": HALO_LAMBDA,
            "n_train": N_TRAIN, "n_eval": N_EVAL, "rcond": RCOND,
            "normalizer": "rms", "construction": "w=gamma*u/rms(u), u=(1-s)*what+s, b=-w*c (centers pinned)",
            "max_center_err": max_center_err, "max_rms_err": max_rms_err,
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
    fig.suptitle(f"expC05 weights -- {target} ({folder}) "
                 r"(Xavier direction $\to$ all-ones; centers pinned, $\mathrm{rms}(w)=\gamma$ fixed; "
                 r"lstsq, fp64, relative $L_2$)",
                 fontsize=15, y=0.999)

    for ri, N in enumerate(widths):
        c = _cell(data["cells"], folder, target, N)
        ss = np.array(c["ss"]); gammas = np.array(c["gammas"])
        lambdas = np.array(c["lambdas"]); err = np.array(c["err"])   # [s, gamma]
        emin = max(err.min(), 1e-16); emax = err.max()

        # --- col 1: K x K heatmap (x = weight uniformness, y = lambda log) ---
        ax = axes[ri][0]
        im = ax.pcolormesh(ss, lambdas, err.T, norm=LogNorm(vmin=emin, vmax=emax),
                           cmap="inferno", shading="auto")
        ax.set_yscale("log")
        ax.set_ylabel(f"N = {N}\n" + r"$\lambda = \mathrm{rms}(w)\,h$")
        ax.set_xlabel("weight uniformness $s$")
        if ri == 0:
            ax.set_title("error heatmap (rel $L_2$)")
        fig.colorbar(im, ax=ax, shrink=0.85, label="rel $L_2$")

        # --- col 2: error vs lambda, one line per weight-uniformness slice ---
        ax = axes[ri][1]
        cmap2, norm2 = plt.cm.viridis, Normalize(0.0, 1.0)
        for si, s in enumerate(ss):
            ax.plot(lambdas, err[si, :], "-", lw=1.1, color=cmap2(norm2(s)))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lam_lo * 0.8, lam_hi * 1.15)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel(r"$\lambda = \mathrm{rms}(w)\,h$"); ax.set_ylabel("rel $L_2$")
        if ri == 0:
            ax.set_title("error vs $\\lambda$ (lines = weight uniformness slices)")
        fig.colorbar(ScalarMappable(norm=norm2, cmap=cmap2), ax=ax,
                     shrink=0.85, label="weight uniformness $s$")

        # --- col 3: error vs weight uniformness, one line per gamma slice ---
        ax = axes[ri][2]
        cmap3, norm3 = plt.cm.plasma, LogNorm(gammas.min(), gammas.max())
        for gi, g in enumerate(gammas):
            ax.plot(ss, err[:, gi], "-", lw=1.1, color=cmap3(norm3(g)))
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel("weight uniformness $s$"); ax.set_ylabel("rel $L_2$")
        if ri == 0:
            ax.set_title(r"error vs weight uniformness (lines = $\gamma$ slices)")
        fig.colorbar(ScalarMappable(norm=norm3, cmap=cmap3), ax=ax,
                     shrink=0.85, label=r"$\gamma$")

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
    if "--plot" in sys.argv:
        if not DATA_PATH.exists():
            print(f"No data at {DATA_PATH}. Run without --plot first.")
            sys.exit(1)
        with open(DATA_PATH) as f:
            data = json.load(f)
    else:
        data = collect_data()
    plot_all(data)
