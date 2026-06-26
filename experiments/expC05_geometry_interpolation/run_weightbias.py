"""expC05 weight+bias mode -- interpolate BOTH the weight pattern and the centers, no confound.

Completes the centers / weights pair. Two independent axes, at a FIXED bandwidth (lambda=0.25):

  - weight uniformness s in [0,1] -- the weight pattern, on the L1 diamond (same as the weights mode):
        w_hat = w_xav / mean|w_xav|,  sigma = sign(w_xav)
        u(s)  = (1-s)*w_hat + s*sigma          # toward the signed-ones octant center
        w(s)  = gamma * u(s) / mean|u(s)|       # mean|w| = gamma, stays in-orthant (no crossing)
  - center uniformness t in [0,1] -- the center POSITIONS:
        c_rand = sorted uniform random over the full grid+halo span
        c(t)   = (1-t)*c_rand + t*c_uniform     # random-uniform layout -> the uniform QI grid
  - bias is DERIVED, never interpolated raw:  b(s,t) = -w(s) * c(t)   (so center -b/w = c(t) exactly)

Feature: tanh(w(s)*x + b(s,t)) = tanh(w(s)*(x - c(t))).

gamma is held CONSTANT at lambda* = 0.25 (gamma = lambda*/h = N/8) -- not swept, not selected. So the
(s,t) grid is one bandwidth slice. Corners: (1,1) = exact QI construction (-> fp64 floor);
(0,0) = the Xavier weight *pattern* at gamma-scale + random-uniform centers (a controlled random
start, NOT the literal small-magnitude Xavier init); (1,0) = uniform weights / random centers;
(0,1) = Xavier-pattern weights / uniform centers.

Consistency check (validate post-run): along t=1 the centers are exactly uniform regardless of s, so
the t=1 edge reduces to the weights mode (uniform centers + w(s)); its error-vs-s should equal the
weights experiment's tanh run at lambda=0.25.

L1 / mean-abs normalization (hold the bandwidth budget); centers and target share the same span (t
varies uniformity only); invariants asserted: mean|w|=gamma, center -b/w=c(t), no sign crossing.

tanh only, single seed. Grid: 4 targets x 4 widths (N=64,128,256,512) x K x K.
Figures: weightbias/interp_<target>.png -- 4x3 grid (rows=widths): col1 = (s,t) error heatmap,
col2 = error vs s (lines = t slices), col3 = error vs t (lines = s slices).

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
SEED = 0
K = 21
LAMBDA_STAR = 0.25                                # bandwidth held fixed here
WDEAD = 1e-12


def mabs(v):
    return float(np.mean(np.abs(np.asarray(v))))


def collect_data():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_tr = np.linspace(-1.0, 1.0, N_TRAIN)
    X_ev = np.linspace(-1.0, 1.0, N_EVAL)
    ss = np.linspace(0.0, 1.0, K)                        # weight-uniformness axis
    ts = np.linspace(0.0, 1.0, K)                        # center-uniformness axis
    Y_tr = np.stack([get_target(t).fn_numpy(X_tr) for t in TARGETS], axis=1)
    Y_ev = np.stack([get_target(t).fn_numpy(X_ev) for t in TARGETS], axis=1)
    y_norms = np.linalg.norm(Y_ev, axis=0)

    cells = []
    t0 = time.time()
    max_center_err = 0.0; max_budget_err = 0.0; n_signflip = 0
    total = len(WIDTHS) * len(TARGETS); done = 0
    for N in WIDTHS:
        c_uniform, h, halo = uniform_geometry(N)
        W = c_uniform.size
        span = (float(c_uniform[0]), float(c_uniform[-1]))
        g = LAMBDA_STAR / h                              # = N/8

        # weight pattern: Xavier draw, ordered (pins neuron i to grid center i, matching weights mode)
        w_x, b_x = xavier_draw(W, SEED, N)
        order = center_order(w_x, b_x)
        w_x = w_x[order]
        what = w_x / mabs(w_x)
        sigma = np.sign(w_x)
        # center anchor: random uniform over the full span (sorted), independent of the weights
        rng = np.random.default_rng([SEED, N])
        c_rand = np.sort(rng.uniform(span[0], span[1], size=W)).astype(np.float64)

        # precompute w(s) on the L1 face
        Wmat = np.zeros((K, W))
        for si, s in enumerate(ss):
            u = (1.0 - s) * what + s * sigma
            Wmat[si] = g * u / mabs(u)                   # mean|w| = g

        for ti, target in enumerate(TARGETS):
            yt = Y_tr[:, ti:ti + 1]; ye = Y_ev[:, ti:ti + 1]; yn = y_norms[ti:ti + 1]
            err = np.zeros((K, K))                        # [s_idx, t_idx]
            for si in range(K):
                w = Wmat[si]
                for tj, t in enumerate(ts):
                    c = (1.0 - t) * c_rand + t * c_uniform
                    b = -w * c
                    live = np.abs(w) > WDEAD
                    if live.any():
                        max_center_err = max(max_center_err, float(np.max(np.abs((-b[live]/w[live]) - c[live]))))
                        n_signflip += int(np.sum((np.sign(w[live]) != sigma[live]) & (sigma[live] != 0)))
                    max_budget_err = max(max_budget_err, abs(mabs(w) - g) / g)
                    Phi_tr = np.tanh(w[None, :] * X_tr[:, None] + b[None, :])
                    Phi_ev = np.tanh(w[None, :] * X_ev[:, None] + b[None, :])
                    err[si, tj] = solve_relL2(Phi_tr, Phi_ev, yt, ye, yn)[0]
            cells.append({
                "target": target, "N": N, "W": int(W), "halo": int(halo), "seed": SEED,
                "lambda": LAMBDA_STAR, "gamma": float(g), "span": list(span),
                "ss": ss.tolist(), "ts": ts.tolist(), "err": err.tolist(),
            })
            done += 1
            print(f"[{done}/{total}] N={N:4d} {target:8s} | (0,0)={err[0,0]:.1e} "
                  f"(1,1)={err[-1,-1]:.1e} (1,0)={err[-1,0]:.1e} (0,1)={err[0,-1]:.1e} "
                  f"min={err.min():.1e} | {time.time()-t0:.0f}s")

    print(f"\n[invariant] max center drift |(-b/w)-c| = {max_center_err:.2e}  (want ~0)")
    print(f"[invariant] max rel budget err |mean|w|-g|/g = {max_budget_err:.2e}  (want ~0)")
    print(f"[invariant] sign flips / origin crossings = {n_signflip}  (want 0)")
    assert max_center_err < 1e-9 and max_budget_err < 1e-9 and n_signflip == 0
    for c in cells:
        if c["target"] == "sine" and c["N"] == 256:
            corner = float(np.array(c["err"])[-1, -1])
            print(f"[sanity] sine N=256 (1,1) corner rel L2 = {corner:.2e}  (want <1e-11)")
            assert corner < 1e-11

    out = {
        "config": {
            "widths": WIDTHS, "targets": TARGETS, "seed": SEED, "K": K,
            "lambda_star": LAMBDA_STAR, "halo_lambda": HALO_LAMBDA,
            "n_train": N_TRAIN, "n_eval": N_EVAL, "rcond": RCOND, "normalizer": "L1 (mean-abs)",
            "construction": "w: L1 face Xavier->sign(Xavier)*1; centers: uniform-random-over-span->uniform grid; "
                            "b=-w*c (derived); gamma fixed at lambda*=0.25",
            "max_center_err": max_center_err, "max_budget_err": max_budget_err, "sign_flips": n_signflip,
        },
        "cells": cells,
    }
    with open(DATA_PATH, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {DATA_PATH} ({time.time()-t0:.0f}s total)")
    return out


# ----------------------------- plotting -----------------------------

def _cell(cells, target, N):
    return next(c for c in cells if c["target"] == target and c["N"] == N)


def plot_target(data, target):
    cfg = data["config"]
    widths = cfg["widths"]
    nrows = len(widths)
    fig, axes = plt.subplots(nrows, 3, figsize=(19, 4.6 * nrows))
    fig.suptitle(f"expC05 weight+bias -- {target} "
                 r"(weight: Xavier $\to \sigma\mathbf{1}$ on L1 face; centers: random$\to$uniform; "
                 r"$b=-w c$; $\lambda=0.25$ fixed; lstsq fp64, rel $L_2$)",
                 fontsize=14, y=0.999)

    for ri, N in enumerate(widths):
        c = _cell(data["cells"], target, N)
        ss = np.array(c["ss"]); ts = np.array(c["ts"]); err = np.array(c["err"])   # [s, t]
        emin = max(err.min(), 1e-16); emax = err.max()

        ax = axes[ri][0]
        im = ax.pcolormesh(ss, ts, err.T, norm=LogNorm(vmin=emin, vmax=emax),
                           cmap="inferno", shading="auto")
        ax.set_ylabel(f"N = {N}\ncenter uniformness $t$")
        ax.set_xlabel("weight uniformness $s$")
        if ri == 0:
            ax.set_title("error heatmap (rel $L_2$)")
        fig.colorbar(im, ax=ax, shrink=0.85, label="rel $L_2$")

        ax = axes[ri][1]
        cmap2, norm2 = plt.cm.viridis, Normalize(0.0, 1.0)
        for tj, t in enumerate(ts):
            ax.plot(ss, err[:, tj], "-", lw=1.1, color=cmap2(norm2(t)))
        ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel("weight uniformness $s$"); ax.set_ylabel("rel $L_2$")
        if ri == 0:
            ax.set_title("error vs weight $s$ (lines = center $t$ slices)")
        fig.colorbar(ScalarMappable(norm=norm2, cmap=cmap2), ax=ax,
                     shrink=0.85, label="center uniformness $t$")

        ax = axes[ri][2]
        cmap3, norm3 = plt.cm.plasma, Normalize(0.0, 1.0)
        for si, s in enumerate(ss):
            ax.plot(ts, err[si, :], "-", lw=1.1, color=cmap3(norm3(s)))
        ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel("center uniformness $t$"); ax.set_ylabel("rel $L_2$")
        if ri == 0:
            ax.set_title("error vs center $t$ (lines = weight $s$ slices)")
        fig.colorbar(ScalarMappable(norm=norm3, cmap=cmap3), ax=ax,
                     shrink=0.85, label="weight uniformness $s$")

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"interp_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_all(data):
    for target in data["config"]["targets"]:
        plot_target(data, target)


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
