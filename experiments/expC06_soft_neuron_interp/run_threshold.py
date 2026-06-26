"""expC06 [threshold] -- threshold-protected weight interpolation (soft-neuron TEST).

Part 2 of expC06 (soft-neuron interpolation). The causal TEST half: it protects
the soft-weight tail during the Xavier -> uniform interpolation and watches the
error. Part 1 (run_histogram.py) is the VISUALIZATION (the magnitude distribution
collapsing). Both share expC05 weights' normalization/draw.

expC05 weights showed a hump: as the inner weights interpolate Xavier -> uniform
(all |w|=gamma), the lstsq error first DEGRADES before recovering to the floor at
the exact uniform endpoint. Hypothesis (Sam): the soft/low-bandwidth neurons at
init -- the ones near-linear over [-1,1], a spread of which span a low-degree
POLYNOMIAL basis -- do cheap, useful approximation work for smooth/convex targets
(exp, x^2, ...). Pulling them up to gamma destroys that basis -> the hump.

Test: interpolate as in expC05 weights, but PROTECT (freeze) the soft neurons.
For a magnitude threshold tau, neurons with |w_i(0)| < tau stay frozen at their
init magnitude for all s; the rest interpolate toward gamma. The non-protected
set is renormalized each s so mean|w| = gamma exactly (L1 ball; survivors stay
EXACTLY frozen). If protecting the soft set flattens the hump but a same-count
RANDOM ("placebo") protection does not, the soft neurons are the cause.

Key facts about the geometry (verified):
  * At s=0 weights are rescaled to mean gamma, so |w(0)| ~ U(0, 2*gamma),
    gamma = lambda/h = N/8 at lambda=0.25. Weights are O(gamma), NOT O(0.1).
  * A raw tau catches an ~N-independent small COUNT (near-linearity is a raw
    property); tau=0.125..1.5 catches ~1..12 neurons. Single-seed counts are
    noisy (0..5) for the small ones.
  * A relative tau = frac*gamma catches a stable FRACTION (frac/2 of W):
    0.0625g..1.0g -> ~3%..50%.
  * ALL thresholds use the SAME 6 seeds, aggregated by GEOMETRIC mean over seeds
    (robust to worst-seed outliers; the right average for a near-floor positive
    error). So every curve shares the s=0 point and is directly comparable.
    Per-seed curves are stored too (err_seeds) so re-aggregation needs no rerun.

Two center regimes (run separately): centers pinned UNIFORM (expC05 weights), and
centers pinned at the Xavier INIT positions c_i = -b_i/w_i. tanh only, fp64.

Layout: per (center regime, target) one 4x2 figure -- rows = N in {64,128,256,512},
cols = [threshold (protect smallest) | placebo (protect random)]. Each cell:
rel L2 vs interpolation s, with curves = the 4 constant + 6 relative thresholds,
plus a tau=0 (no-protection) baseline (the hump reference).

Reuse: expC05's common.py supplies the verified geometry/draw/ordering/solve
(uniform_geometry, xavier_draw, center_order, solve_relL2, grids, rcond). The
threshold-clip + non-protected renorm + placebo + init-center logic is new here.

Usage:
    python experiments/expC06_soft_neuron_interp/run_threshold.py --centers uniform   # run one
    python experiments/expC06_soft_neuron_interp/run_threshold.py --centers init      # then the other
    python experiments/expC06_soft_neuron_interp/run_threshold.py --centers uniform --plot  # replot
"""

import argparse
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
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expC05_geometry_interpolation"))
from common import (EXP_RESULTS, N_TRAIN, N_EVAL, uniform_geometry,  # noqa: E402
                    xavier_draw, center_order, solve_relL2)

BASE_DIR = REPO_ROOT / "results" / "checkpoint_C_geometry" / "expC06_soft_neuron_interp"

WIDTHS = [64, 128, 256, 512]
LAMBDA = 0.25
CONST_TAUS = [0.125, 0.25, 0.75, 1.5]          # raw, 6-seed mean
REL_FRACS = [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]  # x gamma, seed 0
N_SEEDS_CONST = 6
SEED0 = 0
WDEAD = 1e-8

# s grid: concentrated near 0 (hump onset is at small s), then out to 1.
S_GRID = np.unique(np.concatenate([
    np.linspace(0.0, 0.30, 16),
    np.linspace(0.30, 1.0, 11),
]))

# Targets (numpy fn only -- no QI construction here, just lstsq on the geometry).
TARGETS = {
    "exp": (lambda x: np.exp(x), r"$e^{x}$"),
    "x2": (lambda x: x ** 2, r"$x^2$"),
    "neg_cos": (lambda x: -np.cos(np.pi * x / 2.0), r"$-\cos(\pi x/2)$"),
    "sin2pi": (lambda x: np.sin(2.0 * np.pi * x), r"$\sin(2\pi x)$"),
}
TARGET_ORDER = ["exp", "x2", "neg_cos", "sin2pi"]


# --------------------------------------------------------------------------
# geometry per (N, seed, center regime)
# --------------------------------------------------------------------------

def draw_geometry(N, seed, center_mode):
    """Return (base, signs, centers, W, gamma) for the interpolation.

    base = |w_xav|/mean|w_xav| (mean 1), signs = sign(w_xav).
    UNIFORM: neurons ordered by Xavier center, pinned to the uniform grid.
    INIT:    centers c_i = -b_i/w_i (dead parked at +-1), no reorder.
    """
    c_uniform, h, halo = uniform_geometry(N)
    W = c_uniform.size
    gamma = LAMBDA / h
    w, b = xavier_draw(W, seed, N)
    if center_mode == "uniform":
        order = center_order(w, b)
        w = w[order]
        centers = c_uniform
    else:  # init centers
        safe = np.abs(w) >= WDEAD
        centers = np.empty(W)
        centers[safe] = -b[safe] / w[safe]
        centers[~safe] = np.where(b[~safe] >= 0, -1.0, 1.0)
    base = np.abs(w) / np.mean(np.abs(w))      # mean 1
    signs = np.sign(w)
    signs[signs == 0] = 1.0
    return base, signs, centers, W, gamma


def protected_mask(base, gamma, W, kind, val, seed=None):
    """Boolean frozen-mask. kind 'const': |w0|<val; 'rel': |w0|<val*gamma;
    'placebo': random k indices (k = count for the matching real threshold)."""
    w0 = gamma * base                          # |w_i(0)|
    if kind == "const":
        return w0 < val
    if kind == "rel":
        return w0 < val * gamma
    raise ValueError(kind)


def placebo_mask(W, k, seed, N, tag):
    """Random k-of-W frozen mask (deterministic per (seed,N,tag))."""
    m = np.zeros(W, dtype=bool)
    if k > 0:
        rng = np.random.default_rng([99, seed, N, int(round(tag * 1000)), k])
        m[rng.choice(W, size=int(k), replace=False)] = True
    return m


def signed_weights(base, signs, mask, s, gamma, W):
    """Frozen survivors at gamma*base; non-protected interp to gamma with the
    NON-PROTECTED set renormalized to hold mean|w| = gamma (survivors exact)."""
    mag = np.empty(W)
    mag[mask] = gamma * base[mask]
    Q = ~mask
    if Q.any():
        u = (1.0 - s) * base[Q] + s * 1.0
        target_sum_Q = gamma * W - gamma * base[mask].sum()
        denom = u.sum()
        kappa = target_sum_Q / denom if denom > 0 else 0.0
        mag[Q] = kappa * u
    return mag * signs


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------

def _err_curve(base, signs, centers, mask, gamma, W, X_tr, X_ev, Ytr, Yev, ynorm):
    """rel L2 (per target) vs s for one frozen-mask."""
    out = np.empty((len(S_GRID), Ytr.shape[1]))
    for i, s in enumerate(S_GRID):
        w = signed_weights(base, signs, mask, s, gamma, W)
        Phi_tr = np.tanh(w[None, :] * (X_tr[:, None] - centers[None, :]))
        Phi_ev = np.tanh(w[None, :] * (X_ev[:, None] - centers[None, :]))
        out[i] = solve_relL2(Phi_tr, Phi_ev, Ytr, Yev, ynorm)
    return out


def collect(center_mode):
    out_dir = BASE_DIR / center_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    X_tr = np.linspace(-1.0, 1.0, N_TRAIN)
    X_ev = np.linspace(-1.0, 1.0, N_EVAL)
    Ytr = np.stack([TARGETS[t][0](X_tr) for t in TARGET_ORDER], axis=1)
    Yev = np.stack([TARGETS[t][0](X_ev) for t in TARGET_ORDER], axis=1)
    ynorm = np.linalg.norm(Yev, axis=0)

    # threshold descriptors: (key, kind, val, seeds). All curves use the SAME 6
    # seeds so they share the s=0 point and are directly comparable.
    seeds6 = list(range(N_SEEDS_CONST))
    threshs = [("baseline", "const", 0.0, seeds6)]
    threshs += [(f"const_{v}", "const", v, seeds6) for v in CONST_TAUS]
    threshs += [(f"rel_{v}", "rel", v, seeds6) for v in REL_FRACS]

    cells = {}   # (N, key, 'thr'|'plc') -> {err: geomean[s,target], err_seeds, count}
    t0 = time.time()
    for N in WIDTHS:
        for key, kind, val, seeds in threshs:
            for col in ("thr", "plc"):
                if key == "baseline" and col == "plc":
                    continue                       # baseline has no placebo
                curves = []                         # per-seed [s, target]
                kk = []
                for seed in seeds:
                    base, signs, centers, W, gamma = draw_geometry(N, seed, center_mode)
                    if key == "baseline":
                        mask = np.zeros(W, dtype=bool)
                    else:
                        real = protected_mask(base, gamma, W, kind, val)
                        k = int(real.sum())
                        mask = real if col == "thr" else placebo_mask(W, k, seed, N, val)
                        kk.append(k)
                    curves.append(_err_curve(base, signs, centers, mask, gamma, W,
                                             X_tr, X_ev, Ytr, Yev, ynorm))
                curves = np.array(curves)           # [nseed, s, target]
                # GEOMETRIC mean over seeds (robust to worst-seed outliers; the
                # right average for a near-floor positive error). Stash per-seed
                # so future aggregation changes never need a rerun.
                geo = np.exp(np.mean(np.log(np.clip(curves, 1e-300, None)), axis=0))
                cells[f"{N}|{key}|{col}"] = {
                    "err": geo.tolist(),
                    "err_seeds": curves.tolist(),
                    "count": float(np.mean(kk)) if kk else 0.0,
                }
            print(f"  [{center_mode}] N={N} {key:12s} done ({time.time()-t0:.0f}s)")

    data = {
        "center_mode": center_mode, "lambda": LAMBDA, "widths": WIDTHS,
        "s_grid": S_GRID.tolist(), "targets": TARGET_ORDER,
        "const_taus": CONST_TAUS, "rel_fracs": REL_FRACS,
        "n_seeds_const": N_SEEDS_CONST, "cells": cells,
    }
    with open(out_dir / "data.json", "w") as f:
        json.dump(data, f)
    print(f"Saved {out_dir/'data.json'} ({time.time()-t0:.0f}s)")
    return data


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

def _curve_styles():
    """(key, label, color, lw, ls) for every curve, in draw order."""
    sty = [("baseline", r"$\tau{=}0$ (none)", "black", 1.6, (0, (4, 2)))]
    blues = plt.cm.winter(np.linspace(0.15, 0.95, len(CONST_TAUS)))
    for v, c in zip(CONST_TAUS, blues):
        sty.append((f"const_{v}", f"raw {v}", c, 1.3, "-"))
    reds = plt.cm.autumn(np.linspace(0.0, 0.85, len(REL_FRACS)))
    for v, c in zip(REL_FRACS, reds):
        sty.append((f"rel_{v}", f"{v}$\\gamma$", c, 1.3, "-"))
    return sty


def plot_regime(data):
    center_mode = data["center_mode"]
    out_dir = BASE_DIR / center_mode
    s = np.array(data["s_grid"]); cells = data["cells"]
    sty = _curve_styles()

    for ti, tkey in enumerate(data["targets"]):
        tlbl = TARGETS[tkey][1]
        fig, axes = plt.subplots(len(WIDTHS), 2, figsize=(13, 3.1 * len(WIDTHS)),
                                 squeeze=False, sharex=True)
        fig.suptitle(f"expC06 [threshold] weight interpolation [{center_mode} centers] -- "
                     f"{tlbl}  (tanh, lstsq fp64, rel $L_2$, $\\lambda$={data['lambda']}, "
                     f"6-seed geometric mean)", fontsize=13, y=0.998)
        for ri, N in enumerate(WIDTHS):
            for ci, col in enumerate(("thr", "plc")):
                ax = axes[ri][ci]
                for key, label, color, lw, ls in sty:
                    ck = f"{N}|{key}|{col}"
                    if ck not in cells:
                        ck = f"{N}|{key}|thr"            # baseline lives under thr
                        if ck not in cells:
                            continue
                    err = np.array(cells[ck]["err"])[:, ti]
                    ax.semilogy(s, err, ls=ls, color=color, lw=lw)
                ax.grid(True, which="both", alpha=0.25)
                if ci == 0:
                    ax.set_ylabel(f"N={N}\nrel $L_2$")
                if ri == len(WIDTHS) - 1:
                    ax.set_xlabel("interpolation $s$  (Xavier $\\to$ uniform)")
                if ri == 0:
                    ax.set_title("threshold (protect smallest)" if ci == 0
                                 else "placebo (protect random, same count)", fontsize=11)
        # shared legend on top
        handles = [plt.Line2D([], [], color=c, lw=lw, ls=ls)
                   for _k, _l, c, lw, ls in sty]
        labels = [l for _k, l, _c, _lw, _ls in sty]
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
                   ncol=6, fontsize=8, columnspacing=1.0, handlelength=1.6)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.06,
                            hspace=0.18, wspace=0.16)
        out = out_dir / f"interp_{tkey}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", choices=["uniform", "init", "both"], default="uniform")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    modes = ["uniform", "init"] if args.centers == "both" else [args.centers]
    for mode in modes:
        dpath = BASE_DIR / mode / "data.json"
        if args.plot:
            if not dpath.exists():
                print(f"No data at {dpath}; run without --plot first.")
                continue
            with open(dpath) as f:
                data = json.load(f)
        else:
            data = collect(mode)
        plot_regime(data)
