"""expC05 weights mode -- weight interpolation along an L1-ball face, no origin crossing.

Fix the centers at the ideal uniform QI grid and the bandwidth budget (L1 / mean-abs) at gamma,
and interpolate the inner weight vector from the Xavier draw to the CENTER of its own L1 octant
face -- the uniform-magnitude vector with the SAME signs as the Xavier init. Because both endpoints
lie in the same orthant and the L1 face is flat, the straight-line interpolation stays on the L1
sphere (mean|w| = gamma held exactly) and NEVER crosses the origin. So `s` is a clean
"per-neuron bandwidth-magnitude uniformity" axis with no sign-crossing artifact.

Three runs (folders):
  - tanh                : base = Xavier (signed),   target = sign(Xavier)*1   (octant center), tanh
  - gelu                : base = Xavier (signed),   target = sign(Xavier)*1   (octant center), gelu
  - gelu_positive_init  : base = |Xavier|,          target = +1               (positive octant), gelu
(tanh's positive-init case is omitted: tanh is odd, so the octant and positive versions differ only
by per-column sign, which the readout absorbs -- they are identical. gelu is not odd, so its sign is
a real degree of freedom and the two gelu runs genuinely differ.)
weights/crossing_0/ is the frozen earlier RMS run that DID cross the origin (kept for reference).

Construction (per neuron i, interpolation s in [0,1], swept bandwidth gamma):
    base, target  per run (see above), with base L1-normalized: mean|base| = 1, mean|target| = 1
    u(s) = (1 - s) * base + s * target        # stays in one orthant -> never crosses 0
    w(s) = gamma * u(s) / mean|u(s)|           # mean|w| = gamma for all s (mean|u| = 1 already)
    b(s) = -w(s) * c_uniform                   # center -b/w = c_i, pinned
    Phi  = act(w(s)[None,:]*x[:,None] + b(s)[None,:])

Why L1 (mean-abs): the per-neuron dimensionless bandwidth is lambda_i = |w_i| h, so sum|w_i| is the
total bandwidth budget; mean-abs holds it fixed. (RMS would hand the uniform endpoint ~15% more
budget, since the uniform vector maximizes L1 at fixed L2.) The L1 face is flat, so the path needs
no projection and stays on the budget sphere.

Invariants asserted at runtime: mean|w| = gamma; center -b/w = c_i (live neurons); and NO sign flip
/ origin crossing (every w_i keeps its target-octant sign). The (s=1, gamma~N/8) corner: for tanh
and gelu_positive_init it is the uniform ideal (span = QI for tanh; all-forward ridges for gelu) and
should reach the activation's floor; for gelu (octant) s=1 is the mixed-sign uniform vector, which is
a valid but not necessarily optimal gelu basis -- its floor may sit higher (expected, not a bug).

Grid: 3 runs x 4 targets x 4 widths (N=64,128,256,512) x K x K, single seed.

Usage:
    python experiments/expC05_geometry_interpolation/run_weights.py            # collect + plot
    python experiments/expC05_geometry_interpolation/run_weights.py --plot     # plot from saved data
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import erf
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
TANH = np.tanh
GELU = lambda z: 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
# (folder, activation, init_mode): init_mode "octant" = signed Xavier -> sign(Xavier)*1;
#                                            "positive" = |Xavier| -> +1
RUNS = [("tanh", TANH, "octant"),
        ("gelu", GELU, "octant"),
        ("gelu_positive_init", GELU, "positive")]
SANITY = {"tanh": 1e-11, "gelu_positive_init": 1e-7, "gelu": None}   # None = print only
SEED = 0
K = 21
GAMMA_MIN = 0.25
GLOBAL_LAMBDA_MAX = 2.0
WDEAD = 1e-12


def mabs(v):
    return float(np.mean(np.abs(np.asarray(v))))


def xavier_weights(W, seed, N):
    """Exact Xavier inner weights, ordered to match the centers mode (sorted by Xavier center
    -b/w, dead |w| parked by bias sign), pinned to the ascending uniform grid. Seeded per (seed, N)."""
    w, b = xavier_draw(W, seed, N)
    return w[center_order(w, b)]


def collect_data():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_tr = np.linspace(-1.0, 1.0, N_TRAIN)
    X_ev = np.linspace(-1.0, 1.0, N_EVAL)
    ss = np.linspace(0.0, 1.0, K)
    Y_tr = np.stack([get_target(t).fn_numpy(X_tr) for t in TARGETS], axis=1)
    Y_ev = np.stack([get_target(t).fn_numpy(X_ev) for t in TARGETS], axis=1)
    y_norms = np.linalg.norm(Y_ev, axis=0)

    cells = []
    t0 = time.time()
    max_center_err = 0.0; max_budget_err = 0.0; n_signflip = 0
    total = len(RUNS) * len(WIDTHS); done = 0
    for run, act, mode in RUNS:
        for N in WIDTHS:
            c_uniform, h, halo = uniform_geometry(N)
            W = c_uniform.size
            wx = xavier_weights(W, SEED, N)
            if mode == "octant":
                base = wx / mabs(wx)                 # signed, L1-normalized
                target = np.sign(wx)                 # octant center: +-1 with Xavier signs
            else:                                    # "positive"
                base = np.abs(wx) / mabs(wx)         # positive magnitudes, L1-normalized
                target = np.ones(W)                  # all +1
            tsign = np.sign(target)
            gammas = np.logspace(np.log10(GAMMA_MIN), np.log10(N), K)
            lambdas = gammas * 2.0 / N

            err = np.zeros((len(TARGETS), K, K))
            for si, s in enumerate(ss):
                u = (1.0 - s) * base + s * target    # stays in the target octant
                u_unit = u / mabs(u)                 # mean|w| = gamma
                for gi, g in enumerate(gammas):
                    w = g * u_unit
                    b = -w * c_uniform
                    live = np.abs(w) > WDEAD
                    if live.any():
                        max_center_err = max(max_center_err,
                                             float(np.max(np.abs((-b[live] / w[live]) - c_uniform[live]))))
                        # crossing check: any live neuron whose sign differs from its target octant
                        n_signflip += int(np.sum((np.sign(w[live]) != tsign[live]) & (tsign[live] != 0)))
                    max_budget_err = max(max_budget_err, abs(mabs(w) - g) / g)
                    Phi_tr = act(w[None, :] * X_tr[:, None] + b[None, :])
                    Phi_ev = act(w[None, :] * X_ev[:, None] + b[None, :])
                    err[:, si, gi] = solve_relL2(Phi_tr, Phi_ev, Y_tr, Y_ev, y_norms)

            for tigt, target_name in enumerate(TARGETS):
                cells.append({
                    "folder": run, "activation": "tanh" if act is TANH else "gelu",
                    "init_mode": mode, "target": target_name, "N": N, "W": int(W),
                    "halo": int(halo), "seed": SEED,
                    "ss": ss.tolist(), "gammas": gammas.tolist(),
                    "lambdas": lambdas.tolist(), "err": err[tigt].tolist(),
                })
            done += 1
            best = {TARGETS[k]: err[k].min() for k in range(len(TARGETS))}
            print(f"[{done}/{total}] {run:18s} N={N:4d} (W={W}) | "
                  + " ".join(f"{k[:5]}={v:.1e}" for k, v in best.items())
                  + f" | {time.time()-t0:.0f}s")

    # --- invariants + sanity ---
    print(f"\n[invariant] max center drift |(-b/w)-c_i| = {max_center_err:.2e}  (want ~0)")
    print(f"[invariant] max rel budget err |mean|w|-g|/g = {max_budget_err:.2e}  (want ~0)")
    print(f"[invariant] sign flips / origin crossings = {n_signflip}  (want 0)")
    assert max_center_err < 1e-9, "centers drifted"
    assert max_budget_err < 1e-9, "budget not held at gamma"
    assert n_signflip == 0, "a weight crossed the origin -- interpolation left its octant"
    s1 = K - 1
    for run, act, mode in RUNS:
        tc = next(c for c in cells if c["folder"] == run and c["target"] == "sine" and c["N"] == 256)
        floor = float(min(np.array(tc["err"])[s1, :]))
        thr = SANITY[run]
        tag = f"(want <{thr:.0e})" if thr else "(print only)"
        print(f"[sanity] {run:18s} sine N=256 s=1 best-over-gamma rel L2 = {floor:.2e}  {tag}")
        if thr:
            assert floor < thr, f"s=1 corner did not reach the floor for {run}"

    out = {
        "config": {
            "widths": WIDTHS, "targets": TARGETS,
            "runs": [r[0] for r in RUNS], "run_modes": {r[0]: r[2] for r in RUNS},
            "seed": SEED, "K": K, "gamma_min": GAMMA_MIN, "halo_lambda": HALO_LAMBDA,
            "n_train": N_TRAIN, "n_eval": N_EVAL, "rcond": RCOND, "normalizer": "L1 (mean-abs)",
            "construction": "u=(1-s)*base+s*target along an L1 octant face (no origin crossing); "
                            "octant target=sign(Xavier)*1, positive target=+1; w=gamma*u/mean|u|, b=-w*c",
            "max_center_err": max_center_err, "max_budget_err": max_budget_err, "sign_flips": n_signflip,
            "global_lambda": [GAMMA_MIN * 2.0 / max(WIDTHS), GLOBAL_LAMBDA_MAX],
            "note": "weights/crossing_0/ is the frozen earlier RMS run that crossed the origin.",
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
    widths = cfg["widths"]; lam_lo, lam_hi = cfg["global_lambda"]
    mode = cfg["run_modes"][folder]
    tgt_lbl = (r"$\mathrm{sign}(w_{\rm xav})\cdot\mathbf{1}$ (octant center)" if mode == "octant"
               else r"$+\mathbf{1}$ (positive)")
    init_lbl = "Xavier" if mode == "octant" else "$|$Xavier$|$"
    nrows = len(widths)
    fig, axes = plt.subplots(nrows, 3, figsize=(19, 4.6 * nrows))
    fig.suptitle(f"expC05 weights [{folder}] -- {target} "
                 f"({init_lbl} $\\to$ {tgt_lbl}; centers pinned, mean$|w|=\\gamma$; lstsq fp64, rel $L_2$)",
                 fontsize=14, y=0.999)

    for ri, N in enumerate(widths):
        c = _cell(data["cells"], folder, target, N)
        ss = np.array(c["ss"]); gammas = np.array(c["gammas"])
        lambdas = np.array(c["lambdas"]); err = np.array(c["err"])
        emin = max(err.min(), 1e-16); emax = err.max()

        ax = axes[ri][0]
        im = ax.pcolormesh(ss, lambdas, err.T, norm=LogNorm(vmin=emin, vmax=emax),
                           cmap="inferno", shading="auto")
        ax.set_yscale("log")
        ax.set_ylabel(f"N = {N}\n" + r"$\lambda = \mathrm{mean}|w|\,h$")
        ax.set_xlabel("weight uniformness $s$")
        if ri == 0:
            ax.set_title("error heatmap (rel $L_2$)")
        fig.colorbar(im, ax=ax, shrink=0.85, label="rel $L_2$")

        ax = axes[ri][1]
        cmap2, norm2 = plt.cm.viridis, Normalize(0.0, 1.0)
        for si, s in enumerate(ss):
            ax.plot(lambdas, err[si, :], "-", lw=1.1, color=cmap2(norm2(s)))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lam_lo * 0.8, lam_hi * 1.15)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel(r"$\lambda = \mathrm{mean}|w|\,h$"); ax.set_ylabel("rel $L_2$")
        if ri == 0:
            ax.set_title("error vs $\\lambda$ (lines = weight uniformness slices)")
        fig.colorbar(ScalarMappable(norm=norm2, cmap=cmap2), ax=ax,
                     shrink=0.85, label="weight uniformness $s$")

        ax = axes[ri][2]
        # color by lambda on the GLOBAL range, so this lines up with cols 1 & 2 (which are in lambda)
        cmap3, norm3 = plt.cm.plasma, LogNorm(lam_lo, lam_hi)
        for gi in range(len(gammas)):
            ax.plot(ss, err[:, gi], "-", lw=1.1, color=cmap3(norm3(lambdas[gi])))
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel("weight uniformness $s$"); ax.set_ylabel("rel $L_2$")
        if ri == 0:
            ax.set_title(r"error vs weight uniformness (lines = $\lambda$ slices)")
        fig.colorbar(ScalarMappable(norm=norm3, cmap=cmap3), ax=ax,
                     shrink=0.85, label=r"$\lambda = \mathrm{mean}|w|\,h$")

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_dir = OUT_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"interp_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_all(data):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for folder in data["config"]["runs"]:
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
