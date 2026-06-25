"""Experiment 7 -- 1D center-geometry comparison (lstsq readout, tanh).

Question: in 1D, the uniform QI grid + least-squares readout reaches machine
precision. How much of that depends on the *uniform* placement of the tanh
centers? We compare four center geometries head-to-head, all solved by the same
fp64 least-squares readout, all sharing the same gamma = lambda/h at each swept
lambda, on the same target (sine, which the uniform geometry fits to ~1e-13).

Four geometries (each places W centers over a common span):
  uniform   -- equispaced (the QI grid baseline)
  random    -- uniform random over the span
  clustered -- two-stage pseudo-clustered (uniform meta-centers + Gaussian draws)
  trained   -- centers x0 = -b/w extracted from a trained width-W tanh net;
               everything else (trained gamma, readout) is thrown away.

Apples-to-apples: for each base grid size N we take the uniform QI geometry
(grid + halo) to fix W (total width), the center span, and h = 2/N. The other
three geometries place exactly W centers over the same span, and every geometry
uses gamma = lambda/h at each swept lambda. Only the *placement* differs.

Headline figure error_vs_width.png: x = width W, y = error, two panels
(left relative L2, right L_inf), one curve per geometry (best over the lambda
sweep, mean over seeds with a min/max band for the stochastic geometries).

The readout solve is isolated in solve_geometry() so an mpmath path is a
one-line swap (set PRECISION = "mpmath").

Usage:
    python experiments/expC04_center_geometry/run.py            # collect + plot
    python experiments/expC04_center_geometry/run.py --plot     # plot from saved data
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

torch.set_default_dtype(torch.float64)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.center_geometry import (
    clustered_centers,
    random_centers,
    regular_clustered_centers,
    trained_centers,
    uniform_centers,
)
from src.construction.qi_mpmath import default_halo
from src.construction.readout import build_phi, solve_readout_with_bias
from src.data.targets import get_target

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_C_geometry" / "expC04_center_geometry"
DATA_PATH = RESULTS_DIR / "data.json"

# --- experiment grid ---
TARGET = "sine"
BASE_N = [32, 64, 128, 256]        # uniform grid sizes -> set W, span, h
HALO_LAMBDA = 0.25                  # reference lambda for the (structural) halo size
LAMBDA_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00]
SEEDS = [0, 1, 2]                   # for the stochastic geometries
N_TRAIN = 1024
N_EVAL = 4096
PRECISION = "fp64"                  # fp64 readout solve (mpmath: future swap)

GEOMETRIES = ["uniform", "random", "clustered", "trained", "reg_clustered"]
GEOM_COLORS = {
    "uniform": "#000000", "random": "#1f77b4", "clustered": "#2ca02c",
    "trained": "#d62728", "reg_clustered": "#9467bd",
}
# Deterministic geometries run a single seed (no seed dependence).
DETERMINISTIC = {"uniform", "reg_clustered"}

# --- reg_clustered (regular grid broken into geometric clusters) ---
CLUSTER_SIZE = 8        # points per cluster
CLUSTER_RATIO = 0.75    # geometric compression per cluster (1 = uniform)

# --- NN training (for the "trained" geometry) ---
NN_STEPS = 4000
NN_LR = 1e-3


def uniform_geometry(N):
    """The QI grid + halo: fixes width W, span, and h = 2/N for this N."""
    h = 2.0 / N
    halo = default_halo(N, lambda_star=HALO_LAMBDA)
    n_idx = np.arange(-halo, N + halo + 1)
    centers = -1.0 + n_idx.astype(np.float64) * h
    span = (float(centers[0]), float(centers[-1]))
    W = centers.size
    return W, span, h


def train_tanh_net(W, x_train, y_train, seed):
    """Train a width-W single-hidden-layer tanh net; return (w, b) of the
    inner layer (per-neuron input weight and bias)."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(1, W), torch.nn.Tanh(), torch.nn.Linear(W, 1)
    ).double()
    xt = torch.tensor(x_train).reshape(-1, 1)
    yt = torch.tensor(y_train).reshape(-1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=NN_LR)
    for _ in range(NN_STEPS):
        opt.zero_grad(set_to_none=True)
        loss = ((net(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    inner = net[0]
    w = inner.weight.detach().cpu().numpy().ravel()
    b = inner.bias.detach().cpu().numpy().ravel()
    return w, b


def make_centers(geometry, W, span, seed, x_train, y_train):
    if geometry == "uniform":
        return uniform_centers(W, span)
    if geometry == "random":
        return random_centers(W, span, seed=seed)
    if geometry == "clustered":
        return clustered_centers(W, span, seed=seed)
    if geometry == "trained":
        w, b = train_tanh_net(W, x_train, y_train, seed)
        return trained_centers(w, b, span)
    if geometry == "reg_clustered":
        return regular_clustered_centers(W, span, CLUSTER_SIZE, CLUSTER_RATIO)
    raise ValueError(geometry)


def solve_geometry(centers, gamma, x_train, y_train, x_eval, y_eval, y_norm):
    """Build Phi on the given centers, lstsq readout. Returns (linf, rel_l2,
    phi_cond) where phi_cond is cond of the augmented [Phi, 1] train matrix."""
    W = centers.size
    gamma_vec = np.full(W, gamma)
    Phi_tr = build_phi(x_train, gamma_vec, centers)
    Phi_ev = build_phi(x_eval, gamma_vec, centers)
    v, b, info = solve_readout_with_bias(Phi_tr, y_train, method="lstsq")
    resid = Phi_ev @ v + b - y_eval
    s = np.linalg.svd(np.hstack([Phi_tr, np.ones((x_train.size, 1))]),
                      compute_uv=False)
    phi_cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    return (float(np.abs(resid).max()),
            float(np.linalg.norm(resid) / y_norm), phi_cond)


def collect_data():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = get_target(TARGET)
    x_train = np.linspace(-1, 1, N_TRAIN)
    x_eval = np.linspace(-1, 1, N_EVAL)
    y_train = target.fn_numpy(x_train)
    y_eval = target.fn_numpy(x_eval)
    y_norm = float(np.linalg.norm(y_eval))

    rows = []
    geo_centers = {g: {} for g in GEOMETRIES}   # seed-0 centers for the number line
    t0 = time.time()
    for N in BASE_N:
        W, span, h = uniform_geometry(N)
        for geometry in GEOMETRIES:
            seeds = [0] if geometry in DETERMINISTIC else SEEDS
            for seed in seeds:
                centers = make_centers(geometry, W, span, seed, x_train, y_train)
                if seed == seeds[0]:
                    geo_centers[geometry][str(N)] = centers.tolist()
                for lam in LAMBDA_GRID:
                    gamma = lam / h
                    linf, rel_l2, phi_cond = solve_geometry(
                        centers, gamma, x_train, y_train, x_eval, y_eval, y_norm)
                    rows.append({
                        "geometry": geometry, "N": N, "W": W, "seed": seed,
                        "lambda": lam, "gamma": gamma, "h": h,
                        "linf": linf, "rel_l2": rel_l2, "phi_cond": phi_cond,
                    })
            best = min(r["linf"] for r in rows
                       if r["geometry"] == geometry and r["N"] == N)
            print(f"[{time.time()-t0:5.0f}s] N={N:4d} W={W:4d} {geometry:9s} "
                  f"best Linf={best:.2e}")

    out = {
        "config": {
            "target": TARGET, "base_N": BASE_N, "halo_lambda": HALO_LAMBDA,
            "lambda_grid": LAMBDA_GRID, "seeds": SEEDS, "n_train": N_TRAIN,
            "n_eval": N_EVAL, "precision": PRECISION, "geometries": GEOMETRIES,
            "nn_steps": NN_STEPS, "nn_lr": NN_LR,
            "cluster_size": CLUSTER_SIZE, "cluster_ratio": CLUSTER_RATIO,
        },
        "rows": rows,
        "centers": geo_centers,
    }
    with open(DATA_PATH, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {DATA_PATH} ({time.time()-t0:.0f}s total)")
    return out


# ----------------------------- plotting -----------------------------

def _best_over_lambda(rows, geometry, N):
    """Per-seed best (min over lambda) for both metrics -> arrays over seeds."""
    seeds = sorted({r["seed"] for r in rows
                    if r["geometry"] == geometry and r["N"] == N})
    linf, rl2 = [], []
    for s in seeds:
        recs = [r for r in rows if r["geometry"] == geometry
                and r["N"] == N and r["seed"] == s]
        linf.append(min(r["linf"] for r in recs))
        rl2.append(min(r["rel_l2"] for r in recs))
    return np.array(rl2), np.array(linf)


def plot_error_vs_width(data):
    cfg = data["config"]
    rows = data["rows"]
    Ws = [uniform_geometry(N)[0] for N in cfg["base_N"]]
    metrics = [(0, r"Best relative $L_2$"), (1, r"Best $L_\infty$")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(f"Exp07: center-geometry comparison vs width "
                 f"(target={cfg['target']}, lstsq, {cfg['precision']})", fontsize=14)
    for mi, mlabel in metrics:
        ax = axes[mi]
        for geometry in cfg["geometries"]:
            means, los, his = [], [], []
            for N in cfg["base_N"]:
                rl2, linf = _best_over_lambda(rows, geometry, N)
                arr = (rl2, linf)[mi]
                means.append(arr.mean()); los.append(arr.min()); his.append(arr.max())
            ax.fill_between(Ws, los, his, color=GEOM_COLORS[geometry], alpha=0.15)
            ax.loglog(Ws, means, "-o", color=GEOM_COLORS[geometry],
                      markersize=5, label=geometry)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel("width $W$ (total centers)")
        ax.set_ylabel(mlabel); ax.set_title(mlabel)
        ax.set_xticks(Ws); ax.set_xticklabels(Ws)
    axes[0].legend(fontsize=10)
    plt.tight_layout()
    out = RESULTS_DIR / "error_vs_width.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_centers_numberline(data):
    """One panel per geometry; within a panel, the centers for each width are
    drawn as ticks on stacked number lines, least dense (small W) at top."""
    cfg = data["config"]
    centers = data["centers"]
    geometries = cfg["geometries"]
    Ns = cfg["base_N"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    fig.suptitle("Exp07: center placement on the number line (seed 0)", fontsize=14)
    axes = axes.ravel()
    for gi, geometry in enumerate(geometries):
        ax = axes[gi]
        for ri, N in enumerate(Ns):                      # top = smallest W
            c = np.array(centers[geometry][str(N)])
            y = len(Ns) - 1 - ri                         # stack: small N on top
            ax.plot(c, np.full_like(c, y), "|", color=GEOM_COLORS[geometry],
                    markersize=14, markeredgewidth=0.6)
            ax.text(c.min(), y + 0.18, f"W={c.size}", fontsize=8, va="bottom")
        ax.axvline(-1, color="gray", ls=":", lw=0.8)     # the unit domain [-1, 1]
        ax.axvline(1, color="gray", ls=":", lw=0.8)
        ax.set_title(geometry)
        ax.set_yticks([]); ax.set_ylim(-0.5, len(Ns) - 0.3)
        ax.set_xlabel("center position")
    for gi in range(len(geometries), len(axes)):
        axes[gi].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = RESULTS_DIR / "centers_numberline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_conditioning(data):
    """cond(Phi) at the best-L_inf lambda (seed 0) vs width, one curve per
    geometry. A single light-touch diagnostic."""
    cfg = data["config"]
    rows = data["rows"]
    Ws = [uniform_geometry(N)[0] for N in cfg["base_N"]]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.suptitle("Exp07: feature-matrix conditioning at the best-$L_\\infty$ cell",
                 fontsize=13)
    for geometry in cfg["geometries"]:
        conds = []
        for N in cfg["base_N"]:
            recs = [r for r in rows if r["geometry"] == geometry
                    and r["N"] == N and r["seed"] == 0]
            best = min(recs, key=lambda r: r["linf"])
            conds.append(best["phi_cond"])
        ax.loglog(Ws, conds, "-o", color=GEOM_COLORS[geometry],
                  markersize=5, label=geometry)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlabel("width $W$ (total centers)"); ax.set_ylabel(r"cond$(\Phi)$")
    ax.set_xticks(Ws); ax.set_xticklabels(Ws); ax.legend(fontsize=9)
    plt.tight_layout()
    out = RESULTS_DIR / "conditioning.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        if not DATA_PATH.exists():
            print(f"No data at {DATA_PATH}. Run without --plot first.")
            sys.exit(1)
        with open(DATA_PATH) as f:
            data = json.load(f)
    else:
        data = collect_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_error_vs_width(data)
    plot_centers_numberline(data)
    plot_conditioning(data)
