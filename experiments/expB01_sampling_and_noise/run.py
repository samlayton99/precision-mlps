"""Experiment 8 -- randomness and noise in the lstsq readout (1D, tanh).

Follows exp07, but now the *sample points* (where we observe (x, f(x)) to solve
the readout) vary too, and we add y-noise. Everything is an overdetermined fp64
least-squares fit: W neurons < N_TRAIN sample points < N_EVAL eval points, with
N_TRAIN and N_EVAL chosen prime so the uniform train and eval grids do not align
(which would otherwise hide error at the sample points).

In exp07 every geometry was fit and evaluated on a *uniform* grid, which favors
uniform centers. Here we also let the sample points be random, so randomness in
the geometry and randomness in the sampling are tested on an equal footing.

Two figures, both x = width N, y = error (left relative L2, right L_inf), best
over a lambda sweep, mean over seeds with a min/max band.

  Figure 1 (geometry x sampling, all clean):
    {uniform centers, random centers} x {uniform samples, random samples}.

  Figure 2 (uniform centers, noise on x and/or y):
    {uniform samples, random samples} x {clean y, noisy y}.
    Random samples are the x-perturbation; noisy y adds Gaussian noise. Error is
    always measured against the clean target (recovery of the true function).

All lstsq, all fp64, target = sine. gamma = lambda/h shared across conditions at
each (N, lambda), so only randomness/noise differ.

Usage:
    python experiments/expB01_sampling_and_noise/run.py            # collect + plot
    python experiments/expB01_sampling_and_noise/run.py --plot     # plot from saved data
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.center_geometry import random_centers, uniform_centers
from src.construction.qi_mpmath import default_halo
from src.construction.readout import build_phi, solve_readout_with_bias
from src.data.targets import get_target

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_B_scaling" / "expB01_sampling_and_noise"
DATA_PATH = RESULTS_DIR / "data.json"

# --- experiment grid ---
TARGET = "sine"
BASE_N = [32, 64, 128, 256]        # uniform grid sizes -> set W (centers), span, h
HALO_LAMBDA = 0.25
LAMBDA_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00]
SEEDS = [0, 1, 2]
N_TRAIN = 1031                     # prime; > max width (461), overdetermined
N_EVAL = 7919                      # prime; ~7.7x N_TRAIN, misaligned grid
Y_NOISE_STD = 1e-3                 # Gaussian noise added to y in the "noisy" cases
PRECISION = "fp64"

# (center_geom, sample_type, noise)
ALL_CONDS = [
    ("uniform", "uniform", False),
    ("uniform", "random", False),
    ("random", "uniform", False),
    ("random", "random", False),
    ("uniform", "uniform", True),
    ("uniform", "random", True),
]

G1_COLORS = {
    ("uniform", "uniform"): "#000000", ("uniform", "random"): "#1f77b4",
    ("random", "uniform"): "#2ca02c", ("random", "random"): "#d62728",
}
G2_COLORS = {
    ("uniform", False): "#000000", ("uniform", True): "#d62728",
    ("random", False): "#1f77b4", ("random", True): "#9467bd",
}

# --- figure 3: sample-count scaling at fixed N, one line per y-noise level ---
SS_N = 64                          # fixed width for the sample-scaling study
SS_LAMBDA = 0.25                   # fixed bandwidth (viable regime, uniform centers)
SS_SAMPLE_COUNTS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
SS_NOISE_LEVELS = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def uniform_geometry(N):
    """QI grid + halo: fixes width W (centers), span, and h = 2/N for this N."""
    h = 2.0 / N
    halo = default_halo(N, lambda_star=HALO_LAMBDA)
    n_idx = np.arange(-halo, N + halo + 1)
    centers = -1.0 + n_idx.astype(np.float64) * h
    return centers.size, (float(centers[0]), float(centers[-1])), h


def make_centers(center_geom, W, span, seed):
    if center_geom == "uniform":
        return uniform_centers(W, span)
    return random_centers(W, span, seed=seed)


def make_samples(sample_type, n, seed):
    if sample_type == "uniform":
        return np.linspace(-1.0, 1.0, n)
    rng = np.random.default_rng(1000 + seed)
    return np.sort(rng.uniform(-1.0, 1.0, size=n))


def is_stochastic(center_geom, sample_type, noise):
    return center_geom == "random" or sample_type == "random" or noise


def run_cell(center_geom, sample_type, noise, N, seed, x_eval, y_eval, y_norm):
    W, span, h = uniform_geometry(N)
    centers = make_centers(center_geom, W, span, seed)
    x_s = make_samples(sample_type, N_TRAIN, seed)
    y_s = get_target(TARGET).fn_numpy(x_s)
    if noise:
        rng = np.random.default_rng(2000 + seed)
        y_s = y_s + rng.normal(0.0, Y_NOISE_STD, size=y_s.shape)

    best = {"linf": np.inf, "rel_l2": np.inf}
    for lam in LAMBDA_GRID:
        gamma_vec = np.full(W, lam / h)
        Phi_tr = build_phi(x_s, gamma_vec, centers)
        Phi_ev = build_phi(x_eval, gamma_vec, centers)
        v, b, info = solve_readout_with_bias(Phi_tr, y_s, method="lstsq")
        resid = Phi_ev @ v + b - y_eval            # eval vs CLEAN target
        linf = float(np.abs(resid).max())
        rel_l2 = float(np.linalg.norm(resid) / y_norm)
        if linf < best["linf"]:
            best = {"linf": linf, "rel_l2": rel_l2, "lambda": lam}
    best["W"] = W
    return best


def collect_sample_scaling(x_eval, y_eval, y_norm):
    """Fixed uniform geometry and N; scale the number of uniform sample points,
    one curve per y-noise level. The augmented Phi (and its SVD) depends only on
    the sample count, so we factor it once per count and reuse across noise
    levels and seeds. Returns rows."""
    W, span, h = uniform_geometry(SS_N)
    centers = uniform_centers(W, span)
    gamma_vec = np.full(W, SS_LAMBDA / h)
    Phi_ev = build_phi(x_eval, gamma_vec, centers)         # fixed across counts

    rows = []
    for n in SS_SAMPLE_COUNTS:
        x_s = np.linspace(-1.0, 1.0, n)
        y_clean = get_target(TARGET).fn_numpy(x_s)
        Phi_aug = np.hstack([build_phi(x_s, gamma_vec, centers), np.ones((n, 1))])
        U, s, Vt = np.linalg.svd(Phi_aug, full_matrices=False)
        s_inv = np.where(s > 1e-13 * s[0], 1.0 / s, 0.0)
        for noise in SS_NOISE_LEVELS:
            seeds = SEEDS if noise > 0 else [0]
            for seed in seeds:
                y_s = y_clean
                if noise > 0:
                    y_s = y_clean + np.random.default_rng(2000 + seed).normal(
                        0.0, noise, size=n)
                sol = Vt.T @ (s_inv * (U.T @ y_s))
                resid = Phi_ev @ sol[:-1] + sol[-1] - y_eval
                rows.append({
                    "n_samples": n, "noise": noise, "seed": seed, "W": W,
                    "linf": float(np.abs(resid).max()),
                    "rel_l2": float(np.linalg.norm(resid) / y_norm),
                })
    return rows


def collect_data():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x_eval = np.linspace(-1.0, 1.0, N_EVAL)
    y_eval = get_target(TARGET).fn_numpy(x_eval)
    y_norm = float(np.linalg.norm(y_eval))

    rows = []
    t0 = time.time()
    for (cg, st, noise) in ALL_CONDS:
        seeds = SEEDS if is_stochastic(cg, st, noise) else [0]
        for N in BASE_N:
            for seed in seeds:
                best = run_cell(cg, st, noise, N, seed, x_eval, y_eval, y_norm)
                rows.append({
                    "center_geom": cg, "sample_type": st, "noise": noise,
                    "N": N, "W": best["W"], "seed": seed,
                    "linf": best["linf"], "rel_l2": best["rel_l2"],
                    "best_lambda": best["lambda"],
                })
            tag = f"{cg[:4]}-ctr/{st[:4]}-samp/{'noisy' if noise else 'clean'}"
            best_linf = min(r["linf"] for r in rows
                            if r["center_geom"] == cg and r["sample_type"] == st
                            and r["noise"] == noise and r["N"] == N)
            print(f"[{time.time()-t0:5.0f}s] {tag:28s} N={N:4d} best Linf={best_linf:.2e}")

    print("collecting sample-scaling (figure 3)...")
    ss_rows = collect_sample_scaling(x_eval, y_eval, y_norm)
    print(f"[{time.time()-t0:5.0f}s] sample-scaling done ({len(ss_rows)} cells)")

    out = {
        "config": {
            "target": TARGET, "base_N": BASE_N, "halo_lambda": HALO_LAMBDA,
            "lambda_grid": LAMBDA_GRID, "seeds": SEEDS, "n_train": N_TRAIN,
            "n_eval": N_EVAL, "y_noise_std": Y_NOISE_STD, "precision": PRECISION,
            "ss_N": SS_N, "ss_lambda": SS_LAMBDA,
            "ss_sample_counts": SS_SAMPLE_COUNTS, "ss_noise_levels": SS_NOISE_LEVELS,
        },
        "rows": rows,
        "sample_scaling": ss_rows,
    }
    with open(DATA_PATH, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {DATA_PATH} ({time.time()-t0:.0f}s total)")
    return out


# ----------------------------- plotting -----------------------------

def _series(rows, N, **filt):
    """Per-seed best (over lambda) values at width N for a condition."""
    recs = [r for r in rows if r["N"] == N
            and all(r[k] == v for k, v in filt.items())]
    seeds = sorted({r["seed"] for r in recs})
    linf = np.array([min(r["linf"] for r in recs if r["seed"] == s) for s in seeds])
    rl2 = np.array([min(r["rel_l2"] for r in recs if r["seed"] == s) for s in seeds])
    return rl2, linf


def _plot_panels(data, conditions, color_fn, label_fn, title, fname):
    cfg = data["config"]
    rows = data["rows"]
    Ns = cfg["base_N"]
    metrics = [("rel_l2", 0, r"Best relative $L_2$"), ("linf", 1, r"Best $L_\infty$")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(title, fontsize=13)
    for _, ax_i, mlabel in metrics:
        ax = axes[ax_i]
        for cond in conditions:
            means, los, his = [], [], []
            for N in Ns:
                rl2, linf = _series(rows, N, **cond)
                arr = rl2 if ax_i == 0 else linf
                means.append(arr.mean()); los.append(arr.min()); his.append(arr.max())
            c = color_fn(cond)
            ax.fill_between(Ns, los, his, color=c, alpha=0.15)
            ax.loglog(Ns, means, "-o", color=c, markersize=5, label=label_fn(cond))
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel("width $N$"); ax.set_ylabel(mlabel); ax.set_title(mlabel)
        ax.set_xticks(Ns)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].legend(fontsize=9)
    plt.tight_layout()
    out = RESULTS_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_geometry(data):
    conds = [{"center_geom": cg, "sample_type": st, "noise": False}
             for cg in ("uniform", "random") for st in ("uniform", "random")]
    _plot_panels(
        data, conds,
        color_fn=lambda c: G1_COLORS[(c["center_geom"], c["sample_type"])],
        label_fn=lambda c: f"{c['center_geom'][:4]} ctr / {c['sample_type'][:4]} samp",
        title=f"Exp08 fig1: geometry x sampling (clean, lstsq, {data['config']['precision']}, "
              f"target={data['config']['target']})",
        fname="error_vs_width_geometry.png")


def plot_noise(data):
    conds = [{"center_geom": "uniform", "sample_type": st, "noise": noise}
             for st in ("uniform", "random") for noise in (False, True)]
    nstd = data["config"]["y_noise_std"]
    _plot_panels(
        data, conds,
        color_fn=lambda c: G2_COLORS[(c["sample_type"], c["noise"])],
        label_fn=lambda c: f"{c['sample_type'][:4]} samp / {'noisy' if c['noise'] else 'clean'}",
        title=f"Exp08 fig2: uniform centers, x/y perturbation "
              f"(y-noise std={nstd:g}, lstsq, {data['config']['precision']})",
        fname="error_vs_width_noise.png")


def plot_sample_scaling(data):
    """Fixed uniform geometry, fixed N; x = number of uniform sample points,
    y = error, one line per y-noise level. Two panels (rel L2, L_inf)."""
    cfg = data["config"]
    rows = data["sample_scaling"]
    counts = cfg["ss_sample_counts"]
    levels = cfg["ss_noise_levels"]
    metrics = [("rel_l2", 0, r"Relative $L_2$"), ("linf", 1, r"$L_\infty$")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.suptitle(f"Exp08 fig3: sample-count scaling under y-noise "
                 f"(uniform centers, N={cfg['ss_N']}, $\\lambda$={cfg['ss_lambda']}, lstsq, "
                 f"{cfg['precision']})", fontsize=13)
    cmap = plt.cm.viridis(np.linspace(0, 0.88, len(levels)))
    for _, ax_i, mlabel in metrics:
        ax = axes[ax_i]
        for li, noise in enumerate(levels):
            means, los, his = [], [], []
            for n in counts:
                recs = [r for r in rows if r["n_samples"] == n and r["noise"] == noise]
                vals = np.array([r["rel_l2"] if ax_i == 0 else r["linf"] for r in recs])
                means.append(vals.mean()); los.append(vals.min()); his.append(vals.max())
            color = "#000000" if noise == 0 else cmap[li]
            label = "clean" if noise == 0 else f"$\\sigma$={noise:.0e}"
            ax.fill_between(counts, los, his, color=color, alpha=0.12)
            ax.loglog(counts, means, "-o", color=color, markersize=4, label=label)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel("number of (uniform) sample points")
        ax.set_ylabel(mlabel); ax.set_title(mlabel)
    axes[1].legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out = RESULTS_DIR / "error_vs_samples_noise.png"
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
    plot_geometry(data)
    plot_noise(data)
    plot_sample_scaling(data)
