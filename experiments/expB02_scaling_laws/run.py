"""Experiment 9 -- scaling laws for the fixed-geometry lstsq readout (1D).

Fixed 1D QI geometry (uniform centers + halo, gamma = lambda/h), readout solved
by least squares -- the method exp03/0B/11/12 showed reaches the fp64 floor on
the right (uniform) geometry. Here we trace two scaling laws, across multiple
targets and multiple activations:

  Figure 1 (error_vs_width.png) -- SIZE scaling.
    x = grid size N on a fine logspace 16..128 (the resolution knob: h = 2/N).
    Overdetermined clean fit; for each (N, activation, target) the error is the
    best over a lambda sweep (the U-shaped bandwidth tradeoff, see exp06). Two
    columns (relative L2, L_inf), three rows (activations), one line per target.

  Figure 2 (error_vs_data_clean.png) -- DATA scaling, clean.
    Fixed geometry (N = DATA_N, lambda = DATA_LAMBDA); x = number of training
    points n_data on a fine logspace that STARTS underdetermined (n_data < W)
    and ENDS heavily overdetermined. Clean targets. Same 3 x 2 panel layout.

  Figure 3 (error_vs_data_noise.png) -- DATA scaling under fixed noise.
    Same as Figure 2 but with fixed Gaussian y-noise (std = NOISE_STD = 1e-4),
    averaged over seeds. Exposes the statistical sigma * n^-1/2 data law: a
    reference slope of -1/2 is drawn for comparison.

Layout of every figure: rows = activations, left col = relative L2, right col =
L_inf, one colored line per target. All fp64, all numpy.

Geometry is fixed by HALO_LAMBDA (sets W, span, h for each N); gamma = lambda/h
is set separately (swept in Figure 1, fixed in Figures 2-3). The readout is a
plain np.linalg lstsq via a truncated SVD with bias (matches exp08).

Usage:
    python experiments/expB02_scaling_laws/run.py            # collect + plot
    python experiments/expB02_scaling_laws/run.py --plot     # plot from saved data
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
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.qi_mpmath import default_halo
from src.data.targets import get_target

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_B_scaling" / "expB02_scaling_laws"
DATA_PATH = RESULTS_DIR / "data.json"

# ----------------------------- experiment grid -----------------------------
TARGETS = ["sine", "sine_8pi", "runge", "sine_mixture", "exp", "abs_cubed"]
ACTIVATIONS = ["tanh", "gelu", "relu"]

HALO_LAMBDA = 0.25            # fixes geometry (W, span, h) per N; gamma set separately
RCOND = 1e-13                 # SVD truncation floor (relative to s_max)

# Figure 1: width / size scaling
WIDTH_N_MIN, WIDTH_N_MAX, WIDTH_N_POINTS = 16, 1024, 32
WIDTH_LAMBDA_GRID = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00]
FIXED_LAMBDA = 0.25          # single bandwidth for the corrected (no best-over-lambda) width law
WIDTH_N_TRAIN = 4001         # prime, overdetermined for W up to ~1845 (N=1024)
WIDTH_N_EVAL = 8009          # prime, misaligned with the train grid
WIDTH_FLOOR_LINE = 5e-14     # black reference line drawn on every width panel

# Figures 2-3: data scaling (fixed geometry)
DATA_N = 128                 # fixed grid size -> width W ~ 269
DATA_LAMBDA = 0.25           # fixed bandwidth (viable regime)
DATA_N_EVAL = 8009           # prime eval grid
# Clean (figure 2): zoom on the under- -> over-determined transition around W+1.
DATA_N_MIN, DATA_N_MAX, DATA_N_POINTS = 16, 512, 30
# Noisy (figure 3): the statistical regime lives far out, so extend the axis.
NOISE_N_MIN, NOISE_N_MAX, NOISE_N_POINTS = 16, 500_000, 42
NOISE_STD = 1e-4             # fixed Gaussian y-noise for figure 3
NOISE_SEEDS = [0, 1, 2]

TARGET_COLORS = {
    "sine": "#1f77b4", "sine_8pi": "#ff7f0e", "runge": "#2ca02c",
    "sine_mixture": "#d62728", "exp": "#9467bd", "abs_cubed": "#8c564b",
}


# ----------------------------- core machinery -----------------------------

def activation(name, z):
    if name == "tanh":
        return np.tanh(z)
    if name == "gelu":
        return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "sin":
        return np.sin(z)
    raise ValueError(f"Unknown activation: {name}")


def build_phi_act(x, gamma, centers, act):
    """Phi[i, k] = act(gamma_k * (x_i - centers_k)). x:[n], gamma/centers:[W]."""
    x = np.asarray(x, dtype=np.float64).ravel()
    gamma = np.asarray(gamma, dtype=np.float64).ravel()
    centers = np.asarray(centers, dtype=np.float64).ravel()
    return activation(act, gamma[None, :] * (x[:, None] - centers[None, :]))


def uniform_geometry(N):
    """QI grid + halo: fixes width W (centers), span, and h = 2/N for this N."""
    h = 2.0 / N
    halo = default_halo(N, lambda_star=HALO_LAMBDA)
    n_idx = np.arange(-halo, N + halo + 1)
    centers = -1.0 + n_idx.astype(np.float64) * h
    return centers, h


def svd_factor(Phi_aug):
    """Truncated-SVD factorization of the bias-augmented train matrix."""
    U, s, Vt = np.linalg.svd(Phi_aug, full_matrices=False)
    smax = s[0] if s.size else 0.0
    s_inv = np.where(s > RCOND * smax, 1.0 / np.where(s > 0, s, 1.0), 0.0)
    return U, s_inv, Vt


def solve_multi(U, s_inv, Vt, Y):
    """Min-norm/least-squares solve for multiple RHS. Y:[n, k] -> sol:[W+1, k]."""
    return Vt.T @ (s_inv[:, None] * (U.T @ Y))


def errors(Phi_ev, sol, Y_ev, y_norms):
    """Per-column eval L_inf and relative L2. sol:[W+1, k], last row is bias."""
    pred = Phi_ev @ sol[:-1, :] + sol[-1:, :]      # [n_eval, k]
    resid = pred - Y_ev
    linf = np.abs(resid).max(axis=0)
    rel_l2 = np.linalg.norm(resid, axis=0) / y_norms
    return linf, rel_l2


def stack_targets(x):
    """Columns = TARGETS evaluated at x. Returns [len(x), n_targets]."""
    return np.stack([get_target(t).fn_numpy(x) for t in TARGETS], axis=1)


# ----------------------------- collectors -----------------------------

def collect_width(t0):
    """Figure 1: size scaling. Best-over-lambda clean fit vs grid size N."""
    Ns = np.unique(np.round(np.logspace(
        np.log10(WIDTH_N_MIN), np.log10(WIDTH_N_MAX), WIDTH_N_POINTS)).astype(int))
    x_tr = np.linspace(-1.0, 1.0, WIDTH_N_TRAIN)
    x_ev = np.linspace(-1.0, 1.0, WIDTH_N_EVAL)
    Y_tr = stack_targets(x_tr)
    Y_ev = stack_targets(x_ev)
    y_norms = np.linalg.norm(Y_ev, axis=0)

    rows = []
    for N in Ns:
        centers, h = uniform_geometry(int(N))
        W = centers.size
        for act in ACTIVATIONS:
            best_linf = np.full(len(TARGETS), np.inf)
            best_rl2 = np.full(len(TARGETS), np.inf)
            for lam in WIDTH_LAMBDA_GRID:
                gamma = np.full(W, lam / h)
                Phi_tr = build_phi_act(x_tr, gamma, centers, act)
                Phi_ev = build_phi_act(x_ev, gamma, centers, act)
                Phi_aug = np.hstack([Phi_tr, np.ones((Phi_tr.shape[0], 1))])
                U, s_inv, Vt = svd_factor(Phi_aug)
                sol = solve_multi(U, s_inv, Vt, Y_tr)
                linf, rl2 = errors(Phi_ev, sol, Y_ev, y_norms)
                best_linf = np.minimum(best_linf, linf)
                best_rl2 = np.minimum(best_rl2, rl2)
            for ti, t in enumerate(TARGETS):
                rows.append({"N": int(N), "W": int(W), "activation": act,
                             "target": t, "linf": float(best_linf[ti]),
                             "rel_l2": float(best_rl2[ti])})
        print(f"[{time.time()-t0:5.0f}s] width N={int(N):4d} (W={W}) done")
    return rows


def collect_width_fixed(t0):
    """Fixed-lambda size scaling. Same as collect_width but a SINGLE fixed
    lambda (FIXED_LAMBDA), no best-over-lambda min -- this is the corrected
    width scaling law that decouples bandwidth selection from resolution.
    """
    Ns = np.unique(np.round(np.logspace(
        np.log10(WIDTH_N_MIN), np.log10(WIDTH_N_MAX), WIDTH_N_POINTS)).astype(int))
    x_tr = np.linspace(-1.0, 1.0, WIDTH_N_TRAIN)
    x_ev = np.linspace(-1.0, 1.0, WIDTH_N_EVAL)
    Y_tr = stack_targets(x_tr)
    Y_ev = stack_targets(x_ev)
    y_norms = np.linalg.norm(Y_ev, axis=0)

    rows = []
    for N in Ns:
        centers, h = uniform_geometry(int(N))
        W = centers.size
        gamma = np.full(W, FIXED_LAMBDA / h)
        for act in ACTIVATIONS:
            Phi_tr = build_phi_act(x_tr, gamma, centers, act)
            Phi_ev = build_phi_act(x_ev, gamma, centers, act)
            Phi_aug = np.hstack([Phi_tr, np.ones((Phi_tr.shape[0], 1))])
            U, s_inv, Vt = svd_factor(Phi_aug)
            sol = solve_multi(U, s_inv, Vt, Y_tr)
            linf, rl2 = errors(Phi_ev, sol, Y_ev, y_norms)
            for ti, t in enumerate(TARGETS):
                rows.append({"N": int(N), "W": int(W), "activation": act,
                             "target": t, "linf": float(linf[ti]),
                             "rel_l2": float(rl2[ti])})
        print(f"[{time.time()-t0:5.0f}s] width-fixed N={int(N):4d} (W={W}) done")
    return rows


def collect_data(t0, noise_std, n_min, n_max, n_points):
    """Figures 2-3: data scaling at fixed geometry. noise_std=0 -> clean.

    The clean and noisy figures use different n_data ranges (clean zooms on the
    W+1 transition; noisy extends into the far statistical regime), so the
    n-grid is passed in.
    """
    centers, h = uniform_geometry(DATA_N)
    W = centers.size
    gamma = np.full(W, DATA_LAMBDA / h)
    x_ev = np.linspace(-1.0, 1.0, DATA_N_EVAL)
    Y_ev = stack_targets(x_ev)
    y_norms = np.linalg.norm(Y_ev, axis=0)
    Phi_ev = {act: build_phi_act(x_ev, gamma, centers, act) for act in ACTIVATIONS}

    ns = np.unique(np.round(np.logspace(
        np.log10(n_min), np.log10(n_max), n_points)).astype(int))
    seeds = NOISE_SEEDS if noise_std > 0 else [0]

    rows = []
    for n in ns:
        x_tr = np.linspace(-1.0, 1.0, int(n))
        Y_clean = stack_targets(x_tr)
        for act in ACTIVATIONS:
            Phi_tr = build_phi_act(x_tr, gamma, centers, act)
            Phi_aug = np.hstack([Phi_tr, np.ones((Phi_tr.shape[0], 1))])
            del Phi_tr                                 # free the big train matrix early
            U, s_inv, Vt = svd_factor(Phi_aug)        # factor once, reuse across seeds
            del Phi_aug                                # release before the seed loop
            for seed in seeds:
                Y_tr = Y_clean
                if noise_std > 0:
                    rng = np.random.default_rng(2000 + seed)
                    Y_tr = Y_clean + rng.normal(0.0, noise_std, size=Y_clean.shape)
                sol = solve_multi(U, s_inv, Vt, Y_tr)
                linf, rl2 = errors(Phi_ev[act], sol, Y_ev, y_norms)
                for ti, t in enumerate(TARGETS):
                    rows.append({"n_data": int(n), "W": int(W), "activation": act,
                                 "target": t, "seed": seed,
                                 "linf": float(linf[ti]), "rel_l2": float(rl2[ti])})
        tag = "clean" if noise_std == 0 else f"noise={noise_std:g}"
        print(f"[{time.time()-t0:5.0f}s] data {tag} n={int(n):7d} done")
    return rows


def collect_all():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("collecting width scaling (figure 1)...")
    width_rows = collect_width(t0)
    print("collecting width scaling, fixed lambda (figure 4)...")
    width_fixed_rows = collect_width_fixed(t0)
    print("collecting data scaling, clean (figure 2)...")
    data_clean = collect_data(t0, 0.0, DATA_N_MIN, DATA_N_MAX, DATA_N_POINTS)
    print("collecting data scaling, noisy (figure 3)...")
    data_noise = collect_data(t0, NOISE_STD, NOISE_N_MIN, NOISE_N_MAX, NOISE_N_POINTS)

    out = {
        "config": {
            "targets": TARGETS, "activations": ACTIVATIONS,
            "halo_lambda": HALO_LAMBDA, "rcond": RCOND,
            "width_n_range": [WIDTH_N_MIN, WIDTH_N_MAX, WIDTH_N_POINTS],
            "width_lambda_grid": WIDTH_LAMBDA_GRID,
            "fixed_lambda": FIXED_LAMBDA,
            "width_n_train": WIDTH_N_TRAIN, "width_n_eval": WIDTH_N_EVAL,
            "data_N": DATA_N, "data_lambda": DATA_LAMBDA,
            "data_n_range": [DATA_N_MIN, DATA_N_MAX, DATA_N_POINTS],
            "noise_n_range": [NOISE_N_MIN, NOISE_N_MAX, NOISE_N_POINTS],
            "data_n_eval": DATA_N_EVAL, "noise_std": NOISE_STD,
            "noise_seeds": NOISE_SEEDS,
        },
        "width": width_rows,
        "width_fixed": width_fixed_rows,
        "data_clean": data_clean,
        "data_noise": data_noise,
    }
    with open(DATA_PATH, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {DATA_PATH} ({time.time()-t0:.0f}s total)")
    return out


# ----------------------------- plotting -----------------------------

def _panels(activations):
    fig, axes = plt.subplots(len(activations), 2,
                             figsize=(13, 4.2 * len(activations)), squeeze=False)
    return fig, axes


def _add_top_legend(fig, extra=None):
    """Single shared legend above all subplots (one entry per target + extras)."""
    handles = [Line2D([0], [0], color=TARGET_COLORS[t], marker="o", ms=4, lw=1.5,
                      label=t) for t in TARGETS]
    if extra:
        handles += extra
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=len(handles), frameon=True, fontsize=9, handlelength=1.8,
               columnspacing=1.2)


def _finish_panel(ax, xlabel, ylabel, title):
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


def plot_width(data):
    cfg = data["config"]
    rows = data["width"]
    acts = cfg["activations"]
    fig, axes = _panels(acts)
    fig.suptitle(
        f"Exp09 fig1: size scaling -- fixed uniform geometry, best-over-$\\lambda$ "
        f"lstsq, clean (fp64)", fontsize=13, y=0.995)
    for ri, act in enumerate(acts):
        for ci, (metric, mlabel) in enumerate(
                [("rel_l2", r"relative $L_2$"), ("linf", r"absolute $L_\infty$")]):
            ax = axes[ri][ci]
            for t in cfg["targets"]:
                recs = sorted((r for r in rows if r["activation"] == act
                               and r["target"] == t), key=lambda r: r["N"])
                xs = [r["N"] for r in recs]
                ys = [r[metric] for r in recs]
                ax.loglog(xs, ys, "-o", ms=3, color=TARGET_COLORS[t], label=t)
            ax.axhline(WIDTH_FLOOR_LINE, color="black", lw=1.3, alpha=0.85)
            _finish_panel(ax, "grid size $N$ ($h=2/N$)", mlabel, f"{act} -- {mlabel}")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _add_top_legend(fig, extra=[Line2D([0], [0], color="black", lw=1.3,
                                       label=f"{WIDTH_FLOOR_LINE:g} floor")])
    out = RESULTS_DIR / "error_vs_width.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_width_fixed(data):
    """Figure 4: corrected size scaling at a single fixed lambda (no best-over-lambda)."""
    cfg = data["config"]
    rows = data["width_fixed"]
    lam = cfg.get("fixed_lambda", FIXED_LAMBDA)
    acts = cfg["activations"]
    fig, axes = _panels(acts)
    fig.suptitle(
        f"expB02 fig4: size scaling -- fixed uniform geometry, "
        f"FIXED $\\lambda={lam}$ lstsq, clean (fp64)", fontsize=13, y=0.995)
    for ri, act in enumerate(acts):
        for ci, (metric, mlabel) in enumerate(
                [("rel_l2", r"relative $L_2$"), ("linf", r"absolute $L_\infty$")]):
            ax = axes[ri][ci]
            for t in cfg["targets"]:
                recs = sorted((r for r in rows if r["activation"] == act
                               and r["target"] == t), key=lambda r: r["N"])
                xs = [r["N"] for r in recs]
                ys = [r[metric] for r in recs]
                ax.loglog(xs, ys, "-o", ms=3, color=TARGET_COLORS[t], label=t)
            ax.axhline(WIDTH_FLOOR_LINE, color="black", lw=1.3, alpha=0.85)
            _finish_panel(ax, "grid size $N$ ($h=2/N$)", mlabel, f"{act} -- {mlabel}")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _add_top_legend(fig, extra=[Line2D([0], [0], color="black", lw=1.3,
                                       label=f"{WIDTH_FLOOR_LINE:g} floor")])
    out = RESULTS_DIR / "fixed_lambda_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def _plot_data(data, key, noisy, fname, title):
    cfg = data["config"]
    rows = data[key]
    acts = cfg["activations"]
    W = rows[0]["W"] if rows else None
    fig, axes = _panels(acts)
    fig.suptitle(title, fontsize=13, y=0.995)
    for ri, act in enumerate(acts):
        for ci, (metric, mlabel) in enumerate(
                [("rel_l2", r"relative $L_2$"), ("linf", r"absolute $L_\infty$")]):
            ax = axes[ri][ci]
            for t in cfg["targets"]:
                recs = [r for r in rows if r["activation"] == act and r["target"] == t]
                ns = sorted({r["n_data"] for r in recs})
                means = [np.mean([r[metric] for r in recs if r["n_data"] == n])
                         for n in ns]
                ax.loglog(ns, means, "-o", ms=3, color=TARGET_COLORS[t], label=t)
            # mark the underdetermined -> overdetermined threshold (n_data = W+1)
            if W is not None:
                ax.axvline(W + 1, color="gray", ls=":", lw=1.2, alpha=0.7)
            _finish_panel(ax, "number of training points $n_{data}$", mlabel,
                          f"{act} -- {mlabel}")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    extra = ([Line2D([0], [0], color="gray", ls=":", lw=1.2, label="$n=W+1$")]
             if W is not None else None)
    _add_top_legend(fig, extra=extra)
    out = RESULTS_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_data_clean(data):
    cfg = data["config"]
    _plot_data(
        data, "data_clean", noisy=False, fname="error_vs_data_clean.png",
        title=f"Exp09 fig2: data scaling (clean) -- fixed geometry "
              f"N={cfg['data_N']}, $\\lambda$={cfg['data_lambda']}, lstsq (fp64); "
              f"dotted line = n_data = W+1 threshold")


def plot_data_noise(data):
    cfg = data["config"]
    _plot_data(
        data, "data_noise", noisy=True, fname="error_vs_data_noise.png",
        title=f"Exp09 fig3: data scaling under y-noise $\\sigma$={cfg['noise_std']:g} "
              f"-- fixed geometry N={cfg['data_N']}, $\\lambda$={cfg['data_lambda']}, "
              f"lstsq (fp64); mean over {len(cfg['noise_seeds'])} seeds")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        if not DATA_PATH.exists():
            print(f"No data at {DATA_PATH}. Run without --plot first.")
            sys.exit(1)
        with open(DATA_PATH) as f:
            data = json.load(f)
        # Augment older data.json (pre fixed-lambda) with the cheap single-lambda
        # width law, without recollecting the expensive width/data/noise sweeps.
        if "width_fixed" not in data:
            print("width_fixed missing -- collecting only the fixed-lambda width law...")
            data["width_fixed"] = collect_width_fixed(time.time())
            data.setdefault("config", {})["fixed_lambda"] = FIXED_LAMBDA
            with open(DATA_PATH, "w") as f:
                json.dump(data, f)
            print(f"Augmented {DATA_PATH} with width_fixed")
    else:
        data = collect_all()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_width(data)
    plot_width_fixed(data)
    plot_data_clean(data)
    plot_data_noise(data)
