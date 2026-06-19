"""Experiment 3 -- Phase 1: Optimizers on frozen QI geometry (ladder level 3).

Question: with the inner-layer geometry FIXED in the correct regime (correct
lambda, correct uniform spacing, gamma = lambda*/h), how well do different
optimizers recover the readout weights from random initialization? Since the
readout is linear and the loss is MSE, this subproblem is CONVEX, so the
least-squares solution is the global optimum the optimizers are chasing.

Setup (geometry frozen, only readout trained):
  - Geometry: lambda* = 0.25, uniform grid with halo = default_halo(N, 0.25),
    gamma = lambda*/h. gamma and centers are frozen; only readout (v, bias)
    trains, from random (Xavier-uniform) init -- matching continuous-mlps.
  - Phi is constant (geometry frozen) so we precompute it once and train on
    pred = Phi @ v + bias. All fp64, full-batch MSE.

Optimizer setups compared (high-LR start -> cosine decay to near-zero, with a
short linear warmup, over a long step budget):
  - lstsq : numpy least-squares (SVD min-norm). Global-optimum baseline.
  - sgd   : SGD. Peak LR = SGD_PEAK_FRAC * divergence threshold (n/smax^2),
            per cell (the inherited lr=2e-2 diverges; cond(Phi) ~ 1e19).
  - adam  : Adam, peak LR = ADAM_PEAK (betas/eps = torch defaults).
  Both use linear warmup (WARMUP) then cosine decay to END_FRAC * peak over
  TOTAL_STEPS. LBFGS is omitted for this pass.

Two figures (all error on the eval set):
  1. error_vs_width.png       -- 3 targets (rows) x {rel L2, L_inf} (cols).
  2. convergence_<target>.png -- 4 widths (rows) x {rel L2, L_inf} (cols).

Usage:
    python experiments/exp03_geometry_ladder/run.py            # collect + plot
    python experiments/exp03_geometry_ladder/run.py --plot     # plot from saved data
"""

import json
import math
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

from src.construction.qi_mpmath import default_halo
from src.construction.readout import build_phi, solve_readout_with_bias
from src.data.targets import get_target

RESULTS_DIR = REPO_ROOT / "results" / "exp03_geometry_ladder"
DATA_PATH = RESULTS_DIR / "phase1_data.json"

# --- experiment grid ---
TARGETS = ["sine", "sine_8pi", "runge"]
WIDTHS = [32, 64, 128, 256]
SEEDS = [0, 1, 2, 3, 4]
LAMBDA_STAR = 0.25            # exp01's shared lstsq optimum; the "correct regime"
N_TRAIN = 1024               # matches continuous-mlps 1D fits
N_EVAL = 4096                # matches continuous-mlps 1D fits

# --- schedule: short warmup -> high peak -> cosine decay to END_FRAC * peak ---
TOTAL_STEPS = 200000
WARMUP = 2000                # ~1% linear warmup
END_FRAC = 1e-7              # cosine floor as a fraction of peak (extremely low)
ADAM_PEAK = 1e-1             # Adam peak LR (much higher than the old constant 1e-3)
SGD_PEAK_FRAC = 0.9          # peak = SGD_PEAK_FRAC * (2n/smax^2) divergence threshold

SETUPS = [
    {"name": "sgd",  "label": "SGD",  "color": "#1f77b4"},
    {"name": "adam", "label": "Adam", "color": "#ff7f0e"},
]
LSTSQ_COLOR = "#000000"


def build_geometry(N):
    """Frozen QI geometry: gamma = lambda*/h and the halo'd uniform grid.

    Only the geometry is needed here (not the cardinal coefficients), so we
    build it directly -- exact, and avoids the fp64 Toeplitz ill-conditioning
    that the cardinal-coefficient solve hits at lambda=0.25.
    """
    h = 2.0 / N
    halo = default_halo(N, lambda_star=LAMBDA_STAR)
    n_idx = np.arange(-halo, N + halo + 1)
    centers = -1.0 + n_idx.astype(np.float64) * h          # x_n = -1 + n*h
    gamma = LAMBDA_STAR / h                                  # gamma = lambda*/h
    return gamma, centers, h, halo


def lr_at(step, peak, warmup, total):
    """Linear warmup to peak, then cosine decay to END_FRAC * peak."""
    end = END_FRAC * peak
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return end + 0.5 * (peak - end) * (1.0 + math.cos(math.pi * progress))


def make_eval_steps(total, k=200):
    """Log-spaced global step indices to evaluate at, plus a few early linear
    points. Deterministic so per-seed traces share an x-axis (averageable)."""
    pts = np.unique(np.geomspace(1, total, k).astype(int))
    early = np.arange(1, 21)
    return {int(p) for p in set(pts.tolist()) | set(early.tolist())}


def xavier_readout(neurons, seed):
    """Xavier-uniform readout init (matches continuous-mlps), bias = 0."""
    g = torch.Generator().manual_seed(seed)
    bound = float(np.sqrt(6.0 / (neurons + 1)))   # sqrt(6/(fan_in+fan_out))
    v = (torch.rand(neurons, generator=g, dtype=torch.float64) * 2 - 1) * bound
    b = torch.zeros(1, dtype=torch.float64)
    return v, b


def final_metrics(Phi_ev_np, v_np, gamma, h):
    """Violation diagnostics at the final state (computed once per cell)."""
    s = np.linalg.svd(Phi_ev_np, compute_uv=False)
    smax = float(s[0]); smin = float(s[-1])
    tol = 1e-12 * smax
    return {
        "gamma": float(gamma),
        "lambda": float(gamma * h),
        "max_v": float(np.abs(v_np).max()),
        "v_l2": float(np.linalg.norm(v_np)),
        "feature_rank": int((s > tol).sum()),
        "stable_rank": float((s ** 2).sum() / smax ** 2),
        "phi_cond": float(smax / smin) if smin > 0 else float("inf"),
        "phi_smin": smin, "phi_smax": smax,
    }


def train_scheduled(opt_name, peak, Phi_tr, y_tr, Phi_ev, y_ev, y_norm, gamma, h, seed):
    """Train readout (v, bias) with warmup+cosine schedule. Returns traces+final."""
    neurons = Phi_tr.shape[1]
    v0, b0 = xavier_readout(neurons, seed)
    v = v0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)

    if opt_name == "adam":
        opt = torch.optim.Adam([v, b], lr=peak)
    elif opt_name == "sgd":
        opt = torch.optim.SGD([v, b], lr=peak)
    else:
        raise ValueError(opt_name)

    eval_set = make_eval_steps(TOTAL_STEPS)
    steps_log, linf_log, rl2_log, lr_log = [], [], [], []

    def evaluate():
        with torch.no_grad():
            resid = Phi_ev @ v + b - y_ev
            return float(resid.abs().max()), float(torch.linalg.norm(resid) / y_norm)

    l, r = evaluate()
    steps_log.append(0); linf_log.append(l); rl2_log.append(r); lr_log.append(lr_at(0, peak, WARMUP, TOTAL_STEPS))

    for step in range(TOTAL_STEPS):
        lr = lr_at(step, peak, WARMUP, TOTAL_STEPS)
        for grp in opt.param_groups:
            grp["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss = ((Phi_tr @ v + b - y_tr) ** 2).mean()
        loss.backward()
        opt.step()
        gstep = step + 1
        if gstep in eval_set:
            li, rl = evaluate()
            steps_log.append(gstep); linf_log.append(li); rl2_log.append(rl); lr_log.append(lr)

    fm = final_metrics(Phi_ev.detach().cpu().numpy(), v.detach().cpu().numpy(), gamma, h)
    return {"steps": steps_log, "linf": linf_log, "rel_l2": rl2_log, "lr": lr_log, "final": fm}


def collect_data():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x_train = np.linspace(-1, 1, N_TRAIN)
    x_eval = np.linspace(-1, 1, N_EVAL)

    trained, lstsq_rows = [], []
    t0 = time.time()
    total_cells = len(TARGETS) * len(WIDTHS)
    cell = 0

    for target_name in TARGETS:
        target = get_target(target_name)
        y_train_np = target.fn_numpy(x_train)
        y_eval_np = target.fn_numpy(x_eval)
        y_norm = float(np.linalg.norm(y_eval_np))

        for N in WIDTHS:
            cell += 1
            gamma, centers, h, halo = build_geometry(N)
            neurons = len(centers)
            gamma_vec = np.full(neurons, gamma)
            Phi_tr_np = build_phi(x_train, gamma_vec, centers)
            Phi_ev_np = build_phi(x_eval, gamma_vec, centers)

            # --- lstsq baseline (seed-independent) ---
            v_ls, b_ls, info = solve_readout_with_bias(Phi_tr_np, y_train_np, method="lstsq")
            err_ls = np.abs(Phi_ev_np @ v_ls + b_ls - y_eval_np)
            ls_linf = float(err_ls.max())
            ls_rl2 = float(np.linalg.norm(err_ls) / y_norm)
            lstsq_rows.append({
                "target": target_name, "N": N, "neurons": neurons,
                "linf": ls_linf, "rel_l2": ls_rl2,
                "rank": info.get("rank"), "max_v": float(np.abs(v_ls).max()),
            })

            # per-cell peak LRs: full-batch GD on the MEAN-squared loss diverges
            # for lr >= n/smax(A)^2 (Hessian top eigenvalue = (2/n) smax^2). Take
            # SGD_PEAK_FRAC of that threshold as the cosine peak.
            A_aug = np.hstack([Phi_tr_np, np.ones((N_TRAIN, 1))])
            smax_aug = float(np.linalg.svd(A_aug, compute_uv=False)[0])
            sgd_peak = SGD_PEAK_FRAC * (N_TRAIN / smax_aug ** 2)
            peaks = {"sgd": sgd_peak, "adam": ADAM_PEAK}

            Phi_tr = torch.tensor(Phi_tr_np); Phi_ev = torch.tensor(Phi_ev_np)
            y_tr = torch.tensor(y_train_np); y_ev = torch.tensor(y_eval_np)
            y_norm_t = torch.linalg.norm(y_ev)

            cell_best = {}
            for setup in SETUPS:
                finals = []
                for seed in SEEDS:
                    res = train_scheduled(setup["name"], peaks[setup["name"]],
                                          Phi_tr, y_tr, Phi_ev, y_ev, y_norm_t,
                                          gamma, h, seed)
                    finals.append(res["linf"][-1])
                    trained.append({
                        "setup": setup["name"], "target": target_name, "N": N,
                        "neurons": neurons, "seed": seed,
                        "peak_lr": peaks[setup["name"]],
                        "steps": res["steps"], "linf": res["linf"],
                        "rel_l2": res["rel_l2"], "lr": res["lr"], "final": res["final"],
                    })
                cell_best[setup["name"]] = min(finals)

            elapsed = time.time() - t0
            print(f"[{cell}/{total_cells}] {target_name} N={N} "
                  f"(neurons={neurons}, halo={halo}) | lstsq Linf={ls_linf:.2e} | "
                  f"sgd_peak={sgd_peak:.1e} | "
                  + " ".join(f"{k}={v:.1e}" for k, v in cell_best.items())
                  + f" | {elapsed:.0f}s")

    out = {
        "config": {
            "targets": TARGETS, "widths": WIDTHS, "seeds": SEEDS,
            "lambda_star": LAMBDA_STAR, "n_train": N_TRAIN, "n_eval": N_EVAL,
            "total_steps": TOTAL_STEPS, "warmup": WARMUP, "end_frac": END_FRAC,
            "adam_peak": ADAM_PEAK, "sgd_peak_frac": SGD_PEAK_FRAC,
        },
        "trained": trained, "lstsq": lstsq_rows,
    }
    with open(DATA_PATH, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {DATA_PATH} ({time.time() - t0:.0f}s total)")
    return out


# ----------------------------- plotting -----------------------------

def _agg(records):
    """Stack per-seed traces sharing an x-axis -> (steps, mean, lo, hi) per metric."""
    steps = records[0]["steps"]
    out = {}
    for key in ("linf", "rel_l2"):
        arr = np.array([r[key] for r in records])  # [n_seeds, n_steps]
        out[key] = (np.array(steps), arr.mean(0), arr.min(0), arr.max(0))
    return out


def plot_error_vs_width(data):
    trained, lstsq = data["trained"], data["lstsq"]
    metrics = [("rel_l2", r"Relative $L_2$ error"), ("linf", r"$L_\infty$ error")]
    nrows = len(TARGETS)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4.2 * nrows), sharex=True)
    fig.suptitle("Phase 1: final eval error vs width (frozen QI geometry, "
                 r"$\lambda^*=0.25$; warmup+cosine schedule)", fontsize=14, y=0.995)

    for ri, target_name in enumerate(TARGETS):
        for ci, (mkey, mlabel) in enumerate(metrics):
            ax = axes[ri][ci] if nrows > 1 else axes[ci]
            for setup in SETUPS:
                means, los, his = [], [], []
                for N in WIDTHS:
                    recs = [r for r in trained if r["setup"] == setup["name"]
                            and r["target"] == target_name and r["N"] == N]
                    finals = np.array([r[mkey][-1] for r in recs])
                    means.append(finals.mean()); los.append(finals.min()); his.append(finals.max())
                ax.fill_between(WIDTHS, los, his, color=setup["color"], alpha=0.18)
                ax.semilogy(WIDTHS, means, "-o", color=setup["color"],
                            markersize=4, label=setup["label"])
            ls = [next(r for r in lstsq if r["target"] == target_name and r["N"] == N)[mkey]
                  for N in WIDTHS]
            ax.semilogy(WIDTHS, ls, "--s", color=LSTSQ_COLOR, markersize=4, label="lstsq")
            ax.set_xscale("log", base=2)
            ax.set_xticks(WIDTHS); ax.set_xticklabels(WIDTHS)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_ylabel(f"{target_name}\n{mlabel}" if ci == 0 else mlabel)
            if ri == 0:
                ax.set_title(mlabel)
            if ri == nrows - 1:
                ax.set_xlabel("width $N$")

    handles, labels = (axes[0][0] if nrows > 1 else axes[0]).get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = RESULTS_DIR / "error_vs_width.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_convergence(data, target_name):
    trained, lstsq = data["trained"], data["lstsq"]
    metrics = [("rel_l2", r"Relative $L_2$ error"), ("linf", r"$L_\infty$ error")]
    nrows = len(WIDTHS)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 3.6 * nrows), sharex=True)
    fig.suptitle(f"Phase 1: convergence vs step -- {target_name} "
                 r"(frozen QI geometry, $\lambda^*=0.25$; warmup+cosine)", fontsize=14, y=0.997)

    for ri, N in enumerate(WIDTHS):
        for ci, (mkey, mlabel) in enumerate(metrics):
            ax = axes[ri][ci]
            for setup in SETUPS:
                recs = [r for r in trained if r["setup"] == setup["name"]
                        and r["target"] == target_name and r["N"] == N]
                agg = _agg(recs)
                steps, mean, lo, hi = agg[mkey]
                xs = np.maximum(steps, 1)
                ax.fill_between(xs, lo, hi, color=setup["color"], alpha=0.18)
                ax.loglog(xs, mean, "-", color=setup["color"], linewidth=1.4,
                          label=setup["label"])
            ls = next(r for r in lstsq if r["target"] == target_name and r["N"] == N)[mkey]
            ax.axhline(ls, ls="--", color=LSTSQ_COLOR, linewidth=1.2, label="lstsq")
            ax.grid(True, alpha=0.3, which="both")
            ax.set_ylabel(f"N={N}\n{mlabel}" if ci == 0 else mlabel)
            if ri == 0:
                ax.set_title(mlabel)
            if ri == nrows - 1:
                ax.set_xlabel("optimization step")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.005))
    plt.tight_layout(rect=[0, 0.025, 1, 0.975])
    out = RESULTS_DIR / f"convergence_{target_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_all(data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_error_vs_width(data)
    for target_name in TARGETS:
        plot_convergence(data, target_name)


if __name__ == "__main__":
    plot_only = "--plot" in sys.argv
    if plot_only:
        if not DATA_PATH.exists():
            print(f"No data at {DATA_PATH}. Run without --plot first.")
            sys.exit(1)
        with open(DATA_PATH) as f:
            data = json.load(f)
    else:
        data = collect_data()
    plot_all(data)
