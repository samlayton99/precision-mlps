"""Experiment 12 -- geometry ladder, Phase 1: training the readout on frozen geometry.

Question: with the inner-layer geometry FIXED in the correct regime (correct
lambda, uniform spacing, gamma = lambda*/h), how close does a standard optimizer
get to the exact readout? The readout is linear and the loss is MSE, so this
subproblem is CONVEX and the least-squares solution is the global optimum the
optimizer is chasing -- the cleanest possible rung of the ladder.

Setup (geometry frozen, only the readout trained):
  - Geometry: lambda* = 0.25, uniform grid with halo = default_halo(N, 0.25),
    gamma = lambda*/h. gamma and centers are frozen; only the readout (v, bias)
    trains, from random (Xavier-uniform) init.
  - Phi is constant (geometry frozen), precomputed once; train on pred = Phi@v+bias.
  - Optimizer: Adam, peak LR with a short warmup then cosine decay. Compared to
    the lstsq global-optimum baseline. (All fp64, full-batch MSE.)

This is deliberately simple: one optimizer against the exact solve. The earlier
version's per-cell SGD divergence-threshold tuning, 200k-step budget, lr-sweep
side study, and full violation-diagnostic SVDs were removed as over-engineering.

Two figures (all error on the eval set):
  1. error_vs_width.png       -- 3 targets (rows) x {rel L2, L_inf} (cols).
  2. convergence_<target>.png -- 4 widths (rows) x {rel L2, L_inf} (cols).

Usage:
    python experiments/exp12_geometry_ladder/run.py            # collect + plot
    python experiments/exp12_geometry_ladder/run.py --plot     # plot from saved data
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

RESULTS_DIR = REPO_ROOT / "results" / "exp12_geometry_ladder"
DATA_PATH = RESULTS_DIR / "phase1_data.json"

# --- experiment grid ---
TARGETS = ["sine", "sine_8pi", "runge"]
WIDTHS = [32, 64, 128, 256]
SEEDS = [0, 1, 2]
LAMBDA_STAR = 0.25            # exp02's shared lstsq optimum; the "correct regime"
N_TRAIN = 1024
N_EVAL = 4096

# --- Adam schedule: short warmup -> peak -> cosine decay to END_FRAC * peak ---
TOTAL_STEPS = 50000
WARMUP = 1000
END_FRAC = 1e-6
ADAM_PEAK = 1e-1

ADAM_COLOR = "#ff7f0e"
LSTSQ_COLOR = "#000000"


def build_geometry(N):
    """Frozen QI geometry: gamma = lambda*/h and the halo'd uniform grid.

    Only the geometry is needed (not the cardinal coefficients), so we build it
    directly -- exact, and avoids the fp64 Toeplitz ill-conditioning the
    cardinal-coefficient solve hits at lambda=0.25.
    """
    h = 2.0 / N
    halo = default_halo(N, lambda_star=LAMBDA_STAR)
    n_idx = np.arange(-halo, N + halo + 1)
    centers = -1.0 + n_idx.astype(np.float64) * h          # x_n = -1 + n*h
    gamma = LAMBDA_STAR / h                                  # gamma = lambda*/h
    return gamma, centers, h, halo


def lr_at(step):
    """Linear warmup to ADAM_PEAK, then cosine decay to END_FRAC * peak."""
    end = END_FRAC * ADAM_PEAK
    if step < WARMUP:
        return ADAM_PEAK * (step + 1) / WARMUP
    progress = min(1.0, (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP))
    return end + 0.5 * (ADAM_PEAK - end) * (1.0 + math.cos(math.pi * progress))


def make_eval_steps(total, k=160):
    """Log-spaced step indices to evaluate at (+ a few early linear points), so
    per-seed traces share an x-axis and can be averaged."""
    pts = np.unique(np.geomspace(1, total, k).astype(int))
    early = np.arange(1, 21)
    return {int(p) for p in set(pts.tolist()) | set(early.tolist())}


def xavier_readout(neurons, seed):
    """Xavier-uniform readout init, bias = 0."""
    g = torch.Generator().manual_seed(seed)
    bound = float(np.sqrt(6.0 / (neurons + 1)))
    v = (torch.rand(neurons, generator=g, dtype=torch.float64) * 2 - 1) * bound
    b = torch.zeros(1, dtype=torch.float64)
    return v, b


def train_adam(Phi_tr, y_tr, Phi_ev, y_ev, y_norm, gamma, h, seed):
    """Train the readout (v, bias) with Adam + warmup/cosine. Returns traces+final."""
    neurons = Phi_tr.shape[1]
    v0, b0 = xavier_readout(neurons, seed)
    v = v0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    opt = torch.optim.Adam([v, b], lr=ADAM_PEAK)

    eval_set = make_eval_steps(TOTAL_STEPS)
    steps_log, linf_log, rl2_log = [], [], []

    def evaluate():
        with torch.no_grad():
            resid = Phi_ev @ v + b - y_ev
            return float(resid.abs().max()), float(torch.linalg.norm(resid) / y_norm)

    l, r = evaluate()
    steps_log.append(0); linf_log.append(l); rl2_log.append(r)

    for step in range(TOTAL_STEPS):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss = ((Phi_tr @ v + b - y_tr) ** 2).mean()
        loss.backward()
        opt.step()
        gstep = step + 1
        if gstep in eval_set:
            li, rl = evaluate()
            steps_log.append(gstep); linf_log.append(li); rl2_log.append(rl)

    v_np = v.detach().cpu().numpy()
    final = {"gamma": float(gamma), "lambda": float(gamma * h),
             "max_v": float(np.abs(v_np).max()), "v_l2": float(np.linalg.norm(v_np))}
    return {"steps": steps_log, "linf": linf_log, "rel_l2": rl2_log, "final": final}


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

            # --- lstsq baseline (seed-independent global optimum) ---
            v_ls, b_ls, info = solve_readout_with_bias(Phi_tr_np, y_train_np, method="lstsq")
            err_ls = np.abs(Phi_ev_np @ v_ls + b_ls - y_eval_np)
            ls_linf = float(err_ls.max())
            ls_rl2 = float(np.linalg.norm(err_ls) / y_norm)
            lstsq_rows.append({
                "target": target_name, "N": N, "neurons": neurons,
                "linf": ls_linf, "rel_l2": ls_rl2,
                "rank": info.get("rank"), "max_v": float(np.abs(v_ls).max()),
            })

            Phi_tr = torch.tensor(Phi_tr_np); Phi_ev = torch.tensor(Phi_ev_np)
            y_tr = torch.tensor(y_train_np); y_ev = torch.tensor(y_eval_np)
            y_norm_t = torch.linalg.norm(y_ev)

            finals = []
            for seed in SEEDS:
                res = train_adam(Phi_tr, y_tr, Phi_ev, y_ev, y_norm_t, gamma, h, seed)
                finals.append(res["linf"][-1])
                trained.append({
                    "target": target_name, "N": N, "neurons": neurons, "seed": seed,
                    "steps": res["steps"], "linf": res["linf"],
                    "rel_l2": res["rel_l2"], "final": res["final"],
                })

            print(f"[{cell}/{total_cells}] {target_name} N={N} "
                  f"(neurons={neurons}, halo={halo}) | lstsq Linf={ls_linf:.2e} | "
                  f"adam Linf={min(finals):.2e} | {time.time()-t0:.0f}s")

    out = {
        "config": {
            "targets": TARGETS, "widths": WIDTHS, "seeds": SEEDS,
            "lambda_star": LAMBDA_STAR, "n_train": N_TRAIN, "n_eval": N_EVAL,
            "total_steps": TOTAL_STEPS, "warmup": WARMUP, "end_frac": END_FRAC,
            "adam_peak": ADAM_PEAK,
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
        arr = np.array([r[key] for r in records])
        out[key] = (np.array(steps), arr.mean(0), arr.min(0), arr.max(0))
    return out


def plot_error_vs_width(data):
    trained, lstsq = data["trained"], data["lstsq"]
    metrics = [("rel_l2", r"Relative $L_2$ error"), ("linf", r"$L_\infty$ error")]
    nrows = len(TARGETS)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4.2 * nrows), sharex=True)
    fig.suptitle("Exp12 Phase 1: final eval error vs width (frozen QI geometry, "
                 r"$\lambda^*=0.25$; Adam vs lstsq)", fontsize=14, y=0.995)

    for ri, target_name in enumerate(TARGETS):
        for ci, (mkey, mlabel) in enumerate(metrics):
            ax = axes[ri][ci] if nrows > 1 else axes[ci]
            means, los, his = [], [], []
            for N in WIDTHS:
                recs = [r for r in trained if r["target"] == target_name and r["N"] == N]
                finals = np.array([r[mkey][-1] for r in recs])
                means.append(finals.mean()); los.append(finals.min()); his.append(finals.max())
            ax.fill_between(WIDTHS, los, his, color=ADAM_COLOR, alpha=0.18)
            ax.semilogy(WIDTHS, means, "-o", color=ADAM_COLOR, markersize=4, label="Adam")
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
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10,
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
    fig.suptitle(f"Exp12 Phase 1: convergence vs step -- {target_name} "
                 r"(frozen QI geometry, $\lambda^*=0.25$; Adam)", fontsize=14, y=0.997)

    for ri, N in enumerate(WIDTHS):
        for ci, (mkey, mlabel) in enumerate(metrics):
            ax = axes[ri][ci]
            recs = [r for r in trained if r["target"] == target_name and r["N"] == N]
            agg = _agg(recs)
            steps, mean, lo, hi = agg[mkey]
            xs = np.maximum(steps, 1)
            ax.fill_between(xs, lo, hi, color=ADAM_COLOR, alpha=0.18)
            ax.loglog(xs, mean, "-", color=ADAM_COLOR, linewidth=1.4, label="Adam")
            ls = next(r for r in lstsq if r["target"] == target_name and r["N"] == N)[mkey]
            ax.axhline(ls, ls="--", color=LSTSQ_COLOR, linewidth=1.2, label="lstsq")
            ax.grid(True, alpha=0.3, which="both")
            ax.set_ylabel(f"N={N}\n{mlabel}" if ci == 0 else mlabel)
            if ri == 0:
                ax.set_title(mlabel)
            if ri == nrows - 1:
                ax.set_xlabel("optimization step")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10,
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
    if "--plot" in sys.argv:
        if not DATA_PATH.exists():
            print(f"No data at {DATA_PATH}. Run without --plot first.")
            sys.exit(1)
        with open(DATA_PATH) as f:
            data = json.load(f)
    else:
        data = collect_data()
    plot_all(data)
