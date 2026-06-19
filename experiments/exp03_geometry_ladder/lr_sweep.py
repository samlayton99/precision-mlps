"""Experiment 3 -- Phase 1 side study: learning-rate sweep.

A short, single-cell study isolating the effect of the (constant) learning rate
on the readout-only training trajectory, with the QI geometry frozen in the
correct regime (lambda* = 0.25). One optimizer (Adam by default), constant LR,
no schedule -- so the only thing varying is the learning rate.

Produces ONE figure (lr_sweep.png): eval error vs optimization step, one colored
line per learning rate, each the geometric mean over 5 seeds with a 95% CI band.
The lstsq optimum is drawn as a dashed reference.

Usage:
    python experiments/exp03_geometry_ladder/lr_sweep.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy import stats

torch.set_default_dtype(torch.float64)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.readout import build_phi, solve_readout_with_bias
from src.data.targets import get_target
from run import build_geometry, xavier_readout, N_TRAIN, N_EVAL, LAMBDA_STAR  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "exp03_geometry_ladder"

# --- study config (one cell, short run) ---
OPT = "adam"
TARGET = "sine"
N = 64
SEEDS = [0, 1, 2, 3, 4]
STEPS = 10000                                  # short
LRS = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1]           # constant LRs to sweep


def eval_steps(total, k=120):
    pts = np.unique(np.geomspace(1, total, k).astype(int))
    return sorted(set(pts.tolist()) | set(range(1, 21)))


def train_const_lr(lr, Phi_tr, y_tr, Phi_ev, y_ev, y_norm, seed, eval_set):
    """Constant-LR training; returns (steps, linf[], rel_l2[])."""
    neurons = Phi_tr.shape[1]
    v0, b0 = xavier_readout(neurons, seed)
    v = v0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    opt = (torch.optim.Adam if OPT == "adam" else torch.optim.SGD)([v, b], lr=lr)
    es = set(eval_set)

    def ev():
        with torch.no_grad():
            resid = Phi_ev @ v + b - y_ev
            return float(resid.abs().max()), float(torch.linalg.norm(resid) / y_norm)

    steps, linf, rl2 = [0], [], []
    li, rl = ev(); linf.append(li); rl2.append(rl)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = ((Phi_tr @ v + b - y_tr) ** 2).mean()
        loss.backward()
        opt.step()
        if (step + 1) in es:
            li, rl = ev(); steps.append(step + 1); linf.append(li); rl2.append(rl)
    return steps, linf, rl2


def collect():
    target = get_target(TARGET)
    gamma, centers, h, _halo = build_geometry(N)
    neurons = len(centers)
    gvec = np.full(neurons, gamma)
    x_train = np.linspace(-1, 1, N_TRAIN); x_eval = np.linspace(-1, 1, N_EVAL)
    y_train = target.fn_numpy(x_train); y_eval = target.fn_numpy(x_eval)
    y_norm = float(np.linalg.norm(y_eval))
    Phi_tr_np = build_phi(x_train, gvec, centers); Phi_ev_np = build_phi(x_eval, gvec, centers)

    # lstsq reference
    v_ls, b_ls, _ = solve_readout_with_bias(Phi_tr_np, y_train, method="lstsq")
    err = np.abs(Phi_ev_np @ v_ls + b_ls - y_eval)
    lstsq = {"linf": float(err.max()), "rel_l2": float(np.linalg.norm(err) / y_norm)}

    Phi_tr = torch.tensor(Phi_tr_np); Phi_ev = torch.tensor(Phi_ev_np)
    y_tr = torch.tensor(y_train); y_ev = torch.tensor(y_eval)
    y_norm_t = torch.linalg.norm(y_ev)
    es = eval_steps(STEPS)

    runs = {}
    for lr in LRS:
        per_seed = {"linf": [], "rel_l2": []}
        steps = None
        for seed in SEEDS:
            steps, linf, rl2 = train_const_lr(lr, Phi_tr, y_tr, Phi_ev, y_ev,
                                              y_norm_t, seed, es)
            per_seed["linf"].append(linf); per_seed["rel_l2"].append(rl2)
        runs[lr] = {"steps": steps, "linf": per_seed["linf"], "rel_l2": per_seed["rel_l2"]}
        fin = np.array(per_seed["linf"])[:, -1]
        print(f"lr={lr:.0e}: final Linf geomean={10**np.mean(np.log10(fin)):.2e} "
              f"(min={fin.min():.2e}, max={fin.max():.2e})")
    return {"runs": runs, "lstsq": lstsq, "cell": {"target": TARGET, "N": N,
            "neurons": neurons, "opt": OPT, "steps": STEPS, "lambda_star": LAMBDA_STAR}}


def plot(data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs, lstsq = data["runs"], data["lstsq"]
    cell = data["cell"]
    tval = stats.t.ppf(0.975, len(SEEDS) - 1)   # 95% CI, t-dist with n-1 dof
    cmap = plt.get_cmap("viridis")
    colors = {lr: cmap(i / max(1, len(LRS) - 1)) for i, lr in enumerate(LRS)}

    metrics = [("rel_l2", r"Relative $L_2$ error"), ("linf", r"$L_\infty$ error")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(f"LR sweep ({cell['opt']}, constant LR) -- {cell['target']} "
                 f"N={cell['N']} (frozen QI geometry, "
                 r"$\lambda^*=0.25$); mean $\pm$ 95% CI over "
                 f"{len(SEEDS)} seeds", fontsize=13)

    for ax, (mkey, mlabel) in zip(axes, metrics):
        for lr in LRS:
            r = runs[lr]
            steps = np.maximum(np.array(r["steps"]), 1)
            arr = np.array(r[mkey])                    # [n_seeds, n_steps]
            logs = np.log10(np.clip(arr, 1e-300, None))
            m = logs.mean(0)
            sem = logs.std(0, ddof=1) / np.sqrt(arr.shape[0])
            lo = 10 ** (m - tval * sem); hi = 10 ** (m + tval * sem)
            ax.fill_between(steps, lo, hi, color=colors[lr], alpha=0.20)
            ax.loglog(steps, 10 ** m, "-", color=colors[lr], linewidth=1.6,
                      label=f"lr={lr:.0e}")
        ax.axhline(lstsq[mkey], ls="--", color="k", linewidth=1.2, label="lstsq")
        ax.set_xlabel("optimization step"); ax.set_ylabel(mlabel)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULTS_DIR / "lr_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    data = collect()
    with open(RESULTS_DIR / "lr_sweep_data.json", "w") as f:
        json.dump({k: v for k, v in data.items() if k != "runs"}, f)  # light metadata
    plot(data)
