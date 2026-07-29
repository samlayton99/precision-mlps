"""expD07 optimizer registry. Append configs here; run.py picks up new entries
and runs only the missing (problem, optimizer) pairs (runs.jsonl is the cache).

lr semantics:
  family "sgd":  lr = lr_rel / L with L = 2 smax^2 / R (grad-Lipschitz constant
                 of the mean-MSE objective). lr_rel = 1.0 is half the 2/L
                 stability limit -- safe for plain GD and heavy ball (mom 0.9).
  family "adam": lr is absolute (per-coordinate scale-adaptive), UNLESS
                 lr_mode == "eps_coupled": lr = eps * (1 + beta1) / L, which
                 enforces lr/eps < 2(1+beta1)/L so the worst-case update
                 (denominator -> eps) is a stable preconditioned heavy-ball
                 step (Sam's Impl A).
  families "nesterov" / "bb" / "cgls" / "lbfgs" are custom loops in run.py
                 (no torch.optim object); their specs carry family-specific
                 knobs only.

Every run is full-batch from v0 = 0, so one grad eval per step and runs are
deterministic -- no seed axis. Budgets count grad evals: 1 per step for the
gradient methods, 1 per CGLS iteration (2 matvecs, same cost as a grad), and
1 per closure call for L-BFGS (line searches are charged fairly).
"""
from __future__ import annotations

import torch

OPTIMIZERS = [
    {"id": "sgd", "label": "SGD (lr=1/L)",
     "family": "sgd", "lr_rel": 1.0, "momentum": 0.0},
    {"id": "sgd_mom09", "label": "SGD+mom 0.9 (lr=1/L)",
     "family": "sgd", "lr_rel": 1.0, "momentum": 0.9},
    {"id": "adam_lr1e-3", "label": "Adam (lr=1e-3)",
     "family": "adam", "lr": 1e-3},
    # --- round 2: Adam-fix hypothesis (Impl A/B) + acceleration + matched methods ---
    {"id": "adam_epscoupled", "label": "Adam eps-coupled (e=1e-3)",
     "family": "adam", "eps": 1e-3, "lr_mode": "eps_coupled"},
    {"id": "amsgrad_lr1e-3", "label": "AMSGrad (lr=1e-3)",
     "family": "adam", "lr": 1e-3, "amsgrad": True},
    {"id": "nesterov", "label": "Nesterov (convex, lr=1/L)",
     "family": "nesterov", "lr_rel": 1.0},
    {"id": "gd_bb", "label": "GD + BB step",
     "family": "bb"},
    {"id": "cgls", "label": "CGLS (Krylov)",
     "family": "cgls"},
    {"id": "lbfgs", "label": "L-BFGS (Wolfe, m=100)",
     "family": "lbfgs", "history": 100, "max_iter": 50},
    # --- round 3: DL-compatible spectral methods + the iterative ceiling ---
    {"id": "sgd_mom099", "label": "SGD+mom 0.99 (lr=1/L)",
     "family": "sgd", "lr_rel": 1.0, "momentum": 0.99},
    {"id": "sgd_mom0999", "label": "SGD+mom 0.999 (lr=1/L)",
     "family": "sgd", "lr_rel": 1.0, "momentum": 0.999},
    {"id": "nlcg_pr", "label": "NLCG Polak-Ribiere (2 evals/it)",
     "family": "nlcg"},
    {"id": "bb_rms", "label": "BB + frozen RMS diag",
     "family": "bb_rms", "warmup": 100},
    {"id": "lsqr", "label": "LSQR (Golub-Kahan)",
     "family": "lsqr"},
    # --- round 4: fp64 orthogonality-loss fixes (rank ~ 277 << 20k iters,
    #     so the Krylov stall is roundoff, not iteration count) ---
    {"id": "nlcg_restart", "label": "NLCG-PR + restart/refresh 250",
     "family": "nlcg", "restart": 250},
    {"id": "cgls_refine", "label": "CGLS + refine restarts 300",
     "family": "cgls_refine", "cycle": 300},
    {"id": "cgls_reortho", "label": "CGLS reortho (2 evals/it)",
     "family": "cgls_reortho"},
]


def spec_by_id() -> dict[str, dict]:
    return {s["id"]: s for s in OPTIMIZERS}


def make_optimizer(spec: dict, params, lipschitz: float):
    if spec["family"] == "sgd":
        return torch.optim.SGD(params, lr=spec["lr_rel"] / lipschitz,
                               momentum=spec.get("momentum", 0.0))
    if spec["family"] == "adam":
        betas = spec.get("betas", (0.9, 0.999))
        if spec.get("lr_mode") == "eps_coupled":
            lr = spec["eps"] * (1.0 + betas[0]) / lipschitz
        else:
            lr = spec["lr"]
        return torch.optim.Adam(params, lr=lr, eps=spec.get("eps", 1e-8),
                                betas=betas,
                                amsgrad=spec.get("amsgrad", False))
    raise ValueError(f"No torch optimizer for family: {spec['family']}")
