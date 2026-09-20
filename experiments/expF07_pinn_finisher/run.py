"""Experiment expF07 -- lstsq precision finisher for a trained PINN.

Train a vanilla torch tanh-MLP PINN (4x64, Adam) on the expF06 Burgers
problem (nu=0.1) to its plateau, freeze it, then run the expF06 Newton-lstsq
loop warm-started at the PINN (base_fields). Full: 50k Adam steps, polish
W=1024, 6 Newton steps. Smoke: 400 steps, W=256, 2 steps.

Known mechanism (from the smoke-scale test): the ridge correction must
represent u* - PINN, so the achievable floor scales with the magnitude and
roughness of the PINN's error field ("representation ceiling").

Outputs (results/checkpoint_F_applications/expF07_pinn_finisher/):
  finisher_convergence.png  rel L2(u): Adam curve then polish steps
  pinn_ckpt.pt              trained PINN state_dict
  data.json                 training history + polish history + wall clocks

Usage:
  uv run --extra dev python experiments/expF07_pinn_finisher/run.py [--smoke] [--plot]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF05_spline_ridge"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF06_newton_burgers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pinn
import newton as nt

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF07_pinn_finisher"
DATA_PATH = RESULTS_DIR / "data.json"
NU = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot()
        return
    steps, W, n_polish = (400, 256, 2) if args.smoke else (50000, 1024, 6)
    t0 = time.time()
    net, train_hist = pinn.train_pinn(nu=NU, steps=steps, seed=0)
    t_train = time.time() - t0
    torch.save(net.state_dict(), RESULTS_DIR / "pinn_ckpt.pt")
    t0 = time.time()
    res = nt.newton_burgers(nu=NU, W=W, lam=0.25, max_iter=n_polish, seed=0,
                            base_fields=pinn.pinn_fields(net))
    t_polish = time.time() - t0
    data = dict(nu=NU, steps=steps, W=W, t_train_s=t_train, t_polish_s=t_polish,
                train_history=train_hist, polish_history=res["history"])
    DATA_PATH.write_text(json.dumps(data, indent=1))
    print(f"train {t_train:.0f}s -> rel_l2 {train_hist[-1]['rel_l2_u']:.2e}; "
          f"polish {t_polish:.0f}s -> rel_l2 {res['history'][-1]['rel_l2_u']:.2e}",
          flush=True)
    plot()


def plot():
    data = json.loads(DATA_PATH.read_text())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    th = data["train_history"]
    ax.semilogy([h["step"] for h in th], [h["rel_l2_u"] for h in th],
                "-", label=f"Adam ({data['t_train_s']:.0f}s)")
    last_step = th[-1]["step"]
    ph = data["polish_history"]
    steps = [last_step + (i + 1) * max(1, last_step // 20) for i in range(len(ph))]
    ax.semilogy(steps, [h["rel_l2_u"] for h in ph], "o-",
                label=f"Newton-lstsq polish ({data['t_polish_s']:.0f}s)")
    ax.set_xlabel("Adam step (polish appended)")
    ax.set_ylabel("rel L2(u)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "finisher_convergence.png", dpi=140)
    print("plot saved to", RESULTS_DIR)


if __name__ == "__main__":
    main()
