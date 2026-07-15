"""Experiment expF03 -- steady 2D Burgers via Newton-lstsq (solve, don't train).

Grid: nu in {0.1, 0.01} x W in {256, 576, 1024, 2304} (smoke: nu=0.1, W=256),
max 12 Newton iterations, tanh family, lam=0.25.

Outputs (results/checkpoint_F_applications/expF03_newton_burgers/):
  newton_convergence.png   res_norm + rel_l2(u) vs iteration, best W per nu
  error_vs_width.png       final rel_l2(u) vs W per nu
  data.json                every (nu, W) cell with its full Newton history

Usage:
  uv run --extra dev python experiments/expF03_newton_burgers/run.py [--smoke] [--plot]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF02_spline_ridge"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import newton as nt

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF03_newton_burgers"
DATA_PATH = RESULTS_DIR / "data.json"
LAM = 0.25


def load_data():
    if DATA_PATH.exists():
        return json.loads(DATA_PATH.read_text())
    return []


def save_data(data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(data, indent=1))


def sweep(smoke):
    nus = [0.1] if smoke else [0.1, 0.01]
    ws = [256] if smoke else [256, 576, 1024, 2304]
    data = load_data()
    done = {(c["nu"], c["W"]) for c in data}
    for nu in nus:
        for W in ws:
            if (nu, W) in done:
                continue
            print(f"=== nu={nu} W={W} ===", flush=True)
            res = nt.newton_burgers(nu=nu, W=W, lam=LAM,
                                    max_iter=6 if smoke else 12)
            data.append(dict(nu=nu, W=W, lam=LAM, history=res["history"]))
            save_data(data)


def plot():
    data = load_data()
    if not data:
        return
    nus = sorted({c["nu"] for c in data}, reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for nu in nus:
        cells = [c for c in data if c["nu"] == nu]
        best = max(cells, key=lambda c: c["W"])
        it = [h["iter"] for h in best["history"]]
        axes[0].semilogy(it, [h["res_norm"] for h in best["history"]], "o-",
                         label=f"nu={nu} W={best['W']} residual")
        axes[0].semilogy(it, [h["rel_l2_u"] for h in best["history"]], "s--",
                         label=f"nu={nu} rel L2(u)")
        ws = sorted({c["W"] for c in cells})
        finals = [min(c["history"][-1]["rel_l2_u"] for c in cells if c["W"] == w)
                  for w in ws]
        axes[1].loglog(ws, finals, "o-", label=f"nu={nu}")
    axes[0].set_xlabel("Newton iteration")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].set_xlabel("W")
    axes[1].set_ylabel("final rel L2(u)")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "newton_convergence.png", dpi=140)
    print("plots saved to", RESULTS_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.plot:
        sweep(args.smoke)
    plot()


if __name__ == "__main__":
    main()
