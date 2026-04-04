"""Demonstrate that mpmath construction reaches machine epsilon.

Compares fp64 construction (lambda=0.30, fast) vs mpmath construction
(lambda=0.25, machine-precision). Outputs:
  - results/setup/qi_machine_eps.png
  - results/setup/qi_machine_eps_data.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.construction.qi_mpmath import construct_qi, evaluate_qi
from src.data.targets import get_target

MACHINE_EPS = 2.22e-16
N_EVAL = 4001

# Widths to test
WIDTHS = [16, 32, 48, 64, 96, 128]

# Four targets that should reach machine eps
TARGETS = ["sine", "cosine", "exp", "runge"]

# Two parameter configurations
FP64_PARAMS = dict(lambda_star=0.30, Kc=120, halo=50)
MPMATH_PARAMS = dict(lambda_star=0.25, Kc=160, halo=80, use_mpmath=True, mp_dps=30)


def sweep(name: str, params: dict, label: str) -> dict:
    target = get_target(name)
    x_eval = np.linspace(-1, 1, N_EVAL)
    y_true = target.fn_numpy(x_eval)
    results = {"widths": [], "linf": [], "time_s": []}

    for N in WIDTHS:
        t0 = time.time()
        qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=N, **params)
        t1 = time.time()
        y_pred = evaluate_qi(qi, x_eval)
        err = float(np.max(np.abs(y_pred - y_true)))
        W = len(qi.centers)
        results["widths"].append(W)
        results["linf"].append(err)
        results["time_s"].append(t1 - t0)
        print(f"    [{label}] N={N:3d} W={W:4d}: L_inf={err:.3e}, t={t1-t0:.1f}s")

    return results


def main():
    out_dir = Path("results/setup")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for name in TARGETS:
        print(f"\n=== {name} ===")
        print("  fp64 (lambda=0.30):")
        fp64_res = sweep(name, FP64_PARAMS, "fp64")
        print("  mpmath (lambda=0.25, mp_dps=30):")
        mp_res = sweep(name, MPMATH_PARAMS, "mpmath")
        all_data[name] = {"fp64": fp64_res, "mpmath": mp_res}

    with open(out_dir / "qi_machine_eps_data.json", "w") as f:
        json.dump(all_data, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, name in zip(axes, TARGETS):
        d = all_data[name]
        ax.semilogy(d["fp64"]["widths"], d["fp64"]["linf"], "o-",
                    label="fp64 (lambda=0.30)", linewidth=1.5, markersize=7)
        ax.semilogy(d["mpmath"]["widths"], d["mpmath"]["linf"], "s-",
                    label="mpmath (lambda=0.25)", linewidth=1.5, markersize=7)
        ax.axhline(MACHINE_EPS, color="red", linestyle="--", linewidth=1,
                   label=f"fp64 eps ({MACHINE_EPS:.2e})")
        ax.set_title(f"{name}", fontsize=13)
        ax.set_xlabel("width W")
        ax.set_ylabel("L_inf error")
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    plt.suptitle("QI construction: fp64 vs mpmath paths", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "qi_machine_eps.png", dpi=120)
    plt.close()
    print(f"\nSaved: {out_dir / 'qi_machine_eps.png'}")

    # Summary
    print("\n=== Summary (best L_inf) ===")
    print(f"{'target':<10s}{'fp64':>14s}{'mpmath':>14s}{'ratio':>10s}")
    for name in TARGETS:
        d = all_data[name]
        best_f = min(d["fp64"]["linf"])
        best_m = min(d["mpmath"]["linf"])
        print(f"{name:<10s}{best_f:>14.3e}{best_m:>14.3e}{best_f/best_m:>10.1f}x")


if __name__ == "__main__":
    main()
