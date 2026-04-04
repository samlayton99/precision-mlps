"""Sweep QI construction over widths to verify machine-epsilon convergence.

Produces:
  - results/setup/qi_convergence_all.png    Grid of all target curves
  - results/setup/qi_convergence_<name>.png Per-target curves
  - results/setup/qi_convergence_data.json  Raw error data
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.construction.qi_mpmath import construct_qi, evaluate_qi
from src.data.targets import get_target


# Sweep configuration
WIDTHS = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
LAMBDA_STAR = 0.30
KC = 120
HALO = 50
N_EVAL = 8001

TARGETS = [
    "sine", "cosine", "exp", "poly5",
    "runge", "sine_8pi", "sine_mixture", "tanh_steep",
]

MACHINE_EPS = 2.22e-16
SUCCESS_THRESHOLD = 1e-13  # "machine-epsilon precision" for our purposes
STAGNATION_TOL = 0.5       # log10 improvement below this = stagnant


def sweep_target(name: str, widths: list[int]) -> dict:
    """Run QI construction at each width; stop early if converged/stagnant."""
    target = get_target(name)
    x_eval = np.linspace(-1, 1, N_EVAL)
    y_true = target.fn_numpy(x_eval)
    y_norm = float(np.linalg.norm(y_true))

    results = {"name": name, "widths": [], "N_values": [],
               "linf": [], "linf_kahan": [],
               "rel_l2": [], "rel_l2_kahan": [],
               "time_s": [], "toeplitz_resid": []}
    best_err = np.inf
    stagnant_count = 0

    for N in widths:
        t0 = time.time()
        qi = construct_qi(
            target.fn_numpy, target.deriv_numpy, N=N,
            lambda_star=LAMBDA_STAR, Kc=KC, halo=HALO,
        )
        t1 = time.time()

        # Naive summation
        y_pred = evaluate_qi(qi, x_eval, kahan=False)
        linf = float(np.max(np.abs(y_pred - y_true)))
        rel_l2 = float(np.linalg.norm(y_pred - y_true) / y_norm)

        # Kahan summation
        y_pred_k = evaluate_qi(qi, x_eval, kahan=True)
        linf_k = float(np.max(np.abs(y_pred_k - y_true)))
        rel_l2_k = float(np.linalg.norm(y_pred_k - y_true) / y_norm)

        W = len(qi.centers)

        results["widths"].append(W)
        results["N_values"].append(N)
        results["linf"].append(linf)
        results["linf_kahan"].append(linf_k)
        results["rel_l2"].append(rel_l2)
        results["rel_l2_kahan"].append(rel_l2_k)
        results["time_s"].append(t1 - t0)
        results["toeplitz_resid"].append(qi.toeplitz_residual)

        print(f"  N={N:5d} (W={W:5d}): L_inf={linf:.3e} (kahan={linf_k:.3e}), "
              f"rel_L2={rel_l2:.3e}, t={t1-t0:.2f}s")

        # Early stopping based on Kahan error (the true lower bound)
        if linf_k < SUCCESS_THRESHOLD:
            print(f"  -> Reached target precision (<{SUCCESS_THRESHOLD:.0e})")
            break
        if best_err < np.inf:
            improvement = math.log10(best_err) - math.log10(max(linf_k, 1e-17))
            if improvement < STAGNATION_TOL:
                stagnant_count += 1
                if stagnant_count >= 2:
                    print(f"  -> Stagnant (2 steps with <{STAGNATION_TOL} OOM improvement)")
                    break
            else:
                stagnant_count = 0
        best_err = min(best_err, linf_k)

    return results


def plot_all(all_results: list[dict], out_dir: Path) -> None:
    """Plot a grid of per-target convergence curves."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, res in zip(axes, all_results):
        widths = res["widths"]
        ax.semilogy(widths, res["linf"], "o-", linewidth=1.5, markersize=6,
                    label="naive sum")
        ax.semilogy(widths, res["linf_kahan"], "s--", linewidth=1.5, markersize=5,
                    label="Kahan sum", alpha=0.8)
        ax.axhline(MACHINE_EPS, color="red", linestyle="--", linewidth=1,
                   label=f"fp64 eps")
        ax.axhline(SUCCESS_THRESHOLD, color="orange", linestyle=":", linewidth=1,
                   label="1e-13")
        ax.set_title(f"{res['name']}", fontsize=12)
        ax.set_xlabel("width W = N + 2R + 1")
        ax.set_ylabel("L_inf error")
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    plt.suptitle(
        f"QI construction convergence (lambda={LAMBDA_STAR}, Kc={KC}, halo={HALO}, fp64)",
        fontsize=14,
    )
    plt.tight_layout()
    path = out_dir / "qi_convergence_all.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_combined(all_results: list[dict], out_dir: Path) -> None:
    """One plot, all targets overlaid (Kahan-sum errors)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for res in all_results:
        ax.semilogy(res["widths"], res["linf_kahan"], "o-", label=res["name"],
                    linewidth=1.5, markersize=5)
    ax.axhline(MACHINE_EPS, color="red", linestyle="--", linewidth=1.5,
               label=f"fp64 eps")
    ax.axhline(SUCCESS_THRESHOLD, color="orange", linestyle=":", linewidth=1.5,
               label="target (1e-13)")
    ax.set_xlabel("width W = N + 2R + 1")
    ax.set_ylabel("L_inf error")
    ax.set_xscale("log")
    ax.set_title(
        f"QI construction: L_inf vs width "
        f"(lambda={LAMBDA_STAR}, Kc={KC}, halo={HALO})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    path = out_dir / "qi_convergence_combined.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def main():
    out_dir = Path("results/setup")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweeping widths {WIDTHS} on {len(TARGETS)} targets...")
    print(f"Parameters: lambda={LAMBDA_STAR}, Kc={KC}, halo={HALO}")
    print()

    all_results = []
    for name in TARGETS:
        print(f"[{name}]")
        res = sweep_target(name, WIDTHS)
        all_results.append(res)
        print()

    # Save raw data
    data_path = out_dir / "qi_convergence_data.json"
    with open(data_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {data_path}")

    # Plot
    plot_all(all_results, out_dir)
    plot_combined(all_results, out_dir)

    # Summary
    print("\n=== Summary (Kahan-sum evaluator) ===")
    print(f"{'target':<15s}{'best L_inf':>14s}{'at width':>10s}{'verdict':>20s}")
    for res in all_results:
        best_i = int(np.argmin(res["linf_kahan"]))
        best_err = res["linf_kahan"][best_i]
        best_w = res["widths"][best_i]
        if best_err < 1e-14:
            verdict = "machine eps"
        elif best_err < SUCCESS_THRESHOLD:
            verdict = "reached 1e-13"
        elif best_err < 1e-11:
            verdict = "near eps"
        elif best_err < 1e-8:
            verdict = "converging"
        else:
            verdict = "slow"
        print(f"{res['name']:<15s}{best_err:>14.3e}{best_w:>10d}{verdict:>20s}")


if __name__ == "__main__":
    main()
