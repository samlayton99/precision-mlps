"""Experiment 0A: QI vs Learned Readout -- fair 4-way comparison.

Compares QI construction vs least-squares readout, each in both fp64 and
mpmath, on identical QI geometry. Isolates whether the QI convolution
formula is fundamentally better than direct solve.

Four methods at each (target, N, lambda):
  1. QI mpmath   -- full construction in extended precision
  2. QI fp64     -- full construction in fp64
  3. lstsq mpmath -- SVD solve in extended precision on same geometry
  4. lstsq fp64   -- numpy lstsq on same geometry

Usage:
    python3 experiments/exp0A_QI_vs_learn/run.py           # collect + plot
    python3 experiments/exp0A_QI_vs_learn/run.py --plot     # plot only
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.qi_mpmath import construct_qi, evaluate_qi, default_halo
from src.construction.readout import build_phi, solve_readout_with_bias
from src.data.targets import get_target

RESULTS_DIR = REPO_ROOT / "results" / "exp0A_QI_vs_learn"
DATA_PATH = RESULTS_DIR / "data.json"

WIDTHS = [32, 64, 96, 128]
LAMBDA_SWEEP = [0.20, 0.25, 0.30]
TARGETS = ["sine", "runge", "exp", "sine_mixture"]
N_EVAL = 2048
KC = 160
MP_DPS = 50


def solve_readout_mpmath_svd(x_train, y_train, gamma, centers, mp_dps=50):
    """Solve least-squares readout in mpmath via truncated SVD.

    Builds Phi (with bias column) in mpmath, computes SVD, applies
    truncated pseudoinverse, rounds result to fp64.

    Returns (v_fp64, bias_fp64, info_dict).
    """
    mp.mp.dps = mp_dps
    n = len(x_train)
    w = len(centers)
    ncols = w + 1

    gamma_mp = mp.mpf(gamma)

    # Build augmented Phi in mpmath: [tanh(gamma*(x-c)), 1]
    Phi_aug = mp.matrix(n, ncols)
    for i in range(n):
        xi = mp.mpf(float(x_train[i]))
        for k in range(w):
            Phi_aug[i, k] = mp.tanh(gamma_mp * (xi - mp.mpf(float(centers[k]))))
        Phi_aug[i, ncols - 1] = mp.mpf(1)

    y_mp = mp.matrix(n, 1)
    for i in range(n):
        y_mp[i] = mp.mpf(float(y_train[i]))

    # SVD: Phi_aug = U @ diag(S) @ V
    U, S, V = mp.svd_r(Phi_aug, full_matrices=False)

    nsv = S.rows
    s_max = abs(S[0])

    # Truncation threshold: keep singular values above eps_mach * s_max
    # (we want the best fp64-representable solution)
    tol = mp.mpf(10) ** (-15) * s_max
    rank = sum(1 for i in range(nsv) if abs(S[i]) > tol)

    # Pseudoinverse: sol = V^T @ diag(1/s_i) @ U^T @ y
    Uty = U.T * y_mp

    coeffs = mp.matrix(nsv, 1)
    for i in range(nsv):
        if abs(S[i]) > tol:
            coeffs[i] = Uty[i] / S[i]
        else:
            coeffs[i] = mp.mpf(0)

    sol = V.T * coeffs

    v_fp64 = np.array([float(sol[k]) for k in range(w)], dtype=np.float64)
    bias_fp64 = float(sol[w])

    info = {
        "rank": rank,
        "ncols": ncols,
        "smax": float(s_max),
        "smin": float(abs(S[nsv - 1])),
    }
    return v_fp64, bias_fp64, info


def collect_data():
    """Run the 4-way comparison sweep and save results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_eval = np.linspace(-1, 1, N_EVAL)
    results = []

    t0 = time.time()
    total = len(TARGETS) * len(WIDTHS) * len(LAMBDA_SWEEP)
    done = 0

    for target_name in TARGETS:
        target = get_target(target_name)
        y_eval = target.fn_numpy(x_eval)
        y_norm = np.linalg.norm(y_eval)
        if y_norm == 0:
            y_norm = 1.0

        for N in WIDTHS:
            h = 2.0 / N

            for lam in LAMBDA_SWEEP:
                done += 1
                halo = default_halo(N, lambda_star=lam)

                # --- QI mpmath ---
                try:
                    qi_mp = construct_qi(
                        target.fn_numpy, target.deriv_numpy, N,
                        precision="mpmath", lambda_star=lam, Kc=KC,
                        halo=halo, mp_dps=MP_DPS, use_cache=True,
                    )
                    pred = evaluate_qi(qi_mp, x_eval)
                    err = np.abs(pred - y_eval)
                    qi_mp_linf = float(np.max(err))
                    qi_mp_rel_l2 = float(np.linalg.norm(err) / y_norm)
                except Exception as e:
                    qi_mp = None
                    qi_mp_linf = float("nan")
                    qi_mp_rel_l2 = float("nan")
                    print(f"  QI mpmath failed: {target_name} N={N} lam={lam}: {e}")

                # Use mpmath geometry as reference
                if qi_mp is not None:
                    centers = qi_mp.centers
                    gamma = qi_mp.gamma
                else:
                    n_idx = np.arange(-halo, N + halo + 1)
                    centers = -1.0 + n_idx * h
                    gamma = lam / h

                w = len(centers)
                gamma_vec = np.full(w, gamma)
                n_train = max(512, 2 * w)
                x_train = np.linspace(-1, 1, n_train)
                y_train = target.fn_numpy(x_train)

                # --- QI fp64 ---
                try:
                    qi_f64 = construct_qi(
                        target.fn_numpy, target.deriv_numpy, N,
                        precision="fp64", lambda_star=lam, Kc=KC,
                        halo=halo, use_cache=True,
                    )
                    pred = evaluate_qi(qi_f64, x_eval)
                    err = np.abs(pred - y_eval)
                    qi_f64_linf = float(np.max(err))
                    qi_f64_rel_l2 = float(np.linalg.norm(err) / y_norm)
                except Exception as e:
                    qi_f64_linf = float("nan")
                    qi_f64_rel_l2 = float("nan")
                    print(f"  QI fp64 failed: {target_name} N={N} lam={lam}: {e}")

                # --- Lstsq fp64 ---
                try:
                    Phi_train = build_phi(x_train, gamma_vec, centers)
                    v, bias, _ = solve_readout_with_bias(Phi_train, y_train, method="lstsq")
                    Phi_eval = build_phi(x_eval, gamma_vec, centers)
                    pred = Phi_eval @ v + bias
                    err = np.abs(pred - y_eval)
                    ls_f64_linf = float(np.max(err))
                    ls_f64_rel_l2 = float(np.linalg.norm(err) / y_norm)
                except Exception as e:
                    ls_f64_linf = float("nan")
                    ls_f64_rel_l2 = float("nan")
                    print(f"  Lstsq fp64 failed: {target_name} N={N} lam={lam}: {e}")

                # --- Lstsq mpmath SVD ---
                try:
                    t_solve = time.time()
                    v_mp, bias_mp, svd_info = solve_readout_mpmath_svd(
                        x_train, y_train, gamma, centers, mp_dps=MP_DPS,
                    )
                    solve_time = time.time() - t_solve
                    Phi_eval = build_phi(x_eval, gamma_vec, centers)
                    pred = Phi_eval @ v_mp + bias_mp
                    err = np.abs(pred - y_eval)
                    ls_mp_linf = float(np.max(err))
                    ls_mp_rel_l2 = float(np.linalg.norm(err) / y_norm)
                except Exception as e:
                    ls_mp_linf = float("nan")
                    ls_mp_rel_l2 = float("nan")
                    solve_time = 0.0
                    svd_info = {}
                    print(f"  Lstsq mpmath failed: {target_name} N={N} lam={lam}: {e}")

                results.append({
                    "target": target_name,
                    "N": N,
                    "lambda_star": lam,
                    "gamma": gamma,
                    "halo": halo,
                    "width": w,
                    "qi_mp_linf": qi_mp_linf,
                    "qi_mp_rel_l2": qi_mp_rel_l2,
                    "qi_f64_linf": qi_f64_linf,
                    "qi_f64_rel_l2": qi_f64_rel_l2,
                    "ls_f64_linf": ls_f64_linf,
                    "ls_f64_rel_l2": ls_f64_rel_l2,
                    "ls_mp_linf": ls_mp_linf,
                    "ls_mp_rel_l2": ls_mp_rel_l2,
                    "svd_rank": svd_info.get("rank", -1),
                    "svd_ncols": svd_info.get("ncols", -1),
                    "svd_solve_time": solve_time,
                })

                elapsed = time.time() - t0
                print(f"  [{done}/{total}] {target_name} N={N} lam={lam:.2f}  "
                      f"qi_mp={qi_mp_linf:.2e}  qi_f64={qi_f64_linf:.2e}  "
                      f"ls_f64={ls_f64_linf:.2e}  ls_mp={ls_mp_linf:.2e}  "
                      f"rank={svd_info.get('rank', '?')}/{w+1}  "
                      f"svd={solve_time:.0f}s  ({elapsed:.0f}s total)")

    elapsed = time.time() - t0
    print(f"\nData collection done in {elapsed:.1f}s ({total} configs)")

    with open(DATA_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {DATA_PATH}")

    return results


def plot_results(data_path=None):
    """Read saved data and produce 2x2 plots with 4 methods."""
    if data_path is None:
        data_path = DATA_PATH
    data_path = Path(data_path)

    with open(data_path) as f:
        results = json.load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_widths = sorted(set(r["N"] for r in results))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    colors = {w: palette[i % len(palette)] for i, w in enumerate(all_widths)}

    plot_targets = [t for t in TARGETS if any(r["target"] == t for r in results)]

    # 4 methods: QI mpmath, QI fp64, lstsq fp64, lstsq mpmath
    methods = [
        ("qi_mp", "QI mpmath", "--", "o"),
        ("qi_f64", "QI fp64", "--", "^"),
        ("ls_f64", "lstsq fp64", "-", "s"),
        ("ls_mp", "lstsq mpmath", "-", "D"),
    ]

    for metric_key, metric_label, filename in [
        ("linf", r"$L_\infty$ error", "qi_vs_learn_linf.png"),
        ("rel_l2", r"Relative $L_2$ error", "qi_vs_learn_rel_l2.png"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"QI vs Lstsq (fp64 and mpmath): {metric_label}", fontsize=14, y=0.98)

        for idx, target_name in enumerate(plot_targets):
            ax = axes[idx // 2][idx % 2]
            target_data = [r for r in results if r["target"] == target_name]

            for N in all_widths:
                width_data = sorted(
                    [r for r in target_data if r["N"] == N],
                    key=lambda r: r["lambda_star"],
                )
                lambdas = [r["lambda_star"] for r in width_data]
                color = colors[N]

                for method_key, method_name, ls, marker in methods:
                    vals = [r[f"{method_key}_{metric_key}"] for r in width_data]
                    lam_valid = [l for l, v in zip(lambdas, vals) if v == v]
                    val_valid = [v for v in vals if v == v]
                    if val_valid:
                        ax.semilogy(lam_valid, val_valid, ls, color=color,
                                    marker=marker, markersize=4, linewidth=1.2,
                                    label=f"N={N} {method_name}")

            ax.set_title(target_name, fontsize=12)
            ax.set_ylabel(metric_label)
            ax.set_xlabel(r"$\lambda = \gamma h$")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(LAMBDA_SWEEP)

        for idx in range(len(plot_targets), 4):
            axes[idx // 2][idx % 2].set_visible(False)

        # Compact legend: group by method, not by width
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   ncol=4, fontsize=7,
                   bbox_to_anchor=(0.5, -0.04))

        plt.tight_layout(rect=[0, 0.06, 1, 0.96])
        out_path = RESULTS_DIR / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_only = "--plot" in sys.argv

    if not plot_only:
        collect_data()

    if DATA_PATH.exists():
        plot_results()
    else:
        print(f"No data at {DATA_PATH}. Run without --plot first.")
