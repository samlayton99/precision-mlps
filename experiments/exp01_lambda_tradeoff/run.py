"""Experiment 1: Lambda Tradeoff Verification.

Traces the error-vs-lambda curve predicted by QI theory across multiple
widths and target functions. Compares full QI construction (convolution
with cardinal coefficients) against least-squares readout solve on the
same geometry.

Two modes:
  --fp64   (default) Fast sweep over full lambda range, fp64 precision.
  --mpmath          Focused sweep over viable regime, mpmath precision.
  --plot            Plot only (from saved data, specify --fp64 or --mpmath).

Outputs saved to results/exp01_lambda_tradeoff/.

Usage:
    python experiments/exp01_lambda_tradeoff/run.py                # fp64 collect + plot
    python experiments/exp01_lambda_tradeoff/run.py --mpmath       # mpmath collect + plot
    python experiments/exp01_lambda_tradeoff/run.py --plot         # plot fp64 data
    python experiments/exp01_lambda_tradeoff/run.py --mpmath --plot # plot mpmath data
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.qi_mpmath import construct_qi, evaluate_qi, default_halo
from src.construction.readout import build_phi, solve_readout_with_bias
from src.data.targets import get_target

RESULTS_DIR = REPO_ROOT / "results" / "exp01_lambda_tradeoff"
TARGETS = ["sine", "runge", "exp", "sine_mixture"]
N_EVAL = 2048
KC = 160

# --- fp64 config ---
FP64_WIDTHS = [16, 32, 64, 128, 256]
FP64_LAMBDAS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 1.0]
FP64_DATA_PATH = RESULTS_DIR / "data_fp64.json"

# --- mpmath config ---
MPMATH_WIDTHS = [128, 256, 512]
MPMATH_LAMBDAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
MPMATH_DATA_PATH = RESULTS_DIR / "data_mpmath.json"

# --- fine-grained config (fp64 lstsq + mpmath QI around optimal lambda) ---
FINE_WIDTHS = [16, 32, 64, 128]
FINE_LAMBDAS = [round(0.22 + i * 0.005, 3) for i in range(13)]  # 0.220 to 0.280
FINE_DATA_PATH = RESULTS_DIR / "data_fine.json"

# --- fine-grained fp64 config (fp64 lstsq + fp64 QI, same sweep as fine) ---
FINE_FP64_DATA_PATH = RESULTS_DIR / "data_fine_fp64.json"


def collect_data(precision, widths, lambda_sweep, data_path):
    """Run the lambda sweep for all targets/widths and save results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_eval = np.linspace(-1, 1, N_EVAL)
    results = []

    t0 = time.time()
    total = len(TARGETS) * len(widths) * len(lambda_sweep)
    done = 0

    for target_name in TARGETS:
        target = get_target(target_name)
        y_eval = target.fn_numpy(x_eval)
        y_norm = np.linalg.norm(y_eval)

        for N in widths:
            h = 2.0 / N
            # Training grid for readout solve
            n_train = max(512, 4 * N)
            x_train = np.linspace(-1, 1, n_train)
            y_train = target.fn_numpy(x_train)

            for lam in lambda_sweep:
                done += 1
                gamma = lam / h
                halo = default_halo(N, lambda_star=lam)

                # --- QI construction ---
                try:
                    qi = construct_qi(
                        target.fn_numpy, target.deriv_numpy, N,
                        precision=precision, lambda_star=lam, Kc=KC,
                        halo=halo, use_cache=True,
                    )
                    qi_pred = evaluate_qi(qi, x_eval)
                    qi_err = np.abs(qi_pred - y_eval)
                    qi_linf = float(np.max(qi_err))
                    qi_rel_l2 = float(np.linalg.norm(qi_err) / y_norm) if y_norm > 0 else float(np.linalg.norm(qi_err))
                except Exception as e:
                    qi_linf = float("nan")
                    qi_rel_l2 = float("nan")
                    qi = None
                    print(f"  QI failed: {target_name} N={N} lam={lam}: {e}")

                # --- Lstsq readout on same geometry ---
                try:
                    if qi is not None:
                        centers = qi.centers
                        gamma_vec = np.full(len(centers), qi.gamma)
                    else:
                        n_idx = np.arange(-halo, N + halo + 1)
                        centers = -1.0 + n_idx * h
                        gamma_vec = np.full(len(centers), gamma)

                    Phi_train = build_phi(x_train, gamma_vec, centers)
                    v, bias, _ = solve_readout_with_bias(Phi_train, y_train, method="lstsq")

                    Phi_eval = build_phi(x_eval, gamma_vec, centers)
                    lstsq_pred = Phi_eval @ v + bias
                    lstsq_err = np.abs(lstsq_pred - y_eval)
                    lstsq_linf = float(np.max(lstsq_err))
                    lstsq_rel_l2 = float(np.linalg.norm(lstsq_err) / y_norm) if y_norm > 0 else float(np.linalg.norm(lstsq_err))
                except Exception as e:
                    lstsq_linf = float("nan")
                    lstsq_rel_l2 = float("nan")
                    print(f"  Lstsq failed: {target_name} N={N} lam={lam}: {e}")

                results.append({
                    "target": target_name,
                    "N": N,
                    "lambda_star": lam,
                    "gamma": gamma,
                    "halo": halo,
                    "width": len(centers),
                    "precision": precision,
                    "qi_linf": qi_linf,
                    "qi_rel_l2": qi_rel_l2,
                    "lstsq_linf": lstsq_linf,
                    "lstsq_rel_l2": lstsq_rel_l2,
                })

                elapsed = time.time() - t0
                print(f"  [{done}/{total}] {target_name} N={N} lam={lam:.3f} "
                      f"qi_linf={qi_linf:.2e} lstsq_linf={lstsq_linf:.2e} "
                      f"({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    print(f"\nData collection done in {elapsed:.1f}s ({total} configs)")

    with open(data_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {data_path}")

    return results


def collect_data_fine(widths, lambda_sweep, data_path):
    """Fine-grained sweep: QI in mpmath, lstsq in fp64, on the same geometry."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_eval = np.linspace(-1, 1, N_EVAL)
    results = []

    t0 = time.time()
    total = len(TARGETS) * len(widths) * len(lambda_sweep)
    done = 0

    for target_name in TARGETS:
        target = get_target(target_name)
        y_eval = target.fn_numpy(x_eval)
        y_norm = np.linalg.norm(y_eval)

        for N in widths:
            h = 2.0 / N
            n_train = max(512, 4 * N)
            x_train = np.linspace(-1, 1, n_train)
            y_train = target.fn_numpy(x_train)

            for lam in lambda_sweep:
                done += 1
                gamma = lam / h
                halo = default_halo(N, lambda_star=lam)

                # --- QI construction (mpmath) ---
                try:
                    qi = construct_qi(
                        target.fn_numpy, target.deriv_numpy, N,
                        precision="mpmath", lambda_star=lam, Kc=KC,
                        halo=halo, mp_dps=50, use_cache=True,
                    )
                    qi_pred = evaluate_qi(qi, x_eval)
                    qi_err = np.abs(qi_pred - y_eval)
                    qi_linf = float(np.max(qi_err))
                    qi_rel_l2 = float(np.linalg.norm(qi_err) / y_norm) if y_norm > 0 else float(np.linalg.norm(qi_err))
                except Exception as e:
                    qi_linf = float("nan")
                    qi_rel_l2 = float("nan")
                    qi = None
                    print(f"  QI mpmath failed: {target_name} N={N} lam={lam}: {e}")

                # --- Lstsq readout (fp64) on same geometry ---
                try:
                    if qi is not None:
                        centers = qi.centers
                        gamma_vec = np.full(len(centers), qi.gamma)
                    else:
                        n_idx = np.arange(-halo, N + halo + 1)
                        centers = -1.0 + n_idx * h
                        gamma_vec = np.full(len(centers), gamma)

                    Phi_train = build_phi(x_train, gamma_vec, centers)
                    v, bias, _ = solve_readout_with_bias(Phi_train, y_train, method="lstsq")

                    Phi_eval = build_phi(x_eval, gamma_vec, centers)
                    lstsq_pred = Phi_eval @ v + bias
                    lstsq_err = np.abs(lstsq_pred - y_eval)
                    lstsq_linf = float(np.max(lstsq_err))
                    lstsq_rel_l2 = float(np.linalg.norm(lstsq_err) / y_norm) if y_norm > 0 else float(np.linalg.norm(lstsq_err))
                except Exception as e:
                    lstsq_linf = float("nan")
                    lstsq_rel_l2 = float("nan")
                    print(f"  Lstsq fp64 failed: {target_name} N={N} lam={lam}: {e}")

                results.append({
                    "target": target_name,
                    "N": N,
                    "lambda_star": lam,
                    "gamma": gamma,
                    "halo": halo,
                    "width": len(centers),
                    "precision": "fine",
                    "qi_linf": qi_linf,
                    "qi_rel_l2": qi_rel_l2,
                    "lstsq_linf": lstsq_linf,
                    "lstsq_rel_l2": lstsq_rel_l2,
                })

                elapsed = time.time() - t0
                print(f"  [{done}/{total}] {target_name} N={N} lam={lam:.3f} "
                      f"qi_mp={qi_linf:.2e} ls_f64={lstsq_linf:.2e} "
                      f"({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    print(f"\nFine sweep done in {elapsed:.1f}s ({total} configs)")

    with open(data_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {data_path}")

    return results


def plot_results(data_path, widths, suffix=""):
    """Read saved data and produce 2x2 plots."""
    data_path = Path(data_path)

    with open(data_path) as f:
        results = json.load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover widths from data
    all_widths = sorted(set(r["N"] for r in results))
    # Use provided widths to filter, or fall back to all
    plot_widths = [w for w in widths if w in all_widths] or all_widths

    # Color map: assign colors to widths dynamically
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    colors = {w: palette[i % len(palette)] for i, w in enumerate(plot_widths)}

    # Discover targets from data
    plot_targets = [t for t in TARGETS if any(r["target"] == t for r in results)]
    n_targets = len(plot_targets)
    ncols = 2
    nrows = (n_targets + 1) // 2

    prec_raw = results[0].get("precision", "fp64") if results else "fp64"
    prec_label = "QI mpmath + lstsq fp64" if prec_raw == "fine" else prec_raw

    for metric_key, metric_label, filename in [
        ("linf", r"$L_\infty$ error", f"lambda_tradeoff_linf{suffix}.png"),
        ("rel_l2", r"Relative $L_2$ error", f"lambda_tradeoff_rel_l2{suffix}.png"),
    ]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]
        fig.suptitle(f"Lambda Tradeoff ({prec_label}): {metric_label}", fontsize=14, y=0.98)

        for idx, target_name in enumerate(plot_targets):
            ax = axes[idx // ncols][idx % ncols]
            target_data = [r for r in results if r["target"] == target_name]

            for N in plot_widths:
                width_data = sorted(
                    [r for r in target_data if r["N"] == N],
                    key=lambda r: r["lambda_star"],
                )

                lambdas = [r["lambda_star"] for r in width_data]
                qi_vals = [r[f"qi_{metric_key}"] for r in width_data]
                lstsq_vals = [r[f"lstsq_{metric_key}"] for r in width_data]

                # Filter NaNs
                lam_qi = [l for l, v in zip(lambdas, qi_vals) if v == v]
                val_qi = [v for v in qi_vals if v == v]
                lam_ls = [l for l, v in zip(lambdas, lstsq_vals) if v == v]
                val_ls = [v for v in lstsq_vals if v == v]

                color = colors[N]
                if val_qi:
                    ax.semilogy(lam_qi, val_qi, "--o", color=color, markersize=3,
                                linewidth=1.5, label=f"N={N} QI")
                if val_ls:
                    ax.semilogy(lam_ls, val_ls, "-s", color=color, markersize=3,
                                linewidth=1.5, label=f"N={N} lstsq")

            ax.set_title(target_name, fontsize=12)
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_targets, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        for ax in axes[-1]:
            if ax.get_visible():
                ax.set_xlabel(r"$\lambda = \gamma h$")

        # Legend
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(handles), 6), fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        out_path = RESULTS_DIR / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    use_mpmath = "--mpmath" in sys.argv
    use_fine = "--fine" in sys.argv
    use_fine_fp64 = "--fine-fp64" in sys.argv
    plot_only = "--plot" in sys.argv

    if use_fine_fp64:
        precision = "fp64"
        widths = FINE_WIDTHS
        lambdas = FINE_LAMBDAS
        data_path = FINE_FP64_DATA_PATH
        suffix = "_fine_fp64"
    elif use_fine:
        precision = "mpmath"
        widths = MPMATH_WIDTHS
        lambdas = MPMATH_LAMBDAS
        data_path = MPMATH_DATA_PATH
        suffix = "_mpmath"
    else:
        precision = "fp64"
        widths = FP64_WIDTHS
        lambdas = FP64_LAMBDAS
        data_path = FP64_DATA_PATH
        suffix = "_fp64"

    if not plot_only:
        if use_fine:
            collect_data_fine(widths, lambdas, data_path)
        else:
            collect_data(precision, widths, lambdas, data_path)

    if data_path.exists():
        plot_results(data_path, widths, suffix)
    else:
        print(f"No data at {data_path}. Run without --plot first.")
