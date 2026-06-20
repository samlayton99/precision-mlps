"""Experiment 6: optimal lambda vs target frequency, QI vs lstsq.

Maps how the best bandwidth lambda depends on the target frequency, separately
for the QI construction (cardinal-coefficient convolution) and the lstsq readout
on the same geometry. Frequency ladder sin(k*pi*x), fp64, fine lambda grid.

Usage:
    python3 experiments/exp06_lambda_vs_frequency/run.py            # collect + plot
    python3 experiments/exp06_lambda_vs_frequency/run.py --plot      # plot only
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.construction.qi_mpmath import construct_qi, evaluate_qi, default_halo
from src.construction.readout import build_phi, solve_readout_with_bias
import sweep_utils as su

RESULTS_DIR = REPO_ROOT / "results" / "exp06_lambda_vs_frequency"
DATA_PATH = RESULTS_DIR / "data.json"

FREQS = [1, 2, 4, 8, 16]
WIDTHS = [128, 256]
LAMBDAS = [round(x, 3) for x in np.arange(0.02, 0.405, 0.01)]
N_EVAL = 4096
KC = 160


def _qi_linf(fn, deriv, N, lam, x_ev, y_ev):
    try:
        qi = construct_qi(fn, deriv, N, precision="fp64", lambda_star=lam, Kc=KC,
                          halo=default_halo(N, lambda_star=lam), use_cache=True)
        return float(np.max(np.abs(evaluate_qi(qi, x_ev) - y_ev)))
    except Exception:
        return float("nan")


def _ls_linf(fn, N, lam, x_ev, y_ev):
    h = 2.0 / N
    halo = default_halo(N, lambda_star=lam)
    centers = -1.0 + np.arange(-halo, N + halo + 1) * h
    gamma = np.full(len(centers), lam / h)
    x_tr = np.linspace(-1, 1, max(512, 4 * N))
    Phi = build_phi(x_tr, gamma, centers)
    v, b, _ = solve_readout_with_bias(Phi, fn(x_tr), method="lstsq")
    return float(np.max(np.abs(build_phi(x_ev, gamma, centers) @ v + b - y_ev)))


def collect_data():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x_ev = np.linspace(-1, 1, N_EVAL)
    rows = []
    t0 = time.time()
    for N in WIDTHS:
        for k in FREQS:
            fn, deriv = su.sin_target(k)
            y_ev = fn(x_ev)
            qi = [_qi_linf(fn, deriv, N, lam, x_ev, y_ev) for lam in LAMBDAS]
            ls = [_ls_linf(fn, N, lam, x_ev, y_ev) for lam in LAMBDAS]
            qi_best = su.best_lambda(LAMBDAS, qi)
            ls_best = su.best_lambda(LAMBDAS, ls)
            rows.append({"N": N, "k": k, "lambdas": LAMBDAS,
                         "qi_linf": qi, "ls_linf": ls,
                         "qi_best_lam": qi_best[0], "qi_best_err": qi_best[1],
                         "ls_best_lam": ls_best[0], "ls_best_err": ls_best[1]})
            print(f"  N={N:>3} k={k:>2}  QI best lam={qi_best[0]:.2f} ({qi_best[1]:.1e})  "
                  f"lstsq best lam={ls_best[0]:.2f} ({ls_best[1]:.1e})")
    with open(DATA_PATH, "w") as f:
        json.dump({"rows": rows, "freqs": FREQS, "widths": WIDTHS}, f, indent=2)
    print(f"\nDone in {time.time() - t0:.1f}s. Saved {DATA_PATH}")
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _plot_optimum(rows):
    """Optimal lambda vs frequency: QI (dashed) vs lstsq (solid), per width."""
    fig, ax = plt.subplots(figsize=(9, 6))
    wstyles = {WIDTHS[0]: ("#1f77b4", "o"), WIDTHS[1]: ("#d62728", "s")}
    for N in WIDTHS:
        rn = sorted([r for r in rows if r["N"] == N], key=lambda r: r["k"])
        ks = [r["k"] for r in rn]
        color, mk = wstyles.get(N, ("#333333", "^"))
        ax.plot(ks, [r["ls_best_lam"] for r in rn], "-", color=color, marker=mk,
                ms=6, lw=1.6, label=f"lstsq N={N}")
        ax.plot(ks, [r["qi_best_lam"] for r in rn], "--", color=color, marker=mk,
                ms=6, lw=1.4, mfc="none", label=f"QI N={N}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(FREQS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(r"target frequency $k$  in  $\sin(k\pi x)$")
    ax.set_ylabel(r"optimal $\lambda = \gamma h$")
    ax.set_title("exp06: optimal bandwidth vs frequency (fp64)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    caption = (
        "HOW TO READ: best lstsq (solid) and best QI (dashed/open) lambda vs target "
        "frequency, per width. Flat dashed lines => QI optimum is frequency-"
        "independent; rising solid lines => lstsq optimum tracks frequency."
    )
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8,
             wrap=True, style="italic")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = RESULTS_DIR / "optimal_lambda_vs_frequency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_curves(rows):
    """Error vs lambda U-curves, one panel per width; color = frequency."""
    fig, axes = plt.subplots(1, len(WIDTHS), figsize=(7 * len(WIDTHS), 6), sharey=True)
    if len(WIDTHS) == 1:
        axes = [axes]
    colors = {k: _PALETTE[i % len(_PALETTE)] for i, k in enumerate(FREQS)}
    for ax, N in zip(axes, WIDTHS):
        for r in sorted([r for r in rows if r["N"] == N], key=lambda r: r["k"]):
            c = colors[r["k"]]
            ax.semilogy(r["lambdas"], r["ls_linf"], "-", color=c, lw=1.4,
                        label=f"k={r['k']}")
            ax.semilogy(r["lambdas"], r["qi_linf"], "--", color=c, lw=1.1, alpha=0.7)
        ax.set_title(f"N={N}")
        ax.set_xlabel(r"$\lambda = \gamma h$")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"eval $L_\infty$ error")
    axes[0].legend(fontsize=8, title="frequency")
    fig.suptitle("exp06: error vs lambda (solid = lstsq, dashed = QI)", fontsize=13, y=0.99)
    caption = (
        "HOW TO READ: eval error vs lambda, one color per frequency; lstsq solid, "
        "QI dashed. QI U-curves all bottom at the same lambda; lstsq U-curves shift "
        "their minimum rightward (and their low-lambda wall rises) with frequency."
    )
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8,
             wrap=True, style="italic")
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = RESULTS_DIR / "error_vs_lambda_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_results(data_path=None):
    data_path = Path(data_path) if data_path else DATA_PATH
    with open(data_path) as f:
        data = json.load(f)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _plot_optimum(data["rows"])
    _plot_curves(data["rows"])


if __name__ == "__main__":
    if "--plot" not in sys.argv:
        collect_data()
    if DATA_PATH.exists():
        plot_results()
    else:
        print(f"No data at {DATA_PATH}. Run without --plot first.")
