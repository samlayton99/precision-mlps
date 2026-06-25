"""Experiment A05: weight-blowup study -- QI vs fp64 lstsq readout NORM.

Question (roadmap Checkpoint-A follow-up #5): characterize the paper's
"weight blowup" violation. The QI construction is NOT the minimum-norm
representative of the readout; numpy lstsq returns the minimum-norm
least-squares solution on the SAME geometry. So how do the two readout-weight
norms compare as width N grows, and does carrying a larger norm cost any
precision?

Two methods at each (target, N), at fixed lambda = 0.25:
  1. QI fp64  -- the explicit construction (fp64 path). Readout weights are the
                 outer coefficients a_n.
  2. lstsq fp64 -- numpy least-squares (min-norm) solve on the identical
                   geometry (same centers incl. halo, same gamma).

Both methods are fp64 (the mpmath path is ~minutes/N cold and unnecessary here).
The headline quantity -- the readout NORM -- is identical to mpmath: the norm is
dominated by the well-determined main-lobe coefficients, so fp64 ill-conditioning
at lambda=0.25 does not move it (verified: N=64 gives |a|2=1.2747 both ways).

CAVEAT (norm-vs-error scatter only): at lambda=0.25 the fp64 QI construction
floors at L_inf ~1e-10 (convolution cancellation, |c_j|~338 alternating) rather
than machine eps -- a construction-PRECISION artifact, NOT weight-driven. The
fp64 lstsq solve never forms the convolution and reaches ~1e-13. So read the
weight-blowup signal off the NORM (left column); the scatter's QI error axis
carries this fp64 floor. Cardinal coefficients are cached to disk.

Norm = L2 norm of the full readout-weight vector (all centers incl. halo,
bias excluded). max|.| is also recorded.

Figure (4 rows = 4 targets, 2 cols):
  left  -- readout norm vs N: QI (orange) and lstsq (blue) lines.
  right -- norm (x) vs eval L_inf error (y) scatter; QI = orange gradient,
           lstsq = blue gradient, light = low N -> dark = high N.

Usage:
    python3 experiments/expA05_weight_blowup/run.py            # collect + plot
    python3 experiments/expA05_weight_blowup/run.py --plot     # plot only
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.qi_mpmath import construct_qi, evaluate_qi, default_halo
from src.construction.readout import build_phi, solve_readout_with_bias
from src.data.targets import get_target

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_A_numerics" / "expA05_weight_blowup"
DATA_PATH = RESULTS_DIR / "data.json"

LAMBDA = 0.25
N_MIN, N_MAX, N_POINTS = 32, 512, 20
TARGETS = ["sine", "runge", "exp", "sine_mixture"]
KC = 160
N_EVAL = 2048


def width_grid():
    """20 unique integer widths, log-spaced from N_MIN to N_MAX."""
    raw = np.logspace(np.log10(N_MIN), np.log10(N_MAX), N_POINTS)
    return sorted(set(int(round(v)) for v in raw))


def collect_data():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    widths = width_grid()
    x_eval = np.linspace(-1, 1, N_EVAL)
    results = []

    t0 = time.time()
    total = len(TARGETS) * len(widths)
    done = 0

    for target_name in TARGETS:
        target = get_target(target_name)
        y_eval = target.fn_numpy(x_eval)
        y_norm = np.linalg.norm(y_eval) or 1.0

        for N in widths:
            done += 1
            h = 2.0 / N
            halo = default_halo(N, lambda_star=LAMBDA)

            # --- QI (fp64 construction, the reference) ---
            qi = construct_qi(
                target.fn_numpy, target.deriv_numpy, N,
                precision="fp64", lambda_star=LAMBDA, Kc=KC,
                halo=halo, use_cache=True,
            )
            centers = qi.centers                       # full grid incl. halo
            gamma = qi.gamma
            w = len(centers)
            gamma_vec = np.full(w, gamma)

            pred = evaluate_qi(qi, x_eval)
            err = np.abs(pred - y_eval)
            qi_linf = float(np.max(err))
            qi_rel_l2 = float(np.linalg.norm(err) / y_norm)
            qi_norm = float(np.linalg.norm(qi.a_coeffs))
            qi_maxabs = float(np.max(np.abs(qi.a_coeffs)))

            # --- lstsq fp64 (min-norm) on the IDENTICAL geometry ---
            n_train = max(512, 2 * w)
            x_train = np.linspace(-1, 1, n_train)
            y_train = target.fn_numpy(x_train)
            Phi_train = build_phi(x_train, gamma_vec, centers)
            v, bias, _ = solve_readout_with_bias(Phi_train, y_train, method="lstsq")
            Phi_eval = build_phi(x_eval, gamma_vec, centers)
            pred = Phi_eval @ v + bias
            err = np.abs(pred - y_eval)
            ls_linf = float(np.max(err))
            ls_rel_l2 = float(np.linalg.norm(err) / y_norm)
            ls_norm = float(np.linalg.norm(v))
            ls_maxabs = float(np.max(np.abs(v)))

            results.append({
                "target": target_name, "N": N, "width": w, "halo": halo,
                "gamma": gamma, "lambda_star": LAMBDA,
                "qi_norm": qi_norm, "qi_maxabs": qi_maxabs,
                "qi_linf": qi_linf, "qi_rel_l2": qi_rel_l2,
                "ls_norm": ls_norm, "ls_maxabs": ls_maxabs,
                "ls_linf": ls_linf, "ls_rel_l2": ls_rel_l2,
            })

            elapsed = time.time() - t0
            print(f"  [{done}/{total}] {target_name} N={N} w={w}  "
                  f"|a|_QI={qi_norm:.3e} (Linf {qi_linf:.1e})  "
                  f"|v|_ls={ls_norm:.3e} (Linf {ls_linf:.1e})  "
                  f"({elapsed:.0f}s)")

    with open(DATA_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} rows to {DATA_PATH} ({time.time()-t0:.0f}s)")
    return results


# Weight-norm variants to plot: (data-key suffix, axis label, filename).
NORM_VARIANTS = [
    ("norm", r"$\|v\|_2$", "weight_blowup.png"),
    ("maxabs", r"$\|v\|_\infty = \max_n |v_n|$", "weight_blowup_linf.png"),
]


def plot_results(data_path=None):
    data_path = Path(data_path or DATA_PATH)
    with open(data_path) as f:
        results = json.load(f)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for key, label, fname in NORM_VARIANTS:
        _plot_one(results, key, label, fname)


def _plot_one(results, norm_key, norm_label, filename):

    targets = [t for t in TARGETS if any(r["target"] == t for r in results)]
    all_N = sorted(set(r["N"] for r in results))
    norm_N = LogNorm(vmin=min(all_N), vmax=max(all_N))
    # Light -> dark = low -> high N. Clamp the light end so points stay visible.
    def shade(cmap, N):
        return cmap(0.30 + 0.70 * norm_N(N))

    qi_line, ls_line = "#ff7f0e", "#1f77b4"

    fig, axes = plt.subplots(len(targets), 2, figsize=(12, 3.4 * len(targets)),
                             squeeze=False)
    fig.suptitle(
        rf"Readout-weight blowup ({norm_label}): QI vs fp64 lstsq (min-norm), $\lambda={LAMBDA}$",
        fontsize=14, y=0.997)
    qi_col = f"qi_{norm_key}"
    ls_col = f"ls_{norm_key}"

    for row, target_name in enumerate(targets):
        rows = sorted([r for r in results if r["target"] == target_name],
                      key=lambda r: r["N"])
        Ns = [r["N"] for r in rows]
        qi_norm = [r[qi_col] for r in rows]
        ls_norm = [r[ls_col] for r in rows]

        # --- left: norm vs N ---
        axL = axes[row][0]
        axL.loglog(Ns, qi_norm, "-o", color=qi_line, ms=4, lw=1.4, label="QI")
        axL.loglog(Ns, ls_norm, "-s", color=ls_line, ms=4, lw=1.4, label="lstsq (fp64)")
        axL.set_xlabel("width $N$")
        axL.set_ylabel(rf"readout norm {norm_label}")
        axL.grid(True, which="both", alpha=0.3)
        axL.xaxis.set_minor_formatter(NullFormatter())
        axL.text(-0.32, 0.5, target_name, transform=axL.transAxes,
                 rotation=90, va="center", ha="center", fontsize=13, fontweight="bold")
        axL.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                   borderaxespad=0, fontsize=9)

        # --- right: norm vs L_inf error, gradient by N ---
        axR = axes[row][1]
        for r in rows:
            axR.scatter(r[qi_col], r["qi_linf"], color=shade(plt.cm.Oranges, r["N"]),
                        s=34, edgecolors="0.3", linewidths=0.3, zorder=3)
            axR.scatter(r[ls_col], r["ls_linf"], color=shade(plt.cm.Blues, r["N"]),
                        s=34, marker="s", edgecolors="0.3", linewidths=0.3, zorder=3)
        axR.set_xscale("log"); axR.set_yscale("log")
        axR.set_xlabel(rf"readout norm {norm_label}")
        axR.set_ylabel(r"eval $L_\infty$ error")
        axR.grid(True, which="both", alpha=0.3)
        # legend: method (color) + shade convention
        h_qi = axR.scatter([], [], color=plt.cm.Oranges(0.8), s=34,
                           edgecolors="0.3", linewidths=0.3, label="QI")
        h_ls = axR.scatter([], [], color=plt.cm.Blues(0.8), s=34, marker="s",
                           edgecolors="0.3", linewidths=0.3, label="lstsq (fp64)")
        axR.legend(handles=[h_qi, h_ls], loc="lower center",
                   bbox_to_anchor=(0.5, 1.02), ncol=2, borderaxespad=0, fontsize=9)

    # shared N colorbar (grayscale -> shade convention) on the right margin
    sm = ScalarMappable(norm=norm_N, cmap="Greys")
    cb = fig.colorbar(sm, ax=axes[:, 1], fraction=0.025, pad=0.02)
    cb.set_label("width $N$ (light $\\to$ dark)")

    fig.subplots_adjust(left=0.13, right=0.90, top=0.95, bottom=0.05,
                        hspace=0.45, wspace=0.30)
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
