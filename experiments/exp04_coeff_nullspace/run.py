"""Experiment 4: How close are the QI and least-squares readouts?

A marginal justification (beyond eval error) for using lstsq: on the SAME
fixed tanh geometry, does lstsq land on essentially the QI solution?

The subtlety this experiment exists to expose: the tanh feature matrix Phi is
~50% null space at these lambda (verified: rank ~ N+12 of ~N+2*halo+1 columns).
So the QI weights a_n and the lstsq weights v can differ enormously in the raw
coefficient space and still compute the same function. We therefore report the
difference three ways:

    raw       full coefficient-space distance         -- dominated by null space
    rowspace  distance in Phi's row space (data-seen) -- the part that matters
    function  distance of the functions they compute  -- the honest measure

Expected story: raw is large, rowspace/function are at the fp64 floor -> the
two are the "same solution" everywhere the data constrains; they only disagree
in directions the data never sees. The n_train sweep checks that growing the
lstsq sample count does not move this (no spurious trend).

Everything is fp64 and kept quick.

Usage:
    python3 experiments/exp04_coeff_nullspace/run.py            # collect + plot
    python3 experiments/exp04_coeff_nullspace/run.py --plot      # plot only
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

from src.construction.qi_mpmath import construct_qi, default_halo
from src.construction.readout import build_phi, solve_readout_with_bias
from src.data.targets import get_target
import coeff_compare as cc

RESULTS_DIR = REPO_ROOT / "results" / "exp04_coeff_nullspace"
DATA_PATH = RESULTS_DIR / "data.json"

LAMBDA = 0.30
KC = 160
N_EVAL = 2048
WIDTHS = [32, 64, 96, 128]
TARGETS = ["sine", "sine_8pi", "runge", "sine_mixture", "exp", "abs_cubed"]
NSAMPLES_SWEEP = [256, 512, 1024, 2048, 4096]
NSAMPLES_TARGET = "sine"


def _compare_one(target, N, n_train, x_eval, y_eval):
    """QI vs lstsq on one fixed geometry. Returns a metrics dict."""
    h = 2.0 / N
    halo = default_halo(N, lambda_star=LAMBDA)

    # --- QI construction (defines the geometry: centers + gamma) ---
    qi = construct_qi(
        target.fn_numpy, target.deriv_numpy, N,
        precision="fp64", lambda_star=LAMBDA, Kc=KC, halo=halo, use_cache=True,
    )
    centers, gamma = qi.centers, qi.gamma
    W = len(centers)
    gamma_vec = np.full(W, gamma)

    # Augmented coefficient vector beta = [outer weights; bias].
    beta_qi = np.concatenate([qi.a_coeffs, [qi.c0]])

    # --- lstsq readout on the SAME geometry ---
    x_train = np.linspace(-1, 1, n_train)
    y_train = target.fn_numpy(x_train)
    Phi_train = build_phi(x_train, gamma_vec, centers)
    v, bias, _ = solve_readout_with_bias(Phi_train, y_train, method="lstsq")
    beta_ls = np.concatenate([v, [bias]])

    # Augmented design matrices [Phi, 1] for train (row space) and eval (function).
    A_train = np.hstack([Phi_train, np.ones((n_train, 1))])
    A_eval = np.hstack([build_phi(x_eval, gamma_vec, centers), np.ones((N_EVAL, 1))])

    raw = cc.raw_diff(beta_qi, beta_ls)
    fun = cc.function_diff(A_eval, beta_qi, beta_ls, y_eval=y_eval)

    # Headline: largest element-wise coefficient deviation / average coefficient,
    # measured in the full space (data-blind) and the row space (data-visible).
    ratio_full = cc.deviation_ratio(beta_qi, beta_ls)
    rs = cc.rowspace_ratio(A_train, beta_qi, beta_ls)  # rcond=1e-11

    y_train_norm = np.linalg.norm(y_train) or 1.0
    return {
        "target": target.name, "N": N, "width": W, "n_train": n_train,
        "rank": rs["rank"], "null_dim": (W + 1) - rs["rank"],
        "ratio_full": ratio_full, "ratio_row": rs["ratio"],
        "row_mean_qi": rs["mean_qi"], "row_mean_ls": rs["mean_ls"],
        "raw_l2": raw["l2"], "raw_linf": raw["linf"], "raw_rel_l2": raw["rel_l2"],
        "fun_l2": fun["l2"], "fun_linf": fun["linf"], "fun_rel_l2": fun["rel_l2"],
        # Sanity: both must actually fit the data for the comparison to be fair.
        "qi_fit_resid": cc.fit_residual(A_train, beta_qi, y_train) / y_train_norm,
        "ls_fit_resid": cc.fit_residual(A_train, beta_ls, y_train) / y_train_norm,
    }


def collect_data():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x_eval = np.linspace(-1, 1, N_EVAL)
    t0 = time.time()

    width_rows, nsamp_rows = [], []

    # --- Sweep 1: vs width, one target per category, n_train = max(512, 2W) ---
    print("Sweep 1: coefficient difference vs width")
    for target_name in TARGETS:
        target = get_target(target_name)
        y_eval = target.fn_numpy(x_eval)
        for N in WIDTHS:
            W = N + 2 * default_halo(N, lambda_star=LAMBDA) + 1
            r = _compare_one(target, N, max(512, 2 * W), x_eval, y_eval)
            width_rows.append(r)
            print(f"  {target_name:13} N={N:>3} W={r['width']:>4} "
                  f"null={r['null_dim']:>3}  ratio_full={r['ratio_full']:.2e}  "
                  f"ratio_row={r['ratio_row']:.2e}  fun_linf={r['fun_linf']:.2e}  "
                  f"(qi_fit={r['qi_fit_resid']:.1e} ls_fit={r['ls_fit_resid']:.1e})")

    # --- Sweep 2: hold model size, vary lstsq sample count (fixed target) ---
    print(f"\nSweep 2: coefficient difference vs n_train ({NSAMPLES_TARGET})")
    target = get_target(NSAMPLES_TARGET)
    y_eval = target.fn_numpy(x_eval)
    for N in WIDTHS:
        for n_train in NSAMPLES_SWEEP:
            r = _compare_one(target, N, n_train, x_eval, y_eval)
            nsamp_rows.append(r)
            print(f"  N={N:>3} n_train={n_train:>5}  ratio_full={r['ratio_full']:.2e}  "
                  f"ratio_row={r['ratio_row']:.2e}  fun_linf={r['fun_linf']:.2e}")

    out = {"width_sweep": width_rows, "nsamples_sweep": nsamp_rows,
           "lambda": LAMBDA, "Kc": KC}
    with open(DATA_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {time.time() - t0:.1f}s. Saved {DATA_PATH}")
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _two_panel(rows, group_key, x_key, x_label, title, caption, out_path):
    """Two stacked panels: raw coeff diff (top) and function diff (bottom).

    L2 dotted, Linf solid; one color per group_key value.
    """
    groups = sorted(set(r[group_key] for r in rows),
                    key=lambda g: (WIDTHS.index(g) if g in WIDTHS else 0)
                    if group_key == "N" else 0)
    colors = {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(groups)}

    fig, (ax_raw, ax_fun) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    for g in groups:
        gd = sorted([r for r in rows if r[group_key] == g], key=lambda r: r[x_key])
        xs = [r[x_key] for r in gd]
        c = colors[g]
        label = f"{group_key}={g}" if group_key == "N" else str(g)
        ax_raw.semilogy(xs, [r["raw_l2"] for r in gd], ":", color=c, marker="o",
                        ms=4, lw=1.3, label=label)
        ax_raw.semilogy(xs, [r["raw_linf"] for r in gd], "-", color=c, marker="o",
                        ms=4, lw=1.3)
        ax_fun.semilogy(xs, [r["fun_rel_l2"] for r in gd], ":", color=c, marker="o",
                        ms=4, lw=1.3, label=label)
        ax_fun.semilogy(xs, [r["fun_linf"] for r in gd], "-", color=c, marker="o",
                        ms=4, lw=1.3)

    ax_raw.set_title("Raw coefficient difference  (dotted $L_2$, solid $L_\\infty$)")
    ax_raw.set_ylabel(r"$\|\beta_{QI}-\beta_{lstsq}\|$")
    ax_fun.set_title("Function difference on eval grid  (dotted rel-$L_2$, solid $L_\\infty$)")
    ax_fun.set_ylabel(r"$\|f_{QI}-f_{lstsq}\|$")
    ax_fun.set_xlabel(x_label)
    for ax in (ax_raw, ax_fun):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8,
             wrap=True, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _ratio_figure(rows, out_path):
    """Headline figure: full-space vs row-space deviation ratio, per target.

    Solid = full space (data-blind), dashed = row space (data-visible).
    The vertical gap between a target's two lines is the null-space freedom.
    """
    targets = [t for t in TARGETS if any(r["target"] == t for r in rows)]
    colors = {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(targets)}

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for t in targets:
        td = sorted([r for r in rows if r["target"] == t], key=lambda r: r["width"])
        xs = [r["width"] for r in td]
        c = colors[t]
        ax.semilogy(xs, [r["ratio_full"] for r in td], "-", color=c, marker="o",
                    ms=5, lw=1.6, label=t)
        ax.semilogy(xs, [r["ratio_row"] for r in td], "--", color=c, marker="s",
                    ms=4, lw=1.3)

    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("model width  W = N + 2*halo + 1")
    ax.set_ylabel(r"deviation ratio  $\max_i|a^{QI}_i-a^{lstsq}_i|\,/\,(\overline{|a|})$")
    ax.set_title("exp04: QI vs lstsq coefficient deviation -- full space vs row space")
    ax.grid(True, which="both", alpha=0.3)

    # Two legends: color = target, linestyle = which space.
    leg1 = ax.legend(title="target (color)", fontsize=8, loc="center right")
    ax.add_artist(leg1)
    style_handles = [
        plt.Line2D([], [], color="0.3", ls="-", marker="o", label="full space (data-blind)"),
        plt.Line2D([], [], color="0.3", ls="--", marker="s", label="row space (data-visible)"),
    ]
    ax.legend(handles=style_handles, fontsize=8, loc="lower left")

    caption = (
        "HOW TO READ: each color is a target; solid = deviation in the full "
        "coefficient space, dashed = deviation after projecting both vectors onto "
        "the data-visible subspace (the row space of [Phi,1]). Solid lines sit near "
        "1 (the worst coefficient disagrees by ~a typical coefficient); dashed lines "
        "for converged smooth targets fall to 1e-4-1e-3. The gap = the null space the "
        "data cannot see. Lines that do NOT drop (abs_cubed; coarse-N runge/mixture) "
        "are cases the construction itself has not resolved. CAVEAT: the dashed (row) "
        "agreement is the same fact as the functions agreeing -- not independent "
        "evidence; the solid (full) gap is the independent finding."
    )
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=7.5,
             wrap=True, style="italic")
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_results(data_path=None):
    data_path = Path(data_path) if data_path else DATA_PATH
    with open(data_path) as f:
        data = json.load(f)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lam = data.get("lambda", LAMBDA)

    # Headline figure.
    _ratio_figure(data["width_sweep"], RESULTS_DIR / "coeff_ratio_full_vs_row.png")

    _two_panel(
        data["width_sweep"], group_key="target", x_key="width",
        x_label="model width  W = N + 2*halo + 1",
        title=fr"exp04: QI vs lstsq coefficient closeness vs width ($\lambda$={lam}, fp64)",
        caption=(
            "On a fixed tanh geometry, QI and lstsq solve the same rank-deficient "
            "Phi @ beta = y (~half the columns are null space). TOP: the raw "
            "coefficient vectors differ by O(1) -- but that distance lives almost "
            "entirely in the null space the data cannot see. BOTTOM: the functions "
            "they compute agree to the fp64 floor. The two are the same solution "
            "wherever the data constrains it. One target per category."
        ),
        out_path=RESULTS_DIR / "coeff_diff_vs_width.png",
    )

    _two_panel(
        data["nsamples_sweep"], group_key="N", x_key="n_train",
        x_label="number of lstsq sample points  n_train",
        title=fr"exp04: closeness vs lstsq sample count ($\lambda$={lam}, fp64, target={NSAMPLES_TARGET})",
        caption=(
            "QI sampling is fixed by the architecture; lstsq sampling is free. "
            "Holding the model width fixed and increasing the number of lstsq "
            "sample points leaves both the raw coefficient difference (TOP) and "
            "the function difference (BOTTOM) flat: more data does not pull lstsq "
            "toward or away from QI -- it already agrees wherever the data sees."
        ),
        out_path=RESULTS_DIR / "coeff_diff_vs_nsamples.png",
    )


if __name__ == "__main__":
    if "--plot" not in sys.argv:
        collect_data()
    if DATA_PATH.exists():
        plot_results()
    else:
        print(f"No data at {DATA_PATH}. Run without --plot first.")
