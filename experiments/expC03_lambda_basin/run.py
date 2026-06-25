"""Experiment 18 -- lambda basin: center + confidence range of the optimal bandwidth.

Stage 0 (this file, --check): visual stress-test of two candidate "basin
estimators" before committing to the full sweep. For each (target, width) we
sweep lambda finely on the uniform QI geometry (single shared gamma = lambda/h,
centers on the grid + halo), refit the readout by fp64 least squares, and record
eval relative L2 and L_inf. We then overlay, as vertical lines, the proposed
CENTER and BOUNDS from each estimator, to check whether they pin the bottom of
the basin.

Two estimators (both run on rel L2):
  M1  threshold band + parabola vertex:
        floor E0 = median of the lowest few errors; band = largest contiguous
        lambda-interval with E <= T*E0 (edges interpolated to the T*E0 crossing
        in log-error); center = vertex of a quadratic fit of log10(E) vs lambda
        over the in-band points.
  M2  basin posterior (softmin):
        weight w(lambda) = max(0, 1 - (log10 E - log10 E0)/D) (linear tent of
        depth D decades above the floor); center = sum(lambda*w)/sum(w);
        bounds = center +/- weighted std.

The deliverable quantity is the inner-layer magnitude: weight ~ gamma =
lambda*N/2, bias ~ gamma*|center| (centers span ~[-1.8,1.8]). So lambda* and its
range translate directly into "how big should the weights/biases be".

Usage:
    python experiments/expC03_lambda_basin/run.py --check
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.qi_mpmath import default_halo
from src.construction.readout import build_phi, solve_readout_with_bias
from src.data.targets import get_target

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_C_geometry" / "expC03_lambda_basin"

TARGETS = ["sine", "sine_8pi", "runge", "exp"]   # quick-check rows: broad -> pointy basins
WIDTHS_N = [64, 128, 256, 512]                   # quick-check cols

# Full sweep: the 6-category target family x widths.
FULL_TARGETS = ["sine", "sine_8pi", "runge", "sine_mixture", "exp", "abs_cubed"]
FULL_WIDTHS = [32, 64, 128, 256, 512, 1024]
RESOLVE_FLOOR = 1e-10        # a cell whose best rel L2 is worse than this is "unresolved"

HALO_LAMBDA = 0.25                               # fixes halo/W per N (W constant along the sweep)
LAMBDAS = np.round(np.arange(0.02, 0.601, 0.005), 4)  # finely interpolated x-axis
N_EVAL = 8001


# ----------------------------------------------------------------------------
# Sweep: relative L2 and L_inf vs lambda on the uniform geometry.
# ----------------------------------------------------------------------------
def sweep_curve(target, N):
    h = 2.0 / N
    halo = default_halo(N, lambda_star=HALO_LAMBDA)
    centers = -1.0 + np.arange(-halo, N + halo + 1) * h
    W = centers.size

    n_train = max(2048, 4 * W)
    x_train = np.linspace(-1, 1, n_train)
    y_train = target.fn_numpy(x_train)
    x_eval = np.linspace(-1, 1, N_EVAL)
    y_eval = target.fn_numpy(x_eval)
    y_norm = np.linalg.norm(y_eval)

    rel_l2 = np.empty(LAMBDAS.size)
    linf = np.empty(LAMBDAS.size)
    for i, lam in enumerate(LAMBDAS):
        gamma_vec = np.full(W, lam / h)
        Phi_tr = build_phi(x_train, gamma_vec, centers)
        v, bias, _ = solve_readout_with_bias(Phi_tr, y_train, method="lstsq")
        pred = build_phi(x_eval, gamma_vec, centers) @ v + bias
        err = np.abs(pred - y_eval)
        rel_l2[i] = np.linalg.norm(err) / y_norm
        linf[i] = err.max()
    return rel_l2, linf, W


# ----------------------------------------------------------------------------
# Estimators.
# ----------------------------------------------------------------------------
def _floor_log10(err, m_floor=3):
    return float(np.median(np.sort(np.log10(err))[:m_floor]))


def _interp_edge(lam, le, i_in, i_out, thr_log):
    """Lambda where log-error crosses thr_log between in-band i_in and out-band i_out."""
    x0, y0 = lam[i_in], le[i_in]
    x1, y1 = lam[i_out], le[i_out]
    if y1 == y0:
        return x0
    t = (thr_log - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))


def estimator_m1(lam, err, T=10.0, m_floor=3):
    """Threshold band + parabola vertex."""
    le = np.log10(err)
    thr_log = _floor_log10(err, m_floor) + np.log10(T)
    below = le <= thr_log
    imin = int(np.argmin(err))
    lo_i = imin
    while lo_i > 0 and below[lo_i - 1]:
        lo_i -= 1
    hi_i = imin
    while hi_i < len(lam) - 1 and below[hi_i + 1]:
        hi_i += 1

    lo = lam[lo_i] if lo_i == 0 else _interp_edge(lam, le, lo_i, lo_i - 1, thr_log)
    hi = lam[hi_i] if hi_i == len(lam) - 1 else _interp_edge(lam, le, hi_i, hi_i + 1, thr_log)

    band = slice(lo_i, hi_i + 1)
    center = lam[imin]
    if hi_i - lo_i >= 2:
        a, b, c = np.polyfit(lam[band], le[band], 2)
        if a > 0:
            vtx = -b / (2 * a)
            if lo <= vtx <= hi:
                center = float(vtx)
    return center, lo, hi


def estimator_m2(lam, err, D=2.0, m_floor=3):
    """Basin posterior (softmin tent): weighted mean +/- weighted std."""
    le = np.log10(err)
    E0 = _floor_log10(err, m_floor)
    w = np.clip(1.0 - (le - E0) / D, 0.0, None)
    Wsum = w.sum()
    center = float((lam * w).sum() / Wsum)
    spread = float(np.sqrt(((lam - center) ** 2 * w).sum() / Wsum))
    return center, center - spread, center + spread


def estimator_m2_modeseek(lam, err, D=2.0, m_floor=3):
    """M2 + mode seeking: keep the posterior band, then recenter on the argmin
    within the central half of that band. Returns (center, lo, hi, in_lo, in_hi)
    where [lo, hi] is the full posterior band and [in_lo, in_hi] the search window.
    """
    lam = np.asarray(lam, float)
    err = np.asarray(err, float)
    c, lo, hi = estimator_m2(lam, err, D, m_floor)
    half = (hi - lo) / 4.0                       # half of the band's half-width
    in_lo, in_hi = c - half, c + half            # central half of the band
    mask = (lam >= in_lo) & (lam <= in_hi)
    center = float(lam[mask][np.argmin(err[mask])]) if mask.any() else c
    return center, lo, hi, in_lo, in_hi


# ----------------------------------------------------------------------------
# Figure.
# ----------------------------------------------------------------------------
METHODS = {
    "m1": ("M1: threshold band + parabola vertex", "#1f77b4"),
    "m2": ("M2: basin posterior (softmin)", "#d62728"),
    "m2ms": ("M2 + mode seeking (argmin in central half of band)", "#2ca02c"),
}


def estimators_for_curve(rel_l2):
    """Compute all estimator placements from one error curve."""
    m1 = estimator_m1(LAMBDAS, rel_l2)
    m2 = estimator_m2(LAMBDAS, rel_l2)
    ms = estimator_m2_modeseek(LAMBDAS, rel_l2)
    return {
        "m1_center": m1[0], "m1_lo": m1[1], "m1_hi": m1[2],
        "m2_center": m2[0], "m2_lo": m2[1], "m2_hi": m2[2],
        "m2ms_center": ms[0], "m2ms_lo": ms[1], "m2ms_hi": ms[2],
        "m2ms_inlo": ms[3], "m2ms_inhi": ms[4],
    }


def plot_method(data, method, title, color):
    """One 4x4 figure for a single estimator: curves + center line + shaded band.

    If the cell carries `<method>_inlo`/`_inhi`, a darker inner span marks the
    mode-seek search window.
    """
    nrows, ncols = len(TARGETS), len(WIDTHS_N)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                             sharex=True, squeeze=False)
    for r, tname in enumerate(TARGETS):
        for c, N in enumerate(WIDTHS_N):
            d = data[f"{tname}_N{N}"]
            lam = np.array(d["lambdas"])
            center, lo, hi = d[f"{method}_center"], d[f"{method}_lo"], d[f"{method}_hi"]
            ax = axes[r][c]
            ax.axvspan(lo, hi, color=color, alpha=0.15, lw=0)        # full posterior band
            if f"{method}_inlo" in d:                                # mode-seek window
                ax.axvspan(d[f"{method}_inlo"], d[f"{method}_inhi"],
                           color=color, alpha=0.22, lw=0)
            ax.axvline(center, color=color, lw=1.8)                  # center
            ax.semilogy(lam, d["rel_l2"], "-", color="black", lw=1.4, label="rel L2")
            ax.semilogy(lam, d["linf"], "-", color="0.6", lw=1.0, label="L_inf")
            gstar = center * N / 2.0
            ax.set_title(f"{tname}  N={N} (W={d['W']})\n"
                         f"center {center:.3f}  band [{lo:.2f}, {hi:.2f}]  "
                         f"gamma*~{gstar:.0f}", fontsize=8)
            ax.grid(True, alpha=0.3)
            if c == 0:
                ax.set_ylabel("error")
            if r == nrows - 1:
                ax.set_xlabel(r"$\lambda = \gamma h$")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 1.003))
    fig.suptitle(f"exp18 basin check -- {title}  "
                 "(shaded = proposed range, line = center; derived from rel L2)",
                 fontsize=11, y=1.025)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = RESULTS_DIR / f"basin_check_{method}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


def _plot_all(data):
    for method, (title, color) in METHODS.items():
        plot_method(data, method, title, color)


def run_check():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    data = {}
    for tname in TARGETS:
        target = get_target(tname)
        for N in WIDTHS_N:
            rel_l2, linf, W = sweep_curve(target, N)
            est = estimators_for_curve(rel_l2)
            data[f"{tname}_N{N}"] = {
                "N": N, "W": int(W), "lambdas": LAMBDAS.tolist(),
                "rel_l2": rel_l2.tolist(), "linf": linf.tolist(), **est,
            }
            print(f"  {tname} N={N}: M2 c={est['m2_center']:.3f} "
                  f"-> mode-seek {est['m2ms_center']:.3f} ({time.time()-t0:.0f}s)")

    with open(RESULTS_DIR / "basin_method_check.json", "w") as f:
        json.dump(data, f, indent=2)
    _plot_all(data)
    print(f"\nDone ({time.time()-t0:.0f}s)")


def run_replot():
    """Reuse cached error curves: recompute estimator placements and redraw."""
    with open(RESULTS_DIR / "basin_method_check.json") as f:
        data = json.load(f)
    for key, d in data.items():
        d.update(estimators_for_curve(np.array(d["rel_l2"])))
        print(f"  {key}: M2 c={d['m2_center']:.3f} -> mode-seek {d['m2ms_center']:.3f}")
    with open(RESULTS_DIR / "basin_method_check.json", "w") as f:
        json.dump(data, f, indent=2)
    _plot_all(data)
    print("Replot done.")


# ----------------------------------------------------------------------------
# Full sweep: 6-target family x widths, locked estimator = M2 + mode seeking.
# ----------------------------------------------------------------------------
FULL_DATA_PATH = lambda: RESULTS_DIR / "full_sweep.json"
GREEN = "#2ca02c"


def plot_basin_grid(data):
    """6 targets (rows) x widths (cols): error-vs-lambda + outer band + center."""
    nrows, ncols = len(FULL_TARGETS), len(FULL_WIDTHS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 2.9 * nrows),
                             sharex=True, squeeze=False)
    for r, tname in enumerate(FULL_TARGETS):
        for c, N in enumerate(FULL_WIDTHS):
            d = data[f"{tname}_N{N}"]
            lam = np.array(d["lambdas"])
            ax = axes[r][c]
            if d["resolved"]:
                ax.axvspan(d["m2ms_lo"], d["m2ms_hi"], color=GREEN, alpha=0.18, lw=0)
                ax.axvline(d["m2ms_center"], color=GREEN, lw=1.8)
                tag = f"c {d['m2ms_center']:.3f}  g*~{d['m2ms_center']*N/2:.0f}"
            else:
                tag = "UNRESOLVED"
            ax.semilogy(lam, d["rel_l2"], "-", color="black", lw=1.3, label="rel L2")
            ax.semilogy(lam, d["linf"], "-", color="0.6", lw=0.9, label="L_inf")
            ax.set_title(f"{tname}  N={N} (W={d['W']})\n{tag}", fontsize=8)
            ax.grid(True, alpha=0.3)
            if c == 0:
                ax.set_ylabel("error")
            if r == nrows - 1:
                ax.set_xlabel(r"$\lambda = \gamma h$")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.004))
    fig.suptitle("exp18 full basin grid -- M2 + mode seeking (shaded = range, line = center; from rel L2)",
                 fontsize=12, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    out = RESULTS_DIR / "basin_grid_m2ms.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_lambda_vs_size(data):
    """6 targets (rows) x {N, W} (cols): lambda* center + shaded range vs size."""
    nrows = len(FULL_TARGETS)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 2.5 * nrows), squeeze=False)
    for r, tname in enumerate(FULL_TARGETS):
        cells = [data[f"{tname}_N{N}"] for N in FULL_WIDTHS if data[f"{tname}_N{N}"]["resolved"]]
        for c, (xkey, xlabel) in enumerate([("N", "N (grid size)"), ("W", "W (neurons)")]):
            ax = axes[r][c]
            ax.axhline(0.25, color="0.7", lw=1.0, ls="--", label=r"$\lambda=0.25$ (ref)")
            if cells:
                xs = [d[xkey] for d in cells]
                ctr = [d["m2ms_center"] for d in cells]
                lo = [d["m2ms_lo"] for d in cells]
                hi = [d["m2ms_hi"] for d in cells]
                ax.fill_between(xs, lo, hi, color=GREEN, alpha=0.22, label="range")
                ax.plot(xs, ctr, "-o", color=GREEN, lw=1.8, ms=5, label=r"$\lambda^*$ (center)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(FULL_WIDTHS if xkey == "N" else [data[f"{tname}_N{N}"]["W"] for N in FULL_WIDTHS])
            ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
            ax.set_ylim(0.0, 0.45)
            ax.grid(True, alpha=0.3)
            if c == 0:
                ax.set_ylabel(f"{tname}\n" + r"$\lambda^*$")
            if r == nrows - 1:
                ax.set_xlabel(xlabel)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(r"exp18 -- optimal $\lambda^*$ vs size  (weight/bias magnitude $\gamma=\lambda^* N/2$);"
                 " center = M2+mode-seek, band = posterior range", fontsize=12, y=1.025)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = RESULTS_DIR / "lambda_vs_size.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def run_full():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = {}
    for tname in FULL_TARGETS:
        target = get_target(tname)
        for N in FULL_WIDTHS:
            rel_l2, linf, W = sweep_curve(target, N)
            est = estimators_for_curve(rel_l2)
            floor = float(np.min(rel_l2))
            data[f"{tname}_N{N}"] = {
                "N": N, "W": int(W), "lambdas": LAMBDAS.tolist(),
                "rel_l2": rel_l2.tolist(), "linf": linf.tolist(),
                "floor": floor, "resolved": bool(floor <= RESOLVE_FLOOR), **est,
            }
            flag = "" if floor <= RESOLVE_FLOOR else "  [UNRESOLVED]"
            print(f"  {tname} N={N} (W={W}): floor={floor:.1e} "
                  f"lambda*={est['m2ms_center']:.3f} g*~{est['m2ms_center']*N/2:.0f}{flag} "
                  f"({time.time()-t0:.0f}s)")
    with open(FULL_DATA_PATH(), "w") as f:
        json.dump(data, f, indent=2)
    plot_basin_grid(data)
    plot_lambda_vs_size(data)
    print(f"\nFull sweep done ({time.time()-t0:.0f}s)")


def run_full_replot():
    with open(FULL_DATA_PATH()) as f:
        data = json.load(f)
    for d in data.values():
        d.update(estimators_for_curve(np.array(d["rel_l2"])))
    plot_basin_grid(data)
    plot_lambda_vs_size(data)
    print("Full replot done.")


if __name__ == "__main__":
    if "--full" in sys.argv:
        run_full()
    elif "--full-replot" in sys.argv:
        run_full_replot()              # reuse full_sweep.json, just redraw
    elif "--replot" in sys.argv:
        run_replot()                   # fast: reuse 4x4 cached curves
    elif "--check" in sys.argv or len(sys.argv) == 1:
        run_check()
    else:
        print("Usage: run.py [--check | --replot | --full | --full-replot]")
