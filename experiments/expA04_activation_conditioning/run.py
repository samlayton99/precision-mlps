"""Experiment expA04: GELU vs tanh feature-matrix conditioning.

GELU has no QI construction (its derivative is a sigmoid, not a bump kernel), so
the QI-dependent metrics of expA03 (ratio_full/ratio_row/qi_fit) do not apply. We
instead study what IS defined for GELU: the conditioning of the feature matrix
and the lstsq fit, with tanh overlaid for contrast.

Step 1: GELU lstsq readout on sin(pi x), fixed geometry, sweep lambda in
[0.1, 0.5]; pick the lambda that minimizes eval L_inf.

Step 2: at that best lambda, sweep width and target. Conditioning (cond, rank,
null) is target-independent, so it is reported once per width for GELU vs tanh;
the lstsq fit residual and eval error are per target.

Usage:
    python3 experiments/expA04_activation_conditioning/run.py            # collect + plot
    python3 experiments/expA04_activation_conditioning/run.py --plot      # plot only
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.construction.qi_mpmath import default_halo
from src.construction.readout import build_phi  # tanh
from src.data.targets import get_target
import gelu_phi as gp

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_A_numerics" / "expA04_activation_conditioning"
DATA_PATH = RESULTS_DIR / "data.json"

N_STEP1 = 128
LAMBDAS = np.linspace(0.1, 0.5, 21)
WIDTHS = [32, 64, 96, 128]
TARGETS = ["sine", "sine_8pi", "runge", "sine_mixture", "exp", "abs_cubed"]
N_EVAL = 2048

ACTS = {"gelu": gp.build_phi_gelu, "tanh": build_phi}


def _geometry(N, lam):
    h = 2.0 / N
    halo = default_halo(N, lambda_star=lam)
    nidx = np.arange(-halo, N + halo + 1)
    centers = -1.0 + nidx * h
    return centers, lam / h


def _solve_eval(build, x_tr, y_tr, x_ev, y_ev, gamma, centers):
    """lstsq solve + eval. Returns (eval_linf, ls_fit_rel, conditioning_dict)."""
    Phi = build(x_tr, np.full(len(centers), gamma), centers)
    A = np.hstack([Phi, np.ones((len(x_tr), 1))])
    cond = gp.conditioning(A)
    beta, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    A_ev = np.hstack([build(x_ev, np.full(len(centers), gamma), centers),
                      np.ones((len(x_ev), 1))])
    eval_linf = float(np.max(np.abs(A_ev @ beta - y_ev)))
    ls_fit = float(np.linalg.norm(A @ beta - y_tr) / (np.linalg.norm(y_tr) or 1.0))
    return eval_linf, ls_fit, cond


def collect_data():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x_ev = np.linspace(-1, 1, N_EVAL)

    # ---- Step 1: GELU lstsq on sin(pi x), lambda sweep ----
    print("Step 1: GELU lstsq on sin(pi x), lambda sweep")
    f1 = lambda x: np.sin(np.pi * x)
    y_ev1 = f1(x_ev)
    step1 = []
    for lam in LAMBDAS:
        centers, gamma = _geometry(N_STEP1, lam)
        n_tr = max(512, 2 * len(centers))
        x_tr = np.linspace(-1, 1, n_tr)
        y_tr = f1(x_tr)
        row = {"lambda": float(lam)}
        for name, build in ACTS.items():
            linf, lsf, cond = _solve_eval(build, x_tr, y_tr, x_ev, y_ev1, gamma, centers)
            row[f"{name}_linf"] = linf
            row[f"{name}_cond"] = cond["cond"]
            row[f"{name}_rank"] = cond["rank"]
            row[f"{name}_null"] = cond["null"]
        step1.append(row)
        print(f"  lam={lam:.3f}  gelu_linf={row['gelu_linf']:.2e} "
              f"(rank {row['gelu_rank']})  tanh_linf={row['tanh_linf']:.2e} "
              f"(rank {row['tanh_rank']})")

    best = min(step1, key=lambda r: r["gelu_linf"])
    best_lambda = best["lambda"]
    print(f"  -> best GELU lambda = {best_lambda:.3f} (eval L_inf {best['gelu_linf']:.2e})")

    # ---- Step 2: at best lambda, conditioning (per width) + errors (per target) ----
    print(f"\nStep 2: width/target sweep at lambda={best_lambda:.3f}")
    cond_rows, err_rows = [], []
    for N in WIDTHS:
        centers, gamma = _geometry(N, best_lambda)
        W = len(centers)
        n_tr = max(512, 2 * W)
        x_tr = np.linspace(-1, 1, n_tr)
        crow = {"N": N, "width": W}
        # Conditioning is target-independent: use a smooth probe target once.
        for name, build in ACTS.items():
            Phi = build(x_tr, np.full(W, gamma), centers)
            c = gp.conditioning(np.hstack([Phi, np.ones((n_tr, 1))]))
            crow[f"{name}_cond"] = c["cond"]
            crow[f"{name}_rank"] = c["rank"]
            crow[f"{name}_null"] = c["null"]
        cond_rows.append(crow)
        print(f"  N={N:>3} W={W}  GELU rank/null={crow['gelu_rank']}/{crow['gelu_null']} "
              f"cond={crow['gelu_cond']:.1e}  |  tanh rank/null={crow['tanh_rank']}/{crow['tanh_null']} "
              f"cond={crow['tanh_cond']:.1e}")
        for tname in TARGETS:
            t = get_target(tname)
            y_tr = t.fn_numpy(x_tr)
            y_ev = t.fn_numpy(x_ev)
            erow = {"target": tname, "N": N}
            for name, build in ACTS.items():
                linf, lsf, _ = _solve_eval(build, x_tr, y_tr, x_ev, y_ev, gamma, centers)
                erow[f"{name}_linf"] = linf
                erow[f"{name}_ls_fit"] = lsf
            err_rows.append(erow)

    out = {"step1": step1, "best_lambda": best_lambda,
           "cond_sweep": cond_rows, "err_sweep": err_rows,
           "N_step1": N_STEP1}
    with open(DATA_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {DATA_PATH}")
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _plot_step1(data):
    s1 = data["step1"]
    lams = [r["lambda"] for r in s1]
    best_lambda = data["best_lambda"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy(lams, [r["gelu_linf"] for r in s1], "-o", color="#1f77b4", lw=1.6,
                ms=4, label="GELU")
    ax.semilogy(lams, [r["tanh_linf"] for r in s1], "--s", color="#ff7f0e", lw=1.4,
                ms=4, label="tanh")
    ax.axvline(best_lambda, color="#1f77b4", ls=":", lw=1.0)
    ymin = min(r["gelu_linf"] for r in s1)
    ax.annotate(f"GELU min at $\\lambda$={best_lambda:.3f}\n(flat band -- no sharp optimum)",
                xy=(best_lambda, ymin), xytext=(best_lambda + 0.02, ymin * 30),
                fontsize=9, color="#1f77b4")
    ax.set_xlabel(r"$\lambda = \gamma h$")
    ax.set_ylabel(r"eval $L_\infty$ error  (lstsq fp64)")
    ax.set_title(rf"expA04 step 1: lstsq error vs $\lambda$ on $\sin(\pi x)$, N={data['N_step1']}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    caption = (
        "HOW TO READ: lstsq eval error vs bandwidth lambda for GELU (solid) and "
        "tanh (dashed) on the same geometry, target sin(pi x). Neither has a sharp "
        "optimum -- both are flat-bottomed in lambda until the high-lambda fp64 "
        "cancellation wall (see expC02/expC01). The dotted line marks the GELU "
        "minimum, but it is just a representative point in the flat band."
    )
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8,
             wrap=True, style="italic")
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out = RESULTS_DIR / "gelu_lambda_ushape.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_step2(data):
    cs = sorted(data["cond_sweep"], key=lambda r: r["N"])
    widths = [r["width"] for r in cs]
    best_lambda = data["best_lambda"]
    targets = [t for t in TARGETS if any(r["target"] == t for r in data["err_sweep"])]
    colors = {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(targets)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_cond, ax_rank), (ax_null, ax_err) = axes

    # (1) condition number, (2) rank, (3) null -- target-independent, GELU vs tanh.
    for name, color, ls, mk in (("gelu", "#1f77b4", "-", "o"), ("tanh", "#ff7f0e", "--", "s")):
        ax_cond.semilogy(widths, [r[f"{name}_cond"] for r in cs], ls, color=color,
                         marker=mk, lw=1.6, ms=5, label=name)
        ax_rank.plot(widths, [r[f"{name}_rank"] for r in cs], ls, color=color,
                     marker=mk, lw=1.6, ms=5, label=name)
        ax_null.plot(widths, [r[f"{name}_null"] for r in cs], ls, color=color,
                     marker=mk, lw=1.6, ms=5, label=name)
    ax_rank.plot(widths, [r["width"] + 1 for r in cs], ":", color="0.6", lw=1.0,
                 label="W+1 (full)")

    ax_cond.set_title("Effective condition number (target-independent)")
    ax_cond.set_ylabel(r"$\sigma_{\max}/\sigma_{\min}$ (kept)")
    ax_rank.set_title("Effective rank")
    ax_rank.set_ylabel("rank")
    ax_null.set_title("Null dimension")
    ax_null.set_ylabel(r"$\dim\ker(A)$")
    for ax in (ax_cond, ax_rank, ax_null):
        ax.set_xlabel("model width  W = N + 2*halo + 1")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    # (4) per-target lstsq fit + eval error (GELU); color = target.
    for t in targets:
        td = sorted([r for r in data["err_sweep"] if r["target"] == t], key=lambda r: r["N"])
        ws = [next(c["width"] for c in cs if c["N"] == r["N"]) for r in td]
        ax_err.semilogy(ws, [r["gelu_linf"] for r in td], "-", color=colors[t],
                        marker="o", ms=4, lw=1.3, label=t)
        ax_err.semilogy(ws, [r["tanh_linf"] for r in td], "--", color=colors[t],
                        marker="s", ms=3, lw=1.0)
    ax_err.set_title("eval $L_\\infty$ per target: GELU (solid) vs tanh (dashed)")
    ax_err.set_ylabel("error")
    ax_err.set_xlabel("model width  W = N + 2*halo + 1")
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.legend(fontsize=7, ncol=2, title="target")

    fig.suptitle(rf"expA04 step 2: GELU vs tanh conditioning at $\lambda$={best_lambda:.3f} (fp64)",
                 fontsize=14, y=0.99)
    caption = (
        "HOW TO READ: panels 1-3 are the feature-matrix conditioning, which is "
        "target-independent, so each shows GELU (solid) vs tanh (dashed) -- GELU "
        "has higher null dimension / lower rank (more collinear features). Panel 4 "
        "is the eval error per target: GELU (solid) vs tanh (dashed); tanh reaches "
        "the fp64 floor (~1e-13) while GELU is 1-3 orders worse. The dotted W+1 "
        "line in panel 2 is the full column count."
    )
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8,
             wrap=True, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = RESULTS_DIR / "gelu_vs_tanh_conditioning.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_results(data_path=None):
    data_path = Path(data_path) if data_path else DATA_PATH
    with open(data_path) as f:
        data = json.load(f)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _plot_step1(data)
    _plot_step2(data)


if __name__ == "__main__":
    if "--plot" not in sys.argv:
        collect_data()
    if DATA_PATH.exists():
        plot_results()
    else:
        print(f"No data at {DATA_PATH}. Run without --plot first.")
