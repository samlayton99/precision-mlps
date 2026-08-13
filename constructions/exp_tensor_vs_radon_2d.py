"""Head-to-head: tensor-product QI (no solve) vs Radon-ridge-lstsq on [-1,1]^2.

Two ways to reconstruct a 2D function with a tanh MLP:

- **tensor-QI** (this folder's ``TensorQI2D``): an EXPLICIT tensor-product
  convolution from tensor-Hermite edge/corner data. No linear solve, no
  conditioning wall -- the coefficients are the repo's cardinal filter applied to
  derivative samples.
- **Radon-ridge-lstsq** (the repo's existing 2D approach, expE01_geometry_zoo_2d):
  place J directions x M signed offsets of ridge neurons
  tanh(gamma*(w.x - t)) and solve the readout by fp64 SVD least squares. We reuse
  expE01's exact Phi + truncated-SVD lstsq machinery, adapted to the square
  [-1,1]^2 (expE01 works on the unit disk).

Both get a MATCHED neuron budget: tensor-QI at per-axis width N uses ~N^2 basis
units, so Radon gets J*M ~= (N+1)^2 ridges (J = M = N+1).

Metric: rel-L2 and L_inf on a dense interior grid over [-0.9, 0.9]^2, swept over
per-axis width N in {16, 32, 48, 64}. Targets: product sines, radial bump, x*y,
and an asymmetric mixed-scale target. One figure (error-vs-N, log-y; rel-L2 and
L_inf panels; one line per method per target), saved under results/ (gitignored).

Run:
    cd /scr/cdeng/precision-mlps
    python constructions/exp_tensor_vs_radon_2d.py            # full sweep + plot
    python constructions/exp_tensor_vs_radon_2d.py --smoke    # N in {16} only
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tensor_qi_2d import TensorQI2D  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "constructions" / "exp_tensor_vs_radon_2d"

# ---------------------------------------------------------------------------
# Target suite: (f, fx, fy, fxy) callables on stacked (P,2) points -> (P,).
# ---------------------------------------------------------------------------
_TP = 2.0 * np.pi


def _product_sines():
    return (
        lambda P: np.sin(_TP * P[:, 0]) * np.sin(_TP * P[:, 1]),
        lambda P: _TP * np.cos(_TP * P[:, 0]) * np.sin(_TP * P[:, 1]),
        lambda P: _TP * np.sin(_TP * P[:, 0]) * np.cos(_TP * P[:, 1]),
        lambda P: _TP * _TP * np.cos(_TP * P[:, 0]) * np.cos(_TP * P[:, 1]),
    )


def _radial_bump():
    e = lambda P: np.exp(-3.0 * (P[:, 0] ** 2 + P[:, 1] ** 2))
    return (
        lambda P: e(P),
        lambda P: -6.0 * P[:, 0] * e(P),
        lambda P: -6.0 * P[:, 1] * e(P),
        lambda P: 36.0 * P[:, 0] * P[:, 1] * e(P),
    )


def _x_times_y():
    return (
        lambda P: P[:, 0] * P[:, 1],
        lambda P: P[:, 1],
        lambda P: P[:, 0],
        lambda P: np.ones(len(P)),
    )


def _asymmetric():
    # f = exp(sin(pi x) + cos(2 pi y)); mixed-scale, x/y asymmetric.
    e = lambda P: np.exp(np.sin(np.pi * P[:, 0]) + np.cos(_TP * P[:, 1]))
    sx = lambda P: np.pi * np.cos(np.pi * P[:, 0])          # d/dx of exponent
    sy = lambda P: -_TP * np.sin(_TP * P[:, 1])             # d/dy of exponent
    return (
        lambda P: e(P),
        lambda P: sx(P) * e(P),
        lambda P: sy(P) * e(P),
        lambda P: sx(P) * sy(P) * e(P),                     # cross deriv (product rule)
    )


TARGETS = {
    "product_sines": _product_sines(),
    "radial_bump": _radial_bump(),
    "x*y": _x_times_y(),
    "asym_mixed": _asymmetric(),
}
TARGET_COLORS = {
    "product_sines": "#1f77b4", "radial_bump": "#2ca02c",
    "x*y": "#d62728", "asym_mixed": "#9467bd",
}

WIDTHS = [16, 32, 48, 64]
EVAL_SIDE = 60          # dense interior grid over [-0.9, 0.9]^2
RCOND = 1e-13           # truncated-SVD cutoff (matches expE01)
LAMBDA_GRID = [0.08, 0.12, 0.18, 0.26, 0.38, 0.55, 0.80]   # Radon lambda sweep

_xs = np.linspace(-0.9, 0.9, EVAL_SIDE)
_X, _Y = np.meshgrid(_xs, _xs)
EVAL_P = np.stack([_X.ravel(), _Y.ravel()], 1)


# ---------------------------------------------------------------------------
# Radon-ridge geometry on the SQUARE [-1,1]^2 (adapted from expE01 radon_tensor).
# ---------------------------------------------------------------------------
def _radon_square_geometry(J: int, M: int, T: float):
    """J directions x M signed offsets over [-T, T]; returns (dirs[Mn,2], offs[Mn]).

    Directions are equispaced in [0, pi) with a half-step offset to avoid the
    coordinate axes (same trick as expE01). The offset span T > 1 provides the
    node-count collar so interior [-1,1] ridges are well surrounded.
    """
    thetas = 0.5 * np.pi / J + np.arange(J) * np.pi / J
    h_step = 2.0 * T / M
    t1d = -T + (np.arange(M) + 0.5) * h_step
    dirs, offs = [], []
    for th in thetas:
        w = np.array([np.cos(th), np.sin(th)])
        for t in t1d:
            dirs.append(w)
            offs.append(t)
    return np.asarray(dirs), np.asarray(offs)


def _radon_lstsq(f_fn, N):
    """Radon-ridge fp64 SVD least squares, best over lambda. Reuses expE01's
    Phi + truncated-SVD readout, matched to ~ (N+1)^2 ridges."""
    J = N + 1
    M = N + 1
    T = 1.6                                          # offset collar (>1)
    dirs, offs = _radon_square_geometry(J, M, T)
    n_ridge = len(offs)
    h_ref = 2.0 / N                                  # same interior spacing as tensor-QI

    # Overdetermined training sample on [-1,1]^2 (> 2x neuron count).
    n_train = max(4 * n_ridge, 4000)
    rng = np.random.default_rng(0)
    X_tr = rng.uniform(-1.0, 1.0, (n_train, 2))
    y_tr = f_fn(X_tr)
    y_ev = f_fn(EVAL_P)
    y_norm = float(np.linalg.norm(y_ev))

    DXt = X_tr @ dirs.T                              # [n_train, n_ridge]
    DXe = EVAL_P @ dirs.T
    best = {"linf": np.inf, "rel_l2": np.inf, "lambda": None, "n_ridge": n_ridge}
    for lam in LAMBDA_GRID:
        gamma = lam / h_ref
        Phi_tr = np.tanh(gamma * (DXt - offs[None, :]))
        Phi_ev = np.tanh(gamma * (DXe - offs[None, :]))
        Aug = np.hstack([Phi_tr, np.ones((n_train, 1))])
        U, s, Vt = np.linalg.svd(Aug, full_matrices=False)
        s_inv = np.where(s > RCOND * s[0], 1.0 / s, 0.0)
        sol = Vt.T @ (s_inv * (U.T @ y_tr))
        resid = Phi_ev @ sol[:-1] + sol[-1] - y_ev
        linf = float(np.abs(resid).max())
        rel_l2 = float(np.linalg.norm(resid) / y_norm)
        if linf < best["linf"]:
            best.update({"linf": linf, "rel_l2": rel_l2, "lambda": lam})
    return best


def _tensor_qi(target, N, precision="fp64"):
    f, fx, fy, fxy = target
    qi = TensorQI2D(N, precision=precision).fit(f, fx, fy, fxy)
    pred = qi(EVAL_P)
    tgt = f(EVAL_P)
    y_norm = float(np.linalg.norm(tgt))
    rel_l2 = float(np.linalg.norm(pred - tgt) / y_norm)
    linf = float(np.abs(pred - tgt).max())
    return {"linf": linf, "rel_l2": rel_l2, "n_units": (N + 1) ** 2}


def run(widths, precision="fp64"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for name, target in TARGETS.items():
        f_fn = target[0]
        for N in widths:
            tq = _tensor_qi(target, N, precision=precision)
            rd = _radon_lstsq(f_fn, N)
            rows.append({"target": name, "N": N,
                         "tensor_rel_l2": tq["rel_l2"], "tensor_linf": tq["linf"],
                         "radon_rel_l2": rd["rel_l2"], "radon_linf": rd["linf"],
                         "radon_lambda": rd["lambda"], "n_ridge": rd["n_ridge"]})
            print(f"[{time.time()-t0:5.0f}s] {name:14s} N={N:3d} | "
                  f"tensor rel={tq['rel_l2']:.2e} Linf={tq['linf']:.2e} | "
                  f"radon rel={rd['rel_l2']:.2e} Linf={rd['linf']:.2e} "
                  f"(lam={rd['lambda']})", flush=True)
    return rows


def summary_table(rows):
    print("\n" + "=" * 78)
    print(f"{'target':14s} {'N':>3s} | {'tensor relL2':>13s} {'radon relL2':>13s} "
          f"| {'tensor Linf':>13s} {'radon Linf':>13s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['target']:14s} {r['N']:3d} | {r['tensor_rel_l2']:13.2e} "
              f"{r['radon_rel_l2']:13.2e} | {r['tensor_linf']:13.2e} "
              f"{r['radon_linf']:13.2e}")
    print("=" * 78)


def plot(rows, widths):
    metrics = [("rel_l2", r"Relative $L_2$"), ("linf", r"$L_\infty$")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    for ci, (mkey, mlabel) in enumerate(metrics):
        ax = axes[ci]
        for name in TARGETS:
            col = TARGET_COLORS[name]
            tens = [next(r[f"tensor_{mkey}"] for r in rows
                         if r["target"] == name and r["N"] == N) for N in widths]
            rad = [next(r[f"radon_{mkey}"] for r in rows
                        if r["target"] == name and r["N"] == N) for N in widths]
            ax.semilogy(widths, tens, "-o", color=col, markersize=5,
                        label=f"{name} (tensor-QI)")
            ax.semilogy(widths, rad, "--s", color=col, markersize=5,
                        markerfacecolor="none", label=f"{name} (Radon-lstsq)")
        ax.set_xlabel("per-axis width $N$")
        ax.set_ylabel(mlabel)
        ax.set_xticks(widths)
        ax.grid(True, alpha=0.3, which="both")
        # legend ABOVE the axes (CLAUDE.md convention)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                  fontsize=7.5, borderaxespad=0, frameon=False)
    fig.suptitle("Tensor-product QI (no solve) vs Radon-ridge lstsq on "
                 r"$[-1,1]^2$ (fp64)", y=0.02, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.86])
    out = RESULTS_DIR / "tensor_vs_radon.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out}")


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    widths = [16] if smoke else WIDTHS
    prec = "fp64"
    rows = run(widths, precision=prec)
    summary_table(rows)
    plot(rows, widths)
