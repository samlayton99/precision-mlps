"""expA07/smoothness -- does the norm law lose predictive power when the target
lacks the kernel-order derivative?

The generalized inner-band norm law, by kernel order r (kernel K = psi^(r)),
with each activation at its expC07 aliasing-rule lambda:

  r=0 (sech2, gaussian):  v_k ~ (lam/M) f(c_k)        ||v_in|| = (lam/M) sqrt(N/2) ||f||      slope +1/2
  r=1 (tanh, sigmoid):    v_k ~ (h/Delta) f'(c_k)     ||v_in|| = (1/Delta) sqrt(2/N) ||f'||   slope -1/2
  r=2 (gelu, swish):      v_k ~ (h^2/(s lam)) f''(c_k) ||v_in|| = (2/N)^{3/2} ||f''||/(s lam) slope -3/2

  M = int K (sech2: 2, gaussian: sqrt(pi));  Delta = psi(inf)-psi(-inf) (tanh: 2,
  sigmoid: 1);  s = asymptotic ramp slope (gelu, swish: 1).  All L2 norms on [-1,1].

Targets span smoothness classes C0 (kinked), C1 (one derivative), C2 (two),
two sets of three (one figure each). The law needs ||f^(r)||_L2 < inf: for r=2 on
C0 targets f'' contains Dirac deltas and the prediction does not exist -- the
cells where predictive power should be LOST.

Figures: 3x3 grid per target set; row = kernel order (both activations of that
order per panel), column = smoothness class; log-log ||v_inner|| vs N, dashed
zero-parameter law lines, tail slopes (N>=384) annotated.

Run:  uv run python experiments/expA07_inner_norm_rule/smoothness/run.py   (--smoke, --fig-only)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expA06_readout_structure"))
from app import geometry, make_fn   # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint_A_numerics" / "expA07_inner_norm_rule" / "smoothness"
N_GRID = list(range(32, 1025, 32))

# activation -> (kernel order r, expC07 aliasing-rule lambda, law prefactor C)
# law: ||v_in|| = C * ||f^(r)||_L2 * (2/N)^(r - 1/2)
ACT = {
    "sech2":    (0, 0.25,   0.25 / 2.0),
    "gaussian": (0, 0.5302, 0.5302 / np.sqrt(np.pi)),
    "tanh":     (1, 0.25,   1.0 / 2.0),
    "sigmoid":  (1, 0.50,   1.0),
    "gelu":     (2, 0.7070, 1.0 / 0.7070),
    "swish":    (2, 0.4554, 1.0 / 0.4554),
}
ORDER_ROWS = [("order 0 (bump)", ["sech2", "gaussian"]),
              ("order 1 (step)", ["tanh", "sigmoid"]),
              ("order 2 (ramp)", ["gelu", "swish"])]

# target sets: {name: (sympy expr, smoothness class)}; one figure per set
SETS = {
    "A": {"abs":            ("Abs(x)", 0),
          "xabs":           ("x*Abs(x)", 1),
          "abs_cubed":      ("Abs(x)**3", 2)},
    "B": {"abs_sine":       ("Abs(sin(2*pi*x))", 0),
          "sine_xabs":      ("sin(2*pi*x)*Abs(sin(2*pi*x))", 1),
          "abs_sine_cubed": ("Abs(sin(2*pi*x))**3", 2)},
}
TARGETS = {k: v[0] for s in SETS.values() for k, v in s.items()}
SMOOTH = {k: v[1] for s in SETS.values() for k, v in s.items()}
_X = sp.Symbol("x", real=True)


def deriv_l2(text, order):
    """L2 norm of the a.e. r-th derivative on [-1,1] (deltas dropped); None if the
    law is undefined (delta content, i.e. order exceeds class by 2+)."""
    if order - SMOOTH[[k for k, t in TARGETS.items() if t == text][0]] >= 2:
        return None
    e = sp.sympify(text, locals={"x": _X})
    d = sp.lambdify(_X, sp.diff(e, _X, order),
                    modules=[{"DiracDelta": lambda *a: 0.0}, "numpy"])
    xs = np.linspace(-1, 1, 400001)
    vals = np.broadcast_to(np.asarray(d(xs), dtype=np.float64), xs.shape)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)  # isolated kink pts
    return float(np.sqrt(np.trapezoid(vals ** 2, xs)))


def apply_act(z, name):
    from scipy.special import erf, expit
    if name == "sech2":
        return 1.0 / np.cosh(np.clip(z, -300, 300)) ** 2
    if name == "gaussian":
        return np.exp(-z ** 2)
    if name == "sigmoid":
        return expit(z)
    if name == "gelu":
        return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
    if name == "swish":
        return z * expit(z)
    return np.tanh(z)


def prediction(act, fn, N):
    r, _, C = ACT[act]
    nrm = deriv_l2(TARGETS[fn], r)
    if nrm is None:
        return None
    return C * nrm * (2.0 / np.asarray(N, float)) ** (r - 0.5)


def solve_cell(job):
    from scipy.linalg import lstsq as sp_lstsq
    act, fn_name, N = job["act"], job["fn"], job["N"]
    r, lam, _ = ACT[act]
    f_vec = make_fn(TARGETS[fn_name])
    centers, gamma, _ = geometry(N, lam)      # halo sized at reference lambda 0.25
    x = np.linspace(-1, 1, max(2003, 2 * centers.size + 3))
    Phi = apply_act(gamma * (x[:, None] - centers[None, :]), act)
    A = np.hstack([Phi, np.ones((x.size, 1))])
    sol, *_ = sp_lstsq(A, f_vec(x), lapack_driver="gelsd")
    v = sol[:-1]
    inner = np.abs(centers) <= 1.0
    # eval error for context (rough targets will not reach the floor)
    xe = np.linspace(-1, 1, 4001)
    Phi_e = apply_act(gamma * (xe[:, None] - centers[None, :]), act)
    resid = Phi_e @ v + sol[-1] - f_vec(xe)
    rel = float(np.linalg.norm(resid) / np.linalg.norm(f_vec(xe)))
    return {"act": act, "fn": fn_name, "N": N,
            "inner_norm": float(np.linalg.norm(v[inner])),
            "full_norm": float(np.linalg.norm(v)), "rel_l2": rel}


def tail_slope(N, y):
    m = N >= 384
    if m.sum() < 3:
        return None
    return float(np.polyfit(np.log10(N[m]), np.log10(y[m] + 1e-300), 1)[0])


def make_figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = ["#1f4e8c", "#d1352b"]
    cls_name = {0: "C0 (kinked)", 1: "C1", 2: "C2"}
    for set_name, targets in SETS.items():
        fns = sorted(targets, key=lambda k: targets[k][1])
        fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.5), sharex=True)
        for i, (row_label, acts) in enumerate(ORDER_ROWS):
            for j, fn in enumerate(fns):
                ax = axes[i, j]
                handles = []
                for act, col in zip(acts, colors):
                    pts = sorted([r for r in rows if r["act"] == act and r["fn"] == fn],
                                 key=lambda r: r["N"])
                    if not pts:
                        continue
                    N = np.array([r["N"] for r in pts], float)
                    y = np.array([r["inner_norm"] for r in pts])
                    sl = tail_slope(N, y)
                    lbl = f"{act} (tail {sl:+.2f})" if sl is not None else act
                    ax.loglog(N, y, "o", ms=3.5, color=col)
                    handles.append(Line2D([0], [0], marker="o", color=col, lw=0,
                                          ms=4, label=lbl))
                    pred = prediction(act, fn, N)
                    if pred is not None:
                        ax.loglog(N, pred, "--", lw=1.4, color=col)
                r = ACT[acts[0]][0]
                if prediction(acts[0], fn, np.array([64.0])) is None:
                    ax.text(0.5, 0.08, r"law undefined: $\|f^{(%d)}\|_{L_2}=\infty$" % r,
                            transform=ax.transAxes, ha="center", fontsize=9,
                            color="#666666")
                pred_slope = r - 0.5
                handles.append(Line2D([0], [0], ls="--", color="gray", lw=1.4,
                                      label=f"law (slope {-pred_slope:+.1f})"))
                ax.legend(handles=handles, loc="lower center",
                          bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=7.5,
                          frameon=False, borderaxespad=0)
                ax.grid(True, which="both", alpha=0.25)
                if i == 0:
                    ax.set_title(f"{fn}   [{cls_name[targets[fn][1]]}]",
                                 fontsize=10, pad=32)
                if i == 2:
                    ax.set_xlabel("width N")
                if j == 0:
                    ax.set_ylabel(f"{row_label}\n" + r"$\|v_{inner}\|_2$")
        fig.suptitle(f"expA07/smoothness -- norm law vs target smoothness, set {set_name} "
                     "(rows: kernel order, cols: smoothness class)", y=1.005, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        p = OUT_DIR / "figures" / f"expA07s_norm_law_set{set_name}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {p}")


def make_ratio_figures(rows):
    """Compensated view: measured / predicted vs N. Riding the law = flat at 1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = ["#1f4e8c", "#d1352b"]
    cls_name = {0: "C0 (kinked)", 1: "C1", 2: "C2"}
    for set_name, targets in SETS.items():
        fns = sorted(targets, key=lambda k: targets[k][1])
        fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.5), sharex=True)
        for i, (row_label, acts) in enumerate(ORDER_ROWS):
            for j, fn in enumerate(fns):
                ax = axes[i, j]
                handles = []
                for act, col in zip(acts, colors):
                    pts = sorted([r for r in rows if r["act"] == act and r["fn"] == fn],
                                 key=lambda r: r["N"])
                    pred = prediction(act, fn, np.array([r["N"] for r in pts], float))
                    if not pts or pred is None:
                        continue
                    N = np.array([r["N"] for r in pts], float)
                    y = np.array([r["inner_norm"] for r in pts]) / pred
                    ax.loglog(N, y, "o-", ms=3.5, lw=0.8, color=col, alpha=0.85)
                    handles.append(Line2D([0], [0], marker="o", color=col, lw=0.8,
                                          ms=4, label=act))
                r = ACT[acts[0]][0]
                if prediction(acts[0], fn, np.array([64.0])) is None:
                    ax.text(0.5, 0.5, r"law undefined: $\|f^{(%d)}\|_{L_2}=\infty$" % r,
                            transform=ax.transAxes, ha="center", fontsize=10,
                            color="#666666")
                ax.axhline(1.0, color="k", lw=1.2, ls="--")
                handles.append(Line2D([0], [0], ls="--", color="k", lw=1.2,
                                      label="on the law (ratio 1)"))
                ax.legend(handles=handles, loc="lower center",
                          bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8,
                          frameon=False, borderaxespad=0)
                ax.grid(True, which="both", alpha=0.25)
                if i == 0:
                    ax.set_title(f"{fn}   [{cls_name[targets[fn][1]]}]",
                                 fontsize=10, pad=32)
                if i == 2:
                    ax.set_xlabel("width N")
                if j == 0:
                    ax.set_ylabel(f"{row_label}\n" + r"$\|v\|_{meas}\, /\, \|v\|_{law}$")
        fig.suptitle(f"expA07/smoothness -- COMPENSATED: measured/predicted norm, set {set_name} "
                     "(flat at 1 = zero-parameter law holds)", y=1.005, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        p = OUT_DIR / "figures" / f"expA07s_ratio_set{set_name}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {p}")


def summarize(rows):
    print("\ntail slope (N>=384) and data/prediction ratio at N=1024:")
    print(f"{'act':9s} {'r':>2s} " + " ".join(f"{fn:>16s}" for fn in TARGETS))
    print(f"{'':12s}" + " ".join(f"{'slope ratio':>16s}" for _ in TARGETS))
    for act in ACT:
        r, _, _ = ACT[act]
        line = f"{act:9s} {r:>2d} "
        for fn in TARGETS:
            pts = sorted([q for q in rows if q["act"] == act and q["fn"] == fn],
                         key=lambda q: q["N"])
            if not pts:
                line += f"{'--':>16s} "
                continue
            N = np.array([q["N"] for q in pts], float)
            y = np.array([q["inner_norm"] for q in pts])
            sl = tail_slope(N, y)
            pred = prediction(act, fn, np.array([N[-1]]))
            ratio = y[-1] / pred[0] if pred is not None else np.nan
            sl_s = f"{sl:+.2f}" if sl is not None else "  --"
            ra_s = f"{ratio:9.3g}" if np.isfinite(ratio) else "      inf"
            line += f"{sl_s:>6s} {ra_s} "
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fig-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    n_grid = [128, 512] if args.smoke else N_GRID
    rows_path = OUT_DIR / "expA07s_rows.json"

    if args.fig_only:
        rows = json.loads(rows_path.read_text())
    else:
        jobs = sorted([{"act": a, "fn": f, "N": n}
                       for a in ACT for f in TARGETS for n in n_grid],
                      key=lambda j: j["N"])
        print(f"expA07/smoothness | {len(jobs)} solves | acts={list(ACT)} | "
              f"targets={list(TARGETS)} | N={n_grid[0]}..{n_grid[-1]} ({len(n_grid)} pts)")
        rows, t0 = [], time.time()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(solve_cell, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if i % 100 == 0 or i == len(jobs):
                    rows_path.write_text(json.dumps(rows))
                    print(f"  [{i}/{len(jobs)}] ({time.time()-t0:.0f}s)", flush=True)

    summarize(rows)
    if not args.smoke:
        make_figures(rows)
        make_ratio_figures(rows)


if __name__ == "__main__":
    main()
