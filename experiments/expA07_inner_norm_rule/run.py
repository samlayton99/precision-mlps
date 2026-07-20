"""expA07 -- the inner-band norm rule: what magnitude should the readout start at?

Sweeps the SVD min-norm readout on the QI geometry (lambda*=0.25) and records the
L2 norm of the INNER coefficients only (centers in [-1,1], halo excluded), over
width N = 32..1024 (linspace, step 32), for the 6-target family x {tanh, gelu}.

Fits candidate laws per curve (power / power-to-floor / exponential / power-log) and
also overlays the ZERO-free-parameter predictions from the derivative law:
  tanh:  ||v_in|| = ||f'||_L2 / sqrt(2N)          (slope -1/2, prefactor known)
  gelu:  ||v_in|| = (2/N)^{3/2} ||f''||_L2 / lam  (slope -3/2, prefactor known)

Figures: one 2x3 log-log grid per activation (data + best fit + prediction).

Run:  uv run python experiments/expA07_inner_norm_rule/run.py   (--smoke, --fig-only)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expA06_readout_structure"))
from app import build_phi, geometry, make_fn   # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint_A_numerics" / "expA07_inner_norm_rule"
LAM = 0.25
N_GRID = list(range(32, 1025, 32))             # linspace, 32 points
ACTS = ["tanh", "gelu"]
TARGETS = {
    "sine": "sin(2*pi*x)", "sine_8pi": "sin(8*pi*x)", "runge": "1/(1+25*x**2)",
    "sine_mixture": "sin(2*pi*x)+0.5*sin(6*pi*x)+0.25*sin(14*pi*x)",
    "exp": "exp(x)", "abs_cubed": "Abs(x)**3",
}
_X = sp.Symbol("x", real=True)


def deriv_l2(text, order):
    """Continuous L2 norm of f^(order) over [-1,1] by fine quadrature."""
    e = sp.sympify(text, locals={"x": _X})
    # DiracDelta appears in d^2/dx^2 of Abs terms (sign' = 2*delta); it is zero a.e.,
    # so map it to 0 for quadrature purposes.
    d = sp.lambdify(_X, sp.diff(e, _X, order),
                    modules=[{"DiracDelta": lambda *a: 0.0}, "numpy"])
    xs = np.linspace(-1, 1, 200001)
    vals = np.asarray(d(xs), dtype=np.float64)
    vals = np.broadcast_to(vals, xs.shape)
    return float(np.sqrt(np.trapezoid(vals ** 2, xs)))


def solve_cell(job):
    from scipy.linalg import lstsq as sp_lstsq
    act, fn_name, N = job["act"], job["fn"], job["N"]
    f_vec = make_fn(TARGETS[fn_name])
    centers, gamma, _ = geometry(N, LAM)
    x = np.linspace(-1, 1, max(2003, 2 * centers.size + 3))
    A = np.hstack([build_phi(x, gamma, centers, act), np.ones((x.size, 1))])
    sol, *_ = sp_lstsq(A, f_vec(x), lapack_driver="gelsd")
    v = sol[:-1]
    inner = np.abs(centers) <= 1.0
    return {"act": act, "fn": fn_name, "N": N,
            "inner_norm": float(np.linalg.norm(v[inner])),
            "full_norm": float(np.linalg.norm(v))}


# ----------------------------- fits -----------------------------

FORMS = {
    "power":      (lambda N, a, b: a + b * np.log10(N), [0, -0.5]),
    "powerfloor": (lambda N, a, b, c: np.log10(10.0**a * np.power(N, b) + 10.0**c + 1e-300), [0, -0.5, -3]),
    "exponential": (lambda N, a, b: a + b * N, [0, -1e-3]),
    "powerlog":   (lambda N, a, b, c: a + b * np.log10(N) + c * np.log10(np.log(N)), [0, -0.5, 0]),
}


def _r2(t, t_hat):
    ss = np.sum((t - t.mean()) ** 2)
    return 1.0 - np.sum((t - t_hat) ** 2) / ss if ss > 0 else np.nan


def fit_curves(rows):
    warnings.filterwarnings("ignore")
    out = {}
    for act in ACTS:
        for fn in TARGETS:
            pts = sorted([r for r in rows if r["act"] == act and r["fn"] == fn],
                         key=lambda r: r["N"])
            if len(pts) < 5:
                continue
            N = np.array([r["N"] for r in pts], float)
            t = np.log10(np.array([r["inner_norm"] for r in pts]) + 1e-300)
            res = {}
            for name, (fm, p0) in FORMS.items():
                try:
                    popt, _ = curve_fit(fm, N, t, p0=p0, maxfev=20000)
                    res[name] = {"r2": float(_r2(t, fm(N, *popt))),
                                 "params": [float(p) for p in popt]}
                except Exception:
                    res[name] = {"r2": None, "params": None}
            out[(act, fn)] = res
    return out


def prediction(act, fn, N):
    N = np.asarray(N, float)
    if act == "tanh":
        return deriv_l2(TARGETS[fn], 1) / np.sqrt(2.0 * N)
    return (2.0 / N) ** 1.5 * deriv_l2(TARGETS[fn], 2) / LAM


# ----------------------------- figures -----------------------------

def make_figures(rows, fits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    for act in ACTS:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, fn in zip(axes.flat, TARGETS):
            pts = sorted([r for r in rows if r["act"] == act and r["fn"] == fn],
                         key=lambda r: r["N"])
            if not pts:
                continue
            N = np.array([r["N"] for r in pts], float)
            y = np.array([r["inner_norm"] for r in pts])
            ax.loglog(N, y, "o", ms=4, color="#1f4e8c", label="data")
            # best empirical fit
            res = fits.get((act, fn), {})
            ok = {k: v for k, v in res.items() if v["r2"] is not None}
            if ok:
                best = max(ok, key=lambda k: ok[k]["r2"])
                fm, _ = FORMS[best]
                Nf = np.geomspace(N.min(), N.max(), 200)
                ax.loglog(Nf, 10 ** fm(Nf, *ok[best]["params"]), "-", lw=1.4,
                          color="#d1352b",
                          label=f"best fit: {best} (R2={ok[best]['r2']:.3f})")
            else:
                pass
            # tail slope (N >= 384): the asymptotic law, transient excluded
            m = N >= 384
            tail = float(np.polyfit(np.log10(N[m]), np.log10(y[m] + 1e-300), 1)[0]) if m.sum() > 2 else None
            title_extra = f"  tail slope={tail:.2f}" if tail is not None else ""
            # zero-parameter prediction
            Nf = np.geomspace(N.min(), N.max(), 200)
            ax.loglog(Nf, prediction(act, fn, Nf), "--", lw=1.4, color="#2ca02c",
                      label="derivative-law prediction (0 params)")
            ax.set_title(fn + title_extra, fontsize=10)
            ax.set_xlabel("width N"); ax.grid(True, which="both", alpha=0.25)
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), fontsize=7,
                      ncol=1, frameon=False)
        axes[0, 0].set_ylabel(r"$\|v_{inner}\|_2$")
        axes[1, 0].set_ylabel(r"$\|v_{inner}\|_2$")
        fig.suptitle(f"expA07 -- inner-band readout norm vs width ({act}, lambda*=0.25)",
                     y=1.00, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = OUT_DIR / "figures" / f"expA07_inner_norm_{act}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {p}")


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fig-only", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    n_grid = [32, 128, 512] if args.smoke else N_GRID
    rows_path = OUT_DIR / "expA07_rows.json"

    if args.fig_only:
        rows = json.loads(rows_path.read_text())
    else:
        jobs = sorted([{"act": a, "fn": f, "N": n}
                       for a in ACTS for f in TARGETS for n in n_grid],
                      key=lambda j: j["N"])
        print(f"expA07 | {len(jobs)} solves | acts={ACTS} | N={n_grid[0]}..{n_grid[-1]} "
              f"({len(n_grid)} pts, linspace)")
        rows, t0 = [], time.time()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(solve_cell, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if i % 50 == 0 or i == len(jobs):
                    rows_path.write_text(json.dumps(rows))
                    print(f"  [{i}/{len(jobs)}] ({time.time()-t0:.0f}s)", flush=True)

    fits = fit_curves(rows)
    print("\nper-curve best form and fitted power-law slope:")
    print(f"{'act':6s} {'target':13s} {'best form':>11s} {'R2':>7s} {'power slope':>12s} "
          f"{'pred slope':>10s} {'pred R2':>8s}")
    for act in ACTS:
        for fn in TARGETS:
            res = fits.get((act, fn), {})
            ok = {k: v for k, v in res.items() if v["r2"] is not None}
            if not ok:
                continue
            best = max(ok, key=lambda k: ok[k]["r2"])
            pw = res["power"]
            slope = pw["params"][1] if pw["params"] else np.nan
            # R2 of the zero-parameter prediction
            pts = sorted([r for r in rows if r["act"] == act and r["fn"] == fn],
                         key=lambda r: r["N"])
            N = np.array([r["N"] for r in pts], float)
            t = np.log10(np.array([r["inner_norm"] for r in pts]) + 1e-300)
            pr2 = _r2(t, np.log10(prediction(act, fn, N) + 1e-300))
            print(f"{act:6s} {fn:13s} {best:>11s} {ok[best]['r2']:>7.4f} {slope:>12.3f} "
                  f"{-0.5 if act=='tanh' else -1.5:>10.1f} {pr2:>8.4f}")
    (OUT_DIR / "expA07_fits.json").write_text(json.dumps(
        {f"{a}|{f}": v for (a, f), v in fits.items()}))
    make_figures(rows, fits)


if __name__ == "__main__":
    main()
