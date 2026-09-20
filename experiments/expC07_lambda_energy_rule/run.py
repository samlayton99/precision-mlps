"""expC07 -- what invariant of the kernel predicts the optimal lambda per activation?

Hypothesis (Sam): anchored at tanh's lambda* = 0.25, the optimal lambda of any other
activation follows from matching an energy-like invariant of its KERNEL -- the r-th
derivative sigma^(r) that does the approximating -- to tanh's kernel sech^2.

Resolution (matches the data): the invariant is NOT real-space energy but the
kernel's FOURIER TAIL at the first grid harmonic -- Theorem 1's intrinsic-aliasing
term. With Khat the kernel FT, the normalized aliasing at bandwidth lambda is

    A(lambda) = |Khat(2*pi/lambda)| / |Khat(0)|,

and the predicted optimum solves A(lambda) = eps* where eps* = A_tanh(0.25)
= 4*pi^2/sinh(4*pi^2) = 5.65e-16 (machine eps -- tanh's 0.25 is exactly where
intrinsic aliasing e^{-pi^2/lambda} hits fp64 eps). Derivative order r only
multiplies Khat by (i*omega)^r -- a log correction; what sets lambda* is the
kernel's analyticity class (pole distance, or entire/Gaussian).

Aliasing-rule predictions: sech2 0.25 | gaussian 0.5302 | tanh 0.25 |
    sigmoid 0.50 (exact, affine identity) | gelu 0.7070 | swish 0.4554.
Energy-rule (mass-normalized width, original hypothesis, shown for contrast):
    sech2 0.25 | gaussian 0.2089 | tanh 0.25 | sigmoid 0.50 | gelu 0.1074 |
    swish 0.1881.

Sweep: 6 activations x 4 targets x lambda in geomspace(0.02, 1.5, 36) x N in
{64, 128, 256}; frozen QI geometry, SVD lstsq readout, eval rel L2 on 4001 points.
Figures: one 2x3 grid per target (panel = activation); 3 width lines; vertical line
at the predicted lambda; dot at each line's measured minimum.

Run:  uv run python experiments/expC07_lambda_energy_rule/run.py   (--smoke, --fig-only)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expA06_readout_structure"))
from app import geometry, make_fn   # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint_C_geometry" / "expC07_lambda_energy_rule"
LAM_GRID = [float(f"{l:.5g}") for l in np.geomspace(0.02, 1.5, 36)]
WIDTHS = [64, 128, 256]
TARGETS = {"sine": "sin(2*pi*x)", "sine_8pi": "sin(8*pi*x)",
           "runge": "1/(1+25*x**2)", "exp": "exp(x)"}

# activation -> predicted lambda from the aliasing rule A(lambda) = eps*
PRED = {"sech2": 0.25, "gaussian": 0.5302, "tanh": 0.25,
        "sigmoid": 0.5, "gelu": 0.7070, "swish": 0.4554}
# original energy-rule predictions (mass-normalized width match), for contrast
PRED_ENERGY = {"sech2": 0.25, "gaussian": 0.20889, "tanh": 0.25,
               "sigmoid": 0.5, "gelu": 0.10742, "swish": 0.18811}
ACT_ORDER = ["sech2", "gaussian", "tanh", "sigmoid", "gelu", "swish"]


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


def solve_cell(job):
    from scipy.linalg import lstsq as sp_lstsq
    act, fn_name, lam, N = job["act"], job["fn"], job["lam"], job["N"]
    f_vec = make_fn(TARGETS[fn_name])
    centers, gamma, _ = geometry(N, lam)      # halo sized at reference lambda 0.25
    x = np.linspace(-1, 1, max(2003, 2 * centers.size + 3))
    Phi = apply_act(gamma * (x[:, None] - centers[None, :]), act)
    A = np.hstack([Phi, np.ones((x.size, 1))])
    sol, *_ = sp_lstsq(A, f_vec(x), lapack_driver="gelsd")
    xe = np.linspace(-1, 1, 4001)
    Phi_e = apply_act(gamma * (xe[:, None] - centers[None, :]), act)
    resid = Phi_e @ sol[:-1] + sol[-1] - f_vec(xe)
    rel = float(np.linalg.norm(resid) / np.linalg.norm(f_vec(xe)))
    return {"act": act, "fn": fn_name, "lam": lam, "N": N, "rel_l2": rel}


def make_figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {64: "tab:blue", 128: "tab:orange", 256: "tab:green"}
    for fn in TARGETS:
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
        for ax, act in zip(axes.flat, ACT_ORDER):
            for N in WIDTHS:
                pts = sorted([r for r in rows if r["act"] == act and r["fn"] == fn
                              and r["N"] == N], key=lambda r: r["lam"])
                if not pts:
                    continue
                lam = np.array([r["lam"] for r in pts])
                rel = np.array([r["rel_l2"] for r in pts])
                ax.loglog(lam, rel, "-", color=colors[N], lw=1.4)
                i = int(np.argmin(rel))
                ax.plot(lam[i], rel[i], "o", color=colors[N], ms=6,
                        mec="k", mew=0.5, zorder=5)
            ax.axvline(PRED[act], color="#d1352b", lw=1.6, ls="--")
            ax.axvline(PRED_ENERGY[act], color="gray", lw=1.2, ls=":")
            ax.set_title(f"{act}   (aliasing-rule lambda = {PRED[act]:.3f})", fontsize=10)
            ax.set_xlabel("lambda"); ax.grid(True, which="both", alpha=0.25)
        for ax in axes[:, 0]:
            ax.set_ylabel("eval rel $L_2$")
        handles = [Line2D([0], [0], color=colors[N], lw=1.6, label=f"N={N}") for N in WIDTHS]
        handles += [Line2D([0], [0], color="#d1352b", lw=1.6, ls="--", label="aliasing-rule prediction"),
                    Line2D([0], [0], color="gray", lw=1.2, ls=":", label="energy-rule (original)"),
                    Line2D([0], [0], marker="o", color="gray", lw=0, ms=6, label="measured minimum")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=5, frameon=False)
        fig.suptitle(f"expC07 -- lambda U-curves vs aliasing-rule prediction, target = {fn}",
                     y=1.06, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = OUT_DIR / "figures" / f"expC07_ucurves_{fn}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {p}")


def summarize(rows):
    print("\nmeasured argmin lambda vs prediction (per activation, N=256, by target):")
    print(f"{'act':9s} {'pred':>6s} " + " ".join(f"{fn:>10s}" for fn in TARGETS))
    for act in ACT_ORDER:
        mins = []
        for fn in TARGETS:
            pts = sorted([r for r in rows if r["act"] == act and r["fn"] == fn and r["N"] == 256],
                         key=lambda r: r["lam"])
            mins.append(pts[int(np.argmin([r["rel_l2"] for r in pts]))]["lam"] if pts else np.nan)
        print(f"{act:9s} {PRED[act]:>6.3f} " + " ".join(f"{m:>10.3f}" for m in mins))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fig-only", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    lam_grid, widths, fns = LAM_GRID, WIDTHS, list(TARGETS)
    if args.smoke:
        lam_grid, widths, fns = [0.05, 0.25, 0.5], [64], ["sine"]

    rows_path = OUT_DIR / "expC07_rows.json"
    if args.fig_only:
        rows = json.loads(rows_path.read_text())
    else:
        jobs = sorted([{"act": a, "fn": f, "lam": l, "N": n}
                       for a in ACT_ORDER for f in fns for l in lam_grid for n in widths],
                      key=lambda j: j["N"])
        print(f"expC07 | {len(jobs)} solves | acts={ACT_ORDER} | targets={fns} | "
              f"lambda {lam_grid[0]}..{lam_grid[-1]} ({len(lam_grid)} pts) | N={widths}")
        rows, t0 = [], time.time()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(solve_cell, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if i % 200 == 0 or i == len(jobs):
                    rows_path.write_text(json.dumps(rows))
                    print(f"  [{i}/{len(jobs)}] ({time.time()-t0:.0f}s)", flush=True)

    summarize(rows)
    make_figures(rows)


if __name__ == "__main__":
    main()
