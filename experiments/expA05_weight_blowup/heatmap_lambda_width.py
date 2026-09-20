"""expA05 supplement -- lambda x width heatmap of the solved readout-coefficient norm.

For each target in the preset family, sweeps the SVD min-norm readout over
  width N  in logspace 32..1024   (16 points)
  lambda   in linspace 0.02..0.5  (25 points)
and records log10 ||v||_2 of the solved coefficients (tanh by default; --act to
change). Renders a 2x4 grid of heatmaps (7 targets + median-across-targets), color =
log10 ||v||, with a line at lambda*=0.25. Incremental row saves, so a partial sweep
still renders.

Run:  uv run python experiments/expA05_weight_blowup/heatmap_lambda_width.py
      (--smoke for a coarse pipeline check; --act gelu; --fig-only to re-render)
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
from app import PRESETS, build_phi, geometry, make_fn   # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint_A_numerics" / "expA05_weight_blowup"
FN_SHORT = {PRESETS[0]: "sine", PRESETS[1]: "sine_8pi", PRESETS[2]: "runge",
            PRESETS[3]: "sine_mixture", PRESETS[4]: "exp", PRESETS[5]: "poly5",
            PRESETS[6]: "abs_cubed"}

N_GRID = sorted(set(int(round(n)) for n in np.geomspace(32, 1024, 16)))
LAM_GRID = [round(l, 3) for l in np.linspace(0.02, 0.5, 25)]


def solve_cell(job):
    from scipy.linalg import lstsq as sp_lstsq
    act, fn_text, lam, N = job["act"], job["fn"], job["lam"], job["N"]
    f_vec = make_fn(fn_text)
    centers, gamma, _ = geometry(N, lam)
    W = centers.size
    x = np.linspace(-1.0, 1.0, max(2003, 2 * W + 3))
    A = np.hstack([build_phi(x, gamma, centers, act), np.ones((x.size, 1))])
    sol, _, _, _ = sp_lstsq(A, f_vec(x), lapack_driver="gelsd")
    return {"act": act, "fn": FN_SHORT.get(fn_text, fn_text), "lam": lam, "N": N,
            "W": int(W), "norm_v": float(np.linalg.norm(sol[:-1]))}


def rows_path(act):
    return OUT_DIR / f"heatmap_rows_{act}.json"


def sweep(act, lam_grid, n_grid, fns, workers):
    jobs = sorted([{"act": act, "fn": f, "lam": l, "N": n}
                   for f in fns for l in lam_grid for n in n_grid],
                  key=lambda j: j["N"])
    rows, t0 = [], time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(solve_cell, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 50 == 0 or i == len(jobs):
                rows_path(act).write_text(json.dumps(rows))
                print(f"  [{i}/{len(jobs)}] ({time.time()-t0:.0f}s)", flush=True)
    return rows


def make_figure(rows, act, lam_grid, n_grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fns = [FN_SHORT[p] for p in PRESETS if FN_SHORT[p] in {r["fn"] for r in rows}]
    look = {(r["fn"], r["lam"], r["N"]): r["norm_v"] for r in rows}
    grids = {}
    for fn in fns:
        Z = np.full((len(lam_grid), len(n_grid)), np.nan)
        for i, l in enumerate(lam_grid):
            for j, n in enumerate(n_grid):
                v = look.get((fn, l, n))
                if v is not None:
                    Z[i, j] = np.log10(v + 1e-300)
        grids[fn] = Z
    med = np.nanmedian(np.stack(list(grids.values())), axis=0)

    panels = fns + ["median (all targets)"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), sharex=True, sharey=True)
    vmin = min(np.nanmin(Z) for Z in grids.values())
    vmax = max(np.nanmax(Z) for Z in grids.values())
    for ax, name in zip(axes.flat, panels):
        Z = med if name.startswith("median") else grids[name]
        im = ax.pcolormesh(np.arange(len(n_grid) + 1),
                           np.concatenate([lam_grid, [lam_grid[-1] + (lam_grid[-1]-lam_grid[-2])]]),
                           Z, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.axhline(0.25, color="w", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(name, fontsize=11)
        ax.set_xticks(np.arange(len(n_grid)) + 0.5)
        ax.set_xticklabels([str(n) if i % 3 == 0 else "" for i, n in enumerate(n_grid)],
                           fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("N (log-spaced)")
    for ax in axes[:, 0]:
        ax.set_ylabel("lambda")
    fig.colorbar(im, ax=axes, shrink=0.85, label=r"$\log_{10}\|v\|_2$ (solved readout)")
    fig.suptitle(f"expA05 supplement -- readout-coefficient norm over (lambda, width), "
                 f"activation = {act}  (dashed line: lambda* = 0.25)", y=0.98)
    p = OUT_DIR / "figures" / f"lambda_width_norm_heatmap_{act}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="tanh")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fig-only", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    lam_grid, n_grid, fns = LAM_GRID, N_GRID, PRESETS
    if args.smoke:
        lam_grid = [0.05, 0.25, 0.45]
        n_grid = [32, 128]
        fns = PRESETS[:2]

    if args.fig_only:
        rows = json.loads(rows_path(args.act).read_text())
    else:
        total = len(fns) * len(lam_grid) * len(n_grid)
        print(f"sweep: act={args.act} | {len(fns)} targets x {len(lam_grid)} lambdas x "
              f"{len(n_grid)} widths = {total} solves")
        rows = sweep(args.act, lam_grid, n_grid, fns, args.workers)
    make_figure(rows, args.act, lam_grid, n_grid)


if __name__ == "__main__":
    main()
