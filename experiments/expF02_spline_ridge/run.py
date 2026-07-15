"""Experiment expF02 -- KAN-style B-spline ridge basis.

Part A (default): tanh vs cubic-B-spline floor on poisson + smooth darcy
control, W in {144,256,576,1024,2304}, lam in {0.2,0.25,0.3}, best-of-lam.
Part B (--adaptive): rough darcy_421 instance 0, sigma=0. Baselines: dense
tanh and dense spline at W=2304 (uniform). Adaptive: spline, start W=1024,
4 rounds x 320 residual-guided knots -> 2304 total (width-matched).

Outputs (results/checkpoint_F_applications/expF02_spline_ridge/):
  error_vs_width.png    rel L2 vs W, 2 problems x 2 families
  adaptive_rounds.png   rel L2 + n_knots per adaptive round vs dense baselines
  data.json             all cells, written incrementally

Usage:
  uv run --extra dev python experiments/expF02_spline_ridge/run.py [--smoke] [--plot] [--adaptive] [--darcy-path PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ridge_core as rc
import problems as pb
import adaptive as ad
import darcy_data as dd

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF02_spline_ridge"
DATA_PATH = RESULTS_DIR / "data.json"

FAMILIES = {"tanh": rc.tanh_family, "bspline": rc.bspline_family}
LAMS = [0.2, 0.25, 0.3]


def load_data():
    if DATA_PATH.exists():
        return json.loads(DATA_PATH.read_text())
    return []


def save_data(data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(data, indent=1))


def eval_grid(n=120):
    g = np.linspace(-0.995, 0.995, n)
    return np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)


def bc_zero(n_per_edge=480):
    Pb = rc.boundary_points_square(n_per_edge)
    return [dict(points=Pb, terms=[((0, 0), 1.0)], values=np.zeros(len(Pb)))]


def part_a(smoke):
    w_grid = [144, 400] if smoke else [144, 256, 576, 1024, 2304]
    data = load_data()
    done = {(c["part"], c.get("problem"), c.get("family"), c.get("W"), c.get("lam"))
            for c in data}
    Pe = eval_grid()
    for prob_fn in pb.PROBLEMS:
        prob = prob_fn()
        u_true = prob["exact"](Pe)
        for fam_name, family in FAMILIES.items():
            for W in w_grid:
                for lam in LAMS:
                    key = ("A", prob["name"], fam_name, W, lam)
                    if key in done:
                        continue
                    t0 = time.time()
                    model = rc.solve_square(prob["terms"], prob["forcing"], bc_zero(),
                                            W=W, lam=lam, family=family)
                    err = rc.rel_l2(rc.eval_model(model, Pe), u_true)
                    cell = dict(part="A", problem=prob["name"], family=fam_name,
                                W=W, lam=lam, rel_l2=err, t_solve=time.time() - t0)
                    print(cell, flush=True)
                    data.append(cell)
                    save_data(data)


def part_b(smoke, darcy_path):
    a_all, u_all = dd.load_darcy_test(darcy_path, n_instances=1)
    coef = dd.DarcyCoefficient(a_all[0], sigma_px=0.0, cell_centered=True)
    P_eval, u_ref = dd.eval_points_and_ref(u_all[0], stride=3)
    lam = 0.25
    data = load_data()
    done = {(c["part"], c.get("method"), c.get("round")) for c in data}

    def record(method, rnd, model, n_knots, t0):
        err = rc.rel_l2(rc.eval_model(model, P_eval), u_ref)
        cell = dict(part="B", method=method, round=rnd, n_knots=int(n_knots),
                    rel_l2=err, t_solve=time.time() - t0)
        print(cell, flush=True)
        data.append(cell)
        save_data(data)

    w_dense = 400 if smoke else 2304
    for fam_name, family in FAMILIES.items():
        if ("B", f"dense_{fam_name}", 0) not in done:
            t0 = time.time()
            model = rc.solve_square(coef.terms(), dd.DARCY_FORCING, bc_zero(),
                                    W=w_dense, lam=lam, family=family)
            record(f"dense_{fam_name}", 0, model, w_dense, t0)

    # adaptive spline: start smaller, insert knots up to the dense budget
    w0 = 144 if smoke else 1024
    n_rounds, n_add = (2, 128) if smoke else (4, 320)
    dirs, offs, _ = rc.radon_geometry(w0, lam)
    rng = np.random.default_rng(1)
    P_res = rng.uniform(-1, 1, (20000, 2))
    for rnd in range(n_rounds + 1):
        if ("B", "adaptive_bspline", rnd) in done:
            continue
        gammas = ad.local_gammas(dirs, offs, lam)
        t0 = time.time()
        model = rc.solve_square(coef.terms(), dd.DARCY_FORCING, bc_zero(),
                                family=rc.bspline_family,
                                geometry=(dirs, offs, gammas))
        record("adaptive_bspline", rnd, model, len(offs), t0)
        if rnd == n_rounds:
            break
        resid = np.abs(rc.eval_model(model, P_res, terms=coef.terms())
                       - dd.DARCY_FORCING)
        nd, no = ad.insert_knots(dirs, offs, P_res, resid, n_new=n_add)
        dirs = np.vstack([dirs, nd])
        offs = np.concatenate([offs, no])


def plot():
    data = load_data()
    a = [c for c in data if c["part"] == "A"]
    if a:
        probs = sorted({c["problem"] for c in a})
        fig, axes = plt.subplots(1, len(probs), figsize=(6 * len(probs), 4.5))
        axes = np.atleast_1d(axes)
        for axi, prob in zip(axes, probs):
            for fam in sorted({c["family"] for c in a}):
                cells = [c for c in a if c["problem"] == prob and c["family"] == fam]
                ws = sorted({c["W"] for c in cells})
                best = [min(c["rel_l2"] for c in cells if c["W"] == w) for w in ws]
                axi.loglog(ws, best, "o-", label=fam)
            axi.set_title(prob)
            axi.set_xlabel("W")
            axi.set_ylabel("rel L2")
            axi.grid(True, which="both", alpha=0.3)
            axi.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "error_vs_width.png", dpi=140)
    b = [c for c in data if c["part"] == "B"]
    if b:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ada = sorted([c for c in b if c["method"] == "adaptive_bspline"],
                     key=lambda c: c["round"])
        if ada:
            ax.semilogy([c["n_knots"] for c in ada], [c["rel_l2"] for c in ada],
                        "o-", label="adaptive bspline")
        for c in b:
            if c["method"].startswith("dense"):
                ax.axhline(c["rel_l2"], ls="--", alpha=0.7,
                           label=f"{c['method']} W={c['n_knots']}")
        ax.set_xlabel("total knots")
        ax.set_ylabel("rel L2 vs reference")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "adaptive_rounds.png", dpi=140)
    print("plots saved to", RESULTS_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--darcy-path", default=dd.DEFAULT_NPZ)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot()
        return
    if args.adaptive:
        part_b(args.smoke, args.darcy_path)
    else:
        part_a(args.smoke)
    plot()


if __name__ == "__main__":
    main()
