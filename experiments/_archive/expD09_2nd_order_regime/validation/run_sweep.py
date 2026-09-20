"""expD09 validation sweep -- runs experiments A-D, caches to JSON, then plots.

  A  trajectories     3x3 (target x width), one line per block size k
  B  k-vs-d coupling  does the required k grow with d?  THE O(d) question
  C  noise            3x3, one line per noise level; shows semiconvergence
  D  batching         D1: trajectories, one line per batch size
                      D2: rows needed for the QR, vs k

Usage:
  uv run python experiments/expD09_2nd_order_regime/validation/run_sweep.py
  ... --only A B          # subset
  ... --plot              # replot from cache
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import common_val as cv  # noqa: E402

K_TRAJ = [32, 64, 128]
K_SCALE = [8, 16, 32, 64, 128, 256]
NOISE = [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
BATCH_FRAC = [1.0, 0.5, 0.25, 0.125]
QR_ROW_MULT = [1, 2, 4, 8, 16, None]      # s = mult * k ; None = all rows


def iters_for(k, N):
    base = max(600, 60000 // max(k, 1))
    return int(base * (1.5 if N >= 512 else 1.0))


def _list(a):
    return np.asarray(a).tolist()


# ------------------------------------------------------------------ A ------

def run_A():
    rows = []
    t0 = time.time()
    for fn in cv.TARGETS:
        for N in cv.WIDTHS:
            prob = cv.make_problem(fn, N)
            for k in K_TRAJ:
                it = iters_for(k, N)
                r = cv.solve_traj(prob, k, it, stride=2)
                rows.append({"fn": fn, "N": N, "k": k, "d": prob["d"],
                             "stride": 2, "iters": it,
                             "floor": prob["floor_svd"],
                             "eval": _list(r["eval"]), "train": _list(r["train"]),
                             "best": r["best"]})
                print(f"[A {time.time()-t0:5.0f}s] {fn:10s} N={N:4d} k={k:4d} "
                      f"best={r['best']:.2e} floor={prob['floor_svd']:.2e}")
    cv.save_rows("A_traj", rows)


# ------------------------------------------------------------------ B ------

def run_B():
    rows = []
    t0 = time.time()
    for fn in cv.TARGETS:
        for N in cv.WIDTHS_SCALING:
            prob = cv.make_problem(fn, N)
            for k in K_SCALE:
                if k > prob["d"]:
                    continue
                it = iters_for(k, N)
                r = cv.solve_traj(prob, k, it, stride=10)
                rows.append({"fn": fn, "N": N, "d": prob["d"], "k": k,
                             "rank": prob["rank"], "iters": it,
                             "floor": prob["floor_svd"], "best": r["best"],
                             "best_it": r["best_it"] * 10})
                print(f"[B {time.time()-t0:5.0f}s] {fn:10s} N={N:4d} d={prob['d']:4d} "
                      f"k={k:4d} best={r['best']:.2e} floor={prob['floor_svd']:.2e}")
    cv.save_rows("B_scaling", rows)


# ------------------------------------------------------------------ C ------

def run_C():
    rows = []
    t0 = time.time()
    for fn in cv.TARGETS:
        for N in cv.WIDTHS:
            for nz in NOISE:
                prob = cv.make_problem(fn, N, noise_rel=nz)
                it = iters_for(128, N)
                r = cv.solve_traj(prob, 128, it, stride=2)
                rows.append({"fn": fn, "N": N, "noise": nz, "stride": 2,
                             "floor": prob["floor_svd"],
                             "floor_stat": prob["floor_stat"],
                             "eval": _list(r["eval"]),
                             "best": r["best"], "final": r["final"],
                             "best_it": r["best_it"] * 2})
                print(f"[C {time.time()-t0:5.0f}s] {fn:10s} N={N:4d} "
                      f"noise={nz:.0e} best={r['best']:.2e} final={r['final']:.2e}")
    cv.save_rows("C_noise", rows)


# ------------------------------------------------------------------ D ------

def run_D():
    rng = np.random.default_rng(0)
    traj, thresh = [], []
    t0 = time.time()
    # D1 -- trajectories vs batch size (both QR and solve use the batch)
    for fn in cv.TARGETS:
        for N in cv.WIDTHS:
            prob = cv.make_problem(fn, N)
            n = prob["n"]
            for frac in BATCH_FRAC:
                b = max(int(frac * n), prob["d"] // 2)
                rows_b = np.sort(rng.choice(n, size=min(b, n), replace=False))
                it = iters_for(128, N)
                r = cv.solve_traj(prob, 128, it, rows_qr=rows_b,
                                  solve_rows=rows_b, stride=2)
                traj.append({"fn": fn, "N": N, "frac": frac, "b": int(b),
                             "n": n, "d": prob["d"], "stride": 2,
                             "floor": prob["floor_svd"],
                             "eval": _list(r["eval"]), "best": r["best"]})
                print(f"[D1 {time.time()-t0:5.0f}s] {fn:10s} N={N:4d} "
                      f"b/n={frac:.3f} b={b:5d} best={r['best']:.2e}")
    # D2 -- how many rows does the QR itself need? (solve always full)
    for fn in cv.TARGETS:
        for N in cv.WIDTHS:
            prob = cv.make_problem(fn, N)
            n = prob["n"]
            for k in [64, 128]:
                for mult in QR_ROW_MULT:
                    s = n if mult is None else min(mult * k, n)
                    rq = None if mult is None else np.sort(
                        rng.choice(n, size=s, replace=False))
                    it = iters_for(k, N)
                    r = cv.solve_traj(prob, k, it, rows_qr=rq, stride=10)
                    thresh.append({"fn": fn, "N": N, "k": k,
                                   "mult": (mult if mult else -1), "s": int(s),
                                   "n": n, "floor": prob["floor_svd"],
                                   "best": r["best"]})
            print(f"[D2 {time.time()-t0:5.0f}s] {fn:10s} N={N:4d} done")
    cv.save_rows("D1_batch_traj", traj)
    cv.save_rows("D2_qr_rows", thresh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    todo = args.only or ["A", "B", "C", "D"]
    if not args.plot:
        for name in todo:
            {"A": run_A, "B": run_B, "C": run_C, "D": run_D}[name]()
    import plots
    plots.render_all()


if __name__ == "__main__":
    main()
