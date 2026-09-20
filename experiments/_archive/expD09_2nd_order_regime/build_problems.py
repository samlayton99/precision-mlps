"""Build the 9 frozen-Phi least-squares problems and their fp64 SVD floors.

3 targets (sine, sine_8pi, runge) x 3 widths (N = 64, 128, 256). Each is
stored as its defining grid (x_tr, y_tr, x_ev, y_ev, centers, gamma) plus
the two things every run needs and nobody should recompute:

  a_svd   the truncated-SVD (min-norm) solution at rcond = 1e-15 -- THE floor
  V_null  orthonormal basis of the numerical null space of A_train, for the
          null-space-drift diagnostic

The manifest additionally records the spectrum summary each solver's step
rule needs (smax for the 1/L gradient step, rank for the k-vs-rank story).

Usage: uv run python experiments/expD09_2nd_order_regime/build_problems.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from common import (  # noqa: E402
    MANIFEST_PATH, N_EVAL, PROBLEMS_DIR, RCOND, ROW_RATIO, TARGETS, WIDTHS,
    build_phi_aug, geometry, problem_id,
)
from src.data.targets import get_target  # noqa: E402


def make_problem(fn_name: str, N: int) -> tuple[dict, dict]:
    centers, gamma = geometry(N)
    d = centers.size + 1                      # [Phi, 1]
    n = int(ROW_RATIO * d)
    x_tr = np.linspace(-1.0, 1.0, n)
    x_ev = np.linspace(-1.0, 1.0, N_EVAL)
    f = get_target(fn_name).fn_numpy
    y_tr, y_ev = f(x_tr), f(x_ev)

    A = build_phi_aug(x_tr, gamma, centers)
    A_ev = build_phi_aug(x_ev, gamma, centers)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > RCOND * s[0]
    rank = int(keep.sum())
    a_svd = Vt.T @ (np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
                    * (U.T @ y_tr))
    V_null = Vt[~keep].T                      # d x (d - rank), orthonormal

    r_tr = A @ a_svd - y_tr
    r_ev = A_ev @ a_svd - y_ev
    arrays = dict(x_tr=x_tr, y_tr=y_tr, x_ev=x_ev, y_ev=y_ev,
                  centers=centers, gamma=np.float64(gamma),
                  a_svd=a_svd, V_null=V_null, svals=s)
    meta = {
        "id": problem_id(fn_name, N), "fn": fn_name, "N": N,
        "n": n, "d": d, "rank": rank, "halo": int((centers.size - N - 1) // 2),
        "gamma": float(gamma),
        "smax": float(s[0]),
        "s_rank_ratio": float(s[rank - 1] / s[0]),
        "floor_train_rel_l2": float(np.linalg.norm(r_tr)
                                    / np.linalg.norm(y_tr)),
        "floor_eval_rel_l2": float(np.linalg.norm(r_ev)
                                   / np.linalg.norm(y_ev)),
        "floor_eval_linf": float(np.abs(r_ev).max()),
        "a_svd_norm": float(np.linalg.norm(a_svd)),
        "path": f"{problem_id(fn_name, N)}.npz",
    }
    return arrays, meta


def main() -> None:
    t0 = time.time()
    PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for fn in TARGETS:
        for N in WIDTHS:
            arrays, meta = make_problem(fn, N)
            np.savez_compressed(PROBLEMS_DIR / meta["path"], **arrays)
            manifest.append(meta)
            print(f"  {meta['id']:22s} n={meta['n']:5d} d={meta['d']:4d} "
                  f"rank={meta['rank']:4d}  s_r/s_1={meta['s_rank_ratio']:.1e}"
                  f"  floor(train)={meta['floor_train_rel_l2']:.2e}"
                  f"  floor(eval)={meta['floor_eval_rel_l2']:.2e}")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1))
    print(f"\nSaved {MANIFEST_PATH}: {len(manifest)} problems "
          f"in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
