"""E5: noise + batching floors for the finalist configuration.

Finalist ("hardened block-QR"): pivoted-QR whitening at k=128 in a known or
seriation-recovered column order, LSMR with Fong-Saunders stopping
(atol=btol=1e-15, conlim=0), single unwhiten. No oracle anywhere.

Noise arm: noise_rel in {0,1e-8,1e-6,1e-4,1e-2} x 9 cells. Success =
stopped error tracks the statistical floor 0.272*sigma (recipe Sec 8.7).

Batching arm: whiten AND solve on the same random row batch b in
{8d,4d,2d,d}; eval always on the clean 4001-point grid. Success = floor at
b>=4d, graceful degradation below (recipe Sec 8.8).

Min-norm audit: ||w|| recorded for every solve, vs ||a_svd|| per cell.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core10 as c                      # noqa: E402
from scipy.sparse.linalg import lsmr    # noqa: E402

OUT = c.RESULTS
OUT.mkdir(parents=True, exist_ok=True)


def solve_stop(A, y, k=128, order=None, maxiter=8000):
    pieces, B = c.whiten(A, k, order=order)
    res = lsmr(B, y, atol=1e-15, btol=1e-15, conlim=0, maxiter=maxiter)
    w = c.unwhiten(pieces, res[0], A.shape[1])
    return w, int(res[2])


def main():
    t0 = time.time()
    rows = []

    # ---- noise arm -------------------------------------------------------
    for nz in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
        for fn in c.cv.TARGETS:
            for N in c.cv.WIDTHS:
                prob = c.cv.make_problem(fn, N, noise_rel=nz)
                A, y = prob["A"], prob["y"]
                w, it = solve_stop(A, y)
                rows.append({
                    "arm": "noise", "fn": fn, "N": N, "noise": nz,
                    "eval": c.eval_err(prob, w), "iters": it,
                    "wnorm": float(np.linalg.norm(w)),
                    "floor_svd": prob["floor_svd"],
                    "floor_stat": prob["floor_stat"],
                })
                print(f"noise {nz:8.0e} {fn:9s} N={N:3d}  eval={rows[-1]['eval']:.2e}"
                      f"  stat={prob['floor_stat']:.2e}  it={it}")

    # ---- batching arm ----------------------------------------------------
    rng = np.random.default_rng(11)
    for fn in c.cv.TARGETS:
        for N in c.cv.WIDTHS:
            prob = c.cv.make_problem(fn, N)          # n = 8d pool
            A, y = prob["A"], prob["y"]
            n, d = A.shape
            for mult in (8, 4, 2, 1):
                b = min(n, mult * d)
                sel = rng.choice(n, b, replace=False)
                w, it = solve_stop(A[sel], y[sel])
                rows.append({
                    "arm": "batch", "fn": fn, "N": N, "bmult": mult,
                    "eval": c.eval_err(prob, w), "iters": it,
                    "wnorm": float(np.linalg.norm(w)),
                    "floor_svd": prob["floor_svd"],
                })
                print(f"batch b={mult}d {fn:9s} N={N:3d}  eval={rows[-1]['eval']:.2e}  it={it}")

    (OUT / "floors.json").write_text(json.dumps(rows, indent=1))
    print(f"\nsaved {OUT/'floors.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
