"""E6 -- what IS achievable at O(d*k) state, and is precision or kappa binding?

Arms, all at state d*k (block-Jacobi factors) + O(d):
  fp64 LSQR + block-Jacobi, LONG budget   -- is there a hard fp64 floor?
  dd   LSQR + block-Jacobi, LONG budget   -- does extra precision remove it?
and the pre-solve diagnostic kappa_loc(k) from E5 next to the outcome, to test
whether kappa_loc(k)/kappa predicts which cells can be finished at O(d*k).

Also the drift arm (requirement 7): the block factors are STEERING (they only
change the metric of the search, correctness comes from residuals of the current
operator), so a stale factor should cost iterations and not accuracy.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb, core11 as C, e1_precision as E1, e5_local_global as E5   # noqa: E402
from scipy.sparse.linalg import lsqr, LinearOperator                 # noqa: E402

PROBE = [50, 100, 200, 400, 700, 1100, 1600]

if __name__ == "__main__":
    cases = [(C.prob_qi1d, (128,)), (C.prob_qi1d, (256,)),
             (C.prob_qi2d, (576,)), (C.prob_twin, (512, "unstruct")),
             (C.prob_twin, (512, "cluster"))]
    recs = []
    for f, a in cases:
        p = f(*a); A, y = p["A"], p["y"]
        cb = lambda it, w: C.eval_err(p, w)
        print("== %s d=%d rank=%d kappa=%.2e floor=%.2e"
              % (p["label"], p["d"], p["rank"], p["kappa"], p["floor_pool"]))
        for k in (64, 128):
            blocks = lb.blocks_of(p, k)
            P = E1.BJdd(A, blocks)
            klm, klx = E5.kappa_loc(A, k)
            op = LinearOperator(A.shape, matvec=lambda v: A @ P.half64(v),
                                rmatvec=lambda u: P.half64(A.T @ u), dtype=float)
            g = [cb(it, P.half64(lsqr(op, y, atol=0, btol=0, conlim=0,
                                      iter_lim=it)[0])) for it in PROBE]
            t = time.time()
            h = E1.dd_prec_lsqr(A, y, P, max(PROBE), set(PROBE), cb)
            print("   k=%3d kappa_loc/kappa=%.1e  fp64: %s" %
                  (k, klm / p["kappa"], " ".join("%.1e" % v for v in g)))
            print("            %14s  dd  : %s  (%.0fs)"
                  % ("", " ".join("%.1e" % h[i] for i in PROBE), time.time() - t))
            recs.append(dict(cell=p["label"], k=k, probe=PROBE, fp64=g,
                             dd=[h[i] for i in PROBE], floor=p["floor_pool"],
                             kappa=p["kappa"], kappa_loc=klm, d=p["d"],
                             rank=p["rank"]))
    lb.OUT.mkdir(parents=True, exist_ok=True)
    (lb.OUT / "e6_best_odk.json").write_text(json.dumps(recs))
