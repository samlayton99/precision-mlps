"""E5 -- local vs global conditioning: why the whole block family has a floor.

E4 showed that after ONE stage of block whitening, every later stage finds its
blocks already well conditioned (max kappa(R_l) drops from 1e13 to O(10)) while
kappa(B) is unchanged at 1e14.  If that is literally true it kills the entire
block-diagonal family -- any blocking, any depth, any pattern -- because the
family's fixed point is not well conditioned.

Measured here, directly:

    kappa_loc(k) = max over sampled k-column subsets S of kappa(A[:,S])
    kappa_glob   = kappa(A)

A block-diagonal preconditioner built from k-column windows can only remove
conditioning that is VISIBLE inside a k-column window, so the reduction it can
achieve is bounded by kappa_loc(k).  The claim to test is:

    after one whitening stage,  kappa_loc(k) = O(1)  while  kappa_glob = 1e14,

i.e. the surviving ill-conditioning is invisible to every k-column window and
therefore unreachable by any amount of further block work.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import scipy.linalg as sla
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb, core11 as C, e4_butterfly as E4                 # noqa: E402


def kappa_loc(M, k, reps=60, seed=0):
    rng = np.random.default_rng(seed)
    d = M.shape[1]
    out = []
    for _ in range(reps):
        S = rng.choice(d, min(k, d), replace=False)
        s = np.linalg.svd(M[:, S], compute_uv=False)
        keep = s > 1e-15 * s[0]
        out.append(s[0] / s[keep][-1])
    return float(np.median(out)), float(np.max(out))


def kap(M):
    s = np.linalg.svd(M, compute_uv=False)
    keep = s > 1e-15 * s[0]
    return float(s[0] / s[keep][-1])


if __name__ == "__main__":
    cases = [(C.prob_qi1d, (128,)), (C.prob_qi1d, (256,)),
             (C.prob_qi2d, (576,)), (C.prob_twin, (512, "unstruct")),
             (C.prob_twin, (512, "cluster"))]
    recs = []
    for f, a in cases:
        p = f(*a); A = p["A"]
        print("== %s d=%d  kappa_glob=%.2e" % (p["label"], p["d"], kap(A)))
        row = dict(cell=p["label"], d=p["d"], kappa=kap(A))
        for k in (16, 32, 64, 128):
            lm, lx = kappa_loc(A, k)
            B, st, kaps = E4.cascade(A, k, 1, pattern="contig")
            lm2, lx2 = kappa_loc(B, k)
            print("   k=%3d  kappa_loc(A)  med=%.2e max=%.2e   -> after 1 whitening"
                  " stage: kappa_glob=%.2e  kappa_loc med=%.2e max=%.2e"
                  % (k, lm, lx, kaps[0][1], lm2, lx2))
            row["k%d" % k] = dict(loc_med=lm, loc_max=lx, glob_after=kaps[0][1],
                                  loc_med_after=lm2, loc_max_after=lx2)
        recs.append(row)
    lb.OUT.mkdir(parents=True, exist_ok=True)
    (lb.OUT / "e5_local_global.json").write_text(json.dumps(recs))
