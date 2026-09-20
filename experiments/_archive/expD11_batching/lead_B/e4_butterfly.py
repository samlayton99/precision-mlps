"""E4 -- cascaded (butterfly) block whitening.

One-shot block-QR whitening (expD09 round 6) puts ALL of kappa into a single
block-diagonal factor and stalls whenever the columns do not localise, because a
block-diagonal factor can only remove the within-block part of the conditioning.
The campaign tried exactly ONE two-level variant (k/2-offset blocks) and
rejected it.  A k/2 offset is the wrong communication pattern: after two stages
most column pairs have still never shared a block.

Here: L = log_k(d) stages with a radix-k BUTTERFLY pattern.  At stage l the
blocks are the index sets {i, i+s, i+2s, ...} with stride s = k^l, so after L
stages every pair of columns has shared a block at some stage -- the same
communication argument that makes the FFT work.

    B_0 = A ;   B_{l+1} = B_l Z_l ,  Z_l = P_l^T blockdiag(R_{l,j}^{-1}) P_l
    solve min ||B_L u - y|| ,  w = Z_0 Z_1 ... Z_{L-1} u

Budget:
  state    L * d * k floats  (the R factors; L = log_k d, so O(d k log d))
  setup    L * (d/k) QRs of n x k  = O(n d k log d) flops -- sub-cubic
  per iter one matvec with B_L
Secondary hope: kappa is SPLIT over L factors, so each R_l is well conditioned
(kappa ~ kappa(A)^{1/L}), which is exactly what the "never apply an
ill-conditioned factor" lesson wants.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import scipy.linalg as sla
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb, core11 as C                                    # noqa: E402
from scipy.sparse.linalg import lsqr, LinearOperator       # noqa: E402


def butterfly_blocks(d, k, stride):
    """Blocks {i, i+s, ..., i+(k-1)s} covering 0..d-1 exactly once."""
    out = []
    per = k * stride
    for base in range(0, d, per):
        for i in range(min(stride, d - base)):
            blk = np.arange(base + i, min(base + per, d), stride)
            if blk.size:
                out.append(blk)
    return out


def contiguous_blocks(d, k):
    return [np.arange(t, min(t + k, d)) for t in range(0, d, k)]


def whiten_stage(B, blocks, rcond):
    """Pivoted-QR whiten each block of B.  Returns the whitened operator and
    the stage's factors (the persistent state)."""
    n = B.shape[0]
    cols, facs = [], []
    for blk in blocks:
        Ac = B[:, blk]
        Q, R, piv = sla.qr(Ac, mode="economic", pivoting=True)
        dg = np.abs(np.diag(R))
        keep = dg > rcond * max(dg[0], 1e-300)
        kk = int(keep.sum())
        if kk == 0:
            continue
        cols.append(Q[:, :kk])
        facs.append((R[:kk, :kk], piv[:kk], blk, kk))
    Bn = np.hstack(cols) if cols else np.zeros((n, 0))
    return Bn, facs


def cascade(A, k, L, rcond=1e-13, pattern="butterfly", verbose=False):
    B = A
    stages = []
    kaps = []
    for l in range(L):
        d = B.shape[1]
        if pattern == "butterfly":
            blocks = butterfly_blocks(d, k, k ** l if k ** l < d else 1)
        else:
            blocks = contiguous_blocks(d, k)
        B, facs = whiten_stage(B, blocks, rcond)
        s = np.linalg.svd(B, compute_uv=False)
        keep = s > 1e-15 * s[0]
        kap = float(s[0] / s[keep][-1])
        kaps.append((B.shape[1], kap, int(keep.sum()),
                     max(np.linalg.cond(f[0]) for f in facs)))
        stages.append(facs)
        if verbose:
            print("      stage %d: cols=%3d kappa(B)=%.3e nrank=%3d max kappa(R_l)=%.2e"
                  % (l, B.shape[1], kap, int(keep.sum()), kaps[-1][3]))
    return B, stages, kaps


def unwhiten(stages, u, d0):
    """Apply Z_0 Z_1 ... Z_{L-1} to u (reverse order through the stages)."""
    z = u
    for facs in reversed(stages):
        din = max(int(f[2].max()) for f in facs) + 1
        out = np.zeros(din)
        o = 0
        for (R, piv, blk, kk) in facs:
            out[blk[piv]] = sla.solve_triangular(R, z[o:o + kk], lower=False)
            o += kk
        z = out
    return z


if __name__ == "__main__":
    cases = [(C.prob_qi1d, (128,)), (C.prob_qi1d, (256,)),
             (C.prob_qi2d, (576,)), (C.prob_twin, (512, "unstruct"))]
    recs = []
    for f, a in cases:
        p = f(*a); A, y = p["A"], p["y"]; d0 = p["d"]
        print("== %s d=%d rank=%d kappa=%.2e floor=%.2e"
              % (p["label"], d0, p["rank"], p["kappa"], p["floor_pool"]))
        for k in (8, 16, 32):
            L = int(np.ceil(np.log(d0) / np.log(k))) + 1
            for pat in ("butterfly", "contig"):
                t = time.time()
                B, st, kaps = cascade(A, k, L, pattern=pat,
                                      verbose=(pat == "butterfly" and k == 16))
                out = []
                for it in (10, 30, 60, 120, 300):
                    r = lsqr(lb.opA(B), y, atol=0, btol=0, conlim=0, iter_lim=it)
                    w = unwhiten(st, r[0], d0)
                    out.append(C.eval_err(p, w))
                print("   k=%2d L=%d %-9s kappa: %s | ev@it: %s (%.1fs) state=%dd"
                      % (k, L, pat, " ".join("%.1e" % x[1] for x in kaps),
                         " ".join("%.1e" % v for v in out), time.time() - t,
                         sum(sum(f[3] ** 2 for f in s) for s in st) // d0))
                recs.append(dict(cell=p["label"], k=k, L=L, pattern=pat,
                                 kaps=kaps, ev=out, floor=p["floor_pool"]))
    lb.OUT.mkdir(parents=True, exist_ok=True)
    (lb.OUT / "e4_butterfly.json").write_text(json.dumps(recs))
