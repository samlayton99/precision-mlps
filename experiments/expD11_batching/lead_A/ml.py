"""lead_A core: multi-level (butterfly) block whitening + measurement helpers.

Everything here is read-only w.r.t. the rest of the repo. Cost model:
  state    L * d * k floats  (the per-level, per-block R factors)  -> O(d*k)
  setup    L * (d/k) QRs of (n x k)  = O(L*n*d*k) flops            -> sub-cubic
  per-iter one matvec with the whitened operator                   -> O(b*d)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, lsmr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "experiments" / "expD11_batching"),
          str(REPO / "experiments" / "expD09_2nd_order_regime")):
    if p not in sys.path:
        sys.path.insert(0, p)
import core11  # noqa: E402

EPS = np.finfo(float).eps


# ------------------------------------------------------------ diagnostics --
def kap(B):
    """kappa from the matrix directly (never a Gram)."""
    s = np.linalg.svd(B, compute_uv=False)
    s = s[s > 0]
    return float(s[0] / s[-1]), s


def kap_eff(B, rcond=1e-15):
    s = np.linalg.svd(B, compute_uv=False)
    keep = s > rcond * s[0]
    return float(s[0] / s[keep][-1]), int(keep.sum()), s


# ------------------------------------------------------- permutation zoo --
def perm_of(d, kind, level, k, seed=0):
    if level == 0 or kind == "none":
        return np.arange(d)
    if kind == "stride":
        nb = int(np.ceil(d / k))
        idx = np.concatenate([np.arange(j, d, nb) for j in range(nb)])
        return idx
    if kind == "random":
        return np.random.default_rng(1000 * seed + level).permutation(d)
    if kind == "offset":
        return np.roll(np.arange(d), (k // 2) * level)
    raise ValueError(kind)


# --------------------------------------------------- multi-level whitening -
class MLWhiten:
    """L levels of pivoted-QR block whitening, each under its own column
    permutation.  Level l stores, per block, (piv, R, kept) -- d*k floats/level.

    The whitened operator B is built by *direct QR of the previous level*, so
    every level's blocks are orthonormal by construction (never by applying an
    inverse factor to data -- the round-6 lesson).
    """

    def __init__(self, k=64, L=2, kind="stride", rcond=1e-13, seed=0):
        self.k, self.L, self.kind, self.rcond, self.seed = k, L, kind, rcond, seed
        self.levels = []

    def fit(self, A):
        B = np.array(A, dtype=float, copy=True)
        cols = np.arange(A.shape[1])          # surviving original columns
        self.levels = []
        for l in range(self.L):
            d = B.shape[1]
            pm = perm_of(d, self.kind, l, self.k, self.seed)
            nb = int(np.ceil(d / self.k))
            blocks = [pm[t:t + self.k] for t in range(0, d, self.k)]
            lev = []
            newcols, pieces = [], []
            for C in blocks:
                X = B[:, C]
                Q, R, piv = sla.qr(X, mode="economic", pivoting=True)
                dg = np.abs(np.diag(R))
                keep = dg > self.rcond * max(dg[0], 1e-300)
                kk = int(keep.sum())
                lev.append(dict(C=C, piv=piv, R=R[:kk, :kk].copy(), kk=kk))
                pieces.append(Q[:, :kk])
                newcols.append(C[piv[:kk]])
            order = np.concatenate(newcols)
            B = np.hstack(pieces)
            # relabel: B's columns now correspond to `order` positions of old B
            cols = cols[order]
            self.levels.append(dict(blocks=lev, order=order, dprev=d))
            self._reorder = True
        self.B = B
        self.cols = cols
        return B

    def state_floats(self):
        return sum(sum(bl["kk"] ** 2 for bl in lv["blocks"]) for lv in self.levels)

    def unwhiten(self, z):
        """Map a coefficient vector in B's columns back to original columns."""
        v = np.asarray(z, float)
        for lv in reversed(self.levels):
            out = np.zeros(lv["dprev"])
            off = 0
            for bl in lv["blocks"]:
                kk = bl["kk"]
                if kk:
                    y = sla.solve_triangular(bl["R"], v[off:off + kk], lower=False)
                    out[bl["C"][bl["piv"][:kk]]] = y
                off += kk
            v = out
        return v


def solve_ml(prob, k=64, L=2, kind="stride", rcond=1e-13, maxiter=4000,
             atol=1e-15, A=None, y=None, seed=0):
    A = prob["A"] if A is None else A
    y = prob["y"] if y is None else y
    W = MLWhiten(k=k, L=L, kind=kind, rcond=rcond, seed=seed)
    B = W.fit(A)
    res = lsmr(B, y, atol=atol, btol=atol, conlim=0, maxiter=maxiter)
    w = W.unwhiten(res[0])
    return w, dict(kappa=kap(B)[0], iters=int(res[2]), dB=B.shape[1],
                   state=W.state_floats(), eval=core11.eval_err(prob, w))
