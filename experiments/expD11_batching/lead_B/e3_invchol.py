"""E3 -- Gate 1 for sparse inverse-Cholesky preconditioning.

The campaign killed "banded Cholesky" by measuring that the GRAM's off-diagonals
do not decay.  That is the wrong object.  For Green's-function / covariance-like
Grams the classical fact (Markov property; Schaefer-Katzfuss-Owhadi 2021) is the
opposite: the Gram is dense and slowly decaying while the INVERSE Cholesky
factor is exponentially localised.  Brownian covariance min(s,t) is the extreme
case -- dense, but with a tridiagonal inverse.

A = QR (thin QR of the pool).  The ideal preconditioner is M = R^{-1}, which is
upper triangular and costs d^2/2 to store.  Question: does band_k(R^{-1}) with a
FIXED k already give kappa(A M) = O(1)?  If yes, the state is O(d k), the setup
can be done by d independent local regressions (O(d k^3), sub-cubic), and the
whole thing is a steering preconditioner.

Measurement discipline: A @ M is a product of two factors each with
kappa ~ 1e14 whose product is ~ I, so fp64 formation injects eps*kappa ~ 1e-2
and the measured kappa is meaningless.  The product is therefore formed in
double-double and only then rounded to fp64 for the SVD.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import scipy.linalg as sla
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb, core11 as C, dd                              # noqa: E402


def dd_matmul(A, X):
    """A (n,d) fp64 times X (d,m) fp64, accumulated in double-double, returned
    rounded to fp64.  Entries then carry ~eps relative error w.r.t. |A||X|
    *after* cancellation, which is what the naive fp64 matmul cannot do."""
    n, d = A.shape
    m = X.shape[1]
    acc = (np.zeros((n, m)), np.zeros((n, m)))
    for k in range(d):
        p, e = dd.two_prod(A[:, k][:, None], X[k][None, :])
        acc = dd.add(acc, (p, e))
    return acc[0] + acc[1]


def kappa_prod(A, M):
    W = dd_matmul(A, M)
    s = np.linalg.svd(W, compute_uv=False)
    keep = s > 1e-14 * s[0]
    return float(s[0] / s[keep][-1]), int(keep.sum())


def band(M, k):
    d = M.shape[0]
    i = np.arange(d)
    mask = np.abs(i[:, None] - i[None, :]) <= k
    return M * mask


def blockdiag(M, k):
    d = M.shape[0]
    out = np.zeros_like(M)
    for t in range(0, d, k):
        e = min(t + k, d)
        out[t:e, t:e] = M[t:e, t:e]
    return out


def fullrank_cols(A, rcond=1e-15):
    """Drop numerically dependent columns (pivoted QR) but RESTORE the natural
    ordering afterwards -- bandedness is an ordering-dependent property, so the
    pivot order must not be kept."""
    Q, R, piv = sla.qr(A, mode="economic", pivoting=True)
    dg = np.abs(np.diag(R))
    return np.sort(piv[:int((dg > rcond * dg[0]).sum())])


if __name__ == "__main__":
    cases = [(C.prob_qi1d, (128,)), (C.prob_qi1d, (256,)),
             (C.prob_qi2d, (576,)), (C.prob_twin, (512, "unstruct")),
             (C.prob_twin, (512, "cluster"))]
    recs = []
    for f, a in cases:
        p = f(*a)
        A = p["A"][:, fullrank_cols(p["A"])]; d = A.shape[1]
        Q, R = sla.qr(A, mode="economic")
        Rin = sla.solve_triangular(R, np.eye(d), lower=False)
        k0, r0 = kappa_prod(A, Rin)
        print("== %s d=%d rank=%d kappa(A)=%.2e   kappa(A R^-1) full = %.3e (rk %d)"
              % (p["label"], d, p["rank"], np.linalg.cond(A), k0, r0))
        row = dict(cell=p["label"], d=d, kappa=float(np.linalg.cond(A)), full=k0,
                   eps_rank=int((np.linalg.svd(np.triu(Rin,1),compute_uv=False)
                                 > 1.0/np.linalg.norm(A,2)).sum()))
        for k in (8, 16, 32, 64, 128):
            if k >= d:
                break
            kb, rb = kappa_prod(A, band(Rin, k))
            kd, rd = kappa_prod(A, blockdiag(Rin, k))
            # incomplete Cholesky: band R itself then invert the band
            Rb = band(R, k)
            try:
                Rbi = sla.solve_triangular(Rb, np.eye(d), lower=False)
                ki, ri = kappa_prod(A, Rbi)
            except Exception:
                ki, ri = float("inf"), 0
            print("   k=%3d  band(R^-1) kappa=%.3e rk=%3d | blkdiag(R^-1) %.3e"
                  " | band(R)^-1 %.3e" % (k, kb, rb, kd, ki))
            row["band_%d" % k] = kb
            row["blk_%d" % k] = kd
            row["ichol_%d" % k] = ki
        recs.append(row)
    lb.OUT.mkdir(parents=True, exist_ok=True)
    (lb.OUT / "e3_invchol.json").write_text(json.dumps(recs))
