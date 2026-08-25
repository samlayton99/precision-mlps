"""lead_B -- shared helpers. Read-only w.r.t. core11 / expD09 / expD10."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, lsmr, lsqr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "experiments" / "expD11_batching"),
          str(REPO / "experiments" / "expD09_2nd_order_regime")):
    if p not in sys.path:
        sys.path.insert(0, p)
import core11 as C   # noqa: E402

OUT = REPO / "results" / "checkpoint_D_optimizers" / "expD11_batching" / "lead_B"


def diag(prob, w):
    """Everything we care about, in one place."""
    A, y = prob["A"], prob["y"]
    r = y - A @ w
    nr = np.linalg.norm(r)
    ny = np.linalg.norm(y)
    nA = prob.setdefault("_nA", float(np.linalg.norm(prob["A"], 2)))
    nw = np.linalg.norm(w)
    return dict(ev=C.eval_err(prob, w), res=nr / ny, wnorm=float(nw),
                # Stewart/Higham-style LS backward error surrogate
                bwd=float(np.linalg.norm(A.T @ r) / max(nA * nr, 1e-300)),
                # what backward stability would buy on this cell
                bwd_floor=float(np.finfo(float).eps * nA * max(nw, 1e-300) / ny))


def opA(A):
    return LinearOperator(A.shape, matvec=lambda v: A @ v,
                          rmatvec=lambda u: A.T @ u, dtype=float)


def blocks_of(prob, k):
    return C.make_blocks(prob, k, seed=0)


def block_qr_whiten(A, blocks, rcond=1e-15):
    """Per-block pivoted QR whitening (expD09 round 6). Returns B (n x dk),
    and an unwhiten map dk -> d.  Persistent state = the R factors, d*k."""
    n, d = A.shape
    cols, Bs, maps = [], [], []
    Rs = []
    for Cb in blocks:
        Ac = A[:, Cb]
        Q, R, piv = sla.qr(Ac, mode="economic", pivoting=True)
        dg = np.abs(np.diag(R))
        keep = dg > rcond * max(dg[0], 1e-300)
        kk = int(keep.sum())
        Bs.append(Q[:, :kk])
        Rs.append((R[:kk, :kk], piv, Cb, kk))
    B = np.hstack(Bs)

    def unwhiten(z):
        w = np.zeros(d)
        o = 0
        for (R, piv, Cb, kk) in Rs:
            zz = sla.solve_triangular(R, z[o:o + kk], lower=False)
            w[Cb[piv[:kk]]] = zz
            o += kk
        return w
    return B, unwhiten


class BJ:
    """block-Jacobi (SVD-of-rows, never a Gram) half-inverse preconditioner.
    state d*k."""

    def __init__(self, A, blocks, rcond=1e-12):
        self.blocks = blocks
        self.f = []
        for Cb in blocks:
            _, s, Vt = np.linalg.svd(A[:, Cb], full_matrices=False)
            keep = s > rcond * max(s[0], 1e-300)
            iv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
            self.f.append((Vt.T, iv))

    def half(self, v):
        out = np.zeros_like(v)
        for Cb, (V, iv) in zip(self.blocks, self.f):
            out[Cb] = V @ (iv * (V.T @ v[Cb]))
        return out


def kappa_of(M):
    s = np.linalg.svd(M, compute_uv=False)
    keep = s > 1e-15 * s[0]
    return float(s[0] / s[keep][-1]), int(keep.sum())
