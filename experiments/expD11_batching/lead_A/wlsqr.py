"""LSQR with a FIXED-SIZE reorthogonalization window (state O(d*c), c fixed).

Motivation: the exact-arithmetic Krylov rate on these systems is exponential
and front-loaded (~10^(-1/12) per iteration on QI-1D), so the iteration count
is not the problem -- maintaining orthogonality is.  Full one-sided reorth
costs O(d*m) with m ~ r.  This measures what a FIXED window buys.

Modes for the window contents:
  'recent'  the last c basis vectors (classic local reorth)
  'first'   the first c basis vectors (the largest singular directions)
  'both'    c/2 first + c/2 recent
  'ritz'    thick-restart style: c vectors carried as an orthonormal set that
            is periodically re-compressed to the c most-converged directions
"""
from __future__ import annotations
import numpy as np


def _orth_against(v, V, ncols, passes=2):
    if ncols == 0:
        return v
    Q = V[:, :ncols]
    for _ in range(passes):
        v = v - Q @ (Q.T @ v)
    return v


def lsqr_win(A, y, iters, c=32, mode="recent", Aev=None, yev=None, ynorm=1.0,
             probe_every=1, dtype=float):
    n, d = A.shape
    keep = min(c, iters + 2) if c > 0 else 0
    V = np.zeros((d, max(keep, 1)))
    nV = 0                      # number of stored vectors
    u = y.astype(dtype).copy()
    beta = np.linalg.norm(u); u /= beta
    v = A.T @ u
    alpha = np.linalg.norm(v); v /= alpha
    if keep:
        V[:, 0] = v; nV = 1
    w = v.copy(); x = np.zeros(d)
    phibar, rhobar = beta, alpha
    hist = []
    for it in range(1, iters + 1):
        u = A @ v - alpha * u
        beta = np.linalg.norm(u)
        if not np.isfinite(beta) or beta == 0.0:
            break
        u = u / beta
        v = A.T @ u - beta * v
        if keep:
            v = _orth_against(v, V, nV)
        alpha = np.linalg.norm(v)
        if not np.isfinite(alpha) or alpha == 0.0:
            break
        v = v / alpha
        if keep:
            if nV < keep:
                V[:, nV] = v; nV += 1
            elif mode == "recent":
                V[:, :-1] = V[:, 1:]; V[:, -1] = v
            elif mode == "both":
                h = keep // 2
                V[:, h:-1] = V[:, h + 1:]; V[:, -1] = v
            # 'first': window frozen at the first c vectors
        rho = np.hypot(rhobar, beta)
        cs, sn = rhobar / rho, beta / rho
        theta = sn * alpha; rhobar = -cs * alpha
        phi = cs * phibar; phibar = sn * phibar
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w
        if it % probe_every == 0 or it == iters:
            hist.append(dict(it=it,
                             ev=float(np.linalg.norm(Aev @ x - yev) / ynorm)
                             if Aev is not None else None,
                             phibar=float(abs(phibar))))
    return x, hist
