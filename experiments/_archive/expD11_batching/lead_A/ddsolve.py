"""Precision-graded LSQR: isolates whether the fp64 stall is the MATVEC
rounding or the RECURRENCE rounding.

Three arms share one code path; only the matvec kernel and the scalar/vector
recurrence precision change:
  'fp64'   everything float64                      (scipy-equivalent)
  'mixed'  fp64 matvecs, dd recurrence vectors     (~1.0x flops)
  'dd'     dd matvecs, dd recurrence               (~10-20x flops)

Memory in every arm is a fixed number of length-d/length-n vectors: 'dd' costs
exactly 2 words per number, so state is O(d*k) with k=2.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "experiments" / "expD11_batching"),
          str(REPO / "experiments" / "expD09_2nd_order_regime")):
    if p not in sys.path:
        sys.path.insert(0, p)
import dd as D  # noqa: E402


# ------------------------------------------------------------- kernels -----
def mv_fp64(A, X):
    """fp64 matvec of a dd vector -> dd (products/sum in fp64, so the result
    carries the fp64 dot-product error ~eps*|A||x|, but no cancellation in the
    hi/lo split)."""
    h = A @ X[0]
    l = A @ X[1]
    return D.quick_two_sum(*D.two_sum(h, l))


def rmv_fp64(A, X):
    h = A.T @ X[0]
    l = A.T @ X[1]
    return D.quick_two_sum(*D.two_sum(h, l))


def mv_dd(A, X):
    return D.matvec(A, X)


def rmv_dd(A, X):
    return D.rmatvec(A, X)


# ------------------------------------------------------------- LSQR --------
def lsqr(A, b, iters, arm="dd", Aev=None, yev=None, probe_every=1, x0=None,
         Pinv=None):
    """Golub-Kahan LSQR.  `Pinv(v)` optional right preconditioner applied as
    B = A*Pinv (matvec: A@Pinv(v); rmatvec: Pinv^T(A.T@u)) -- supply as a pair
    (fwd, adj) of dd->dd callables."""
    n, d = A.shape
    if arm == "fp64":
        return _lsqr64(A, b, iters, Aev, yev, probe_every, x0, Pinv)
    MV = mv_dd if arm == "dd" else mv_fp64
    RMV = rmv_dd if arm == "dd" else rmv_fp64
    if Pinv is None:
        Amv = lambda v: MV(A, v)
        Armv = lambda u: RMV(A, u)
    else:
        fwd, adj = Pinv
        Amv = lambda v: MV(A, fwd(v))
        Armv = lambda u: adj(RMV(A, u))

    x = D.dd(np.zeros(d)) if x0 is None else (np.array(x0, float), np.zeros(d))
    r0 = D.sub(D.dd(b), Amv(x)) if x0 is not None else D.dd(b)
    u = r0
    beta = D.norm(u)
    u = D.scale(u, D.sdiv((1.0, 0.0), beta))
    v = Armv(u)
    alpha = D.norm(v)
    v = D.scale(v, D.sdiv((1.0, 0.0), alpha))
    w = (v[0].copy(), v[1].copy())
    phibar, rhobar = beta, alpha
    hist = []
    ynorm = np.linalg.norm(yev) if yev is not None else 1.0
    for it in range(1, iters + 1):
        u = D.sub(Amv(v), D.scale(u, alpha))
        beta = D.norm(u)
        u = D.scale(u, D.sdiv((1.0, 0.0), beta))
        v = D.sub(Armv(u), D.scale(v, beta))
        alpha = D.norm(v)
        v = D.scale(v, D.sdiv((1.0, 0.0), alpha))
        rho = D.ssqrt(D.sadd(D.smul(rhobar, rhobar), D.smul(beta, beta)))
        c, s = D.sdiv(rhobar, rho), D.sdiv(beta, rho)
        theta = D.smul(s, alpha)
        rhobar = D.smul((-c[0], -c[1]), alpha)
        phi = D.smul(c, phibar)
        phibar = D.smul(s, phibar)
        x = D.add(x, D.scale(w, D.sdiv(phi, rho)))
        w = D.sub(v, D.scale(w, D.sdiv(theta, rho)))
        if it % probe_every == 0 or it == iters:
            xf = x[0] + x[1]
            xx = x[0] if Pinv is None else None
            wfull = xf if Pinv is None else None
            hist.append(dict(it=it, phibar=float(abs(phibar[0])),
                             xnorm=float(np.linalg.norm(xf)),
                             ev=_ev(Aev, yev, ynorm, x, Pinv)))
    return x, hist


def _ev(Aev, yev, ynorm, x, Pinv):
    if Aev is None:
        return None
    z = x if Pinv is None else Pinv[0](x)
    zf = z[0] + z[1]
    return float(np.linalg.norm(Aev @ zf - yev) / ynorm)


def _lsqr64(A, b, iters, Aev, yev, probe_every, x0, Pinv):
    n, d = A.shape
    if Pinv is None:
        Amv, Armv = (lambda v: A @ v), (lambda u: A.T @ u)
    else:
        fwd, adj = Pinv
        Amv = lambda v: A @ fwd((v, np.zeros_like(v)))[0]
        Armv = lambda u: adj((A.T @ u, np.zeros(d)))[0]
    x = np.zeros(d) if x0 is None else np.array(x0, float)
    u = b - Amv(x) if x0 is not None else b.copy()
    beta = np.linalg.norm(u); u = u / beta
    v = Armv(u); alpha = np.linalg.norm(v); v = v / alpha
    w = v.copy(); phibar, rhobar = beta, alpha
    hist = []
    ynorm = np.linalg.norm(yev) if yev is not None else 1.0
    for it in range(1, iters + 1):
        u = Amv(v) - alpha * u
        beta = np.linalg.norm(u); u = u / beta
        v = Armv(u) - beta * v
        alpha = np.linalg.norm(v); v = v / alpha
        rho = np.hypot(rhobar, beta)
        c, s = rhobar / rho, beta / rho
        theta = s * alpha; rhobar = -c * alpha
        phi = c * phibar; phibar = s * phibar
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w
        if it % probe_every == 0 or it == iters:
            z = x if Pinv is None else Pinv[0]((x, np.zeros(d)))[0]
            hist.append(dict(it=it, phibar=float(abs(phibar)),
                             xnorm=float(np.linalg.norm(x)),
                             ev=None if Aev is None else
                             float(np.linalg.norm(Aev @ z - yev) / ynorm)))
    return x, hist
