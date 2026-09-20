"""Extended-precision LSQR with fp64 instrumentation.

Uses expD09's `dd` module (read-only import) for double-double arithmetic.
State: a fixed number of dd vectors = O(d) float64 words.  No preconditioner,
no setup.  The point of this file is to separate two hypotheses:

  H_info      Krylov cannot reach the floor because the small singular
              directions are not recoverable from matvecs.
  H_prec      Krylov cannot reach the floor because fp64 rounding destroys
              the exact-arithmetic finite termination at r iterations.

If dd-LSQR terminates near iteration r, H_prec is the operative one.
"""
from __future__ import annotations
import numpy as np
import dd


def _split(A):
    """Dekker split of a fp64 matrix into two 26-bit halves (exact)."""
    S = 2.0 ** 27 + 1.0
    Ah = (S * A) - ((S * A) - A)
    return Ah, A - Ah


def matvec(M, X):
    n = M.shape[0]
    acc = (np.zeros(n), np.zeros(n))
    for k in range(M.shape[1]):
        p, e = dd.two_prod(M[:, k], X[0][k])
        e = e + M[:, k] * X[1][k]
        acc = dd.add(acc, (p, e))
    return acc


def rmatvec(M, X):
    P, E = dd.two_prod(M, X[0][:, None])
    E = E + M * X[1][:, None]
    return dd._csum0(P, E)


def lsqr_x(A, b, iters, probe=None, cb=None, precond=None):
    """LSQR in dd.  precond: optional pair (half, halfT) of fp64 callables
    applied as A -> A P (right preconditioning); they are applied in dd via
    a matrix if given as an ndarray."""
    n, d = A.shape
    u = dd.dd(b)
    beta = dd.norm(u)
    u = dd.scale(u, dd.sdiv((1.0, 0.0), beta))
    v = rmatvec(A, u)
    alpha = dd.norm(v)
    v = dd.scale(v, dd.sdiv((1.0, 0.0), alpha))
    w = (v[0].copy(), v[1].copy())
    x = (np.zeros(d), np.zeros(d))
    phibar, rhobar = beta, alpha
    probe = set(probe or ())
    out = {}
    for it in range(1, iters + 1):
        u = dd.sub(matvec(A, v), dd.scale(u, alpha))
        beta = dd.norm(u)
        u = dd.scale(u, dd.sdiv((1.0, 0.0), beta))
        v = dd.sub(rmatvec(A, u), dd.scale(v, beta))
        alpha = dd.norm(v)
        v = dd.scale(v, dd.sdiv((1.0, 0.0), alpha))
        rho = dd.ssqrt(dd.sadd(dd.smul(rhobar, rhobar), dd.smul(beta, beta)))
        c = dd.sdiv(rhobar, rho)
        s = dd.sdiv(beta, rho)
        theta = dd.smul(s, alpha)
        rhobar = dd.smul((-c[0], -c[1]), alpha)
        phi = dd.smul(c, phibar)
        phibar = dd.smul(s, phibar)
        x = dd.add(x, dd.scale(w, dd.sdiv(phi, rho)))
        w = dd.sub(v, dd.scale(w, dd.sdiv(theta, rho)))
        if it in probe and cb is not None:
            out[it] = cb(it, x, float(abs(phibar[0])))
    return x, out
