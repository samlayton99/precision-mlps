"""LSQR entirely in double-double. Memory: a fixed number of dd vectors,
i.e. 2 x O(d) float64 words -- still O(d)."""
import numpy as np, dd


def lsqr_dd(A, b, iters, Aev=None, yev=None, yev_norm=None, probe=()):
    n, d = A.shape
    u = dd.dd(b)
    beta = dd.norm(u); u = dd.scale(u, dd.sdiv((1.0, 0.0), beta))
    v = dd.rmatvec(A, u)
    alpha = dd.norm(v); v = dd.scale(v, dd.sdiv((1.0, 0.0), alpha))
    w = (v[0].copy(), v[1].copy())
    x = (np.zeros(d), np.zeros(d))
    phibar = beta; rhobar = alpha
    out = {}
    for it in range(1, iters + 1):
        # u <- A v - alpha u
        u = dd.sub(dd.matvec(A, v), dd.scale(u, alpha))
        beta = dd.norm(u); u = dd.scale(u, dd.sdiv((1.0, 0.0), beta))
        # v <- A^T u - beta v
        v = dd.sub(dd.rmatvec(A, u), dd.scale(v, beta))
        alpha = dd.norm(v); v = dd.scale(v, dd.sdiv((1.0, 0.0), alpha))
        rho = dd.ssqrt(dd.sadd(dd.smul(rhobar, rhobar), dd.smul(beta, beta)))
        c = dd.sdiv(rhobar, rho); s = dd.sdiv(beta, rho)
        theta = dd.smul(s, alpha)
        rhobar = dd.smul((-c[0], -c[1]), alpha)
        phi = dd.smul(c, phibar); phibar = dd.smul(s, phibar)
        x = dd.add(x, dd.scale(w, dd.sdiv(phi, rho)))
        w = dd.sub(v, dd.scale(w, dd.sdiv(theta, rho)))
        if it in probe:
            r = dd.sub(dd.matvec(Aev, x), dd.dd(yev))
            out[it] = float(dd.norm(r)[0]) / yev_norm
    return x, out
