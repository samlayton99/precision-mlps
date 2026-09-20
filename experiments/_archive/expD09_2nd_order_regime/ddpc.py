"""Block-Jacobi preconditioned LSQR in double-double.

State: d/k blocks of a k x k factor  ->  O(d*k), k fixed  =>  O(d).
Vectors are dd: 2 float64 words each  =>  still O(d).
"""
import numpy as np, dd


def block_factors(A, k, rcond=1e-15):
    """Block M^{-1/2} = V S^{-1} V^T from the SVD of the n x k block ITSELF.

    NOT from eigh(Ac^T Ac): forming the block Gram squares its condition
    number, so the small singular values come back as fp64 noise and the
    truncation then deletes real directions permanently -- a hard floor that
    no amount of working precision downstream can undo (measured: the dd
    solve stalled at exactly the fp64 stall).
    """
    d = A.shape[1]; fac = []
    for t in range(0, d, k):
        C = np.arange(t, min(t + k, d)); Ac = A[:, C]
        _, sv, Vt = np.linalg.svd(Ac, full_matrices=False)
        kp = sv > rcond * max(sv[0], 1e-300)
        fac.append((C, Vt.T, np.where(kp, 1.0 / np.where(kp, sv, 1.0), 0.0)))
    return fac


def apply_M(fac, X, d):
    """Apply block-Jacobi M^{-1/2} to a dd vector, blockwise, in dd."""
    oh = np.zeros(d); ol = np.zeros(d)
    for C, Q, ih in fac:
        sub = (X[0][C], X[1][C])
        t = dd.rmatvec(Q, sub)                 # Q^T v_C
        t = (t[0] * ih, t[1] * ih)             # scale by ih (fp64 exact-ish)
        r = dd.matvec(Q, t)                    # Q (...)
        oh[C] = r[0]; ol[C] = r[1]
    return (oh, ol)


def plsqr_dd(A, b, fac, iters, Aev, yev, yev_norm, probe=()):
    n, d = A.shape
    Av = lambda V: dd.matvec(A, apply_M(fac, V, d))
    Atv = lambda U: apply_M(fac, dd.rmatvec(A, U), d)
    u = dd.dd(b)
    beta = dd.norm(u); u = dd.scale(u, dd.sdiv((1.0, 0.0), beta))
    v = Atv(u); alpha = dd.norm(v); v = dd.scale(v, dd.sdiv((1.0, 0.0), alpha))
    w = (v[0].copy(), v[1].copy()); x = (np.zeros(d), np.zeros(d))
    phibar = beta; rhobar = alpha; out = {}
    for it in range(1, iters + 1):
        u = dd.sub(Av(v), dd.scale(u, alpha))
        beta = dd.norm(u); u = dd.scale(u, dd.sdiv((1.0, 0.0), beta))
        v = dd.sub(Atv(u), dd.scale(v, beta))
        alpha = dd.norm(v); v = dd.scale(v, dd.sdiv((1.0, 0.0), alpha))
        rho = dd.ssqrt(dd.sadd(dd.smul(rhobar, rhobar), dd.smul(beta, beta)))
        c = dd.sdiv(rhobar, rho); s = dd.sdiv(beta, rho)
        theta = dd.smul(s, alpha); rhobar = dd.smul((-c[0], -c[1]), alpha)
        phi = dd.smul(c, phibar); phibar = dd.smul(s, phibar)
        x = dd.add(x, dd.scale(w, dd.sdiv(phi, rho)))
        w = dd.sub(v, dd.scale(w, dd.sdiv(theta, rho)))
        if it in probe:
            wsol = apply_M(fac, x, d)
            r = dd.sub(dd.matvec(Aev, wsol), dd.dd(yev))
            out[it] = float(dd.norm(r)[0]) / yev_norm
    return x, out
