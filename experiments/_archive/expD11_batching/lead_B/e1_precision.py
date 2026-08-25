"""E1 -- what does extended precision actually buy?

Three arms at matched iteration counts:
  fp64 LSQR                  (scipy, reference)
  dd   LSQR                  (xlsqr, eps ~ 1e-32)
  dd/fp64 LSQR + block-Jacobi (O(d*k) state, O(d k^2) setup)

Separates "Krylov convergence rate" from "fp64 rounding floor".
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb, core11 as C, xlsqr, dd                       # noqa: E402
from scipy.sparse.linalg import lsqr, lsmr, LinearOperator  # noqa: E402


class BJdd:
    """block-Jacobi half-inverse, applicable to dd vectors. State d*k."""

    def __init__(self, A, blocks, rcond=1e-12):
        self.blocks, self.f = blocks, []
        for Cb in blocks:
            _, s, Vt = np.linalg.svd(A[:, Cb], full_matrices=False)
            keep = s > rcond * max(s[0], 1e-300)
            iv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
            self.f.append((Vt.T * iv, Vt))          # (V*iv, V^T)

    def half64(self, v):
        out = np.zeros_like(v)
        for Cb, (Vi, Vt) in zip(self.blocks, self.f):
            out[Cb] = Vi @ (Vt @ v[Cb])
        return out

    def half_dd(self, X):
        h = np.zeros_like(X[0]); l = np.zeros_like(X[1])
        for Cb, (Vi, Vt) in zip(self.blocks, self.f):
            t = xlsqr.rmatvec(Vt.T, (X[0][Cb], X[1][Cb]))   # Vt @ v  (k x k)
            o = xlsqr.matvec(Vi, t)
            h[Cb], l[Cb] = o
        return (h, l)


def dd_prec_lsqr(A, b, P, iters, probe, cb):
    """LSQR in dd on the right-preconditioned system A P (P = P^T = BJ half)."""
    n, d = A.shape
    mv = lambda V: xlsqr.matvec(A, P.half_dd(V))
    rmv = lambda U: P.half_dd(xlsqr.rmatvec(A, U))
    u = dd.dd(b); beta = dd.norm(u); u = dd.scale(u, dd.sdiv((1., 0.), beta))
    v = rmv(u); alpha = dd.norm(v); v = dd.scale(v, dd.sdiv((1., 0.), alpha))
    w = (v[0].copy(), v[1].copy()); z = (np.zeros(d), np.zeros(d))
    phibar, rhobar = beta, alpha
    out = {}
    for it in range(1, iters + 1):
        u = dd.sub(mv(v), dd.scale(u, alpha))
        beta = dd.norm(u); u = dd.scale(u, dd.sdiv((1., 0.), beta))
        v = dd.sub(rmv(u), dd.scale(v, beta))
        alpha = dd.norm(v); v = dd.scale(v, dd.sdiv((1., 0.), alpha))
        rho = dd.ssqrt(dd.sadd(dd.smul(rhobar, rhobar), dd.smul(beta, beta)))
        c = dd.sdiv(rhobar, rho); s = dd.sdiv(beta, rho)
        theta = dd.smul(s, alpha); rhobar = dd.smul((-c[0], -c[1]), alpha)
        phi = dd.smul(c, phibar); phibar = dd.smul(s, phibar)
        z = dd.add(z, dd.scale(w, dd.sdiv(phi, rho)))
        w = dd.sub(v, dd.scale(w, dd.sdiv(theta, rho)))
        if it in probe:
            xx = P.half_dd(z)
            out[it] = cb(it, xx[0] + xx[1])
    return out


PROBE = [5, 10, 20, 40, 70, 110, 160, 220, 300, 400]

if __name__ == "__main__":
    cases = [(C.prob_qi1d, (128,)), (C.prob_qi1d, (256,)),
             (C.prob_twin, (512, "unstruct")), (C.prob_qi2d, (576,))]
    recs = []
    for f, a in cases:
        p = f(*a); A, y = p["A"], p["y"]; ny = np.linalg.norm(y)
        cb = lambda it, w: (float(np.linalg.norm(y - A @ w) / ny),
                            C.eval_err(p, w))
        print("== %s d=%d rank=%d kappa=%.2e floor=%.2e"
              % (p["label"], p["d"], p["rank"], p["kappa"], p["floor_pool"]))
        # fp64 LSQR reference
        f64 = {}
        for it in PROBE:
            r = lsqr(lb.opA(A), y, atol=0, btol=0, conlim=0, iter_lim=it)
            f64[it] = cb(it, r[0])
        print("   fp64 LSQR      ", " ".join("%.1e" % f64[i][1] for i in PROBE))
        # dd LSQR
        t = time.time()
        _, ddo = xlsqr.lsqr_x(A, y, max(PROBE), probe=PROBE,
                              cb=lambda it, x, ph: cb(it, x[0] + x[1]))
        print("   dd   LSQR      ", " ".join("%.1e" % ddo[i][1] for i in PROBE),
              " (%.0fs)" % (time.time() - t))
        for k in (64, 128):
            blocks = lb.blocks_of(p, k)
            P = BJdd(A, blocks)
            kap, rk = lb.kappa_of(np.column_stack(
                [A @ P.half64(e) for e in np.eye(p["d"])]))
            g = {}
            op = LinearOperator(A.shape, matvec=lambda v: A @ P.half64(v),
                                rmatvec=lambda u: P.half64(A.T @ u), dtype=float)
            for it in PROBE:
                r = lsqr(op, y, atol=0, btol=0, conlim=0, iter_lim=it)
                g[it] = cb(it, P.half64(r[0]))
            print("   fp64 +BJ k=%3d " % k,
                  " ".join("%.1e" % g[i][1] for i in PROBE),
                  " kappa(AM)=%.2e rk=%d" % (kap, rk))
            t = time.time()
            h = dd_prec_lsqr(A, y, P, max(PROBE), set(PROBE), cb)
            print("   dd   +BJ k=%3d " % k,
                  " ".join("%.1e" % h[i][1] for i in PROBE),
                  " (%.0fs)" % (time.time() - t))
            recs.append(dict(cell=p["label"], k=k, kappa_AM=kap,
                             fp64=[f64[i] for i in PROBE],
                             dd=[ddo[i] for i in PROBE],
                             fp64_bj=[g[i] for i in PROBE],
                             dd_bj=[h[i] for i in PROBE], probe=PROBE,
                             floor=p["floor_pool"], d=p["d"], rank=p["rank"]))
    lb.OUT.mkdir(parents=True, exist_ok=True)
    (lb.OUT / "e1_precision.json").write_text(json.dumps(recs))
