"""Mechanism test for the window law.

In EXACT arithmetic the Golub-Kahan three-term recurrence is globally
orthogonal -- no reorthogonalization is needed at all.  So the measured need
for a window of size c ~ 0.8r must be a ROUNDING effect.  If it is, raising the
working precision (double-double, eps 1e-32, 2 words/number => still O(d*k))
should let a SMALL fixed window reach the floor.  If it does not, the window
requirement is not about eps and no precision trick will fix it.

Arms: 'fp64' | 'mixed' (fp64 matvec, dd recurrence+reorth) | 'dd' (all dd).
"""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np, core11, dd as D, ddsolve as S

OUT = HERE / "ddwin.jsonl"


def orth_dd(v, Vh, Vl, nV):
    """Two-pass classical GS of dd vector v against dd basis columns."""
    for _ in range(2):
        for j in range(nV):
            cj = D.dsum(D.mul((Vh[:, j], Vl[:, j]), v))
            v = D.sub(v, D.scale((Vh[:, j], Vl[:, j]), cj))
    return v


def lsqr_win_dd(A, y, iters, c, arm="dd", Aev=None, yev=None, ynorm=1.0,
                probe_every=10):
    n, d = A.shape
    MV = S.mv_dd if arm == "dd" else S.mv_fp64
    RMV = S.rmv_dd if arm == "dd" else S.rmv_fp64
    keep = min(c, iters + 2) if c > 0 else 0
    Vh = np.zeros((d, max(keep, 1))); Vl = np.zeros_like(Vh); nV = 0
    u = D.dd(y)
    beta = D.norm(u); u = D.scale(u, D.sdiv((1.0, 0.0), beta))
    v = RMV(A, u)
    alpha = D.norm(v); v = D.scale(v, D.sdiv((1.0, 0.0), alpha))
    if keep:
        Vh[:, 0], Vl[:, 0] = v[0], v[1]; nV = 1
    w = (v[0].copy(), v[1].copy()); x = D.dd(np.zeros(d))
    phibar, rhobar = beta, alpha
    hist = []
    for it in range(1, iters + 1):
        u = D.sub(MV(A, v), D.scale(u, alpha))
        beta = D.norm(u)
        if not np.isfinite(beta[0]) or beta[0] == 0: break
        u = D.scale(u, D.sdiv((1.0, 0.0), beta))
        v = D.sub(RMV(A, u), D.scale(v, beta))
        if keep:
            v = orth_dd(v, Vh, Vl, nV)
        alpha = D.norm(v)
        if not np.isfinite(alpha[0]) or alpha[0] == 0: break
        v = D.scale(v, D.sdiv((1.0, 0.0), alpha))
        if keep:
            if nV < keep:
                Vh[:, nV], Vl[:, nV] = v[0], v[1]; nV += 1
            else:
                Vh[:, :-1] = Vh[:, 1:]; Vl[:, :-1] = Vl[:, 1:]
                Vh[:, -1], Vl[:, -1] = v[0], v[1]
        rho = D.ssqrt(D.sadd(D.smul(rhobar, rhobar), D.smul(beta, beta)))
        cs, sn = D.sdiv(rhobar, rho), D.sdiv(beta, rho)
        th = D.smul(sn, alpha); rhobar = D.smul((-cs[0], -cs[1]), alpha)
        phi = D.smul(cs, phibar); phibar = D.smul(sn, phibar)
        x = D.add(x, D.scale(w, D.sdiv(phi, rho)))
        w = D.sub(v, D.scale(w, D.sdiv(th, rho)))
        if it % probe_every == 0 or it == iters:
            xf = x[0] + x[1]
            hist.append(dict(it=it, ev=float(np.linalg.norm(Aev @ xf - yev) / ynorm)))
    return x, hist


if __name__ == "__main__":
    for nm, mk in [("qi1d128", lambda: core11.prob_qi1d(128)),
                   ("twinU512", lambda: core11.prob_twin(512, "unstruct"))]:
        p = mk()
        it = int(2.2 * p["rank"] + 60)
        print(f"\n=== {nm} d={p['d']} r={p['rank']} floor={p['floor_pool']:.2e} "
              f"iters={it}", flush=True)
        for c in [0, 16, 32, 64]:
            for arm in ["fp64", "mixed", "dd"]:
                t = time.time()
                if arm == "fp64":
                    import wlsqr
                    x, h = wlsqr.lsqr_win(p["A"], p["y"], it, c=c, mode="recent",
                                          Aev=p["A_ev"], yev=p["y_ev"],
                                          ynorm=p["ye_norm"], probe_every=10)
                else:
                    x, h = lsqr_win_dd(p["A"], p["y"], it, c, arm=arm,
                                       Aev=p["A_ev"], yev=p["y_ev"],
                                       ynorm=p["ye_norm"], probe_every=10)
                vv = [q["ev"] for q in h if np.isfinite(q["ev"])]
                b = min(vv) if vv else np.nan
                bi = [q["it"] for q in h if q["ev"] == b][0] if vv else -1
                core11.append(OUT, dict(case=nm, d=p["d"], rank=p["rank"], c=c,
                                        arm=arm, best=float(b), best_it=int(bi),
                                        floor=p["floor_pool"], secs=time.time() - t))
                print(f"  c={c:3d} {arm:5s}: {b:.2e} @{bi:4d}  "
                      f"({time.time()-t:5.1f}s)", flush=True)
