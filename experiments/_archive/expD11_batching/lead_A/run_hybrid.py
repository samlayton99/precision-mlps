"""Hybrid: block-QR whitening (state d*k) + windowed reorth LSQR (state d*c).
Total persistent state (k+c)*d with BOTH constants fixed -> the only
configuration in this study that can satisfy requirement 2."""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np, core11, wlsqr, ml

OUT = HERE / "hybrid.jsonl"


def best(h):
    v = [q["ev"] for q in h if q["ev"] is not None and np.isfinite(q["ev"])]
    return (min(v), [q["it"] for q in h if q["ev"] == min(v)][0]) if v else (np.nan, -1)


def hybrid(p, k, c, L=1, kind="stride", rcond=1e-13, itmul=3.0):
    t0 = time.time()
    W = ml.MLWhiten(k=k, L=L, kind=kind, rcond=rcond)
    B = W.fit(p["A"])
    tset = time.time() - t0
    kb = ml.kap(B)[0]
    it = int(min(2600, itmul * p["rank"] + 80))
    # eval side must be whitened too -> instead unwhiten each probe via W
    Aev, yev = p["A_ev"], p["y_ev"]

    class EvProxy:
        pass
    z, h = wlsqr.lsqr_win(B, p["y"], it, c=c, mode="recent", Aev=None,
                          yev=None, probe_every=10)
    # recover trajectory endpoints only for the final z (cheap, exact)
    w = W.unwhiten(z)
    ev = core11.eval_err(p, w)
    return dict(kappa_B=kb, dB=int(B.shape[1]), setup_s=tset, iters=it,
                state_over_d=(k + (c if c < 1e5 else it)), eval=ev)


def hybrid_traj(p, k, c, L=1, rcond=1e-13, itmul=3.0, probe=10):
    """As hybrid() but probes eval along the trajectory (unwhitens per probe)."""
    W = ml.MLWhiten(k=k, L=L, kind="stride", rcond=rcond)
    t0 = time.time(); B = W.fit(p["A"]); tset = time.time() - t0
    it = int(min(2600, itmul * p["rank"] + 80))
    n, dB = B.shape
    keep = min(c, it + 2) if c > 0 else 0
    V = np.zeros((dB, max(keep, 1))); nV = 0
    u = p["y"].astype(float).copy(); beta = np.linalg.norm(u); u /= beta
    v = B.T @ u; alpha = np.linalg.norm(v); v /= alpha
    if keep: V[:, 0] = v; nV = 1
    wv = v.copy(); x = np.zeros(dB); phibar, rhobar = beta, alpha
    traj = []
    for i in range(1, it + 1):
        u = B @ v - alpha * u; beta = np.linalg.norm(u)
        if not np.isfinite(beta) or beta == 0: break
        u = u / beta
        v = B.T @ u - beta * v
        if keep: v = wlsqr._orth_against(v, V, nV)
        alpha = np.linalg.norm(v)
        if not np.isfinite(alpha) or alpha == 0: break
        v = v / alpha
        if keep:
            if nV < keep: V[:, nV] = v; nV += 1
            else: V[:, :-1] = V[:, 1:]; V[:, -1] = v
        rho = np.hypot(rhobar, beta); cs, sn = rhobar / rho, beta / rho
        th = sn * alpha; rhobar = -cs * alpha
        phi = cs * phibar; phibar = sn * phibar
        x = x + (phi / rho) * wv; wv = v - (th / rho) * wv
        if i % probe == 0 or i == it:
            traj.append((i, core11.eval_err(p, W.unwhiten(x))))
    return dict(kappa_B=ml.kap(B)[0], dB=int(dB), setup_s=tset,
                traj=traj, best=min(t[1] for t in traj),
                best_it=min(traj, key=lambda t: t[1])[0])


if __name__ == "__main__":
    cases = [("qi1d256", lambda: core11.prob_qi1d(256)),
             ("qi1d512", lambda: core11.prob_qi1d(512)),
             ("qi2d576", lambda: core11.prob_qi2d(576)),
             ("qi2d1024", lambda: core11.prob_qi2d(1024)),
             ("twinU1024", lambda: core11.prob_twin(1024, "unstruct")),
             ("twinC1024", lambda: core11.prob_twin(1024, "cluster"))]
    for nm, mk in cases:
        p = mk()
        print(f"\n{nm} d={p['d']} r={p['rank']} floor={p['floor_pool']:.2e} "
              f"kappa={p['kappa']:.1e}", flush=True)
        for k in (0, 128):
            for c in (0, 64, 128, 256):
                t = time.time()
                if k == 0:
                    it = int(min(2600, 3.0 * p["rank"] + 80))
                    x, h = wlsqr.lsqr_win(p["A"], p["y"], it, c=c, mode="recent",
                                          Aev=p["A_ev"], yev=p["y_ev"],
                                          ynorm=p["ye_norm"], probe_every=10)
                    b, bi = best(h); kb, st = p["kappa"], 0.0
                else:
                    R = hybrid_traj(p, k, c)
                    b, bi, kb, st = R["best"], R["best_it"], R["kappa_B"], R["setup_s"]
                rec = dict(case=nm, d=p["d"], rank=p["rank"], floor=p["floor_pool"],
                           k=k, c=c, best=float(b), best_it=int(bi),
                           kappa_op=float(kb), setup_s=st, secs=time.time() - t,
                           state_over_d=k + c)
                core11.append(OUT, rec)
                print(f"  k={k:4d} c={c:4d} state={k+c:4d}d kappa_op={kb:.1e} "
                      f"best={b:.2e} @{bi}", flush=True)
