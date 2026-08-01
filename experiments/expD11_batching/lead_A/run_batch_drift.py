"""Requirements 5 and 7 for the reorthogonalized-Krylov family.

5  batching: a fresh random row batch of b rows per matvec (literal reading),
   vs the accumulate-then-solve reading (legitimate on a frozen Phi).
7  drift: the operator is perturbed by relative eta at every matvec.

Reference lines: prob['floor_pool'] (pooled truncated SVD) and
core11.floor_1batch(prob, b) (best from ONE batch, i.e. no accumulation).
"""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np, core11, wlsqr

OUT = HERE / "batch_drift.jsonl"


def lsqr_stoch(A, y, iters, b=None, c=10**6, eta=0.0, seed=0, Aev=None,
               yev=None, ynorm=1.0, probe_every=10, pool_grow=False):
    """Golub-Kahan with one fresh row batch per matvec pair, and/or a fresh
    relative perturbation eta of the operator per matvec."""
    n, d = A.shape
    rng = np.random.default_rng(seed)
    keep = min(c, iters + 2) if c > 0 else 0
    V = np.zeros((d, max(keep, 1))); nV = 0

    def draw(t):
        if b is None or b >= n:
            S = np.arange(n)
        elif pool_grow:
            S = np.arange(min(n, b * t))
        else:
            S = rng.choice(n, b, replace=False)
        Ab = A[S]
        if eta > 0:
            Ab = Ab * (1.0 + eta * rng.standard_normal(d)[None, :])
        return Ab, y[S]

    Ab, yb = draw(1)
    u = yb.copy(); beta = np.linalg.norm(u); u /= beta
    v = Ab.T @ u; alpha = np.linalg.norm(v); v /= alpha
    if keep: V[:, 0] = v; nV = 1
    w = v.copy(); x = np.zeros(d); phibar, rhobar = beta, alpha
    hist = []
    for it in range(1, iters + 1):
        Ab, yb = draw(it + 1)
        if Ab.shape[0] != u.shape[0]:
            u = np.zeros(Ab.shape[0])          # batch size changed (pool_grow)
        u = Ab @ v - alpha * u
        beta = np.linalg.norm(u)
        if not np.isfinite(beta) or beta == 0: break
        u = u / beta
        v = Ab.T @ u - beta * v
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
        x = x + (phi / rho) * w
        w = v - (th / rho) * w
        if it % probe_every == 0 or it == iters:
            hist.append(dict(it=it, ev=float(np.linalg.norm(Aev @ x - yev) / ynorm)))
    return x, hist


def best(h):
    v = [q["ev"] for q in h if np.isfinite(q["ev"])]
    return (min(v), [q["it"] for q in h if q["ev"] == min(v)][0]) if v else (np.nan, -1)


if __name__ == "__main__":
    for nm, mk in [("qi1d128", lambda: core11.prob_qi1d(128)),
                   ("twinU512", lambda: core11.prob_twin(512, "unstruct"))]:
        p = mk(); d, n = p["d"], p["n"]
        it = int(3.0 * p["rank"] + 80)
        print(f"\n=== {nm} d={d} n={n} r={p['rank']} floor={p['floor_pool']:.2e}",
              flush=True)
        print("-- req5 batching (full reorth, fresh batch per matvec)", flush=True)
        for div in [1, 2, 4, 8, 16, 64]:
            b = max(8, d // div) if div > 1 else None
            f1 = core11.floor_1batch(p, b) if b else p["floor_pool"]
            for grow in ([False, True] if b else [False]):
                x, h = lsqr_stoch(p["A"], p["y"], it, b=b, Aev=p["A_ev"],
                                  yev=p["y_ev"], ynorm=p["ye_norm"],
                                  pool_grow=grow)
                bb, bi = best(h)
                core11.append(OUT, dict(case=nm, test="batch", b=(b or n),
                                        grow=grow, best=float(bb), best_it=int(bi),
                                        floor1=f1, floor=p["floor_pool"]))
                print(f"   b={(b or n):5d} (d/{div}) grow={int(grow)}: "
                      f"best={bb:.2e} @{bi}   [1-batch floor {f1:.1e}]", flush=True)
        print("-- req7 drift (full reorth, fresh eta per matvec, all rows)",
              flush=True)
        for eta in [0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-3]:
            x, h = lsqr_stoch(p["A"], p["y"], it, b=None, eta=eta, Aev=p["A_ev"],
                              yev=p["y_ev"], ynorm=p["ye_norm"])
            bb, bi = best(h)
            core11.append(OUT, dict(case=nm, test="drift", eta=eta,
                                    best=float(bb), best_it=int(bi),
                                    floor=p["floor_pool"]))
            print(f"   eta={eta:.0e}: best={bb:.2e} @{bi}", flush=True)
