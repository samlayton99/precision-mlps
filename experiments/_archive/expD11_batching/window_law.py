"""THE decisive measurement for step 2.

On a FROZEN operator with a FIXED accumulated buffer (so batching, drift and
sampling are all out of the picture), LSQR stalls at ~1e-7 while full
reorthogonalization reaches ~1e-16.  The only difference is how many previous
Golub-Kahan vectors are kept and orthogonalized against.

Call that number c.  Requirement 2 (memory O(d*k), k FIXED) is satisfied iff
c* -- the window needed to reach a target accuracy -- is bounded as d grows.
If c* grows like r, the state is O(d*r) and the approach is disqualified.

This script measures c*(d) directly.  Nothing else varies.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core11 as c

OUT = c.RESULTS / "window_law.jsonl"
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def bjac(Ar, d, k=64, rcond=1e-13):
    """Block-Jacobi M^{-1/2}; state d*k.  Identity when k is None."""
    if k is None:
        return (lambda v: v)
    fac = []
    for t in range(0, d, k):
        C = np.arange(t, min(t + k, d))
        _, s, Vt = np.linalg.svd(Ar[:, C], full_matrices=False)
        kp = s > rcond * max(s[0], 1e-300)
        fac.append((C, Vt.T, np.where(kp, 1.0 / np.where(kp, s, 1.0), 0.0)))

    def Mh(v):
        o = np.zeros(d)
        for C, V, ih in fac:
            o[C] = V @ (ih * (V.T @ v[C]))
        return o
    return Mh


def lsqr_window(A, b, Mh, d, maxiter, cwin, tol=0.0, probe=None):
    """LSQR on A*Mh with a SLIDING reorthogonalization window of cwin vectors.

    cwin = 0    plain LSQR (the shipped solver)
    cwin = inf  full reorthogonalization (the O(d*r) reference)

    Returns (x_in_preconditioned_coords, history).  Standard Paige-Saunders
    recurrence; the only addition is the two-pass Gram-Schmidt against the
    stored window.
    """
    Av = lambda z: A @ Mh(z)
    Atu = lambda u: Mh(A.T @ u)

    u = b.copy()
    beta = np.linalg.norm(u)
    if beta == 0:
        return np.zeros(d), []
    u /= beta
    v = Atu(u)
    alpha = np.linalg.norm(v)
    if alpha == 0:
        return np.zeros(d), []
    v /= alpha

    Us, Vs = [u.copy()], [v.copy()]
    w = v.copy()
    x = np.zeros(d)
    phibar, rhobar = beta, alpha
    hist = []

    def reorth(z, store):
        if cwin == 0 or not store:
            return z
        keep = store if cwin == np.inf else store[-int(cwin):]
        for _ in range(2):                      # two passes = numerically safe
            for q in keep:
                z -= (q @ z) * q
        return z

    for it in range(1, maxiter + 1):
        # bidiagonalization step
        u_new = Av(v) - alpha * u
        u_new = reorth(u_new, Us)
        beta = np.linalg.norm(u_new)
        if beta > 0:
            u = u_new / beta
        else:
            break
        v_new = Atu(u) - beta * v
        v_new = reorth(v_new, Vs)
        alpha = np.linalg.norm(v_new)
        if alpha > 0:
            v = v_new / alpha
        else:
            break
        if cwin != 0:
            Us.append(u.copy()); Vs.append(v.copy())
            if cwin != np.inf:
                Us = Us[-int(cwin):]; Vs = Vs[-int(cwin):]
        # Givens
        rho = np.hypot(rhobar, beta)
        cs, sn = rhobar / rho, beta / rho
        theta = sn * alpha
        rhobar = -cs * alpha
        phi = cs * phibar
        phibar = sn * phibar
        x += (phi / rho) * w
        w = v - (theta / rho) * w
        if probe is not None and (it % probe[0] == 0 or it == maxiter):
            hist.append((it, probe[1](Mh(x))))
        if phibar <= tol * np.linalg.norm(b):
            break
    return x, hist


def run_cell(fam, kw, cwins, maxiter_mult=2.0, k=64, rounds=1, seed=0):
    p = (c.prob_qi1d(**kw) if fam == "qi1d" else
         c.prob_qi2d(**kw) if fam == "qi2d" else c.prob_twin(**kw))
    A, y, n, d = p["A"], p["y"], p["n"], p["d"]
    rng = np.random.default_rng(500 + seed)
    idx = rng.choice(n, int(min(n, 4 * d)), replace=False)
    Aa, ya = A[idx], y[idx]
    U, s, Vt = np.linalg.svd(Aa, full_matrices=False)
    kp = s > 1e-15 * s[0]
    r = int(kp.sum())
    aex = (Vt.T * np.where(kp, 1 / np.where(kp, s, 1), 0)) @ (U.T @ ya)
    buf_floor = c.eval_err(p, aex)
    Mh = bjac(Aa, d, k)
    mi = int(maxiter_mult * r)
    log(f"{p['label']}: d={d} r={r} buffer-exact={buf_floor:.2e} maxiter={mi}")
    out = []
    for cw in cwins:
        cwv = np.inf if cw == "full" else int(cw)
        w = np.zeros(d)
        for _ in range(rounds):
            xz, _ = lsqr_window(Aa, ya - Aa @ w, Mh, d, mi, cwv)
            w = w + Mh(xz)
        ev = c.eval_err(p, w)
        rec = dict(family=p["family"], label=p["label"], d=d, r=r, n_acc=len(idx),
                   k=k, cwin=(-1 if cwv == np.inf else int(cwv)),
                   cwin_label=str(cw), maxiter=mi, rounds=rounds,
                   eval=ev, buf_floor=buf_floor, floor_pool=p["floor_pool"],
                   kappa=p["kappa"])
        c.append(OUT, rec)
        out.append(rec)
        log(f"    c={str(cw):<6} c/r={(cwv/r if cwv!=np.inf else 1.0):<6.3f} "
            f"eval={ev:.3e}")
    return out


CELLS = [("qi1d", dict(N=N, row_mult=8)) for N in (48, 64, 96, 128, 192, 256, 384, 512)]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxN", type=int, default=512)
    a = ap.parse_args()
    done = {(q["label"], q["cwin_label"], q["k"]) for q in c.load(OUT)}
    for fam, kw in CELLS:
        if kw["N"] > a.maxN:
            continue
        for k in (64, None):
            cws = [0, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, "full"]
            run_cell(fam, kw, [x for x in cws], k=k)
    log("DONE")
