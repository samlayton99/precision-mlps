"""E2 -- Tikhonov continuation (regularization-path homotopy).

Idea.  Solve a decreasing sequence of damped problems
    w_l = argmin ||A w - y||^2 + lam_l ||w||^2 ,      lam_l = (mu_l)^2 decreasing,
each by LSQR on the *increment*, warm-started from w_{l-1}:
    min_delta  || [A ; mu I] delta - [ r_{l-1} ; -mu w_{l-1} ] ||.
The stacked operator has singular values sqrt(sig_i^2 + mu^2): the spectrum is
FLOORED at mu, so kappa = sig_1/mu is chosen by the schedule, and everything
below mu collapses into one cluster (Krylov eats clusters).  The right-hand
side of stage l is concentrated in the directions the previous stage could not
see, i.e. a narrow spectral window.

Why this is worth a shot here:
  * state O(d) -- three or four vectors, no factors at all
  * setup: NONE.  No O(d^3), no O(d k^2), nothing.
  * per iteration: one A-matvec + one A^T-matvec (the mu*I part is O(d))
  * nothing is stored between stages except w, so operator drift is free
  * front-loaded by construction: stage 0 is well conditioned
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb, core11 as C                                   # noqa: E402
from scipy.sparse.linalg import lsqr, lsmr, LinearOperator  # noqa: E402


def stacked(A, mu):
    n, d = A.shape
    def mv(v):
        return np.concatenate([A @ v, mu * v])
    def rmv(u):
        return A.T @ u[:n] + mu * u[n:]
    return LinearOperator((n + d, d), matvec=mv, rmatvec=rmv, dtype=float)


def continuation(prob, mu0_rel=1.0, decade_step=1.0, mu_min_rel=1e-17,
                 inner=60, solver="lsqr", w0=None, verbose=False,
                 rows=None, batch=None, seed=0, drift=0.0, atol=0.0):
    """rows: optional fixed row subset.  batch: if set, a fresh random batch of
    this many rows per stage (requirement 5)."""
    A_full, y_full = prob["A"], prob["y"]
    n, d = A_full.shape
    s1 = prob.setdefault("_nA", float(np.linalg.norm(A_full, 2)))
    rng = np.random.default_rng(seed)
    w = np.zeros(d) if w0 is None else w0.copy()
    mus = []
    m = mu0_rel * s1
    while m > mu_min_rel * s1:
        mus.append(m); m /= 10.0 ** decade_step
    mus.append(0.0)
    tr = []
    tot = 0
    for l, mu in enumerate(mus):
        if batch is None:
            A, y = A_full, y_full
        else:
            S = rng.choice(n, min(batch, n), replace=False)
            A, y = A_full[S], y_full[S]
        if drift > 0:
            A = A * (1.0 + drift * rng.standard_normal(d)[None, :])
        r = y - A @ w
        rhs = np.concatenate([r, -mu * w])
        op = stacked(A, mu)
        f = lsqr if solver == "lsqr" else lsmr
        res = f(op, rhs, atol=atol, btol=atol, conlim=0,
                **({"iter_lim": inner} if solver == "lsqr" else {"maxiter": inner}))
        w = w + res[0]
        tot += int(res[2])
        dg = lb.diag(prob, w)
        tr.append(dict(stage=l, mu_rel=mu / s1, it=int(res[2]), tot=tot, **dg))
        if verbose:
            print("   l=%2d mu/s1=%8.1e it=%4d tot=%5d res=%.3e ev=%.3e |w|=%.3e"
                  % (l, mu / s1, res[2], tot, dg["res"], dg["ev"], dg["wnorm"]))
    return w, tr


if __name__ == "__main__":
    cases = [(C.prob_qi1d, (128,)), (C.prob_qi1d, (256,)),
             (C.prob_twin, (512, "unstruct")), (C.prob_qi2d, (576,))]
    recs = []
    for f, a in cases:
        p = f(*a)
        print("== %s d=%d rank=%d kappa=%.2e floor=%.2e"
              % (p["label"], p["d"], p["rank"], p["kappa"], p["floor_pool"]))
        for step, inner in [(1.0, 60), (0.5, 40), (0.25, 25)]:
            w, tr = continuation(p, decade_step=step, inner=inner)
            best = min(t["ev"] for t in tr)
            print("   step=%.2f inner=%3d stages=%2d totit=%5d  final ev=%.3e"
                  " best=%.3e res=%.3e" % (step, inner, len(tr), tr[-1]["tot"],
                                           tr[-1]["ev"], best, tr[-1]["res"]))
            recs.append(dict(cell=p["label"], step=step, inner=inner, tr=tr,
                             floor=p["floor_pool"], d=p["d"]))
        print("   --- verbose, step=0.5 inner=40")
        continuation(p, decade_step=0.5, inner=40, verbose=True)
    lb.OUT.mkdir(parents=True, exist_ok=True)
    (lb.OUT / "e2_continuation.json").write_text(json.dumps(recs))
