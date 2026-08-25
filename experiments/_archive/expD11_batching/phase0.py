"""PHASE 0 -- ballpark probe for the progressively-unregularized solve.

No figures, no full battery.  Answers only the questions that set the scales
for the real experiment:

  1. sigma_1, r, kappa for each Phi (so mu can be stated scale-free)
  2. what accuracy does each mu converge to?          -> mu spacing
  3. how many LSQR iterations does each stage need?   -> iteration budget
  4. does warm-starting from the previous stage cut those iterations?
  5. where does c=0 stop working?                     -> the handoff point

Ladder is parameterised scale-free as mu = (alpha * sigma_1)^2, so alpha is the
singular value the damping sits at, relative to the largest.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import core11 as c
from window_law import lsqr_window
import disproof as D

T0 = time.time()
I = (lambda v: v)
MAXIT = 4000


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def probs():
    out = []
    A, y, Ae, ye = D.qi_system(256, 0.30)
    out.append(("QI-1D N=256", A, y, Ae, ye))
    p = c.prob_qi2d(N=256, geom="radon_tensor", lam=0.12, row_mult=6)
    out.append(("QI-2D N=256", p["A"], p["y"], p["A_ev"], p["y_ev"]))
    p = c.prob_twin(d=512, kind="unstruct", row_mult=4)
    out.append(("twin-unstruct d=512", p["A"], p["y"], p["A_ev"], p["y_ev"]))
    return out


def aug(A, b, mu, d):
    if mu <= 0:
        return A, b
    return np.vstack([A, np.sqrt(mu) * np.eye(d)]), np.concatenate([b, np.zeros(d)])


def run(name, A, y, Ae, ye):
    d = A.shape[1]
    ny = float(np.linalg.norm(ye))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = int((s > 1e-15 * s[0]).sum())
    ferr = lambda w: float(np.linalg.norm(Ae @ w - ye) / ny)
    floor = ferr(Vt.T @ ((U.T @ y) / np.where(s > 1e-15 * s[0],
                                              np.where(s > 1e-15 * s[0], s, 1), np.inf)))
    log(f"=== {name}: d={d} r={r} sigma1={s[0]:.3e} "
        f"kappa={s[0]/s[r-1]:.2e} floor={floor:.2e}")
    print(f"    {'alpha':>9}{'mu':>11}{'kap_mu':>9}{'dampopt':>11}"
          f"{'iter cold':>11}{'iter warm':>11}{'reached':>11}")
    alphas = [10.0 ** (-k) for k in range(1, 15)]
    w = np.zeros(d)
    tot = 0
    prev_reach = None
    for a in alphas:
        mu = float((a * s[0]) ** 2)
        kap = float(np.sqrt((s[0] ** 2 + mu) / (s[r - 1] ** 2 + mu)))
        wex = Vt.T @ ((s / (s ** 2 + mu)) * (U.T @ y))
        dop = ferr(wex)
        tgt = 1.3 * dop

        def iters_to(w0):
            lo, hi = 1, MAXIT
            Aa, ya = aug(A, y - A @ w0, mu, d)
            best = None
            for T in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4000):
                x, _ = lsqr_window(Aa, ya, I, d, T, 0)
                if ferr(w0 + x) <= tgt:
                    best = T
                    break
            return best, (w0 + x)

        it_cold, _ = iters_to(np.zeros(d))
        it_warm, w_new = iters_to(w)
        reached = ferr(w_new)
        if it_warm is not None:
            w = w_new
            tot += it_warm
        print(f"    {a:>9.0e}{mu:>11.1e}{kap:>9.1e}{dop:>11.2e}"
              f"{str(it_cold):>11}{str(it_warm):>11}{reached:>11.2e}")
        if it_warm is None and it_cold is None:
            print(f"    -> c=0 can no longer reach this level: HANDOFF at "
                  f"alpha={a:.0e}, accuracy {prev_reach:.2e}")
            break
        prev_reach = reached
    print(f"    total warm-start iterations down the ladder: {tot}")
    return tot


if __name__ == "__main__":
    for nm, A, y, Ae, ye in probs():
        run(nm, A, y, Ae, ye)
    log("phase 0 done")
