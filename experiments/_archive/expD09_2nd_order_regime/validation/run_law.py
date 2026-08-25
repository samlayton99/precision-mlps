"""Experiment H -- the (d, k, iterations) law, measured without confounds.

Everything before this was confounded one way or another:

  B  fixed iteration budget          -> k looked coupled
  E  ratio-to-floor (saturates)      -> k=32 looked fine
  F  best-over-trajectory + saturating metric -> everything looked exactly 1.0
  G  budget scaled as 1/k, which UNDER-fed small k (k=32 got 8436 iterations
     when E had already measured it needs ~16750 at d=922) -> k=32 looked
     coupled again

Fixes here:

1. NO ITERATION BUDGET AS AN INDEPENDENT VARIABLE. Every run uses the same
   STOPPING CRITERION -- patience on the running minimum -- and runs until it
   triggers or hits a hard cap. Cells that hit the cap are flagged and are
   NOT used for the flat-vs-rising verdict.
2. ALL METRICS AT A SINGLE ITERATE. Reporting min(eval) and min(disagreement)
   separately compares different points on the path, which is how F produced
   an algebraically impossible pair. Here the iterate is chosen once (the
   argmin of the disagreement, i.e. semiconvergence-aware early stopping,
   which Sam has explicitly accepted) and every number is read there.
3. NON-SATURATING TARGETS. Smooth targets bottom out at machine epsilon, so
   the floor stops falling and the metric loses resolution. Limited-smoothness
   targets keep improving with d, which is what makes a scaling study
   meaningful. |x| is C^0, |x|^3 is C^2; both have polynomially-decaying
   floors that never reach eps at these widths.

Outputs, per (target, d, k): iterations to the minimum, the achieved eval
error and SVD-disagreement AT that iterate, the floor, and a capped flag.

Usage: uv run python .../validation/run_law.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import common_val as cv  # noqa: E402

cv.ROW_MULT = 4
NS = [64, 96, 128, 192, 256, 384, 512]
KS = [16, 32, 64, 128, 256]
STRIDE = 100
# NO early stopping. A patience rule was measured to starve small k: at
# abs_cubed d=922, k=32 stopped at 6475 iterations with dis=1.0e-9, but
# running on to 60000 reaches 1.9e-13 at iteration 42400 -- four orders
# better. Any stopping rule that fires before the true minimum reproduces
# the same budget confound that invalidated experiments B and G. So: run to
# a generous per-cell cap, take the min over the whole trajectory (Sam has
# explicitly accepted min/early-stopping as the operating point), and FLAG
# cells whose minimum lands near the cap as not-converged.
HARD_CAP = 100000


def cap_for(k, d):
    """Empirical law from E/H: iterations ~ d^1.6 / k^2.5. Generous x3."""
    est = 6000.0 * (d / 206.0) ** 1.6 * (64.0 / k) ** 2.5
    return int(min(HARD_CAP, max(3000, est)))

# Limited-smoothness targets: the floor keeps falling with d, so the study
# never runs out of resolution.
ROUGH = {
    "abs": lambda x: np.abs(x),                 # C^0
    "abs_cubed": lambda x: np.abs(x) ** 3,      # C^2
}


def make_rough(fn, N):
    centers, gamma = cv.geometry(N)
    d = centers.size + 1
    n = cv.ROW_MULT * d
    x_tr = np.linspace(-1.0, 1.0, n)
    x_ev = np.linspace(-1.0, 1.0, cv.N_EVAL)
    f = ROUGH[fn]
    y, y_ev = f(x_tr), f(x_ev)
    A = cv.build_phi_aug(x_tr, gamma, centers)
    A_ev = cv.build_phi_aug(x_ev, gamma, centers)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > cv.RCOND_SVD * s[0]
    w_svd = Vt.T @ (np.where(keep, 1 / np.where(keep, s, 1), 0) * (U.T @ y))
    yen = float(np.linalg.norm(y_ev))
    return {"fn": fn, "N": N, "d": d, "n": n, "rank": int(keep.sum()),
            "A": A, "y": y, "A_ev": A_ev, "y_ev": y_ev, "ye_norm": yen,
            "w_svd": w_svd,
            "floor": float(np.linalg.norm(A_ev @ w_svd - y_ev) / yen)}


def run_cell(prob, k, cap):
    A, y, d = prob["A"], prob["y"], prob["d"]
    Ae, ye, yen, wsvd = prob["A_ev"], prob["y_ev"], prob["ye_norm"], prob["w_svd"]
    B, pieces = cv.block_qr(A, k)
    dis, ev = [], []

    def hook(it, z):
        if it % STRIDE:
            return
        w = cv.unwhiten(pieces, z, d)
        dis.append(float(np.linalg.norm(Ae @ (w - wsvd)) / yen))
        ev.append(float(np.linalg.norm(Ae @ w - ye) / yen))

    n_, d_ = B.shape
    u = y.copy(); beta = np.linalg.norm(u); u /= beta
    v = B.T @ u; alpha = np.linalg.norm(v); v /= alpha
    w_ = v.copy(); x = np.zeros(d_); phibar, rhobar = beta, alpha
    it = 0
    while it < cap:
        it += 1
        uu = B @ v - alpha * u; beta = np.linalg.norm(uu)
        if beta <= 0:
            break
        u = uu / beta
        vv = B.T @ u - beta * v; alpha = np.linalg.norm(vv)
        if alpha <= 0:
            break
        v = vv / alpha
        rho = np.hypot(rhobar, beta); c, s = rhobar / rho, beta / rho
        theta = s * alpha; rhobar = -c * alpha
        phi = c * phibar; phibar = s * phibar
        x = x + (phi / rho) * w_
        w_ = v - (theta / rho) * w_
        hook(it, x)

    dis = np.asarray(dis); ev = np.asarray(ev)
    if dis.size == 0:
        return None
    i = int(dis.argmin())                     # ONE iterate; all metrics read here
    it_min = (i + 1) * STRIDE
    # thresholds crossed, for the iterations-vs-(d,k) law
    cross = {}
    for tol in (1e-8, 1e-10, 1e-12):
        h = np.nonzero(dis <= tol)[0]
        cross[f"it_{tol:.0e}"] = int(h[0] + 1) * STRIDE if h.size else -1
    return {"k": k, "it_run": it, "it_min": it_min,
            "dis": float(dis[i]), "ev": float(ev[i]),
            # minimum in the last 20% of the run => probably still improving
            "not_converged": bool(i >= 0.8 * len(dis)),
            "cap": cap, **cross}


def main():
    rows = []
    t0 = time.time()
    for fn in ROUGH:
        for N in NS:
            prob = make_rough(fn, N)
            for k in KS:
                if k > prob["d"]:
                    continue
                r = run_cell(prob, k, cap_for(k, prob["d"]))
                if r is None:
                    continue
                rows.append({"fn": fn, "N": N, "d": prob["d"],
                             "rank": prob["rank"], "floor": prob["floor"],
                             **r})
                print(f"[H {time.time()-t0:6.0f}s] {fn:10s} d={prob['d']:5d} "
                      f"k={k:4d} | it_min={r['it_min']:7d} cap={r['cap']:7d}"
                      f"{' NOTCONV' if r['not_converged'] else '        '} | "
                      f"dis={r['dis']:.2e} ev={r['ev']:.2e} "
                      f"floor={prob['floor']:.2e}", flush=True)
                cv.save_rows("H_law", rows)
    print("\nsaved H_law")


if __name__ == "__main__":
    main()
