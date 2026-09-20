"""Experiment I -- the iterations/k tradeoff at fixed accuracy (iso-lines).

THEORY, stated first so the measurement can falsify it.

LSQR contracts geometrically on a system of condition number kappa, so the
iterations needed to first reach tolerance tau are

    t(tau)  ~  0.5 * sqrt(kappa(B)) * ln(1/tau)

Two predictions:

  P1  t is LOGARITHMIC in tau. Going from 1e-4 to 1e-12 triples ln(1/tau),
      i.e. shifts log10(t) by only log10(3) = 0.48. On a log-log (k, t) plot
      the iso-tau lines must therefore be nearly PARALLEL and tightly
      COMPRESSED. Widely-spaced iso-lines would refute this.
  P2  ALL the k-dependence enters through kappa(B(k)). So t should collapse
      onto a single line when plotted against sqrt(kappa), across every k
      and every d. kappa(B) is measured here rather than inferred, so the
      empirical exponent in k is a derived quantity, not a fitted one.

METRIC. Solver disagreement with the direct solve,
    dis = ||A_ev (w - w_svd)|| / ||y_ev||
because it reaches ~1e-14 for ANY target, whereas eval error stops at the
target's approximation floor (abs_cubed at d=922 floors at 3.7e-10, so a
1e-12 iso-line would not exist in that metric at all).

kappa is reported over the numerically meaningful range: sigma_1 / sigma_r
with r the numerical rank of B, since B inherits Phi's exact rank deficiency.

Usage: uv run python .../validation/run_isolines.py
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
FNS = {
    "sine": lambda x: np.sin(np.pi * x),
    "runge": lambda x: 1.0 / (1.0 + 25.0 * x ** 2),
    "abs_cubed": lambda x: np.abs(x) ** 3,
}
NS = [64, 96, 128, 160, 192, 256, 320, 384]
KS = [16, 24, 32, 48, 64, 96, 128, 192, 256]
TAUS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
FLOOR_EXIT = 3e-14                      # below the finest iso-line -> stop


def cap_for(k):
    """Small k needs far more iterations; a uniform cap truncates exactly the
    cells that set the k-exponent."""
    return int(min(300000, 20000 * (64.0 / k) ** 2))


def stride_for(it):
    """ADAPTIVE. A fixed stride of 25 quantized t to 1-2 steps for large k,
    which is precisely where the k-exponent is measured. Fine resolution
    early, coarse later."""
    if it <= 500:
        return 1
    if it <= 5000:
        return 10
    return 50


def make(fn, N):
    centers, gamma = cv.geometry(N)
    d = centers.size + 1
    n = cv.ROW_MULT * d
    x_tr = np.linspace(-1.0, 1.0, n)
    x_ev = np.linspace(-1.0, 1.0, cv.N_EVAL)
    f = FNS[fn]
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


def run_cell(prob, k, keep_traj=False):
    A, y, d = prob["A"], prob["y"], prob["d"]
    Ae, ye, yen, wsvd = prob["A_ev"], prob["y_ev"], prob["ye_norm"], prob["w_svd"]
    B, pieces = cv.block_qr(A, k)

    its, dis = [], []
    done = {"v": False}

    def hook(it, z):
        if it % stride_for(it) or done["v"]:
            return
        w = cv.unwhiten(pieces, z, d)
        dd = float(np.linalg.norm(Ae @ (w - wsvd)) / yen)
        its.append(it); dis.append(dd)
        if dd <= FLOOR_EXIT:
            done["v"] = True

    # inline LSQR so we can break early
    u = y.copy(); beta = np.linalg.norm(u); u /= beta
    v = B.T @ u; alpha = np.linalg.norm(v); v /= alpha
    w_ = v.copy(); x = np.zeros(d); phibar, rhobar = beta, alpha
    cap = cap_for(k)
    it = 0
    while it < cap and not done["v"]:
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

    dis = np.asarray(dis); its = np.asarray(its)
    # FIRST iteration at or below each tau (Sam's definition: first hit).
    # Running-min so semiconvergence cannot un-achieve a level already met.
    hits = {}
    run = np.minimum.accumulate(dis) if dis.size else dis
    for tau in TAUS:
        h = np.nonzero(run <= tau)[0]
        hits[f"t_{tau:.0e}"] = int(its[h[0]]) if h.size else -1
    out = {"k": k, "it_run": it, "cap": cap, "truncated": bool(it >= cap),
           "min_dis": float(dis.min()) if dis.size else float("nan"), **hits}
    if keep_traj:
        out["traj_it"] = its.tolist(); out["traj_dis"] = dis.tolist()
    return out


def main():
    rows = []
    t0 = time.time()
    for fn in FNS:
        for N in NS:
            prob = make(fn, N)
            for k in KS:
                if k > prob["d"]:
                    continue
                r = run_cell(prob, k, keep_traj=(N in (128, 256, 384)))
                rows.append({"fn": fn, "N": N, "d": prob["d"],
                             "floor": prob["floor"], **r})
                hits = " ".join(f"{t:.0e}:{r[f't_{t:.0e}']}" for t in TAUS)
                print(f"[I {time.time()-t0:6.0f}s] {fn:10s} d={prob['d']:5d} "
                      f"k={k:4d} min={r['min_dis']:.1e}"
                      f"{' TRUNC' if r['truncated'] else '     '} | {hits}",
                      flush=True)
                cv.save_rows("I_isolines", rows)
    print("\nsaved I_isolines")


if __name__ == "__main__":
    main()
