"""Final hardening battery for the step-2 part. Attacks the claims register:

  H1  A1/A2/A3: full 6-target family x N in {64,128,256,512}, FS stopping
  H2  C1/C2/D4: scale to d=1846 (and 3690 if it fits), iterations vs d,
                one implicit-compensated parity point at large d
  H3  B1/B4: spectral-twin random matrices --
        (i)  unstructured twin: same spectrum/rank/energy profile as the QI
             system but random singular vectors. Round-4's information
             argument predicts NO O(d) method reaches the floor here; the
             point is to MEASURE the boundary and the pre-solve detector.
        (ii) hidden-cluster twin: within-cluster correlated, across-cluster
             independent, columns randomly permuted -- ordering exists but
             is a grouping, not a band. Seriation must recover it.
        detector: within-block |corr| mass fraction rho_blk from the same
        s-row sample seriation uses, computed BEFORE any solve.
  H4  D1/D2: noise x batching jointly (sigma=1e-4, b in {4d,2d})
  H5  A2: knob sweeps rcond x atol on two cells (plateau or cliff?)
  H6  B2: seriated arm with the cap raised 4x (was 8000) on the worst cell
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import lsmr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core10 as c                      # noqa: E402

OUT = c.RESULTS
EV = {}
RNG = np.random.default_rng(2029)
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def fs_solve(A, y, k=128, order=None, maxiter=8000, atol=1e-15, rcond=1e-13):
    pieces, B = c.whiten(A, k, order=order, rcond=rcond)
    res = lsmr(B, y, atol=atol, btol=atol, conlim=0, maxiter=maxiter)
    return c.unwhiten(pieces, res[0], A.shape[1]), int(res[2])


# ---------------------------------------------------------------- H1 -------
def h1_targets():
    rows = []
    for fn in ("sine", "sine_8pi", "runge", "sine_mixture", "exp", "abs_cubed"):
        for N in (64, 128, 256, 512):
            prob = c.cv.make_problem(fn, N)
            A, y = prob["A"], prob["y"]
            w, it = fs_solve(A, y)
            rows.append({"fn": fn, "N": N, "d": A.shape[1],
                         "eval": c.eval_err(prob, w), "iters": it,
                         "floor": prob["floor_svd"]})
            log(f"H1 {fn:12s} N={N:4d} eval={rows[-1]['eval']:.2e} "
                f"floor={prob['floor_svd']:.2e} it={it}")
    EV["h1"] = rows


# ---------------------------------------------------------------- H2 -------
def h2_scale():
    rows = []
    for fn, N in (("sine", 512), ("abs_cubed", 512), ("sine", 1024),
                  ("abs_cubed", 1024)):
        prob = c.cv.make_problem(fn, N)
        A, y = prob["A"], prob["y"]
        w, it = fs_solve(A, y, maxiter=30000)
        rows.append({"fn": fn, "N": N, "d": A.shape[1],
                     "eval": c.eval_err(prob, w), "iters": it,
                     "floor": prob["floor_svd"]})
        log(f"H2 {fn} N={N} d={A.shape[1]} eval={rows[-1]['eval']:.2e} "
            f"floor={prob['floor_svd']:.2e} it={it}")
    # implicit-compensated parity at d=1846
    prob = c.cv.make_problem("sine", 1024)
    A, y = prob["A"], prob["y"]
    pieces, _ = c.whiten(A, 128)
    op = c.implicit_dd_operator(A, pieces)
    res = lsmr(op, y, atol=1e-15, btol=1e-15, conlim=0, maxiter=30000)
    e = c.eval_err(prob, c.unwhiten(pieces, res[0], A.shape[1]))
    rows.append({"fn": "sine", "N": 1024, "d": A.shape[1], "arm": "implicit_dd",
                 "eval": e, "iters": int(res[2])})
    log(f"H2 implicit-dd sine N=1024: eval={e:.2e} it={int(res[2])}")
    EV["h2"] = rows


# ---------------------------------------------------------------- H3 -------
def _spectral_twin(d, n_tr, n_ev, seed, kind):
    """Random matrix with the QI system's spectral fingerprint.

    Spectrum: log-linear decay sigma_1=1 -> sigma_r=1e-14 at r=0.6d, hard
    zero after (rank cliff). Target energy along u_i proportional to
    sigma_i. `kind`:
      unstructured    U, V Haar-random -- no correlation structure at all
      hidden_cluster  columns are 32-wide clusters, each cluster an
                      ill-conditioned mix of its own 32-dim random factor
                      (within-cluster corr, across-cluster independent),
                      columns then randomly permuted
    """
    rng = np.random.default_rng(seed)
    n = n_tr + n_ev
    if kind == "unstructured":
        r = int(0.6 * d)
        sig = np.logspace(0, -14, r)
        U = np.linalg.qr(rng.standard_normal((n, r)))[0]
        V = np.linalg.qr(rng.standard_normal((d, r)))[0]
        A = (U * sig) @ V.T
        g = rng.standard_normal(r)
        y = U @ (sig * g)
        perm = np.arange(d)
    else:
        kc = 32
        m = d // kc
        cols = []
        for j in range(m):
            X = rng.standard_normal((n, kc))
            W = (np.linalg.qr(rng.standard_normal((kc, kc)))[0]
                 * np.logspace(0, -12, kc)) @ \
                np.linalg.qr(rng.standard_normal((kc, kc)))[0]
            cols.append(X @ W)
        A = np.hstack(cols)
        perm = rng.permutation(A.shape[1])
        A = A[:, perm]
        Uf, sf, Vtf = np.linalg.svd(A, full_matrices=False)
        keep = sf > 1e-15 * sf[0]
        g = rng.standard_normal(int(keep.sum()))
        y = Uf[:, keep] @ (sf[keep] * g)
    y = y / np.linalg.norm(y) * np.sqrt(n)
    return {"A": A[:n_tr], "y": y[:n_tr], "A_ev": A[n_tr:], "y_ev": y[n_tr:],
            "ye_norm": float(np.linalg.norm(y[n_tr:])), "perm": perm}


def _detector(A_sample, order, k=128):
    """rho_blk: fraction of off-diagonal |corr| mass captured inside blocks
    of the given order. Computed from the SAME sample seriation uses."""
    ns = np.linalg.norm(A_sample, axis=0)
    cn = A_sample / np.maximum(ns, 1e-300)
    W = np.abs(cn.T @ cn)
    np.fill_diagonal(W, 0.0)
    tot = W.sum()
    inb = 0.0
    for t in range(0, W.shape[0], k):
        C = order[t:t + k]
        inb += W[np.ix_(C, C)].sum()
    return float(inb / max(tot, 1e-300))


def _eval_tw(tw, w):
    return float(np.linalg.norm(tw["A_ev"] @ w - tw["y_ev"]) / tw["ye_norm"])


def _svd_floor_tw(tw):
    U, s, Vt = np.linalg.svd(tw["A"], full_matrices=False)
    keep = s > 1e-15 * s[0]
    a = (Vt.T * np.where(keep, 1 / np.where(keep, s, 1), 0)) @ (U.T @ tw["y"])
    return _eval_tw(tw, a), a


def _spir_tw(tw, rounds=8, inner=200, seed=0):
    A, y = tw["A"], tw["y"]
    n, d = A.shape
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((min(n, 2 * d), n)) / np.sqrt(min(n, 2 * d))
    U2, s2, V2t = np.linalg.svd(S @ A, full_matrices=False)
    rk = int((s2 > 1e-15 * s2[0]).sum())
    P = V2t[:rk].T / s2[:rk]
    from scipy.sparse.linalg import LinearOperator
    B = LinearOperator((n, rk), matvec=lambda u: A @ (P @ u),
                       rmatvec=lambda v: P.T @ (A.T @ v), dtype=float)
    w = np.zeros(d)
    for _ in range(rounds):
        w = w + P @ lsmr(B, y - A @ w, atol=0, btol=0, conlim=0,
                         maxiter=inner)[0]
    return w


def h3_twins():
    rows = []
    for kind in ("unstructured", "hidden_cluster"):
        for d in (270, 922):
            tw = _spectral_twin(d, 4 * d, 1000, seed=d, kind=kind)
            floor, _ = _svd_floor_tw(tw)
            sr = RNG.choice(tw["A"].shape[0], min(256, tw["A"].shape[0]),
                            replace=False)
            order = c.seriate(tw["A"][sr])
            det_nat = _detector(tw["A"][sr], np.arange(tw["A"].shape[1]))
            det_ser = _detector(tw["A"][sr], order)
            w_nat, it_n = fs_solve(tw["A"], tw["y"], maxiter=20000)
            w_ser, it_s = fs_solve(tw["A"], tw["y"], order=order, maxiter=20000)
            w_sp = _spir_tw(tw)
            rec = {"kind": kind, "d": d, "floor": floor,
                   "natural": _eval_tw(tw, w_nat), "it_nat": it_n,
                   "seriated": _eval_tw(tw, w_ser), "it_ser": it_s,
                   "spir": _eval_tw(tw, w_sp),
                   "rho_natural": det_nat, "rho_seriated": det_ser}
            rows.append(rec)
            log(f"H3 {kind:14s} d={d}: floor={floor:.1e} nat={rec['natural']:.1e} "
                f"ser={rec['seriated']:.1e} spir={rec['spir']:.1e} "
                f"rho_nat={det_nat:.2f} rho_ser={det_ser:.2f}")
    # QI control detector values
    prob = c.cv.make_problem("sine", 256)
    sr = RNG.choice(prob["A"].shape[0], 256, replace=False)
    ordq = c.seriate(prob["A"][sr])
    EV["h3_control_rho"] = {
        "qi_natural": _detector(prob["A"][sr], np.arange(prob["A"].shape[1])),
        "qi_seriated": _detector(prob["A"][sr], ordq)}
    log(f"H3 control rho (QI sine N=256): {EV['h3_control_rho']}")
    EV["h3"] = rows


# ---------------------------------------------------------------- H4 -------
def h4_noise_batch():
    rows = []
    rng = np.random.default_rng(4)
    for fn in ("sine", "runge", "sine_8pi"):
        for N in (128, 256):
            prob = c.cv.make_problem(fn, N, noise_rel=1e-4)
            A, y = prob["A"], prob["y"]
            n, d = A.shape
            r = prob["rank"]
            for mult in (4, 2):
                b = mult * d
                sel = rng.choice(n, b, replace=False)
                w, it = fs_solve(A[sel], y[sel])
                stat_b = 1e-4 * np.sqrt(r / b)
                rows.append({"fn": fn, "N": N, "bmult": mult,
                             "eval": c.eval_err(prob, w), "iters": it,
                             "floor_stat_batch": float(stat_b)})
                log(f"H4 {fn} N={N} b={mult}d: eval={rows[-1]['eval']:.2e} "
                    f"stat(b)={stat_b:.2e} it={it}")
    EV["h4"] = rows


# ---------------------------------------------------------------- H5 -------
def h5_knobs():
    rows = []
    for fn, N in (("sine", 256), ("abs_cubed", 256)):
        prob = c.cv.make_problem(fn, N)
        A, y = prob["A"], prob["y"]
        for rcond in (1e-11, 1e-13, 1e-15):
            for atol in (1e-14, 1e-15, 1e-16):
                w, it = fs_solve(A, y, rcond=rcond, atol=atol, maxiter=20000)
                rows.append({"fn": fn, "N": N, "rcond": rcond, "atol": atol,
                             "eval": c.eval_err(prob, w), "iters": it})
        log(f"H5 {fn} N={N}: " + " ".join(
            f"rc={r['rcond']:.0e}/at={r['atol']:.0e}:{r['eval']:.1e}"
            for r in rows[-9:]))
    EV["h5"] = rows


# ---------------------------------------------------------------- H6 -------
def h6_cap():
    prob = c.cv.make_problem("sine_8pi", 256)
    A, y = prob["A"], prob["y"]
    d = A.shape[1]
    perm = RNG.permutation(d)
    Ap = A[:, perm]
    sr = RNG.choice(A.shape[0], 256, replace=False)
    order = c.seriate(Ap[sr])
    out = {}
    for cap in (8000, 32000):
        wp, it = fs_solve(Ap, y, order=order, maxiter=cap)
        w = np.zeros(d)
        w[perm] = wp
        out[str(cap)] = {"eval": c.eval_err(prob, w), "iters": it}
        log(f"H6 cap={cap}: eval={out[str(cap)]['eval']:.2e} it={it}")
    EV["h6"] = out


def main():
    h1_targets()
    h4_noise_batch()
    h5_knobs()
    h6_cap()
    h3_twins()
    h2_scale()
    (OUT / "hardening.json").write_text(json.dumps(EV, indent=1))
    log(f"saved {OUT/'hardening.json'}")


if __name__ == "__main__":
    main()
