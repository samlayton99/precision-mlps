"""E8 -- corrected requirement 5 / 7 / 3 arms.

Fixes two problems in E7:
  * the block-QR (baked-in) drift arm never applied the drifted operator, so it
    reported a flat line.  Correct test: the R factors are STALE (built from A)
    and the whitened operator is rebuilt from the DRIFTED rows,
    B' = A'_kept R_stale^{-1}, which is the ill-conditioned application the
    campaign's lesson T1/lesson-2 is about.
  * the batching arm was a hand-rolled restart loop and under-tuned.  Use the
    repo's own tuned streaming solver core11.type_a as the reference instead.

Setup cost is extended to larger d so the sketch SVD leaves the
overhead-dominated regime.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import scipy.linalg as sla
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb, core11 as C, e1_precision as E1                  # noqa: E402
from scipy.sparse.linalg import lsqr, LinearOperator          # noqa: E402


def stale_bqr(A, blocks, rcond=1e-13):
    """Persistent state: the R factors + kept-column bookkeeping (d*k)."""
    facs = []
    for Cb in blocks:
        Q, R, piv = sla.qr(A[:, Cb], mode="economic", pivoting=True)
        dg = np.abs(np.diag(R))
        keep = dg > rcond * max(dg[0], 1e-300)
        kk = int(keep.sum())
        facs.append((R[:kk, :kk], Cb[piv[:kk]], kk))
    return facs


def apply_bqr(facs, Aq, d):
    """Build the whitened operator from the CURRENT rows and the stale factors:
    B = [ A[:,cols_j] R_j^{-1} ]_j ."""
    cols = []
    for (R, cols_j, kk) in facs:
        cols.append(sla.solve_triangular(R.T, Aq[:, cols_j].T, lower=True).T)
    return np.hstack(cols)


def unwhiten_bqr(facs, z, d):
    w = np.zeros(d)
    o = 0
    for (R, cols_j, kk) in facs:
        w[cols_j] = sla.solve_triangular(R, z[o:o + kk], lower=False)
        o += kk
    return w


def drift_arm(p, k, etas, it=800, seed=0):
    A, y = p["A"], p["y"]; d = p["d"]
    blocks = lb.blocks_of(p, k)
    P = E1.BJdd(A, blocks)                # stale block-Jacobi (steering)
    facs = stale_bqr(A, blocks)           # stale block-QR (baked in)
    rng = np.random.default_rng(seed)
    out = []
    for eta in etas:
        Ad = A if eta == 0 else A * (1.0 + eta * rng.standard_normal(d)[None, :])
        op = LinearOperator(Ad.shape, matvec=lambda v: Ad @ P.half64(v),
                            rmatvec=lambda u: P.half64(Ad.T @ u), dtype=float)
        w_bj = P.half64(lsqr(op, y, atol=0, btol=0, conlim=0, iter_lim=it)[0])
        B = apply_bqr(facs, Ad, d)
        kapB = float(np.linalg.cond(B))
        z = lsqr(lb.opA(B), y, atol=0, btol=0, conlim=0, iter_lim=it)[0]
        w_bq = unwhiten_bqr(facs, z, d)
        out.append((eta, C.eval_err(p, w_bj), C.eval_err(p, w_bq), kapB))
    return out


if __name__ == "__main__":
    recs = {}
    print("### requirement 7 -- drift, corrected (k=128, 800 it)")
    for f, a in [(C.prob_qi1d, (128,)), (C.prob_twin, (512, "unstruct"))]:
        p = f(*a)
        o = drift_arm(p, 128, [0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4])
        print("  %-22s floor=%.1e" % (p["label"], p["floor_pool"]))
        print("     eta        :", " ".join("%9.0e" % t[0] for t in o))
        print("     BJ steering:", " ".join("%9.1e" % t[1] for t in o))
        print("     BQR baked  :", " ".join("%9.1e" % t[2] for t in o))
        print("     kappa(B')  :", " ".join("%9.1e" % t[3] for t in o))
        recs["drift_" + p["label"]] = o

    print("### requirement 5 -- core11.type_a (the repo's tuned streaming solver)")
    for f, a in [(C.prob_qi1d, (128,)), (C.prob_twin, (512, "unstruct"))]:
        p = f(*a); d = p["d"]
        print("  %-22s floor=%.1e" % (p["label"], p["floor_pool"]))
        row = []
        for b in [max(4, d // 64), d // 16, d // 4, d, 4 * d]:
            w, info = C.type_a(p, k=128, b=b, tau=200, n_eps=120,
                               variant="rotqr", seed=0)
            fl = C.floor_1batch(p, b)
            row.append((b, info["best"], fl, info["passes"]))
            print("     b=%5d (d/%4.1f)  type_a best %.2e  floor_1batch %.2e"
                  "  passes %.0f" % (b, d / b, info["best"], fl, info["passes"]))
        recs["batch_" + p["label"]] = row

    print("### requirement 3 -- setup cost, larger d")
    for N in (128, 256, 384, 512, 768):
        p = C.prob_qi1d(N); A = p["A"]; d = p["d"]
        blocks = lb.blocks_of(p, 128)
        t0 = time.time(); E1.BJdd(A, blocks); t_bj = time.time() - t0
        t0 = time.time(); stale_bqr(A, blocks); t_bqr = time.time() - t0
        rng = np.random.default_rng(0)
        t0 = time.time()
        Sk = rng.standard_normal((2 * d, A.shape[0])) / np.sqrt(2 * d)
        np.linalg.svd(Sk @ A, full_matrices=False); t_sp = time.time() - t0
        print("  d=%5d n=%6d  BJ %7.3f  blockQR %7.3f  sketchSVD %7.3f"
              % (d, A.shape[0], t_bj, t_bqr, t_sp))
        recs.setdefault("cost", []).append((d, t_bj, t_bqr, t_sp))
    c = np.array(recs["cost"])
    for i, nm in enumerate(["BJ", "blockQR", "sketchSVD"], start=1):
        e = np.polyfit(np.log(c[:, 0]), np.log(c[:, i]), 1)[0]
        e2 = np.polyfit(np.log(c[-3:, 0]), np.log(c[-3:, i]), 1)[0]
        print("  exponent in d: %-10s all=%.2f  top3=%.2f" % (nm, e, e2))
    lb.OUT.mkdir(parents=True, exist_ok=True)
    (lb.OUT / "e8_reqs.json").write_text(json.dumps(recs))
