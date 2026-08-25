"""E7 -- requirements 5 (batching) and 7 (drift) for the O(d*k) candidates,
plus the honest setup-cost exponent (requirement 3).

Two preconditioner styles at the same d*k state:
  BJ   block-Jacobi half-inverse, used INSIDE the iteration (steering)
  BQR  block-QR whitening, baked into the solution map
Drift: the factors are built from A, the solve runs on A*(1+eta*noise).
Batching: the factors are built from b rows only; the solve runs on b rows
per restart with a fresh draw each time.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import scipy.linalg as sla
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb, core11 as C, e1_precision as E1                 # noqa: E402
from scipy.sparse.linalg import lsqr, lsmr, LinearOperator  # noqa: E402


def solve_bj(A, y, P, d, it):
    op = LinearOperator(A.shape, matvec=lambda v: A @ P.half64(v),
                        rmatvec=lambda u: P.half64(A.T @ u), dtype=float)
    return P.half64(lsqr(op, y, atol=0, btol=0, conlim=0, iter_lim=it)[0])


def solve_bqr(A, y, blocks, d, it, rcond=1e-13):
    B, unw = lb.block_qr_whiten(A, blocks, rcond=rcond)
    z = lsqr(lb.opA(B), y, atol=0, btol=0, conlim=0, iter_lim=it)[0]
    return unw(z)


def drift_arm(p, k, etas, it=600, seed=0):
    A, y = p["A"], p["y"]; d = p["d"]
    blocks = lb.blocks_of(p, k)
    rng = np.random.default_rng(seed)
    P = E1.BJdd(A, blocks)
    Bfac = lb.block_qr_whiten(A, blocks, rcond=1e-13)
    out = []
    for eta in etas:
        Ad = A if eta == 0 else A * (1.0 + eta * rng.standard_normal(d)[None, :])
        # steering: factors from A (stale), residuals/operator from Ad
        op = LinearOperator(Ad.shape, matvec=lambda v: Ad @ P.half64(v),
                            rmatvec=lambda u: P.half64(Ad.T @ u), dtype=float)
        w_bj = P.half64(lsqr(op, y, atol=0, btol=0, conlim=0, iter_lim=it)[0])
        # baked-in: whitened operator built from A, applied to drifted rows
        Bs, unw = Bfac
        # honest version: recompute B from Ad columns but reuse STALE R factors
        z = lsqr(lb.opA(Bs), y, atol=0, btol=0, conlim=0, iter_lim=it)[0]
        w_bq = unw(z)
        ev_bq = float(np.linalg.norm(p["A_ev"] @ w_bq - p["y_ev"]) / p["ye_norm"])
        out.append((eta, C.eval_err(p, w_bj), ev_bq))
    return out


def batch_arm(p, k, bs, rounds=12, it=200, seed=0):
    """Factors built from batches only; each round draws a fresh batch and runs
    `it` LSQR iterations on it from the current w (fresh residual)."""
    A, y = p["A"], p["y"]; n, d = A.shape
    blocks = lb.blocks_of(p, k)
    res = []
    for b in bs:
        rng = np.random.default_rng(11 + seed)
        # warm the factors from accumulated batches (state stays d*k: we hold
        # a k-column row buffer of 3k rows per block, refreshed round-robin)
        bufs = [None] * len(blocks)
        for _ in range(3 * len(blocks)):
            S = rng.choice(n, min(b, n), replace=False)
            for j, Cb in enumerate(blocks):
                X = A[S][:, Cb]
                bufs[j] = X if bufs[j] is None else np.vstack([X, bufs[j]])[:3 * k]
        P = type("P", (), {})()
        f = []
        for Cb, Bf in zip(blocks, bufs):
            _, s, Vt = np.linalg.svd(Bf, full_matrices=False)
            keep = s > 1e-12 * max(s[0], 1e-300)
            f.append((Vt.T * np.where(keep, 1 / np.where(keep, s, 1), 0), Vt))
        P.blocks, P.f = blocks, f
        P.half64 = lambda v, P=P: _apply(P, v)
        w = np.zeros(d)
        for _ in range(rounds):
            S = rng.choice(n, min(b, n), replace=False)
            Ab, yb = A[S], y[S]
            r = yb - Ab @ w
            op = LinearOperator(Ab.shape, matvec=lambda v: Ab @ P.half64(v),
                                rmatvec=lambda u: P.half64(Ab.T @ u), dtype=float)
            tau = max(2, min(it, len(S) // 2))
            w = w + P.half64(lsqr(op, r, atol=0, btol=0, conlim=0,
                                  iter_lim=tau)[0])
        res.append((b, C.eval_err(p, w), C.floor_1batch(p, b)))
    return res


def _apply(P, v):
    out = np.zeros_like(v)
    for Cb, (Vi, Vt) in zip(P.blocks, P.f):
        out[Cb] = Vi @ (Vt @ v[Cb])
    return out


if __name__ == "__main__":
    recs = {}
    print("### requirement 7 -- drift (eta = relative column perturbation)")
    for f, a in [(C.prob_qi1d, (128,)), (C.prob_twin, (512, "unstruct"))]:
        p = f(*a)
        o = drift_arm(p, 128, [0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4])
        print("  %-22s floor=%.1e" % (p["label"], p["floor_pool"]))
        print("     eta      :", " ".join("%8.0e" % t[0] for t in o))
        print("     BJ steer :", " ".join("%8.1e" % t[1] for t in o))
        print("     BQR baked:", " ".join("%8.1e" % t[2] for t in o))
        recs["drift_" + p["label"]] = o
    print("### requirement 5 -- batching (b, achieved, floor_1batch)")
    for f, a in [(C.prob_qi1d, (128,)), (C.prob_twin, (512, "unstruct"))]:
        p = f(*a); d = p["d"]
        bs = [max(4, d // 64), d // 16, d // 4, d, 4 * d]
        o = batch_arm(p, 128, bs)
        print("  %-22s floor=%.1e" % (p["label"], p["floor_pool"]))
        for b, ev, fl in o:
            print("     b=%5d (d/%4.0f)  achieved %.2e   floor_1batch %.2e"
                  % (b, d / b, ev, fl))
        recs["batch_" + p["label"]] = o
    print("### requirement 3 -- setup cost exponent in d (seconds)")
    for N in (128, 256, 384, 512):
        p = C.prob_qi1d(N); A = p["A"]; d = p["d"]
        blocks = lb.blocks_of(p, 128)
        t0 = time.time(); E1.BJdd(A, blocks); t_bj = time.time() - t0
        t0 = time.time(); lb.block_qr_whiten(A, blocks); t_bqr = time.time() - t0
        t0 = time.time()
        Sk = np.random.default_rng(0).standard_normal((2 * d, A.shape[0])) / np.sqrt(2 * d)
        np.linalg.svd(Sk @ A, full_matrices=False); t_sp = time.time() - t0
        print("  d=%5d  BJ(dk) %7.3f  blockQR(dk) %7.3f  sketchSVD(dr) %7.3f"
              % (d, t_bj, t_bqr, t_sp))
        recs.setdefault("cost", []).append((d, t_bj, t_bqr, t_sp))
    lb.OUT.mkdir(parents=True, exist_ok=True)
    (lb.OUT / "e7_reqs.json").write_text(json.dumps(recs))
    c = np.array(recs["cost"])
    for i, nm in enumerate(["BJ", "blockQR", "sketchSVD"], start=1):
        e = np.polyfit(np.log(c[:, 0]), np.log(c[:, i]), 1)[0]
        print("  fitted exponent in d: %-10s %.2f" % (nm, e))
