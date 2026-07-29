"""expD10 -- hardening the block-QR recipe into the step-2 part.

Four engineering fixes are tested against the expD09 recipe's four failures:

  E1  ordering    column order unknown -> recover it by spectral seriation
                  of the |correlation| graph from an s-row Gram sample
  E2  memory      never materialize B: apply R^{-1} implicitly with the
                  matvec computed end-to-end in double-double (Krylov
                  recurrence stays fp64)
  E3  order-blind dd-LSQR + block-Jacobi with RANDOM blocks (candidate K2)
  E4  stopping    refinement rounds with fresh residuals; stop on the
                  round-level observable (||A^T r|| plateau + ||w|| guard)

Problems and floors come from expD09's validation suite (same geometry,
same noise protocol, same SVD reference).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, lsmr

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
D09 = REPO_ROOT / "experiments" / "expD09_2nd_order_regime"
for p in (str(REPO_ROOT), str(D09), str(D09 / "validation")):
    if p not in sys.path:
        sys.path.insert(0, p)

import common_val as cv                      # noqa: E402  problems + floors
import dd                                    # noqa: E402  dd primitives

RESULTS = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD10_step2_hardening"
FIGS = RESULTS / "figures"
RCOND_BLK = 1e-13


# ------------------------------------------------------------ fast dd ------

def _tree_csum(H, L, axis):
    """Compensated dd sum along `axis` by pairwise (tree) reduction.

    Replaces dd._csum0's python loop over the reduction axis (O(n) numpy
    calls) with O(log n) vectorized levels. Error ~ eps_dd * log2(n)."""
    H = np.moveaxis(H, axis, 0)
    L = np.moveaxis(L, axis, 0)
    while H.shape[0] > 1:
        m = H.shape[0]
        if m % 2 == 1:                        # carry the odd row forward
            H = np.concatenate([H, np.zeros((1,) + H.shape[1:])])
            L = np.concatenate([L, np.zeros((1,) + L.shape[1:])])
            m += 1
        h0, h1 = H[0::2], H[1::2]
        l0, l1 = L[0::2], L[1::2]
        s, e = dd.two_sum(h0, h1)
        e = e + (l0 + l1)
        H, L = dd.quick_two_sum(s, e)
    return H[0], L[0]


def dd_mat_ddvec(M, X):
    """M fp64 (n,m) @ X dd(m) -> dd(n), fully vectorized."""
    P, E = dd.two_prod(M, X[0][None, :])
    E = E + M * X[1][None, :]
    return _tree_csum(P, E, axis=1)


def dd_rmat_ddvec(M, X):
    """M.T fp64 (n,m) @ X dd(n) -> dd(m), fully vectorized."""
    P, E = dd.two_prod(M, X[0][:, None])
    E = E + M * X[1][:, None]
    return _tree_csum(P, E, axis=0)


def patch_dd_fast():
    """Route ddlsqr/ddpc's inner loops through the tree-reduced versions."""
    dd.matvec = dd_mat_ddvec
    dd.rmatvec = dd_rmat_ddvec


def dd_trisolve_upper(R, b_hi, b_lo=None):
    """Solve R x = b with R fp64 upper-triangular, arithmetic in dd.

    Column-oriented back substitution: nk python steps of vectorized dd ops.
    R's entries are exact fp64 inputs; all products/sums are dd."""
    nk = R.shape[0]
    xh = np.zeros(nk)
    xl = np.zeros(nk)
    bh = b_hi.astype(float).copy()
    bl = np.zeros(nk) if b_lo is None else b_lo.copy()
    for j in range(nk - 1, -1, -1):
        q = dd.sdiv((bh[j], bl[j]), (R[j, j], 0.0))
        xh[j], xl[j] = q
        if j:
            col = R[:j, j]
            p, e = dd.two_prod(col, q[0])
            e = e + col * q[1]
            s, e2 = dd.two_sum(bh[:j], -p)
            bl[:j] = bl[:j] + e2 - e
            bh[:j], bl[:j] = dd.quick_two_sum(s, bl[:j])
    return xh, xl


def dd_trisolve_lower(Rt, b_hi, b_lo=None):
    """Solve R^T x = b (R^T lower-triangular), arithmetic in dd."""
    nk = Rt.shape[0]
    xh = np.zeros(nk)
    xl = np.zeros(nk)
    bh = b_hi.astype(float).copy()
    bl = np.zeros(nk) if b_lo is None else b_lo.copy()
    for j in range(nk):
        q = dd.sdiv((bh[j], bl[j]), (Rt[j, j], 0.0))
        xh[j], xl[j] = q
        if j < nk - 1:
            col = Rt[j + 1:, j]
            p, e = dd.two_prod(col, q[0])
            e = e + col * q[1]
            s, e2 = dd.two_sum(bh[j + 1:], -p)
            bl[j + 1:] = bl[j + 1:] + e2 - e
            bh[j + 1:], bl[j + 1:] = dd.quick_two_sum(s, bl[j + 1:])
    return xh, xl


# ---------------------------------------------------------- blocking -------

def cluster_blocks(A_sample, kmax=128):
    """THE shipped blocking: agglomerative average-linkage on 1-|corr| from
    an s-row sample, cut so no cluster exceeds kmax, small clusters merged
    up to kmax. Subsumes band structure (measured: beats Fiedler seriation
    on the permuted QI system) and recovers hidden cluster structure exactly
    (measured: cluster twin d=922 to 9.7e-15 in 42 iterations vs a 20000-
    iteration stall for stride blocking)."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    ns = np.linalg.norm(A_sample, axis=0)
    cn = A_sample / np.maximum(ns, 1e-300)
    W = np.abs(cn.T @ cn)
    np.fill_diagonal(W, 1.0)
    D = squareform(np.maximum(1.0 - W, 0.0), checks=False)
    Z = linkage(D, method="average")
    lab = None
    for t in np.linspace(0.99, 0.2, 40):
        lab = fcluster(Z, t=t, criterion="distance")
        if np.bincount(lab).max() <= kmax:
            break
    blocks = [np.where(lab == j)[0] for j in np.unique(lab)]
    blocks.sort(key=len)
    merged = []
    cur = np.array([], dtype=int)
    for b in blocks:
        if len(cur) + len(b) <= kmax:
            cur = np.concatenate([cur, b])
        else:
            merged.append(cur)
            cur = b
    if len(cur):
        merged.append(cur)
    return merged


def kappa_gate(A_sample, blocks, rcond=RCOND_BLK):
    """Pre-solve applicability gate: whiten the SAMPLE with the candidate
    blocks and return the effective kappa of B_sample. Measured separation
    (at s >= d): QI band ~1e8-1e10 (works), matched clusters ~1e0
    (instant), structure-free spectral twin ~5e12 (block-QR cannot reach
    the floor; use an O(d*r) method or accept the stall). Gate at ~1e11.

    REQUIRES s >= d sample rows: a wide sample (s < d) is rank-deficient
    and silently UNDERESTIMATES kappa -- measured false-GO at d=922 with
    s=384 (read 6e8 against a true ~1e13). The clustering itself is fine
    at small s; only this kappa estimate needs the rows. In batched use
    the solve batch (b >= 4d) more than covers it."""
    s, d = A_sample.shape
    if s < d:
        raise ValueError(
            f"kappa_gate needs s >= d sample rows (got s={s}, d={d}); "
            "a wide sample rank-deficiently underestimates kappa (false GO)")
    B = np.zeros((s, d))
    for C in blocks:
        Q, R, piv = sla.qr(A_sample[:, C], mode="economic", pivoting=True)
        dg = np.abs(np.diag(R))
        nk = int((dg > rcond * max(dg[0], 1e-300)).sum())
        B[:, C[:nk]] = Q[:, :nk]
    sv = np.linalg.svd(B, compute_uv=False)
    sv = sv[sv > 1e-13]
    return float(sv[0] / sv[-1])


def plan_blocks(A_sample, k=128, k_budget=512, gate_go=1e11):
    """Ledger F1 policy: adaptive-k escalation behind the gate, on the
    SAMPLE only. Try k, 2k, ... up to k_budget; return the first blocking
    whose kappa_gate clears `gate_go`, else a no-go verdict. Never a silent
    stall: the caller always learns whether block-QR can reach the floor
    and at what state cost, before touching the full data.

    Returns dict(go, k, blocks, kappa, history)."""
    hist = []
    kk = k
    while kk <= k_budget:
        blocks = cluster_blocks(A_sample, kmax=kk)
        kap = kappa_gate(A_sample, blocks)
        hist.append({"k": kk, "kappa": kap})
        if kap < gate_go:
            return {"go": True, "k": kk, "blocks": blocks, "kappa": kap,
                    "history": hist}
        kk *= 2
    return {"go": False, "k": None, "blocks": None,
            "kappa": hist[-1]["kappa"], "history": hist}


def seriate(A_sample):
    """Recover a block-friendly column order from an s-row sample.

    Spectral seriation: Fiedler vector of the normalized Laplacian of the
    |correlation| graph. For a 1-D banded Gram this provably recovers the
    band order (up to reversal); the point of E1 is to measure it end to
    end through the whitening. Cost: O(s d^2) for the Gram + O(d^3) eigh
    on d = a few hundred; on a subset of a real model d_L is capped, and
    only ONE eigenvector is needed (Lanczos at scale).
    """
    ns = np.linalg.norm(A_sample, axis=0)
    cn = A_sample / np.maximum(ns, 1e-300)
    W = np.abs(cn.T @ cn)
    np.fill_diagonal(W, 0.0)
    deg = W.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-300))
    Lsym = np.eye(W.shape[0]) - (dinv[:, None] * W) * dinv[None, :]
    evals, evecs = np.linalg.eigh(Lsym)
    fiedler = dinv * evecs[:, 1]              # v = D^{-1/2} u_2
    return np.argsort(fiedler)


# ------------------------------------------------ whitening + solvers ------

def whiten(A, k, order=None, blocks=None, rcond=RCOND_BLK):
    """Stage 1. Blocking precedence: explicit `blocks` (from cluster_blocks)
    > contiguous runs of `order` > natural contiguous. Batching is handled
    at the call site by slicing rows (whiten and solve on the same rows).
    Returns (pieces, B). pieces: (C, piv, R[:nk,:], nk)."""
    n, d = A.shape
    if blocks is None:
        order = np.arange(d) if order is None else np.asarray(order)
        blocks = [order[t:t + k] for t in range(0, d, k)]
    pieces = []
    B = np.zeros((n, d))
    for C in blocks:
        Q, R, piv = sla.qr(A[:, C], mode="economic", pivoting=True)
        dg = np.abs(np.diag(R))
        nk = int((dg > rcond * max(dg[0], 1e-300)).sum())
        pieces.append((C, piv, R[:nk, :], nk))
        B[:, C[:nk]] = Q[:, :nk]
    return pieces, B


def unwhiten(pieces, z, d):
    w = np.zeros(d)
    for C, piv, R, nk in pieces:
        if nk == 0:
            continue
        c = sla.solve_triangular(R[:, :nk], z[C[:nk]], lower=False)
        out = np.zeros(len(C))
        out[piv[:nk]] = c
        w[C] = out
    return w


def dd_trisolve_lower_multi(Rt, Bh):
    """Solve R^T X = B for a MATRIX rhs (nk x m), arithmetic in dd,
    vectorized over the rhs columns: nk python steps of (m,)-vector dd ops."""
    nk, m = Rt.shape[0], Bh.shape[1]
    Xh = np.zeros((nk, m)); Xl = np.zeros((nk, m))
    bh = Bh.astype(float).copy(); bl = np.zeros((nk, m))
    for j in range(nk):
        # x_j = b_j / Rt[j,j]  (dd row / fp64 scalar)
        q = dd.two_prod(bh[j], 1.0 / Rt[j, j])
        qh, ql = dd.quick_two_sum(q[0], q[1] + bl[j] / Rt[j, j])
        # correction step for the scalar division
        p, e = dd.two_prod(qh, Rt[j, j])
        r = ((bh[j] - p) - e) + bl[j] - ql * Rt[j, j]
        qh, ql = dd.quick_two_sum(qh, ql + r / Rt[j, j])
        Xh[j], Xl[j] = qh, ql
        if j < nk - 1:
            col = Rt[j + 1:, j][:, None]
            p, e = dd.two_prod(col, qh[None, :])
            e = e + col * ql[None, :]
            s, e2 = dd.two_sum(bh[j + 1:], -p)
            bl[j + 1:] = bl[j + 1:] + e2 - e
            bh[j + 1:], bl[j + 1:] = dd.quick_two_sum(s, bl[j + 1:])
    return Xh, Xl


def refine_pieces(A, pieces):
    """Ledger F6 fix, correct version. The implicit operator W~ = A_kept
    R^{-1} (R^{-1} applied EXACTLY, in dd) deviates from orthonormal by the
    QR factorization's backward error ~eps*kappa(R) ~ 1e-6. Correct it with
    one CholeskyQR step measured in the right metric:

      X = R^{-T} A_kept^T   (dd multi-rhs trisolve => X = W~^T to ~eps64)
      G = X X^T              (fp64 fine: deviation 1e-6 >> eps64)
      S = chol(G)^T          (upper; kappa(S) ~ 1 + 1e-6)

    S is stored SEPARATELY and applied as a second stage (R^{-1} then
    S^{-1}); composing S@R in fp64 would reinject eps*kappa(R). Then
    W~ S^{-1} is orthonormal to ~deviation^2 ~ 1e-12. State: + nk^2 per
    block (still O(dk)); setup: one dd k-pass.
    Returns pieces with entries (C, piv, R, nk, S)."""
    out = []
    for C, piv, R, nk in pieces:
        if nk == 0:
            out.append((C, piv, R, nk, None))
            continue
        Xh, Xl = dd_trisolve_lower_multi(R[:, :nk].T, A[:, C[piv[:nk]]].T)
        X = Xh + Xl
        S = np.linalg.cholesky(X @ X.T).T
        out.append((C, piv, R, nk, S))
    return out


def implicit_dd_operator_refined(A, rpieces):
    """Implicit whitened operator through R^{-1} then S^{-1}, dd throughout
    the ill-conditioned part. fp64 in/out."""
    n, d = A.shape

    def mv(v):
        acc_h = np.zeros(n); acc_l = np.zeros(n)
        for C, piv, R, nk, S in rpieces:
            if nk == 0:
                continue
            t = sla.solve_triangular(S, v[C[:nk]], lower=False)  # kappa~1
            xh, xl = dd_trisolve_upper(R[:, :nk], t)
            h, l = dd_mat_ddvec(A[:, C[piv[:nk]]], (xh, xl))
            s, e = dd.two_sum(acc_h, h)
            acc_l = acc_l + e + l
            acc_h, acc_l = dd.quick_two_sum(s, acc_l)
        return acc_h + acc_l

    def rmv(u):
        out = np.zeros(d)
        ud = dd.dd(u)
        for C, piv, R, nk, S in rpieces:
            if nk == 0:
                continue
            h, l = dd_rmat_ddvec(A[:, C[piv[:nk]]], ud)
            xh, xl = dd_trisolve_lower(R[:, :nk].T, h, l)
            out[C[:nk]] = sla.solve_triangular(S.T, xh + xl, lower=True)
        return out

    return LinearOperator((n, d), matvec=mv, rmatvec=rmv, dtype=float)


def unwhiten_refined(rpieces, z, d):
    w = np.zeros(d)
    for C, piv, R, nk, S in rpieces:
        if nk == 0:
            continue
        t = sla.solve_triangular(S, z[C[:nk]], lower=False)
        cvec = sla.solve_triangular(R[:, :nk], t, lower=False)
        out = np.zeros(len(C))
        out[piv[:nk]] = cvec
        w[C] = out
    return w


def implicit_dd_operator(A, pieces):
    """E2: the whitened operator with R^{-1} applied per matvec in dd.

    matvec  Bv = sum_C A[:,C[piv[:nk]]] @ (R^{-1} v_C)   (trisolve + matvec in dd)
    rmatvec B^T u = per block R^{-T} (A[:,C[piv]]^T u)   (rmatvec + trisolve in dd)

    fp64 in, fp64 out; every intermediate that suffers kappa(R)
    amplification is carried in dd, so the rounded result is accurate to
    ~eps64 relative -- the same contract a materialized B provides."""
    n, d = A.shape

    def mv(v):
        acc_h = np.zeros(n)
        acc_l = np.zeros(n)
        for C, piv, R, nk in pieces:
            if nk == 0:
                continue
            xh, xl = dd_trisolve_upper(R[:, :nk], v[C[:nk]])
            h, l = dd_mat_ddvec(A[:, C[piv[:nk]]], (xh, xl))
            s, e = dd.two_sum(acc_h, h)
            acc_l = acc_l + e + l
            acc_h, acc_l = dd.quick_two_sum(s, acc_l)
        return acc_h + acc_l

    def rmv(u):
        out = np.zeros(d)
        ud = dd.dd(u)
        for C, piv, R, nk in pieces:
            if nk == 0:
                continue
            h, l = dd_rmat_ddvec(A[:, C[piv[:nk]]], ud)
            xh, xl = dd_trisolve_lower(R[:, :nk].T, h, l)
            out[C[:nk]] = xh + xl
        return out

    return LinearOperator((n, d), matvec=mv, rmatvec=rmv, dtype=float)


def solve_ir(A, y, k, order=None, rows=None, inner=100, max_rounds=12,
             w0=None, operator=None, pieces=None, plateau=0.7, guard=1.05,
             record=None):
    """E4: refinement rounds around the whitened LSMR solve, with an
    OBSERVABLE stopping rule. No oracle anywhere in the control path.

    Per round: r = y - A w (fresh, fp64); solve min ||B z - r|| with LSMR
    (inner iterations); w <- w + unwhiten(z).
    Observables: t_j = ||y - A w||/||y||, g_j = ||A^T r||, nw_j = ||w||.
    Stop when g fails to improve by `plateau` twice in a row, or the guard
    fires (norm grows while the residual doesn't improve -- the null-drift
    signature). Returns w at the best OBSERVABLE round.

    `record(w, meta)` is called per round for oracle reporting only.
    """
    n, d = A.shape
    if pieces is None:
        pieces, B = whiten(A, k, order=order, rows=rows)
        op = B
    else:
        op = operator
    w = np.zeros(d) if w0 is None else w0.copy()
    ynorm = np.linalg.norm(y)
    hist = []
    best = None
    bad = 0
    passes = k if pieces is not None else 0
    for j in range(max_rounds):
        r = y - A @ w
        g = np.linalg.norm(A.T @ r)
        t = np.linalg.norm(r) / ynorm
        nw = np.linalg.norm(w)
        passes += 2
        if best is None or g < best[0]:
            best = (g, w.copy(), j)
        hist.append({"round": j, "train": t, "grad": g, "wnorm": nw,
                     "passes": passes})
        if record is not None:
            record(w, hist[-1])
        if j > 0:
            g_prev = hist[-2]["grad"]
            t_prev = hist[-2]["train"]
            nw_prev = max(hist[-2]["wnorm"], 1e-300)
            if nw > guard * nw_prev and t > 0.9 * t_prev:
                hist[-1]["stop"] = "guard"
                break
            bad = bad + 1 if g > plateau * g_prev else 0
            if bad >= 2:
                hist[-1]["stop"] = "plateau"
                break
        res = lsmr(op, r, atol=0.0, btol=0.0, conlim=0, maxiter=inner)
        w = w + unwhiten(pieces, res[0], d)
        passes += int(res[2])
    return best[1], hist


# ------------------------------------------------------- other solvers -----

def lsmr_bjac(A, y, k, iters, order=None, w0=None):
    """Tier-1 reference: LSMR + block-Jacobi M^{-1/2}, blocks by `order`."""
    n, d = A.shape
    order = np.arange(d) if order is None else np.asarray(order)
    fac = []
    for t in range(0, d, k):
        C = order[t:t + k]
        _, sv, Vt = np.linalg.svd(A[:, C], full_matrices=False)
        kp = sv > 1e-15 * max(sv[0], 1e-300)
        fac.append((C, Vt.T, np.where(kp, 1.0 / np.where(kp, sv, 1.0), 0.0)))

    def Mh(v):
        o = np.zeros_like(v)
        for C, Q, ih in fac:
            o[C] = Q @ (ih * (Q.T @ v[C]))
        return o

    B = LinearOperator((n, d), matvec=lambda v: A @ Mh(v),
                       rmatvec=lambda u: Mh(A.T @ u), dtype=float)
    x0 = None if w0 is None else w0            # solve in preconditioned coords
    res = lsmr(B, y, atol=0.0, btol=0.0, conlim=0, maxiter=iters,
               x0=None)                        # cold; warm handled via residual
    if w0 is not None:
        res = lsmr(B, y - A @ w0, atol=0.0, btol=0.0, conlim=0, maxiter=iters)
        return w0 + Mh(res[0])
    return Mh(res[0])


def eval_err(prob, w):
    return float(np.linalg.norm(prob["A_ev"] @ w - prob["y_ev"]) / prob["ye_norm"])


def geomean(v):
    v = np.asarray(v, dtype=float)
    return float(np.exp(np.log(np.maximum(v, 1e-300)).mean()))
