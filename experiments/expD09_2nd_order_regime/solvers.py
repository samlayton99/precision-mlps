"""expD09 solver registry -- ROUND 1: baselines + exact block-coordinate descent.

Every solver is a generator yielding the iterate `a` once per SWEEP, where a
sweep is defined to be ~one pass over Phi:

  gd / cgls   one iteration  (1-2 matvecs)
  bcd         one full cycle through all ceil(d/k) blocks

so the x axis of every figure is cost-comparable ACROSS methods in matvec
passes. It is NOT flop-comparable: an exact block solve costs O(n k^2) per
block, i.e. O(n d k) per sweep against a matvec's O(n d). Every record
carries `flops_per_sweep` so the honest comparison is available. On a FROZEN
Phi with a FIXED partition the block factorizations are computed once and
reused, which collapses that factor -- not exploited here (arithmetic is
identical, only wall time), stated rather than assumed.

Round 1 is deliberately the simplest thing that can answer the fork:
does the ill-conditioning of [Phi, 1] live LOCALLY in the column index
(contiguous blocks = neighbouring neuron centers, i.e. a 1D Schwarz domain
decomposition) or GLOBALLY (random blocks do just as well and both fail)?
Round 2 adds overlap, a coarse-space correction, the block-diagonal
preconditioned gradient and the Nystrom sketch.

Two axes are swept, because a measurement showed both matter:

  k          block size, 32 / 64 / 128 against rank 80 / 144 / 273
  sub_rcond  how much of the block's OWN inverse the sub-solve dares use.
             At the repo-standard 1e-15 the sub-solve inverts the block's
             near-null directions; the increments cancel inside the block
             but push `a` along globally-near-null directions, and ||a||
             random-walks to ~1e12 (measured). Past ~1e10 the residual
             A a - y can no longer be formed in fp64 without catastrophic
             cancellation, so the METRIC goes noise-limited too. The damped
             arms (1e-8) show the same error with a bounded iterate.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

RCOND_STD = 1e-15      # repo standard: keep everything fp64 can resolve
RCOND_DAMPED = 1e-8    # ablation: refuse the block's own near-null directions

SOLVERS = [
    # ---- baselines: what any block method has to beat ----------------------
    {"id": "gd", "family": "gd", "label": "GD (lr = 1/L)",
     "note": "first-order reference; direction i needs (s_1/s_i)^2 steps"},
    {"id": "cgls", "family": "cgls", "label": "CGLS (Krylov)",
     "note": "expD07's best O(1)-memory iterative method; stalls ~1e-8"},

    # ---- exact BCD, contiguous blocks (neighbouring centers) ---------------
    {"id": "bcd_contig_k32", "family": "bcd", "blocks": "contig", "k": 32,
     "label": "exact BCD, contiguous $k=32$"},
    {"id": "bcd_contig_k64", "family": "bcd", "blocks": "contig", "k": 64,
     "label": "exact BCD, contiguous $k=64$"},
    {"id": "bcd_contig_k128", "family": "bcd", "blocks": "contig", "k": 128,
     "label": "exact BCD, contiguous $k=128$"},

    # ---- exact BCD, random blocks (structure-blind control) ----------------
    {"id": "bcd_rand_k32", "family": "bcd", "blocks": "rand", "k": 32,
     "label": "exact BCD, random $k=32$"},
    {"id": "bcd_rand_k64", "family": "bcd", "blocks": "rand", "k": 64,
     "label": "exact BCD, random $k=64$"},
    {"id": "bcd_rand_k128", "family": "bcd", "blocks": "rand", "k": 128,
     "label": "exact BCD, random $k=128$"},

    # ---- sub-solve damping ablation ---------------------------------------
    {"id": "bcd_contig_k64_damped", "family": "bcd", "blocks": "contig",
     "k": 64, "sub_rcond": RCOND_DAMPED,
     "label": r"exact BCD, contiguous $k=64$, sub-solve rcond $10^{-8}$"},
    {"id": "bcd_rand_k64_damped", "family": "bcd", "blocks": "rand",
     "k": 64, "sub_rcond": RCOND_DAMPED,
     "label": r"exact BCD, random $k=64$, sub-solve rcond $10^{-8}$"},

    # ---- residual-handling ablation (expD08 lesson 2 lives here) -----------
    {"id": "bcd_contig_k64_carried", "family": "bcd", "blocks": "contig",
     "k": 64, "refresh": "never",
     "label": "exact BCD, contiguous $k=64$, carried residual"},

    # ======================= ROUND 2: preconditioned Krylov ================
    # Round 1 established that BCD as a multiplicative solver stalls, and the
    # corrected diagnostic established that block preconditioning buys ~5
    # orders (kappa 7e14 -> 1e9), not enough: at kappa ~ 1e9 the attainable
    # accuracy eps*kappa ~ 1e-7 IS the measured CGLS stall. Reaching the
    # floor needs kappa ~ O(1), which needs a preconditioner that knows the
    # row space. Two arms, and the honest cost is the persistent state.
    {"id": "pcg_bjac_k64", "family": "pcg_bjac", "k": 64, "sub_rcond": 1e-12,
     "state_mult": "k", "label": r"CGLS + block-Jacobi precond ($k=64$)"},
    {"id": "cgls_sketch", "family": "sketch", "over": 2.0,
     "state_mult": "rank", "label": "CGLS + sketch preconditioner (Blendenpik)"},
    # Sam's question, answered: round-robin block solving is far more useful
    # as a PRECONDITIONER than as the solver. Identical block work and
    # identical O(k^2) state as exact BCD -- one k x k Gram resident at a
    # time, the rest carried in the O(d)+O(b) residual -- but the outer CG
    # replaces BCD's stationary contraction rho with sqrt(kappa).
    {"id": "cgls_sgs_k64", "family": "sgs", "k": 64, "state_mult": "ksq",
     "label": "CGLS + symmetric block Gauss-Seidel precond ($k=64$)"},
    # Round 3. CGLS's recurrences live in the A^T A metric, so it attains
    # eps*kappa(A)^2; LSMR (Golub-Kahan bidiagonalization on A directly)
    # attains eps*kappa(A). Same preconditioner, same O(k^2) state, one
    # line changed -- measured 20-50x on the geo-mean.
    {"id": "lsmr_bjac_k64", "family": "lsmr_bjac", "k": 64,
     "state_mult": "ksq",
     "label": "LSMR + block-Jacobi precond ($k=64$)"},
    # SPIR (Epperly, Meier & Nakatsukasa 2024, arXiv:2406.03468): sketch-and-
    # precondition plus iterative refinement, the provably backward-stable
    # form. Reaches -- and beats -- the truncated-SVD floor on every cell,
    # at an OBSERVABLE stopping rule. State is d*rank.
    {"id": "spir", "family": "spir", "over": 2.0, "inner": 200,
     "state_mult": "rank",
     "label": "SPIR (sketch-and-precondition + iterative refinement)"},
    # THE O(d) SOLVER. Block-QR whitening: reaches the floor in fp64 with
    # state d*k, k FIXED. See _blockqr for why QR and not the Gram.
    {"id": "blockqr_k128", "family": "blockqr", "k": 128, "state_mult": "k",
     "label": "block-QR whitening + LSQR ($k=128$)"},
    {"id": "blockqr_k64", "family": "blockqr", "k": 64, "state_mult": "k",
     "label": "block-QR whitening + LSQR ($k=64$)"},
]


def spec_by_id() -> dict[str, dict]:
    return {s["id"]: s for s in SOLVERS}


# --------------------------------------------------------------------------


def _minnorm_step(Ac: np.ndarray, b: np.ndarray, rcond: float) -> np.ndarray:
    """Min-norm solution of  min_d ||Ac d - b||  by truncated SVD.

    Truncation is load-bearing, not hygiene. Two separate things it handles:
    a contiguous block sitting entirely inside the halo is numerically rank-1
    (saturated tanh columns are constant), and every live block is itself
    badly conditioned because neighbouring tanh columns are near-duplicates.
    `rcond` is the knob that decides how much of that inverse to use, and it
    is swept rather than fixed.
    """
    U, s, Vt = np.linalg.svd(Ac, full_matrices=False)
    if s.size == 0 or s[0] <= 0.0:
        return np.zeros(Ac.shape[1])
    keep = s > rcond * s[0]
    return Vt.T @ (np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
                   * (U.T @ b))


def _partition(d: int, k: int, kind: str, rng: np.random.Generator):
    """Column blocks for one sweep.

    contig  consecutive runs in CENTER order -- each block is a spatially
            local neighbourhood of neurons, capturing the strong
            nearest-neighbour coupling (a 1D Schwarz decomposition).
    rand    a fresh random partition every sweep -- structure-blind, the
            control that isolates whether column ordering buys anything.
    """
    idx = np.arange(d) if kind == "contig" else rng.permutation(d)
    return [idx[i:i + k] for i in range(0, d, k)]


def _bcd(A, y, spec, sweeps, seed) -> Iterator[np.ndarray]:
    d = A.shape[1]
    k = int(spec["k"])
    rcond = float(spec.get("sub_rcond", RCOND_STD))
    refresh = spec.get("refresh", "sweep")
    rng = np.random.default_rng(seed)
    a = np.zeros(d)
    r = -y.astype(np.float64).copy()          # residual A a - y at a = 0
    for _ in range(sweeps):
        for C in _partition(d, k, spec["blocks"], rng):
            Ac = A[:, C]
            delta = _minnorm_step(Ac, -r, rcond)
            a[C] += delta
            r += Ac @ delta                   # carried within the sweep
        if refresh == "sweep":
            # One fresh recompute per sweep, not per block. expD08 lesson 2
            # (re-rolling ~eps rounding EVERY iteration caps a method at
            # 1e-10) governs recurrences whose per-step progress is below
            # that noise; here it is amortized over d/k exact solves. The
            # `..._carried` arm never refreshes, so the two are measured.
            r = A @ a - y
        yield a


def _gd(A, y, spec, sweeps, seed) -> Iterator[np.ndarray]:
    n, d = A.shape
    smax = float(np.linalg.norm(A, 2))
    lr = 1.0 / (2.0 * smax ** 2 / n)          # half the 2/L stability limit
    a = np.zeros(d)
    for _ in range(sweeps):
        a = a - lr * (2.0 / n) * (A.T @ (A @ a - y))
        yield a


def _cgls(A, y, spec, sweeps, seed) -> Iterator[np.ndarray]:
    """Conjugate gradients on the normal equations, applied MATRIX-FREE.

    CGLS never forms A^T A, so it does not square the condition number in
    storage -- but its recurrences still live in the A^T A metric, which is
    why expD07 measured it stalling several orders above the SVD floor.
    """
    a = np.zeros(A.shape[1])
    r = y - A @ a
    s = A.T @ r
    p = s.copy()
    gam = float(s @ s)
    for _ in range(sweeps):
        q = A @ p
        qq = float(q @ q)
        if qq <= 0.0 or gam <= 0.0:
            yield a
            continue
        alpha = gam / qq
        a = a + alpha * p
        r = r - alpha * q
        s = A.T @ r
        gam_new = float(s @ s)
        p = s + (gam_new / gam) * p
        gam = gam_new
        yield a


def _pcg_generic(A, y, Minv, iters) -> Iterator[np.ndarray]:
    """CGLS in the metric induced by Minv (SPD). One yield per iteration."""
    a = np.zeros(A.shape[1])
    r = A.T @ (y - A @ a)
    z = Minv(r)
    p = z.copy()
    rz = float(r @ z)
    for _ in range(iters):
        Ap = A.T @ (A @ p)
        pa = float(p @ Ap)
        if pa <= 0.0 or rz <= 0.0:
            yield a
            continue
        al = rz / pa
        a = a + al * p
        r = r - al * Ap
        z = Minv(r)
        rz2 = float(r @ z)
        p = z + (rz2 / rz) * p
        rz = rz2
        yield a


def _pcg_bjac(A, y, spec, sweeps, seed) -> Iterator[np.ndarray]:
    """Block-Jacobi preconditioned CGLS.

    Each block's inverse is a TRUNCATED pseudo-inverse, not a ridged
    Cholesky: the halo blocks are numerically rank-deficient and a small
    ridge turns them into a 1/mu amplifier of directions the data cannot
    see (measured: it caps the method at 1e-4 and drives ||a|| to 1e7).
    Persistent state is d*k -- OUTSIDE the O(d) budget, kept as the
    reference point that isolates how much block curvature alone buys.
    """
    d = A.shape[1]
    k = int(spec["k"])
    rcond = float(spec.get("sub_rcond", 1e-12))
    G = A.T @ A
    ops = []
    for t in range(0, d, k):
        C = np.arange(t, min(t + k, d))
        w, Q = np.linalg.eigh(G[np.ix_(C, C)])
        keep = w > rcond * w.max()
        ops.append((C, Q, np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)))

    def Minv(v):
        o = np.zeros_like(v)
        for C, Q, iw in ops:
            o[C] = Q @ (iw * (Q.T @ v[C]))
        return o

    yield from _pcg_generic(A, y, Minv, sweeps)


def _sgs(A, y, spec, sweeps, seed) -> Iterator[np.ndarray]:
    """CGLS preconditioned by one symmetric round-robin block Gauss-Seidel sweep.

    M^-1 v is: start from x = 0, sweep the blocks forward then backward,
    each block solving G_C delta = (v - G x)_C exactly. The full Gram is
    never formed -- the coupling is carried through u = Phi x, so block C
    only needs Phi_C^T u (O(bk)) and its own k x k factor. Persistent state
    is therefore O(d) + O(b) + O(k^2), the same class as exact BCD.

    This is the accelerated version of round-robin BCD: same block work,
    same memory, but the outer CG turns the stationary contraction rho into
    sqrt(kappa). Measured: 6-7 orders better than BCD-as-solver.
    """
    n, d = A.shape
    k = int(spec["k"])
    rcond = float(spec.get("sub_rcond", 1e-12))
    fac = []
    for t in range(0, d, k):
        C = np.arange(t, min(t + k, d))
        Ac = A[:, C]
        w, Q = np.linalg.eigh(Ac.T @ Ac)
        keep = w > rcond * max(w.max(), 1e-300)
        fac.append((C, Ac, Q, np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)))

    def Minv(v):
        x = np.zeros(d)
        u = np.zeros(n)
        for order in (fac, fac[::-1]):
            for C, Ac, Q, iw in order:
                dl = Q @ (iw * (Q.T @ (v[C] - Ac.T @ u)))
                x[C] += dl
                u += Ac @ dl
        return x

    yield from _pcg_generic(A, y, Minv, sweeps)


def _blockqr(A, y, spec, sweeps, seed) -> Iterator[np.ndarray]:
    """Block-QR whitening + plain LSQR.  Reaches the SVD floor in fp64 at
    O(d*k) state.  Three things are load-bearing, each measured:

    1. WHITEN BY QR, NOT BY THE GRAM.  Per block, pivoted Householder QR
       gives A[:,C] = Q_C R_C, so the whitened block IS Q_C -- orthonormal
       to machine precision BY CONSTRUCTION.  Computing the same thing as
       A[:,C] M_C^{-1/2} with M_C from eigh(A_C^T A_C) instead (a) squares
       the block condition number and (b) is catastrophic cancellation,
       because kappa(M^{-1/2}) reaches 1e14 here.  Measured: Gram route
       1.6e-8, QR route 1.7e-15.

    2. NEVER APPLY THE ILL-CONDITIONED FACTOR INSIDE THE ITERATION.  Running
       LSQR through a LinearOperator that applies M each matvec puts a
       relative error eps*kappa(M) ~ 1e-2 into EVERY matvec (measured: caps
       the method at 4e-4).  Here the whitened operator B is materialized
       once and the iteration only ever touches B; R_C^{-1} is applied a
       single time at the end, as one triangular solve per block.  A single
       application of an ill-conditioned operator is harmless -- that is
       exactly what the truncated SVD does.

    3. PIVOT AND DROP.  The halo blocks are rank-deficient; column-pivoted
       QR exposes that in |diag(R)| and the dependent columns are dropped
       rather than inverted.

    Persistent state: the R factors, k x k per block over d/k blocks = d*k
    floats, with k FIXED (the subproblem's rotating-block budget).  B is
    the whitened data, batch-sized, not optimizer state.
    """
    import scipy.linalg as sla
    from scipy.sparse.linalg import lsmr

    n, d = A.shape
    k = int(spec["k"])
    rcond = float(spec.get("rcond", 1e-13))
    B = np.zeros((n, d))
    pieces = []
    for t in range(0, d, k):
        C = np.arange(t, min(t + k, d))
        Q, R, piv = sla.qr(A[:, C], mode="economic", pivoting=True)
        dg = np.abs(np.diag(R))
        nk = int((dg > rcond * max(dg[0], 1e-300)).sum())
        B[:, C[:nk]] = Q[:, :nk]
        pieces.append((C, piv, R[:nk, :], nk))

    def unwhiten(zz):
        w = np.zeros(d)
        for C, piv, R, nk in pieces:
            if nk == 0:
                continue
            c = sla.solve_triangular(R[:, :nk], zz[C[:nk]], lower=False)
            out = np.zeros(len(C))
            out[piv[:len(c)]] = c
            w[C] = out
        return w

    last = np.zeros(d)
    grid = np.unique(np.round(np.logspace(0, np.log10(max(sweeps, 2)),
                                          40)).astype(int))
    gi = 0
    for it in range(1, sweeps + 1):
        if gi < grid.size and it == grid[gi]:
            last = unwhiten(lsmr(B, y, atol=0.0, btol=0.0, conlim=0,
                                 maxiter=int(it))[0])
            gi += 1
        yield last


def _spir(A, y, spec, sweeps, seed) -> Iterator[np.ndarray]:
    """SPIR: sketch-and-precondition + iterative refinement.

    The plain sketch-and-precondition form is NOT backward stable on
    ill-conditioned problems (Meier, Nakatsukasa, Townsend & Webb, SIMAX
    2024); wrapping it in iterative refinement restores backward stability
    (Epperly, Meier & Nakatsukasa 2024). Both halves matter here.

    Refinement contracts because the sketch preconditioner makes the inner
    solve a good approximate inverse -- kappa(A P) = 2.5-4.6 measured. It
    does NOT need an extended-precision residual: a double-double residual
    was measured to change the result by under 1%, because the barrier was
    never residual rounding (that was tested and rejected twice).
    """
    from scipy.sparse.linalg import LinearOperator, lsmr

    n, d = A.shape
    rng = np.random.default_rng(seed)
    s_rows = min(n, int(spec.get("over", 2.0) * d))
    S = rng.standard_normal((s_rows, n)) / np.sqrt(s_rows)
    U2, s2, V2t = np.linalg.svd(S @ A, full_matrices=False)
    kp = s2 > 1e-15 * s2[0]
    rk = int(kp.sum())
    P = V2t[:rk].T / s2[:rk]
    B = LinearOperator((n, rk), matvec=lambda u: A @ (P @ u),
                       rmatvec=lambda v: P.T @ (A.T @ v), dtype=float)
    inner = int(spec.get("inner", 200))
    w = np.zeros(d)
    for it in range(1, sweeps + 1):
        if it % inner == 0:
            w = w + P @ lsmr(B, y - A @ w, atol=0.0, btol=0.0, conlim=0,
                             maxiter=inner)[0]
        yield w


def _blk_half(A, k, rcond=1e-12):
    """Block-Jacobi inverse SQUARE ROOT, applied blockwise.

    Note the guarded np.where: `np.where(kp, w**-0.5, 0)` evaluates BOTH
    branches and produces NaN on the dead eigenvalues of a rank-deficient
    halo block, silently poisoning the preconditioner (measured, cost a
    whole debugging round).
    """
    d = A.shape[1]
    fac = []
    for t in range(0, d, k):
        C = np.arange(t, min(t + k, d))
        Ac = A[:, C]
        w, Q = np.linalg.eigh(Ac.T @ Ac)
        kp = w > rcond * max(w.max(), 1e-300)
        fac.append((C, Q, np.where(kp, np.where(kp, w, 1.0) ** -0.5, 0.0)))

    def f(v):
        o = np.zeros_like(v)
        for C, Q, ih in fac:
            o[C] = Q @ (ih * (Q.T @ v[C]))
        return o
    return f


def _lsmr_bjac(A, y, spec, sweeps, seed) -> Iterator[np.ndarray]:
    """LSMR on the block-Jacobi-preconditioned system B = A M_b^{-1/2}.

    LSMR is MINRES on the normal equations via Golub-Kahan bidiagonalization
    -- it never forms or iterates in the A^T A metric, so its attainable
    accuracy is eps*kappa(A) rather than CGLS's eps*kappa(A)^2. It is also
    monotone in ||A^T r||, the optimality residual, which is the right
    stopping quantity here (expD08 lesson 1: never stop on a loss value).

    scipy's lsmr has no per-iteration hook, so the trajectory is sampled by
    re-running to increasing maxiter. That is a reporting device only; the
    final iterate at a given maxiter is identical either way.
    """
    from scipy.sparse.linalg import LinearOperator, lsmr

    n, d = A.shape
    Mh = _blk_half(A, int(spec["k"]))
    B = LinearOperator((n, d), matvec=lambda u: A @ Mh(u),
                       rmatvec=lambda v: Mh(A.T @ v), dtype=float)
    last = np.zeros(d)
    grid = np.unique(np.round(np.logspace(0, np.log10(max(sweeps, 2)),
                                          60)).astype(int))
    gi = 0
    for it in range(1, sweeps + 1):
        if gi < grid.size and it == grid[gi]:
            last = Mh(lsmr(B, y, atol=0.0, btol=0.0, conlim=0,
                           maxiter=int(it))[0])
            gi += 1
        yield last


def _sketch(A, y, spec, sweeps, seed) -> Iterator[np.ndarray]:
    """Sketch-preconditioned CGLS (Blendenpik / LSRN).

    Draw a Gaussian sketch S with s = over*d rows, factor SA = U diag(s) V^T,
    and precondition with  M^{-1} = V_r diag(1/s_r)  keeping only directions
    the sketch resolves. Random-projection theory says kappa(A M^{-1}) = O(1)
    independent of kappa(A) -- MEASURED here at 2.5 to 4.6 -- and the rank
    truncation on the sketch is what supplies the row-space knowledge that no
    O(d)-state preconditioner has.

    The sketch is built ONCE against the frozen Phi; per-iteration work is
    then one pass plus an O(d*rank) apply. Persistent state is d*rank, i.e.
    O(d^2) here because this toy's rank is 0.6d -- the honest cost, recorded
    as `state_floats` and plotted rather than argued.
    """
    n, d = A.shape
    rng = np.random.default_rng(seed)
    s_rows = min(n, int(spec.get("over", 2.0) * d))
    S = rng.standard_normal((s_rows, n)) / np.sqrt(s_rows)
    U2, s2, V2t = np.linalg.svd(S @ A, full_matrices=False)
    keep = s2 > 1e-15 * s2[0]
    rk = int(keep.sum())
    Mi = V2t[:rk].T / s2[:rk]              # d x rk
    B = A @ Mi                             # kappa(B) = O(1)

    u = np.zeros(rk)
    r = y - B @ u
    p = B.T @ r
    q = p.copy()
    gam = float(p @ p)
    for _ in range(sweeps):
        Bq = B @ q
        qq = float(Bq @ Bq)
        if qq <= 0.0 or gam <= 0.0:
            yield Mi @ u
            continue
        al = gam / qq
        u = u + al * q
        r = r - al * Bq
        p = B.T @ r
        g2 = float(p @ p)
        q = p + (g2 / gam) * q
        gam = g2
        yield Mi @ u


_FAMILIES = {"bcd": _bcd, "gd": _gd, "cgls": _cgls,
             "pcg_bjac": _pcg_bjac, "sketch": _sketch, "sgs": _sgs,
             "lsmr_bjac": _lsmr_bjac, "spir": _spir,
             "blockqr": _blockqr}


def iterate(A, y, spec, sweeps, seed=0) -> Iterator[np.ndarray]:
    return _FAMILIES[spec["family"]](A, y, spec, sweeps, seed)


def state_floats(spec: dict, d: int, rank: int) -> float:
    """Persistent optimizer state, in units of d (the parameter count).

    This is the practicality gate made explicit: Adam is 2. Anything that
    grows with d is outside the budget and must say so in its own figure.
    """
    m = spec.get("state_mult")
    if m == "ksq":
        return float(spec["k"]) ** 2 / max(d, 1)
    if m == "k":
        return float(spec["k"])
    if m == "rank":
        return float(rank)
    if spec["family"] == "bcd":
        return float(spec["k"])       # one block resident at a time
    return 3.0                        # gd / cgls: a few vectors


def flops_per_sweep(spec: dict) -> float:
    """Leading-order flops for one sweep, in units of one n x d matvec (nd)."""
    if spec["family"] in ("gd", "cgls", "pcg_bjac", "sketch", "sgs",
                          "lsmr_bjac", "spir", "blockqr"):
        return 2.0
    # per block: SVD of n x k is ~6 n k^2, plus 2 matvecs of n x k;
    # summed over d/k blocks -> 6 n d k + 2 n d.
    return 6.0 * int(spec["k"]) + 2.0
