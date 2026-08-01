"""expD11 -- batching hardening for the two step-2 solvers.

NO ADAM ANYWHERE. Geometry drift, where it appears, is a synthetic schedule.

Type A  streaming: LSMR/heavy-ball + k-constant block-diagonal preconditioner,
        factors maintained from batches (EMA Gram, or rotating pivoted QR).
        Sees one batch per step, any b, state O(d*k) + O(d).
Type B  finale: SPIR (sketch preconditioner + LSMR + iterative refinement) on
        rows ACCUMULATED from the same batch stream (frozen Phi), so the
        4d-row requirement is satisfied by the buffer, never by the user's b.

Two reference floors are computed for every problem, and they are the whole
point of the batching story:
  floor_pool     truncated SVD on the entire pool -- the target
  floor_1batch(b) truncated SVD on ONE batch of size b -- what a method that
                 does NOT accumulate information across batches can do
A solver earns its keep by sitting near floor_pool while b shrinks, i.e. by
beating floor_1batch(b) by orders. That is the "soft win"; the bars are the
"strong win".

Cost is reported in PASSES-EQUIVALENT (1 pass = one O(b*d) traversal of the
current batch), so no accuracy claim is separable from its compute price.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, lsmr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
D09 = REPO / "experiments" / "expD09_2nd_order_regime"
E01 = REPO / "experiments" / "expE01_geometry_zoo_2d"
for p in (str(REPO), str(D09), str(D09 / "validation"), str(E01)):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS = REPO / "results" / "checkpoint_D_optimizers" / "expD11_batching"
FIGS = RESULTS / "figures"
RCOND = 1e-15


# ============================================================ problems =====

def _finish(A, y, A_ev, y_ev, meta):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > RCOND * s[0]
    r = int(keep.sum())
    a = (Vt.T * np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)) @ (U.T @ y)
    ye = float(np.linalg.norm(y_ev))
    out = dict(A=A, y=y, A_ev=A_ev, y_ev=y_ev, ye_norm=ye,
               d=int(A.shape[1]), n=int(A.shape[0]), rank=r,
               kappa=float(s[0] / s[keep][-1]),
               floor_pool=float(np.linalg.norm(A_ev @ a - y_ev) / ye),
               a_pool=a, **meta)
    return out


def prob_qi1d(N, row_mult=8, noise_rel=0.0, seed=0):
    """1-D QI system (expD09 geometry): tanh ridges on a uniform grid+halo."""
    import common_val as cv
    p = cv.make_problem_custom(N, row_mult, noise_rel, seed) if hasattr(
        cv, "make_problem_custom") else None
    if p is None:
        centers, gamma = cv.geometry(N)
        d = centers.size + 1
        n = row_mult * d
        x_tr = np.linspace(-1.0, 1.0, n)
        x_ev = np.linspace(-1.0, 1.0, 4001)
        from src.data.targets import get_target
        f = get_target("sine").fn_numpy
        y_c, y_ev = f(x_tr), f(x_ev)
        rng = np.random.default_rng(1000 * seed + N)
        y = y_c.copy()
        if noise_rel > 0:
            e = rng.standard_normal(n)
            e *= noise_rel * np.linalg.norm(y_c) / np.linalg.norm(e)
            y = y_c + e
        A = cv.build_phi_aug(x_tr, gamma, centers)
        A_ev = cv.build_phi_aug(x_ev, gamma, centers)
    return _finish(A, y, A_ev, y_ev,
                   dict(family="qi1d", label=f"QI-1D N={N}", noise_rel=noise_rel))


def prob_qi2d(N, geom="radon_tensor", lam=0.12, row_mult=8, noise_rel=0.0, seed=0):
    """2-D ridge system, expE01 (checkpoint E) conventions: radius 2.5 for
    Radon geometries, gauss_bump target, h_ref = 2.8/sqrt(N)."""
    from geometries import GEOMETRIES
    from src.data.sampling2d import disk_grid, disk_uniform
    from src.data.targets2d import get_target_2d
    radius = 2.5 if geom.startswith("radon") else 1.6
    dirs, offs = GEOMETRIES[geom](N, radius, seed=0)
    gamma = lam / (2.8 / np.sqrt(N))
    d = len(offs) + 1
    n = row_mult * d
    X_tr = disk_uniform(n, radius=1.0, seed=seed)
    X_ev = disk_grid(120, radius=1.0)
    f = get_target_2d("gauss_bump").fn_numpy
    y_c, y_ev = f(X_tr), f(X_ev)
    rng = np.random.default_rng(7000 + N)
    y = y_c.copy()
    if noise_rel > 0:
        e = rng.standard_normal(n)
        e *= noise_rel * np.linalg.norm(y_c) / np.linalg.norm(e)
        y = y_c + e
    ph = lambda P: np.hstack([np.tanh(gamma * (P @ dirs.T - offs[None, :])),
                              np.ones((len(P), 1))])
    return _finish(ph(X_tr), y, ph(X_ev), y_ev,
                   dict(family="qi2d", label=f"QI-2D {geom} N={N}",
                        noise_rel=noise_rel))


def prob_twin(d, kind="cluster", row_mult=4, n_ev=1500, noise_rel=0.0, seed=0):
    """Random matrix with the QI spectral fingerprint: exponential decay with
    no gap, rank cliff at 0.6d, target energy proportional to sigma_i.

    cluster    columns in 32-wide correlated groups, randomly permuted --
               structure exists but must be discovered
    unstruct   Haar singular vectors -- no column structure at all
    """
    rng = np.random.default_rng(9000 + seed * 31 + d)
    n = row_mult * d + n_ev
    if kind == "unstruct":
        r = int(0.6 * d)
        sig = np.logspace(0, -14, r)
        U = np.linalg.qr(rng.standard_normal((n, r)))[0]
        V = np.linalg.qr(rng.standard_normal((d, r)))[0]
        A = (U * sig) @ V.T
        y_all = U @ (sig * rng.standard_normal(r))
        del U, V
    else:
        kc = 32
        cols = []
        for _ in range(d // kc):
            X = rng.standard_normal((n, kc))
            W = (np.linalg.qr(rng.standard_normal((kc, kc)))[0]
                 * np.logspace(0, -12, kc)) @ np.linalg.qr(
                     rng.standard_normal((kc, kc)))[0]
            cols.append(X @ W)
        A = np.hstack(cols)
        del cols
        A = A[:, rng.permutation(A.shape[1])]
        Uf, sf, _ = np.linalg.svd(A, full_matrices=False)
        kp = sf > RCOND * sf[0]
        y_all = Uf[:, kp] @ (sf[kp] * rng.standard_normal(int(kp.sum())))
        del Uf
    y_all = y_all / np.linalg.norm(y_all) * np.sqrt(n)
    ntr = row_mult * d
    y_tr = y_all[:ntr].copy()
    if noise_rel > 0:
        e = rng.standard_normal(ntr)
        e *= noise_rel * np.linalg.norm(y_tr) / np.linalg.norm(e)
        y_tr = y_tr + e
    return _finish(A[:ntr], y_tr, A[ntr:], y_all[ntr:],
                   dict(family=f"twin_{kind}", label=f"twin-{kind} d={d}",
                        noise_rel=noise_rel))


def eval_err(prob, w):
    return float(np.linalg.norm(prob["A_ev"] @ w - prob["y_ev"]) / prob["ye_norm"])


def floor_1batch(prob, b, seed=0, reps=2):
    """Best achievable by a method that sees only ONE batch of b rows: the
    truncated-SVD (min-norm) solve on that batch, evaluated on the clean grid.
    This is the no-accumulation reference the streaming solvers must beat."""
    A, y = prob["A"], prob["y"]
    n = A.shape[0]
    out = []
    for rep in range(reps):
        rng = np.random.default_rng(500 + 17 * rep + seed)
        S = rng.choice(n, min(b, n), replace=False)
        U, s, Vt = np.linalg.svd(A[S], full_matrices=False)
        kp = s > RCOND * s[0]
        a = (Vt.T * np.where(kp, 1 / np.where(kp, s, 1), 0)) @ (U.T @ y[S])
        out.append(eval_err(prob, a))
    return float(np.exp(np.mean(np.log(np.maximum(out, 1e-300)))))


# ======================================================== blocking ========

def make_blocks(prob, k, sample_rows=None, seed=0):
    """Fixed column partition. Uses cluster_blocks on a row sample when the
    problem is small enough for it to be affordable (it is O(d^2)); contiguous
    otherwise. At Type A's accuracy ceiling ordering is a mild optimization."""
    d = prob["d"]
    if d <= 2200:
        rng = np.random.default_rng(seed)
        s = sample_rows or min(prob["n"], max(384, 2 * k))
        S = rng.choice(prob["n"], min(s, prob["n"]), replace=False)
        try:
            sys.path.insert(0, str(REPO / "experiments" / "expD10_step2_hardening"))
            import core10
            return core10.cluster_blocks(prob["A"][S], kmax=k)
        except Exception:
            pass
    return [np.arange(t, min(t + k, d)) for t in range(0, d, k)]


# ======================================================== Type A =========

class BlockPrec:
    """k-constant block-diagonal preconditioner maintained from batches.

    variant 'ema'   : EMA of the block Gram (any b, including b << k -- the EMA
                      accumulates rank a single batch cannot have). Damped.
    variant 'rotqr' : per-block row buffer refreshed by pivoted QR of the rows
                      themselves (never a Gram -- avoids the squaring loss).

    Refresh is ROTATING: one block per call, so per-step factor cost is
    independent of d. Gram rows are capped at gram_mult*k so the cost is
    independent of b as well.
    """

    def __init__(self, blocks, d, variant="ema", beta=0.99, damp_rel=1e-12,
                 gram_mult=4, buf_mult=3, seed=0):
        # damp_rel is a TRUNCATION level on block EIGENvalues relative to the
        # block's largest, matching expD09's _blk_half (rcond=1e-12). This is
        # load-bearing, not hygiene: measured on sine N=128, k=64, full batch,
        # best-over-trajectory --  1e-16: 5.9e-9 | 1e-14: 2.1e-10 |
        # 1e-12: 8.5e-11 | 1e-10: 9.7e-10 | 1e-8: 6.4e-8.  Inverting a block's
        # near-null directions amplifies rounding; zeroing them removes them
        # from the search space instead. Optimum is a plateau near 1e-12.
        self.blocks, self.d, self.variant = blocks, d, variant
        self.beta, self.damp_rel = beta, damp_rel
        self.gram_mult, self.buf_mult = gram_mult, buf_mult
        self.nb = len(blocks)
        self.G = [None] * self.nb
        self.buf = [None] * self.nb
        self.fac = [None] * self.nb          # (V, inv_eig, inv_sqrt_eig)
        self.seen = np.zeros(self.nb, dtype=int)
        self.ptr = 0
        self.rng = np.random.default_rng(seed)
        self.flops = 0
        self.nfac = 0

    def _refactor(self, j):
        C = self.blocks[j]
        kj = len(C)
        if self.variant == "ema":
            G = self.G[j]
            if G is None:
                return
            ev, V = np.linalg.eigh(G)
        else:
            Bf = self.buf[j]
            if Bf is None or Bf.shape[0] < 2:
                return
            _, sv, Vt = np.linalg.svd(Bf, full_matrices=False)
            V = Vt.T
            ev = sv ** 2 / max(Bf.shape[0], 1)
            if V.shape[1] < kj:                       # keep only measured dirs
                pass
        keep = ev > self.damp_rel * max(ev.max(), 1e-300)
        safe = np.where(keep, ev, 1.0)
        self.fac[j] = (V, np.where(keep, 1.0 / safe, 0.0),
                       np.where(keep, 1.0 / np.sqrt(safe), 0.0))
        self.flops += kj ** 3
        self.nfac += 1

    def update(self, A_batch):
        """Refresh ONE block from the current batch (rotating)."""
        j = self.ptr
        self.ptr = (self.ptr + 1) % self.nb
        C = self.blocks[j]
        kj = len(C)
        b = A_batch.shape[0]
        m = min(b, self.gram_mult * kj)
        rows = slice(None) if m == b else self.rng.choice(b, m, replace=False)
        Xc = A_batch[rows][:, C]
        if self.variant == "ema":
            Gn = (Xc.T @ Xc) / max(m, 1)
            self.flops += m * kj * kj
            if self.G[j] is None:
                self.G[j] = Gn
            else:
                bb = self.beta
                self.G[j] = bb * self.G[j] + (1 - bb) * Gn
        else:
            cap = self.buf_mult * kj
            self.buf[j] = Xc if self.buf[j] is None else np.vstack(
                [Xc, self.buf[j]])[:cap]
            self.flops += m * kj
        self.seen[j] += 1
        self._refactor(j)
        return j

    def ready(self):
        return any(f is not None for f in self.fac)

    def _apply(self, v, which):
        out = np.zeros_like(v)
        for j, C in enumerate(self.blocks):
            f = self.fac[j]
            if f is None:
                out[C] = v[C]                       # identity until warmed
                continue
            V, ie, ise = f
            sc = ie if which == "inv" else ise
            out[C] = V @ (sc * (V.T @ v[C]))
        return out

    def inv(self, v):
        return self._apply(v, "inv")

    def half(self, v):
        return self._apply(v, "half")


def type_a(prob, k, b, tau, n_eps, variant="ema", beta=0.99, damp_rel=1e-12,
           seed=0, w0=None, blocks=None, drift=None, prewarm=True,
           rec_every=1):
    """Type A -- EPISODIC streaming solver.

    Each episode: draw a fresh batch of b rows, refresh one block's factor
    from it (rotating), then run `tau` LSMR iterations on that batch's
    preconditioned system starting from the current w. Fresh residual every
    episode (never carried). State O(d*k) + O(d).

    `tau` is the solver's budget knob and it is LOAD-BEARING. Measured on a
    FIXED operator with identical total iterations (3000), restarting every
    tau iterations costs orders against running unrestarted:
        tau=2: 1.3e-3 | 8: 1.6e-4 | 32: 1.1e-6 | 128: 1.6e-8 | 3000: 8.5e-11
    A Krylov process needs runway; there is no useful tau~1 regime. Type A is
    therefore a mini-solve, not a per-step interleave.

    THE TAU CLAMP (load-bearing): tau is clamped to b//2. Without it, an
    episode with tau > rank(batch) drives LSMR to interpolate an
    underdetermined batch, which throws w into directions the batch cannot
    see and DESTROYS prior progress -- measured worse than doing nothing
    (b=d/16: 5.1e-1 unclamped vs floor_1batch 1.6e-1). With the clamp the
    same cell gives 1.5e-2, i.e. 10x BETTER than the single-batch floor, and
    degradation in b is monotone. Step damping and Polyak averaging were both
    measured to add nothing on top of the clamp.

    drift: optional callable(episode)->eta perturbing the operator (a
    synthetic stand-in for a moving geometry; NO Adam in this module).
    """
    A, y = prob["A"], prob["y"]
    n, d = A.shape
    b = int(min(max(b, 2), n))
    tau = int(max(2, min(tau, b // 2)))
    if blocks is None:
        blocks = make_blocks(prob, k, seed=seed)
    P = BlockPrec(blocks, d, variant=variant, beta=beta, damp_rel=damp_rel,
                  seed=seed)
    rng = np.random.default_rng(1234 + seed)
    w = np.zeros(d) if w0 is None else w0.copy()
    hist = []
    best = (np.inf, w.copy(), 0)
    core_flops = 0
    if prewarm:                       # one-time: give every block a factor
        for _ in range(len(blocks)):
            S0 = np.arange(n) if b >= n else rng.choice(n, b, replace=False)
            P.update(A[S0])
    Aeff = A
    for e in range(1, n_eps + 1):
        if drift is not None:
            eta = drift(e)
            Aeff = A if eta <= 0 else A * (
                1.0 + eta * rng.standard_normal(d)[None, :])
        S = np.arange(n) if b >= n else rng.choice(n, b, replace=False)
        Ab, yb = Aeff[S], y[S]
        P.update(Ab)
        r = yb - Ab @ w
        core_flops += Ab.size
        op = LinearOperator((len(S), d), matvec=lambda z: Ab @ P.half(z),
                            rmatvec=lambda u: P.half(Ab.T @ u), dtype=float)
        res = lsmr(op, r, atol=0, btol=0, conlim=0, maxiter=tau)
        w = w + P.half(res[0])
        core_flops += 2 * int(res[2]) * Ab.size
        if e % rec_every == 0 or e == n_eps:
            ev = eval_err(prob, w)
            if ev < best[0]:
                best = (ev, w.copy(), e)
            hist.append(dict(ep=e, eval=ev, wnorm=float(np.linalg.norm(w)),
                             iters=int(e * tau), rows_seen=int(e * b),
                             passes=float((core_flops + P.flops)
                                          / max(b * d, 1))))
    return best[1], dict(history=hist, best=best[0], best_ep=best[2],
                         passes=hist[-1]["passes"] if hist else 0.0,
                         tau_used=int(tau), total_iters=int(n_eps * tau),
                         prec_frac=float(P.flops / max(core_flops + P.flops, 1)))


# ======================================================== Type B =========

def type_b(prob, b, n_acc_mult=4.0, rounds=8, inner=200, over=2.0,
           refine_rows="buffer", w0=None, seed=0, drift_eta=0.0,
           noise_rcond=None):
    """Finale Type B: SPIR on rows ACCUMULATED from the batch stream.

    The 4d-row requirement is met by the buffer, so the user's b is free.
    refine_rows: 'buffer' (fixed buffer), 'fresh' (a new batch per round),
                 'grow' (buffer keeps growing during refinement).
    Guard: keep-best by the observable ||A^T r||; if never better than entry,
    return w0 unchanged (worst case is a no-op).
    """
    A, y = prob["A"], prob["y"]
    n, d = A.shape
    rng = np.random.default_rng(4321 + seed)
    Aeff = A if drift_eta == 0 else A * (
        1.0 + drift_eta * rng.standard_normal(d)[None, :])
    target = int(min(n, np.ceil(n_acc_mult * d)))
    idx = []
    steps = 0
    while len(idx) < target:
        S = rng.choice(n, min(b, n), replace=False)
        idx.extend(S.tolist())
        steps += 1
    idx = np.array(idx[:target])
    Ab, yb = Aeff[idx], y[idx]
    w = np.zeros(d) if w0 is None else w0.copy()
    obs0 = float(np.linalg.norm(Ab.T @ (yb - Ab @ w)))
    s_rows = min(len(idx), int(over * d))
    Sk = rng.standard_normal((s_rows, len(idx))) / np.sqrt(s_rows)
    U2, s2, V2t = np.linalg.svd(Sk @ Ab, full_matrices=False)
    cut = (noise_rcond or 1e2 * np.finfo(float).eps) * s2[0]
    rk = int((s2 > cut).sum())
    P = V2t[:rk].T / s2[:rk]
    kap = float(np.linalg.svd(Ab @ P, compute_uv=False)[[0, -1]].tolist()[0]
                / max(np.linalg.svd(Ab @ P, compute_uv=False)[-1], 1e-300))
    op = LinearOperator((len(idx), rk), matvec=lambda u: Ab @ (P @ u),
                        rmatvec=lambda vv: P.T @ (Ab.T @ vv), dtype=float)
    hist = []
    best = (obs0, w.copy(), 0)
    passes = float(len(idx) / max(b, 1))       # accumulation cost, in b-passes
    prev = None
    for j in range(1, rounds + 1):
        if refine_rows == "fresh":
            S = rng.choice(n, min(max(b, 2 * d), n), replace=False)
            Ar, yr = Aeff[S], y[S]
        elif refine_rows == "grow":
            extra = rng.choice(n, min(b, n), replace=False)
            idx = np.concatenate([idx, extra])
            Ab, yb = Aeff[idx], y[idx]
            op = LinearOperator((len(idx), rk), matvec=lambda u: Ab @ (P @ u),
                                rmatvec=lambda vv: P.T @ (Ab.T @ vv), dtype=float)
            Ar, yr = Ab, yb
        else:
            Ar, yr = Ab, yb
        r = yr - Ar @ w
        obs = float(np.linalg.norm(Ar.T @ r))
        opj = op if refine_rows != "fresh" else LinearOperator(
            (len(Ar), rk), matvec=lambda u: Ar @ (P @ u),
            rmatvec=lambda vv: P.T @ (Ar.T @ vv), dtype=float)
        res = lsmr(opj, r, atol=5e-16, btol=5e-16, conlim=0, maxiter=inner)
        w = w + P @ res[0]
        passes += 2 * int(res[2]) * len(Ar) / max(b, 1)
        obs_new = float(np.linalg.norm(Ar.T @ (yr - Ar @ w)))
        if obs_new < best[0]:
            best = (obs_new, w.copy(), j)
        hist.append(dict(round=j, obs=obs_new, eval=eval_err(prob, w),
                         iters=int(res[2]), passes=passes,
                         wnorm=float(np.linalg.norm(w))))
        if prev is not None and obs_new > 0.5 * prev:
            break
        prev = obs_new
    reverted = best[0] >= obs0
    w_out = (w0.copy() if (reverted and w0 is not None)
             else (np.zeros(d) if reverted else best[1]))
    return w_out, dict(history=hist, kappa_AP=kap, rank_sketch=rk,
                       n_acc=int(len(idx)), acc_steps=steps,
                       reverted=bool(reverted), passes=passes,
                       eval=eval_err(prob, w_out),
                       best_round=best[2])


# ========================================================= io ============

def append(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def geomean(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return float("nan")
    return float(np.exp(np.log(np.maximum(v, 1e-300)).mean()))
