"""expD12 -- the progressively-unregularized solve.

PROCEDURE UNDER TEST (fixed Phi; no Adam, no moving geometry)
  1. damping ladder mu = (alpha*sigma_1)^2 with alpha descending
  2. at each alpha, run LSQR at c=0 (NO stored state) warm-started from the
     previous level, until that level converges or a cap is hit
  3. when c=0 can no longer reach the next level, hand off: allocate O(r*d) and
     run one full-reorthogonalization solve from that starting point

Everything is parameterised by alpha = sqrt(mu)/sigma_1, the singular value the
damping sits at relative to the largest, because kappa_mu = 1/alpha exactly
(measured, phase 0).  That makes the ladder transferable across matrices.

Damping is applied IMPLICITLY through the operator, never by stacking a d x d
identity (which costs 238 MB at d=2060 and is rebuilt every stage).
Reorthogonalization is vectorised (CGS2): 12x faster than the per-vector loop
at d=2048, agreeing to 1.9e-15.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
E01 = REPO / "experiments" / "expE01_geometry_zoo_2d"
for p in (str(REPO), str(E01)):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS = REPO / "results" / "checkpoint_D_optimizers" / "expD12_mu_ladder"
FIGS = RESULTS / "figures"
RCOND = 1e-15


# ========================================================== problems =====

def _finish(A, y, A_ev, y_ev, meta):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > RCOND * s[0]
    r = int(keep.sum())
    inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    a = (Vt.T * inv) @ (U.T @ y)
    ny = float(np.linalg.norm(y_ev))
    return dict(A=A, y=y, A_ev=A_ev, y_ev=y_ev, ny=ny,
                d=int(A.shape[1]), n=int(A.shape[0]), rank=r,
                sigma=s, U=U, Vt=Vt,
                sigma1=float(s[0]), kappa=float(s[0] / s[keep][-1]),
                floor_pool=float(np.linalg.norm(A_ev @ a - y_ev) / ny),
                a_pool=a, **meta)


def prob_qi1d(N, lam=0.30, row_mult=6, noise_rel=0.0, seed=0):
    from src.construction.qi_mpmath import default_halo
    h = 2.0 / N
    gamma = lam / h
    halo = int(default_halo(N, lambda_star=lam))
    cen = -1.0 + h * np.arange(-halo, N + halo + 1)
    f = lambda z: np.sin(4.0 * z)
    xt = np.linspace(-1, 1, row_mult * (cen.size + 1))
    xe = np.linspace(-1, 1, 4001)
    ph = lambda x: np.hstack([np.tanh(gamma * (x[:, None] - cen[None, :])),
                              np.ones((x.size, 1))])
    y = f(xt)
    if noise_rel > 0:
        rng = np.random.default_rng(11 * seed + N)
        e = rng.standard_normal(xt.size)
        y = y + e * noise_rel * np.linalg.norm(y) / np.linalg.norm(e)
    return _finish(ph(xt), y, ph(xe), f(xe),
                   dict(family="qi1d", label=f"QI-1D  N={N}", noise_rel=noise_rel))


def prob_qi2d(N, geom="radon_tensor", lam=0.12, row_mult=6, noise_rel=0.0, seed=0):
    from geometries import GEOMETRIES
    from src.data.sampling2d import disk_grid, disk_uniform
    from src.data.targets2d import get_target_2d
    radius = 2.5 if geom.startswith("radon") else 1.6
    dirs, offs = GEOMETRIES[geom](N, radius, seed=0)
    gamma = lam / (2.8 / np.sqrt(N))
    d = len(offs) + 1
    Xt = disk_uniform(row_mult * d, radius=1.0, seed=seed)
    Xe = disk_grid(120, radius=1.0)
    f = get_target_2d("gauss_bump").fn_numpy
    ph = lambda P: np.hstack([np.tanh(gamma * (P @ dirs.T - offs[None, :])),
                              np.ones((len(P), 1))])
    y = f(Xt)
    if noise_rel > 0:
        rng = np.random.default_rng(77 + N + seed)
        e = rng.standard_normal(len(Xt))
        y = y + e * noise_rel * np.linalg.norm(y) / np.linalg.norm(e)
    return _finish(ph(Xt), y, ph(Xe), f(Xe),
                   dict(family="qi2d", label=f"QI-2D  N={N}", noise_rel=noise_rel))


def prob_twin(d, spectrum=None, rank_frac=0.6, row_mult=4, n_ev=1500,
              noise_rel=0.0, seed=0, label=None):
    """Random matrix with Haar singular vectors and a PRESCRIBED spectrum.

    spectrum=None gives the default gapless logspace(0,-14,r).  Passing an
    explicit array reproduces another matrix's spectrum exactly with no other
    structure -- that is the 'spectrally identical but random' control."""
    rng = np.random.default_rng(9000 + 31 * seed + d)
    sig = np.asarray(spectrum, float) if spectrum is not None else \
        np.logspace(0, -14, int(rank_frac * d))
    sig = sig[:d]
    r = sig.size
    ntr = row_mult * d
    n = ntr + n_ev
    U = np.linalg.qr(rng.standard_normal((n, r)))[0]
    V = np.linalg.qr(rng.standard_normal((d, r)))[0]
    A = (U * sig) @ V.T
    y_all = U @ (sig * rng.standard_normal(r))      # target energy ~ sigma
    y_all = y_all / np.linalg.norm(y_all) * np.sqrt(n)
    del U, V
    y = y_all[:ntr].copy()
    if noise_rel > 0:
        e = rng.standard_normal(ntr)
        y = y + e * noise_rel * np.linalg.norm(y) / np.linalg.norm(e)
    return _finish(A[:ntr], y, A[ntr:], y_all[ntr:],
                   dict(family="twin", label=label or f"twin  d={d}",
                        noise_rel=noise_rel))


def eval_err(p, w):
    return float(np.linalg.norm(p["A_ev"] @ w - p["y_ev"]) / p["ny"])


def damped_exact(p, mu):
    s = p["sigma"]
    return p["Vt"].T @ ((s / (s ** 2 + mu)) * (p["U"].T @ p["y"]))


def floor_1batch(p, b, seed=0, reps=2):
    """Best achievable seeing only ONE batch of b rows -- the no-accumulation
    reference the batched solver must beat (the 'soft win')."""
    A, y, n = p["A"], p["y"], p["n"]
    out = []
    for rep in range(reps):
        rng = np.random.default_rng(500 + 17 * rep + seed)
        S = rng.choice(n, min(b, n), replace=False)
        U, s, Vt = np.linalg.svd(A[S], full_matrices=False)
        kp = s > RCOND * s[0]
        a = (Vt.T * np.where(kp, 1 / np.where(kp, s, 1), 0)) @ (U.T @ y[S])
        out.append(eval_err(p, a))
    return float(np.exp(np.mean(np.log(np.maximum(out, 1e-300)))))


# ============================================================ solver =====

def probe_sched(maxiter, dense=24, ratio=1.18):
    s = list(range(1, min(dense, maxiter) + 1))
    v = float(dense)
    while v < maxiter:
        v *= ratio
        if int(v) > s[-1]:
            s.append(int(v))
    if s and s[-1] > maxiter:
        s[-1] = maxiter
    if not s or s[-1] != maxiter:
        s.append(maxiter)
    return sorted(set(x for x in s if 1 <= x <= maxiter))


def _reorth(z, buf, cnt):
    """Vectorised two-pass Gram-Schmidt (CGS2) against buf[:cnt]."""
    if cnt == 0:
        return z
    Q = buf[:cnt]
    z = z - Q.T @ (Q @ z)
    return z - Q.T @ (Q @ z)


def lsqr_damped(A, rhs, mu, d, maxiter, cwin=0, probe=None, probe_at=None,
                stop_below=None, brk=1e-14, reorth_u=True):
    """LSQR on [A; sqrt(mu) I] with rhs [rhs; 0], damping applied implicitly.

    cwin = 0    plain short recurrence, O(d) state (the ladder's solver)
    cwin = -1   full reorthogonalization, O(d*r) state (the terminal solve)
    otherwise   sliding window of cwin vectors

    Returns (x, history) where history is a list of (iteration, probe value,
    observable ||A_aug^T r||).
    """
    n = A.shape[0]
    sq = float(np.sqrt(mu)) if mu > 0 else 0.0
    damp = sq > 0

    def av(z):
        return (np.concatenate([A @ z, sq * z]) if damp else A @ z)

    def atu(u):
        return (A.T @ u[:n] + sq * u[n:]) if damp else A.T @ u

    m = n + d if damp else n
    u = np.concatenate([rhs, np.zeros(d)]) if damp else rhs.copy()
    beta = float(np.linalg.norm(u))
    if beta == 0:
        return np.zeros(d), []
    u /= beta
    v = atu(u)
    alpha = float(np.linalg.norm(v))
    if alpha == 0:
        return np.zeros(d), []
    v /= alpha

    full = (cwin == -1)
    cap = (maxiter + 1) if full else int(cwin)
    Ub = Vb = None
    cnt = pos = 0
    if cap > 0:
        Ub = np.empty((cap, m)) if reorth_u else None
        Vb = np.empty((cap, d))
        if reorth_u:
            Ub[0] = u
        Vb[0] = v
        cnt, pos = 1, 1 % cap

    w = v.copy()
    x = np.zeros(d)
    phibar, rhobar = beta, alpha
    at = set(probe_at or [])
    hist = []
    # Breakdown guard. With reorthogonalization the Krylov space is exhausted
    # after ~rank steps; the next vector is then pure rounding noise, and
    # normalising it destroys the iterate (measured: residual 2.5e15 without
    # this guard). alpha/beta are bidiagonal entries, not singular values --
    # they stay O(||A||) until the space runs out -- so a relative threshold is
    # safe even at kappa = 1e14.
    #
    # brk is LOAD-BEARING and worth two orders. Measured on the terminal solve,
    # eval rel L2 vs SVD floor:
    #   random d=512   1e-12: 9.7e-13 | 1e-14: 2.7e-15 | 1e-16: 2.0e-15 | 0: nan
    #   QI-1D N=256    1e-12: 4.1e-14 | 1e-14: 3.0e-15 | 1e-16: 3.0e-15 | 0: nan
    #   random d=2048  1e-12: 2.6e-13 | 1e-14: 2.1e-15 | 1e-16: 2.7e-15 | 0: nan
    # 1e-12 stops at ~0.79r and leaves the smallest directions unresolved; 0
    # normalises pure noise and destroys the iterate. The optimum is a plateau
    # over 1e-14..1e-16.
    anorm2 = alpha ** 2

    for it in range(1, maxiter + 1):
        un = av(v) - alpha * u
        if cap and reorth_u:
            un = _reorth(un, Ub, cnt)
        beta = float(np.linalg.norm(un))
        if beta <= brk * np.sqrt(anorm2):
            break
        u = un / beta
        vn = atu(u) - beta * v
        if cap:
            vn = _reorth(vn, Vb, cnt)
        alpha = float(np.linalg.norm(vn))
        anorm2 += alpha ** 2 + beta ** 2
        if alpha <= brk * np.sqrt(anorm2):
            break
        v = vn / alpha
        if cap:
            if reorth_u:
                Ub[pos] = u
            Vb[pos] = v
            pos = (pos + 1) % cap
            cnt = min(cnt + 1, cap)
        rho = float(np.hypot(rhobar, beta))
        cs, sn = rhobar / rho, beta / rho
        theta = sn * alpha
        rhobar = -cs * alpha
        phi = cs * phibar
        phibar = sn * phibar
        x += (phi / rho) * w
        w = v - (theta / rho) * w
        if it in at and probe is not None:
            pv = probe(x)
            hist.append((it, pv, float(abs(phibar * alpha * cs))))
            if stop_below is not None and pv <= stop_below:
                break
    return x, hist


# ================================================================ io =====

def append(path, rec):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
