"""expD09 validation suite -- shared machinery.

Validates the block-QR whitening recipe (results/checkpoint_D_optimizers/expD09_2nd_order_regime/expD09_recipe_results.md)
along the
four axes that decide whether it is deployable:

  A  trajectories      does it converge cleanly across targets and widths?
  B  k-vs-d coupling   is the state coefficient k INDEPENDENT of d?
                       (if k must grow with d the method is not O(d))
  C  noise             does it reach the statistical floor and stop, or
                       overfit the noise tail?
  D  batching          how does accuracy depend on rows used, for the
                       whitening and for the solve separately?

Everything here is self-contained: the problem builder, the solver, and the
metrics. Nothing imports from the parent experiment except repo geometry.

Conventions kept from the parent experiment: uniform centers + halo,
gamma = lambda*/h with lambda* = 0.25, augmented [Phi, 1], eval on 4001
equispaced points, all fp64.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RESULTS = (REPO_ROOT / "results" / "checkpoint_D_optimizers"
           / "expD09_2nd_order_regime" / "validation")
FIGS = RESULTS / "figures"

LAMBDA_STAR = 0.25
RCOND_SVD = 1e-15          # truncated-SVD reference floor
RCOND_BLK = 1e-13          # block QR drop tolerance (recipe detail 4)
N_EVAL = 4001
ROW_MULT = 8               # n = ROW_MULT * d, so batching can go down to b ~ d

TARGETS = ["sine", "sine_8pi", "runge"]
WIDTHS = [64, 128, 256]
WIDTHS_SCALING = [64, 128, 256, 512]     # experiment B needs the extra decade
YLIM = (1e-17, 1e1)


def geometry(N: int):
    from src.construction.qi_mpmath import default_halo
    h = 2.0 / N
    halo = default_halo(N, lambda_star=LAMBDA_STAR)
    centers = -1.0 + np.arange(-halo, N + halo + 1, dtype=np.float64) * h
    return centers, LAMBDA_STAR / h


def build_phi_aug(x, gamma, centers):
    phi = np.tanh(gamma * (x[:, None] - centers[None, :]))
    return np.hstack([phi, np.ones((x.size, 1))])


def make_problem(fn: str, N: int, noise_rel: float = 0.0, seed: int = 0):
    """Frozen system with optional y-noise.

    `noise_rel` is the relative noise level: eps ~ N(0, s^2) with s chosen so
    ||eps|| / ||y_clean|| ~ noise_rel. The solver only ever sees y_noisy; all
    reporting is against the CLEAN target on the eval grid, which is what we
    actually care about and what makes over-fitting visible.
    """
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.data.targets import get_target

    centers, gamma = geometry(N)
    d = centers.size + 1
    n = ROW_MULT * d
    x_tr = np.linspace(-1.0, 1.0, n)
    x_ev = np.linspace(-1.0, 1.0, N_EVAL)
    f = get_target(fn).fn_numpy
    y_clean, y_ev = f(x_tr), f(x_ev)

    rng = np.random.default_rng(1000 * seed + N)
    if noise_rel > 0.0:
        e = rng.standard_normal(n)
        e *= noise_rel * np.linalg.norm(y_clean) / np.linalg.norm(e)
        y = y_clean + e
    else:
        y = y_clean.copy()

    A = build_phi_aug(x_tr, gamma, centers)
    A_ev = build_phi_aug(x_ev, gamma, centers)

    # reference: truncated-SVD solve on the DATA THE SOLVER SEES
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > RCOND_SVD * s[0]
    a_svd = Vt.T @ (np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
                    * (U.T @ y))
    ye_norm = np.linalg.norm(y_ev)
    return {
        "fn": fn, "N": N, "d": d, "n": n, "rank": int(keep.sum()),
        "noise_rel": noise_rel,
        "A": A, "y": y, "y_clean": y_clean, "A_ev": A_ev, "y_ev": y_ev,
        "ye_norm": float(ye_norm),
        "floor_svd": float(np.linalg.norm(A_ev @ a_svd - y_ev) / ye_norm),
        # the statistical floor: what noise alone permits, ~ sigma/sqrt(n)
        # expressed on the same relative-L2 scale
        "floor_stat": float(noise_rel / np.sqrt(max(n, 1)) * np.sqrt(
            max(int(keep.sum()), 1))),
    }


# ---------------------------------------------------------------- solver ---

def block_qr(A, k, rcond=RCOND_BLK, rows=None):
    """Recipe stage 1 + details 3/4: contiguous blocks, pivoted QR, drop.

    `rows` restricts the FACTORIZATION to a row subset (the batching axis);
    the returned B is still whitened on every row of A.
    """
    n, d = A.shape
    B = np.zeros((n, d))
    pieces = []
    for t in range(0, d, k):
        C = np.arange(t, min(t + k, d))
        src = A[:, C] if rows is None else A[np.ix_(rows, C)]
        Q, R, piv = sla.qr(src, mode="economic", pivoting=True)
        dg = np.abs(np.diag(R))
        nk = int((dg > rcond * max(dg[0], 1e-300)).sum())
        if nk == 0:
            pieces.append((C, piv, R[:0, :], 0))
            continue
        if rows is None:
            B[:, C[:nk]] = Q[:, :nk]
        else:
            # apply the subset-derived R to all rows
            B[:, C[:nk]] = sla.solve_triangular(
                R[:nk, :nk], A[:, C[piv[:nk]]].T, lower=False, trans="T").T
        pieces.append((C, piv, R[:nk, :], nk))
    return B, pieces


def unwhiten(pieces, z, d):
    """Recipe stage 3: apply R^{-1} exactly ONCE, at the end."""
    w = np.zeros(d)
    for C, piv, R, nk in pieces:
        if nk == 0:
            continue
        c = sla.solve_triangular(R[:, :nk], z[C[:nk]], lower=False)
        out = np.zeros(len(C))
        out[piv[:len(c)]] = c
        w[C] = out
    return w


def lsqr_traj(B, b, d, iters, on_iter=None):
    """Plain LSQR (Paige-Saunders) on the whitened operator, with an exact
    per-iteration hook -- no re-running to sample a trajectory."""
    u = b.copy()
    beta = np.linalg.norm(u)
    if beta == 0:
        return np.zeros(d)
    u /= beta
    v = B.T @ u
    alpha = np.linalg.norm(v)
    if alpha == 0:
        return np.zeros(d)
    v /= alpha
    w = v.copy()
    x = np.zeros(d)
    phibar, rhobar = beta, alpha
    for it in range(1, iters + 1):
        uu = B @ v - alpha * u
        beta = np.linalg.norm(uu)
        if beta <= 0:
            break
        u = uu / beta
        vv = B.T @ u - beta * v
        alpha = np.linalg.norm(vv)
        if alpha <= 0:
            break
        v = vv / alpha
        rho = np.hypot(rhobar, beta)
        c, s = rhobar / rho, beta / rho
        theta = s * alpha
        rhobar = -c * alpha
        phi = c * phibar
        phibar = s * phibar
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w
        if on_iter is not None:
            on_iter(it, x)
    return x


def solve_traj(prob, k, iters, rows_qr=None, solve_rows=None, stride=1):
    """Full recipe with a recorded trajectory.

    rows_qr     rows used for the QR factorization (None = all)
    solve_rows  rows used for the LSQR solve       (None = all)
    Returns dict of per-iteration eval/train relative L2 (vs the CLEAN target
    on the eval grid, and vs the observed y on the train rows).
    """
    A, y = prob["A"], prob["y"]
    d = prob["d"]
    B, pieces = block_qr(A, k, rows=rows_qr)
    sr = np.arange(A.shape[0]) if solve_rows is None else solve_rows
    Bs, ys = B[sr], y[sr]
    A_ev, y_ev, ye_norm = prob["A_ev"], prob["y_ev"], prob["ye_norm"]
    yn = np.linalg.norm(ys)
    ev, tr = [], []

    def hook(it, z):
        if it % stride and it != iters:
            return
        w = unwhiten(pieces, z, d)
        ev.append(float(np.linalg.norm(A_ev @ w - y_ev) / ye_norm))
        tr.append(float(np.linalg.norm(A[sr] @ w - ys) / yn))

    lsqr_traj(Bs, ys, d, iters, on_iter=hook)
    ev = np.asarray(ev)
    return {"eval": ev, "train": np.asarray(tr), "stride": stride,
            "best": float(ev.min()) if ev.size else float("nan"),
            "best_it": int(ev.argmin()) + 1 if ev.size else -1,
            "final": float(ev[-1]) if ev.size else float("nan")}


# --------------------------------------------------------------- caching ---

def save_rows(name, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(rows))


def load_rows(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None
