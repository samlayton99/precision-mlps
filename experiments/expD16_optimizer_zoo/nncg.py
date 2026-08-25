"""Nystrom Newton-CG (NNCG) finisher, after Rathore et al. 2024
("Challenges in Training PINNs: A Loss Landscape Perspective", Alg. 3).

Per outer step:
  1. (every `precond_freq` steps) build a rank-`rank` Nystrom approximation
     U diag(lam) U^T of the loss Hessian H at the current point, via `rank`
     Hessian-vector products against a random orthonormal sketch.
  2. Solve (H + mu*I) d = -g by Nystrom-preconditioned CG (one HVP per CG
     iteration), with a Steihaug-style negative-curvature bailout.
  3. Armijo backtracking line search along d.

Everything is flat-vector fp64. The HVP oracle is exact double-backward
autograd -- no Gauss-Newton approximation, matching the published method.
"""

from __future__ import annotations

import math

import torch


def nystrom_approx(hvp, m, rank, device, generator=None):
    """Rank-`rank` Nystrom approximation of the (symmetric) Hessian.

    Returns (U [m,rank], lam [rank]) with H ~= U diag(lam) U^T.
    Uses the numerically stable shifted construction (Frangella et al. 2021).
    """
    Om = torch.randn(m, rank, dtype=torch.float64, device=device, generator=generator)
    Om, _ = torch.linalg.qr(Om)                       # orthonormal sketch
    Y = torch.stack([hvp(Om[:, j]) for j in range(rank)], dim=1)   # H @ Om
    nu = math.sqrt(m) * torch.finfo(torch.float64).eps * torch.linalg.norm(Y)
    Y_nu = Y + nu * Om
    C = Om.T @ Y_nu
    C = 0.5 * (C + C.T)
    try:
        L = torch.linalg.cholesky(C)
        B = torch.linalg.solve_triangular(L, Y_nu.T, upper=False).T
    except torch.linalg.LinAlgError:
        # Indefinite core (nonconvex point): fall back to eigendecomposition
        # of the projected Hessian -- still a usable preconditioner.
        evals, evecs = torch.linalg.eigh(C)
        keep = evals.abs() > 1e-14 * evals.abs().max()
        B = Y_nu @ (evecs[:, keep] / evals[keep].abs().sqrt())
    U, S, _ = torch.linalg.svd(B, full_matrices=False)
    lam = torch.clamp(S**2 - nu, min=0.0)
    return U, lam


def nystrom_pcg(hvp, g, U, lam, mu, max_iter, tol):
    """Preconditioned CG for (H + mu I) d = -g with the Nystrom preconditioner
    P^{-1} v = U (lam+mu)^{-1} U^T v + (lam_r+mu)^{-1} (v - U U^T v).
    Steihaug bailout on nonpositive curvature. Returns d."""
    lam_r = lam[-1] if lam.numel() else torch.tensor(0.0, dtype=g.dtype, device=g.device)

    def precond(v):
        Utv = U.T @ v
        return U @ (Utv / (lam + mu)) + (v - U @ Utv) / (lam_r + mu)

    d = torch.zeros_like(g)
    r = -g.clone()                                    # residual of (H+mu)d = -g at d=0
    z = precond(r)
    p = z.clone()
    rz = torch.dot(r, z)
    g_norm = torch.linalg.norm(g)
    for _ in range(max_iter):
        Hp = hvp(p) + mu * p
        pHp = torch.dot(p, Hp)
        if pHp <= 0:                                  # negative curvature: stop here
            if torch.linalg.norm(d) == 0:
                return z                              # first iter: preconditioned SD
            return d
        alpha = rz / pHp
        d = d + alpha * p
        r = r - alpha * Hp
        if torch.linalg.norm(r) <= tol * g_norm:
            break
        z = precond(r)
        rz_new = torch.dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return d


def nncg_minimize(x0, fg, hvp_at, max_steps=1000, rank=100, mu=1e-8,
                  precond_freq=25, cg_iters=25, cg_tol=1e-12,
                  armijo_c=1e-4, ls_shrink=0.5, max_ls=40,
                  step_callback=None, seed=0):
    """Minimise via NNCG. fg(x) -> (f, g); hvp_at(x) -> hvp(v) closure.

    mu is a *relative* damping: the system solved is (H + mu*lam_max*I) d = -g
    with lam_max the Nystrom top eigenvalue (floored at 1e-30 absolute).
    """
    x = x0.detach().clone()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    f, g = fg(x)
    U = lam = None
    mu_abs = 1e-30
    for step in range(int(max_steps)):
        if U is None or step % int(precond_freq) == 0:
            hvp = hvp_at(x)
            U, lam = nystrom_approx(hvp, x.numel(), min(rank, x.numel()), x.device, gen)
            mu_abs = max(float(mu * (lam[0] if lam.numel() else 0.0)), 1e-30)
        hvp = hvp_at(x)
        d = nystrom_pcg(hvp, g, U, lam, mu_abs, int(cg_iters), cg_tol)
        gd = float(torch.dot(g, d))
        if not math.isfinite(gd) or gd >= 0:
            d, gd = -g, -float(torch.dot(g, g))       # fallback: steepest descent
        # Armijo backtracking
        alpha, accepted = 1.0, False
        f_new, g_new = f, g
        for _ in range(int(max_ls)):
            f_try, g_try = fg(x + alpha * d)
            if math.isfinite(f_try) and f_try <= f + armijo_c * alpha * gd:
                f_new, g_new, accepted = f_try, g_try, True
                break
            alpha *= ls_shrink
        if not accepted:
            break                                     # no progress at any scale
        x = x + alpha * d
        f, g = f_new, g_new
        if step_callback is not None:
            step_callback(step + 1, x, f, float(torch.linalg.norm(g)))
        if float(torch.linalg.norm(g)) <= 1e-23:
            break
    return x
