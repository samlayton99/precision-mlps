"""Frozen-ridge collocation-lstsq core on [-1,1]^2, generalized over the
univariate family and per-neuron gamma.

u(p) = sum_m c_m phi(gamma_m (w_m . p - t_m)) + poly_deg<=3(p).
family(order, Z) returns d^order/dz^order phi(z) elementwise. gammas is [M].
All coefficients from ONE min-norm lstsq over stacked PDE + BC rows.
Vendored/generalized from expF01 (and continuous-mlps precision_pde/core.py).

Operator terms: [((ax, ay), coeff)], coeff a float, callable(P)->[n], or [n] array.
"""
from __future__ import annotations

import numpy as np

RCOND = 1e-13
COLLAR_SQUARE = 1.6

MONO_2D = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
           (3, 0), (2, 1), (1, 2), (0, 3)]


def tanh_family(order, Z):
    t = np.tanh(Z)
    if order == 0:
        return t
    if order == 1:
        return 1.0 - t * t
    if order == 2:
        return -2.0 * t * (1.0 - t * t)
    if order == 3:
        s = 1.0 - t * t
        return -2.0 * s * (1.0 - 3.0 * t * t)
    raise ValueError(order)


def bspline_family(order, Z):
    """Cubic B-spline bump: support [-2,2], C^2, closed-form derivatives.
    z in [0,1]: B = 2/3 - z^2 + z^3/2;  z in [1,2]: B = (2-z)^3/6; even in z."""
    z = np.abs(Z)
    sgn = np.sign(Z)
    inner = z <= 1.0
    outer = (z > 1.0) & (z < 2.0)
    out = np.zeros_like(np.asarray(Z, dtype=np.float64))
    zi, zo = z[inner], z[outer]
    if order == 0:
        out[inner] = 2.0 / 3.0 - zi**2 + 0.5 * zi**3
        out[outer] = (2.0 - zo)**3 / 6.0
    elif order == 1:
        out[inner] = (-2.0 * zi + 1.5 * zi**2) * sgn[inner]
        out[outer] = -0.5 * (2.0 - zo)**2 * sgn[outer]
    elif order == 2:
        out[inner] = -2.0 + 3.0 * zi
        out[outer] = 2.0 - zo
    elif order == 3:
        out[inner] = 3.0 * sgn[inner]
        out[outer] = -1.0 * sgn[outer]
    else:
        raise ValueError(order)
    return out


def _ffact(d, o):
    out = 1
    for k in range(o):
        out *= (d - k)
    return out


def _coeff_col(coeff, pts):
    """Coefficient -> column [n,1], scalar float, for a term at pts."""
    if callable(coeff):
        return np.asarray(coeff(pts), dtype=np.float64).reshape(-1, 1)
    arr = np.asarray(coeff, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    return arr.reshape(-1, 1)


def pi_thetas(J):
    return np.pi * (np.arange(J) + 0.5) / J


def radon_geometry(W, lam, collar=COLLAR_SQUARE):
    """Uniform Radon tensor geometry. Returns dirs [M,2], offs [M], gammas [M]."""
    J = int(round(np.sqrt(W)))
    M = W // J
    thetas = pi_thetas(J)
    ts = np.linspace(-collar, collar, M)
    dirs = np.repeat(np.stack([np.cos(thetas), np.sin(thetas)], axis=1), M, axis=0)
    offs = np.tile(ts, J)
    h_ref = 2.8 / np.sqrt(J * M)
    gammas = np.full(len(offs), lam / h_ref)
    return dirs, offs, gammas


def rows_2d(P, dirs, offs, gammas, terms, family):
    """[L Phi | L poly] rows at P."""
    Z = (P @ dirs.T - offs[None, :]) * gammas[None, :]
    A = np.zeros_like(Z)
    polys = np.zeros((len(P), len(MONO_2D)))
    x, y = P[:, 0], P[:, 1]
    for (ax, ay), coeff in terms:
        o = ax + ay
        cc = _coeff_col(coeff, P)
        dir_fac = (dirs[:, 0] ** ax * dirs[:, 1] ** ay)[None, :]
        A += cc * (gammas[None, :] ** o) * dir_fac * family(o, Z)
        ccr = cc.ravel() if np.ndim(cc) else cc
        for k, (px, py) in enumerate(MONO_2D):
            if ax <= px and ay <= py:
                mono = (_ffact(px, ax) * _ffact(py, ay)
                        * x ** (px - ax) * y ** (py - ay))
                polys[:, k] += ccr * mono
    return np.hstack([A, polys])


def boundary_points_square(n_per_edge=480):
    s = np.linspace(-1.0, 1.0, n_per_edge)
    edges = [np.stack([s, np.full_like(s, -1.0)], axis=1),
             np.stack([s, np.full_like(s, 1.0)], axis=1),
             np.stack([np.full_like(s, -1.0), s], axis=1),
             np.stack([np.full_like(s, 1.0), s], axis=1)]
    return np.concatenate(edges, axis=0)


def interior_points_square(n_feat, rng):
    n = max(5 * n_feat, 2000)
    return rng.uniform(-1.0, 1.0, (n, 2))


def solve_square(terms, forcing, bc_blocks, W=None, lam=None, family=tanh_family,
                 seed=42, geometry=None):
    """One stacked min-norm lstsq. Either pass W+lam (uniform Radon geometry) or
    geometry=(dirs, offs, gammas) explicitly (adaptive knots)."""
    rng = np.random.default_rng(seed)
    if geometry is None:
        dirs, offs, gammas = radon_geometry(W, lam)
    else:
        dirs, offs, gammas = geometry
    P = interior_points_square(len(offs), rng)
    A_pde = rows_2d(P, dirs, offs, gammas, terms, family)
    y_pde = forcing(P) if callable(forcing) else np.full(len(P), float(forcing))
    s = np.abs(A_pde).max()
    rows, vals = [A_pde / s], [y_pde / s]
    for blk in bc_blocks:
        Pb = np.asarray(blk["points"], dtype=np.float64)
        r = rows_2d(Pb, dirs, offs, gammas, blk["terms"], family)
        w = np.sqrt(len(P) / len(Pb))
        rows.append(w * r)
        vals.append(w * np.asarray(blk["values"], dtype=np.float64))
    A = np.vstack(rows)
    y = np.concatenate(vals)
    sol = np.linalg.lstsq(A, y, rcond=RCOND)[0]
    return dict(dirs=dirs, offs=offs, gammas=gammas, sol=sol, family=family)


def eval_model(model, P, terms=(((0, 0), 1.0),), chunk=4096):
    """Evaluate L[u_hat] at P; default L = identity."""
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        R = rows_2d(P[i:i + chunk], model["dirs"], model["offs"],
                    model["gammas"], list(terms), model["family"])
        out[i:i + chunk] = R @ model["sol"]
    return out


def rel_l2(u_hat, u_true):
    return float(np.linalg.norm(u_hat - u_true) / np.linalg.norm(u_true))


def linf(u_hat, u_true):
    return float(np.max(np.abs(u_hat - u_true)))
