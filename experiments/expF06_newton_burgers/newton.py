"""Damped Newton for steady Burgers; every step is one block collocation lstsq
in a FROZEN ridge basis, so iterates add in coefficient space.

newton_burgers(..., base_fields=None): base_fields(P) -> dict with keys
u, ux, uy, lap_u, v, vx, vy, lap_v (numpy [n]) is an optional frozen warm
start (the trained PINN in expF07); the ridge expansion carries corrections.
"""
from __future__ import annotations

import time

import numpy as np

import ridge_core as rc
import burgers as bp

FIELD_TERMS = {
    "": (((0, 0), 1.0),),
    "x": (((1, 0), 1.0),),
    "y": (((0, 1), 1.0),),
    "lap": (((2, 0), 1.0), ((0, 2), 1.0)),
}


def _ridge_fields(geom, family, sol_u, sol_v, P):
    dirs, offs, gammas = geom
    out = {}
    for name, sol in (("u", sol_u), ("v", sol_v)):
        m = dict(dirs=dirs, offs=offs, gammas=gammas, sol=sol, family=family)
        out[name] = rc.eval_model(m, P)
        out[name + "x"] = rc.eval_model(m, P, terms=FIELD_TERMS["x"])
        out[name + "y"] = rc.eval_model(m, P, terms=FIELD_TERMS["y"])
        out["lap_" + name] = rc.eval_model(m, P, terms=FIELD_TERMS["lap"])
    return out


def _total_fields(geom, family, sol_u, sol_v, P, base_fields):
    f = _ridge_fields(geom, family, sol_u, sol_v, P)
    if base_fields is not None:
        b = base_fields(P)
        for k in f:
            f[k] = f[k] + b[k]
    return f


def _residuals(f, P, nu):
    F_u = f["u"] * f["ux"] + f["v"] * f["uy"] - nu * f["lap_u"] - bp.f_u(P, nu)
    F_v = f["u"] * f["vx"] + f["v"] * f["vy"] - nu * f["lap_v"] - bp.f_v(P, nu)
    return F_u, F_v


def newton_burgers(nu, W, lam, family=rc.tanh_family, max_iter=12, seed=42,
                   base_fields=None, u_exact=bp.u_exact, v_exact=bp.v_exact,
                   n_eval=120, init_sol=None):
    """init_sol=(sol_u, sol_v): warm-start the ridge coefficients (same W/lam/
    family geometry) instead of zero. Used by the nu-continuation ladder to
    seed a low-nu solve from a converged higher-nu one."""
    rng = np.random.default_rng(seed)
    geom = rc.radon_geometry(W, lam)
    n_feat = len(geom[1]) + len(rc.MONO_2D)
    P = rc.interior_points_square(len(geom[1]), rng)
    Pb = rc.boundary_points_square()
    g = np.linspace(-0.995, 0.995, n_eval)
    Pe = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    ue, ve = u_exact(Pe), v_exact(Pe)

    if init_sol is None:
        sol_u = np.zeros(n_feat)
        sol_v = np.zeros(n_feat)
    else:
        sol_u = np.array(init_sol[0], dtype=np.float64)
        sol_v = np.array(init_sol[1], dtype=np.float64)
    history = []
    t0 = time.time()
    for it in range(max_iter + 1):
        f = _total_fields(geom, family, sol_u, sol_v, P, base_fields)
        F_u, F_v = _residuals(f, P, nu)
        res_norm = float(np.sqrt(np.mean(F_u**2 + F_v**2)))
        fe = _total_fields(geom, family, sol_u, sol_v, Pe, base_fields)
        history.append(dict(iter=it, res_norm=res_norm,
                            rel_l2_u=rc.rel_l2(fe["u"], ue),
                            rel_l2_v=rc.rel_l2(fe["v"], ve),
                            t=time.time() - t0))
        print(history[-1], flush=True)
        if it == max_iter or res_norm < 1e-13:
            break
        # block Jacobian rows: J_uu du + J_uv dv = -F_u ; J_vu du + J_vv dv = -F_v
        A_uu = rc.rows_2d(P, *geom, terms=[((1, 0), f["u"]), ((0, 1), f["v"]),
                                           ((0, 0), f["ux"]),
                                           ((2, 0), -nu), ((0, 2), -nu)],
                          family=family)
        A_uv = rc.rows_2d(P, *geom, terms=[((0, 0), f["uy"])], family=family)
        A_vu = rc.rows_2d(P, *geom, terms=[((0, 0), f["vx"])], family=family)
        A_vv = rc.rows_2d(P, *geom, terms=[((1, 0), f["u"]), ((0, 1), f["v"]),
                                           ((0, 0), f["vy"]),
                                           ((2, 0), -nu), ((0, 2), -nu)],
                          family=family)
        A_pde = np.block([[A_uu, A_uv], [A_vu, A_vv]])
        y_pde = np.concatenate([-F_u, -F_v])
        s = np.abs(A_pde).max()
        # Dirichlet BC on the correction: delta = exact - current on the boundary
        fb = _total_fields(geom, family, sol_u, sol_v, Pb, base_fields)
        Rb = rc.rows_2d(Pb, *geom, terms=[((0, 0), 1.0)], family=family)
        Zb = np.zeros_like(Rb)
        wb = np.sqrt(len(P) / len(Pb))
        A_bc = np.block([[Rb, Zb], [Zb, Rb]])
        y_bc = np.concatenate([u_exact(Pb) - fb["u"], v_exact(Pb) - fb["v"]])
        A = np.vstack([A_pde / s, wb * A_bc])
        y = np.concatenate([y_pde / s, wb * y_bc])
        dsol = np.linalg.lstsq(A, y, rcond=rc.RCOND)[0]
        du, dv = dsol[:n_feat], dsol[n_feat:]
        # backtracking line search on the collocation residual norm
        alpha = 1.0
        accepted = False
        while alpha > 1.0 / 256:
            tu, tv = sol_u + alpha * du, sol_v + alpha * dv
            ft = _total_fields(geom, family, tu, tv, P, base_fields)
            Fu_t, Fv_t = _residuals(ft, P, nu)
            if np.sqrt(np.mean(Fu_t**2 + Fv_t**2)) < res_norm:
                sol_u, sol_v = tu, tv
                accepted = True
                break
            alpha /= 2
        if not accepted:
            history[-1]["stalled"] = True
            break
    return dict(geom=geom, family=family, sol_u=sol_u, sol_v=sol_v,
                history=history)
