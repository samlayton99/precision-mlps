"""Training-free 3D Navier-Stokes solve in the tensor-product sech^2 basis:
frozen QI geometry + Gauss-Newton collocation lstsq for the readout (expF12).

This is the checkpoint-F recipe (expF02/expF06 Newton-lstsq, expF09 multi-field
blocks) carried into the d=4 tensor-product architecture. The geometry is the
QI one -- per-axis uniform centers with halo, gamma = lambda / h, frozen -- and
the readout is SOLVED, never trained.

The obstacle: the Newton-linearized NS operator has variable coefficients
(u0 . grad), which breaks Kronecker separability, so the readout solve cannot
be mode-wise, and a dense lstsq over all 4 Nt^4 coefficients is impossible
(~10^5 unknowns). The fix comes from the ceiling study: fp64 can only use the
product-SVD directions whose PRODUCT singular value survives truncation anyway
(~5-15%). So the Gauss-Newton steps are solved in the REDUCED product-SVD
basis: per axis Phi_a = U_a S_a V_a^T on a reference grid, transformed features
psi_a = feats_a V_a diag(1/S_a) (orthonormal columns on the grid), and only the
top-K tuples by product singular value s_i s_j s_k s_l are kept per field.
Each Newton step is then one dense min-norm lstsq with ~4K unknowns.

Newton linearization of the quadratic advection term B(u,u) = (u.grad)u at u0:
    B(u0, u) + B(u, u0) - B(u0, u0)
so each step solves the linear system (x-momentum shown; y,z cyclic):
    u_t + (u0.grad) u + u dx(u0) + v dy(u0) + w dz(u0) + p_x - nu lap u
        = (u0.grad) u0
    div u = 0;  u = g on faces;  u = u0_ic at t=0;  pressure tap rows.
theta = 0 start makes the first step exactly the Stokes solve. Damped step if
the interior NS residual increases. All numpy float64; block rows scaled to
O(1) max entry before stacking (core_system convention); scipy gelsd lstsq.
"""
from __future__ import annotations

import json
import time

import numpy as np
import scipy.linalg
import torch

import beltrami as bel
import pinn
import tensor_basis as tb

RCOND = 1e-13


class ReducedTensorBasis:
    """Per-axis transformed sech^2 features + top-K product-direction tuples."""

    def __init__(self, n_centers=12, lam=0.15, domains=bel.DOMAINS,
                 K=2000, m_svd=None):
        self.net = tb.TensorBasisNet(n_centers=n_centers, lam=lam,
                                     domains=domains)
        m = m_svd or 4 * n_centers
        self.T = []                       # per-axis transform [Nt, R]
        Ss = []
        with torch.no_grad():
            for a, (lo, hi) in enumerate(domains):
                g = torch.tensor(np.linspace(lo, hi, m))
                Phi = self.net.axis_features(g, a, [0])[0].numpy()
                _, S, Vt = np.linalg.svd(Phi, full_matrices=False)
                self.T.append(Vt.T / S[None, :])
                Ss.append(S)
        prod = np.einsum("i,j,k,l->ijkl", *Ss)
        flat = prod.ravel()
        order = np.argsort(flat)[::-1][:K]
        self.idx = np.stack(np.unravel_index(order, prod.shape), axis=0)  # [4,K]
        self.K = len(order)
        self.prod_sv = flat[order]

    def axis_feats(self, X, orders_per_axis):
        """X [B,4] -> per-axis dict {order: [B, R]} of transformed features."""
        out = []
        with torch.no_grad():
            for a in range(4):
                fa = self.net.axis_features(torch.tensor(X[:, a]), a,
                                            sorted(orders_per_axis[a]))
                out.append({o: f.numpy() @ self.T[a] for o, f in fa.items()})
        return out

    def op_columns(self, X, terms, feats=None):
        """[B, K] columns of sum_terms coeff * d^orders applied to the K kept
        basis functions. terms: list of (orders 4-tuple, coeff scalar or [B])."""
        need = [set() for _ in range(4)]
        for orders, _ in terms:
            for a in range(4):
                need[a].add(orders[a])
        F = feats or self.axis_feats(X, need)
        out = np.zeros((len(X), self.K))
        for orders, coeff in terms:
            col = (F[0][orders[0]][:, self.idx[0]]
                   * F[1][orders[1]][:, self.idx[1]]
                   * F[2][orders[2]][:, self.idx[2]]
                   * F[3][orders[3]][:, self.idx[3]])
            out += (np.asarray(coeff).reshape(-1, 1) * col
                    if np.ndim(coeff) else float(coeff) * col)
        return out


VAL = (0, 0, 0, 0)
DT = (0, 0, 0, 1)
DX, DY, DZ = (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)
DXX, DYY, DZZ = (2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0)
GRADS = (DX, DY, DZ)


def _field_ops(basis, X, theta, ops):
    """Evaluate operators `ops` of the 3 velocity fields (+ pressure last) at X.
    theta: [3K + Kp] stacked (u, v, w, p). Returns {op: [B, 3]} velocities only
    (pressure handled separately where needed)."""
    K = basis.K
    out = {}
    for op in ops:
        C = basis.op_columns(X, [(op, 1.0)])
        out[op] = np.stack([C @ theta[i * K:(i + 1) * K] for i in range(3)],
                           axis=1)
    return out


def ns_residual_interior(basis_v, basis_p, X, theta, nu):
    """Full nonlinear NS residual (RMS over momentum comps + max |div|)."""
    Kv, Kp = basis_v.K, basis_p.K
    d = _field_ops(basis_v, X, theta, [VAL, DT, DX, DY, DZ, DXX, DYY, DZZ])
    u = d[VAL]
    adv = (u[:, 0:1] * d[DX] + u[:, 1:2] * d[DY] + u[:, 2:3] * d[DZ])
    lap = d[DXX] + d[DYY] + d[DZZ]
    gp = np.stack([basis_p.op_columns(X, [(g, 1.0)]) @ theta[3 * Kv:]
                   for g in GRADS], axis=1)
    mom = d[DT] + adv + gp - nu * lap
    div = d[DX][:, 0] + d[DY][:, 1] + d[DZ][:, 2]
    return float(np.sqrt((mom ** 2).mean())), float(np.abs(div).max())


def solve_navier_stokes(n_centers=12, lam=0.15, K_vel=2000, K_p=1000,
                        n_int=5000, n_ic=1500, n_bc=2000, newton_iters=6,
                        seed=0, rcond=RCOND, log_path=None, verbose=True):
    """Gauss-Newton NS solve. Returns (basis_v, basis_p, theta, history)."""
    rng = np.random.default_rng(seed)
    basis_v = ReducedTensorBasis(n_centers, lam, K=K_vel)
    basis_p = ReducedTensorBasis(n_centers, lam, K=K_p)
    Kv, Kp = basis_v.K, basis_p.K
    ncols = 3 * Kv + Kp

    Xi = pinn.sample_interior(rng, n_int).numpy()
    Xic = pinn.sample_ic(rng, n_ic).numpy()
    Xbc = pinn.sample_bc(rng, n_bc).numpy()
    Xg = pinn.gauge_points().numpy()
    Fic, Fbc, Fg = bel.fields(Xic), bel.fields(Xbc), bel.fields(Xg)
    Xe, Fe = pinn.make_eval_set()
    nu = bel.NU

    # constant (theta-independent) row blocks
    blocks_const = []

    def add_block(A, y):
        s = np.abs(A).max()
        s = s if s > 0 else 1.0
        blocks_const.append((A / s, np.asarray(y) / s))

    # continuity
    zeros_p = np.zeros((n_int, Kp))
    A_cont = np.hstack([basis_v.op_columns(Xi, [(DX, 1.0)]),
                        basis_v.op_columns(Xi, [(DY, 1.0)]),
                        basis_v.op_columns(Xi, [(DZ, 1.0)]), zeros_p])
    add_block(A_cont, np.zeros(n_int))
    # IC + BC velocity value rows
    for X, F, n in [(Xic, Fic, n_ic), (Xbc, Fbc, n_bc)]:
        Cv = basis_v.op_columns(X, [(VAL, 1.0)])
        Z = np.zeros((n, Kv))
        Zp = np.zeros((n, Kp))
        for i in range(3):
            row = [Z, Z, Z, Zp]
            row[i] = Cv
            add_block(np.hstack(row), F[:, i])
    # pressure gauge (tap line)
    Cg = basis_p.op_columns(Xg, [(VAL, 1.0)])
    add_block(np.hstack([np.zeros((len(Xg), 3 * Kv)), Cg]), Fg[:, 3])

    # precompute interior features (all orders) once -- reused every iteration
    need = [{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1}]
    Fi_v = basis_v.axis_feats(Xi, need)
    Fi_p = basis_p.axis_feats(Xi, [{0, 1}] * 4)
    visc = [(DT, 1.0), (DXX, -nu), (DYY, -nu), (DZZ, -nu)]
    grad_p_cols = [basis_p.op_columns(Xi, [(g, 1.0)], feats=Fi_p)
                   for g in GRADS]

    theta = np.zeros(ncols)
    history = []
    t0 = time.time()

    def record(it, extra):
        mom_rms, div_max = ns_residual_interior(basis_v, basis_p, Xi[:3000],
                                                theta, nu)
        pred_v = np.stack(
            [basis_v.op_columns(Xe.numpy(), [(VAL, 1.0)])
             @ theta[i * Kv:(i + 1) * Kv] for i in range(3)], axis=1)
        pred_p = basis_p.op_columns(Xe.numpy(), [(VAL, 1.0)]) @ theta[3 * Kv:]
        E = Fe.numpy()
        rel_v = float(np.linalg.norm(pred_v - E[:, :3])
                      / np.linalg.norm(E[:, :3]))
        dp = pred_p - E[:, 3]
        dp -= dp.mean()
        rel_p = float(np.linalg.norm(dp)
                      / np.linalg.norm(E[:, 3] - E[:, 3].mean()))
        rec = dict(iter=it, wall=time.time() - t0, rel_l2_v=rel_v,
                   rel_l2_p=rel_p, mom_rms=mom_rms, div_max=div_max, **extra)
        history.append(rec)
        if verbose:
            print(rec, flush=True)
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    record(0, dict(phase="init"))
    for it in range(1, newton_iters + 1):
        d0 = _field_ops(basis_v, Xi, theta, [VAL, DX, DY, DZ])
        u0 = d0[VAL]
        adv0 = (u0[:, 0:1] * d0[DX] + u0[:, 1:2] * d0[DY]
                + u0[:, 2:3] * d0[DZ])                       # (u0.grad)u0 [B,3]
        rows, vals = [], []
        for i in range(3):                                   # momentum-i
            terms_i = visc + [(DX, u0[:, 0]), (DY, u0[:, 1]), (DZ, u0[:, 2])]
            Ai = basis_v.op_columns(Xi, terms_i, feats=Fi_v)
            row = []
            for j in range(3):
                blk = Ai.copy() if j == i else np.zeros((n_int, basis_v.K))
                # + u_j * d_j(u0_i) coupling
                blk += (d0[GRADS[j]][:, i:i + 1]
                        * basis_v.op_columns(Xi, [(VAL, 1.0)], feats=Fi_v))
                row.append(blk)
            row.append(grad_p_cols[i])
            A = np.hstack(row)
            y = adv0[:, i]
            s = np.abs(A).max()
            rows.append(A / s)
            vals.append(y / s)
        for A, y in blocks_const:
            rows.append(A)
            vals.append(y)
        Amat = np.vstack(rows)
        yvec = np.concatenate(vals)
        del rows, vals
        t_s = time.time()
        theta_new = scipy.linalg.lstsq(Amat, yvec, cond=rcond,
                                       lapack_driver="gelsy",
                                       overwrite_a=True)[0]
        t_solve = time.time() - t_s
        del Amat
        # damping on the nonlinear interior residual
        base, _ = ns_residual_interior(basis_v, basis_p, Xi[:3000], theta, nu)
        alpha = 1.0
        for _ in range(4):
            cand = theta + alpha * (theta_new - theta)
            r, _ = ns_residual_interior(basis_v, basis_p, Xi[:3000], cand, nu)
            if r < base or it == 1:
                break
            alpha *= 0.5
        theta = theta + alpha * (theta_new - theta)
        record(it, dict(phase="newton", alpha=alpha, t_solve=t_solve))
    return basis_v, basis_p, theta, history
