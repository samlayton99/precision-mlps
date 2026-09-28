"""Own-state GD probes that freeze one factor of the effective slope force.

Parameters are flat, in a,b,c,d order. The empirical polynomial basis is fixed;
its orthogonal residual remains in the ordinary remainder in every arm.
Configure JAX_ENABLE_X64=true at launch, as in the existing D34 runners.
"""
from __future__ import annotations

import jax.numpy as jnp

from . import transport

ARMS = ('joint', 'freeze_map', 'clamp_residual')


def _features(p, x):
    width = (p.shape[0]-1)//3
    a, b, c = p[:-1].reshape(3, width)
    u = x[:, None]*a+b
    h = jnp.tanh(u)
    exp = jnp.exp(-2*jnp.abs(u))
    s = 4*exp/(1+exp)**2
    return c, h, s


def _pullback(v, x, c, h, s):
    """Sample residual to flat parameter gradient, with the empirical 1/m."""
    m = x.shape[0]
    return jnp.concatenate((c*((x*v) @ s)/m, c*(v @ s)/m,
                            h.T @ v/m, jnp.mean(v)[None]))


def _modal_jacobian(q, x, c, h, s):
    m = x.shape[0]
    return jnp.concatenate(((q.T @ (x[:, None]*s)/m)*c,
                            (q.T @ s/m)*c, q.T @ h/m,
                            jnp.mean(q, axis=0)[:, None]), axis=1)


def matrices(p, context):
    """Materialize modal Jacobians and the full effective map only on demand."""
    x, q = context['x'], context['q']
    c, h, s = _features(p, x)
    J = _modal_jacobian(q, x, c, h, s)
    JC, JH = J[:2], J[2:]
    C = JC @ JC.T
    B = jnp.linalg.solve(C, JC @ JH.T)
    T = JH.T-JC.T @ B
    return dict(J=J, J_C=JC, J_H=JH, C=C, B=B, T=T,
                T_a=T[:c.shape[0]])


def fork_context(p0, x, y, q=None, degree=65):
    """Capture the fixed basis, target, initial map, and initial fine residual.

    Pass q explicitly when vmapping this function. With q omitted, the existing
    NumPy QR basis builder is used once outside compiled training.
    """
    if q is None:
        q = transport.basis(x, degree)
    p0, x, y, q = map(jnp.asarray, (p0, x, y, q))
    context = dict(x=x, y=y, q=q, p0=p0)
    c, h, _ = _features(p0, x)
    eH0 = q[:, 2:].T @ (h @ c+p0[-1]-y)/x.shape[0]
    return context | dict(T_a0=matrices(p0, context)['T_a'], eH0=eH0)


def field(p, context, arm='joint'):
    """Return applied flat gradient and own-state force channels.

    JIT with arm static, or close over it. Fine Jacobians are never formed here.
    The retained remainder is exactly g_a-F_a at the current arm state, not a
    frozen remainder or one borrowed from the baseline trajectory. No inverse
    regularization is applied; callers must reject coarse_resolved=False.
    """
    if arm not in ARMS:
        raise ValueError(f'Unknown effective feedback arm: {arm}')
    x, y, q = context['x'], context['y'], context['q']
    m = x.shape[0]
    c, h, s = _features(p, x)
    width = c.shape[0]
    residual = h @ c+p[-1]-y
    e = q.T @ residual/m
    JC = _modal_jacobian(q[:, :2], x, c, h, s)
    C = JC @ JC.T
    g = _pullback(residual, x, c, h, s)
    fine_raw = _pullback(q[:, 2:] @ e[2:], x, c, h, s)
    balance = jnp.linalg.solve(C, JC @ fine_raw)
    effective = fine_raw-JC.T @ balance
    zC = e[:2]+balance
    tracking = JC.T @ zC
    omitted = _pullback(residual-q @ e, x, c, h, s)
    current = effective[:width]
    remainder = g[:width]-current
    if arm == 'joint':
        replacement = current
        applied = g
    else:
        if arm == 'freeze_map':
            replacement = context['T_a0'] @ e[2:]
        else:
            initial_raw = _pullback(q[:, 2:] @ context['eH0'], x, c, h, s)
            replacement = (initial_raw-JC.T @ jnp.linalg.solve(C, JC @ initial_raw))[:width]
        applied = g.at[:width].set(remainder+replacement)
    eigen = jnp.linalg.eigvalsh(C)
    resolved = eigen[0] > 64*jnp.finfo(p.dtype).eps*jnp.maximum(1., eigen[-1])
    channels = dict(full_gradient=g, effective_a=current,
        applied_effective_a=replacement, remainder_a=remainder,
        tracking_a=tracking[:width], omitted_a=omitted[:width],
        eH=e[2:], eC=e[:2], zC=zC, C=C,
        loss=jnp.mean(residual**2)/2, residual=residual,
        coarse_min_eigenvalue=eigen[0], coarse_resolved=resolved,
        reconstruction_norm=jnp.linalg.norm(g-effective-tracking-omitted))
    return applied, channels
