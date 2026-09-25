"""Directional reinforcement identities; no parameter Hessian is materialized.

All derivatives use physical Euclidean gradient-flow time. F includes coarse
compensation; R is the separate coarse disequilibrium force. Numerical callers
must enable FP64 and use Modal, as in the existing population-flow campaign.
"""
from functools import partial

import jax
import jax.numpy as jnp

from . import mechanism_persistence_kernel as kernel
from .population_feedback_flow import rk4


def loaded_operator(p, x, y, v):
    """B v = J_H^* J_H v + <e_H - Q_C ell, D²f> v."""
    state = kernel.decomposition(p, x, y)
    a, b, c = p[:-1].reshape(3, -1)
    va, vb, vc = v[:-1].reshape(3, -1)
    h = jnp.tanh(x[:, None]*a+b)
    s = 1-h*h
    du = x[:, None]*va+vb
    load = state['eH']-state['basis']@state['balance']
    dh = -2*c*h*s*du+vc*s
    curvature = jnp.concatenate((jnp.mean(load[:, None]*dh*x[:, None], axis=0),
                                  jnp.mean(load[:, None]*dh, axis=0),
                                  jnp.mean(load[:, None]*s*du, axis=0),
                                  jnp.zeros(1, dtype=p.dtype)))
    jhv = kernel._fine(state['J']@v, state['basis'])
    return state['J'].T@jhv/len(x)+curvature


def effective_rate(p, x, y):
    f = kernel.effective(p, x, y)
    q = jnp.linalg.norm(f)
    u = f/q
    return -u@loaded_operator(p, x, y, u)


@jax.jit
def diagnostics(p, x, y):
    state = kernel.decomposition(p, x, y)
    f, r = state['F'], state['R']
    q = jnp.linalg.norm(f)
    u = f/q  # q=0 is explicitly unresolved in the scalar output.
    dfu = jax.jvp(lambda z: kernel.effective(z, x, y), (p,), (u,))[1]
    dfr = jax.jvp(lambda z: kernel.effective(z, x, y), (p,), (r,))[1]
    bu = loaded_operator(p, x, y, u)
    kappa = -u@bu
    udot = -dfu+u*(u@dfu)
    rotation = -2*udot@bu
    dbu = jax.jvp(lambda z: loaded_operator(z, x, y, u), (p,), (u,))[1]
    coefficient = u@dbu
    state_change = q*coefficient
    udot_r = (-dfr+u*(u@dfr))/q
    db_r = jax.jvp(lambda z: loaded_operator(z, x, y, u), (p,), (r,))[1]
    tracking_kappa_drift = -2*udot_r@bu+u@db_r
    tracking_log_rate = -u@dfr/q
    jhu = kernel._fine(state['J']@u, state['basis'])
    second = kernel._second_output(p, x, u)
    geometry = -jnp.mean(state['eH']*second)
    compensation = state['balance']@(state['basis'].T@second/len(x))
    residual = kernel.output(p, x)-y
    w = (len(p)-1)//3
    return dict(q=q, kappa=kappa, rotation=rotation, coefficient=coefficient,
                state_change=state_change, kappa_dot=rotation+state_change,
                tracking_kappa_drift=tracking_kappa_drift,
                tracking_log_rate=tracking_log_rate, tracking_norm=jnp.linalg.norm(r),
                tracking_energy=jnp.mean(state['eH']*(state['J']@r)),
                relaxation=jnp.mean(jhu*jhu), geometry=geometry,
                compensation=compensation, rate_identity_error=kappa+u@dfu,
                Y2=jnp.mean(state['eH']**2), target_norm=jnp.sqrt(jnp.mean(y*y)),
                relative_error=jnp.sqrt(jnp.mean(residual**2)/jnp.mean(y*y)),
                M=jnp.sum(p[:-1]**2), slope_rms=jnp.linalg.norm(p[:w])/jnp.sqrt(w),
                coarse_min=jnp.linalg.eigvalsh(state['gram'])[0],
                force_resolved=q > 0)


@partial(jax.jit, static_argnames=('steps', 'kind'))
def advance(p, x, y, dt, steps, kind):
    def body(_, p):
        if kind == 'effective':
            return rk4(p, dt, lambda z: kernel.effective(z, x, y))
        return p-dt*kernel.ordinary_gradient(p, x, y)
    return jax.lax.fori_loop(0, steps, body, p)
