"""Exact full-residual force channels and ordinary Adam step attribution.

The auxiliary moment buffers share the real optimizer's second moment. They
account for its realized steps and never change the training update.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from . import targets

BLENDS = dict(blend_m010=-.1, blend_p001=.01, blend_p010=.1, blend_p030=.3)
TARGETS = (*targets.TARGETS, 'mixed_sine', 'localized_sine', 'chirp', 'moment4', *BLENDS)
CONTROLS = ('moment3', 'moment9', 'mixed_sine', 'chirp')
CHANNELS = ('effective', 'tracking', 'unresolved')


def target_values(name, x, mapping):
    if name in targets.TARGETS:
        return targets.values(name, x, mapping)
    q = np.polynomial.legendre.legvander(x, 9) @ mapping
    if name == 'moment4':
        return .3*q[:, 0]+.4*q[:, 1]+np.sqrt(.75)*q[:, 4]
    if name in BLENDS:
        s = BLENDS[name]
        return .3*q[:, 0]+.4*q[:, 1]+np.sqrt(.75)*(s*q[:, 3]+np.sqrt(1-s*s)*q[:, 9])
    if name in ('mixed_sine', 'localized_sine'):
        value = np.sin(2*np.pi*x)+.5*np.sin(6*np.pi*x)+.25*np.sin(14*np.pi*x)
        return value if name == 'mixed_sine' else value*np.exp(-.5*(x/.4)**2)
    if name == 'chirp':
        u = (x+1)/2
        return np.sin(2*np.pi*(u+4*u*u))
    raise ValueError(name)


def data(name, m=2048):
    """Target definitions and normalization always come from the original grid."""
    original = targets.grid(2048)
    mapping = targets.polynomial_map(original)
    scale = (np.sqrt(np.mean(target_values(name, original, mapping)**2))
             if name in ('mixed_sine', 'localized_sine', 'chirp') else 1.)
    x = targets.grid(m)
    return x, target_values(name, x, mapping)/scale, mapping, float(scale)


def unpack(p):
    return p[:-1].reshape(3, -1), p[-1]


def field(p, x, y):
    """Gradient and exact constant/linear Jacobian in raw physical coordinates."""
    (a, b, c), d = unpack(p)
    u = x[:, None]*a+b
    h = jnp.tanh(u)
    exponential = jnp.exp(-2*jnp.abs(u))
    s = 4*exponential/(1+exponential)**2
    r = h @ c+d-y
    m = len(x)
    gradient = jnp.concatenate((c*((r*x) @ s)/m, c*(r @ s)/m, h.T @ r/m, jnp.array([r.mean()])))
    q = jnp.stack((jnp.ones_like(x), x/jnp.sqrt(jnp.mean(x*x))))
    ja = (q @ (x[:, None]*s)/m)*c
    jb = (q @ s/m)*c
    jc = q @ h/m
    coarse_jacobian = jnp.concatenate((ja, jb, jc, q.mean(axis=1)[:, None]), axis=1)
    return gradient, r, coarse_jacobian, q @ r/m


def split(gradient, coarse_jacobian, e_coarse, mobility=None):
    """GD-reference or instantaneous diagonal-mobility balance.

An unresolved solve enters an explicit unknown channel, preserving exact
accounting and ordinary training without silently regularizing the balance.
"""
    j = coarse_jacobian
    mobility = jnp.ones_like(gradient) if mobility is None else mobility
    C = (j*mobility) @ j.T
    eig = jnp.linalg.eigvalsh(C)
    resolved = eig[0] > 64*jnp.finfo(gradient.dtype).eps*jnp.maximum(1., eig[-1])
    fine = gradient-j.T @ e_coarse
    b = jnp.linalg.solve(jnp.where(resolved, C, jnp.eye(2)), j @ (mobility*fine))
    balanced = -j.T @ b
    effective = fine+balanced
    tracking = j.T @ (e_coarse+b)
    channels = jnp.stack((jnp.where(resolved, effective, 0.),
                          jnp.where(resolved, tracking, 0.),
                          jnp.where(resolved, 0., gradient)))
    return channels, dict(fine=fine, balanced=balanced, z=e_coarse+b, C=C,
                          resolved=resolved, min_eigenvalue=eig[0])


def moments(gradient, channels, m, v, channel_m, count, beta1, beta2, epsilon, adaptive):
    """Bias-corrected EMA (including beta1=0) and shared Adam scaling."""
    mn = beta1*m+(1-beta1)*gradient
    vn = beta2*v+(1-beta2)*gradient**2
    cn = beta1*channel_m+(1-beta1)*channels
    correction = 1-beta1**count
    mh = mn/correction
    ch = cn/correction
    vh = vn/(1-beta2**count)
    inverse = jnp.where(adaptive, 1/(jnp.sqrt(vh)+epsilon), jnp.ones_like(vh))
    return mn, vn, cn, mh, ch, inverse
