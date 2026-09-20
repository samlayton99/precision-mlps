"""Maps and numerical formulas for the declared full frozen-gamma sweep."""
from __future__ import annotations

import numpy as np
from . import core


def config():
    return core.config(core.HERE/'full_config.yaml')


def target(x, name, xp=np):
    if name == 'sine_mix_2_6_10':
        return xp.sin(2*xp.pi*x)+.5*xp.sin(6*xp.pi*x)+.25*xp.sin(10*xp.pi*x)
    if name == 'exp_sin_3pi':
        return xp.exp(xp.sin(3*xp.pi*x))
    if name == 'runge_25':
        return 1/(1+25*x*x)
    if name == 'quadratic':
        return xp.sqrt(5.)*x*x
    if name == 'sine_2pi':
        return xp.sqrt(2.)*xp.sin(2*xp.pi*x)
    raise ValueError(name)


def map_spec(g, name):
    """Return column scales and whether scale-before-difference is applied."""
    a = g.alpha.copy()
    neighbor = name.endswith('_neighbor')
    if neighbor:
        a[1:] = np.cumsum(a[1:])
    if name == 'raw':
        scale = np.ones_like(a)
    elif name == 'uniform_sqrt_h':
        scale = np.r_[1., np.full(g.width, np.sqrt(g.h))]
    elif name.startswith('collective'):
        if name == 'collective_ordinary_halo':
            a[1:][g.corrected_halo] = g.ordinary_alpha
        scale = np.sqrt(a)
        if name == 'collective_unscaled_bias':
            scale[0] = 1.
    elif name in ['individual', 'individual_neighbor']:
        scale = a
    else:
        raise ValueError(name)
    return scale, neighbor


def map_matrix(g, name):
    scale, neighbor = map_spec(g, name)
    r = np.diag(scale)
    if neighbor:
        indices = np.arange(1, g.width)
        r[indices+1, indices] = -scale[indices]
    return r


def design_from_physical(a, scale, neighbor):
    if neighbor:
        a = np.column_stack([a[:, 0], a[:, 1:-1]-a[:, 2:], a[:, -1]])
    return a*scale


def decode(theta, scale, neighbor, xp=np):
    """Parameter axis is penultimate; broadcast scales end with (d, 1)."""
    q = theta*scale[..., :, None]
    if not neighbor:
        return q
    return xp.concatenate([q[..., :2, :], q[..., 2:, :]-q[..., 1:-1, :]], axis=-2)


def encode(c, scale, neighbor):
    q = c.copy()
    if neighbor:
        q[..., 1:, :] = np.cumsum(c[..., 1:, :], axis=-2)
    return q/scale[..., :, None]


def initial_physical(g, family, seed):
    xi = np.random.default_rng(seed).standard_normal(g.width)
    multiplier = np.sqrt(g.alpha[1:]) if family == 'sqrt_alpha_xavier' else g.alpha[1:]
    if family not in ['sqrt_alpha_xavier', 'alpha_xavier']:
        raise ValueError(family)
    return np.r_[0., multiplier*np.sqrt(2/(g.width+1))*xi]


def envelopes(gamma, degrees, r):
    e = core.log_feature_envelope(gamma, degrees)
    cap_scale = (r.shape[0]-1)*np.linalg.norm(r, 2)**2
    column_scale = np.sum(np.sum(np.abs(r[1:]), axis=0)**2)
    cap, columns = 2*e+np.log(cap_scale), 2*e+np.log(column_scale)
    beta = np.arcsinh(np.pi/(2*gamma))
    coefficient = 4/np.expm1(beta)*(1/np.sqrt(gamma*gamma+np.pi**2/4)+1/np.pi)
    abbreviated = np.log(cap_scale)+2*np.log(coefficient)-2*beta*np.asarray(degrees)
    return dict(cap=cap, columns=columns, used=np.minimum(cap, columns), abbreviated=abbreviated)


def bound(e, log_access, epsilon, curvature, chi=.5):
    """C2 with arbitrary chi=eta*L in (0,1), in initial-residual units."""
    if epsilon >= 1:
        return dict(k=None, bound=0., log10_bound=None, status='already_reached')
    if not 0 < epsilon < 1 or not 0 < chi < 1:
        raise ValueError('Invalid tolerance or stable-step fraction')
    result = core.bound(e, log_access, epsilon, curvature)
    if result['log10_bound'] is not None:
        result['log10_bound'] += float(np.log10(np.log(2)/(-np.log1p(-chi))))
        value = result['log10_bound']
        result['bound'] = float(np.ceil(10**value)) if value < np.log10(2**53) else None
    return result


def damping(j, u, singular, vh, q, penalty):
    loading = u.T@q
    step = -vh.T@(singular/(singular**2+penalty)*loading)
    residual = q+j@step
    mu = float(np.linalg.norm(j.T@q)**2)
    return step, residual, penalty/(mu+penalty)
