"""Moment-matched evaluation of the original empirical training measure."""
from __future__ import annotations
import numpy as np
from . import targets


def empirical_rule(x, nodes=64):
    """Gaussian rule for the discrete uniform measure on x, not Lebesgue measure."""
    if not 1 < nodes < len(x): raise ValueError('Expected fewer quadrature nodes than samples')
    q, _ = np.linalg.qr(np.polynomial.legendre.legvander(x, nodes-1))
    diagonal = np.sum(x[:, None]*q*q, axis=0)
    off = np.sum(x[:, None]*q[:, 1:]*q[:, :-1], axis=0)
    jacobi = np.diag(diagonal)+np.diag(off, 1)+np.diag(off, -1)
    points, vectors = np.linalg.eigh(jacobi)
    mass = vectors[0]**2
    return points, mass


def analytic_force_remainder(p, empirical_x, nodes, model, active_coefficients=None):
    """Cauchy-series quadrature error in real arithmetic, excluding rounding.

Both exact measures integrate degree < 2*nodes equally. On |z|=3,
|tanh(az+b)| <= max(1,tan(3|a|)) and |sech^2| <= sec^2(3|a|).
"""
    w = (len(p)-1)//3; a, _, c = p[:-1].reshape(3, w); radius = 3.
    angle = radius*np.max(abs(a))
    if angle >= np.pi/2: return np.inf
    h = max(1., np.tan(angle)); s = 1/np.cos(angle)**2
    mapping = targets.polynomial_map(empirical_x)
    polynomials = [np.polynomial.legendre.leg2poly(mapping[:, k]) for k in range(10)]
    qb = np.array([abs(v) @ radius**np.arange(len(v)) for v in polynomials])
    yb = .3*qb[0]+.4*qb[1]+np.sqrt(.75)*qb[9]
    rb = np.sum(abs(c))*h+abs(p[-1])+yb
    tangent = np.r_[abs(c)*radius*s, abs(c)*s, np.full(w, h), 1.]
    factor = 2*radius**(-2*nodes)/(1-1/radius)
    if model == 'full': return factor*rb*np.linalg.norm(tangent)
    columns = [0, 1, 2, 3, 9] if model == 'five_mode' else list(range(10))
    real_tangent = np.r_[abs(c), abs(c), np.ones(w), 1.]
    return factor*(rb*qb[columns].sum()*np.linalg.norm(real_tangent)
                   +(abs(active_coefficients) @ qb[columns])*np.linalg.norm(tangent))
