"""Fixed-geometry Fourier filtering of the finite raw tanh dictionary.

The exact-arithmetic remainder covers the distant square-wave transition and
the omitted harmonics. The floating-point allowance is a sensitivity estimate,
not an interval enclosure. See docs/gamma_factorized_readout.md.
"""
from __future__ import annotations

import math

import numpy as np


def multiplier(gamma, omega):
    """Fourier multiplier of gamma/2 * sech(gamma*x)**2, with M(0)=1."""
    if gamma <= 0:
        raise ValueError('The common slope must be positive')
    z = np.pi*np.abs(np.asarray(omega, dtype=float))/(2*gamma)
    out = np.ones_like(z)
    nonzero = z > 0
    value = z[nonzero]
    out[nonzero] = 2*value*np.exp(-value)/(-np.expm1(-2*value))
    return out


def geometry(x, centers, harmonics, half_period=8.):
    """Build gamma-independent F and C, retaining the native readout metric.

    Rows of C and columns of F are bias, sin(w1*x), cos(w1*x), ... .
    Neither the target nor gamma is an input to this construction.
    """
    x, centers = np.asarray(x), np.asarray(centers)
    if harmonics < 1:
        raise ValueError('At least one harmonic is required')
    radius = float(max(abs(x.min()-centers.max()), abs(x.max()-centers.min())))
    if half_period <= radius:
        raise ValueError('The half-period must exceed every sample-center displacement')
    odd = np.arange(1, 2*harmonics, 2)
    omega = np.pi*odd/half_period
    coefficient = 4/(np.pi*odd)
    f = np.empty((len(x), 2*harmonics+1))
    f[:, 0] = 1
    f[:, 1::2] = np.sin(x[:, None]*omega)
    f[:, 2::2] = np.cos(x[:, None]*omega)
    f /= np.sqrt(len(x))
    c = np.zeros((2*harmonics+1, len(centers)+1))
    c[0, 0] = 1
    c[1::2, 1:] = coefficient[:, None]*np.cos(omega[:, None]*centers)
    c[2::2, 1:] = -coefficient[:, None]*np.sin(omega[:, None]*centers)
    return dict(f=f, c=c, omega=omega, coefficient=coefficient,
                harmonics=int(harmonics), half_period=float(half_period), radius=radius,
                phase_radius=float(np.max(np.abs(x))+np.max(np.abs(centers))))


def feature_remainder(gamma, harmonics, half_period, radius):
    """Separate uniform errors relative to the nonperiodic tanh features."""
    if gamma <= 0 or harmonics < 1 or half_period <= radius:
        raise ValueError('Require gamma>0, harmonics>=1, and half_period>radius')
    a = np.pi**2/(2*gamma*half_period)
    first_omitted = a*(2*harmonics+1)
    extension = 4*np.exp(-2*gamma*(half_period-radius))
    tail = (4*np.pi/(gamma*half_period)*np.exp(-first_omitted)
            /((-np.expm1(-2*first_omitted))*(-np.expm1(-2*a))))
    return dict(extension=float(extension), tail=float(tail))


def synthesize(g, gamma):
    """Evaluate F D_gamma C with short dot products and pairwise accumulation.

    G=C C^T is retained implicitly; no diagonal or coupling approximation is
    made. Blocking limits summation roundoff without changing the operator.
    """
    m = multiplier(gamma, g['omega'])
    diagonal = np.r_[1., np.repeat(m, 2)]
    f, c = g['f'], g['c']
    block = 128
    partials = []
    blocks = 0
    for start in range(0, len(diagonal), block):
        stop = min(start+block, len(diagonal))
        term = f[:, start:stop]@(diagonal[start:stop, None]*c[start:stop])
        level = 0
        while level < len(partials) and partials[level] is not None:
            term += partials[level]
            partials[level] = None
            level += 1
        if level == len(partials):
            partials.append(term)
        else:
            partials[level] = term
        blocks += 1
    approximate = sum(p for p in partials if p is not None)
    remainder = feature_remainder(gamma, g['harmonics'], g['half_period'], g['radius'])
    width = c.shape[1]-1
    eps = np.finfo(float).eps
    # This propagates a short-dot-product roundoff scale and explicitly exposes
    # an assumed transcendental/phase allowance. It is NOT a rigorous libm bound.
    operations = block+math.ceil(math.log2(blocks))+2
    dot_scale = operations*eps/(1-operations*eps)
    absolute_envelope = np.sum(np.linalg.norm(f, axis=0)*diagonal*np.linalg.norm(c, axis=1))
    phase_envelope = np.sqrt(width)*np.sum(g['coefficient']*m*(1+g['omega']*g['phase_radius']))
    allowance = dot_scale*absolute_envelope+16*eps*phase_envelope
    return approximate, dict(**remainder,
        synthesis_remainder=float(np.sqrt(width)*(remainder['extension']+remainder['tail'])),
        construction_allowance=float(allowance), block_terms=block)
