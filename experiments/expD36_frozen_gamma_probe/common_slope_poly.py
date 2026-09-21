"""Polynomial-kernel transfer bounds for a fixed common slope.

The inequalities are exact-arithmetic statements. This module evaluates them
in FP64 and reports an explicit arithmetic sensitivity allowance, not an
interval certificate. See docs/common_slope_polynomial_kernel.md.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dct
from scipy.linalg import svd
from scipy.optimize import minimize_scalar

from . import core


def interpolate(x, centers, gamma, degree):
    """Chebyshev-Lobatto interpolation; constant readout column stays exact."""
    if degree < 1:
        raise ValueError('degree must be positive')
    nodes = np.cos(np.pi*np.arange(degree+1)/degree)
    values = np.tanh(gamma*(nodes[:, None]-np.asarray(centers)))
    coefficients = dct(values, type=1, axis=0)/degree
    coefficients[[0, -1]] *= .5
    # Clenshaw avoids a high-degree empirical orthogonal-polynomial recurrence.
    hidden = np.polynomial.chebyshev.chebval(x, coefficients).T
    j = np.column_stack((np.ones(len(x)), hidden))/np.sqrt(len(x))
    envelope = float(np.exp(core.log_feature_envelope(gamma, np.array([degree]))[0]))
    # The DCT coefficients give ||I_D||_{infinity->infinity} <= 2D.
    truncation = np.sqrt(len(centers))*(1+2*degree)*envelope
    return j, coefficients, truncation


def factor(j, y, eta):
    """Rectangular SVD, with all target components and all computed modes."""
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    u, s, vt = svd(j, full_matrices=False, lapack_driver='gesdd')
    rates = eta*s*s
    if np.any(rates >= 1):
        raise ValueError('The archived step does not contract this approximant')
    norm = np.linalg.norm(y, axis=0)
    loading = u.T@y
    perpendicular = y-u@loading
    orth_error = np.linalg.norm(u.T@u-np.eye(len(s)), 'fro')
    reconstruction = np.linalg.norm(j-(u*s)@vt, 'fro')
    return dict(u=u, singular=s, vt=vt, rates=rates, loading=loading,
                perpendicular=perpendicular, norm=norm,
                weights=(loading/norm)**2,
                floor=np.sum(perpendicular**2, axis=0)/norm**2,
                orthogonality_error=float(orth_error),
                reconstruction_error=float(reconstruction))


def defect(j, approximant, model, eta, synthesis_error, arithmetic_error=0.):
    """Two independent upper estimates on the telescoping transfer error.

    The analytic route uses only the feature remainder. The action route also
    evaluates the original synthesis matrix, never a GD trajectory or its SVD.
    The latter is a numerical operator diagnostic, not a gamma-only bound.
    """
    u, s = model['u'], model['singular']
    difference = j-approximant

    def apply(values):
        # K-Ktilde = (J-Jtilde) Jtilde^T + J (J-Jtilde)^T;
        # avoid subtracting two large, nearly equal kernel products.
        return difference@(approximant.T@values)+j@(difference.T@values)

    action = np.linalg.norm(apply(u), axis=0)
    null_action = np.linalg.norm(apply(model['perpendicular']), axis=0)/model['norm']
    delta = synthesis_error+arithmetic_error
    analytic_kernel = (2*s[0]+delta)*delta
    arithmetic_kernel = (2*s[0]+arithmetic_error)*arithmetic_error
    return dict(eta=float(eta), analytic_kernel=float(analytic_kernel),
                arithmetic_kernel=float(arithmetic_kernel),
                mode_action=action, null_action=null_action,
                measured_synthesis_frobenius=float(np.linalg.norm(difference, 'fro')))


def error(model, steps):
    decay = np.exp(2*steps*np.log1p(-model['rates']))
    return np.sqrt(model['floor']+decay@model['weights'])


def transfer(model, bounds, steps, method='combined'):
    analytic = steps*bounds['eta']*bounds['analytic_kernel']
    if method == 'analytic':
        return np.full(len(model['norm']), analytic)
    rates = model['rates']
    sums = np.full_like(rates, float(steps))
    nonzero = rates > 0
    sums[nonzero] = -np.expm1(steps*np.log1p(-rates[nonzero]))/rates[nonzero]
    action = bounds['eta']*(
        (sums*bounds['mode_action'])@np.abs(model['loading']/model['norm'])
        +steps*bounds['null_action'])
    action += steps*bounds['eta']*bounds['arithmetic_kernel']
    if method == 'action':
        return action
    if method != 'combined':
        raise ValueError(method)
    return np.minimum(analytic, action)


def band(model, bounds, steps, method='combined'):
    center = error(model, steps)
    radius = transfer(model, bounds, steps, method)
    return np.maximum(0., center-radius), np.minimum(1., center+radius)


def crossing_bracket(model, bounds, target, epsilon=.01, method='combined', cap=10**18):
    """Necessary time and a verified sufficient-time witness, if found.

    The lower curve is nonincreasing. The upper curve need not be: first find
    a witness below epsilon on a log grid and with local minimization. Any
    returned upper endpoint is checked, even if an earlier pocket was missed.
    """
    lower = lambda n: float(band(model, bounds, int(n), method)[0][target])
    upper = lambda n: float(band(model, bounds, int(n), method)[1][target])
    if upper(0) <= epsilon:
        return dict(necessary=0, sufficient=0, status='already_reached')
    lo, hi = 0, 1
    while hi < cap and lower(hi) > epsilon:
        lo, hi = hi, min(cap, 2*hi)
    if lower(hi) > epsilon:
        return dict(necessary=cap+1, sufficient=None, status='beyond_search_cap')
    while hi-lo > 1:
        mid = (hi+lo)//2
        if lower(mid) > epsilon:
            lo = mid
        else:
            hi = mid
    necessary = hi  # E_(hi-1)>epsilon rules out every earlier true iterate.
    candidates = np.unique(np.maximum(necessary, np.geomspace(max(1, necessary), cap, 160).astype(np.int64)))
    values = [upper(n) for n in candidates]
    witnesses = [int(n) for n, e in zip(candidates, values) if e <= epsilon]
    if not witnesses:
        for i in range(1, len(candidates)-1):
            if values[i] <= values[i-1] and values[i] <= values[i+1] and values[i] < 1:
                fit = minimize_scalar(lambda t: upper(round(np.exp(t))),
                    bounds=(np.log(candidates[i-1]), np.log(candidates[i+1])), method='bounded')
                n = int(round(np.exp(fit.x)))
                if upper(n) <= epsilon:
                    witnesses.append(n)
    if not witnesses:
        return dict(necessary=necessary, sufficient=None, status='no_upper_witness')
    lo, hi = max(0, necessary-1), min(witnesses)
    # This keeps a true upper witness; it does not assume global monotonicity.
    while hi-lo > 1:
        mid = (hi+lo)//2
        if upper(mid) <= epsilon:
            hi = mid
        else:
            lo = mid
    assert upper(hi) <= epsilon
    return dict(necessary=necessary, sufficient=hi, status='fp64_estimate')
