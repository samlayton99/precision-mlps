"""Finite common-slope Gram identity and independently constructed forecasts."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, svd


def gram(x, centers, gamma, mapping=None):
    x, centers = np.asarray(x), np.asarray(centers)
    phi = np.tanh(gamma*(x[:, None]-centers))
    means = np.mean(phi, axis=0)
    delta = gamma*(centers[None, :]-centers[:, None])
    close = np.abs(delta) < 1e-3
    safe = np.where(close, 1., delta)
    hidden = 1-(means[:, None]-means[None, :])/np.tanh(safe)
    # Repeated and nearly repeated centers use direct stable inner products.
    for j, k in zip(*np.where(close)):
        hidden[j, k] = np.mean(phi[:, j]*phi[:, k])
    h = np.empty((len(centers)+1, len(centers)+1))
    h[0, 0] = 1.
    h[0, 1:] = h[1:, 0] = means
    h[1:, 1:] = (hidden+hidden.T)/2
    return h if mapping is None else mapping.T@h@mapping


def rectangular_forecast(j, y, eta=None):
    """FP64 measured-spectrum reference, with explicit perpendicular residual."""
    u, singular, _ = svd(j, full_matrices=False, lapack_driver='gesdd')
    norm = np.linalg.norm(y, axis=0)
    if y.ndim == 1:
        y = y[:, None]
        norm = np.atleast_1d(norm)
    loading = u.T@y
    floor = np.sum((y-u@loading)**2, axis=0)/norm**2
    return dict(rates=singular**2*(.5/singular[0]**2 if eta is None else eta),
                weights=loading**2/norm**2, floor=floor,
                L=float(singular[0]**2))


def gram_forecast(h, correlations, norm_sq, eta):
    """Gram-only prediction. Roundoff uncertainty is reported, not hidden.

    Small eigenvalues cannot be safely resolved from a FP64 Gram matrix.
    Omitted target energy is an unresolved floor, not an exact nullspace.
    """
    eigenvalues, vectors = eigh(h)
    threshold = 32*np.finfo(float).eps*len(h)*max(np.linalg.norm(h, 2), 1.)
    keep = eigenvalues > threshold
    corr = np.asarray(correlations)
    if corr.ndim == 1:
        corr = corr[:, None]
    norm_sq = np.atleast_1d(norm_sq)
    weights = (vectors[:, keep].T@corr)**2/eigenvalues[keep, None]/norm_sq
    remainder = 1-np.sum(weights, axis=0)
    return dict(rates=eta*eigenvalues[keep], weights=weights,
                floor=np.maximum(remainder, 0.), unresolved_mass=remainder,
                eigenvalue_threshold=threshold, min_eigenvalue=float(eigenvalues[0]))


def error(model, step):
    rates = np.asarray(model['rates'])
    if np.any(rates < 0) or np.any(rates >= 1):
        raise ValueError('Forecast requires 0<=eta*eigenvalue<1')
    decay = np.exp(2*float(step)*np.log1p(-rates))
    return np.sqrt(model['floor']+decay@model['weights'])


def first_hit(model, epsilon=.01, cap=10**18):
    """Integer crossings; None means unresolved/unattainable within this model."""
    answers = []
    for column in range(model['weights'].shape[1]):
        if error(model, 0)[column] <= epsilon:
            answers.append(0)
            continue
        if np.sqrt(model['floor'][column]) >= epsilon:
            answers.append(None)
            continue
        lo, hi = 0, 1
        while hi < cap and error(model, hi)[column] > epsilon:
            hi = min(2*hi, cap)
        if error(model, hi)[column] > epsilon:
            answers.append(None)
            continue
        while hi-lo > 1:
            mid = (lo+hi)//2
            if error(model, mid)[column] <= epsilon:
                hi = mid
            else:
                lo = mid
        answers.append(hi)
    return answers
