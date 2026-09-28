"""Slope distributions, analytic feature tails, and target-weighted spectra.

These are detached CPU diagnostics for the raw readout. Pole coefficients use
slopes and centers, not measured dictionary eigenvectors or GD trajectories.
The inequalities are exact; their FP64 evaluations are not interval enclosures.
"""
from __future__ import annotations

import numpy as np

from . import core
from .mechanism import forced_slow_mass


def distribution_bound(slopes, degrees, threshold):
    """Bound sum_{i > k+1+q(G)} nu_i by the capped columns' squared tails."""
    slopes = np.abs(np.asarray(slopes, dtype=float))
    degrees = np.asarray(degrees, dtype=int)
    if threshold < 0 or np.any(degrees < 0):
        raise ValueError('Require a nonnegative threshold and degrees')
    low = slopes <= threshold
    column_sq = np.column_stack([
        np.exp(2*core.log_feature_envelope(g, degrees)) for g in slopes])
    count = int(np.count_nonzero(~low))
    return dict(exceptions=count, rank=degrees+1+count,
                bound=column_sq[:, low].sum(axis=1),
                cap_bound=int(low.sum())*np.exp(2*core.log_feature_envelope(threshold, degrees)))


def mean_spectrum_bound(width, mean_cap, degrees, thresholds):
    """Optimize an eigenvalue envelope using only width and mean absolute slope.

    Each threshold uses q <= floor(W*mean/G) and the safe W*e_k(G)^2
    tail budget. The global trace bound W+1 includes the output bias.
    """
    degrees = np.asarray(degrees, int)
    indices = np.arange(1, width+2)
    upper = np.full(width+1, float(width+1))
    for threshold in thresholds:
        if threshold <= 0:
            raise ValueError('Mean-cap thresholds must be positive')
        count = min(width, int(np.floor(width*mean_cap/threshold)))
        rank = degrees+1+count
        offsets = indices[:, None]-rank
        budget = width*np.exp(2*core.log_feature_envelope(threshold, degrees))
        candidate = np.divide(budget, offsets, out=np.full(offsets.shape, np.inf), where=offsets > 0)
        upper = np.minimum(upper, candidate.min(axis=1))
    return upper


def omitted_pole_tail(slopes, degrees, terms):
    """Uniform degree-k tail of pole pairs ell >= terms (not the full feature).

    Uses v_ell >= 1, rho_ell >= 2*v_ell, rho_ell-1 >= v_ell and a
    first-term-plus-integral bound on sum (2*ell+1)^(-k-2).
    """
    g = np.abs(np.asarray(slopes, dtype=float))
    k = np.asarray(degrees, dtype=float)[:, None]
    if np.any(np.pi*(terms+.5) < g):
        raise ValueError('Omitted poles must have imaginary part at least one')
    safe = np.maximum(g, np.finfo(float).tiny)
    log_r = (np.log(16)+(k+1)*np.log(safe)-(k+2)*np.log(np.pi)
             + np.logaddexp(-(k+2)*np.log(2*terms+1),
                 -(k+1)*np.log(2*terms+1)-np.log(2*(k+1))))
    result = np.exp(log_r)
    result[:, g == 0] = 0
    return result


def pole_coefficients(centers, slopes, degree, terms=128):
    """Signed Chebyshev coefficients n=1..degree from conjugate pole pairs.

    The constant coefficient is unnecessary: every projector includes constants.
    Returns a bound on the finite-pole tail beyond degree, separately from the
    omitted-pole tail. Negative slopes change coefficient signs only.
    """
    centers, slopes = np.broadcast_arrays(centers, slopes)
    g = np.abs(slopes)
    terms = max(terms, int(np.ceil(np.max(g)/np.pi)))
    coefficients = np.zeros((degree, len(g)))
    beyond_degree = np.zeros(len(g))
    active = g > 0
    ga = g[active]
    if len(ga):
        z = centers[active]+1j*np.pi*(np.arange(terms)[:, None]+.5)/ga
        s = np.sqrt(z-1)*np.sqrt(z+1)
        s = np.where(np.abs(z+s) > 1, s, -s)
        w = z+s
        power = 1/s
        for n in range(degree):
            power = power/w
            coefficients[n, active] = -4/ga*np.real(power.sum(axis=0))*np.sign(slopes[active])
        beyond_degree[active] = (4/ga*np.sum(np.abs(w)**(-degree)
            / (np.abs(s)*(np.abs(w)-1)), axis=0))
    return coefficients, beyond_degree, terms


def analytic_access(x, centers, slopes, targets, k_max=128, degree=384, terms=128,
                    raw_qr=None):
    """Center-aware polynomial access, with signed coefficients and remainders.

    No dictionary SVD is used. The transformed polynomial matrix retains sample
    orthogonality and cancellation before Frobenius/directional aggregation.
    A floating-point monitor is added, and recorded separately from the analytic
    truncation remainder. It is not a rigorous rounding-error enclosure.
    """
    x = np.asarray(x)
    slopes = np.broadcast_to(slopes, np.shape(centers)).copy()
    y = np.asarray(targets)
    if y.ndim == 1:
        y = y[:, None]
    if not 0 <= k_max < min(degree, len(x)):
        raise ValueError('Require 0 <= k_max < degree and sample count')
    coefficients, beyond, terms = pole_coefficients(centers, slopes, degree, terms)
    cheb = np.polynomial.chebyshev.chebvander(x, degree)[:, 1:]/np.sqrt(len(x))
    polynomial = cheb@coefficients
    raw_qr = core.polynomial_transform(x, k_max) if raw_qr is None else raw_qr
    jh = core.transform(raw_qr, polynomial)
    yh = core.transform(raw_qr, y)
    tails, directional, frobenius = core.access(yh, jh, k_max)
    degrees = np.arange(k_max+1)
    column_remainder = omitted_pole_tail(slopes, degrees, terms)+beyond
    remainder = np.linalg.norm(column_remainder, axis=1)
    monitor = 64*np.finfo(float).eps*np.sqrt(len(x))*np.linalg.norm(polynomial, 'fro')
    uniform = np.column_stack([np.exp(core.log_feature_envelope(g, degrees)) for g in slopes])
    signed_tail = np.cumsum(np.abs(coefficients)[::-1], axis=0)[::-1]
    signed_tail = signed_tail[degrees]+column_remainder
    column = np.minimum(uniform, signed_tail)
    cap = np.sum(uniform**2, axis=1)
    centered = np.minimum(np.sum(column**2, axis=1),
                          (np.sqrt(frobenius)+remainder+monitor)**2)
    directional_bound = np.minimum(centered[:, None],
        (np.sqrt(directional)+remainder[:, None]+monitor)**2)
    tails[tails < 1e-12] = 0
    return dict(tails=tails, cap=cap, centered=centered,
                directional=directional_bound, remainder=remainder,
                roundoff_monitor=float(monitor), terms=terms, degree=degree,
                centered_resolved=np.sqrt(frobenius) > 10*monitor)


def target_cdf_bound(tails, access, thresholds, curvature):
    """Lower bound on initial-residual mass at normalized eigenvalues <= s."""
    tails = np.asarray(tails)
    access = np.asarray(access)
    thresholds = np.asarray(thresholds)
    if curvature <= 0 or np.any(thresholds < 0) or np.any(np.diff(thresholds) <= 0):
        raise ValueError('Require positive curvature and increasing nonnegative thresholds')
    ratio = np.divide(access[:, None], curvature*thresholds,
                      out=np.full((len(access), len(thresholds)), np.inf), where=thresholds > 0)
    ratio[access == 0] = 0
    values = forced_slow_mass(tails[:, None], ratio)
    return np.maximum.accumulate(np.max(values, axis=0))


def cdf_atoms(thresholds, lower_mass):
    """Worst-case spectral distribution consistent with CDF lower bounds.

    Put each increment at its threshold and all remaining mass at rate one.
    This combines thresholds without counting the same target energy twice.
    """
    s, p = np.asarray(thresholds, float), np.asarray(lower_mass, float)
    if (s.ndim != 1 or p.shape != s.shape or np.any(s < 0) or np.any(s > 1)
            or np.any(np.diff(s) <= 0) or np.any(p < 0) or np.any(p > 1)):
        raise ValueError('Require increasing rates in [0, 1] and CDF values in [0, 1]')
    p = np.maximum.accumulate(p)
    rates = np.r_[s, 1.] if s[-1] < 1 else s.copy()
    mass = np.r_[p, 1.] if s[-1] < 1 else np.r_[p[:-1], 1.]
    return rates, np.diff(np.r_[0., mass])


def cdf_error(steps, rates, weights, chi=.5):
    return float(np.sqrt(np.sum(weights*np.exp(2*steps*np.log1p(-chi*rates)))))


def cdf_time_bound(thresholds, lower_mass, epsilon=.01, chi=.5, cap=10**32):
    """First integer step allowed by the combined CDF error lower bound."""
    if not 0 < chi < 1 or not 0 < epsilon < 1:
        raise ValueError('Require chi and epsilon in (0, 1)')
    rates, weights = cdf_atoms(thresholds, lower_mass)
    if weights[rates == 0].sum() >= epsilon**2:
        return dict(bound=None, log10_bound=None, status='proved_zero_mode_obstruction')
    lo, hi = 0, 1
    while cdf_error(hi, rates, weights, chi) > epsilon and hi < cap:
        lo, hi = hi, min(2*hi, cap)
    if cdf_error(hi, rates, weights, chi) > epsilon:
        return dict(bound=None, log10_bound=float(np.log10(float(cap))), status='above_prediction_cap')
    while hi-lo > 1:
        middle = (lo+hi)//2
        if cdf_error(middle, rates, weights, chi) > epsilon:
            lo = middle
        else:
            hi = middle
    return dict(bound=int(hi) if hi < 2**53 else None,
                log10_bound=float(np.log10(float(hi))), status='fp64_lower_bound_estimate')


def exceptional_target_tails(x, high_features, targets, degrees):
    """Distance from targets to polynomials plus the exceptional feature span.

    High-feature columns use the same sample normalization as targets. A dropped
    numerical singular direction makes that witness unresolved, not certified.
    """
    y = np.asarray(targets)
    if y.ndim == 1:
        y = y[:, None]
    norms = np.linalg.norm(y, axis=0)
    tails, resolved = [], []
    for k in degrees:
        p, _ = np.linalg.qr(np.polynomial.chebyshev.chebvander(x, int(k)))
        remainder = y-p@(p.T@y)
        h = high_features-p@(p.T@high_features)
        if h.shape[1]:
            u, singular, _ = np.linalg.svd(h, full_matrices=False)
            keep = singular > 64*np.finfo(float).eps*np.linalg.norm(high_features)
            resolved.append(bool(np.all(keep)))
            remainder -= u[:, keep]@(u[:, keep].T@remainder)
        else:
            resolved.append(True)
        value = np.linalg.norm(remainder, axis=0)/norms
        value[value < 1e-12] = 0
        tails.append(value)
    return np.asarray(tails), np.asarray(resolved)
