"""Initial-data audit of the evolving fine-sensitivity comparison ODE.

Quadrature evaluates an analytic theorem's scalar integral in float64. It is
not interval certification and does not certify ordinary discrete GD.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad


def sensitivity_time_integral(ratio, factor):
    """Integral from 1 to factor of 1/(ratio+2sqrt(2)(u^3-1))."""
    if not math.isfinite(ratio) or not math.isfinite(factor) or ratio <= 0 or factor <= 1:
        raise ValueError('Positive initial sensitivity and factor > 1 required.')
    span = factor - 1
    linear = 6 * math.sqrt(2)
    # Remove the logarithmic endpoint layer before numerical quadrature.
    leading = np.logaddexp(0., math.log(linear) + math.log(span) - math.log(ratio)) / linear

    def regular_part(u):
        if u == 0:
            return 0.
        base = ratio + linear * u
        extra = linear * u * u * (1 + u / 3)
        return -extra / ((base + extra) * base)

    correction, error = quad(regular_part, 0., span, epsabs=1e-11,
                             epsrel=1e-11, limit=100)
    return float(leading + correction), float(error)


def initial_sensitivity_certificate(row):
    """Use the same collective rank alternatives as the sixth-moment audit."""
    B = float(row['B'])
    Y = float(row['fine_norm'])
    j = float(row['fine_jacobian_hs_norm'])
    prefix = 'hs_certificate_'
    required = (B, Y, j, float(row['M']), float(row['Es']),
                float(row['coarse_kappa']), float(row['input_variance']), float(row['target_norm']))
    if not all(math.isfinite(value) and value >= 0 for value in required):
        return {prefix + 'status': 'invalid_initial_data'}
    if float(row['target_norm']) == 0 or float(row['input_variance']) == 0:
        return {prefix + 'status': 'degenerate_initial_data'}
    if not row['resolved'] or B <= 0 or Y <= 0:
        return {prefix + 'status': 'degenerate_or_unresolved'}
    if j == 0:
        return {prefix + 'status': 'stationary_effective_flow',
                prefix + 'relative_fine_floor': 1.}
    if not math.isfinite(j) or j < 0:
        return {prefix + 'status': 'invalid_initial_sensitivity'}
    P = math.sqrt(row['M'])
    es = math.sqrt(row['Es'])
    initial_sigma = math.sqrt(max(0., row['coarse_kappa']))
    variance = float(row['input_variance'])

    def rank_margin(factor):
        travel = B * (factor - 1)
        moment = math.sqrt(variance * max(es - travel, 0.)**2
                           / (1 + 2 * (P + travel)**2)) - 3 * (factor * B)**3
        initial = initial_sigma - (math.sqrt(2) + 4 * (P + travel)) * travel
        return max(moment, initial)

    if rank_margin(1.) <= 0:
        return {prefix + 'status': 'initial_collective_rank_bound_nonpositive'}
    low, high = 1., 2.
    for _ in range(60):
        middle = (low + high) / 2
        if rank_margin(middle) > 0:
            low = middle
        else:
            high = middle
    factor = 1 + .9 * (low - 1)
    if factor <= 1 or B**3 == 0 or Y * B**2 == 0:
        return {prefix + 'status': 'numerically_unresolved_scale'}
    ratio = j / B**3
    if not math.isfinite(ratio):
        return {prefix + 'status': 'numerically_unresolved_scale'}
    integral, error = sensitivity_time_integral(ratio, factor)
    scale = 1 / (Y * B**2)
    # The polynomial uses span to avoid cancellation near factor == 1.
    span = factor - 1
    quartic_difference = 1.5 * span**2 + span**3 + .25 * span**4
    exponent = (j * B * span + 2 * math.sqrt(2) * B**4
                * quartic_difference) / Y
    floor = math.exp(-exponent)
    return {
        prefix + 'status': 'effective_flow_initial_data_fp64_audit',
        prefix + 'factor': factor,
        prefix + 'flow_time': scale * integral,
        prefix + 'eta002_time_units': scale * integral / .002,
        prefix + 'relative_fine_floor': floor,
        prefix + 'absolute_relative_floor': Y * floor / row['target_norm'],
        prefix + 'path': B * span,
        prefix + 'rank_margin': rank_margin(factor),
        prefix + 'quadrature_absolute_error': scale * error,
        prefix + 'initial_sensitivity_ratio': ratio,
    }
