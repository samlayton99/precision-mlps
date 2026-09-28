"""Conditional FP64 I-only effective-flow bounds (Theorem 12 / Corollary 13).

Future force concentration is an assumption. Neither quadrature nor time/eta
certifies discrete GD. No future moments or force-weighted kurtosis are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
from scipy.integrate import IntegrationWarning, quad

PREFIX = 'shape_certificate_'


def force_polynomial(q0, g0, Y, sigma, width, I_star):
    """Ascending coefficients of G(A), using positive binomial increments."""
    s = I_star**.25
    coefficients = np.zeros(7)
    coefficients[0] = g0
    terms = ((3, 2*math.sqrt(2)*Y*s),
             (5, 6*Y/(5*sigma*sigma*s)),
             (6, 2*math.sqrt(2)*Y/(sigma*sigma*math.sqrt(width))))
    for degree, scale in terms:
        for power in range(1, degree+1):
            coefficients[power] += scale*math.comb(degree, power)*q0**(degree-power)*s**power
    return coefficients


def travel_integral(coefficients, travel, tolerance):
    """Integral dA/G; subtract the logarithmic small-initial-force layer."""
    g0, beta0 = coefficients[:2]
    if min(g0, beta0, travel) <= 0:
        raise ValueError('Positive force, growth and travel required.')
    log_ratio = math.log(beta0)+math.log(travel)-math.log(g0)
    leading = float(np.logaddexp(0., log_ratio))/beta0
    remainder_coefficients = coefficients.copy()
    remainder_coefficients[:2] = 0.

    def correction(unit):
        if unit == 0:
            return 0.
        A = unit*travel
        linear = g0+beta0*A
        remainder = np.polynomial.polynomial.polyval(A, remainder_coefficients)
        return -travel*(remainder/(linear+remainder))/linear

    with warnings.catch_warnings():
        warnings.simplefilter('error', IntegrationWarning)
        correction_value, error = quad(correction, 0., 1., epsabs=tolerance,
                                        epsrel=tolerance, limit=200)
    # This is a rounding diagnostic, not an outward-rounded error certificate.
    cancellation_error = 8*np.finfo(float).eps*(abs(leading)+abs(correction_value))
    return leading+correction_value, error+cancellation_error


def evaluate(row, I_star=32.):
    packed = lambda value: {PREFIX+key: item for key, item in value.items()}
    base = dict(I_allowance=I_star, assumptions_certified=False,
                scope='conditional_FP64_effective_flow')

    def failed(status):
        return packed(dict(base, status=status))

    if not math.isfinite(I_star) or I_star <= 0:
        raise ValueError('Require a finite positive I allowance.')
    try:
        W, M, M4, Y, target, target_H = [float(row[key]) for key in
            ('width', 'M', 'M4', 'fine_norm', 'target_norm', 'target_fine_norm')]
        F, concentration, sigma0 = [float(row['reinforcement_'+key]) for key in
            ('F_norm', 'hidden_force_energy_concentration', 'coarse_sigma')]
    except (KeyError, ValueError, TypeError):
        return failed('invalid_initial_data')
    if str(row.get('reinforcement_resolved', row.get('resolved', ''))).lower() not in ('true', '1', '1.0'):
        return failed('unresolved_coarse_solve')
    if not all(math.isfinite(v) and v >= 0 for v in (W, M, M4, Y, target, target_H, F, sigma0)):
        return failed('invalid_initial_data')
    if min(W, target, sigma0) <= 0 or W != int(W):
        return failed('invalid_initial_data')
    if F == 0:
        return failed('stationary_effective_flow')
    if not math.isfinite(concentration) or concentration <= 0 or min(M4, Y) <= 0:
        return failed('invalid_initial_force_shape')
    base['initial_I'] = concentration
    if concentration > I_star:
        return failed('initial_concentration_outside_allowance')
    C = math.sqrt(2)+4*math.sqrt(M)
    travel = sigma0/(math.sqrt(C*C+4*sigma0)+C)
    sigma, s, q0 = sigma0/2, I_star**.25, M4**.25
    qend = q0+s*travel
    coefficients = force_polynomial(q0, W*F, Y, sigma, W, I_star)
    try:
        time1, error1 = travel_integral(coefficients, travel, 1e-9)
        time, error = travel_integral(coefficients, travel, 1e-11)
        energy = sum(value*travel**(power+1)/(power+1) for power, value in enumerate(coefficients))
        # Same endpoint qend; explicit comparison has its own time.
        explicit_time = W*(-math.expm1(-2*math.log1p(s*travel/q0)))/(6*Y*s*s*q0*q0)
        explicit_exponent = 3*q0**4*math.expm1(4*math.log1p(s*travel/q0))/(4*Y*W)
    except IntegrationWarning:
        return failed('quadrature_integration_warning')
    except (ValueError, OverflowError, FloatingPointError):
        return failed('numerical_evaluation_failure')
    if not all(math.isfinite(v) and v >= 0 for v in (time, time1, error, error1, energy, explicit_time, explicit_exponent)) or min(time, time1) == 0:
        return failed('numerical_evaluation_failure')
    difference = abs(time-time1)/time
    relative_error = max(error/time, error1/time1)
    capacity = 2*math.sqrt(2)*qend**4/(3*W)
    floor = max(0., target_H-capacity)
    fine_energy = max(0., Y*Y-2*energy/W)
    status = ('quadrature_tolerance_disagreement' if difference > 1e-6 else
              'quadrature_error_too_large' if relative_error > 1e-6 else
              'conditional_FP64_effective_flow')
    base.update(status=status,
                flow_time=W*time, explicit_flow_time=explicit_time, rescaled_time=time,
                path=travel, rank_margin=sigma, M4_bound=qend**4,
                Omega_bound=s*s*qend*qend, force_bound=np.polynomial.polynomial.polyval(travel, coefficients)/W,
                nonlinear_output_capacity=capacity, output_error_floor=floor,
                relative_output_error_floor=floor/target,
                fine_energy_floor=fine_energy, fine_error_floor=math.sqrt(fine_energy),
                relative_fine_retention=math.sqrt(fine_energy)/Y,
                explicit_relative_fine_retention=math.exp(-explicit_exponent),
                time_relative_tolerance_difference=difference,
                quadrature_relative_error_estimate=relative_error,
                quadrature_time_error=W*error, check_quadrature_time_error=W*error1,
                quadrature_tolerance=1e-11, check_quadrature_tolerance=1e-9)
    try:
        eta = float(row.get('eta', 'nan'))
    except (TypeError, ValueError):
        eta = math.nan
    if math.isfinite(eta) and eta > 0:
        base['nominal_time_over_eta_not_GD_certificate'] = W*time/eta
        base['explicit_nominal_time_over_eta_not_GD_certificate'] = explicit_time/eta
    return packed(base)


def write_audit(source, output, I_star=32.):
    source, output = Path(source), Path(output)
    if source.resolve() == (output/'states.csv').resolve():
        raise ValueError('Output must not overwrite source.')
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Source has no states.')
    results = [dict(row, **evaluate(row, I_star)) for row in rows]
    output.mkdir(parents=True, exist_ok=True)
    with (output/'states.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in results for key in row)))
        writer.writeheader()
        writer.writerows(results)
    counts = {}
    for row in results:
        status = row[PREFIX+'status']
        counts[status] = counts.get(status, 0)+1
    manifest = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    rows=len(rows), statuses=counts, I_allowance=I_star, assumptions_certified=False,
                    scope='I-only conditional FP64 effective flow; no GD or interval certificate')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--I-star', type=float, default=32.)
    args = parser.parse_args()
    print(json.dumps(write_audit(args.source, args.output, args.I_star), indent=2))


if __name__ == '__main__':
    main()
