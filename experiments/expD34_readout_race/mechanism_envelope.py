"""Checkpoint-only frozen-effective-map acquisition envelopes.

These are bounds for the frozen model, not certificates for ordinary GD.
Optional supplied tracking bounds add ONLY its two-channel allowance. Run
numerical panel evaluations on the approved remote CPU allocation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import numpy as np

from . import effective_feedback as ef, effective_feedback_kernel as kernel, transport
from .run import write_json

HORIZONS = (1, 1000, 10000, 20000, 50000, 100000, 200000, 1000000, 5000000, 50000000)


def geometric_gain(singular, eta, horizon):
    """sigma*eta*sum(1-eta*sigma**2)**k, with no positive-mode cutoff."""
    s = np.asarray(singular, dtype=float)
    if eta <= 0 or horizon < 0 or int(horizon) != horizon or np.any(s < 0):
        raise ValueError('Positive eta, nonnegative singular values and integer horizon required')
    z = eta*s*s
    if np.any(z > 1):
        raise ValueError('Nonoscillating spectral step condition eta*sigma**2 <= 1 failed')
    gain = np.zeros_like(s)
    positive = s > 0
    regular = positive & (z > 0) & (z < 1)
    gain[regular] = -np.expm1(horizon*np.log1p(-z[regular]))/s[regular]
    gain[positive & (z == 0)] = eta*horizon*s[positive & (z == 0)]
    if horizon:
        gain[z == 1] = 1/s[z == 1]
    return gain


def tracking_amplification(left_a, singular, eta, horizon):
    """Upper-bound sum of row norms for Phi_0,...,Phi_(N-1).

    Row norms increase under the nonoscillating step condition. Dyadic bins
    use each bin's right endpoint, so the upper bound is analytic, not a
    quadrature estimate. Returned values include the outer eta in (20).
    """
    upper = np.zeros(left_a.shape[0])
    start = 1  # Phi_0 is zero.
    while start < horizon:
        stop = min(2*start, horizon)
        gain = geometric_gain(singular, eta, stop-1)
        upper += (stop-start)*np.linalg.norm(left_a*gain, axis=1)
        start = stop
    endpoint = np.linalg.norm(left_a*geometric_gain(singular, eta, max(horizon-1, 0)), axis=1)
    return eta*upper, endpoint


def spectral_envelope(p, T, e, eta, horizons=HORIZONS, h=1/64, threshold=.25,
                      tracking_u=None, tracking_v=None):
    """Pure model envelope plus optional externally supplied uniform tracking bounds.

    tracking_u: absolute coordinate bounds |(r_C)_a,j|, shape (W,).
    tracking_v: absolute bound ||J_H r_C||, scalar.
    Both must hold uniformly over the claimed interval. No values are inferred
    from small checkpoint ratios. Map drift, omitted modes and finite-step
    defects are NOT covered by the optional correction.
    """
    p, T, e = map(lambda value: np.asarray(value, dtype=float), (p, T, e))
    horizons = np.asarray(horizons, dtype=np.int64)
    width = (len(p)-1)//3
    if T.shape != (len(p), len(e)) or h <= 0 or threshold <= 0:
        raise ValueError('Incompatible map dimensions or nonpositive scale parameters')
    if (tracking_u is None) != (tracking_v is None):
        raise ValueError('Supply both tracking channels, or neither')
    left, singular, right = np.linalg.svd(T, full_matrices=False)
    # Keep every positive computed singular value, including tiny ones.
    loading = right@e
    left_a = left[:width]
    arrays = dict(horizons=horizons, singular=singular, loading=loading,
                  left_a=left_a, p0=p, initial_lambda=h*abs(p[:width]),
                  initial_occupied=h*abs(p[:width]) >= threshold)
    values = {key: [] for key in ('pure_prefix_displacement', 'pure_endpoint_a',
              'pure_prefix_lambda_upper', 'remaining_allowance_a',
              'tracking_residual_amplification_upper', 'tracking_kernel_endpoint_norm')}
    if tracking_u is not None:
        u = np.asarray(tracking_u, dtype=float)
        v = float(tracking_v)
        if u.shape != (width,) or not np.all(np.isfinite(u)) or not np.isfinite(v) or np.any(u < 0) or v < 0:
            raise ValueError('Tracking inputs must be finite nonnegative absolute bounds')
        arrays.update(tracking_u=u, tracking_v=np.asarray(v))
        values.update(tracking_only_allowance=[], tracking_corrected_prefix_lambda_upper=[])
    for horizon in horizons:
        gain = geometric_gain(singular, eta, int(horizon))
        displacement = abs(left_a)@(abs(loading)*gain)
        amplification, endpoint = tracking_amplification(left_a, singular, eta, int(horizon))
        upper = h*(abs(p[:width])+displacement)
        values['pure_prefix_displacement'].append(displacement)
        values['pure_endpoint_a'].append(p[:width]-left_a@(loading*gain))
        values['pure_prefix_lambda_upper'].append(upper)
        values['remaining_allowance_a'].append(threshold/h-abs(p[:width])-displacement)
        values['tracking_residual_amplification_upper'].append(amplification)
        values['tracking_kernel_endpoint_norm'].append(endpoint)
        if tracking_u is not None:
            allowance = eta*horizon*u+amplification*v
            values['tracking_only_allowance'].append(allowance)
            values['tracking_corrected_prefix_lambda_upper'].append(upper+h*allowance)
    arrays.update({key: np.asarray(value) for key, value in values.items()})
    # Infinite means no depletion. These times describe e-folding of the
    # discrete scalar residual, not an acquisition deadline.
    z = eta*singular*singular
    decay_steps = np.full_like(singular, np.inf)
    regular = (z > 0) & (z < 1)
    decay_steps[regular] = -1/np.log1p(-z[regular])
    decay_steps[z == 1] = 0
    arrays['residual_efolding_updates'] = decay_steps
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--degree', type=int, default=65)
    parser.add_argument('--eta', type=float, default=.002)
    parser.add_argument('--h', type=float, default=1/64)
    parser.add_argument('--threshold', type=float, default=.25)
    parser.add_argument('--tracking-bounds', type=Path,
                        help='NPZ with u shape(cases,width), v shape(cases); uniform hypotheses supplied externally')
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError('FP64 required')
    pp, x, yy, cases = ef.load_inputs(args.inputs)
    q = transport.basis(x, args.degree)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    supplied = None
    if args.tracking_bounds:
        with np.load(args.tracking_bounds) as source:
            supplied = {key: source[key].copy() for key in ('u', 'v')}
        if supplied['u'].shape != (len(pp), (pp.shape[1]-1)//3) or supplied['v'].shape != (len(pp),):
            raise ValueError('Tracking bounds must align exactly with input cases')
    rows, statuses = [], []
    for index, (p, y, case) in enumerate(zip(pp, yy, cases)):
        context = kernel.fork_context(p, x, y, q=q)
        matrices = kernel.matrices(context['p0'], context)
        eig = np.linalg.eigvalsh(np.asarray(matrices['C']))
        if eig[0] <= 64*np.finfo(float).eps*max(1., eig[-1]):
            statuses.append(dict(index=index, case=case, status='unresolved_coarse'))
            continue
        try:
            result = spectral_envelope(p, np.asarray(matrices['T']), np.asarray(context['eH0']),
                args.eta, h=args.h, threshold=args.threshold,
                tracking_u=None if supplied is None else supplied['u'][index],
                tracking_v=None if supplied is None else supplied['v'][index])
        except ValueError as error:
            statuses.append(dict(index=index, case=case, status='unsupported', reason=str(error)))
            continue
        np.savez_compressed(args.output/f'{index:03d}.npz', **result)
        initial = result['initial_occupied']
        for k, horizon in enumerate(result['horizons']):
            allowed = result['pure_prefix_lambda_upper'][k] >= args.threshold
            row = dict(index=index, case=json.dumps(case, sort_keys=True), horizon=int(horizon),
                time=float(args.eta*horizon), initial_occupied=int(initial.sum()),
                pure_ever_hit_upper=int(allowed.sum()), pure_new_hit_upper=int((allowed & ~initial).sum()),
                pure_excluded_fraction=float(np.mean(~allowed)),
                max_pure_prefix_lambda=float(result['pure_prefix_lambda_upper'][k].max()),
                minimum_remaining_allowance_a=float(result['remaining_allowance_a'][k].min()),
                max_tracking_residual_amplification=float(result['tracking_residual_amplification_upper'][k].max()))
            if supplied is not None:
                corrected = result['tracking_corrected_prefix_lambda_upper'][k] >= args.threshold
                row['tracking_only_corrected_ever_hit_upper'] = int(corrected.sum())
            rows.append(row)
        statuses.append(dict(index=index, case=case, status='evaluated',
            max_eta_sigma_squared=float(args.eta*result['singular'].max(initial=0)**2)))
    if rows:
        with (args.output/'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_json(args.output/'manifest.json', dict(issued_utc=datetime.now(timezone.utc).isoformat(),
        inputs_sha256=ef.digest(args.inputs), source_sha256=ef.digest(__file__),
        tracking_bounds_sha256=None if supplied is None else ef.digest(args.tracking_bounds),
        eta=args.eta, h=args.h, threshold=args.threshold, degree=args.degree, horizons=HORIZONS,
        scope='Pure frozen-effective-map prefix bounds; optional tracking-only conditional correction. Not an ordinary-GD certificate.',
        numerical_certificate=False, cases=statuses))


if __name__ == '__main__':
    main()
