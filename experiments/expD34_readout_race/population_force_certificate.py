"""FP64 initial-force evaluation of effective-flow Proposition 9.

No future trajectory enters selection. Times are ODE times, not certified GD
updates; quadrature and rank checks are not outward-rounded certification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad


FRACTIONS = (.05, .1, .25, .5, .75, .9)
PREFIX = 'force_certificate_'


def force_time_integral(ratio, zeta, factor):
    """Dimensionless integral; remove its logarithmic endpoint layer."""
    if not all(math.isfinite(v) for v in (ratio, zeta, factor)) or ratio <= 0 or zeta < 0 or factor <= 1:
        raise ValueError('Finite ratio > 0, zeta >= 0, factor > 1 required.')
    span = factor - 1
    linear = 6 * math.sqrt(2) + 4 * zeta
    leading = np.logaddexp(0., math.log(linear) + math.log(span) - math.log(ratio)) / linear

    def regular_part(t):
        if t == 0:
            return 0.
        base = ratio + linear * t
        extra = t*t*(6*math.sqrt(2)+6*zeta+(2*math.sqrt(2)+4*zeta)*t+zeta*t*t)
        return -extra / (base * (base + extra))

    correction, error = quad(regular_part, 0., span, epsabs=1e-11, epsrel=1e-11, limit=100)
    return float(leading + correction), float(error)


def evaluate(row):
    """Return selected summary and all candidates using initial fields only."""
    def packed(value):
        return {PREFIX + key: item for key, item in value.items()}

    def failed(status):
        return packed(dict(status=status)), [packed(dict(status=status, fraction=f)) for f in FRACTIONS]

    keys = ('B', 'fine_norm', 'F_norm', 'M', 'Es', 'coarse_kappa', 'input_variance', 'target_norm')
    try:
        B, Y, F, M, Es, kappa, variance, target = [float(row[key]) for key in keys]
    except (KeyError, TypeError, ValueError):
        return failed('invalid_initial_data')
    if not all(math.isfinite(v) and v >= 0 for v in (B, Y, F, M, Es, kappa, variance, target)):
        return failed('invalid_initial_data')
    if str(row.get('resolved', '')).lower() not in ('true', '1', '1.0'):
        return failed('unresolved_coarse_solve')
    if B == 0 or variance == 0 or target == 0 or kappa == 0:
        return failed('degenerate_initial_data')
    if F == 0:
        stationary = packed(dict(status='stationary_effective_flow', relative_fine_floor=1.,
                                 fine_floor=Y, absolute_relative_floor=Y/target))
        return stationary, [dict(stationary, **{PREFIX+'selected': True})]
    if Y == 0:
        return failed('inconsistent_zero_fine_error')
    P, es, initial_sigma = math.sqrt(M), math.sqrt(Es), math.sqrt(kappa)

    def rank_margin(q):
        travel = B*(q-1)
        HC = math.sqrt(2)+4*(P+travel)
        moment = math.sqrt(variance)*max(es-travel, 0.) / math.sqrt(1+2*(P+travel)**2)-3*(q*B)**3
        return max(moment, initial_sigma-HC*travel), HC

    low, high = 1., 2.
    for _ in range(60):
        middle = (low+high)/2
        if rank_margin(middle)[0] > 0:
            low = middle
        else:
            high = middle
    candidates = []
    for fraction in FRACTIONS:
        q = 1+fraction*(low-1)
        sigma, HC = rank_margin(q)
        result = dict(fraction=fraction, factor=q, rank_margin=sigma, H_C=HC, selected=False)
        if q <= 1 or sigma <= 0:
            candidates.append(dict(result, status='rank_or_scale_unresolved'))
            continue
        try:
            ratio, zeta = F/(Y*B**3), 3*HC*B/(4*sigma)
            integral, error = force_time_integral(ratio, zeta, q)
            time = integral/(Y*B**2)
            span = q-1
            fourth = 1.5*span**2+span**3+.25*span**4
            fifth = 2*span**2+2*span**3+span**4+.2*span**5
            energy = Y*B**4*(ratio*span+2*math.sqrt(2)*fourth+zeta*fifth)
            floor = math.sqrt(max(0., 1-2*energy/Y**2))
            error /= Y*B**2
            if not all(math.isfinite(v) and v >= 0 for v in (time, energy, floor, error)):
                raise ValueError('Unresolved numerical integral or energy allowance.')
        except (ValueError, OverflowError, ZeroDivisionError):
            candidates.append(dict(result, status='numerical_evaluation_failure'))
            continue
        candidates.append(dict(result, status='eligible_half_retention' if floor >= .5 else 'fine_floor_below_half',
                               flow_time=time, eta002_time_units=time/.002, energy_allowance=energy,
                               relative_fine_floor=floor, fine_floor=Y*floor, absolute_relative_floor=Y*floor/target,
                               path=B*span, initial_force_ratio=ratio, compensation_curvature=zeta,
                               quadrature_absolute_error=error))
    eligible = [c for c in candidates if c['status'] == 'eligible_half_retention']
    if eligible:
        selected = max(eligible, key=lambda c: c['flow_time'])
        selected['selected'] = True
        summary = dict(selected, status='effective_flow_initial_force_fp64_audit')
    else:
        summary = dict(status='no_candidate_retains_half_initial_fine_error')
    return packed(summary), [packed(c) for c in candidates]


def write_audit(source, output):
    source, output = Path(source), Path(output)
    if source.resolve() == (output/'states.csv').resolve():
        raise ValueError('Output must not overwrite the source states archive.')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Source contains no states.')
    states, candidates, counts = [], [], {}
    for index, row in enumerate(rows):
        summary, choices = evaluate(row)
        summary[PREFIX+'source_row'] = index
        states.append(dict(row, **summary))
        identity = {key: row[key] for key in ('target', 'width', 'seed', 'step', 'optimizer') if key in row}
        candidates.extend(dict(identity, **choice, **{PREFIX+'source_row': index}) for choice in choices)
        status = summary[PREFIX+'status']
        counts[status] = counts.get(status, 0)+1
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (('states.csv', states), ('candidates.csv', candidates)):
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in data for key in row)))
            writer.writeheader()
            writer.writerows(data)
    manifest = dict(source=str(source), source_sha256=source_hash,
                    helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    rows=len(states), candidates=len(candidates), statuses=counts, fractions=FRACTIONS,
                    selection='longest effective-flow time retaining at least half the initial fine-error norm',
                    scope='initial-data FP64 audit; neither discrete-GD nor interval certification')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(write_audit(args.source, args.output), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
