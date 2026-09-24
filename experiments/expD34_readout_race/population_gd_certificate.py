"""FP64 evaluation of Proposition 5's initial-data ordinary-GD enclosure.

No trajectory information enters the recurrence. Numerical values are checked
floating-point evaluations, not outward-rounded interval certificates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


TOLERANCES = (.1, .01, .001, .0001, .000001)


def constants(row, q, eta):
    B = float(row['B'])
    M = float(row['M'])
    Es = float(row['Es'])
    variance = float(row['input_variance'])
    norm = float(row['target_norm'])
    initial_fine = float(row['fine_norm'])
    residual = float(row['relative_l2'])*norm
    Z = float(row['tracking_z_norm'])
    if not all(math.isfinite(x) and x >= 0 for x in (B, M, Es, variance, norm, initial_fine, residual, Z)):
        return dict(status='nonfinite_or_negative_initial_data')
    if B == 0 or variance == 0 or norm == 0:
        return dict(status='degenerate_initial_data')
    A = (q-1)*B
    b = q*B
    P = math.sqrt(M)+A
    J = math.sqrt(1+2*P*P)
    HC = math.sqrt(2)+4*P
    sigma_moment = math.sqrt(variance)*max(0., math.sqrt(Es)-A)/J-3*b**3
    sigma_travel = math.sqrt(max(0., float(row['coarse_kappa'])))-HC*A
    sigma = max(sigma_moment, sigma_travel)
    result = dict(q=q, eta=eta, B_star=b, A_star=A, sigma=sigma,
                  J_star=J, initial_z=Z, initial_fine=initial_fine, target_norm=norm,
                  sigma_moment=sigma_moment, sigma_travel=sigma_travel)
    if sigma <= 0:
        return dict(result, status='coarse_margin_nonpositive')
    Yseg = residual+J*A
    D = 6*HC*Yseg*b**3/sigma**2+(9*b**6+6*math.sqrt(2)*Yseg*b*b)/sigma
    u = 3*residual*b**3
    descent = eta*(J*J+Yseg*HC)
    fine_step = 9*eta*b**6
    result.update(H_C=HC, Y_segment=Yseg, D_ell=D, u_star=u,
                  descent_product=descent, fine_step_product=fine_step,
                  tracking_stability_rate=sigma*sigma-D*J)
    if descent > 1:
        return dict(result, status='descent_step_condition_fails')
    if fine_step > 1:
        return dict(result, status='fine_step_condition_fails')
    return dict(result, status='eligible')


def evaluate(row, q, eta=.002, max_steps=600000):
    c = constants(row, q, eta)
    if c['status'] != 'eligible':
        return dict(c, accepted_updates=0)
    A, Z, floor = 0., c['initial_z'], c['initial_fine']
    initial_relative = floor/c['target_norm']
    last_positive = 0
    half_horizon = 0
    retained_half_floor = initial_relative
    horizons = {f'error_above_{tol:g}_through_update': 0 if initial_relative > tol else -1
                for tol in TOLERANCES}
    checkpoints = {}
    status = 'budget_reached'
    for n in range(max_steps):
        speed = c['u_star']+c['J_star']*Z
        next_A = A+eta*speed
        if not math.isfinite(next_A) or next_A >= c['A_star']:
            status = 'travel_envelope_exhausted'
            break
        next_Z = ((1-eta*c['sigma']**2)*Z+eta*c['D_ell']*speed
                  +.5*c['H_C']*eta*eta*speed*speed)
        next_floor = max(0., (1-c['fine_step_product'])*floor
                         -3*eta*c['B_star']**3*c['J_star']*Z
                         -3*math.sqrt(2)*eta*eta*c['B_star']**2*speed*speed)
        if not math.isfinite(next_Z):
            status = 'floating_point_overflow'
            break
        A, Z, floor = next_A, next_Z, next_floor
        relative = floor/c['target_norm']
        if floor > 0:
            last_positive = n+1
        if relative >= .5*initial_relative:
            half_horizon = n+1
            retained_half_floor = relative
        for tol in TOLERANCES:
            if relative > tol:
                horizons[f'error_above_{tol:g}_through_update'] = n+1
        if n+1 in (1, 10, 100, 1000, 10000, 20000, 100000, 600000):
            checkpoints[f'error_floor_at_{n+1}'] = relative
    else:
        n = max_steps
    accepted = n if status != 'budget_reached' else max_steps
    return dict(c, status=status, accepted_updates=accepted, flow_time_units=eta*accepted,
                error_floor_final=floor/c['target_norm'], travel_upper=A, z_upper=Z,
                last_positive_error_floor_update=last_positive,
                half_initial_fine_error_through_update=half_horizon,
                half_initial_retained_floor=retained_half_floor, **horizons, **checkpoints)


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    with args.source.open() as handle:
        states = list(csv.DictReader(handle))
    result, best = [], []
    for i, row in enumerate(states):
        if row.get('status') not in (None, '', 'finite'):
            continue
        identity = {k: v for k, v in row.items() if k in
                    ('target', 'seed', 'width', 'panel', 'role', 'horizon', 'state_sha256', 'source', 'index', 'arm')}
        candidates = []
        for q in (1.1, 1.25, 1.5, 2.):
            values = evaluate(row, q, args.eta, args.max_steps)
            out = dict(input_row=i, **identity, **values)
            candidates.append(out)
            result.append(out)
        winner = max(candidates, key=lambda r: (r.get('half_initial_fine_error_through_update', 0),
                                                r['accepted_updates'], r.get('error_floor_final', 0.)))
        best.append(dict(winner, selection='maximum half-initial-fine-error horizon; then accepted updates'))
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output/'candidates.csv', result)
    write_csv(args.output/'best.csv', best)
    (args.output/'manifest.json').write_text(json.dumps(dict(
        source=str(args.source), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        eta=args.eta, max_steps=args.max_steps, input_rows=len(states), audited_rows=len(best),
        provenance='Initial-data recurrence only; FP64 evaluation, not interval arithmetic.'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eta', type=float, default=.002)
    parser.add_argument('--max-steps', type=int, default=600000)
    run(parser.parse_args())
