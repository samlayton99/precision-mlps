"""Initial-data evaluation of the exact-tanh effective-flow persistence theorem.

FP64 evaluations are not certificates. Numerical execution belongs under Slurm.
Ordinary-GD trajectory comparisons require the separate proved GD corollary.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import population_coverage as coverage

MULTIPLIERS = (1.01, 1.025, 1.05, 1.1, 1.25, 1.5, 2.)


def initial_state(p, x, y, h, eta=None):
    p, x, y = (np.asarray(value, dtype=np.float64) for value in (p, x, y))
    w = (len(p)-1)//3
    if len(p) != 3*w+1 or w < 1 or x.ndim != 1 or y.shape != x.shape:
        raise ValueError('Invalid parameter or sample shape')
    if not all(np.all(np.isfinite(value)) for value in (p, x, y)):
        raise ValueError('Nonfinite input')
    if np.max(abs(x)) > 1 or abs(np.mean(x)) > 64*np.finfo(float).eps:
        raise ValueError('The theorem requires centered inputs in [-1,1]')
    ordered = np.sort(x)
    if np.max(abs(ordered+ordered[::-1])) > 64*np.finfo(float).eps:
        raise ValueError('The theorem requires a symmetric empirical input grid')
    v = float(np.mean(x*x))
    if not 0 < v <= 1:
        raise ValueError('Nondegenerate empirical variance required')
    if not np.isfinite(h) or h <= 0:
        raise ValueError('Verified positive construction spacing required')
    if eta is not None and (not np.isfinite(eta) or eta <= 0):
        raise ValueError('Verified positive learning rate required')
    abc = np.sqrt(w)*p[:-1].reshape(3, w)
    radii2 = np.sum(abc*abc, axis=0)
    state = coverage.geometry(p, x, y)
    q = coverage.basis(x)
    ec = q.T@state['residual']/len(x)
    eh = state['residual']-q@ec
    return dict(W=w, h=float(h), eta=eta, R0=float(np.sqrt(radii2.max())),
        M0=float(radii2.mean()), Es0=float(np.mean(abc[0]**2+abc[2]**2)),
        Y0=float(np.sqrt(np.mean(eh*eh))), v=v,
        z0=float(np.linalg.norm(state['z'])), coarse_k0=float(state['kappa']),
        eC0=float(np.linalg.norm(ec)), total_loss0=float(np.mean(state['residual']**2)/2),
        F0=float(np.linalg.norm(state['F'])), Fa0=float(np.linalg.norm(state['F'][:w])),
        Rtracking0=float(np.linalg.norm(state['R'])),
        lambda_max0=float(h*np.max(abs(p[:w]))),
        max_input_abs=float(np.max(abs(x))), input_mean=float(np.mean(x)))


def effective_bounds(initial, radius, time):
    """Evaluate at reduced time T=t/W; all returned slacks must be positive."""
    if time < 0 or radius < 0:
        raise ValueError('Nonnegative time and radius required')
    w, y0 = initial['W'], initial['Y0']
    exponent = 8*y0*radius**2*time
    base = dict(radius=radius, time=time, moment_exponent=exponent,
                support_initial_slack=radius-initial['R0'])
    if exponent > 700:
        return dict(base, valid=False, reason='moment_overflow', gap=-np.inf,
                    support_slack=-np.inf)
    m = initial['M0']*np.exp(exponent)
    es = initial['Es0']*np.exp(-exponent)
    gap = np.sqrt(initial['v']*es/(1+2*m))-3*radius**3/w
    base.update(M_star=m, Es_star=es, gap=gap)
    if gap <= 0 or not np.isfinite(gap):
        return dict(base, valid=False, reason='conditioning', support_slack=-np.inf)
    kappa = gap**2
    b = 4*y0*radius**3*(1+np.sqrt(2*m/kappa))
    ell = (9*radius**6/w+6*np.sqrt(2)*y0*radius**2
           +8*y0*radius**2*np.sqrt(m/kappa)*(np.sqrt(2)+6*np.sqrt(2)*radius**2/w))
    slack = radius-initial['R0']-time*b
    q_exponent = ell*time
    q_bound = initial['F0']*np.exp(q_exponent) if q_exponent < 700 else np.inf
    if initial['F0'] == 0:
        q_bound = 0.
    # Physical time is W*T; integrate the physical parameter speed envelope.
    travel = initial['F0']*w*np.expm1(q_exponent)/ell if ell > 0 and q_exponent < 700 else (initial['F0']*w*time if ell == 0 else np.inf)
    if initial['F0'] == 0:
        travel = 0.
    base.update(kappa=kappa, B=b, L=ell, support_slack=slack,
        force_bound=q_bound, parameter_travel_bound=travel,
        normalized_support_bound=initial['h']*radius/np.sqrt(w),
        valid=bool(slack > 0), reason='valid' if slack > 0 else 'support')
    return base


def evaluate_margin(initial, multiplier):
    if multiplier <= 1:
        raise ValueError('Outer multiplier must exceed one')
    radius = multiplier*initial['R0']
    at_zero = effective_bounds(initial, radius, 0.)
    record = dict(multiplier=multiplier, radius=radius,
        flow='pure_effective_gradient_flow', ordinary_gd_bound=False,
        fp64_certificate=False, **{f'initial_{k}': v for k, v in at_zero.items()})
    if not at_zero['valid']:
        return dict(record, status='no_initial_region', limiting_condition=at_zero['reason'],
                    max_time_lower=0., max_time_upper=0., effective_flow_integer_updates=0)
    if initial['Y0'] == 0:
        return dict(record, status='stationary_effective_flow', max_time_lower=np.inf,
                    max_time_upper=np.inf, limiting_condition='none', effective_flow_integer_updates=None)
    low, high = 0., 1.
    for _ in range(1024):
        if not effective_bounds(initial, radius, high)['valid']:
            break
        low, high = high, high*2
    else:
        raise RuntimeError('Unable to bracket first-exit root')
    for _ in range(96):
        middle = low+(high-low)/2
        if middle in (low, high):
            break
        if effective_bounds(initial, radius, middle)['valid']:
            low = middle
        else:
            high = middle
    limit = effective_bounds(initial, radius, high)
    record.update(status='evaluated', max_time_lower=low, max_time_upper=high,
                  limiting_condition=limit['reason'], effective_flow_integer_updates=None)
    # This count is a clock conversion ONLY, not an ordinary-GD guarantee.
    if initial['eta'] is not None:
        count = max(0, int(np.floor(low*initial['W']/initial['eta'])))
        while count > 0 and not effective_bounds(initial, radius, count*initial['eta']/initial['W'])['valid']:
            count -= 1
        record['effective_flow_integer_updates'] = count
        record.update({f'at_integer_{k}': v for k, v in effective_bounds(initial, radius, count*initial['eta']/initial['W']).items()})
    return record


def verified_eta(case, pack):
    """Read explicit archive metadata; never substitute the campaign default."""
    if case.get('eta') is not None:
        return float(case['eta']), 'archive_case'
    if 'eta' in pack and np.size(pack['eta']) == 1:
        return float(np.asarray(pack['eta']).reshape(-1)[0]), 'archive_scalar'
    return None, 'missing'


def gd_constants(initial, multiplier):
    w, eta = initial['W'], initial['eta']
    r, mbar, slower = multiplier*initial['R0'], multiplier**2*initial['M0'], initial['Es0']/multiplier**2
    result = dict(multiplier=multiplier, radius=r, Mbar=mbar, Sunder=slower,
                  ordinary_gd_bound=True, fp64_certificate=False)
    if eta is None:
        return dict(result, valid=False, reason='missing_eta')
    gap = np.sqrt(initial['v']*slower/(1+2*mbar))-3*r**3/w
    result['gap'] = gap
    if gap <= 0:
        return dict(result, valid=False, reason='conditioning')
    kappa = gap**2
    ybar = np.sqrt(2*initial['total_loss0']); y = 2*ybar
    c = 4*y; j = np.sqrt(1+2*mbar)
    hc = np.sqrt(2)+4*np.sqrt(2)*r*r/w
    jf, hf = 3*r**3, (8+4*np.sqrt(2))*r*r
    f = jf*y
    dell = 2*hc*f/kappa+(hf*y+jf*jf/w)/np.sqrt(kappa)
    b = c*r**3*(1+np.sqrt(2)*np.sqrt(mbar/kappa))
    lf = hf*y+jf*jf/w+2*hc*f/np.sqrt(kappa)
    step_loss = eta*(j*j+2*ybar*hc)
    a = eta*c*r*r/w
    result.update(W=w, eta=eta, kappa=kappa, Ybar=ybar, Y=y, J=j,
        Hc=hc, Jf=jf, Hf=hf, f=f, Dell=dell, B=b, LF=lf, a=a,
        step_loss=step_loss, valid=bool(step_loss <= 1 and a <= 1),
        reason='valid' if step_loss <= 1 and a <= 1 else 'step_size')
    return result


def gd_step(state, constants, force_coupled=False):
    q, m, s, r, u = (state[k] for k in ('q', 'm', 's', 'r', 'u'))
    eta, w, j = constants['eta'], constants['W'], constants['J']
    if force_coupled:
        speed = u+j*q
        return dict(q=(1-eta*constants['kappa'])*q+eta*constants['Dell']/w*speed+eta**2*constants['Hc']/2*speed**2,
            m=m+eta*speed, s=s-eta*speed,
            r=r+eta*(min(constants['B']/w, np.sqrt(w)*u)+np.sqrt(2)*constants['radius']*q),
            u=u+eta*constants['LF']/w*speed)
    speed = constants['f']/w+j*q
    return dict(q=(1-eta*constants['kappa'])*q+eta*constants['Dell']/w*speed+eta**2*constants['Hc']/2*speed**2,
        m=(1+constants['a'])*m+eta*j*q,
        s=(1-constants['a'])*s-eta*j*q,
        r=r+eta*(constants['B']/w+np.sqrt(2)*constants['radius']*q),
        u=(1+eta*constants['LF']/w)*u+eta*constants['LF']*j*q/w)


def gd_exit_reason(state, constants):
    if not all(np.isfinite(value) for value in state.values()):
        return 'nonfinite_recurrence'
    if state['r'] >= constants['radius']:
        return 'support'
    if state['m'] >= np.sqrt(constants['Mbar']):
        return 'upper_moment'
    if state['s'] <= np.sqrt(constants['Sunder']):
        return 'lower_slope_readout_moment'
    return None


def evaluate_gd_margin(initial, multiplier, requested=(), force_coupled=False):
    constants = gd_constants(initial, multiplier)
    result = dict(constants, valid_updates=0,
                  recurrence_variant='force_coupled' if force_coupled else 'primary')
    if not constants['valid']:
        return result, {}
    state = dict(q=initial['z0'], m=np.sqrt(initial['M0']), s=np.sqrt(initial['Es0']), r=initial['R0'], u=initial['F0'])
    records = {0: dict(state)} if 0 in requested else {}
    if constants['a'] == 0 and state['q'] == 0:
        return dict(result, reason='stationary', valid_updates=None), {n: dict(state) for n in requested}
    n = 0
    while True:
        trial = gd_step(state, constants, force_coupled)
        reason = gd_exit_reason(trial, constants)
        if reason is not None:
            break
        n += 1; state = trial
        if n in requested:
            records[n] = dict(state)
    result.update(valid_updates=n, reason=reason,
        **{f'endpoint_{k}': v for k, v in state.items()},
        normalized_support_bound=initial['h']*state['r']/np.sqrt(initial['W']))
    gap = .25-initial['lambda_max0']
    result['ever_fraction_bound_lambda025'] = (min(1., initial['h']**2*(state['m']-np.sqrt(initial['M0']))**2/(initial['W']*gap**2)) if gap > 0 else 1.)
    return result, records


def natural_audit(base, output, hashes, force_coupled=False):
    initials, margin_rows, path_rows, missing, failures = [], [], [], [], []
    expected_panels = [f'feedback_{cohort}_{panel}' for cohort in ('development', 'confirmation') for panel in ('N128', 'N512', 'N1024', 'late')]
    found_cases = 0
    for name in expected_panels:
        prediction = base/'evidence/persistence_1bf7138'/name
        source = prediction/'inputs.npz'; run = prediction.with_name(prediction.name+'_run')
        if not source.exists() or not run.exists():
            missing.append(str(source if not source.exists() else run))
            continue
        manifest_path = prediction/'manifest.json'; run_manifest = run/'manifest.json'
        if not manifest_path.exists() or not run_manifest.exists():
            missing.append(str(manifest_path if not manifest_path.exists() else run_manifest))
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest['input_sha256'] != coverage.digest(source) or json.loads(run_manifest.read_text())['prediction_sha256'] != coverage.digest(manifest_path):
            raise ValueError(f'Archive pairing changed: {prediction}')
        for path in (source, manifest_path, run_manifest):
            hashes[str(path)] = coverage.digest(path)
        pack, cases = coverage.load(source)
        snapshots = sorted((run/'snapshots').glob('*.npz'))
        if not snapshots:
            missing.append(str(run/'snapshots'))
        requested = [int(path.stem) for path in snapshots]
        for n in manifest['horizons']:
            if int(n) not in requested:
                missing.append(str(run/'snapshots'/f'{int(n):09d}.npz'))
        points = {}
        for snapshot in snapshots:
            hashes[str(snapshot)] = coverage.digest(snapshot)
            with np.load(snapshot) as data:
                points[int(snapshot.stem)] = {key: data[key].copy() for key in ('p', 'failed', 'count', 'positive', 'negative', 'first_hit', 'travel') if key in data}
                for counter in ('positive', 'negative', 'first_hit'):
                    if counter not in data:
                        failures.append(dict(panel=prediction.name, horizon=int(snapshot.stem), stage='counter', reason='missing_'+counter))
        for index, case in enumerate(cases):
            if case.get('arm') != 'natural':
                continue
            found_cases += 1
            meta = dict(case, panel=prediction.name, index=index)
            h = coverage.spacing(pack, case, index)
            try:
                initial = initial_state(pack['p'][index], pack['x'], pack['y'][index], h, float(manifest['eta']))
            except ValueError as error:
                failures.append(dict(meta, stage='initial', reason=str(error)))
                continue
            initials.append(dict(meta, **initial, eta_source='verified_prediction_manifest'))
            evaluations = []
            for multiplier in MULTIPLIERS:
                result, envelopes = evaluate_gd_margin(initial, multiplier, requested, force_coupled)
                margin_rows.append(dict(meta, **result)); evaluations.append((result, envelopes))
            chosen, envelopes = max(evaluations, key=lambda pair: np.inf if pair[0].get('valid_updates') is None else pair[0]['valid_updates'])
            for n in requested:
                snapshot = points[n]
                row = dict(meta, horizon=n, actual_updates=int(snapshot['count'][index]), failed=bool(snapshot['failed'][index]),
                           selected_multiplier=chosen['multiplier'], valid_gd_updates=chosen['valid_updates'])
                if row['failed']:
                    row['status'] = 'failed_archived_state'
                    failures.append(dict(meta, horizon=n, stage='snapshot', reason=row['status']))
                    path_rows.append(row); continue
                try:
                    current = initial_state(snapshot['p'][index], pack['x'], pack['y'][index], h, float(manifest['eta']))
                except ValueError as error:
                    row['status'] = 'unresolved_snapshot'
                    failures.append(dict(meta, horizon=n, stage='snapshot', reason=str(error)))
                    path_rows.append(row); continue
                if row['actual_updates'] != n:
                    failures.append(dict(meta, horizon=n, stage='snapshot', reason='update_count_mismatch'))
                row.update({f'observed_{key}': current[key] for key in ('R0', 'M0', 'Es0', 'Y0', 'coarse_k0', 'z0', 'F0', 'total_loss0', 'lambda_max0')})
                bound = envelopes.get(n)
                row['within_valid_gd_interval'] = bound is not None and row['actual_updates'] == n
                if row['within_valid_gd_interval']:
                    row.update({f'bound_{key}': value for key, value in bound.items()})
                    row.update(support_slack=bound['r']-current['R0'], moment_upper_slack=bound['m']**2-current['M0'],
                        moment_lower_slack=current['Es0']-bound['s']**2,
                        tracking_slack=bound['q']-current['z0'], force_slack=bound['u']-current['F0'],
                        conditioning_slack=current['coarse_k0']-chosen['kappa'])
                    gap = .25-initial['lambda_max0']
                    row['ever_fraction_bound_lambda025'] = min(1., h*h*(bound['m']-np.sqrt(initial['M0']))**2/(initial['W']*gap*gap)) if gap > 0 else 1.
                    if 'first_hit' in snapshot:
                        hits = snapshot['first_hit'][index]
                        # Runner encoding: -1 never, 0 initially above, n first hit.
                        if np.any((hits < -1) | (hits > row['actual_updates']) | (hits != np.floor(hits))):
                            failures.append(dict(meta, horizon=n, stage='counter', reason='invalid_first_hit_encoding'))
                        else:
                            row['observed_ever_fraction_lambda025'] = float(np.mean(hits >= 0))
                    # Full Euclidean travel controls abc-label travel but includes d;
                    # do not compare it against the abc-only moment travel bound.
                    row['abc_label_travel_bound'] = bound['m']-np.sqrt(initial['M0'])
                    if 'positive' in snapshot and 'negative' in snapshot:
                        per_particle = np.sqrt(initial['W'])/h*(snapshot['positive'][index]+snapshot['negative'][index])
                        row['observed_label_slope_absolute_travel_rms'] = float(np.sqrt(np.mean(per_particle**2)))
                        row['slope_travel_slack'] = row['abc_label_travel_bound']-row['observed_label_slope_absolute_travel_rms']
                path_rows.append(row)
    coverage.write_csv(output/'natural_initial_states.csv', initials)
    coverage.write_csv(output/'natural_gd_margins.csv', margin_rows)
    coverage.write_csv(output/'natural_path_checks.csv', path_rows)
    coverage.write_csv(output/'natural_failures.csv', failures)
    return dict(natural_cases=len(initials), found_natural_cases=found_cases,
        natural_path_rows=len(path_rows), missing_natural=missing, natural_failures=len(failures),
        natural_panel_complete=bool(found_cases == 40 and len(path_rows) == 320 and not missing and not failures),
        expected_natural_cases=40, expected_natural_path_rows=320,
        first_hit_encoding='-1 never reached lambda=.25; 0 initially above; positive update index first acquisition')


def audit(args):
    args.output.mkdir(parents=True, exist_ok=False)
    inventory_dir = args.base/'evidence/population_coverage_structural_v3'
    inventory = inventory_dir/'static.csv'
    manifest_path = inventory_dir/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    with inventory.open() as stream:
        rows = list(csv.DictReader(stream))
    inputs, margins, gd_margins, failures, best, hashes = [], [], [], [], [], {}
    hashes[str(inventory)] = coverage.digest(inventory)
    hashes[str(manifest_path)] = coverage.digest(manifest_path)
    cached = {}
    for row in rows:
        source = Path(row['source']); index = int(row['index'])
        if source not in cached:
            actual_hash = coverage.digest(source)
            if actual_hash != manifest['sources'][str(source)]:
                raise ValueError(f'Changed structural source: {source}')
            hashes[str(source)] = actual_hash
            cached[source] = coverage.load(source)
        pack, cases = cached[source]
        case = cases[index]
        meta = dict(source=str(source), index=index, state_sha256=row['state_sha256'],
                    target=case.get('target'), seed=case.get('seed'), start=case.get('start'))
        eta, eta_source = verified_eta(case, pack)
        try:
            initial = initial_state(pack['p'][index], pack['x'], pack['y'][index], float(row['h']), eta)
        except ValueError as error:
            failures.append(dict(meta, stage='initial', reason=str(error))); continue
        inputs.append(dict(meta, **initial, eta_source=eta_source))
        selected = []
        for multiplier in MULTIPLIERS:
            value = evaluate_margin(initial, multiplier)
            margins.append(dict(meta, **value)); selected.append(value)
            gd_value, _ = evaluate_gd_margin(initial, multiplier, force_coupled=args.force_coupled)
            gd_margins.append(dict(meta, **gd_value))
        chosen = max(selected, key=lambda value: value['max_time_lower'])
        best.append(dict(meta, **chosen))
    coverage.write_csv(args.output/'initial_states.csv', inputs)
    coverage.write_csv(args.output/'effective_flow_margins.csv', margins)
    coverage.write_csv(args.output/'effective_flow_best.csv', best)
    coverage.write_csv(args.output/'ordinary_gd_margins.csv', gd_margins)
    coverage.write_csv(args.output/'failures.csv', failures)
    natural = natural_audit(args.base, args.output, hashes, args.force_coupled)
    result = dict(source_sha256=coverage.digest(__file__), input_hashes=hashes, **natural,
        inventory_states=len(rows), evaluated_states=len(inputs), failures=len(failures),
        missing_eta_states=sum(row['eta'] is None for row in inputs),
        recurrence_variant='force_coupled' if args.force_coupled else 'primary',
        force_coupled_policy='Use proved recurrence from initial F0; no future force observations enter the envelope',
        dependency_sha256={str(Path(coverage.__file__)): coverage.digest(coverage.__file__)},
        multipliers=MULTIPLIERS, clock='T=eta*N/W', role='FP64 theorem-condition evaluation; GF and ordinary-GD rows separate; no rigorous certificates',
        eta_policy='Explicit archive metadata only; missing rates do not prevent continuous-time evaluation',
        selection='Largest initially computed valid horizon, independent of trajectories')
    (args.output/'manifest.json').write_text(json.dumps(coverage.clean(result), indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--force-coupled', action='store_true', help='Evaluate the separately proved initial-force recurrence; default remains the primary recurrence')
    audit(parser.parse_args())
