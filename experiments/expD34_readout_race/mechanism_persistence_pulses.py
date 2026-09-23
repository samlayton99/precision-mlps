"""Physical b/c/d pulses changing intrinsic force amplification at a GD fork.

Slopes are unchanged exactly. Coarse output, coarse disequilibrium, the actual
slope gradient, and effective-force squared norm are matched to first order.
Only the initial state changes; subsequent training must use ordinary GD.
Numerical preparation and tests are run through the campaign's CPU allocation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_run as ar, effective_feedback as ef
from . import mechanism_persistence_kernel as kernel
from .run import write_json

AMPLITUDES = (0., .01, -.01, .005, -.005, .0025, -.0025)


def _matched_values(p, x, y):
    state = kernel.decomposition(p, x, y)
    w = (p.size-1)//3
    return jnp.concatenate((state['z'], state['g'][:w],
                            jnp.reshape(jnp.vdot(state['F'], state['F']), (1,))))


@jax.jit
def observables(p, x, y):
    """Actual own-state matching values plus intrinsic and loaded rates."""
    state = kernel.decomposition(p, x, y)
    w = (p.size-1)//3
    a, b, c = p[:-1].reshape(3, w)
    residual = jnp.tanh(x[:, None]*a+b)@c+p[-1]-y
    centered = x-jnp.mean(x)
    coarse = jnp.stack((jnp.mean(residual),
                        jnp.mean(centered*residual)/jnp.sqrt(jnp.mean(centered**2))))
    return dict(coarse=coarse, z=state['z'], slope_gradient=state['g'][:w],
                JC=state['JC'], F=state['F'], R=state['R'],
                diagnostics=kernel.diagnostics(p, x, y))


@jax.jit
def _derivatives(p, x, y):
    jacobian = jax.jacrev(_matched_values)(p, x, y)
    objective = jax.grad(lambda point: kernel.diagnostics(point, x, y)['k_pure'])(p)
    return jacobian, objective


def local_system(p, x, y):
    """Linear constraints and derivative of k_pure at the unperturbed fork."""
    point, xx, yy = map(jnp.asarray, (p, x, y))
    base = jax.tree.map(np.asarray, observables(point, xx, yy))
    eigen = np.linalg.eigvalsh(base['JC']@base['JC'].T)
    tolerance = 64*np.finfo(float).eps*max(1., eigen[-1])
    if eigen[0] <= tolerance:
        raise ValueError('Numerically unresolved coarse Gram matrix')
    if not np.isfinite(base['diagnostics']['k_pure']):
        raise ValueError('Intrinsic amplification is undefined at the fork')
    derivative, objective = map(np.asarray, _derivatives(point, xx, yy))
    if not np.all(np.isfinite(derivative)) or not np.all(np.isfinite(objective)):
        raise ValueError('Nonfinite matching or amplification derivative')
    w = (len(p)-1)//3
    return dict(base=base, constraints=np.vstack((base['JC'], derivative)),
                objective_gradient=objective, Dz=derivative[:2],
                Dga=derivative[2:2+w], grad_force_squared=derivative[-1],
                coarse_min=float(eigen[0]))


def matched_direction(p, system):
    """Maximize Dk_pure on the constrained block-scaled unit RMS sphere.

The block RMS scales, row normalization, SVD rank threshold, projected-gain
threshold, and full-coordinate RMS normalization match mechanism_pulses.
The slope coordinates are eliminated before the solve, then filled with zeros.
"""
    p = np.asarray(p, dtype=float)
    w = (len(p)-1)//3
    fallback = np.sqrt(np.mean(p*p))
    if not np.isfinite(fallback) or fallback == 0:
        raise ValueError('Zero or nonfinite state has no relative pulse scale')
    scales = np.empty_like(p)
    for block in (slice(0, w), slice(w, 2*w), slice(2*w, 3*w), slice(3*w, None)):
        rms = np.sqrt(np.mean(p[block]**2))
        scales[block] = rms if rms > 0 else fallback
    active = slice(w, None)
    matrix = np.asarray(system['constraints'])[:, active]*scales[active]
    rownorm = np.linalg.norm(matrix, axis=1)
    matrix = matrix[rownorm > 0]/rownorm[rownorm > 0, None]
    if len(matrix):
        _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
        tolerance = np.finfo(float).eps*max(matrix.shape)*singular[0]
        rank = int(np.count_nonzero(singular > tolerance))
        basis = vt[:rank]
    else:
        singular, basis, tolerance, rank = np.empty(0), np.empty((0, len(p)-w)), 0., 0
    objective = scales[active]*np.asarray(system['objective_gradient'])[active]
    projected = objective-basis.T@(basis@objective)
    gain = np.linalg.norm(projected)
    if not np.isfinite(gain) or gain <= 64*np.finfo(float).eps*np.linalg.norm(objective):
        raise ValueError('No numerically resolved amplification direction in constraint nullspace')
    relative = projected/gain*np.sqrt(len(p))
    direction = np.zeros_like(p)
    direction[active] = scales[active]*relative
    info = dict(rank=rank, nullity=int(len(p)-w-rank), rank_tolerance=float(tolerance),
        singular_min_retained=float(singular[rank-1]) if rank else None,
        normalized_constraint_residual=float(np.linalg.norm(matrix@relative)),
        constraint_residual_norm=float(np.linalg.norm(system['constraints']@direction)),
        intrinsic_k_derivative=float(system['objective_gradient']@direction),
        projected_objective_norm=float(gain), objective_norm=float(np.linalg.norm(objective)),
        scaled_rms=float(np.sqrt(np.mean((direction/scales)**2))),
        slope_direction_norm=float(np.linalg.norm(direction[:w])))
    return direction, info


def _scalar(value):
    value = float(value)
    return value if np.isfinite(value) else None


def matching_record(p, direction, amplitude, x, y, baseline):
    pulse = p+amplitude*direction
    state = jax.tree.map(np.asarray, observables(jnp.asarray(pulse), jnp.asarray(x), jnp.asarray(y)))
    d0, d1 = baseline['diagnostics'], state['diagnostics']
    w = (len(p)-1)//3
    eigen = np.linalg.eigvalsh(state['JC']@state['JC'].T)
    return dict(amplitude=float(amplitude), slopes_unchanged=bool(np.array_equal(pulse[:w], p[:w])),
        coarse_change=_scalar(np.linalg.norm(state['coarse']-baseline['coarse'])),
        z_change=_scalar(np.linalg.norm(state['z']-baseline['z'])),
        slope_gradient_change=_scalar(np.linalg.norm(state['slope_gradient']-baseline['slope_gradient'])),
        force_squared_change=_scalar(abs(d1['q_squared']-d0['q_squared'])),
        intrinsic_k_change=_scalar(d1['k_pure']-d0['k_pure']),
        diagnostics={key: _scalar(d1[key]) for key in
            ('q_squared', 'k_pure', 'k', 'D', 'C', 'C_generated', 'C_target', 'loaded')},
        tracking_norm=_scalar(np.linalg.norm(state['R'])),
        coarse_min=_scalar(eigen[0]),
        coarse_resolved=bool(eigen[0] > 64*np.finfo(float).eps*max(1., eigen[-1])),
        finite=bool(np.all(np.isfinite(state['F'])) and np.isfinite(d1['k_pure'])))


def build_family(p, x, y):
    """Return physical states and metadata; unresolved cases retain a baseline.

    Metadata's ``direction`` is a NumPy vector. All other fields are JSON-safe.
    Positive amplitudes maximize the first derivative of intrinsic k, not its
    finite-amplitude change, whose actual value is retained in ``pulses``.
    """
    p = np.asarray(p)
    try:
        system = local_system(p, x, y)
        direction, info = matched_direction(p, system)
    except ValueError as error:
        return p[None, :].copy(), dict(status='unresolved', reason=str(error),
            direction=np.zeros_like(p), amplitudes=(0.,), pulses=[], halving_ratios={})
    matches = [matching_record(p, direction, amplitude, x, y, system['base'])
               for amplitude in AMPLITUDES]
    ratios = {}
    for sign in (1, -1):
        selected = [m for m in matches if sign*m['amplitude'] > 0]
        for key in ('coarse_change', 'z_change', 'slope_gradient_change', 'force_squared_change'):
            values = [m[key] for m in selected]
            ratios[f'{key}_{"positive" if sign > 0 else "negative"}'] = [
                a/b if a is not None and b is not None and b > 0 else None
                for a, b in zip(values, values[1:])]
    return p+np.asarray(AMPLITUDES)[:, None]*direction, dict(status='resolved',
        direction=direction, amplitudes=AMPLITUDES, direction_diagnostics=info,
        coarse_min=system['coarse_min'], pulses=matches, halving_ratios=ratios)


def prepare(inputpack, out, seed=None, targets=None):
    """Write ordinary-GD inputs, including a baseline for every supplied case.

Unresolved directions are recorded rather than replaced by arbitrary pulses;
their unmodified baselines remain in the output. Resolved cases each contribute
the baseline and six paired physical pulses. Evaluation grids pass through.
"""
    inputpack, out = Path(inputpack), Path(out)
    pp, x, yy, cases = ef.load_inputs(inputpack)
    if isinstance(targets, str):
        targets = {name.strip() for name in targets.split(',') if name.strip()}
    elif targets is not None:
        targets = set(targets)
    selected = [i for i, case in enumerate(cases)
                if (seed is None or int(case['seed']) == seed)
                and (targets is None or case['target'] in targets)]
    if not selected:
        raise ValueError('No source cases match the seed/target selection')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    diagnostics, states, labels, records, directions, baseline_indices, source_indices = [], [], [], [], [], [], []
    with np.load(inputpack) as archive:
        evaluation = {key: archive[key].copy() for key in ('x_eval', 'y_eval') if key in archive}
    for ci in selected:
        p, y, case = pp[ci], yy[ci], cases[ci]
        base_index = len(states)
        family, family_info = build_family(p, x, y)
        direction = family_info.pop('direction')
        amplitudes = family_info['amplitudes']
        item = dict(family_info, case=case, source_index=ci, reference_baseline_index=base_index)
        diagnostics.append(item)
        for amplitude, pulse in zip(amplitudes, family):
            states.append(pulse); labels.append(y); directions.append(direction)
            source_indices.append(ci); baseline_indices.append(base_index)
            records.append(dict(case, source_index=ci, amplitude=amplitude,
                pulse_status=item['status'], reference_baseline_index=base_index,
                pulse_objective='increase_intrinsic_k' if amplitude > 0 else
                                'decrease_intrinsic_k' if amplitude < 0 else 'baseline'))
        print(json.dumps(item, allow_nan=False), flush=True)
    payload = dict(p=np.asarray(states), x=x, y=np.asarray(labels),
        cases=np.array(json.dumps(records)),
        sources=np.array(json.dumps({str(inputpack): ef.digest(inputpack)})))
    if 'x_eval' in evaluation:
        payload['x_eval'] = evaluation['x_eval']
    if 'y_eval' in evaluation:
        payload['y_eval'] = evaluation['y_eval'][source_indices]
    ar.atomic_npz(out/'inputs.npz', **payload)
    ar.atomic_npz(out/'directions.npz', direction=np.asarray(directions),
        source_index=np.asarray(source_indices), reference_baseline=np.asarray(baseline_indices))
    write_json(out/'diagnostics.json', diagnostics)
    manifest = dict(input_sha256=ef.digest(inputpack), source_sha256=ef.digest(__file__),
        kernel_sha256=ef.digest(kernel.__file__), pulse_input_sha256=ef.digest(out/'inputs.npz'),
        direction_sha256=ef.digest(out/'directions.npz'), amplitudes=AMPLITUDES,
        source_cases=len(selected), available_source_cases=len(cases),
        selected_source_indices=selected,
        selection=dict(seed=seed, targets=sorted(targets) if targets is not None else None),
        resolved_cases=sum(d['status'] == 'resolved' for d in diagnostics),
        retained_baselines=len(selected), output_cases=len(records), coordinate_order='a,b,c,d',
        force_space='full empirical complement of constant and linear modes',
        constraints=['delta_a=0 exactly', 'JC delta=0', 'DzC delta=0',
                     'Dg_a delta=0', 'D||F||² delta=0'],
        objective='k_pure=-(D+C)/||F||²; positive amplitude increases it to first order',
        amplitude_units='RMS over all coordinates of delta divided by fork block RMS',
        optimizer='ordinary GD after the initial physical perturbation',
        issued_utc=datetime.now(timezone.utc).isoformat(), numerical_certificate=False)
    write_json(out/'manifest.json', manifest)
    return manifest


def _comparison(actual, predicted):
    norm = np.linalg.norm(actual)
    predicted_norm = np.linalg.norm(predicted)
    error = np.linalg.norm(predicted-actual)
    return dict(actual_norm=_scalar(norm), predicted_norm=_scalar(predicted_norm),
        error_norm=_scalar(error), relative_error=_scalar(error/norm) if norm > 0 else None,
        alignment=_scalar(actual@predicted/(norm*predicted_norm))
                  if norm > 0 and predicted_norm > 0 else None)


def analyze(predictions, run, out):
    """Compare own-initial-state movement and antisymmetric physical responses.

    The runner's source_index identifies each expanded input row; the retained
    reference_baseline_index identifies its unpulsed input row. No terminal
    readout difference is mistaken for a difference acquired during training.
    """
    predictions, run, out = map(Path, (predictions, run, out))
    if out.exists():
        raise FileExistsError(out)
    with np.load(predictions/'inputs.npz') as saved:
        initial = saved['p'].copy()
        q0 = np.linalg.norm(saved['reference_F0'], axis=1)
        cases = json.loads(str(saved['cases']))
    with np.load(predictions/'forecasts.npz') as saved:
        forecast = {key: saved[key].copy() for key in saved.files}
    width = (initial.shape[1]-1)//3
    by_source = {int(case['source_index']): i for i, case in enumerate(cases)}
    if len(by_source) != len(cases):
        raise ValueError('Physical analysis requires one natural arm per pulse input')
    groups = {}
    for i, case in enumerate(cases):
        if case['arm'] != 'natural':
            raise ValueError('Physical pulse analysis requires ordinary GD')
        baseline_source = int(case['reference_baseline_index'])
        if baseline_source not in by_source:
            raise ValueError('Missing retained unpulsed baseline')
        groups.setdefault(baseline_source, {})[float(case['amplitude'])] = i
    rows, paired = [], []
    for path in sorted((run/'snapshots').glob('*.npz')):
        n = int(path.stem)
        if n not in forecast['horizons']:
            raise ValueError(f'No issued forecast at snapshot {n}')
        hi = list(forecast['horizons']).index(n)
        with np.load(path) as saved:
            actual = {key: saved[key].copy() for key in saved.files}
        movement = actual['p']-initial
        predicted_movement = forecast['amplification'][:, hi]-initial
        gamma_change = np.abs(actual['p'][:, :width])-np.abs(initial[:, :width])
        predicted_gamma = np.abs(forecast['amplification'][:, hi, :width])-np.abs(initial[:, :width])
        q_change = actual['sampled'][:, 0]-q0
        predicted_q_change = forecast['q'][:, hi]-q0
        for baseline_source, family in groups.items():
            base = by_source[baseline_source]
            if float(cases[base]['amplitude']) != 0:
                raise ValueError('Reference baseline has a nonzero pulse amplitude')
            scale = float(cases[base]['h'])
            for amplitude, i in family.items():
                if amplitude == 0:
                    continue
                observed = movement[i]-movement[base]
                predicted = predicted_movement[i]-predicted_movement[base]
                measured_gamma = scale*(gamma_change[i]-gamma_change[base])
                forecast_gamma = scale*(predicted_gamma[i]-predicted_gamma[base])
                row = dict(cases[i], horizon=n,
                    failed=bool(actual['failed'][i] or actual['failed'][base]),
                    forecast_supported=bool(forecast['valid'][i, hi] and forecast['valid'][base, hi]),
                    actual_mean_lambda_change_contrast=_scalar(measured_gamma.mean()),
                    predicted_mean_lambda_change_contrast=_scalar(forecast_gamma.mean()),
                    actual_q_change_contrast=_scalar(q_change[i]-q_change[base]),
                    predicted_q_change_contrast=_scalar(predicted_q_change[i]-predicted_q_change[base]),
                    initial_q_difference=_scalar(q0[i]-q0[base]))
                for label, block in (('slope', slice(0, width)),
                                     ('readout', slice(2*width, 3*width)), ('full', slice(None))):
                    row.update({label+'_'+key: value for key, value in
                                _comparison(observed[block], predicted[block]).items()})
                rows.append(row)
            responses = {}
            for amplitude in (.01, .005, .0025):
                if amplitude not in family or -amplitude not in family:
                    continue
                plus, minus = family[amplitude], family[-amplitude]
                response = (movement[plus]-movement[minus])/(2*amplitude)
                prediction = (predicted_movement[plus]-predicted_movement[minus])/(2*amplitude)
                measured_gamma = scale*(gamma_change[plus]-gamma_change[minus])/(2*amplitude)
                forecast_gamma = scale*(predicted_gamma[plus]-predicted_gamma[minus])/(2*amplitude)
                row = dict(cases[plus], horizon=n,
                    failed=bool(actual['failed'][plus] or actual['failed'][minus]),
                    forecast_supported=bool(forecast['valid'][plus, hi] and forecast['valid'][minus, hi]),
                    actual_mean_lambda_response=_scalar(measured_gamma.mean()),
                    predicted_mean_lambda_response=_scalar(forecast_gamma.mean()),
                    actual_q_change_response=_scalar((q_change[plus]-q_change[minus])/(2*amplitude)),
                    predicted_q_change_response=_scalar((predicted_q_change[plus]-predicted_q_change[minus])/(2*amplitude)),
                    initial_mean_lambda_difference=_scalar(scale*np.mean(
                        np.abs(initial[plus, :width])-np.abs(initial[minus, :width]))))
                for label, block in (('slope', slice(0, width)),
                                     ('readout', slice(2*width, 3*width)), ('full', slice(None))):
                    row.update({label+'_'+key: value for key, value in
                                _comparison(response[block], prediction[block]).items()})
                responses[amplitude] = (response, row)
            for amplitude, (response, row) in responses.items():
                if amplitude/2 in responses:
                    fine, _ = responses[amplitude/2]
                    for label, block in (('slope', slice(0, width)),
                                         ('readout', slice(2*width, 3*width)), ('full', slice(None))):
                        den = np.linalg.norm(fine[block])
                        row[label+'_halving_response_difference'] = _scalar(
                            np.linalg.norm(response[block]-fine[block])/den) if den > 0 else None
                paired.append(row)
    out.mkdir(parents=True)
    for filename, values in (('contrasts.csv', rows), ('paired.csv', paired)):
        if not values:
            continue
        with (out/filename).open('w', newline='') as stream:
            fields = list(dict.fromkeys(key for row in values for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(values)
    manifest = dict(source_sha256=ef.digest(__file__),
        prediction_manifest_sha256=ef.digest(predictions/'manifest.json'),
        run_manifest_sha256=ef.digest(run/'manifest.json'),
        baseline_families=len(groups), pulse_contrasts=len(rows), paired_responses=len(paired),
        movement='Each state minus its own perturbed initial state before taking contrasts',
        paired_response='(positive own-state movement - negative own-state movement)/(2 amplitude)',
        amplitude_consistency='Adjacent-amplitude central-response discrepancy relative to finer response',
        numerical_certificate=False)
    write_json(out/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=('prepare', 'analyze'), default='prepare')
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--targets', help='Comma-separated exact target names')
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.inputs is None:
            parser.error('prepare requires --inputs')
        if not jax.config.x64_enabled:
            raise ValueError('FP64 required')
        prepare(args.inputs, args.output, seed=args.seed, targets=args.targets)
    else:
        if args.predictions is None or args.run is None:
            parser.error('analyze requires --predictions and --run')
        analyze(args.predictions, args.run, args.output)


if __name__ == '__main__':
    main()
