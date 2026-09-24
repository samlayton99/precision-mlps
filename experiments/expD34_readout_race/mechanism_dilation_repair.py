"""Finite geometry dilation and locally nearest exact coarse-balance repair.

Only c,d are optimized; a,b are locked to s times their archived values.
The objective is physical Euclidean readout/intercept displacement. The nonlinear
constraint is z=0, not merely vanishing affine output residual. Run numerical
preparation/tests in the campaign CPU allocation with JAX_ENABLE_X64=1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_run as ar, effective_feedback as ef
from . import mechanism_persistence_kernel as kernel

MAX_ITERATIONS = 100
MAX_INCREMENT = 1 / 8
MIN_INCREMENT = 1 / 1024
BALANCE_TOLERANCE = 1e-12
STATIONARITY_TOLERANCE = 1e-8
TRACKING_FRACTION = .001


def _constraints(v, geometry, x, y):
    return kernel.decomposition(jnp.concatenate((geometry, v)), x, y)['z']


_jacobian = jax.jit(jax.jacrev(_constraints))


@jax.jit
def _state(v, geometry, x, y):
    p = jnp.concatenate((geometry, v))
    state = kernel.decomposition(p, x, y)
    # Evaluate tracking directly, avoiding g-F cancellation near balance.
    tracking = state['JC'].T @ state['z']
    return dict(z=state['z'], gram=state['gram'], F=state['F'], R=tracking,
                fine_loss=jnp.mean(state['eH']**2)/2,
                coarse=state['basis'].T @ (kernel.output(p, x)-y)/len(x))


def _resolved(matrix):
    if not np.all(np.isfinite(matrix)):
        return False
    eigen = np.linalg.eigvalsh(matrix)
    return bool(eigen[0] > 64*np.finfo(float).eps*max(1., eigen[-1]))


def _evaluate(v, geometry, x, y):
    state = jax.tree.map(np.asarray, _state(v, geometry, x, y))
    if not _resolved(state['gram']):
        raise ValueError('unresolved_coarse_gram')
    if not all(np.all(np.isfinite(value)) for value in state.values()):
        raise ValueError('nonfinite_state')
    return state


def _stationarity(v, reference, jacobian):
    gram = jacobian @ jacobian.T
    if not _resolved(gram):
        raise ValueError('unresolved_constraint_jacobian')
    r = v-reference
    normal = jacobian.T @ np.linalg.solve(gram, jacobian @ r)
    return float(np.linalg.norm(r-normal))


def _stage(v, geometry, reference, x, y, target_rms):
    """Equality SQP with identity objective Hessian and exact l2 merit."""
    v = v.copy()
    penalty = 1.
    for iteration in range(MAX_ITERATIONS + 1):
        state = _evaluate(v, geometry, x, y)
        jacobian = np.asarray(_jacobian(v, geometry, x, y))
        stationary = _stationarity(v, reference, jacobian)
        r = v-reference
        station_scale = max(np.linalg.norm(reference), np.linalg.norm(r), 1e-12)
        balanced = np.linalg.norm(state['z']) <= BALANCE_TOLERANCE*target_rms
        if balanced and stationary <= STATIONARITY_TOLERANCE*station_scale:
            return v, dict(iterations=iteration, stationarity=stationary,
                           balance_norm=float(np.linalg.norm(state['z'])))
        if iteration == MAX_ITERATIONS:
            raise ValueError('iteration_limit')
        gram = jacobian @ jacobian.T
        multipliers = np.linalg.solve(gram, jacobian @ r-state['z'])
        step = -r+jacobian.T @ multipliers
        penalty = max(penalty, 2*np.linalg.norm(multipliers)+1.)
        merit = .5*(r @ r)+penalty*np.linalg.norm(state['z'])
        directional = r @ step-penalty*np.linalg.norm(state['z'])
        if not np.isfinite(directional) or directional >= 0:
            raise ValueError('non_descent_sqp_step')
        accepted = False
        for backtrack in range(30):
            fraction = 2.**(-backtrack)
            trial = v+fraction*step
            try:
                trial_state = _evaluate(trial, geometry, x, y)
            except ValueError:
                continue
            trial_r = trial-reference
            trial_merit = .5*(trial_r @ trial_r)+penalty*np.linalg.norm(trial_state['z'])
            if trial_merit <= merit+1e-4*fraction*directional:
                v, accepted = trial, True
                break
        if not accepted:
            raise ValueError('line_search_failed')
    raise AssertionError('unreachable')


def repair_case(p, x, y, scale, reference='primary'):
    """Return a repaired state and JSON-safe diagnostics, including failures.

    ``primary`` fixes the distance reference at (c0,d0); ``inverse`` fixes it
    at (c0/scale,d0). These are different objectives, not merely solver starts.
    Continuation never updates that reference. A valid result is a feasible
    stationary local repair, not a certified global distance minimum.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError('Repair requires JAX_ENABLE_X64=1')
    p, x, y = (np.asarray(value, dtype=np.float64) for value in (p, x, y))
    if p.ndim != 1 or (p.size-1) % 3 or x.ndim != 1 or y.shape != x.shape:
        raise ValueError('Invalid parameter or grid shape')
    if scale < 1 or not np.isfinite(scale) or reference not in ('primary', 'inverse'):
        raise ValueError('Expected finite scale >= 1 and primary/inverse reference')
    w = (p.size-1)//3
    geometry0 = p[:2*w].copy()
    fallback = p.copy()
    fallback[:2*w] *= scale
    vref = p[2*w:].copy()
    if reference == 'inverse':
        vref[:w] /= scale
    actual_target_rms = float(np.sqrt(np.mean(y*y)))
    target_rms = max(actual_target_rms, 1e-12)
    info = dict(valid=False, scale=float(scale), reference=reference,
                target_rms=actual_target_rms, target_normalizer=target_rms, stages=[], failures=[],
                distance_metric='sum_dc_squared_plus_dd_squared',
                balance_tolerance=BALANCE_TOLERANCE,
                stationarity_tolerance=STATIONARITY_TOLERANCE,
                tracking_fraction_tolerance=TRACKING_FRACTION)
    if not np.isfinite(target_rms):
        info['reason'] = 'nonfinite_target_normalizer'
        return fallback, info
    try:
        base = _evaluate(p[2*w:], geometry0, x, y)
        v, record = _stage(vref, geometry0, vref, x, y, target_rms)
        info['stages'].append(dict(scale=1., **record))
        current, increment = 1., MAX_INCREMENT
        while current < scale:
            next_scale = min(scale, current+increment)
            try:
                candidate, record = _stage(v, next_scale*geometry0, vref, x, y, target_rms)
            except ValueError as error:
                info['failures'].append(dict(scale=float(next_scale), reason=str(error)))
                increment /= 2
                if increment < MIN_INCREMENT:
                    raise ValueError('continuation_failed') from error
                continue
            v, current = candidate, next_scale
            info['stages'].append(dict(scale=float(current), **record))
            increment = min(MAX_INCREMENT, 2*increment)
        repaired = np.concatenate((scale*geometry0, v))
        state = _evaluate(v, scale*geometry0, x, y)
        jacobian = np.asarray(_jacobian(v, scale*geometry0, x, y))
        tracking = float(np.linalg.norm(state['R'][:w]))
        force = float(np.linalg.norm(state['F'][:w]))
        base_force = float(np.linalg.norm(base['F'][:w]))
        force_scale = max(base_force, force)
        valid_tracking = tracking <= TRACKING_FRACTION*force_scale
        displacement = v-vref
        eigen = np.linalg.eigvalsh(state['gram'])
        info.update(valid=bool(valid_tracking),
            reason='accepted' if valid_tracking else 'initial_tracking_contamination',
            balance_norm=float(np.linalg.norm(state['z'])),
            normalized_balance=float(np.linalg.norm(state['z'])/target_rms),
            stationarity=_stationarity(v, vref, jacobian),
            normalized_stationarity=_stationarity(v, vref, jacobian)/max(
                np.linalg.norm(vref), np.linalg.norm(displacement), 1e-12),
            coarse_gram_min=float(eigen[0]), coarse_gram_max=float(eigen[-1]),
            slope_tracking_norm=tracking, slope_effective_norm=force,
            baseline_slope_effective_norm=base_force,
            tracking_to_effective_scale=tracking/force_scale if force_scale else None,
            correction_norm=float(np.linalg.norm(displacement)),
            physical_change_from_base=float(np.linalg.norm(v-p[2*w:])),
            readout_norm_before=float(np.linalg.norm(p[2*w:3*w])),
            readout_norm_after=float(np.linalg.norm(v[:w])),
            readout_slope_product_after=float(repaired[:w] @ v[:w]),
            fine_loss_before=float(base['fine_loss']), fine_loss_after=float(state['fine_loss']),
            coarse_residual_after=state['coarse'].tolist(),
            achieved_slope_rms_ratio=float(scale), geometry_locked=True)
        return (repaired if valid_tracking else fallback), info
    except (ValueError, np.linalg.LinAlgError) as error:
        info['reason'] = str(error)
        return fallback, info


def prepare(input_path, output_path, nref=None, cohort=None):
    """Six records per case, including invalid attempts for the runner to skip."""
    pp, x, yy, cases = ef.load_inputs(input_path)
    records, states, labels, indices = [], [], [], []
    arms = (('original', 1., 'none'), ('repaired', 1., 'primary'),
            ('s125_primary', 1.25, 'primary'), ('s2_primary', 2., 'primary'),
            ('s125_inverse', 1.25, 'inverse'), ('s2_inverse', 2., 'inverse'))
    for index, (p, y, case) in enumerate(zip(pp, yy, cases)):
        case = dict(case)
        width = (len(p)-1)//3
        resolved_nref = nref if nref is not None else case.get('Nref', case.get('nref'))
        if resolved_nref is None:
            resolved_nref = {177: 128, 705: 512, 1409: 1024}.get(width)
        if resolved_nref is None or int(resolved_nref) != resolved_nref or resolved_nref <= 0:
            raise ValueError('Supply --nref for a width without an archived Nref convention')
        if nref is not None and any(key in case and case[key] != nref for key in ('Nref', 'nref')):
            raise ValueError('--nref conflicts with source metadata')
        case['Nref'] = int(resolved_nref)
        expected_h = 2/case['Nref']
        if 'h' in case and not np.isclose(case['h'], expected_h, rtol=1e-12, atol=0):
            raise ValueError('Source h conflicts with 2/Nref')
        case['h'] = expected_h
        case['cohort'] = cohort if cohort is not None else case.get('cohort', Path(input_path).stem)
        if 'start' not in case:
            raise ValueError('Source case must identify its start update')
        original_id = case.get('original_case_id', case.get('case_id', index))
        for arm, scale, reference in arms:
            if arm == 'original':
                point, repair = p.copy(), dict(valid=bool(np.all(np.isfinite(p))),
                                               reason='unmodified_baseline')
            else:
                point, repair = repair_case(p, x, y, scale, reference)
            record = dict(case, original_case_id=original_id, source_index=index,
                          arm=arm, scale=scale, reference=reference,
                          valid=repair['valid'], repair=repair)
            records.append(record)
            states.append(point)
            labels.append(y)
            indices.append(index)
            print(json.dumps(record, allow_nan=False), flush=True)
    payload = dict(p=np.asarray(states), x=x, y=np.asarray(labels),
                   cases=np.array(json.dumps(records, allow_nan=False)),
                   sources=np.array(json.dumps({str(input_path): ef.digest(input_path)})),
                   code_sources=np.array(json.dumps({
                       'sha256': {Path(path).name: ef.digest(path)
                                  for path in (__file__, kernel.__file__)},
                       'versions': {'jax': jax.__version__, 'numpy': np.__version__}})))
    with np.load(input_path) as archive:
        if 'x_eval' in archive:
            payload['x_eval'] = archive['x_eval'].copy()
        if 'y_eval' in archive:
            payload['y_eval'] = archive['y_eval'][indices].copy()
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ar.atomic_npz(output_path, **payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--nref', type=int)
    parser.add_argument('--cohort')
    args = parser.parse_args()
    prepare(args.input, args.output, args.nref, args.cohort)


if __name__ == '__main__':
    main()
