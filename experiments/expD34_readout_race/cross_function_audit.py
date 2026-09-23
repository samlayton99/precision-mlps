"""Audit archived ordinary-GD forecasts; produce evidence, never report prose.

Run numerical work remotely. Inputs are the immutable effective-feedback
archives (or byte-preserving transfer subsets), not new training trajectories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import effective_feedback_kernel as kernel
from . import transport
from .cross_function_budget import spectral_budget
from .cross_function_forces import BLOCK_NAMES, gain_parts, make_analyzer

HORIZONS = (0, 1000, 10000, 50000, 200000)
THRESHOLDS = (1., 3.2, 16.)
COHORTS = {
    'existing': ('existing', 'raw/existing/joint/run'),
    'fresh': ('fresh_all', 'raw/fresh/joint/run'),
    'heldout': ('heldout_all', 'raw/heldout/joint'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(x, y):
    return float(x/y) if y > 0 else np.nan


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def case_key(case):
    return case['target'], int(case['seed']), int(case['start'])


def load_cohort(root, name):
    stem, relative = COHORTS[name]
    pack_path = root/'inputs'/f'{stem}.npz'
    manifest_path = root/'predictions'/stem/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if digest(pack_path) != manifest['input_sha256']:
        raise ValueError(f'Changed input pack: {pack_path}')
    if (manifest.get('eta', .002) != .002 or manifest.get('degree', 65) != 65
            or any(r['eta'] != .002 or r['degree'] != 65 for r in manifest['records'])):
        raise ValueError('Forecast uses a different step or basis')
    with np.load(pack_path) as z:
        pack = {k: z[k].copy() for k in ('p', 'x', 'y')}
        cases = json.loads(str(z['cases']))
    run = root/relative
    run_manifest = json.loads((run/'manifest.json').read_text())
    status = json.loads((run/'status.json').read_text())
    if not status['complete'] or not status['valid']:
        raise ValueError(f'Incomplete ordinary-GD source: {run}')
    if ([case_key(c) for c in run_manifest['cases']] != [case_key(c) for c in cases]
            or run_manifest['arm'] != 'joint'):
        raise ValueError('Run and input cases differ')
    if run_manifest['eta'] != .002 or run_manifest['degree'] != 65:
        raise ValueError('This audit preserves the primary step and basis')
    if run_manifest['input_sha256'] != digest(pack_path):
        raise ValueError('Run and analysis inputs differ')
    if tuple(run_manifest['thresholds']) != THRESHOLDS:
        raise ValueError('Stored first-hit threshold order differs')
    snapshots = {}
    paths = [pack_path, manifest_path, run/'manifest.json', run/'status.json']
    for n in HORIZONS:
        path = run/'snapshots'/f'{n:09d}.npz'
        with np.load(path) as z:
            snapshots[n] = {k: z[k].copy() for k in z.files}
        data = snapshots[n]
        if int(data['offset']) != n or np.any(data['count'] != n) or np.any(data['failed']):
            raise ValueError(f'Invalid update count or failed state: {path}')
        paths.append(path)
    if not np.array_equal(pack['p'], snapshots[0]['p']):
        raise ValueError('Initial snapshot differs from input pack')
    records = {case_key(r['case']): r for r in manifest['records']}
    if len(records) != len(cases) or set(records) != {case_key(c) for c in cases}:
        raise ValueError('Forecast coverage differs from input coverage')
    return pack, cases, snapshots, manifest_path, records, paths


def analyze(root, output, cohort, limit=None):
    if not jax.config.x64_enabled:
        raise ValueError('Set JAX_ENABLE_X64=true')
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Run the numerical audit in a remote Slurm allocation')
    if any(d.platform != 'cpu' for d in jax.devices()):
        raise ValueError('This postprocessing audit requests CPU resources only')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'complete.json').exists():
        raise ValueError('Use a fresh output directory; completed audits are immutable')
    begun = time.monotonic()
    pack, cases, snapshots, pred_root, records, paths = load_cohort(root, cohort)
    q = jnp.asarray(transport.basis(pack['x'], 65))
    x = jnp.asarray(pack['x'])
    evaluate = jax.jit(make_analyzer())
    get_gains = jax.jit(gain_parts)
    force_rows, budget_rows, modal_rows = [], [], []
    vector_rows = []
    provenance = {str(p.relative_to(root)): digest(p) for p in paths}
    max_errors = dict(reconstruction=0., defect=0., derivative=0., schur=0.,
                      projection=0., issued_spectral=0., signed_travel=0., travel=0.)
    selected = cases if limit is None else cases[:limit]
    for index, case in enumerate(selected):
        p0 = pack['p'][index]
        width = (len(p0)-1)//3
        record = records[case_key(case)]
        forecast_path = pred_root.parent/record['file']
        if digest(forecast_path) != record['sha256']:
            raise ValueError(f'Changed forecast archive: {forecast_path}')
        provenance[str(forecast_path.relative_to(root))] = record['sha256']
        with np.load(forecast_path) as z:
            needed = ('steps', 'p0', 'effective_pure_p', 'effective_pure_eH',
                      'effective_pure_supported', 'joint_p', 'joint_supported')
            forecast = {k: z[k].copy() for k in needed}
        if not np.array_equal(forecast['p0'], p0):
            raise ValueError('Forecast initial parameters do not match')
        context = kernel.fork_context(jnp.asarray(p0), x, jnp.asarray(pack['y'][index]), q=q)
        D0, C0 = get_gains(jnp.asarray(p0), context)
        T0 = np.asarray(D0-C0)
        e0 = np.asarray(context['eH0'])
        budget = spectral_budget(p0, T0, e0, .002, HORIZONS, THRESHOLDS)
        identity = dict(cohort=cohort, target=case['target'], family=case.get('family', 'original'),
                        seed=int(case['seed']), start=int(case['start']))
        for hi, n in enumerate(HORIZONS):
            fi = int(np.flatnonzero(forecast['steps'] == n)[0])
            if not forecast['effective_pure_supported'][fi]:
                raise ValueError('Unsupported primary fixed-map forecast')
            state = snapshots[n]
            p = state['p'][index]
            result = jax.device_get(evaluate(jnp.asarray(p), context, D0, C0,
                                            jnp.asarray(forecast['effective_pure_eH'][fi])))
            if not bool(result['coarse_resolved']):
                raise ValueError(f'Unresolved coarse inverse: {identity}, horizon {n}')
            row = identity | dict(horizon=n)
            row.update({k: float(v) for k, v in result.items() if np.ndim(v) == 0})
            actual_move = p[:width]-p0[:width]
            fixed_error = forecast['effective_pure_p'][fi, :width]-p[:width]
            affine_error = forecast['joint_p'][fi, :width]-p[:width]
            row.update(
                actual_slope_displacement_norm=float(np.linalg.norm(actual_move)),
                actual_slope_path=float(state['path'][index]),
                fixed_motion_error=ratio(np.linalg.norm(fixed_error), np.linalg.norm(actual_move)),
                affine_supported=bool(forecast['joint_supported'][fi]),
                affine_motion_error=(ratio(np.linalg.norm(affine_error), np.linalg.norm(actual_move))
                                     if forecast['joint_supported'][fi] else np.nan),
                actual_mean_gamma=float(np.mean(abs(p[:width]))),
                actual_mean_gamma_change=float(np.mean(abs(p[:width])-abs(p0[:width]))),
                actual_max_gamma=float(np.max(abs(p[:width]))),
                actual_positive_travel=float(np.mean(state['positive'][index])),
                actual_relative_mse=float(state['metric_relative_mse'][index]),
            )
            for name in ('tracking', 'omitted', 'remainder'):
                row[name+'_relative_slope'] = ratio(row[name+'_slope_norm'], row['effective_slope_norm'])
            row['tracking_relative_residual'] = ratio(
                np.linalg.norm(result['fine_tracking_forcing']),
                np.linalg.norm(result['fine_effective_forcing']))
            row['omitted_relative_residual'] = ratio(
                np.linalg.norm(result['fine_omitted_forcing']),
                np.linalg.norm(result['fine_effective_forcing']))
            total_defects = sum(row[k+'_norm'] for k in (
                'map_defect_a', 'residual_defect_a', 'remainder_defect_a'))
            row['force_defect_cancellation'] = ratio(row['forecast_defect_a_norm'], total_defects)
            row['two_term_cancellation'] = ratio(row['effective_slope_norm'],
                row['direct_slope_norm']+row['balanced_slope_norm'])
            for name in ('map_direct_defect_a', 'map_balanced_defect_a', 'map_defect_a',
                         'residual_defect_a', 'remainder_defect_a'):
                row[name+'_relative_effective'] = ratio(row[name+'_norm'], row['effective_slope_norm'])
            for bi, block in enumerate(BLOCK_NAMES):
                for kind in ('direct', 'balanced', 'gain'):
                    for stat in ('norm', 'signed'):
                        row[f'{block}_{kind}_derivative_{stat}'] = float(
                            result[f'block_{kind}_derivative_{stat}'][bi])
            full_dot = result['full_force_derivative_a']
            F = result['effective'][:width]
            for kind in ('direct', 'balanced', 'gain'):
                vectors = result[f'block_{kind}_derivative_a']
                for bi, block in enumerate(BLOCK_NAMES):
                    row[f'{block}_{kind}_force_energy_rate'] = float(F @ vectors[bi])
            row['residual_force_energy_rate'] = float(F @ result['residual_force_derivative_a'])
            row['total_force_energy_rate'] = float(F @ full_dot)
            row['gain_derivative_norm'] = float(np.linalg.norm(result['block_gain_derivative_a'].sum(axis=0)))
            row['residual_derivative_norm'] = float(np.linalg.norm(result['residual_force_derivative_a']))
            force_rows.append(row)
            # Retain vectors needed to inspect cancellation and all modal signs;
            # full gain matrices are reproducible from the archived parameters.
            vector_rows.append({k: np.asarray(v) for k, v in result.items()
                                if np.ndim(v) == 1 or k.startswith('block_') and k.endswith('_a')})
            for mi, e in enumerate(result['eH']):
                mr = identity | dict(horizon=n, mode=mi+2, residual=float(e))
                for kind in ('direct', 'balanced', 'effective'):
                    mr[kind+'_A'] = float(result[f'modal_{kind}_A'][mi])
                    mr[kind+'_velocity'] = float(result[f'modal_{kind}_velocity'][mi])
                mr['tracking_residual_forcing'] = float(result['fine_tracking_forcing'][mi])
                mr['omitted_residual_forcing'] = float(result['fine_omitted_forcing'][mi])
                mr['effective_residual_forcing'] = float(result['fine_effective_forcing'][mi])
                mr['tracking_to_effective_residual_ratio'] = ratio(
                    abs(mr['tracking_residual_forcing']), abs(mr['effective_residual_forcing']))
                modal_rows.append(mr)
            signed = sum(state[k][index] for k in ('effective', 'tracking', 'omitted', 'crossing'))
            change = abs(p[:width])-abs(p0[:width])
            checks = dict(
                reconstruction=float(result['reconstruction_norm'])/(1+np.linalg.norm(result['gradient'])),
                defect=float(np.linalg.norm(result['defect_identity']))/(1+row['forecast_defect_a_norm']),
                derivative=float(np.linalg.norm(result['derivative_identity']))/(1+np.linalg.norm(full_dot)),
                schur=float(result['Schur_identity_norm']),
                projection=float(result['coarse_projector_identity_norm']),
                signed_travel=float(np.max(abs(signed-change))),
                travel=float(np.max(abs(state['positive'][index]-state['negative'][index]-change))),
            )
            if budget['applicable']:
                pred_difference = p0+budget['displacement'][hi]-forecast['effective_pure_p'][fi]
                checks['issued_spectral'] = float(np.linalg.norm(pred_difference))/(1+budget['displacement_norm'][hi])
            for k, v in checks.items():
                if not np.isfinite(v):
                    raise ValueError(f'Nonfinite {k} identity: {identity}, horizon {n}')
                max_errors[k] = max(max_errors[k], v)
            for ti, threshold in enumerate(THRESHOLDS):
                hits = state['first_hit'][index, ti]
                br = identity | dict(horizon=n, threshold=threshold, applicable=budget['applicable'],
                    initial_count=int(budget['initial_occupancy'][ti]),
                    predicted_simultaneous_upper=int(budget['simultaneous_occupancy_upper'][hi, ti]),
                    predicted_ever_upper=int(budget['ever_occupancy_upper'][hi, ti]),
                    actual_current_count=int(np.count_nonzero(abs(p[:width]) >= threshold)),
                    actual_ever_count=int(np.count_nonzero(hits >= 0)),
                    actual_new_ever_count=int(np.count_nonzero(hits > 0)),
                    full_displacement_budget=float(budget['displacement_norm'][hi]),
                    predicted_mean_travel_upper=float(np.mean(budget['slope_travel_upper'][hi])),
                    actual_mean_positive_travel=float(np.mean(state['positive'][index])),
                )
                budget_rows.append(br)
        print(json.dumps(identity | dict(done=index+1, total=len(selected), seconds=time.monotonic()-begun)), flush=True)
    write_csv(output/'forces.csv', force_rows)
    write_csv(output/'budgets.csv', budget_rows)
    write_csv(output/'modes.csv', modal_rows)
    np.savez_compressed(output/'vectors.npz', **{
        key: np.stack([r[key] for r in vector_rows]) for key in vector_rows[0]},
        cases=np.array(json.dumps([{k: r[k] for k in ('cohort', 'target', 'seed', 'start', 'horizon')}
                                  for r in force_rows])))
    checks_ok = all(v < 1e-8 for v in max_errors.values())
    record = dict(cohort=cohort, cases=len(selected), states=len(force_rows), modes=len(modal_rows),
                  complete=True, verification_passed=checks_ok, maximum_identity_errors=max_errors,
                  horizons=HORIZONS, thresholds=THRESHOLDS, seconds=time.monotonic()-begun,
                  jax_version=jax.__version__, numpy_version=np.__version__,
                  devices=[str(d) for d in jax.devices()], slurm_job=os.environ['SLURM_JOB_ID'],
                  sources=provenance, source_hashes={str(p): digest(p)
                    for p in Path('experiments/expD34_readout_race').glob('cross_function_*.py')},
                  scope='sampled retrospective audit; spectral bounds concern frozen model only')
    (output/'complete.json').write_text(json.dumps(record, indent=2)+'\n')
    if not checks_ok:
        raise ValueError(f'Identity verification failed: {max_errors}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cohort', choices=COHORTS, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    analyze(args.evidence, args.output, args.cohort, args.limit)


if __name__ == '__main__':
    main()
