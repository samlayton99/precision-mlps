"""Assemble observed training and theorem evidence for the paper's three panels.

No optimizer runs or spectral forecasts are substituted for observations.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.expD36_frozen_gamma_probe.direct_ratio_interval import error_curve, first_crossing

EPSILONS = (.1, .03, .01, .003, .001)


def grid(m):
    return -1 + (np.arange(m)+.5)*2/m


def target(x):
    return np.sin(2*np.pi*x)+.5*np.sin(6*np.pi*x)+.25*np.sin(14*np.pi*x)


def predict(parameters, x):
    a, b, c = parameters[:-1].reshape(3, -1)
    return np.tanh(x[:, None]*a+b)@c+parameters[-1]


def crossing(steps, errors, epsilon, exact=None):
    indices = np.flatnonzero(errors <= epsilon)
    if not len(indices):
        return dict(executed_updates=exact, bracket_lower=int(steps[-1]),
                    bracket_upper=None, censored=True)
    i = int(indices[0])
    return dict(executed_updates=exact, bracket_lower=int(steps[max(0, i-1)]),
                bracket_upper=int(steps[i]), censored=False)


def collect(frozen, joint_root):
    hashes, arrays, checks = {}, {}, {}
    def source(path):
        path = Path(path)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def read(path):
        return json.loads(source(path).read_text())
    campaign = read(frozen/'refinements/capped_kernel/campaign_summary.json')
    factorized = read(frozen/'refinements/gamma_factorized_kernel/summary.json')
    assert factorized['width'] == 559 and factorized['samples'] == 8193
    assert factorized['targets'][0] == 'sine_mix_2_6_10'
    gd = []
    for gamma in (8, 12, 16, 64):
        name = f'N512_cap{gamma}_common_s0'
        folder = frozen/'refinements/capped_kernel/evidence/gd_trajectories'/name
        meta, training, curve = [read(folder/f'{part}.json') for part in ('meta', 'training', 'curve')]
        case = next(c for c in campaign['cases'] if c['id'] == name)
        dictionary = next(d for d in factorized['dictionaries'] if d['gamma'] == gamma)
        assert training == case['training']
        assert meta['matrix_hash'] == case['matrix_hash'] == dictionary['matrix_hash']
        assert meta['target_hash'] == case['target_hash']
        assert meta['eta'] == training['eta'] == dictionary['eta']
        assert meta['map'] == 'raw' and meta['initialization'] == 'zero'
        assert meta['n'] == 512 and meta['family'] == 'common'
        steps = np.r_[0, [c['step'] for c in curve]].astype(np.int64)
        errors = np.r_[1., [c['train'][0] for c in curve]]
        assert np.all(np.diff(steps)>0) and np.all(np.diff(errors)<=1e-14)
        assert steps[-1] == training['steps'] and errors[-1] == training['final_train'][0]
        bound_path = frozen/f'refinements/gamma_direct_ratio/archive_g{gamma}_q10_p20'
        info = read(bound_path.with_suffix('.json'))
        assert info['gamma'] == gamma and info['spacing'] == 1/256
        assert info['arrays_sha256'] == hashlib.sha256(source(bound_path.with_suffix('.npz')).read_bytes()).hexdigest()
        for relative in ('common/N512/arrays.npz', 'refinements/gamma_factorized_kernel/summary.json'):
            path = source(frozen/relative)
            expected = next(value for key,value in info['source_sha256'].items() if key.endswith('/full_sweep/'+relative))
            assert hashes[str(path)] == expected
        with np.load(source(bound_path.with_suffix('.npz'))) as saved:
            rates = info['eta_mu1']*saved['rho_upper'][:len(saved['actual_target_weights'])]
            weights = saved['actual_target_weights'].copy()
            weights[~saved['resolved']] = 0.
        assert info['eta'] == training['eta'] and info['samples'] == 8193 and info['width'] == 559
        lower = error_curve(rates, weights, 0., steps)
        assert np.all(lower[1:] <= errors[1:]), (gamma, float(np.max(lower-errors)))
        smooth = np.unique(np.r_[0, np.geomspace(1, steps[-1], 420).astype(np.int64)])
        smooth_error = error_curve(rates, weights, 0., smooth)
        crossings = []
        for epsilon in EPSILONS:
            exact = int(training['hits'][0][0]) if epsilon == .01 else None
            c = crossing(steps, errors, epsilon, exact)
            necessary = first_crossing(rates, weights, 0., epsilon)
            if exact is not None:
                assert necessary <= exact and c['bracket_lower'] < exact <= c['bracket_upper']
            crossings.append(dict(epsilon=epsilon, necessary_updates=necessary, **c))
        row = dict(gamma=gamma, steps=steps.tolist(), error=errors.tolist(), lower_error=lower.tolist(),
                   bound_steps=smooth.tolist(), bound_error=smooth_error.tolist(), crossings=crossings,
                   eta=training['eta'], matrix_hash=meta['matrix_hash'], target_hash=meta['target_hash'],
                   numerical_status=info['numerical_status'], lower_actual_ratio=(lower/errors).tolist(),
                   initialization='zero', observed_start=int(steps[1]), observed_horizon=int(steps[-1]))
        gd.append(row)
        for key, value in [('steps', steps), ('error', errors), ('lower_error', lower), ('bound_steps', smooth), ('bound_error', smooth_error)]:
            arrays[f'gd{gamma}_{key}'] = value
        checks[f'gd{gamma}'] = dict(actual_checkpoints=len(curve), maximum_lower_minus_actual=float(np.max(lower[1:]-errors[1:])),
                                    analytic_initialization_weight_closure=float(lower[0]-1),
                                    minimum_lower_actual_ratio=float(np.min(lower/errors)), crossings=crossings)
    # Recompute physical-coordinate network predictions at every archived state.
    train_x, eval_x = grid(2048), grid(8192)
    normalizer = float(np.sqrt(np.mean(target(train_x)**2)))
    train_y, eval_y = target(train_x)/normalizer, target(eval_x)/normalizer
    result = dict(seeds=list(range(5)), target='mixed_sine', width=177, train_samples=2048,
                  eval_samples=8192, original_train_rms=normalizer, slope_bandwidth_divisor=64,
                  target_formula='sin(2*pi*x)+0.5*sin(6*pi*x)+0.25*sin(14*pi*x)',
                  gd=dict(train_error=[], eval_error=[], lambda_rms=[], cases=[]),
                  adam=dict(train_error=[], eval_error=[], lambda_rms=[], cases=[]))
    with gzip.open(source(joint_root/'states.csv.gz'), 'rt') as f:
        archived = {(r['bundle'], int(r['case_index']), int(r['step'])):float(r['relative_train_mse'])
                    for r in csv.DictReader(f) if r['target']=='mixed_sine' and r['bundle'].startswith('primary_')}
    max_mse_gap = 0.
    for seed in range(5):
        bundle = f'primary_{seed}'
        folder = joint_root/'raw'/bundle
        manifest = read(folder/'manifest.json')
        assert manifest['m'] == 2048 and manifest['width'] == 177
        with np.load(source(folder/'snapshots.npz')) as f:
            steps, parameters, failed = f['steps'], f['p'], f['failed']
        assert len(steps) == 39 and steps[0] == 0 and steps[-1] == 600000
        if seed:
            assert result['steps'] == steps.tolist()
        result['steps'] = steps.tolist()
        with np.load(source(folder/'state.npz')) as f:
            assert int(f['cursor']) == 600000
            np.testing.assert_array_equal(f['p'], parameters[:, -1])
        with np.load(source(joint_root/'curated'/bundle/'snapshots.npz')) as f:
            for j, step in enumerate(f['steps']):
                k = int(np.flatnonzero(steps == step)[0])
                np.testing.assert_array_equal(f['p'][:, j], parameters[:, k])
        selected_indices = [next(i for i,c in enumerate(manifest['cases']) if c['target']=='mixed_sine' and c['optimizer']==opt) for opt in ('gd','adam')]
        np.testing.assert_array_equal(parameters[selected_indices[0],0], parameters[selected_indices[1],0])
        for optimizer in ('gd', 'adam'):
            ci = next(i for i,c in enumerate(manifest['cases']) if c['target']=='mixed_sine' and c['optimizer']==optimizer and c['seed']==seed)
            assert manifest['target_normalizers'][ci] == normalizer and not np.any(failed[ci])
            params = parameters[ci]
            train_errors = np.array([np.linalg.norm(predict(p,train_x)-train_y)/np.linalg.norm(train_y) for p in params])
            eval_errors = np.array([np.linalg.norm(predict(p,eval_x)-eval_y)/np.linalg.norm(eval_y) for p in params])
            bandwidth = np.sqrt(np.mean(params[:, :177]**2, axis=1))/64
            for k, step in enumerate(steps):
                key = (bundle, ci, int(step))
                assert key in archived
                gap = abs(train_errors[k]**2-archived[key]); max_mse_gap = max(max_mse_gap, gap)
                assert gap < 1e-11
            for key, values in [('train_error', train_errors), ('eval_error', eval_errors), ('lambda_rms', bandwidth)]:
                result[optimizer][key].append(values.tolist())
            result[optimizer]['cases'].append(manifest['cases'][ci])
            arrays[f'joint_{optimizer}_seed{seed}_parameters'] = params
    for optimizer in ('gd', 'adam'):
        for key in ('train_error', 'eval_error', 'lambda_rms'):
            arrays[f'joint_{optimizer}_{key}'] = np.asarray(result[optimizer][key])
    arrays['joint_steps'] = np.asarray(result['steps'])
    centers = -1+np.arange(-24,153)/64
    design = np.column_stack((np.ones(len(train_x)), np.tanh(16*(train_x[:,None]-centers))))
    coefficients, _, rank, singular = np.linalg.lstsq(design, train_y, rcond=2048*np.finfo(float).eps)
    eval_design = np.column_stack((np.ones(len(eval_x)), np.tanh(16*(eval_x[:,None]-centers))))
    construction = dict(gamma=16, lambda_rms=16/64, width=len(centers), rank=int(rank),
        rcond=2048*np.finfo(float).eps, train_error=float(np.linalg.norm(design@coefficients-train_y)/np.linalg.norm(train_y)),
        eval_error=float(np.linalg.norm(eval_design@coefficients-eval_y)/np.linalg.norm(eval_y)),
        method='Direct SVD least squares with bias; capacity witness, not a training trajectory.')
    fine_x = grid(16384)
    fine_design = np.column_stack((np.ones(len(fine_x)), np.tanh(16*(fine_x[:,None]-centers))))
    fine_y = target(fine_x)/normalizer
    construction['verification_16384_error'] = float(np.linalg.norm(fine_design@coefficients-fine_y)/np.linalg.norm(fine_y))
    arrays.update(construction_centers=centers, construction_coefficients=coefficients,
                  construction_singular_values=singular, train_x=train_x, eval_x=eval_x,
                  train_target=train_y, eval_target=eval_y)
    checks['joint'] = dict(raw_curated_overlap_exact=True, raw_endpoint_state_exact=True,
                          maximum_archived_train_mse_gap=max_mse_gap, snapshots_per_case=39, cases=10, archived_train_mse_comparisons=390)
    checks['joint']['paired_initialization_exact'] = True
    checks['joint']['endpoint_medians'] = {opt:{key:float(np.median(np.asarray(result[opt][key])[:,-1])) for key in ('train_error','eval_error','lambda_rms')} for opt in ('gd','adam')}
    checks['construction'] = construction
    data = dict(gd=gd, joint=result, construction=construction, source_sha256=hashes,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        conventions=dict(gd='Step0 is analytic zero initialization; all remaining errors are actual executed checkpoints.',
            bound='V4 p=0 numerical lower bound; unresolved target weights omitted only in this lower curve.',
            crossings='Except exact1% hits, intervals bracket first threshold crossings between observed checkpoints of monotone frozen GD.',
            joint='All39 raw snapshots per seed; straight plotting segments are not intervening observations.',
            normalization='Targets normalized using original2048-grid RMS; relative L2 denominator evaluated on each grid.'))
    return data, arrays, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-root', required=True, type=Path)
    parser.add_argument('--joint-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data, arrays, validation = collect(args.frozen_root, args.joint_root)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'evidence.npz', **arrays)
    (args.output/'figure_data.json').write_text(json.dumps(data, indent=2)+'\n')
    (args.output/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
