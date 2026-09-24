"""FP64 spectral propagation of ordinary GD with exactly frozen tanh geometry.

All computed singular values participate. This evaluates a linear recurrence,
not a least-squares refit, optimizer change, or interval certificate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import svd

from .mechanism_dilation_analysis import clean, digest, write_csv


def spectral_factors(singular, eta, updates):
    """Return (1-eta*s²)^n and eta*s*sum_{k<n}(1-eta*s²)^k."""
    if eta <= 0 or updates < 0 or int(updates) != updates:
        raise ValueError('Positive eta and nonnegative integer updates required')
    singular = np.asarray(singular, dtype=np.float64)
    x = eta*singular**2
    if updates == 0:
        return np.ones_like(singular), np.zeros_like(singular)
    power = np.empty_like(x); geometric = np.empty_like(x)
    zero = x == 0
    positive = (x > 0)&(x < 1)
    remaining = ~(zero|positive)
    power[zero] = 1.; geometric[zero] = updates
    log_power = updates*np.log1p(-x[positive])
    power[positive] = np.exp(log_power)
    geometric[positive] = -np.expm1(log_power)/x[positive]
    power[remaining] = np.power(1-x[remaining], updates)
    geometric[remaining] = (1-power[remaining])/x[remaining]
    return power, eta*singular*geometric


def setup(design, target, readout):
    U, singular, Vt = svd(design, full_matrices=False, lapack_driver='gesdd')
    residual = design@readout-target
    coefficients = U.T@residual
    orthogonal = residual-U@coefficients
    return dict(U=U, singular=singular, Vt=Vt, coefficients=coefficients,
                orthogonal=orthogonal, readout=np.asarray(readout).copy())


def propagate(spectral, eta, updates):
    power, gain = spectral_factors(spectral['singular'], eta, updates)
    readout = spectral['readout']-spectral['Vt'].T@(gain*spectral['coefficients'])
    residual = spectral['orthogonal']+spectral['U']@(power*spectral['coefficients'])
    return readout, residual, power


def run(args):
    with np.load(args.input) as data:
        pack = {k: data[k].copy() for k in data.files}
    manifest_path = args.full_run/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if digest(args.input) != manifest['input_sha256']:
        raise ValueError('Prepared input hash differs from full-GD run')
    if manifest.get('freeze_geometry', False):
        raise ValueError('Comparison run must be ordinary full GD')
    eta = float(manifest['eta'])
    choices = [(i, case) for i, case in enumerate(manifest['cases'])
               if str(case.get('seed')) in args.seeds.split(',') and case['arm'] in args.arms.split(',')
               and (not args.targets or case['target'] in args.targets.split(','))]
    if not choices:
        raise ValueError('No matching full-GD cases')
    steps = [0, 20000, 100000]
    full = {}
    for count in steps:
        path = args.full_run/f'{count:09d}.npz'
        if not path.exists():
            raise ValueError(f'Missing matched full-GD snapshot: {path}')
        with np.load(path) as data:
            full[count] = {k: data[k].copy() for k in data.files}
    embedded_evaluation = 'x_eval' in pack and 'y_eval' in pack
    evaluation_sources = {}
    if not embedded_evaluation:
        from . import adam_forces as af, effective_feedback_holdout as holdout, mechanism_widths as widths
        if any(case['target'] not in (*af.TARGETS, *holdout.TARGETS) for _, case in choices):
            raise ValueError('Missing evaluation arrays and unknown target definition')
        evaluation_sources = {str(Path(module.__file__)): digest(Path(module.__file__))
                              for module in (widths, af, holdout)}
    args.output.mkdir(parents=True, exist_ok=False)
    rows, band_rows = [], []
    for local, (full_index, case) in enumerate(choices):
        index = case['input_index']; p0 = pack['p'][index]
        if not np.array_equal(p0, full[0]['p'][full_index]):
            raise ValueError('Prepared and full-GD starting parameters differ')
        a, b, c = p0[:-1].reshape(3, -1)
        x, y = pack['x'], pack['y'][index]
        if embedded_evaluation:
            xe, ye = pack['x_eval'], pack['y_eval'][index]
        else:
            xe, ye, _, _ = widths.data(case['target'], 8192)
        design = np.column_stack((np.tanh(x[:, None]*a+b), np.ones(len(x))))/np.sqrt(len(x))
        eval_design = np.column_stack((np.tanh(xe[:, None]*a+b), np.ones(len(xe))))
        target = y/np.sqrt(len(x)); target_norm = np.linalg.norm(target)
        spectral = setup(design, target, np.r_[c, p0[-1]])
        singular = spectral['singular']; coefficients = spectral['coefficients']
        orthogonal_energy = float(spectral['orthogonal']@spectral['orthogonal'])
        readouts, powers = [], []
        metadata = {k: case.get(k) for k in ('target', 'seed', 'start', 'arm', 'scale', 'reference', 'cohort')}
        for count in steps:
            readout, predicted_residual, power = propagate(spectral, eta, count)
            residual = design@readout-target
            evaluation = eval_design@readout-ye
            readouts.append(readout); powers.append(power)
            fp = full[count]['p'][full_index]; fa, fb, fc = fp[:-1].reshape(3, -1)
            full_error = np.tanh(x[:, None]*fa+fb)@fc+fp[-1]-y
            full_eval_error = np.tanh(xe[:, None]*fa+fb)@fc+fp[-1]-ye
            row = dict(metadata, width=len(a), eta=eta, updates=count,
                       full_completed_updates=int(full[count]['count'][full_index]),
                       full_failed=bool(full[count]['failed'][full_index]),
                       initial_state_exact_match=True, frozen_geometry_exact=True,
                       frozen_relative_l2=float(np.linalg.norm(residual)/target_norm),
                       frozen_relative_eval_l2=float(np.linalg.norm(evaluation)/np.linalg.norm(ye)),
                       full_relative_l2=float(np.linalg.norm(full_error)/np.linalg.norm(y)),
                       full_relative_eval_l2=float(np.linalg.norm(full_eval_error)/np.linalg.norm(ye)),
                       frozen_readout_rms=float(np.sqrt(np.mean(readout[:-1]**2))),
                       full_readout_rms=float(np.sqrt(np.mean(fc**2))),
                       spectral_residual_reconstruction_error=float(np.linalg.norm(residual-predicted_residual)),
                       initial_residual_energy=float(np.sum(coefficients**2)+orthogonal_energy),
                       orthogonal_residual_energy=orthogonal_energy,
                       max_eta_eigenvalue=float(eta*singular[0]**2),
                       stable_linear_gd=bool(eta*singular[0]**2 <= 2),
                       computed_singular_values=len(singular), discarded_singular_values=0,
                       smallest_computed_singular_value=float(singular[-1]))
            row['frozen_minus_full_relative_l2'] = row['frozen_relative_l2']-row['full_relative_l2']
            row['frozen_minus_full_relative_eval_l2'] = row['frozen_relative_eval_l2']-row['full_relative_eval_l2']
            h = float(case.get('h', 2/float(case.get('nref', case.get('Nref', 512)))))
            row['frozen_readout_rms_over_h'] = row['frozen_readout_rms']/h
            row['full_readout_rms_over_h'] = row['full_readout_rms']/h
            rows.append(row)
            for cutoff in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, .1):
                slow = eta*singular**2 <= cutoff
                energy = float(np.sum(coefficients[slow]**2))
                attenuation = np.exp(2*count*np.log1p(-cutoff))
                floor = np.sqrt(orthogonal_energy+attenuation*energy)/target_norm
                band_rows.append(dict(metadata, width=len(a), eta=eta, updates=count,
                    cutoff_eta_eigenvalue=cutoff, slow_modes=int(slow.sum()),
                    initial_slow_energy=energy,
                    current_slow_energy=float(np.sum((power[slow]*coefficients[slow])**2)),
                    relative_l2_lower_bound_through_budget=float(floor)))
        np.savez_compressed(args.output/f'case_{local:03d}.npz',
                            singular_values=singular, initial_residual_coefficients=coefficients,
                            orthogonal_residual_energy=orthogonal_energy,
                            readouts=np.asarray(readouts), powers=np.asarray(powers), updates=np.asarray(steps),
                            frozen_geometry=np.stack((a, b)), case=np.array(json.dumps(metadata)))
        write_csv(args.output/'endpoints.csv', rows); write_csv(args.output/'slow_bands.csv', band_rows)
        print(json.dumps(dict(case=local, **metadata)), flush=True)
    record = dict(cases=len(choices), eta=eta, updates=steps,
                  evaluation_source='embedded_arrays' if embedded_evaluation else 'mechanism_widths.data(target,8192)',
                  evaluation_helper_sha256=evaluation_sources,
                  input_sha256=digest(args.input), full_manifest_sha256=digest(manifest_path),
                  full_snapshot_sha256={str(n): digest(args.full_run/f'{n:09d}.npz') for n in steps},
                  helper_sha256=digest(Path(__file__)),
                  scope='Exact linear-GD recurrence evaluated by FP64 SVD; no singular-value truncation; not interval certification',
                  slow_bound='Retained initial residual energy in eta*lambda<=cutoff; valid for every update through the stated budget in exact arithmetic')
    (args.output/'manifest.json').write_text(json.dumps(clean(record), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--full-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', default='30')
    parser.add_argument('--arms', default='repaired,s10_primary,s10_inverse,s100_primary,s100_inverse')
    parser.add_argument('--targets')
    run(parser.parse_args())
