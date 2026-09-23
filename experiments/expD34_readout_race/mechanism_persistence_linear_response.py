"""Retrospective frozen fork-derivative responses to physical GD pulses.

Forecast the derivative of movement relative to each pulse's own initial
state, not the unperturbed trajectory. No coefficient is fitted to outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import mechanism_persistence_kernel as kernel

ETA = .002
HORIZONS = (1000, 10000, 20000)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spectral_movement(eigenvalues, eigenvectors, direction, steps, eta=ETA):
    """Return [(I-eta H)**steps-I]v, preserving signed discrete factors.

    Zero, negative, tiny positive and unstable eigenvalues are not clipped.
    Overflow yields supported=False, never a silently modified operator.
    """
    if steps < 0 or int(steps) != steps:
        raise ValueError('Nonnegative integer step count required')
    eigenvalues = np.asarray(eigenvalues)
    if steps == 0:
        return np.zeros_like(direction), True
    z = -eta*eigenvalues
    base = 1+z
    change = np.empty_like(eigenvalues)
    positive, negative = base > 0, base < 0
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        change[positive] = np.expm1(steps*np.log1p(z[positive]))
        exponent = steps*np.log(-base[negative])
        change[negative] = np.expm1(exponent) if steps % 2 == 0 else -np.exp(exponent)-1
        change[base == 0] = -1.
        response = eigenvectors@(change*(eigenvectors.T@direction))
    return response, bool(np.isfinite(change).all() and np.isfinite(response).all())


def operator_movement(jacobian, direction, steps, eta=ETA):
    with np.errstate(over='ignore', invalid='ignore'):
        response = (np.linalg.matrix_power(np.eye(len(direction))-eta*jacobian, steps)-np.eye(len(direction)))@direction
    return response, bool(np.isfinite(response).all())


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def compare(actual, predicted):
    norm, predicted_norm = np.linalg.norm(actual), np.linalg.norm(predicted)
    error = np.linalg.norm(predicted-actual)
    return dict(actual_norm=norm, predicted_norm=predicted_norm, error_norm=error,
                relative_error=error/norm if norm else None,
                cosine=actual@predicted/(norm*predicted_norm) if norm*predicted_norm else None)


def run(root, output, include_effective=True, max_seconds=600.):
    if not jax.config.x64_enabled or any(d.platform != 'cpu' for d in jax.devices()):
        raise RuntimeError('FP64 CPU allocation required')
    output.mkdir(parents=True, exist_ok=False)
    hessian = jax.jit(jax.jacfwd(kernel.ordinary_gradient, argnums=0))
    effective_jacobian = jax.jit(jax.jacfwd(kernel.effective, argnums=0))
    begin = time.perf_counter()
    rows, sources, skipped = [], {}, []
    timed_out = False
    for prediction in sorted(root.glob('physical_*')):
        if not (prediction/'forecasts.npz').exists():
            continue
        manifest_path = prediction/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('kind') != 'physical':
            continue
        if manifest['source_hashes'][Path(kernel.__file__).name] != digest(kernel.__file__):
            raise ValueError('Kernel differs from original physical experiment')
        input_path = prediction/'inputs.npz'
        if digest(input_path) != manifest['input_sha256']:
            raise ValueError('Physical initial states changed')
        sources[str(manifest_path.relative_to(root))] = digest(manifest_path)
        sources[str(input_path.relative_to(root))] = digest(input_path)
        with np.load(input_path) as data:
            initial, x, labels = data['p'], data['x'], data['y']
            cases = json.loads(str(data['cases']))
        run_path = prediction.with_name(prediction.name+'_run')
        run_manifest = run_path/'manifest.json'
        if json.loads(run_manifest.read_text())['prediction_sha256'] != digest(manifest_path):
            raise ValueError('Physical run refers to different issued predictions')
        sources[str(run_manifest.relative_to(root))] = digest(run_manifest)
        snapshots = {}
        for horizon in HORIZONS:
            path = run_path/'snapshots'/f'{horizon:09d}.npz'
            if path.exists():
                with np.load(path) as data:
                    snapshots[horizon] = (data['p'], data['failed'], data['count'])
                sources[str(path.relative_to(root))] = digest(path)
        # Use archived pulse vectors if present; otherwise disclose recovery.
        directions_path = prediction.with_name(prediction.name.replace('physical_', 'kicks_', 1))/'directions.npz'
        directions = None
        if directions_path.exists():
            kick_manifest = directions_path.parent/'manifest.json'
            if json.loads(kick_manifest.read_text())['direction_sha256'] != digest(directions_path):
                raise ValueError('Archived physical pulse direction changed')
            with np.load(directions_path) as data:
                directions = data['direction']
            sources[str(directions_path.relative_to(root))] = digest(directions_path)
            sources[str(kick_manifest.relative_to(root))] = digest(kick_manifest)
        for base_index, case in enumerate(cases):
            if float(case.get('amplitude', 0)) != 0:
                continue
            family = {float(c['amplitude']): i for i, c in enumerate(cases)
                      if c.get('reference_baseline_index') == case.get('reference_baseline_index')}
            amplitudes = sorted(a for a in family if a > 0 and -a in family)
            if not amplitudes:
                skipped.append(dict(panel=prediction.name, target=case['target'], reason='No resolved paired pulses'))
                continue
            if time.perf_counter()-begin >= max_seconds:
                timed_out = True
                break
            largest = amplitudes[-1]
            v = directions[base_index] if directions is not None else (initial[family[largest]]-initial[family[-largest]])/(2*largest)
            point, xx, yy = map(jnp.asarray, (initial[base_index], x, labels[base_index]))
            h = np.asarray(hessian(point, xx, yy))
            asymmetry = float(np.linalg.norm(h-h.T))
            eigenvalues, eigenvectors = np.linalg.eigh((h+h.T)/2)
            jf = np.asarray(effective_jacobian(point, xx, yy)) if include_effective else None
            width = (len(v)-1)//3
            for horizon, (states, failed, count) in snapshots.items():
                models = {'full_hessian': spectral_movement(eigenvalues, eigenvectors, v, horizon)}
                if include_effective:
                    models['effective_jacobian'] = operator_movement(jf, v, horizon)
                for amplitude in amplitudes:
                    plus, minus = family[amplitude], family[-amplitude]
                    actual = ((states[plus]-initial[plus])-(states[minus]-initial[minus]))/(2*amplitude)
                    pulse_error = np.linalg.norm((initial[plus]-initial[minus])/(2*amplitude)-v)
                    for model, (predicted, supported) in models.items():
                        for block, indices in (('full', slice(None)), ('slope', slice(0, width)), ('readout', slice(2*width, 3*width))):
                            row = dict(panel=prediction.name, cohort=case['cohort'], target=case['target'], seed=case['seed'],
                                width=width, nref=case['nref'], baseline_index=base_index, horizon=horizon,
                                amplitude=amplitude, model=model, block=block, supported=supported,
                                failed=bool(failed[plus] or failed[minus]),
                                actual_updates_plus=int(count[plus]), actual_updates_minus=int(count[minus]),
                                hessian_asymmetry_norm=asymmetry, hessian_min_eigenvalue=float(eigenvalues[0]),
                                hessian_max_eigenvalue=float(eigenvalues[-1]), initial_direction_error=pulse_error,
                                direction_source='saved' if directions is not None else 'largest paired initial states')
                            row.update(compare(actual[indices], predicted[indices]))
                            if block == 'slope':
                                sign = np.sign(initial[base_index, :width])
                                row['actual_initial_sign_mean_lambda_response'] = case['h']*np.mean(sign*actual[:width])
                                row['predicted_initial_sign_mean_lambda_response'] = case['h']*np.mean(sign*predicted[:width])
                            rows.append(clean(row))
            print(json.dumps(dict(panel=prediction.name, target=case['target'], rows=len(rows), elapsed_seconds=time.perf_counter()-begin)), flush=True)
        if timed_out:
            break
    if rows:
        with (output/'responses.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader(); writer.writerows(rows)
    (output/'manifest.json').write_text(json.dumps(clean(dict(source_sha256=digest(__file__),
        kernel_sha256=digest(kernel.__file__), inputs=sources, rows=len(rows), skipped=skipped,
        elapsed_seconds=time.perf_counter()-begin, timed_out=timed_out, max_seconds=max_seconds,
        include_effective=include_effective, eta=ETA, horizons=HORIZONS,
        scope='Retrospective fork-derivative comparison; no future fit, baseline path not predicted',
        interpretation='Central displacement derivative; finite pulse amplitudes require halving checks. Initial slope signs only; not a normalized-scale crossing forecast.',
        numerical_certificate=False)), indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-effective', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=600.)
    args = parser.parse_args()
    run(args.root, args.output, not args.skip_effective, args.max_seconds)
