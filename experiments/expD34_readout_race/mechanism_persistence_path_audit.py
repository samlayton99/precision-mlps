"""Retrospective own-state force rates and rotation; no forecasts are altered."""
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

HORIZONS = (0, 1000, 10000, 20000)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(a, b):
    return jnp.where(b > 0, a/jnp.where(b > 0, b, 1.), jnp.nan)


@jax.jit
def path_diagnostics(p, x, y, reference, arm):
    gradient, applied, natural, remainder = kernel.step_components(p, x, y, reference, arm)
    derivative = jax.jvp(lambda point: kernel.arm_effective(point, x, y, reference, arm),
                         (p,), (-gradient,))[1]
    q = jnp.linalg.norm(applied)
    q0 = jnp.linalg.norm(reference['F0'])
    width = (len(p)-1)//3
    a_norm = jnp.linalg.norm(applied[:width])
    a0_norm = jnp.linalg.norm(reference['F0'][:width])
    state = kernel.decomposition(p, x, y)
    residual_change = jnp.linalg.norm(state['eH']-reference['eH0'])
    return dict(k_arm=ratio(applied@derivative, q*q), q=q, q_over_q0=ratio(q, q0),
                force_cos_initial=ratio(applied@reference['F0'], q*q0),
                slope_force_cos_initial=ratio(applied[:width]@reference['F0'][:width], a_norm*a0_norm),
                slope_fraction=ratio(a_norm, q),
                fine_residual_relative_change=ratio(residual_change, jnp.linalg.norm(reference['eH0'])),
                fine_residual_absolute_change=residual_change/jnp.sqrt(len(x)),
                tracking_norm=jnp.linalg.norm(remainder),
                natural_force_norm=jnp.linalg.norm(natural),
                coarse_disequilibrium_norm=jnp.linalg.norm(state['z']))


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


def audit(root, output, max_seconds=600.):
    if not jax.config.x64_enabled or any(d.platform != 'cpu' for d in jax.devices()):
        raise RuntimeError('FP64 CPU required; run through the campaign CPU allocation')
    output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    rows, sources, missing = [], {}, []
    natural_diagnostics = jax.jit(kernel.diagnostics)
    timed_out = False
    path = output/'path_diagnostics.csv'
    # Fixed columns allow each completed state to survive a deadline interruption.
    fields = ['panel', 'cohort', 'target', 'seed', 'nref', 'width', 'source_index', 'arm',
              'horizon', 'actual_updates', 'failed', 'k_initial', 'k_arm', 'q', 'q_over_q0',
              'force_cos_initial', 'slope_force_cos_initial', 'slope_fraction',
              'fine_residual_relative_change', 'fine_residual_absolute_change',
              'tracking_norm', 'natural_force_norm', 'coarse_disequilibrium_norm',
              'D_over_q_squared', 'C_over_q_squared', 'C_generated_over_q_squared',
              'C_target_over_q_squared', 'loaded_over_q_squared', 'k_pure']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for prediction in sorted(root.glob('feedback_*')):
            if not (prediction/'fork_diagnostics.csv').exists():
                continue
            manifest_path = prediction/'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('kind') != 'feedback':
                continue
            if manifest['source_hashes'].get(Path(kernel.__file__).name) != digest(kernel.__file__):
                raise ValueError(f'Kernel differs from issued source: {prediction}')
            run = prediction.with_name(prediction.name+'_run')
            input_path = prediction/'inputs.npz'
            if digest(input_path) != manifest['input_sha256']:
                raise ValueError(f'Changed issued input: {input_path}')
            run_manifest = run/'manifest.json'
            if run_manifest.exists():
                if json.loads(run_manifest.read_text())['prediction_sha256'] != digest(manifest_path):
                    raise ValueError(f'Run refers to different predictions: {run}')
                sources[str(run_manifest.relative_to(root))] = digest(run_manifest)
            sources[str(input_path.relative_to(root))] = digest(input_path)
            sources[str(manifest_path.relative_to(root))] = digest(manifest_path)
            with np.load(input_path) as data:
                pack = {k: data[k] for k in ('x', 'y', 'arms', 'reference_F0', 'reference_eH0', 'cases')}
            cases = json.loads(str(pack['cases']))
            with (prediction/'fork_diagnostics.csv').open() as source:
                fork_rates = {(int(float(r['source_index'])), r['arm']): float(r['k_actual_arm'] or 'nan') for r in csv.DictReader(source)}
            sources[str((prediction/'fork_diagnostics.csv').relative_to(root))] = digest(prediction/'fork_diagnostics.csv')
            x = jnp.asarray(pack['x'])
            for horizon in HORIZONS:
                snapshot = run/'snapshots'/f'{horizon:09d}.npz'
                if not snapshot.exists():
                    missing.append(str(snapshot.relative_to(root)))
                    continue
                sources[str(snapshot.relative_to(root))] = digest(snapshot)
                with np.load(snapshot) as state:
                    points, counts, failed = state['p'], state['count'], state['failed']
                for i, case in enumerate(cases):
                    if time.perf_counter()-begin >= max_seconds:
                        timed_out = True
                        break
                    point, y, arm = map(jnp.asarray, (points[i], pack['y'][i], pack['arms'][i]))
                    reference = dict(F0=jnp.asarray(pack['reference_F0'][i]), eH0=jnp.asarray(pack['reference_eH0'][i]))
                    value = {k: float(v) for k, v in jax.device_get(path_diagnostics(point, x, y, reference, arm)).items()}
                    row = {k: case.get(k) for k in ('cohort', 'target', 'seed', 'nref', 'width', 'source_index', 'arm')}
                    row.update(value, panel=prediction.name, horizon=horizon,
                               actual_updates=int(counts[i]), failed=bool(failed[i]),
                               k_initial=fork_rates[(int(case['source_index']), case['arm'])])
                    if case['arm'] == 'natural':
                        diag = {k: float(v) for k, v in jax.device_get(natural_diagnostics(point, x, y)).items()}
                        row['k_pure'] = diag['k_pure']
                        for term in ('D', 'C', 'C_generated', 'C_target', 'loaded'):
                            row[term+'_over_q_squared'] = diag[term]/diag['q_squared'] if diag['q_squared'] else None
                    row = clean(row)
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
                if timed_out:
                    break
            if timed_out:
                break
            print(json.dumps(dict(panel=prediction.name, completed_rows=len(rows), elapsed_seconds=time.perf_counter()-begin)), flush=True)
    result = dict(kind='Retrospective path audit, not a newly issued forecast',
                  source_sha256=digest(__file__), kernel_sha256=digest(kernel.__file__),
                  inputs=sources, rows=len(rows), horizons=HORIZONS, timed_out=timed_out,
                  elapsed_seconds=time.perf_counter()-begin, max_seconds=max_seconds,
                  missing_snapshots=missing)
    (output/'manifest.json').write_text(json.dumps(clean(result), indent=2, allow_nan=False))
    print(json.dumps(clean(result)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=600.)
    args = parser.parse_args()
    audit(args.root, args.output, args.max_seconds)
