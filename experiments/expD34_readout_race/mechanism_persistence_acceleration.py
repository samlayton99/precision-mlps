"""Fork-only Hessian drift in the initial acceleration of physical responses.

No future trajectories are read. The identity is local and cannot validate a
20k-step frozen or coupled forecast by itself. CPU allocation and FP64 only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import mechanism_persistence_kernel as kernel
from .mechanism_persistence_summary import digest, write_csv

ETAS = (.002, .001, .0005)


def directional_terms(field, point, direction):
    """Return g, Hv, H²v, and DH[g]v without a third-order tensor."""
    g = field(point)
    hv = jax.jvp(field, (point,), (direction,))[1]
    h2v = jax.jvp(field, (point,), (hv,))[1]
    drift = jax.jvp(lambda p: jax.jvp(field, (p,), (direction,))[1],
                    (point,), (g,))[1]
    return g, hv, h2v, drift


def two_step_acceleration(field, point, direction, eta):
    """Exact tangent of two GD updates, removing the known first-order term.

    Stable form equals (xi2-v+2 eta Hv)/eta², but avoids cancelling v.
    """
    g = field(point)
    hv = jax.jvp(field, (point,), (direction,))[1]
    next_point = point-eta*g
    h1v = jax.jvp(field, (next_point,), (direction,))[1]
    h1hv = jax.jvp(field, (next_point,), (hv,))[1]
    xi2 = direction-eta*hv-eta*jax.jvp(field, (next_point,), (direction-eta*hv,))[1]
    direct = (xi2-direction+2*eta*hv)/eta**2
    stable = (hv-h1v)/eta+h1hv
    return direct, stable


@jax.jit
def diagnose(point, x, y, direction):
    field = lambda p: kernel.ordinary_gradient(p, x, y)
    g, hv, h2v, drift = directional_terms(field, point, direction)
    direct, stable = jax.vmap(lambda eta: two_step_acceleration(field, point, direction, eta))(jnp.asarray(ETAS))
    return dict(g=g, hv=hv, h2v=h2v, drift=drift, direct=direct, stable=stable)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def ratio(a, b):
    return a/b if b else None


def audit(root, output):
    if not jax.config.x64_enabled or any(d.platform != 'cpu' for d in jax.devices()):
        raise RuntimeError('FP64 CPU allocation required')
    output.mkdir(parents=True, exist_ok=False)
    rows, convergence, sources, skipped = [], [], {}, []
    for prediction in sorted(root.glob('physical_*')):
        manifest_path = prediction/'manifest.json'
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('kind') != 'physical':
            continue
        inputs = prediction/'inputs.npz'
        direction_path = prediction.with_name(prediction.name.replace('physical_', 'kicks_', 1))/'directions.npz'
        kick_manifest = direction_path.parent/'manifest.json'
        if manifest['input_sha256'] != digest(inputs):
            raise ValueError('Issued physical input changed')
        if manifest['source_hashes'][Path(kernel.__file__).name] != digest(kernel.__file__):
            raise ValueError('Kernel differs from original physical experiment')
        if json.loads(kick_manifest.read_text())['direction_sha256'] != digest(direction_path):
            raise ValueError('Archived physical direction changed')
        for path in (manifest_path, inputs, direction_path, kick_manifest):
            sources[str(path.relative_to(root))] = digest(path)
        with np.load(inputs) as data:
            points, x, labels = data['p'], data['x'], data['y']
            cases = json.loads(str(data['cases']))
        with np.load(direction_path) as data:
            directions = data['direction']
        for i, case in enumerate(cases):
            if float(case.get('amplitude', 0)) != 0:
                continue
            if not np.any(directions[i]):
                skipped.append(dict(panel=prediction.name, target=case['target'], reason='No resolved physical direction'))
                continue
            value = {k: np.asarray(v) for k, v in jax.device_get(diagnose(*map(jnp.asarray,
                (points[i], x, labels[i], directions[i])))).items()}
            width = (len(points[i])-1)//3
            metadata = dict(panel=prediction.name, cohort=case['cohort'], target=case['target'],
                            seed=case['seed'], nref=case['nref'], width=width, baseline_index=i)
            for block, indices in (('slope', slice(0, width)), ('readout', slice(2*width, 3*width)), ('full', slice(None))):
                h2, drift = value['h2v'][indices], value['drift'][indices]
                total = h2+drift
                hn, dn, tn = map(np.linalg.norm, (h2, drift, total))
                valid = bool(np.isfinite(total).all() and np.isfinite(h2).all() and np.isfinite(drift).all())
                rows.append(clean(dict(metadata, block=block, supported=valid,
                    h2v_norm=hn, drift_norm=dn, total_norm=tn,
                    h2v_drift_cosine=ratio(h2@drift, hn*dn), cancellation_ratio=ratio(tn, hn+dn),
                    full_hessian_only_relative_error=ratio(dn, tn),
                    initial_hv_norm=np.linalg.norm(value['hv'][indices]), initial_direction_norm=np.linalg.norm(directions[i][indices]))))
                previous_error = None
                for ei, eta in enumerate(ETAS):
                    direct, stable = value['direct'][ei, indices], value['stable'][ei, indices]
                    error = np.linalg.norm(stable-total)
                    convergence.append(clean(dict(metadata, block=block, eta=eta,
                        supported=valid and bool(np.isfinite(direct).all() and np.isfinite(stable).all()),
                        stable_error_norm=error, direct_error_norm=np.linalg.norm(direct-total),
                        direct_stable_difference=np.linalg.norm(direct-stable),
                        stable_relative_error=ratio(error, tn),
                        previous_over_current_error=ratio(previous_error, error) if previous_error is not None else None)))
                    previous_error = error
            print(json.dumps(dict(panel=prediction.name, target=case['target'], completed_cases=len(rows)//3)), flush=True)
    write_csv(output/'acceleration.csv', rows)
    write_csv(output/'step_halving.csv', convergence)
    (output/'manifest.json').write_text(json.dumps(clean(dict(source_sha256=digest(__file__),
        kernel_sha256=digest(kernel.__file__), inputs=sources, cases=len(rows)//3, skipped=skipped, etas=list(ETAS),
        identity='xi_second=H_squared_v+DH[g]v at the fork',
        scope='Initial derivative diagnostic, no future trajectory inputs, no finite-window inference',
        step_check='Exact two-step tangent after removing v-2etaHv; O(eta) error after division by eta squared',
        numerical_certificate=False)), indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)
