"""Runpod follow-up at the completed six-million-update Modal checkpoints.

The optimizer, intervention, and forecasting kernels are imported unchanged.
Preparation and analysis run on CPUs; pilot and probes require a Slurm GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_forces as af, adam_run as ar, plateau, plateau_probes as pp
from . import plateau_run as pr
from .plateau_runtime import verify_gpu
from .run import write_json

START = 6000000
HORIZON = 500000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed(folder):
    status = json.loads((folder / 'status.json').read_text())
    if (not status['complete'] or np.any(status['failed'])
            or np.any(status.get('unresolved', status.get('unresolved_steps', 0)))):
        raise ValueError(f'Incomplete or invalid source: {folder}')
    for name in ('identity_max', 'motion_identity', 'channel_identity'):
        if np.max(np.abs(status.get(name, 0))) > 1e-9:
            raise ValueError(f'Failed {name}: {folder}')
    return status


def extract(folder, seed, names, step=START):
    """Select complete state fields by case identity and exact checkpoint step."""
    completed(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    indices = [i for i, c in enumerate(manifest['cases'])
               if c['seed'] == seed and c['target'] in names]
    cases = [manifest['cases'][i] for i in indices]
    if len(cases) != len(names) or {c['target'] for c in cases} != set(names):
        raise ValueError('Missing or duplicate checkpoint cases')
    with np.load(folder / 'snapshots.npz') as f:
        positions = np.flatnonzero(f['steps'] == step)
        if len(positions) != 1:
            raise ValueError(f'Missing checkpoint {step}')
        state = {k: f[k][indices, int(positions[0])].copy()
                 for k in f.files if k != 'steps'}
    if not all(np.all(np.isfinite(v)) for v in state.values()):
        raise ValueError('Nonfinite checkpoint')
    return cases, state


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=True)
    parent = source / 'long/gd'
    completed(parent)
    hashes = {str(parent / n): sha(parent / n)
              for n in ('manifest.json', 'snapshots.npz', 'status.json')}
    for seed in range(3):
        cases, state = extract(parent, seed, pp.ANCHORS)
        folder = output / 'inputs' / f'primary_{seed}'
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / 'manifest.json', dict(cases=cases, parent_hashes=hashes,
                   initial_step=START, adaptation='Exact case/step slices; all state fields retained'))
        ar.atomic_npz(folder / 'snapshots.npz', steps=np.array([START]),
                      **{k: v[:, None] for k, v in state.items()})
        # Immutable forecasts are issued before any late intervention is submitted.
        pr.issue_forecast(folder, state, cases, START)
    artifacts = {str(p.relative_to(output)): sha(p)
                 for p in sorted((output / 'inputs').rglob('*')) if p.is_file()}
    write_json(output / 'prepared.json', dict(issued_unix=time.time(),
               parent_hashes=hashes, artifacts=artifacts, start=START, horizon=HORIZON))


def pilot(source, output):
    output.mkdir(parents=True, exist_ok=True)
    verify_gpu(output, 'slurm')
    if jax.__version__ != '0.10.2' or np.__version__ != '2.4.6':
        raise ValueError('The archived JAX/NumPy environment is required')
    results = []
    for optimizer in ('gd', 'adam'):
        cases, original = extract(source / 'long' / optimizer, 0, ('moment9', 'mixed_sine'))
        state = jax.tree.map(jnp.asarray, original)
        settings = jnp.array([[c[k] for k in ('eta', 'beta1', 'beta2', 'epsilon', 'adaptive')]
                              for c in cases])
        yy = jnp.array(np.stack([af.data(c['target'])[1] for c in cases]))
        xx = jnp.asarray(af.data(cases[0]['target'])[0])
        gradient_error = split_error = 0.
        for p, y in zip(state['p'], yy):
            g, _, jc, ec = af.field(p, xx, y)
            expected = jax.grad(lambda v: .5*jnp.mean((plateau.prediction(v, xx)-y)**2))(p)
            channels, info = af.split(g, jc, ec)
            np.testing.assert_allclose(g, expected, atol=2e-13, rtol=2e-11)
            np.testing.assert_allclose(channels.sum(axis=0), g, atol=2e-13, rtol=2e-11)
            assert bool(info['resolved'])
            gradient_error = max(gradient_error, float(jnp.max(abs(g-expected))))
            split_error = max(split_error, float(jnp.max(abs(channels.sum(axis=0)-g))))
        advance = ar.advance_factory(2048)
        begun = time.monotonic()
        whole = jax.device_get(advance(state, yy, settings, 32)[0])
        part = jax.device_get(advance(state, yy, settings, 13)[0])
        path = output / f'{optimizer}_interrupted.npz'
        ar.atomic_npz(path, **part)
        with np.load(path) as f:
            loaded = {k: jnp.asarray(f[k]) for k in f.files}
        resumed = jax.device_get(advance(loaded, yy, settings, 19)[0])
        for name in whole:
            np.testing.assert_allclose(resumed[name], whole[name], atol=2e-13, rtol=2e-12)
        delta = np.mean(abs(resumed['p'][:, :177])-abs(original['p'][:, :177]), axis=1)
        signed = (resumed['signed_channels']-original['signed_channels']).sum(axis=1)
        signed += resumed['crossing']-original['crossing']
        np.testing.assert_allclose(delta, signed, atol=1e-12, rtol=0)
        assert not np.any(resumed['failed']) and not np.any(resumed['unresolved_steps'])
        assert np.max(resumed['identity_max']) <= 1e-9
        results.append(dict(optimizer=optimizer, gradient_error=gradient_error,
                       split_error=split_error, signed_motion_error=float(np.max(abs(delta-signed))),
                       seconds=time.monotonic()-begun, state_fields=list(whole)))
    write_json(output / 'passed.json', dict(passed=True, checks=results,
               device=jax.devices()[0].device_kind, jax=jax.__version__, numpy=np.__version__))


def probe(output, seed, max_seconds):
    if not json.loads((output / 'pilot/passed.json').read_text())['passed']:
        raise ValueError('GPU pilot has not passed')
    preparation = json.loads((output / 'prepared.json').read_text())
    for name, expected in preparation['artifacts'].items():
        if sha(output / name) != expected:
            raise ValueError(f'Changed prepared input or forecast: {name}')
    folder = output / 'probes' / f'seed{seed}_{START}'
    folder.mkdir(parents=True, exist_ok=True)
    # Slurm requeues are disabled; a duplicate submission must not own this output.
    with (folder / 'owner.json').open('x') as f:
        json.dump(dict(job=os.environ.get('SLURM_JOB_ID'), started=time.time()), f)
    pp.run(SimpleNamespace(runtime='slurm', source=output / 'inputs', output=folder,
           seed=seed, start=START, horizon=HORIZON, samples=2048, half=False,
           max_seconds=max_seconds))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'pilot', 'probe'))
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=range(3))
    p.add_argument('--max-seconds', type=float, default=1080)
    args = p.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError('FP64 is required')
    if args.phase == 'prepare':
        prepare(args.source, args.output)
    elif args.phase == 'pilot':
        pilot(args.source, args.output / 'pilot')
    else:
        if args.seed is None:
            p.error('--seed is required for probes')
        probe(args.output, args.seed, args.max_seconds)


if __name__ == '__main__':
    main()
