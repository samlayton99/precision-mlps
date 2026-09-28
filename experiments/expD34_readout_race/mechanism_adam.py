"""Small preserved-history Adam tracking experiment; all execution is scheduled.

The two inputs to Adam's moment recurrences are changed independently, only
on slopes. Forecasts either freeze both forces or periodically repeat two
phases obtained from the fork and one virtual ordinary Adam update. This is
a local conditional driver model, not a forecast reading future training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_forces as af

ARMS = ((1., 1.), (.9, 1.), (1., .9), (.9, .9))
SAVED = (0, 1, 10, 100, 1000, 2000, 10000, 20000)
H = 2 / 128
METRICS = ('relative_mse_before', 'mean_lambda_before', 'lambda_increment',
           'effective_signed_increment', 'tracking_signed_increment',
           'crossing_increment', 'raw_tracking_norm', 'raw_effective_norm',
           'tracking_step_norm', 'effective_step_norm', 'moment_identity',
           'step_identity', 'coarse_resolved', 'mean_inverse', 'max_inverse')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {p.name: digest(p) for p in (Path(__file__), Path(af.__file__),
                                       Path(af.targets.__file__))}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def initial(pack):
    state = {k: jnp.asarray(pack[k]) for k in ('p', 'm', 'v', 'channel_m', 'count')}
    batch, size = state['p'].shape
    width = (size - 1) // 3
    state.update(positive=jnp.zeros((batch, width)), negative=jnp.zeros((batch, width)),
                 signed=jnp.zeros((batch, 3, width)), crossing=jnp.zeros((batch, width)),
                 force_activity=jnp.zeros((batch, 3)), step_activity=jnp.zeros((batch, 3)),
                 identity_max=jnp.zeros((batch, 2)), unresolved=jnp.zeros(batch, dtype=jnp.int64))
    return state


def moment_step(old, channels, alpha, eta=.002, beta1=.9, beta2=.999, epsilon=1e-8,
                gradient=None):
    """One update with the original second-moment cross terms intact."""
    width = (old['p'].size - 1) // 3
    gm_channels = channels.at[1, :width].multiply(alpha[0])
    gradient = channels.sum(axis=0) if gradient is None else gradient
    gm = gradient.at[:width].add((alpha[0]-1)*channels[1, :width])
    gv = gradient.at[:width].add((alpha[1]-1)*channels[1, :width])
    count = old['count'] + 1
    m = beta1 * old['m'] + (1-beta1) * gm
    v = beta2 * old['v'] + (1-beta2) * gv**2
    cm = beta1 * old['channel_m'] + (1-beta1) * gm_channels
    inv = 1 / (jnp.sqrt(v / (1-beta2**count)) + epsilon)
    delta = -eta * inv * m / (1-beta1**count)
    parts = -eta * inv[None, :] * cm / (1-beta1**count)
    p = old['p'] + delta
    sign = jnp.sign(old['p'][:width])
    change = H * (jnp.abs(p[:width]) - jnp.abs(old['p'][:width]))
    signed = H * parts[:, :width] * sign
    crossing = change - H * delta[:width] * sign
    errors = jnp.array([jnp.linalg.norm(cm.sum(axis=0)-m),
                        jnp.linalg.norm(parts.sum(axis=0)-delta)])
    new = dict(old, p=p, m=m, v=v, channel_m=cm, count=count,
               positive=old['positive']+jnp.maximum(change, 0),
               negative=old['negative']+jnp.maximum(-change, 0),
               signed=old['signed']+signed, crossing=old['crossing']+crossing,
               force_activity=old['force_activity']+jnp.linalg.norm(channels[:, :width], axis=1),
               step_activity=old['step_activity']+jnp.linalg.norm(parts[:, :width], axis=1),
               identity_max=jnp.maximum(old['identity_max'], errors))
    return new, (change, signed, crossing, errors, inv, parts)


def step(old, x, y, alpha):
    g, r, jc, ec = af.field(old['p'], x, y)
    channels, info = af.split(g, jc, ec)
    new, (change, signed, crossing, errors, inv, parts) = moment_step(old, channels, alpha, gradient=g)
    new['unresolved'] = old['unresolved'] + ~info['resolved']
    width = len(change)
    row = jnp.array([jnp.mean(r*r)/jnp.mean(y*y), H*jnp.mean(jnp.abs(old['p'][:width])),
        jnp.mean(change), jnp.mean(signed[0]), jnp.mean(signed[1]), jnp.mean(crossing),
        jnp.linalg.norm(channels[1, :width]), jnp.linalg.norm(channels[0, :width]),
        jnp.linalg.norm(parts[1, :width]), jnp.linalg.norm(parts[0, :width]),
        errors[0], errors[1], info['resolved'], jnp.mean(inv[:width]), jnp.max(inv[:width])])
    return new, row


def prepare(args):
    archive = args.archive
    manifest = json.loads((archive/'manifest.json').read_text())
    z = np.load(archive/'snapshots.npz')
    at = list(z['steps']).index(600000)
    selected = [i for i, c in enumerate(manifest['cases']) if c['optimizer'] == 'adam']
    if len(selected) != 13:
        raise ValueError('Expected thirteen ordinary Adam targets')
    records = []
    values = {k: [] for k in ('p', 'm', 'v', 'channel_m', 'count', 'y', 'alpha')}
    for i in selected:
        case = manifest['cases'][i]
        if (case['eta'], case['beta1'], case['beta2'], case['epsilon']) != (.002, .9, .999, 1e-8):
            raise ValueError('Unexpected optimizer settings')
        if z['count'][i, at] != 600000 or z['failed'][i, at] or z['p'].shape[-1] != 532:
            raise ValueError('Invalid fork state or unexpected width')
        x, y, _, _ = af.data(case['target'], 2048)
        for am, av in ARMS:
            records.append(dict(case, alpha_m=am, alpha_v=av, fork=600000, nref=128, h=H))
            for k in ('p', 'm', 'v', 'channel_m', 'count'):
                values[k].append(z[k][i, at])
            values['y'].append(y); values['alpha'].append((am, av))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, **{k: np.asarray(v) for k, v in values.items()}, x=x,
         cases=np.array(json.dumps(records)))
    write_json(args.output.with_suffix('.json'), dict(input_sha256=digest(args.output),
        sources={str(archive/k): digest(archive/k) for k in ('manifest.json', 'snapshots.npz')},
        source_hashes=source_hashes(), cases=records,
        issued_utc=datetime.now(timezone.utc).isoformat()))


def predict(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'manifest.json').exists():
        raise ValueError('Predictions are immutable; choose a new directory')
    pack = dict(np.load(args.inputs)); old = initial(pack)
    x, y, alpha = map(jnp.asarray, (pack['x'], pack['y'], pack['alpha']))
    def force(p, yy):
        g, _, jc, ec = af.field(p, x, yy)
        return af.split(g, jc, ec)[0], g
    channels, gradient = jax.jit(jax.vmap(force))(old['p'], y)
    first, _ = jax.jit(jax.vmap(step, in_axes=(0, None, 0, 0)))(old, x, y, alpha)
    save(args.output/'first_step.npz', **jax.device_get(first))
    ordinary_first, _ = jax.jit(jax.vmap(step, in_axes=(0, None, 0, 0)))(old, x, y, jnp.ones_like(alpha))
    phase_two, gradient_two = jax.jit(jax.vmap(force))(ordinary_first['p'], y)
    save(args.output/'driver_phases.npz', phase_zero=np.asarray(channels),
         phase_one=np.asarray(phase_two), gradient_zero=np.asarray(gradient),
         gradient_one=np.asarray(gradient_two), virtual_ordinary_p=np.asarray(ordinary_first['p']))
    forecasts = {}
    # Both F and Q evolve between these two checkpoint-derived phases. All
    # arms receive the same driver phases and retain their complete histories.
    def advance(state, periodic, start, end):
        def body(k, current):
            ch = jnp.where(periodic & (k % 2 == 1), phase_two, channels)
            g = jnp.where(periodic & (k % 2 == 1), gradient_two, gradient)
            return jax.vmap(lambda s, c, a, gg: moment_step(s, c, a, gradient=gg)[0])(current, ch, alpha, g)
        return jax.lax.fori_loop(start, end, body, state)
    advance = jax.jit(advance)
    for name, alternating in (('frozen', False), ('two_phase', True)):
        state = old; start = 0
        for end in sorted({k for k in SAVED if 0 < k <= args.horizon} | {args.horizon}):
            state = advance(state, alternating, start, end)
            forecasts[f'{name}_p_{end}'] = np.asarray(state['p'])
            forecasts[f'{name}_signed_{end}'] = np.asarray(state['signed'])
            start = end
    save(args.output/'forecasts.npz', **forecasts)
    write_json(args.output/'manifest.json', dict(input_sha256=digest(args.inputs),
        horizon=args.horizon, source_hashes=source_hashes(),
        first_step_sha256=digest(args.output/'first_step.npz'),
        driver_phases_sha256=digest(args.output/'driver_phases.npz'),
        forecasts_sha256=digest(args.output/'forecasts.npz'),
        issued_utc=datetime.now(timezone.utc).isoformat(),
        interpretation='Frozen force or two-phase F/Q drivers from fork plus one virtual ordinary update; no future trajectory states.'))


def run(args):
    if args.backend == 'slurm':
        for key in ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'CUDA_VISIBLE_DEVICES'):
            if not os.environ.get(key):
                raise RuntimeError(f'Missing {key}; run inside allocated srun step')
        for kind, identifier in (('job', os.environ['SLURM_JOB_ID']),
                                 ('step', os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])):
            subprocess.run(['scontrol', 'show', kind, identifier], check=True)
        if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
            raise RuntimeError('Expected exactly one allocated GPU')
    elif any(d.platform != 'cpu' for d in jax.devices()):
        raise RuntimeError('CPU verification requested')
    pred = json.loads((args.predictions/'manifest.json').read_text())
    if pred['input_sha256'] != digest(args.inputs) or pred['source_hashes'] != source_hashes():
        raise ValueError('Input/source differs from preissued forecast')
    for name, key in (('first_step.npz', 'first_step_sha256'),
                      ('driver_phases.npz', 'driver_phases_sha256'),
                      ('forecasts.npz', 'forecasts_sha256')):
        if digest(args.predictions/name) != pred[key]:
            raise ValueError(f'Changed preissued prediction archive: {name}')
    if args.horizon > pred['horizon']:
        raise ValueError('Forecast does not cover requested horizon')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'snapshots').mkdir(exist_ok=True)
    pack = dict(np.load(args.inputs)); state = initial(pack); offset = 0
    protocol = dict(input_sha256=digest(args.inputs), prediction_sha256=digest(args.predictions/'manifest.json'),
                    source_hashes=source_hashes(), cases=json.loads(str(pack['cases'])), metrics=METRICS)
    manifest = args.output/'manifest.json'
    protocol = json.loads(json.dumps(protocol))
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError('Changed resume protocol')
    write_json(manifest, protocol)
    if (args.output/'state.npz').exists():
        loaded = dict(np.load(args.output/'state.npz')); offset = int(loaded.pop('offset'))
        state = jax.tree.map(jnp.asarray, loaded)
    else:
        save(args.output/'snapshots/000000.npz', **jax.device_get(state))
    x, y, alpha = map(jnp.asarray, (pack['x'], pack['y'], pack['alpha']))
    def advance(state, steps):
        def body(current, _):
            return jax.vmap(step, in_axes=(0, None, 0, 0))(current, x, y, alpha)
        new, rows = jax.lax.scan(body, state, None, length=steps)
        return new, rows.sum(axis=0), rows.min(axis=0), rows.max(axis=0), rows
    advance = jax.jit(advance, static_argnums=1)
    begun = time.monotonic()
    ends = sorted({1, 10, *range(100, args.horizon+1, 100), args.horizon})
    for end in ends:
        if end <= offset:
            continue
        if end > args.horizon:
            continue
        state, total, lo, hi, rows = jax.device_get(advance(state, end-offset))
        if not np.isfinite(state['p']).all() or not np.isfinite(state['v']).all() or state['unresolved'].any():
            raise FloatingPointError(f'Invalid or unresolved state at {end}')
        save(args.output/f'trace_{end:06d}.npz', start=np.array(offset), end=np.array(end),
             total=total, minimum=lo, maximum=hi,
             **({'dense': rows} if end <= 2000 else {}))
        offset = end
        if end in SAVED or end == args.horizon:
            save(args.output/f'snapshots/{end:06d}.npz', **state)
        if end % 100 == 0 or end == args.horizon:
            save(args.output/'state.npz', offset=np.array(offset), **state)
            write_json(args.output/'status.json', dict(offset=offset, complete=offset==args.horizon,
                invocation_seconds=time.monotonic()-begun, identity_max=state['identity_max'].max(axis=0).tolist()))
            if time.monotonic()-begun >= args.max_seconds:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'predict', 'run'))
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon', type=int, default=20000)
    parser.add_argument('--max-seconds', type=float, default=300)
    parser.add_argument('--backend', choices=('cpu', 'slurm'), default='slurm')
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError('Set JAX_ENABLE_X64=true')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
