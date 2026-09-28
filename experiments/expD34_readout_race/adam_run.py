"""Paired D34 GD/Adam sweep with every-update force and movement accounting."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_forces as af, targets
from .run import verify_gpu, write_json

STAGES = ('primary', 'rates', 'controls', 'epsilon')
SNAPSHOTS = {0, 1, 10, 100, 200, 1000, 2000, 5000, 10000, *range(20000, 600001, 20000)}
METRICS = ('half_mse', 'relative_mse', 'mean_gamma', 'max_gamma', 'readout_l2',
           'fine_residual_norm', 'coarse_residual_norm', 'tracking_metric', 'coarse_resolved',
           'direct_fine_norm', 'balanced_coarse_norm')
METRICS += tuple(f'{stage}_{channel}_norm' for stage in ('raw', 'moment', 'scaled_current', 'step')
                 for channel in ('total', *af.CHANNELS))
METRICS += tuple(f'step_{channel}_projection' for channel in af.CHANNELS)
METRICS += tuple(f'outward_{channel}' for channel in af.CHANNELS)
METRICS += ('outward_total', 'actual_gamma_increment', 'crossing_correction',
            'effective_coupling', 'epsilon_dominated_fraction', 'inverse_mean', 'inverse_max')
METRICS += tuple(f'residual_mode_{k}' for k in range(10))
METRICS += ('gradient_identity_error', 'moment_identity_error', 'step_identity_error')


def cases(stage, index):
    def case(target, seed, optimizer, eta=.002, epsilon=1e-8):
        return dict(target=target, seed=seed, optimizer=optimizer, eta=eta, epsilon=epsilon,
                    beta1=0. if optimizer in ('gd', 'adaptive_only') else .9, beta2=.999,
                    adaptive=optimizer in ('adam', 'adaptive_only'))
    if stage == 'primary':
        assert index in range(5)
        return [case(t, index, o) for t in af.TARGETS for o in ('gd', 'adam')]
    if stage == 'rates':
        assert index in range(2)
        return [case(t, 0, 'adam', eta=(.0002, .001)[index]) for t in af.TARGETS]
    if stage == 'controls':
        assert index in range(3)
        return [case(t, index, o) for t in af.CONTROLS for o in ('momentum', 'adaptive_only')]
    if stage == 'epsilon':
        assert index == 0
        return [case(t, 0, 'adam', epsilon=1e-12) for t in af.CONTROLS]
    raise ValueError(stage)


def initial(p):
    p = jnp.asarray(p); width = (len(p)-1)//3
    return dict(p=p, m=jnp.zeros_like(p), v=jnp.zeros_like(p),
        channel_m=jnp.zeros((3, len(p))), count=jnp.array(0, dtype=jnp.int64),
        positive=jnp.zeros(width), negative=jnp.zeros(width), path=jnp.array(0.),
        channel_path=jnp.zeros(3), raw_channel_path=jnp.zeros(3), signed_channels=jnp.zeros(3),
        crossing=jnp.array(0.), step_energy=jnp.array(0.), projected_energy=jnp.zeros(3),
        identity_max=jnp.zeros(3), unresolved_steps=jnp.array(0, dtype=jnp.int64),
        previous_loss=jnp.array(jnp.inf), loss_increases=jnp.array(0, dtype=jnp.int64),
        failed=jnp.array(0, dtype=jnp.int64))


def one_step(old, x, y, modes, settings):
    eta, beta1, beta2, epsilon, adaptive = settings
    p = old['p']; width = (len(p)-1)//3
    g, r, jc, ec = af.field(p, x, y)
    channels, info = af.split(g, jc, ec)
    count = old['count']+1
    m, v, cm, mh, ch, inverse = af.moments(g, channels, old['m'], old['v'], old['channel_m'],
                                         count, beta1, beta2, epsilon, adaptive)
    delta = -eta*inverse*mh
    channel_delta = -eta*inverse*ch
    pn = p+delta
    ga = g[:width]; ca = channels[:, :width]; ma = mh[:width]; cma = ch[:, :width]
    da = delta[:width]; dca = channel_delta[:, :width]
    sign = jnp.sign(p[:width]); change = jnp.abs(pn[:width])-jnp.abs(p[:width])
    outward = jnp.mean(dca*sign, axis=1)
    crossing = jnp.mean(change)-jnp.mean(sign*da)
    energy = da @ da
    projection = dca @ da
    errors = jnp.array([jnp.linalg.norm(channels.sum(axis=0)-g),
                        jnp.linalg.norm(cm.sum(axis=0)-m),
                        jnp.linalg.norm(channel_delta.sum(axis=0)-delta)])
    q1 = x/jnp.sqrt(jnp.mean(x*x))
    fine2 = jnp.mean((r-ec[0]-ec[1]*q1)**2)
    loss = .5*jnp.mean(r*r)
    stage_norms = jnp.concatenate([jnp.r_[jnp.linalg.norm(total), jnp.linalg.norm(parts, axis=1)]
        for total, parts in ((ga, ca), (ma, cma), (inverse[:width]*ga, inverse[:width]*ca), (da, dca))])
    # These are measurements of the state BEFORE the associated update.
    row = jnp.concatenate((jnp.array([loss, 2*loss/jnp.mean(y*y), jnp.mean(jnp.abs(p[:width])),
        jnp.max(jnp.abs(p[:width])), jnp.linalg.norm(p[2*width:3*width]), jnp.sqrt(fine2),
        jnp.linalg.norm(ec), jnp.sqrt(jnp.maximum(0., info['z'] @ info['C'] @ info['z'])),
        info['resolved'], jnp.linalg.norm(info['fine'][:width]), jnp.linalg.norm(info['balanced'][:width])]),
        stage_norms, jnp.where(energy>0, projection/energy, jnp.nan), outward,
        jnp.array([jnp.mean(sign*da), jnp.mean(change), crossing,
            jnp.where(fine2>0, ca[0] @ ca[0]/fine2, jnp.nan),
            jnp.mean(jnp.sqrt(v[:width]/(1-beta2**count))<=epsilon),
            jnp.mean(inverse[:width]), jnp.max(inverse[:width])]), modes.T @ r/len(x), errors))
    new = dict(p=pn, m=m, v=v, channel_m=cm, count=count,
        positive=old['positive']+jnp.maximum(change, 0.), negative=old['negative']+jnp.maximum(-change, 0.),
        path=old['path']+jnp.linalg.norm(da), channel_path=old['channel_path']+jnp.linalg.norm(dca, axis=1),
        raw_channel_path=old['raw_channel_path']+eta*jnp.linalg.norm(ca, axis=1),
        signed_channels=old['signed_channels']+outward, crossing=old['crossing']+crossing,
        step_energy=old['step_energy']+energy, projected_energy=old['projected_energy']+projection,
        identity_max=jnp.maximum(old['identity_max'], errors),
        unresolved_steps=old['unresolved_steps']+~info['resolved'], previous_loss=loss,
        loss_increases=old['loss_increases']+(loss>old['previous_loss']+1e-14), failed=old['failed'])
    finite = jnp.isfinite(loss)&jnp.all(jnp.isfinite(g))&jnp.all(jnp.isfinite(pn))
    active = (old['failed']==0)&finite
    new = jax.tree.map(lambda n, o: jnp.where(active, n, o), new, old)
    new['count'] = count
    new['failed'] = jnp.where((old['failed']==0)&~finite, count, old['failed'])
    return new, jnp.where(active, row, jnp.nan)


@lru_cache(maxsize=8)
def advance_factory(m):
    x = jnp.asarray(targets.grid(m))
    mapping = targets.polynomial_map(np.asarray(x))
    modes = jnp.asarray(np.polynomial.legendre.legvander(np.asarray(x), 9) @ mapping)
    def advance(old, y, settings, steps):
        zero = jnp.zeros(len(METRICS))
        def body(_, carry):
            current, _, lo, hi = carry
            new, row = one_step(current, x, y, modes, settings)
            return new, row, jnp.fmin(lo, row), jnp.fmax(hi, row)
        return jax.lax.fori_loop(0, steps, body, (old, zero, jnp.full_like(zero, jnp.inf), jnp.full_like(zero, -jnp.inf)))
    return jax.jit(jax.vmap(advance, in_axes=(0, 0, 0, None)))


def schedule(end):
    return sorted({*range(1, min(200, end)+1), *range(210, min(2000, end)+1, 10),
                   *range(2100, min(20000, end)+1, 100), *range(21000, end+1, 1000),
                   *(s for s in SNAPSHOTS if 0<s<=end), end})


def atomic_npz(destination, **arrays):
    temporary = destination.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination)


def run(args):
    if not jax.config.x64_enabled:
        raise ValueError('Set JAX_ENABLE_X64=true')
    output = args.output/f'{args.stage}_{args.index}'
    output.mkdir(parents=True, exist_ok=True)
    verify_gpu(output)
    records = cases(args.stage, args.index)
    pp = []; yy = []; normalizers = []
    for case in records:
        z, d = targets.initial(128, 24, case['seed'])
        pp.append(np.r_[z.ravel(), d])
        x, y, _, scale = af.data(case['target'], args.samples)
        yy.append(y); normalizers.append(scale)
    pp = np.array(pp); yy = np.array(yy)
    settings = np.array([[c[k] for k in ('eta', 'beta1', 'beta2', 'epsilon', 'adaptive')] for c in records])
    manifest = dict(cases=records, m=args.samples, width=177, max_updates=600000,
        target_normalizers=normalizers, initial_hash=targets.array_hash(pp), data_hash=targets.array_hash(x, yy),
        source_commit=os.environ.get('RACE_SOURCE_COMMIT'), metrics=list(METRICS),
        trace_convention='row at pre-update index interval_end-1; extrema cover [start,end)',
        balance='GD reference; entire empirical residual beyond constant/linear')
    path = output/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Changed resume protocol')
    write_json(path, manifest)
    state = jax.vmap(initial)(jnp.asarray(pp))
    snapshots = {0: jax.device_get(state)}
    starts = []; ends = []; rows = []; lows = []; highs = []; step = 0
    if (output/'state.npz').exists():
        f = dict(np.load(output/'state.npz'))
        step = int(f.pop('cursor')); state = jax.tree.map(jnp.asarray, f)
        f = dict(np.load(output/'snapshots.npz')); ss = f.pop('steps')
        snapshots = {int(s): {k: v[:, i] for k, v in f.items()} for i, s in enumerate(ss) if s<=step}
        f = dict(np.load(output/'trace.npz'))
        keep = f['ends']<=step
        f = {k: v[keep] if k in ('starts', 'ends') else v[:, keep] for k, v in f.items()}
        starts = list(f['starts']); ends = list(f['ends'])
        rows = list(f['values'].transpose(1, 0, 2)); lows = list(f['minimum'].transpose(1, 0, 2)); highs = list(f['maximum'].transpose(1, 0, 2))
    advance = advance_factory(args.samples)
    begun = time.monotonic()
    def save():
        current = jax.device_get(state)
        snapshots[step] = current
        ss = sorted(snapshots)
        atomic_npz(output/'snapshots.npz', steps=np.array(ss),
                   **{k: np.stack([snapshots[s][k] for s in ss], axis=1) for k in current})
        atomic_npz(output/'trace.npz', starts=np.array(starts), ends=np.array(ends),
                   values=np.stack(rows, axis=1), minimum=np.stack(lows, axis=1), maximum=np.stack(highs, axis=1))
        atomic_npz(output/'state.npz', cursor=np.array(step), **current)
        write_json(output/'status.json', dict(cursor=step, requested_end=args.end_step, complete=step==600000,
            cases=len(records), failed=current['failed'].tolist(), unresolved_steps=current['unresolved_steps'].tolist(),
            identity_max=current['identity_max'].max(axis=0).tolist(),
            motion_identity_max=float(np.max(abs(current['positive']-current['negative']-
                (abs(current['p'][:, :177])-abs(pp[:, :177]))))),
            loss_increases=current['loss_increases'].tolist(), invocation_seconds=time.monotonic()-begun))
    for end in schedule(args.end_step):
        if end<=step:
            continue
        new, row, lo, hi = advance(state, jnp.asarray(yy), jnp.asarray(settings), end-step)
        state, row, lo, hi = jax.device_get((new, row, lo, hi))
        starts.append(step); ends.append(end); rows.append(row); lows.append(lo); highs.append(hi)
        step = end
        if step in SNAPSHOTS:
            snapshots[step] = state
        if step%20000==0 or step==args.end_step or time.monotonic()-begun>args.max_seconds:
            save()
            print(json.dumps(dict(batch=output.name, step=step, seconds=time.monotonic()-begun,
                relative_mse=np.round(row[:, 1], 8).tolist())), flush=True)
        if time.monotonic()-begun>args.max_seconds:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--end-step', type=int, default=600000)
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--max-seconds', type=float, default=3400)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
