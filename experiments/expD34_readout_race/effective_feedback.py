"""Matched effective-force feedback experiments with exact discrete travel.

The finite polynomial basis defines the intervention; ordinary gradients always
use the complete empirical residual. Reports are authored separately.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_forces as af, adam_run as ar, targets, transport
from .run import write_json

ARMS = ('joint', 'freeze_map', 'clamp_residual')
THRESHOLDS = (1., 3.2, 16.)
CONTEXT_AXES = dict(x=None, q=None, y=0, p0=0, T_a0=0, eH0=0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    """Extract labeled forks, or the unchanged fresh-seed initialization."""
    seeds = [int(v) for v in args.seeds.split(',')]
    starts = [int(v) for v in args.starts.split(',')]
    names = args.targets.split(',') if args.targets else list(af.TARGETS)
    if not set(names) <= set(af.TARGETS):
        raise ValueError('Unknown target')
    cases, pp, yy, hashes = [], [], [], {}
    for seed in seeds:
        if args.fresh:
            z, d = targets.initial(128, 24, seed)
            initial_p = np.r_[z.ravel(), d]
            if starts != [0]:
                raise ValueError('Fresh inputs start at zero')
        else:
            folder = args.source / f'primary_{seed}'
            records = json.loads((folder/'manifest.json').read_text())['cases']
            hashes[str(folder/'manifest.json')] = digest(folder/'manifest.json')
            hashes[str(folder/'snapshots.npz')] = digest(folder/'snapshots.npz')
            archive = np.load(folder/'snapshots.npz')
        for start in starts:
            for name in names:
                if args.fresh:
                    p = initial_p
                else:
                    ci = [i for i, c in enumerate(records) if c['target'] == name
                          and c['optimizer'] == 'gd' and c['eta'] == .002]
                    si = np.flatnonzero(archive['steps'] == start)
                    if len(ci) != 1 or len(si) != 1:
                        raise ValueError(f'Nonunique or missing source: {seed}/{name}/{start}')
                    p = archive['p'][ci[0], si[0]]
                x, y, _, scale = af.data(name, args.samples)
                cases.append(dict(target=name, seed=seed, start=start, eta=.002,
                                  width=(len(p)-1)//3, target_scale=scale))
                pp.append(p); yy.append(y)
        if not args.fresh:
            archive.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    ar.atomic_npz(args.output, p=np.asarray(pp), x=x, y=np.asarray(yy),
                  cases=np.array(json.dumps(cases)), sources=np.array(json.dumps(hashes)))
    print(json.dumps(dict(inputs=str(args.output), cases=len(cases), sha256=digest(args.output))))


def load_inputs(path, selection=None):
    with np.load(path) as data:
        cases = json.loads(str(data['cases']))
        indices = np.arange(len(cases)) if selection is None else np.asarray(selection, dtype=int)
        if len(set(indices.tolist())) != len(indices) or not len(indices):
            raise ValueError('Input selection must be nonempty and unique')
        return (data['p'][indices].copy(), data['x'].copy(), data['y'][indices].copy(),
                [cases[i] for i in indices])


def initial(p):
    w = (p.shape[-1]-1)//3
    zero = jnp.zeros(w, dtype=p.dtype)
    hit = jnp.where(jnp.abs(p[:w])[None, :] >= jnp.asarray(THRESHOLDS)[:, None], 0, -1)
    return dict(p=p, positive=zero, negative=zero, effective=zero, tracking=zero,
                omitted=zero, crossing=zero, first_hit=hit, path=jnp.array(0.),
                count=jnp.array(0, dtype=jnp.int64), failed=jnp.array(0, dtype=jnp.int32))


def advance_factory(arm, eta):
    from . import effective_feedback_kernel as kernel

    def one(state, context, length):
        w = (len(state['p'])-1)//3

        def step(_, old):
            if arm == 'joint':
                gradient, channel = kernel.field(old['p'], context, arm)
            else:
                # Evaluate the common first update by the same arithmetic,
                # avoiding a fork difference from dense versus factored products.
                gradient, channel = jax.lax.cond(
                    old['count'] == 0,
                    lambda p: kernel.field(p, context, 'joint'),
                    lambda p: kernel.field(p, context, arm), old['p'])
            delta = -eta*gradient
            pn = old['p']+delta
            change = jnp.abs(pn[:w])-jnp.abs(old['p'][:w])
            sign = jnp.sign(old['p'][:w])
            ef = -eta*sign*channel['applied_effective_a']
            tr = -eta*sign*channel['tracking_a']
            om = -eta*sign*channel['omitted_a']
            count = old['count']+1
            hit = (jnp.abs(pn[:w])[None, :] >= jnp.asarray(THRESHOLDS)[:, None])
            first_hit = jnp.where((old['first_hit'] < 0) & hit, count, old['first_hit'])
            new = dict(p=pn, positive=old['positive']+jnp.maximum(change, 0.),
                       negative=old['negative']+jnp.maximum(-change, 0.),
                       effective=old['effective']+ef, tracking=old['tracking']+tr,
                       omitted=old['omitted']+om,
                       crossing=old['crossing']+change-sign*delta[:w],
                       first_hit=first_hit, path=old['path']+jnp.linalg.norm(delta[:w]),
                       count=count, failed=old['failed'])
            finite = jnp.all(jnp.isfinite(pn)) & jnp.all(jnp.isfinite(gradient))
            resolved = channel['coarse_resolved']
            valid = finite & resolved & (old['failed'] == 0)
            new = jax.tree.map(lambda a, b: jnp.where(valid, a, b), new, old)
            new['failed'] = jnp.where(old['failed'] != 0, old['failed'],
                                     jnp.where(~finite, 1, jnp.where(~resolved, 2, 0)))
            return new

        return jax.lax.fori_loop(0, length, step, state)

    return jax.jit(jax.vmap(one, in_axes=(0, CONTEXT_AXES, None)))


def context_for(pp, x, yy, degree):
    from . import effective_feedback_kernel as kernel
    q = jnp.asarray(transport.basis(np.asarray(x), degree))
    x = jnp.asarray(x)
    arrays = jax.vmap(lambda p, y: kernel.fork_context(p, x, y, q=q))(jnp.asarray(pp), jnp.asarray(yy))
    # Shared immutable arrays are not replicated along the batching axis.
    arrays['x'] = x; arrays['q'] = q
    return arrays


@lru_cache(maxsize=3)
def measure_factory(arm):
    from . import effective_feedback_kernel as kernel

    def one(p, ctx):
        gradient, c = kernel.field(p, ctx, arm)
        w = (len(p)-1)//3
        fa = c['applied_effective_a']
        norm = jnp.linalg.norm(fa)
        participation = jnp.where(norm > 0, norm**4/(w*jnp.sum(fa**4)), 0.)
        return dict(relative_mse=2*c['loss']/jnp.mean(ctx['y']**2),
                    mean_gamma=jnp.mean(jnp.abs(p[:w])), max_gamma=jnp.max(jnp.abs(p[:w])),
                    readout_rms=jnp.sqrt(jnp.mean(p[2*w:3*w]**2)),
                    effective_norm=norm, reference_effective_norm=jnp.linalg.norm(c['effective_a']),
                    outward=-jnp.mean(jnp.sign(p[:w])*fa), participation=participation,
                    tracking_norm=jnp.linalg.norm(c['tracking_a']), omitted_norm=jnp.linalg.norm(c['omitted_a']),
                    coarse_min=c['coarse_min_eigenvalue'], coarse_resolved=c['coarse_resolved'],
                    eH=c['eH'], effective_a=fa, reference_effective_a=c['effective_a'],
                    tracking_a=c['tracking_a'], omitted_a=c['omitted_a'],
                    reconstruction=jnp.linalg.norm(gradient[:w]-fa-c['tracking_a']-c['omitted_a']))

    return jax.jit(jax.vmap(one, in_axes=(0, CONTEXT_AXES)))


def verify_backend(output, backend):
    if not jax.config.x64_enabled:
        raise ValueError('This experiment requires JAX_ENABLE_X64=true')
    devices = jax.devices()
    if backend == 'slurm':
        from .run import verify_gpu
        verify_gpu(output)
    elif backend == 'modal':
        if not os.environ.get('MODAL_TASK_ID') or len(devices) != 1 or devices[0].platform != 'gpu':
            raise RuntimeError('Expected one GPU inside an actual Modal worker')
    elif backend == 'cpu':
        if any(d.platform != 'cpu' for d in devices):
            raise RuntimeError('CPU verification must not expose GPUs')
    write_json(output/'environment.json', dict(backend=backend, devices=[str(d) for d in devices],
               jax=jax.__version__, numpy=np.__version__, x64=jax.config.x64_enabled))


def run(args):
    from . import effective_feedback_kernel as kernel
    selection = [int(v) for v in args.indices.split(',')] if args.indices else None
    pp, x, yy, cases = load_inputs(args.inputs, selection)
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    verify_backend(out, args.backend)
    begun = time.monotonic()
    if args.eta not in (.002, .001):
        raise ValueError('Use the primary or matched-time half step')
    factor = round(.002/args.eta)
    manifest = dict(cases=cases, arm=args.arm, eta=args.eta, reference_eta=.002,
                    degree=args.degree, samples=len(x), input_sha256=digest(args.inputs),
                    selected_indices=selection, source_commit=os.environ.get('RACE_SOURCE_COMMIT'),
                    runner_sha256=digest(__file__), prediction_manifest=args.predictions,
                    kernel_sha256=digest(kernel.__file__),
                    prediction_sha256=(digest(args.predictions) if Path(args.predictions).is_file() else None),
                    thresholds=THRESHOLDS, coordinates='raw a,b,c,d',
                    first_hit_units='actual updates; multiply by eta for physical time',
                    force='finite polynomial basis; ordinary tracking and omitted residual retained')
    mp = out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError('Changed experiment on resume')
    write_json(mp, manifest)
    context = context_for(pp, x, yy, args.degree)
    # Materialize setup before timing steady-state throughput.
    jax.block_until_ready(context)
    state = jax.vmap(initial)(jnp.asarray(pp)); offset = 0; previous_seconds = 0.
    if (out/'state.npz').exists():
        with np.load(out/'state.npz') as old:
            offset = int(old['offset'])
            state = {k: jnp.asarray(old[k]) for k in state}
        previous_seconds = json.loads((out/'status.json').read_text())['seconds']
    if offset > args.horizon:
        raise ValueError('Cannot resume beyond requested horizon')
    advance = advance_factory(args.arm, args.eta)
    measure = measure_factory(args.arm)
    snapshot_dir = out/'snapshots'; snapshot_dir.mkdir(exist_ok=True)

    def save(include_metrics=True):
        host = jax.device_get(state)
        arrays = dict(offset=np.array(offset), **host)
        if include_metrics:
            diagnostic = jax.device_get(measure(state['p'], context))
            arrays.update({'metric_'+k: v for k, v in diagnostic.items()})
        ar.atomic_npz(snapshot_dir/f'{offset:09d}.npz', **arrays)
        ar.atomic_npz(out/'state.npz', offset=np.array(offset), **host)
        w = (pp.shape[1]-1)//3
        displacement = np.abs(host['p'][:, :w])-np.abs(pp[:, :w])
        motion_error = np.max(np.abs(host['positive']-host['negative']-displacement))
        signed_error = np.max(np.abs(host['effective']+host['tracking']+host['omitted']+host['crossing']-displacement))
        status = dict(offset=offset, horizon=args.horizon, complete=offset == args.horizon,
                      valid=bool(np.all(host['failed'] == 0)), failed=host['failed'].tolist(),
                      seconds=previous_seconds+time.monotonic()-begun,
                      motion_identity=float(motion_error), signed_identity=float(signed_error))
        write_json(out/'status.json', status)
        return status

    if offset == 0:
        save()
    schedule = sorted({v for v in (1, 2, 10, 100, 1000) if v <= args.horizon}
                      | set(range(1000, args.horizon+1, args.stride)) | {args.horizon})
    timed = []
    for end in schedule:
        if end <= offset:
            continue
        started = time.monotonic()
        state = advance(state, context, (end-offset)*factor)
        jax.block_until_ready(state)
        timed.append(dict(start=offset, end=end, seconds=time.monotonic()-started))
        offset = end
        status = save()
        if offset <= 1000 or offset % 10000 == 0 or offset == args.horizon:
            print(json.dumps(status), flush=True)
        if time.monotonic()-begun >= args.max_seconds or not status['valid']:
            break
    write_json(out/'timings.json', dict(chunks=timed, invocation_seconds=time.monotonic()-begun))
    if not np.all(np.asarray(state['failed']) == 0):
        raise RuntimeError('Numerical failure; states retained with explicit failure codes')


def export_fork(args):
    """Make the same typed input pack from a newly trained reference snapshot."""
    manifest = json.loads((args.source/'manifest.json').read_text())
    if manifest['arm'] != 'joint':
        raise ValueError('Fresh forks require an ordinary-GD source')
    path = args.source/'snapshots'/f'{args.offset:09d}.npz'
    with np.load(path) as data:
        if np.any(data['failed']):
            raise ValueError('Cannot fork failed states')
        pp = data['p'].copy()
    cases = [dict(c, start=c['start']+args.offset) for c in manifest['cases']]
    x = targets.grid(manifest['samples'])
    yy = np.stack([af.data(c['target'], len(x))[1] for c in cases])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ar.atomic_npz(args.output, p=pp, x=x, y=yy, cases=np.array(json.dumps(cases)),
                  sources=np.array(json.dumps({str(path): digest(path)})))


def predict(args):
    """Issue immutable per-fork predictions before launching continuations."""
    from datetime import datetime, timezone
    from .effective_feedback_predict import predict_case, DEFAULT_HORIZONS
    selection = [int(v) for v in args.indices.split(',')] if args.indices else None
    pp, x, yy, cases = load_inputs(args.inputs, selection)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    records = []
    for i, (p, y, case) in enumerate(zip(pp, yy, cases)):
        path = args.output/f'case_{i:03d}.npz'
        if path.exists():
            raise FileExistsError(path)
        result = predict_case(p, x, y, degree=args.degree, eta=args.eta,
                              horizons=DEFAULT_HORIZONS)
        ar.atomic_npz(path, **result['arrays'])
        record = dict(case=case, file=path.name, sha256=digest(path),
                      issued=datetime.now(timezone.utc).isoformat(), **result['metadata'])
        write_json(path.with_suffix('.json'), record)
        records.append(record)
        print(json.dumps(dict(case=i, target=case['target'], seed=case['seed'],
                              start=case['start'], issued=record['issued'])), flush=True)
    write_json(manifest_path, dict(input_sha256=digest(args.inputs),
               selected_indices=selection, degree=args.degree, eta=args.eta,
               source_commit=os.environ.get('RACE_SOURCE_COMMIT'), records=records))


def analyze(args):
    rows = []
    for path in sorted(args.source.glob('**/manifest.json')):
        manifest = json.loads(path.read_text())
        if 'arm' not in manifest or 'cases' not in manifest:
            continue
        folder = path.parent
        for snapshot in sorted((folder/'snapshots').glob('*.npz')):
            with np.load(snapshot) as data:
                offset = int(data['offset'])
                for i, case in enumerate(manifest['cases']):
                    p = data['p'][i]; w = (len(p)-1)//3
                    row = dict(**case, arm=manifest['arm'], degree=manifest['degree'],
                               run_eta=manifest['eta'], offset=offset, time=.002*offset,
                               run=str(folder), failed=int(data['failed'][i]))
                    for key in data.files:
                        if key.startswith('metric_') and data[key].ndim == 1:
                            row[key[7:]] = float(data[key][i])
                    row.update(positive=float(data['positive'][i].mean()),
                               negative=float(data['negative'][i].mean()),
                               effective_travel=float(data['effective'][i].mean()),
                               tracking_travel=float(data['tracking'][i].mean()),
                               omitted_travel=float(data['omitted'][i].mean()),
                               crossing=float(data['crossing'][i].mean()))
                    for ti, threshold in enumerate(THRESHOLDS):
                        hit = data['first_hit'][i, ti]
                        row[f'initial_fraction_{threshold:g}'] = float(np.mean(hit == 0))
                        row[f'new_fraction_{threshold:g}'] = float(np.mean(hit > 0))
                        row[f'fraction_{threshold:g}'] = float(np.mean(abs(p[:w]) >= threshold))
                    rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No experiment snapshots found')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (args.output/'trajectories.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(rows=len(rows), output=str(args.output))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.set_defaults(function=prepare)
    p.add_argument('--source', type=Path); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', default='0,1,2,3,4'); p.add_argument('--starts', default='100000,400000,600000')
    p.add_argument('--targets'); p.add_argument('--samples', type=int, default=2048)
    p.add_argument('--fresh', action='store_true')
    p = sub.add_parser('run'); p.set_defaults(function=run)
    p.add_argument('--inputs', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--indices'); p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--degree', type=int, default=65); p.add_argument('--eta', type=float, default=.002)
    p.add_argument('--horizon', type=int, required=True); p.add_argument('--stride', type=int, default=1000)
    p.add_argument('--max-seconds', type=float, default=1500)
    p.add_argument('--backend', choices=('cpu', 'slurm', 'modal'), required=True)
    p.add_argument('--predictions', default='unissued-verification-only')
    p = sub.add_parser('export-fork'); p.set_defaults(function=export_fork)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--offset', type=int, required=True)
    p = sub.add_parser('predict'); p.set_defaults(function=predict)
    p.add_argument('--inputs', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--indices'); p.add_argument('--degree', type=int, default=65)
    p.add_argument('--eta', type=float, default=.002)
    p = sub.add_parser('analyze'); p.set_defaults(function=analyze)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.function(args)


if __name__ == '__main__':
    main()
