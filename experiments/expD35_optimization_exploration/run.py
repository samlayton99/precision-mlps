"""Small manifest-driven batched sweep with immutable cases and exact resume."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from experiments.expD34_readout_race.run import write_json, verify_gpu
from . import core

RATES = [a*10.**b for b in range(-5, 0) for a in (1, 3)]
EVAL_COLUMNS = ('validation_mse', 'validation_relative_mse', 'lambda_q10', 'lambda_median',
                'lambda_q90', 'readout_l2', 'max_gamma')


def case(**kwargs):
    return dict(dict(n=128, seed=0, target='sine', optimizer='adam', coordinates='individual',
        eta=.001, initialization='reference_xavier', slope_initialization='physical_xavier',
        epsilon=1e-15, beta1=.9, beta2=.999, ema_alpha=.98, ema_strength=0.,
        ema_location=1, ema_normalized=False, sampling='full', batch_size=1024,
        schedule='constant', reset='none'), **kwargs)


def baseline():
    return [case(optimizer=opt, coordinates=coord, target=target, eta=eta)
            for opt in ('gd', 'adam') for coord in core.COORDINATES
            for target in ('sine', 'mixed', 'moment9') for eta in RATES]


def key(config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    return f"{config['optimizer']}_N{config['n']}_s{config['seed']}_{config['coordinates']}_{config['target']}_{digest}"


def save(path, **arrays):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno()); size = stream.tell()
    if not size or tmp.stat().st_size != size:
        raise OSError(f'Incomplete checkpoint: {tmp}')
    tmp.replace(path)


def restore(path):
    with np.load(path) as data:
        return {k:jnp.asarray(data[k]) for k in data.files if k!='step'}, int(data['step'])


def finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def group_signature(c):
    return tuple(c[k] for k in ('n', 'coordinates', 'optimizer', 'target', 'sampling', 'batch_size', 'schedule', 'reset'))


def prepare(root, config):
    folder = root/key(config); folder.mkdir(parents=True, exist_ok=True)
    if (folder/'case.json').exists():
        assert json.loads((folder/'case.json').read_text()) == config, folder
    else:
        write_json(folder/'case.json', config)
    if (folder/'state.npz').exists():
        previous, at=restore(folder/'state.npz')
        return folder, dict(core.initialize(config), **previous), at
    state = core.initialize(config)
    if 'origin' in config:
        origin = Path(config['origin']['checkpoint'])
        if hashlib.sha256(origin.read_bytes()).hexdigest() != config['origin']['sha256']:
            raise ValueError('Changed origin checkpoint')
        parent = json.loads((origin.parent/'case.json').read_text())
        previous, _ = restore(origin)
        if parent['n'] != config['n'] or parent['target'] != config['target']:
            raise ValueError('Incompatible width/target in handoff')
        g = core.old.geometry(config['n'])
        c, gamma = core.physical(previous['z'], g, parent['coordinates'])
        state['z'] = jnp.asarray(core.encode(np.asarray(c), np.asarray(gamma), g, config['coordinates']))
        if config['origin'].get('carry_optimizer', True):
            if (parent['coordinates'], parent['optimizer']) != (config['coordinates'], config['optimizer']):
                raise ValueError('Optimizer state cannot be copied across maps/algorithms')
            state = dict(previous, eta=jnp.array(config['eta']))
        # EMA interventions start from the current gradient, not a hidden warm history.
        x = jnp.linspace(-1, 1, 16*config['n']+1)
        grad = core.field(state['z'], x, core.target(x, config['target']), g, config['coordinates'])[1]
        state['ema'] = grad
        hp = core.hyperparameters(dict(config, ema_strength=0.))
        state['post_ema'] = core.optimizer_direction(state, grad, hp, config['optimizer'])[0]
    save(folder/'initial.npz', **state, step=0)
    save(folder/'state.npz', **state, step=0)
    return folder, state, 0


def advance(root, cases, frontier, deadline=float('inf')):
    loaded = [prepare(root, c) for c in cases]
    assert len({s for _,_,s in loaded}) == 1, 'Group by current horizon before advancing'
    at = loaded[0][2]; config = cases[0]
    if at >= frontier:
        return True
    if config['reset']!='none':
        raise ValueError('Recycling hook is not enabled yet')
    stack = lambda xs: jax.tree.map(lambda *a:jnp.stack(a), *xs)
    states = stack([s for _,s,_ in loaded]); hp = stack([core.hyperparameters(c) for c in cases])
    evaluate = core.evaluate(config['n'], config['coordinates'], config['target'])
    begin = time.monotonic()
    while at < frontier:
        if time.monotonic() >= deadline:
            return False
        # Every early state, then 100/1000 and regular saved diagnostic states.
        if at < 20: end = at+1
        elif at < 100: end = min(100, frontier)
        else:
            stride = 1000 if at < 20000 else 5000
            end = min((at//stride+1)*stride, frontier)
        if config['schedule']!='constant': end=min(end,(at//3000+1)*3000)
        kernel = core.chunk(config['n'], config['coordinates'], config['optimizer'], config['target'],
                            end-at, config['sampling'], config['batch_size'])
        states, traces = kernel(states, hp, at)
        jax.block_until_ready(states)
        traces = np.asarray(traces)
        agreement_data=None
        if config['schedule']!='constant' and end%3000==0:
            keys=jax.vmap(lambda k:jax.random.fold_in(k,end+9127))(states['key'])
            agreement_data=core.agreement(config['n'],config['coordinates'],config['target'],config['batch_size'])(states['z'],keys)
            metric=agreement_data[0][:,0,0]
            states['agreement_ema']=jnp.where(jnp.isfinite(metric),.9*states['agreement_ema']+.1*metric,states['agreement_ema'])
            ceiling=jnp.asarray([c.get('eta_ceiling',c['eta']) for c in cases])
            threshold=jnp.asarray([c.get('agreement_threshold',.9) for c in cases])
            factor=jnp.where((states['agreement_ema']<threshold)&(end>1000),1/.9,.9)
            if config['schedule']=='decay': factor=jnp.full_like(factor,.9)
            states['eta']=jnp.where(jnp.isfinite(metric),jnp.clip(states['eta']*factor,1e-16,ceiling),states['eta'])
            agreement_data=jax.tree.map(np.asarray,agreement_data)
        # Validation snapshots include all selection-window endpoints.
        evaluations = np.asarray(evaluate(states['z'])) if end>=1000 or end==frontier else None
        for i,(folder,_,_) in enumerate(loaded):
            state = jax.tree.map(lambda a:np.asarray(a[i]), states)
            save(folder/f'trace_{at:09d}_{end:09d}.npz', trace=traces[i], start=at, end=end)
            save(folder/f'snapshot_{end:09d}.npz', z=state['z'], step=end)
            if agreement_data is not None:
                save(folder/f'agreement_{end:09d}.npz',statistics=agreement_data[0][i],
                     batch_gradients=agreement_data[1][i],full_gradient=agreement_data[2][i],
                     metric_ema=state['agreement_ema'],eta_next=state['eta'])
            # Publish progress only after both trace and state writes succeed.
            save(folder/'state.npz', **state, step=end)
            if end%20000==0 or end==frontier:
                save(folder/f'checkpoint_{end:09d}.npz', **state, step=end)
            progress = dict(step=end, failed_update=int(state['failed']),
                last_train_mse=finite(traces[i,-1,0]), complete=end>=frontier,
                requested_frontier=frontier, batch_elapsed_seconds=time.monotonic()-begin)
            if evaluations is not None:
                metrics = dict(zip(EVAL_COLUMNS, map(finite, evaluations[i])))
                write_json(folder/f'evaluation_{end:09d}.json', dict(step=end, **metrics))
                progress.update(metrics)
            write_json(folder/'latest.json', progress)
        at = end
        if at%20000==0 or at==frontier:
            print(json.dumps(dict(group=list(group_signature(config)), step=at, cases=len(cases),
                elapsed=time.monotonic()-begin, failed=int(np.count_nonzero(np.asarray(states['failed']))),
                validation_mse=[finite(v[0]) for v in evaluations])), flush=True)
        if np.all(np.asarray(states['failed'])>0):
            return True
    return True


def summary(root):
    rows=[]
    for path in sorted(root.glob('*/case.json')):
        if not (path.parent/'latest.json').exists(): continue
        c=json.loads(path.read_text()); latest=json.loads((path.parent/'latest.json').read_text())
        rows.append(dict(id=path.parent.name, config=c, **latest))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--frontier', type=int, default=20000)
    p.add_argument('--worker', type=int, default=0);p.add_argument('--workers', type=int, default=1)
    p.add_argument('--seconds', type=float, default=1100)
    p.add_argument('--coordinates', choices=core.COORDINATES);p.add_argument('--optimizer', choices=('gd','adam'))
    p.add_argument('--target');p.add_argument('--require-gpu', action='store_true')
    args=p.parse_args()
    if args.frontier<20000 or args.frontier%20000:
        p.error('Scientific horizons must be multiples of 20k')
    args.root.mkdir(parents=True, exist_ok=True)
    if args.require_gpu:
        verify_gpu(args.root)
    source_files=list(Path(__file__).parent.glob('*.py')) + [
        Path(core.old.__file__), Path(core.maps.__file__),
        Path('experiments/expD34_readout_race/targets.py')]
    sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}
    job=os.environ.get('SLURM_JOB_ID','local')
    write_json(args.root/f'source_{job}_{args.worker}.json', dict(sources=sources,
        commit=os.environ.get('EXPLORATION_SOURCE_COMMIT'), trace_columns=core.TRACE, evaluation_columns=EVAL_COLUMNS))
    cases=json.loads(args.manifest.read_text()) if args.manifest else baseline()
    for name in ('coordinates','optimizer','target'):
        value=getattr(args,name)
        if value: cases=[c for c in cases if c[name]==value]
    signatures=sorted({group_signature(c) for c in cases})
    assigned={s for i,s in enumerate(signatures) if i%args.workers==args.worker}
    groups=defaultdict(list)
    for c in cases:
        if group_signature(c) not in assigned: continue
        path=args.root/key(c)/'state.npz'
        if path.exists():
            with np.load(path) as checkpoint: at=int(checkpoint['step'])
        else: at=0
        if at<args.frontier: groups[(group_signature(c),at)].append(c)
    deadline=time.monotonic()+args.seconds
    for _, group in sorted(groups.items()):
        if not advance(args.root, group, args.frontier, deadline): break
    write_json(args.root/f'progress_{job}_{args.worker}.json', summary(args.root))


if __name__=='__main__':main()
