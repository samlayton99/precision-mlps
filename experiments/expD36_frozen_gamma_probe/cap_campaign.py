"""Direct exploratory capped-gamma campaign; compact evidence, ordinary GD.

Preparation and spectral selection are CPU diagnostics. The GPU stage only
executes frozen-readout updates, with first hits checked on every iterate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh

from . import core, full_core as f, finite_gamma_gram as fg

CAPS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96]
TARGETS = ['sine_mix_2_6_10', 'exp_sin_3pi', 'runge_25', 'quadratic', 'sine_2pi']
EPSILONS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]


def slope_family(centers, cap, family, seed):
    rng = np.random.default_rng(seed)
    w = len(centers)
    if family == 'common':
        return np.full(w, float(cap))
    if family == 'uniform':
        return cap*rng.uniform(0, 1, w)
    if family == 'log_uniform':
        return cap*np.exp(rng.uniform(-np.log(64), 0, w))
    slopes = np.full(w, cap/8)
    count = max(1, w//2)
    if family == 'two_random':
        active = rng.permutation(w)[:count]
    elif family == 'two_center':
        active = np.argsort(np.abs(centers))[:count]
    elif family == 'two_boundary':
        active = np.argsort(-np.abs(centers))[:count]
    elif family == 'sparse_cap':
        slopes[:] = 0
        active = rng.permutation(w)[:max(1, w//8)]
    else:
        raise ValueError(family)
    slopes[active] = cap
    return slopes


def make_case(n, cap, family, seed, slopes=None):
    geometry = core.geometry(n)
    x = np.linspace(-1, 1, 16*n+1)
    slopes = slope_family(geometry.centers, cap, family, seed) if slopes is None else np.asarray(slopes)
    if np.any(np.abs(slopes) > cap) or not np.all(np.isfinite(slopes)):
        raise ValueError('An admissible slope vector is required')
    return dict(n=n, cap=cap, family=family, seed=seed, x=x,
                centers=geometry.centers, slopes=slopes,
                id=f'N{n}_cap{cap:g}_{family}_s{seed}')


def matrices(case):
    j = core.design(case['x'], case['centers'], case['slopes'])
    y = np.column_stack([f.target(case['x'], t) for t in TARGETS])/np.sqrt(len(case['x']))
    return j, y


def fast_screen(j, y):
    h = j.T@j
    values, vectors = eigh(h, check_finite=False)
    curvature = float(values[-1])
    threshold = 32*np.finfo(float).eps*len(h)*max(curvature, 1.)
    keep = values > threshold
    weights = (vectors[:, keep].T@(j.T@y))**2/values[keep, None]/np.sum(y*y, axis=0)
    remainder = 1-weights.sum(axis=0)
    model = dict(rates=.5*values[keep]/curvature, weights=weights,
                 floor=np.maximum(remainder, 0.), L=curvature)
    return model, remainder


def prepare_one(args):
    root, n, cap, family, seed = args
    case = make_case(n, cap, family, seed)
    folder = Path(root)/'cases'/case['id']
    if (folder/'meta.json').exists():
        return json.loads((folder/'meta.json').read_text())
    start = time.monotonic()
    j, y = matrices(case)
    model, remainder = fast_screen(j, y)
    # A small upward buffer protects the intended normalized step in FP64.
    eta = .5/(model['L']*(1+1e-12))
    core.save_arrays(folder/'parameters.npz', x=case['x'], centers=case['centers'], slopes=case['slopes'])
    meta = {k:case[k] for k in ['id', 'n', 'cap', 'family', 'seed']}
    meta.update(map='raw', initialization='zero', eta=eta, L_estimate=model['L'],
                matrix_hash=core.array_hash(j), target_hash=core.array_hash(y),
                screening_kind='fp64_gram_selection_only',
                screen_hits={str(eps):fg.first_hit(model, eps) for eps in EPSILONS[:3]},
                unresolved_mass=remainder.tolist(), seconds=time.monotonic()-start,
                source_commit=os.environ.get('PROBE_SOURCE_COMMIT', 'local'))
    core.write_json(folder/'meta.json', meta)
    return meta


def prepare(root, phase, workers):
    seeds = range(5) if phase == 'development' else range(100, 105)
    families = ['common', 'uniform', 'log_uniform', 'two_random', 'two_center', 'two_boundary', 'sparse_cap']
    jobs = []
    for cap, family in itertools.product(CAPS, families):
        chosen_seeds = [0] if family in ['common', 'two_center', 'two_boundary'] else seeds
        for seed in chosen_seeds:
            jobs.append((str(root), 512, cap, family, seed))
    for n, cap, family in itertools.product([128, 256, 1024], [4, 16, 64], ['common', 'uniform', 'two_center']):
        jobs.append((str(root), n, cap, family, 0 if phase == 'development' else 100))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(prepare_one, jobs):
            rows.append(row)
            print('PREPARE', row['id'], row['screen_hits']['0.01'], flush=True)
            core.write_json(root/f'{phase}_cases.json', [r['id'] for r in rows])
    core.write_json(root/f'{phase}_screen.json', rows)


def make_gd_chunk(steps):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def chunk(theta, count, hits, j, y, eta):
        norm = jnp.linalg.norm(y, axis=1)
        eps = jnp.array(EPSILONS)

        def update(state, _):
            theta, count, hits = state
            residual = j@theta-y
            error = jnp.linalg.norm(residual, axis=1)/norm
            hits = jnp.where((hits < 0)&(error[:, :, None] <= eps), count, hits)
            theta = theta-eta[:, None, None]*(jnp.swapaxes(j, 1, 2)@residual)
            return (theta, count+1, hits), None

        return jax.lax.scan(update, (theta, count, hits), None, length=steps)[0]
    return chunk


def train(root, case_list, frontier, worker, workers, seconds, batch_size):
    import jax
    import jax.numpy as jnp
    from .train import verify_gpu
    jax.config.update('jax_enable_x64', True)
    verify_gpu(root, f'cap_{os.environ.get("SLURM_JOB_ID", "local")}')
    start = time.monotonic()
    deadline = start+seconds
    ids = json.loads(Path(case_list).read_text())[worker::workers]
    groups = {}
    for name in ids:
        folder = root/'cases'/name
        meta = json.loads((folder/'meta.json').read_text())
        at = 0
        if (folder/'state.npz').exists():
            with np.load(folder/'state.npz') as state:
                at = int(state['count'])
        if at < frontier:
            groups.setdefault((meta['n'], at), []).append(name)
    chunk_size = 1000
    chunk = make_gd_chunk(chunk_size)
    for (n, at), names in groups.items():
        for begin in range(0, len(names), batch_size):
            if time.monotonic()+30 >= deadline:
                return
            selected = names[begin:begin+batch_size]
            arrays, metadata, previous, states = [], [], [], []
            for name in selected:
                folder = root/'cases'/name
                case = dict(np.load(folder/'parameters.npz'))
                j, y = matrices(case)
                meta = json.loads((folder/'meta.json').read_text())
                assert core.array_hash(j) == meta['matrix_hash']
                assert core.array_hash(y) == meta['target_hash']
                arrays.append((j, y)); metadata.append(meta)
                if (folder/'state.npz').exists():
                    states.append(dict(np.load(folder/'state.npz')))
                    previous.append(json.loads((folder/'curve.json').read_text()))
                else:
                    states.append(dict(theta=np.zeros((j.shape[1], len(TARGETS))),
                        hits=np.full((len(TARGETS), len(EPSILONS)), -1, dtype=np.int64)))
                    previous.append([])
            j = jnp.asarray(np.stack([a[0] for a in arrays]))
            y = jnp.asarray(np.stack([a[1] for a in arrays]))
            eta = jnp.asarray([m['eta'] for m in metadata])
            theta = jnp.asarray(np.stack([s['theta'] for s in states]))
            hits = jnp.asarray(np.stack([s['hits'] for s in states]))
            count = jnp.array(at, dtype=jnp.int64)
            batch_start = time.monotonic()
            while int(count) < frontier and time.monotonic()+15 < deadline:
                theta, count, hits = chunk(theta, count, hits, j, y, eta)
                jax.block_until_ready(theta)
                now = int(count)
                errors = np.asarray(jnp.linalg.norm(j@theta-y, axis=1)/jnp.linalg.norm(y, axis=1))
                if not np.all(np.isfinite(errors)):
                    raise FloatingPointError('Nonfinite ordinary-GD residual')
                hit_host = np.asarray(hits)
                hit_host = np.where((hit_host < 0)&(errors[:, :, None] <= np.array(EPSILONS)), now, hit_host)
                hits = jnp.asarray(hit_host)
                if now <= 10000 or now % 10000 == 0 or now >= frontier or time.monotonic()+30 >= deadline:
                    theta_host = np.asarray(theta)
                    for k, name in enumerate(selected):
                        folder = root/'cases'/name
                        previous[k].append(dict(step=now, train=errors[k].tolist()))
                        core.write_json(folder/'curve.json', previous[k])
                        core.save_arrays(folder/'state.npz', theta=theta_host[k], hits=hit_host[k], count=now)
                        core.write_json(folder/'training.json', dict(id=name, steps=now,
                            hits=hit_host[k].tolist(), epsilons=EPSILONS, targets=TARGETS,
                            final_train=errors[k].tolist(), complete=now >= frontier,
                            job=os.environ['SLURM_JOB_ID'], eta=metadata[k]['eta'],
                            source_commit=os.environ.get('PROBE_SOURCE_COMMIT', 'local')))
                    print('TRAIN', selected, now, errors[:, 0].tolist(),
                          'seconds', round(time.monotonic()-batch_start, 2), flush=True)
            if time.monotonic()+30 >= deadline:
                return
    core.write_json(root/f'train_completion_{os.environ["SLURM_JOB_ID"]}.json',
        dict(case_list=str(case_list), frontier=frontier, worker=worker,
             seconds=time.monotonic()-start, completed=True))


def reference(root, case_ids):
    rows = []
    for name in case_ids:
        folder = root/'cases'/name
        case = dict(np.load(folder/'parameters.npz'))
        meta = json.loads((folder/'meta.json').read_text())
        j, y = matrices(case)
        model = fg.rectangular_forecast(j, y, eta=meta['eta'])
        curves = json.loads((folder/'curve.json').read_text()) if (folder/'curve.json').exists() else []
        discrepancy = max((float(np.max(np.abs(fg.error(model, row['step'])-row['train']))) for row in curves), default=0.)
        values = dict(id=name, forecast_kind='measured_rectangular_fp64',
            hits={str(eps):fg.first_hit(model, eps) for eps in EPSILONS},
            max_curve_absolute_difference=discrepancy, L=model['L'],
            eta_L=meta['eta']*model['L'], floor=model['floor'].tolist())
        core.write_json(folder/'reference.json', values)
        core.save_arrays(folder/'reference.npz', rates=model['rates'], weights=model['weights'], floor=model['floor'])
        rows.append(values)
        print('REFERENCE', name, values['hits']['0.01'], discrepancy, flush=True)
    return rows


def select(root, source, count=3):
    ids = json.loads(Path(source).read_text())
    grouped = {}
    for name in ids:
        row = json.loads((root/'cases'/name/'meta.json').read_text())
        grouped.setdefault((row['n'], row['cap']), []).append(row)
    selected = []
    for rows in grouped.values():
        ranked = sorted(rows, key=lambda r:r['screen_hits']['0.01'][0] or float('inf'))
        selected.extend(r['id'] for r in ranked[:count])
        selected.extend(r['id'] for r in rows if r['family'] == 'common')
    selected = sorted(set(selected))
    core.write_json(root/'selected_cases.json', selected)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'train', 'reference', 'select'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['development', 'confirmation'], default='development')
    parser.add_argument('--case-list', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--seconds', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=12)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    case_list = args.case_list or args.root/f'{args.phase}_cases.json'
    if args.stage == 'prepare':
        prepare(args.root, args.phase, args.workers)
    elif args.stage == 'train':
        train(args.root, case_list, args.steps, args.worker, args.workers, args.seconds, args.batch_size)
    elif args.stage == 'reference':
        reference(args.root, json.loads(case_list.read_text()))
    else:
        select(args.root, case_list)


if __name__ == '__main__':
    main()
