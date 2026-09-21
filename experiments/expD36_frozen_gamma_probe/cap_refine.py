"""Target-aware certificate search with independent continuous-cap checking."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg import svd

from . import cap_campaign as campaign, cap_certificate as certificate, core
from .slope_spectrum import cdf_time_bound


def witness_bank(x, y, u):
    bank = [('target', y)]
    loadings = u.T@y
    energy = np.sum(loadings**2)
    tails = np.r_[energy, energy-np.cumsum(loadings**2)]/np.dot(y, y)
    used = {0}
    for delta in [.75, .5, .25, .125, .0625, .03125, .015625]:
        cut = int(np.argmin(abs(tails-delta**2)))
        if cut not in used and cut < u.shape[1]:
            residual = y-u[:, :cut]@loadings[:cut]
            if np.linalg.norm(residual) > .005*np.linalg.norm(y):
                bank.append((f'spectral_cut_{cut}', residual))
                used.add(cut)
    for frequency in [2, 6, 10]:
        for power in [1, 2]:
            bank.append((f'windowed_sine_{frequency}_{power}',
                         np.sin(frequency*np.pi*x)*(1-x*x)**power))
    if np.array_equal(y, -y[::-1]):
        bank = [(name, .5*(v-v[::-1])) for name, v in bank]
    elif np.array_equal(y, y[::-1]):
        bank = [(name, .5*(v+v[::-1])) for name, v in bank]
    return [(name, v) for name, v in bank if np.linalg.norm(v) > 0]


def run_case(args):
    root, archive, n, cap, target_index, rank, intervals, verify_count = args
    root, archive = Path(root), Path(archive)
    destination = root/'certificates'/f'N{n}_cap{cap:g}_t{target_index}_r{rank}_i{intervals}'
    if (destination/'result.json').exists():
        return json.loads((destination/'result.json').read_text())
    destination.mkdir(parents=True, exist_ok=True)
    case = campaign.make_case(n, cap, 'common', 0)
    j, yy = campaign.matrices(case)
    y = yy[:, target_index]
    previous = archive/'dictionaries'/f'N{n}_raw_g{cap:g}'
    if (previous/'U.npy').exists():
        meta = json.loads((previous/'meta.json').read_text())
        assert meta['matrix_hash'] == core.array_hash(j)
        u = np.load(previous/'U.npy')
    else:
        u, _, _ = svd(j, full_matrices=False, lapack_driver='gesdd')
    candidates = []
    for label, witness in witness_bank(case['x'], y, u):
        saved = destination/f'{label}.json'
        if saved.exists():
            candidates.append(json.loads(saved.read_text()))
            continue
        started = time.monotonic()
        v = witness/np.linalg.norm(witness)
        delta = abs(float(v@y))/np.linalg.norm(y)
        if delta <= .01:
            continue
        basis = np.column_stack([u[:, :rank], v])
        try:
            proposed = certificate.optimize_candidate(case['x'], case['centers'], cap,
                witness, basis, grid_size=13)
        except Exception as exc:
            core.write_json(saved, dict(label=label, status='solver_failed', detail=str(exc), rough_bound=0))
            continue
        if proposed['factor'] is None:
            continue
        beta = proposed['beta']
        thresholds = np.geomspace(max(min(beta, .999), 1e-20), 1., 256)
        mass = certificate.slow_mass(delta, beta, thresholds)
        score = cdf_time_bound(thresholds, mass)['bound'] or 0
        if label == 'target' and 0 < beta < 1:
            score = max(score, int(np.ceil(np.log(.01)/np.log1p(-.5*beta))))
        row = dict(label=label, delta=delta, beta_grid=beta, rough_bound=score,
                   status='grid_candidate', solver_status=proposed['solver_status'],
                   seconds=time.monotonic()-started)
        core.save_arrays(destination/f'{label}.npz', witness=witness, factor=proposed['factor'], grid=proposed['grid'])
        core.write_json(saved, row)
        candidates.append(row)
        print('CANDIDATE', n, cap, target_index, label, beta, score, flush=True)
    candidates.sort(key=lambda row:row['rough_bound'], reverse=True)
    certified = [certificate.analytic_small_cap(case['x'], len(case['centers']), cap, y)]
    for row in [c for c in candidates if c['status'] == 'grid_candidate'][:verify_count]:
        name = row['label']; path = destination/f'{name}_certificate.json'
        if path.exists():
            result = json.loads(path.read_text())
        else:
            values = dict(np.load(destination/f'{name}.npz'))
            result = certificate.certify(case['x'], case['centers'], cap,
                values['witness'], values['factor'], y, max_intervals=intervals,
                relative_slack=.01, target_witness=name == 'target', progress=True)
            result.update(label=name, factor_hash=core.array_hash(values['factor']),
                          witness_hash=core.array_hash(values['witness']))
            core.write_json(path, result)
        certified.append(result)
        print('CERTIFIED', n, cap, target_index, name, result['beta'],
              'repair', result['bias_repair'], 'seconds', result['seconds'], flush=True)
    bounds = {}
    for epsilon in campaign.EPSILONS:
        bounds[str(epsilon)] = certificate.time_bound(certified, epsilon)
    result = dict(n=n, cap=cap, target=campaign.TARGETS[target_index], rank=rank,
        max_intervals=intervals, certificates=certified, bounds=bounds,
        candidates=candidates, source_commit=os.environ.get('PROBE_SOURCE_COMMIT', 'local'),
        grid_hash=core.array_hash(case['x']), centers_hash=core.array_hash(case['centers']),
        target_hash=core.array_hash(y))
    core.write_json(destination/'result.json', result)
    print('BOUND', n, cap, target_index, bounds['0.01']['bound'], flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--caps', nargs='+', type=float, default=campaign.CAPS)
    p.add_argument('--n', type=int, default=512)
    p.add_argument('--targets', nargs='+', type=int, default=[0])
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--intervals', type=int, default=64)
    p.add_argument('--verify-count', type=int, default=2)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    jobs = [(str(a.root), str(a.archive), a.n, cap, target, a.rank, a.intervals, a.verify_count)
            for cap in a.caps for target in a.targets]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(run_case, jobs):
            pass


if __name__ == '__main__':
    main()
