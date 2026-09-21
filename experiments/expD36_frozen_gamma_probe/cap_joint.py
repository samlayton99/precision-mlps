"""Iterate on cap witnesses by solving the target-aware convex problem jointly."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import svd

from . import core, cap_campaign as campaign, cap_certificate as c, cap_resolvent as resolvent


def run_case(args):
    root, archive, n, cap, ti, rank, intervals = args
    root, archive = Path(root), Path(archive)
    destination = root/'certificates'/f'N{n}_cap{cap:g}_t{ti}_joint_polished_r{rank}_i{intervals}'
    if (destination/'result.json').exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    case = campaign.make_case(n, cap, 'common', 0)
    j, targets = campaign.matrices(case); y = targets[:, ti]
    old = archive/'dictionaries'/f'N{n}_raw_g{cap:g}'
    if (old/'U.npy').exists():
        assert json.loads((old/'meta.json').read_text())['matrix_hash'] == core.array_hash(j)
        u = np.load(old/'U.npy')
    else:
        u, _, _ = svd(j, full_matrices=False, lapack_driver='gesdd')
    model, _ = campaign.fast_screen(j, targets)
    forecast = campaign.fg.first_hit(model)[ti]
    center = .1/forecast if forecast else 1e-9
    shifts = center*np.array([1., 10., 100., 1000.])
    certificates, candidates = [], []
    for index, shift in enumerate(shifts):
        label = f'joint_shift_{index}'
        proof_path = destination/f'{label}_certificate.json'
        if proof_path.exists():
            certificates.append(json.loads(proof_path.read_text()))
            continue
        proposal = c.optimize_joint_candidate(case['x'], case['centers'], cap, y, u[:, :rank], float(shift))
        if proposal['factor'] is None:
            candidates.append(dict(label=label, status=proposal['status'], shift=float(shift)))
            core.write_json(destination/f'{label}.json', candidates[-1])
            continue
        witness = proposal['witness']
        if np.array_equal(y, -y[::-1]):
            witness = .5*(witness-witness[::-1])
        elif np.array_equal(y, y[::-1]):
            witness = .5*(witness+witness[::-1])
        # Re-solve the normalized directional problem before interval checking.
        # Joint conic solves can have small absolute residuals but large residuals
        # after dividing by the small optimized witness norm.
        polished = c.optimize_candidate(case['x'], case['centers'], cap, witness,
            np.column_stack([u[:, :rank], y/np.linalg.norm(y), witness/np.linalg.norm(witness)]), grid_size=17)
        if polished['factor'] is None:
            candidates.append(dict(label=label, status='polish_failed', shift=float(shift)))
            continue
        proposal.update(witness=witness, factor=polished['factor'], beta=polished['beta'])
        core.save_arrays(destination/f'{label}.npz', witness=proposal['witness'], factor=proposal['factor'], grid=proposal['grid'])
        metadata = {k:v for k, v in proposal.items() if k not in ['witness', 'factor', 'grid']}
        metadata['label'] = label
        candidates.append(metadata)
        core.write_json(destination/f'{label}.json', metadata)
        print('JOINT_CANDIDATE', n, cap, ti, rank, shift, proposal['beta'], proposal['delta'], flush=True)
        proof = c.certify(case['x'], case['centers'], cap, proposal['witness'], proposal['factor'],
                          y, max_intervals=intervals, progress=True)
        proof.update(label=label, factor_hash=core.array_hash(proposal['factor']),
                     witness_hash=core.array_hash(proposal['witness']), shift=float(shift))
        core.write_json(proof_path, proof); certificates.append(proof)
        print('JOINT_CERTIFIED', n, cap, ti, rank, proof['beta'], proof['delta'], proof['bias_repair'], flush=True)
    certificates.append(c.analytic_small_cap(case['x'], len(case['centers']), cap, y))
    bounds = {str(e):c.time_bound(certificates, e) for e in campaign.EPSILONS}
    result = dict(n=n, cap=cap, target=campaign.TARGETS[ti], rank=rank,
        max_intervals=intervals, certificates=certificates, candidates=candidates, bounds=bounds,
        source_commit=os.environ.get('PROBE_SOURCE_COMMIT', 'local'),
        grid_hash=core.array_hash(case['x']), centers_hash=core.array_hash(case['centers']), target_hash=core.array_hash(y))
    core.write_json(destination/'result.json', result)
    refined = [resolvent.improve(proof, bounds['0.01']['bound']) for proof in certificates]
    best = max(refined, key=lambda row:row['bound'])
    core.write_json(destination/'resolvent_refinement.json', dict(n=n, cap=cap,
        target=campaign.TARGETS[ti], cdf_bound=bounds['0.01']['bound'], best=best, candidates=refined))
    print('JOINT_BOUND', n, cap, ti, bounds['0.01']['bound'], best['bound'], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--caps', nargs='+', type=float, default=campaign.CAPS)
    p.add_argument('--n', type=int, default=512)
    p.add_argument('--targets', nargs='+', type=int, default=[0])
    p.add_argument('--rank', type=int, default=32)
    p.add_argument('--intervals', type=int, default=256)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    jobs = [(str(a.root), str(a.archive), a.n, cap, ti, a.rank, a.intervals) for cap in a.caps for ti in a.targets]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(run_case, jobs):
            pass


if __name__ == '__main__':
    main()
