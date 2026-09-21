"""Use certificate dual weights to propose adversarial capped dictionaries.

The dual mixture is a search diagnostic, not an admissible finite dictionary.
Rounding or sampling its slopes produces admissible candidates, which still
need ordinary-GD verification. No claim of a global optimum is made.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import svd

from . import cap_campaign as campaign, cap_certificate as certificate, cap_refine, cap_search, core


def slope_probabilities(weights, bias):
    weights = np.maximum(np.asarray(weights), 0.)
    missing = np.maximum(float(bias)-weights.sum(axis=0), 0.)
    values = np.vstack([weights, missing])
    totals = values.sum(axis=0)
    values[-1, totals == 0] = 1.
    return values/values.sum(axis=0)


def run(root, archive, caps, n, rank, round_index):
    names = []
    for cap in caps:
        candidates = []
        for folder in (root/'certificates').glob(f'N{n}_cap{cap:g}_t0_r*'):
            for path in folder.glob('*.npz'):
                if not path.with_suffix('.json').exists():
                    continue
                meta = json.loads(path.with_suffix('.json').read_text())
                if meta.get('status') == 'grid_candidate' and 'beta_grid' in meta:
                    score = cap_refine.candidate_score(meta['delta'], meta['beta_grid'], path.stem == 'target')
                    candidates.append((score, path))
        if not candidates:
            raise ValueError(f'No saved witness for cap {cap}')
        source = max(candidates, key=lambda row:row[0])[1]
        witness = np.load(source)['witness']
        case = campaign.make_case(n, cap, 'common', 0)
        j, _ = campaign.matrices(case)
        old = archive/'dictionaries'/f'N{n}_raw_g{cap:g}'
        if (old/'U.npy').exists():
            assert json.loads((old/'meta.json').read_text())['matrix_hash'] == core.array_hash(j)
            u = np.load(old/'U.npy')
        else:
            u, _, _ = svd(j, full_matrices=False, lapack_driver='gesdd')
        proposal = certificate.optimize_candidate(case['x'], case['centers'], cap, witness,
            np.column_stack([u[:, :rank], witness/np.linalg.norm(witness)]), grid_size=17, include_dual=True)
        if proposal['factor'] is None:
            raise RuntimeError(f'Dual proposal failed: {proposal["status"]}')
        grid = np.r_[proposal['grid'], 0.]
        probabilities = slope_probabilities(proposal['dual_weights'], proposal['dual_bias'])
        destination = root/'dual_proposals'/f'N{n}_cap{cap:g}_r{round_index}'
        core.save_arrays(destination/'mixture.npz', grid=grid, probabilities=probabilities)
        core.write_json(destination/'meta.json', dict(source=str(source.relative_to(root)),
            rank=rank, beta_grid=proposal['beta'], dual_bias=proposal['dual_bias'],
            interpretation='finite-grid dual mixture; selection diagnostic only'))
        selections = [('argmax', grid[np.argmax(probabilities, axis=0)]),
                      ('mean', grid@probabilities)]
        for seed in range(6):
            rng = np.random.default_rng(700000+1000*int(cap)+seed)
            draw = [rng.choice(len(grid), p=probabilities[:, j]) for j in range(len(case['centers']))]
            selections.append((f'sample{seed}', grid[draw]))
        for label, slopes in selections:
            name = f'dual_r{round_index}_{label}'
            history = [dict(initialization='certificate_dual', source=str(source.relative_to(root)), rounding=label)]
            meta = cap_search.save_case(root, n, cap, name, slopes, history)
            names.append(meta['id'])
            core.write_json(root/f'search_round{round_index}_cases.json', names)
            print('DUAL_SELECTED', meta['id'], meta['screen_hits']['0.01'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--caps', type=float, nargs='+', default=[8, 12, 16, 64])
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--round', type=int, default=7)
    args = parser.parse_args()
    run(args.root, args.archive, args.caps, args.n, args.rank, args.round)


if __name__ == '__main__':
    main()
