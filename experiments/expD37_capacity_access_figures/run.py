"""Reuse D36; train only two missing note readout maps, sequentially."""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy.linalg as sla
import torch
from threadpoolctl import threadpool_limits
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D36 = ROOT / 'experiments/expD36_frozen_lambda_sweep'
sys.path.insert(0, str(D36))
# This file can itself be __main__; avoid ambiguous imports of another run.py.
import importlib.util
spec = importlib.util.spec_from_file_location('d36_baseline', D36 / 'run.py')
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)
from ablations import reference_allowances

SOURCE = baseline.RESULTS
RESULTS = ROOT / 'results/checkpoint_D_optimizers/expD37_capacity_access_figures'


def config():
    return yaml.safe_load((HERE / 'config.yaml').read_text())


def readout_map(p, name):
    alpha = reference_allowances(p.cfg['n'], p.cfg['halo_per_side'], p.cfg['lambda_reference'])
    if name == 'alpha_scale':
        return np.diag(alpha), alpha
    if name == 'neighbor_sqrt_cumulative':
        # c = L diag(sqrt(s)) theta, s_j=sum_{l<=j}alpha_l (neurons only).
        scales = np.r_[np.sqrt(alpha[0]), np.sqrt(np.cumsum(alpha[1:]))]
        L = np.eye(len(alpha))
        rows = np.arange(2, len(alpha))
        L[rows, rows-1] = -1
        return L * scales[None, :], alpha
    raise ValueError(name)


def mapped_problem(p, name):
    M, alpha = readout_map(p, name)
    M = torch.from_numpy(M)
    # Avoid subtracting already-scaled columns to construct neighboring features.
    if name == 'neighbor_sqrt_cumulative':
        scales = torch.from_numpy(np.r_[np.sqrt(alpha[0]), np.sqrt(np.cumsum(alpha[1:]))])
        def mapped(A):
            return torch.cat((A[:, :, :1], A[:, :, 1:-1]-A[:, :, 2:], A[:, :, -1:]), 2)*scales
    else:
        def mapped(A):
            return A*torch.from_numpy(alpha)
    A, E = mapped(p.A), mapped(p.E)
    B = A / np.sqrt(len(p.x_train))
    spectrum = np.stack([sla.svdvals(b.numpy(), check_finite=False) for b in B])
    rates = p.cfg['gd_rate_multiplier']/spectrum[:, 0]**2
    return replace(p, cfg=dict(p.cfg, coordinates=name), B=B,
                   BT=B.transpose(1, 2).contiguous(), gd_rates=torch.from_numpy(rates)[:, None, None],
                   readout_map=M, mapped_A=A, mapped_E=E,
                   reference=dict(p.reference, optimizer_singular_values=spectrum, alpha=alpha))


def train_missing():
    meta = json.loads((SOURCE / 'data/metadata.json').read_text())
    p = baseline.make_problem(meta['config'])
    with np.load(SOURCE / 'data/reference.npz') as ref:
        np.testing.assert_array_equal(p.gammas, ref['gammas'])
        np.testing.assert_array_equal(p.x_train.numpy(), ref['x_train'])
        np.testing.assert_array_equal(p.y_train.numpy(), ref['y_train'])
        p.reference = {key: ref[key].copy() for key in p.reference}
    for arm in config()['new_readout_maps']:
        output = RESULTS / 'data/training' / arm
        mp = mapped_problem(p, arm)
        print(f'Training {arm}', flush=True)
        baseline.record_problem(mp, output)
        state, _ = baseline.train(mp, output, config()['training_steps'])
        actual = baseline.physical_coefficients(mp, state)
        discrepancy = torch.linalg.vector_norm(mp.E @ actual - mp.mapped_E @ state['c'], dim=1)
        relative = discrepancy / torch.linalg.vector_norm(mp.y_eval_pair, dim=0)
        baseline.save_json(output / 'data/validation.json', {
            'max_native_vs_physical_prediction_discrepancy': float(relative.max()),
            'full_rank_map': bool(np.linalg.matrix_rank(mp.readout_map.numpy()) == len(mp.readout_map)),
        })
    sources = [SOURCE, SOURCE / 'ablations/sqrt_allowance', SOURCE / 'ablations/neighbor_unscaled']
    baseline.save_json(RESULTS / 'data/provenance.json', {
        'reused_trajectories': {str(path.relative_to(ROOT)):
            hashlib.sha256((path / 'data/trajectory.npz').read_bytes()).hexdigest() for path in sources},
        'config': config(), 'source_config': meta['config'],
        'note': 'papers/optimization_notes/frozen_geometry_capacity_access_note.pdf',
        'new_training': config()['new_readout_maps'],
        'diagnostic_metric': 'raw c=theta, B=A; same centers and training grid as D36',
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train', 'diagnostics', 'plot', 'all'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=4):
        if args.stage in ('train', 'all'):
            train_missing()
        if args.stage in ('diagnostics', 'all'):
            from diagnostics import measure
            measure(config(), SOURCE, RESULTS)
        if args.stage in ('plot', 'all'):
            from figures import plot_all
            plot_all(config(), SOURCE, RESULTS)


if __name__ == '__main__':
    main()
