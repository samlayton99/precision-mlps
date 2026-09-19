"""Compare native block sensitivities at paired Xavier and construction states."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, eigvalsh

from . import core, joint_analysis as analysis, joint_conditioning as campaign, run


def maximum_gram(matrix):
    gram = matrix.T @ matrix
    return float(eigvalsh(gram, subset_by_index=(len(gram)-1, len(gram)-1))[0])


def audit(task):
    path, n, seed, role = task; g = core.geometry(n)
    with np.load(path) as arrays:
        c, gamma = arrays['c'], arrays['gamma']
    if role == 'xavier_readouts_uniform_gamma':
        gamma = np.full(g.width, .25/g.h)
    x, _, features, residual, geometry = analysis.linearize(g, c, gamma)
    gram = geometry.T @ geometry
    values, vectors = eigh(gram, subset_by_index=(len(gram)-1, len(gram)-1))
    vector = vectors[:, 0]
    a = c[1:] / g.h; b = -g.centers * a; variance = np.mean(x*x)
    rank_two = np.array([[variance*(a@a), np.sqrt(variance)*(a@b)],
                         [np.sqrt(variance)*(a@b), b@b]])
    near_linear = (x[:, None]*a + b) / np.sqrt(len(x))
    approximation = float(eigvalsh(rank_two)[-1])
    readouts = {}
    for coordinates in campaign.MAPS:
        matrix = analysis.native_features(features, g, coordinates)
        interior = np.r_[False, g.core] if coordinates=='parameter_scale' else np.r_[False, g.core[:-1]&g.core[1:], False]
        readouts[coordinates] = dict(all_maximum_gram=maximum_gram(matrix),
            interior_maximum_gram=maximum_gram(matrix[:, interior]),
            interior_column_squared_norm_quantiles=np.quantile(np.sum(matrix[:, interior]**2, axis=0), [.1,.5,.9]).tolist())
    return dict(n=n, seed=seed, role=role, h=g.h, training_mse=float(residual@residual),
        uniform_lambda_override=.25 if role=='xavier_readouts_uniform_gamma' else None,
        geometry_all_maximum_gram=float(values[0]), geometry_core_maximum_gram=maximum_gram(geometry[:, g.core]),
        geometry_core_column_squared_norm_quantiles=np.quantile(np.sum(geometry[:, g.core]**2, axis=0), [.1,.5,.9]).tolist(),
        geometry_leading_vector_region_energy={name:float(np.sum(vector[mask]**2)) for name,mask in g.masks.items()},
        near_linear_maximum_gram=approximation,
        near_linear_relative_leading_eigenvalue_error=abs(approximation-values[0])/values[0],
        near_linear_relative_jacobian_error=float(np.linalg.norm(near_linear-geometry)/np.linalg.norm(geometry)),
        readouts=readouts, source_file=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    records = json.loads((args.analysis/'summary.json').read_text())
    tasks = [(args.analysis/r['key']/'spectrum_0.npz', r['case']['n'], r['case']['seed'], role)
             for r in records if r['case']['optimizer']=='gd' and r['case']['coordinates']=='parameter_scale'
             for role in ('paired_xavier','xavier_readouts_uniform_gamma')]
    tasks += [(args.analysis/f'construction_N{n}.npz', n, None, 'construction_lambda_0.25') for n in (512,1024)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(audit, tasks))
    run.write_json(args.output/'scale_sensitivity.json', dict(records=rows, samples_per_cell=16,
        training_states_modified=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
