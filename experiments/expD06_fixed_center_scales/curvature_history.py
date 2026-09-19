"""Detached exact-Hessian history for the two constant-rate GD comparisons."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from . import core, joint_analysis as analysis, joint_conditioning as campaign, run
from .joint_mechanism_probes import native_hessians


def audit(task):
    source, record = task
    case = record['case']; g = core.geometry(case['n'])
    path = source / record['key'] / 'history.npz'
    with np.load(path) as archive:
        history = dict(archive)
    requested = np.r_[0, np.geomspace(1, record['end'], 48), 100000, 300000, 1300000, record['end']]
    indices = np.unique([np.argmin(np.abs(history['step'] - step)) for step in requested])
    rows = []
    for index in indices:
        hessian, _, gradient = native_hessians(g, history['c'][index], history['gamma'][index], case['coordinates'])
        values, vectors = eigh(hessian, subset_by_index=(len(hessian)-1, len(hessian)-1))
        vector = vectors[:, 0]
        rows.append(dict(step=int(history['step'][index]), hessian_max=float(values[0]),
            eta_hessian_max=float(case['eta'] * values[0]),
            gradient_top_curvature_fraction=float((vector @ gradient)**2 / max(gradient @ gradient, 1e-300)),
            top_vector_block_energy=dict(bias=float(vector[0]**2),
                readouts=float(np.sum(vector[1:g.width+1]**2)), geometry=float(np.sum(vector[g.width+1:]**2)))))
    return dict(key=record['key'], case=case, rows=rows, requested_steps=requested.tolist(),
        history_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def plot(records, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for row, coordinates in enumerate(campaign.MAPS):
        for seed in (0, 1):
            ax = axes[row, seed]
            pair = sorted((r for r in records if r['case']['coordinates']==coordinates and r['case']['seed']==seed),
                          key=lambda r: -r['case']['eta'])
            for record, style in zip(pair, ('-', '--')):
                rows = record['rows']
                ax.plot([r['step'] for r in rows], [r['eta_hessian_max'] for r in rows], style,
                    marker='.', ms=4, color=analysis.COLORS[coordinates], label=f"eta={record['case']['eta']:g}")
            ax.axhline(2, color='.4', ls=':', label='Positive-quadratic stability boundary')
            ax.set_xscale('symlog', linthresh=1)
            ax.set_xlim(0, 1.05 * max(r['rows'][-1]['step'] for r in pair))
            ax.set(title=f'{analysis.LABELS[coordinates]}, seed {seed}', xlabel='Updates',
                   ylabel='Shared rate × largest exact Hessian eigenvalue')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('GD curvature along saved trajectories, N=512\nSampled states; full training grid; residual curvature included')
    fig.savefig(output/'curvature_history.png', dpi=160, bbox_inches='tight'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analyses', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(source, record) for source in args.analyses
             for record in json.loads((source/'summary.json').read_text())
             if record['case']['optimizer']=='gd' and record['case']['n']==512]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(audit, tasks))
    run.write_json(args.output/'curvature_history.json', dict(records=records, training_states_modified=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hessian_source_sha256=hashlib.sha256(Path(__file__).with_name('joint_mechanism_probes.py').read_bytes()).hexdigest()))
    plot(records, args.output)


if __name__ == '__main__':
    main()
