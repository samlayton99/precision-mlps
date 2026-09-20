"""CPU-only offline screen; each completed dictionary is independently saved."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits

from . import core


def run(root, cfg):
    root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    g = core.geometry(cfg['n'])
    x = np.linspace(-1, 1, cfg['samples_per_cell']*g.n+1)
    xe = core.grid(cfg['n_eval'])
    xv = -1+2*(np.arange(cfg['n_validation'])+.37)/cfg['n_validation']
    y = np.column_stack([core.target(x, t) for t in cfg['targets']])/np.sqrt(len(x))
    ye = np.column_stack([core.target(xe, t) for t in cfg['targets']])/np.sqrt(len(xe))
    yv = np.column_stack([core.target(xv, t) for t in cfg['targets']])/np.sqrt(len(xv))
    raw_qr = core.polynomial_transform(x, cfg['k_max'])
    yh = core.transform(raw_qr, y)
    q = core.discrete_polynomials(x, cfg['k_max'])
    orthogonality = float(np.linalg.norm(q.T@q-np.eye(q.shape[1]), ord=2))
    independent_coefficients = q.T@y
    # Deep complement norms are never reconstructed by subtracting energies.
    core.save_arrays(root/'data/common.npz', x=x, x_eval=xe, x_validation=xv, y=y,
                     y_eval=ye, y_validation=yv, centers=g.centers, alpha=g.alpha, y_hat=yh)
    manifest = dict(config=cfg, source_commit=os.environ.get('PROBE_SOURCE_COMMIT', 'local'),
        sample_hash=core.array_hash(x), target_hash=core.array_hash(y), width=g.width,
        geometry_hash=core.array_hash(g.centers), polynomial_orthogonality=orthogonality,
        independent_coefficient_abs_error=float(np.max(np.abs(np.abs(independent_coefficients)-np.abs(yh[:cfg['k_max']+1])))),
        versions={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'jax', 'optax', 'mpmath']},
        slurm_job=os.environ.get('SLURM_JOB_ID'), precision_note='FP64 estimates; no interval enclosure')
    core.write_json(root/'manifest.json', manifest)
    all_rows, capacity_rows = [], []
    for gamma in cfg['gammas']:
        a = core.design(x, g.centers, gamma)
        ah = core.transform(raw_qr, a)
        ae = core.design(xe, g.centers, gamma)
        av = core.design(xv, g.centers, gamma)
        for name in cfg['maps']:
            tag = f'{name}_g{gamma}'
            scale = core.scales(g, name)
            j, jh, je = a*scale, ah*scale, ae*scale
            curvature, capacities, spectral = core.spectrum(j, y, je, ye, cfg['cutoffs'], cfg['tolerances'])
            e, mu, frob = core.access(yh, jh, cfg['k_max'])
            log_b = 2*core.log_feature_envelope(gamma, np.arange(cfg['k_max']+1)) + np.log(np.sum(scale[1:]**2))
            log_cap = 2*core.log_feature_envelope(gamma, np.arange(cfg['k_max']+1)) + np.log(g.width*np.max(scale)**2)
            # This is a heuristic resolution monitor, explicitly not a rigorous
            # feature/projector error enclosure. Never clip measured access.
            noise = (64*np.finfo(float).eps*np.linalg.norm(j)/np.maximum(e, np.finfo(float).tiny))**2
            selected = set(range(0, cfg['k_max']+1, 8))
            certs = []
            for ti, target in enumerate(cfg['targets']):
                witness_e = e[:, ti].copy()
                if target == 'quadratic':
                    witness_e[2:] = 0.  # Exact polynomial fact; retain measured E in arrays.
                with np.errstate(divide='ignore'):
                    log_mu = np.log(mu[:, ti])
                for eps in cfg['tolerances']:
                    for kind, denominator in [('analytic', log_b), ('directional', log_mu)]:
                        c = core.bound(witness_e, denominator, eps, curvature)
                        k = c['k']
                        c.update(map=name, gamma=gamma, target=target, epsilon=eps, kind=kind,
                                 L=curvature, eta=.5/curvature)
                        if k is not None:
                            c.update(E=float(e[k, ti]), mu=float(mu[k, ti]), log_B=float(log_b[k]),
                                     access_noise_estimate=float(noise[k, ti]),
                                     resolution=('above_heuristic_floor' if mu[k, ti] > noise[k, ti]
                                                 else 'unresolved_roundoff'))
                            if eps == cfg['tolerances'][0]:
                                selected.add(k)
                        certs.append(c)
            b = np.full(cfg['k_max']+1, np.nan)
            for k in sorted(selected):
                b[k] = svdvals(jh[k+1:])[0]**2
            for ti, target in enumerate(cfg['targets']):
                witness_e = e[:, ti].copy()
                if target == 'quadratic':
                    witness_e[2:] = 0.
                for eps in cfg['tolerances']:
                    denominator = np.where(np.isfinite(b), np.log(b), np.inf)
                    c = core.bound(witness_e, denominator, eps, curvature)
                    c.update(map=name, gamma=gamma, target=target, epsilon=eps,
                             kind='subspace_sampled', L=curvature, eta=.5/curvature)
                    certs.append(c)
            for c in certs:
                if c['k'] is not None:
                    c['b_at_k'] = float(b[c['k']]) if np.isfinite(b[c['k']]) else None
                    c['frobenius_at_k'] = float(frob[c['k']])
            all_rows.extend(certs)
            capacity_rows.extend(dict(map=name, gamma=gamma, target=cfg['targets'][c.pop('target_index')], **c)
                                 for c in capacities)
            core.save_arrays(root/f'data/{tag}.npz', J=j, J_eval=je, J_validation=av*scale,
                scales=scale, L=curvature, E=e, mu=mu, frobenius=frob, b=b, log_B=log_b,
                log_B_cap=log_cap, access_noise=noise, **spectral)
            core.write_json(root/'certificates.json', all_rows)
            core.write_json(root/'capacity.json', capacity_rows)
            print(f'SCREEN {tag}: L={curvature:.6g}, elapsed={time.monotonic()-start:.1f}s', flush=True)
    manifest['screen_seconds'] = time.monotonic()-start
    manifest['complete'] = True
    core.write_json(root/'manifest.json', manifest)
    print('SCREEN_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=core.HERE/'config.yaml')
    args = parser.parse_args()
    with threadpool_limits(limits=8):
        run(args.root, core.config(args.config))


if __name__ == '__main__':
    main()
