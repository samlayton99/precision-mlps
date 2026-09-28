"""Numerical evidence for relaxation, autonomous forecasts, and error bounds.

Writes numerical artifacts only. Scientific exposition is authored separately.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import persistence as pe, persistence_theory as pt, persistence_reduction as pr, plateau, targets, transport


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def audit(source, root):
    x = targets.grid(2048); y = pe.af.data('moment9')[1]; q = transport.basis(x, 65)
    rows = []; modal = []; vectors = []
    settings = jnp.array([.002, 0., .999, 1e-8, 0.])
    states = []
    for step in (20000, 100000, 200000, 400000, 600000):
        pp, _, _, hashes = pe.load_inputs(source, step, range(5)); states.append((step, pp))
    long_path = root/'runs/full_600k_q64/snapshots.npz'
    if long_path.exists():
        with np.load(long_path) as f:
            for step in (1000000, 2000000, 4000000, 6000000):
                matches = np.flatnonzero(f['steps'] == step)
                if len(matches): states.append((step, f['p'][:, matches[0]]))
        hashes[str(long_path)] = hashlib.sha256(long_path.read_bytes()).hexdigest()
    for step, pp in states:
        for seed, p in enumerate(pp):
            state = pt.tensors(p, x, y); J = q.T @ state['J']/len(x); e = q.T @ state['r']/len(x)
            jc = J[:2]; C = jc @ jc.T; K = J @ J.T
            B = np.linalg.solve(C, K[:2, 2:]); T = J[2:].T-jc.T @ B; S = T.T @ T
            z = np.linalg.solve(C, jc @ state['g']); tracking = jc.T @ z
            exact = state['g']-tracking
            derivative = -J[2:] @ state['g']
            relaxation = -S[:7, :7] @ e[2:9]
            hard = -S[:7, 7]*e[9]
            higher = -S[:7, 8:] @ e[10:]
            track = -K[2:9, :2] @ z
            zero = jnp.zeros_like(jnp.array(p))
            diagnostics = jax.device_get(plateau.diagnostics(jnp.array(p), zero, zero,
                jnp.zeros((3, len(p))), jnp.array(step), jnp.array(y), jnp.array(x), settings))
            row = dict(seed=seed, step=step, generated_residual_norm=np.linalg.norm(e[2:9]),
                hard_residual=e[9], generated_energy_rate=e[2:9] @ derivative[:7],
                generated_self_rate=e[2:9] @ relaxation, generated_hard_rate=e[2:9] @ hard,
                generated_higher_rate=e[2:9] @ higher, generated_tracking_rate=e[2:9] @ track,
                modal_force_error=np.linalg.norm(exact-T @ e[2:]),
                modal_derivative_error=np.linalg.norm(derivative[:7]-relaxation-hard-higher-track))
            for key in ('force_norm', 'fine_norm', 'alignment', 'participation', 'derivative_cancellation', 'total_log_rate',
                        'relative_vector_rate', 'force_identity', 'derivative_identity', 'next_step_relative_remainder'):
                row[key] = float(diagnostics[key])
            for i, key in enumerate(plateau.DRIVERS):
                row['rate_'+key] = float(diagnostics['log_rates'][i])
                row['norm_'+key] = float(diagnostics['driver_norms'][i])
            rows.append(row); vectors.append(diagnostics['parts'])
            for k in (2, 3):
                i = k-2
                modal.append(dict(seed=seed, step=step, mode=k, residual=e[k], derivative=derivative[i],
                    own_mode=-S[i, i]*e[k], other_generated=relaxation[i]+S[i, i]*e[k],
                    hard=hard[i], higher=higher[i], tracking=track[i],
                    direct_diagonal_rate=S[i, i]))
        print(json.dumps(dict(audit_step=step)), flush=True)
    table(root/'mechanism.csv', rows); table(root/'generated_evolution.csv', modal)
    np.savez_compressed(root/'derivative_vectors.npz', parts=np.stack(vectors))
    (root/'audit_manifest.json').write_text(json.dumps(dict(input_hashes=hashes,
        states=len(rows), derivative_units='per physical time eta*n', diagnostic_degree=65,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')


def predict(source, root):
    spectra_rows = []; summaries = []; budgets = []; physical_budgets = []
    for start, ends in ((100000, np.arange(100000, 600001, 10000)),
                        (600000, np.arange(600000, 6000001, 100000))):
        pp, x, y, hashes = pe.load_inputs(source, start, range(5))
        predicted = []; gradients = []; effective = []; constants = []; paths = []
        two = []; two_forces = []; truncation = []
        physical = []; physical_forces = []; driven = []; driven_forces = []
        for seed, p in enumerate(pp):
            spectrum = pt.frozen_spectrum(p, x, y)
            reduced_physical = pr.reduced_state(p, x, y, transport.basis(x, 9))
            for is_driven, parameters, forces in ((False, physical, physical_forces), (True, driven, driven_forces)):
                values = [pr.reduced_at(reduced_physical, int(end-start), driven=is_driven) for end in ends]
                parameters.append(np.stack([v['p'] for v in values])); forces.append(np.stack([v['force'][:177] for v in values]))
                physical_budgets.append(dict(seed=seed, start=start, driven=is_driven,
                    transient_slope_budget=values[-1]['transient_budget'], slope_force_floor=values[-1]['force_floor'],
                    finite_slope_path_bound=values[-1]['slope_path_bound'],
                    initial_generated_norm=np.linalg.norm(reduced_physical['initial']),
                    generated_equilibrium_norm=np.linalg.norm(reduced_physical['equilibrium']) if is_driven else 0.,
                    lambda_min=reduced_physical['values'].min(), lambda_max=reduced_physical['values'].max()))
            w = (len(p)-1)//3
            component = spectrum['projected'][:w]*spectrum['loading']
            score = np.linalg.norm(component, axis=0); ranking = np.argsort(score)[::-1]
            force = component.sum(axis=1)
            Jmodal = transport.basis(x, 9).T @ spectrum['initial']['J']/len(x)
            for rank, i in enumerate(ranking):
                mode = Jmodal @ spectrum['vectors'][:, i]/np.sqrt(spectrum['values'][i]) if rank < 2 else np.full(10, np.nan)
                spectra_rows.append(dict(seed=seed, start=start, rank=rank+1, eigenvalue=spectrum['values'][i],
                    gradient_loading=spectrum['loading'][i], slope_force_component=score[i],
                    output_mode2_power=mode[2]**2, output_mode3_power=mode[3]**2,
                    output_coarse_power=mode[:2] @ mode[:2],
                    slope_share=np.sum(spectrum['vectors'][:w, i]**2),
                    bias_share=np.sum(spectrum['vectors'][w:2*w, i]**2),
                    readout_share=np.sum(spectrum['vectors'][2*w:3*w, i]**2)))
            for rank in (1, 2, 3, 4, 8):
                remainder = force-component[:, ranking[:rank]].sum(axis=1)
                summaries.append(dict(seed=seed, start=start, retained=rank,
                    force_relative_error=np.linalg.norm(remainder)/np.linalg.norm(force)))
            pair = ranking[:2]
            future_path = np.sum(abs(spectrum['loading'][pair])*np.linalg.norm(spectrum['vectors'][:w, pair], axis=0)/spectrum['values'][pair])
            budgets.append(dict(seed=seed, start=start, initial_maximum_gamma=np.max(abs(p[:w])),
                two_mode_infinite_slope_path=future_path,
                two_mode_uniform_gamma_cap=np.max(abs(p[:w]))+future_path,
                excludes_gamma_one_in_two_mode_model=bool(np.max(abs(p[:w]))+future_path < 1.)))
            prediction = [pt.frozen_at(spectrum, int(end-start)) for end in ends]
            reduced = dict(spectrum, loading=spectrum['loading']*np.isin(np.arange(len(spectrum['loading'])), ranking[:2]))
            pair_prediction = [pt.frozen_at(reduced, int(end-start)) for end in ends]
            predicted.append(np.stack([r[0] for r in prediction])); gradients.append(np.stack([r[1] for r in prediction]))
            effective.append(np.stack([r[2] for r in prediction])); paths.append(np.array([r[3] for r in prediction]))
            two.append(np.stack([r[0] for r in pair_prediction])); two_forces.append(np.stack([r[2] for r in pair_prediction]))
            truncation.append(np.array([np.linalg.norm(a[0]-b[0]) for a, b in zip(prediction, pair_prediction)]))
            constants.append(p[None, :]-.002*(ends-start)[:, None]*spectrum['initial']['g'][None, :])
            print(json.dumps(dict(prediction_seed=seed, start=start)), flush=True)
        np.savez_compressed(root/f'predictions_{start}.npz', steps=ends, seeds=np.arange(5),
            p0=pp, frozen=np.stack(predicted), gradient=np.stack(gradients), effective=np.stack(effective),
            constant=np.stack(constants), slope_path_bound=np.stack(paths),
            physical=np.stack(physical), physical_effective=np.stack(physical_forces),
            driven=np.stack(driven), driven_effective=np.stack(driven_forces),
            two_mode=np.stack(two), two_effective=np.stack(two_forces), spectral_parameter_error=np.stack(truncation))
    table(root/'spectrum.csv.gz', spectra_rows); table(root/'spectral_reconstruction.csv', summaries)
    table(root/'two_mode_budgets.csv', budgets)
    table(root/'physical_mode_budgets.csv', physical_budgets)


def bounds(source, root):
    local = []; tubes = []; summaries = []
    for start in (100000, 600000):
        pp, x, y, _ = pe.load_inputs(source, start, range(5))
        for seed, p in enumerate(pp):
            candidates = pt.local_horizons(p, x, y)
            local.extend(dict(seed=seed, start=start, **r) for r in candidates)
            result = pt.frozen_enclosure(p, x, y, horizon=5400000, block=1000)
            tubes.extend(dict(seed=seed, start=start, block=1000, **r) for r in result)
            closed = [r for r in result if r['closed']]
            best = max(candidates, key=lambda r: r['updates'])
            row = dict(seed=seed, start=start, local_updates=best['updates'], local_radius=best['radius'],
                enclosed_updates=closed[-1]['end'] if closed else 0,
                last_error=closed[-1]['end_error'] if closed else 0.,
                next_block_closed=result[-1]['closed'])
            for threshold in (1., 3.2, 16.):
                row[f'maximum_acquired_count_{threshold:g}'] = max((r[f'maximum_acquired_count_{threshold:g}'] for r in closed), default=-1)
            summaries.append(row)
            table(root/'local_bounds.csv', local); table(root/'enclosures.csv.gz', tubes); table(root/'bound_summary.csv', summaries)
            print(json.dumps(row), flush=True)


def original_states(source, seed):
    if source.is_file():
        with np.load(source) as f:
            i = np.flatnonzero(f['seeds'] == seed)[0]
            return {int(t): f['p'][i, j] for j, t in enumerate(f['steps'])}
    folder = source/f'primary_{seed}'
    cases = json.loads((folder/'manifest.json').read_text())['cases']
    i = next(i for i, c in enumerate(cases) if c['target'] == 'moment9' and c['optimizer'] == 'gd')
    with np.load(folder/'snapshots.npz') as f:
        return {int(t): f['p'][i, j] for j, t in enumerate(f['steps'])}


def compare(source, root):
    x, y, _, _ = pe.af.data('moment9'); w = 177; rows = []; endpoints = []
    all_modes = transport.basis(x, 9)
    matrices = dict(full=np.empty((len(x), 0)), linear_features=np.empty((len(x), 0)),
                    five_mode=all_modes[:, [0, 1, 2, 3, 9]], ten_mode=all_modes)
    def model_force(p, p0, q, model):
        g, _, jc = pe.field(p, p0, jnp.asarray(x), jnp.asarray(y), q, model)
        return (g-jc.T @ jnp.linalg.solve(jc @ jc.T, jc @ g))[:w]
    effective = jax.jit(model_force, static_argnums=(3,))
    original = [original_states(source, seed) for seed in range(5)]
    full_path = root/'runs/full_600k_q64/snapshots.npz'
    if full_path.exists():
        with np.load(full_path) as f:
            for seed in range(5):
                for j, t in enumerate(f['steps']): original[seed][int(t)] = f['p'][seed, j]
    def add(seed, start, end, model, p, p0, predicted_force=None):
        if end not in original[seed] or end == start: return
        reference = original[seed][end]
        force = np.asarray(effective(reference, reference, matrices['full'], 'full'))
        if predicted_force is None:
            active_model = next((m for m in pe.MODELS if model.startswith(m)), 'full')
            predicted_force = np.asarray(effective(p, p0, matrices[active_model], active_model))
        row = dict(seed=seed, start=start, end=end, model=model,
            slope_displacement=np.linalg.norm(reference[:w]-p0[:w]),
            slope_error=np.linalg.norm(p[:w]-reference[:w]),
            parameter_error=np.linalg.norm(p-reference),
            force_relative_error=np.linalg.norm(predicted_force-force)/np.linalg.norm(force),
            force_norm=np.linalg.norm(predicted_force), reference_force_norm=np.linalg.norm(force),
            mean_gamma=np.mean(abs(p[:w])), mean_gamma_error=np.mean(abs(p[:w])-abs(reference[:w])))
        for block, key in enumerate('abc'):
            row['error_'+key] = np.linalg.norm(p[block*w:(block+1)*w]-reference[block*w:(block+1)*w])
        row['error_d'] = abs(p[-1]-reference[-1]); rows.append(row)
    for start in (100000, 600000):
        path = root/f'predictions_{start}.npz'
        if not path.exists(): continue
        with np.load(path) as f:
            for seed in range(5):
                for j, end in enumerate(f['steps']):
                    add(seed, start, int(end), 'frozen_tangent', f['frozen'][seed, j], f['p0'][seed], f['effective'][seed, j])
                    add(seed, start, int(end), 'constant_gradient', f['constant'][seed, j], f['p0'][seed], f['effective'][seed, 0])
                    if 'two_mode' in f:
                        add(seed, start, int(end), 'two_mode', f['two_mode'][seed, j], f['p0'][seed], f['two_effective'][seed, j])
                    for key in ('physical', 'driven'):
                        if key in f:
                            add(seed, start, int(end), key+'_two_mode', f[key][seed, j], f['p0'][seed], f[key+'_effective'][seed, j])
    for folder in sorted((root/'runs').iterdir()):
        if not (folder/'snapshots.npz').exists(): continue
        manifest = json.loads((folder/'manifest.json').read_text()); start = manifest['start']; model = manifest['model']
        with np.load(folder/'snapshots.npz') as f:
            for i, seed in enumerate(manifest['seeds']):
                for j, end in enumerate(f['steps']):
                    # Keep numerical-control run names separate from model names.
                    label = folder.name if manifest['eta'] != .002 or manifest.get('quadrature_points', 0) == 128 else model
                    add(seed, start, int(end), label, f['p'][i, j], f['p0'][i])
                p = f['p'][i, -1]; t = pt.tensors(p, x, y); e = transport.basis(x, 9).T @ t['r']/len(x)
                _, own_residual, _ = pe.field(p, f['p0'][i], x, y, matrices[model], model)
                endpoints.append(dict(run=folder.name, model=model, seed=seed, start=start, end=int(f['steps'][-1]),
                    mean_gamma=np.mean(abs(p[:w])), max_gamma=np.max(abs(p[:w])),
                    mean_gamma_change=np.mean(abs(p[:w])-abs(f['p0'][i, :w])),
                    positive_travel=np.mean(f['positive'][i, -1]), max_positive_travel=np.max(f['positive'][i, -1]),
                    negative_travel=np.mean(f['negative'][i, -1]), slope_path=f['path'][i, -1],
                    tracking_signed=np.mean(f['tracking'][i, -1]), tracking_path=f['tracking_path'][i, -1],
                    hard_residual=e[9], generated_residual=np.linalg.norm(e[2:9]), relative_mse=np.mean(t['r']**2)/np.mean(y*y),
                    model_hard_residual=float(all_modes[:, 9] @ own_residual/len(x)),
                    model_relative_mse=float(np.mean(np.asarray(own_residual)**2)/np.mean(y*y))))
    table(root/'comparisons.csv', rows); table(root/'endpoints.csv', endpoints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'predict', 'bounds', 'compare'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); args.root.mkdir(parents=True, exist_ok=True)
    globals()[args.action](args.source, args.root)
