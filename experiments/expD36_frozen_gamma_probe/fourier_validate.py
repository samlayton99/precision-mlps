"""Independent numerical evidence for the new note; never writes report prose."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve_triangular
from scipy.special import roots_legendre

from . import core, full_core as f, fourier_law as law, finite_gamma_gram as fg


def periodic_checks(destination, steps=20000):
    rows = []
    for n, gamma, density, offset in itertools.product([64, 128], [4, 16, 64], [4, 16], [0., .5]):
        j = law.periodized_design(n, gamma, density, offset, images=8)
        fourier_j = law.sampled_design(n, gamma, density, offset, aliases=64)
        eigenvalues = law.sampled_spectrum(n, gamma, density, offset, aliases=64)
        vectors = np.exp(2j*np.pi*np.outer(np.arange(n), np.arange(n))/n)/np.sqrt(n)
        measured = vectors.conj().T@(j.T@j)@vectors
        m = n*density
        x = -1+2*(np.arange(m)+offset)/m
        y = np.column_stack([np.sin(k*np.pi*x) for k in [2, 6, 10]]
                            +[f.target(x, 'sine_mix_2_6_10')])/np.sqrt(m)
        keep = eigenvalues > 1e-14*np.max(eigenvalues)
        modes = fourier_j@vectors[:, keep]/np.sqrt(eigenvalues[keep])
        loadings = modes.conj().T@y
        norm_sq = np.sum(y*y, axis=0)
        model = dict(rates=.5*eigenvalues[keep]/(4/n),
                     weights=np.abs(loadings)**2/norm_sq,
                     floor=np.sum(np.abs(y-modes@loadings)**2, axis=0)/norm_sq)
        theta = np.zeros((n, y.shape[1])); hits = np.full(y.shape[1], -1)
        checkpoints = []; max_difference = 0.
        eta = .5/(4/n)
        for step in range(steps+1):
            residual = j@theta-y
            error = np.linalg.norm(residual, axis=0)/np.sqrt(norm_sq)
            hits = np.where((hits < 0)&(error <= .01), step, hits)
            if step in [0, 1, 10, 100, 1000, 10000, steps]:
                prediction = fg.error(model, step)
                max_difference = max(max_difference, float(np.max(abs(error-prediction))))
                checkpoints.append(dict(step=step, actual=error.tolist(), prediction=prediction.tolist()))
            if step < steps:
                theta -= eta*j.T@residual
        values, lo, hi = law.continuous_spectrum(n, gamma, aliases=64)
        row = dict(n=n, gamma=gamma, density=density, offset=offset,
            synthesis_relative=float(np.linalg.norm(j-fourier_j)/np.linalg.norm(j)),
            coefficient_gram_relative=float(np.linalg.norm(measured-np.diag(eigenvalues))/np.linalg.norm(measured)),
            continuous_bracket_violation=float(max(0., np.max(lo-values*n/4), np.max(values*n/4-hi))),
            partition_unity_error=float(np.max(abs(j.sum(axis=1)-2/np.sqrt(m)))),
            max_curve_absolute_difference=max_difference,
            actual_hits=hits.tolist(), predicted_hits=fg.first_hit(model),
            forecast_floor=model['floor'].tolist(), steps=steps, checkpoints=checkpoints)
        rows.append(row)
        core.write_json(destination/'periodic_checks.json', rows)
        print('PERIODIC', n, gamma, density, offset, hits.tolist(), max_difference, flush=True)
    return rows


def finite_checks(archive, destination):
    rows = []
    for folder in sorted((archive/'dictionaries').iterdir()):
        if not (folder/'meta.json').exists():
            continue
        meta = json.loads((folder/'meta.json').read_text())
        n, gamma, name = meta['n'], meta['gamma'], meta['map']
        common = dict(np.load(archive/f'common/N{n}/arrays.npz'))
        x, y = common['x_train'], common['y_train']
        mapping = f.map_matrix(core.geometry(n), name)
        physical = core.design(x, common['centers'], gamma)
        scale, neighbor = f.map_spec(core.geometry(n), name)
        j = f.design_from_physical(physical, scale, neighbor)
        assert core.array_hash(j) == meta['matrix_hash']
        h = fg.gram(x, common['centers'], gamma, mapping)
        direct_h = j.T@j
        # The actual saved GD step is .5/L for these zero-start primary runs.
        eta = .5/meta['L']
        predicted = fg.gram_forecast(h, j.T@y, np.sum(y*y, axis=0), eta)
        saved = dict(np.load(folder/'spectrum.npz'))
        reference = dict(rates=eta*saved['singular']**2,
            weights=saved['loadings']**2/saved['norm_y']**2,
            floor=saved['floor_sq']/saved['norm_y']**2)
        actual_rows = []
        training = archive/'training'/f'N{n}_{name}_gd'
        if (training/'case.json').exists():
            case = json.loads((training/'case.json').read_text())
            if gamma in case['gammas']:
                gi = case['gammas'].index(gamma)
                # All five zero-start columns must use the declared saved clock.
                np.testing.assert_allclose(case['rates'][gi], eta, rtol=1e-14)
                for entry in json.loads((training/'evaluations.json').read_text()):
                    actual_rows.append(dict(step=entry['step'], actual=entry['train'][gi],
                        gram=fg.error(predicted, entry['step']).tolist(),
                        spectrum_reference=fg.error(reference, entry['step']).tolist()))
        row = dict(id=meta['dictionary_id'], n=n, gamma=gamma, map=name, eta=eta,
            gram_relative_difference=float(np.linalg.norm(h-direct_h)/np.linalg.norm(direct_h)),
            eigenvalue_threshold=predicted['eigenvalue_threshold'],
            unresolved_target_mass=predicted['unresolved_mass'].tolist(),
            gram_hits={str(e):fg.first_hit(predicted, e) for e in f.config()['tolerances']},
            reference_hits={str(e):fg.first_hit(reference, e) for e in f.config()['tolerances']},
            comparisons=actual_rows,
            capacity_refit_error=(np.linalg.norm(j@saved['refit_theta']-y, axis=0)/saved['norm_y']).tolist(),
            status='fp64_gram_prediction_with_unresolved_small_spectrum')
        rows.append(row)
        core.write_json(destination/'finite_checks.json', rows)
        print('FINITE', row['id'], row['gram_relative_difference'], row['gram_hits']['0.01'], flush=True)
    return rows


def whole_line_neighbor_gram(width, h, gamma):
    """Exact integral formula, evaluated in FP64; dx/2 normalization.

    Integral psi(t)psi(t-k) dt = 2[F(k+1)-2F(k)+F(k-1)],
    F(k)=k*coth(lambda*k), F(0)=1/lambda.
    """
    offsets = np.arange(width)[:, None]-np.arange(width)
    lam = gamma*h
    def term(k):
        safe = np.where(k == 0, 1., k)
        return np.where(k == 0, 1/lam, safe/np.tanh(lam*safe))
    return h*(term(offsets+1)-2*term(offsets)+term(offsets-1))


def ablation_checks(archive, destination):
    rows = []
    n = 512; geometry = core.geometry(n)
    data = dict(np.load(archive/f'common/N{n}/arrays.npz'))
    x, y = data['x_train'], data['y_train']
    nodes, weights = roots_legendre(2048)
    difference = np.eye(geometry.width+1)
    ii = np.arange(1, geometry.width)
    difference[ii+1, ii] = -1.
    for gamma in [4, 16, 64]:
        raw = core.design(x, geometry.centers, gamma)
        c = raw@difference
        continuous = np.column_stack([np.ones(len(nodes)), np.tanh(gamma*(nodes[:, None]-geometry.centers))])
        continuous = (continuous@difference)*np.sqrt(weights[:, None]/2)
        continuous_gram = continuous.T@continuous
        whole = continuous_gram.copy()
        whole[1:-1, 1:-1] = whole_line_neighbor_gram(geometry.width-1, geometry.h, gamma)
        exterior = whole[1:-1, 1:-1]-continuous_gram[1:-1, 1:-1]
        for name in ['raw', 'collective_neighbor']:
            mapping = f.map_matrix(geometry, name)
            transform = solve_triangular(difference, mapping, lower=True)
            j = raw@mapping
            meta = json.loads((archive/'dictionaries'/f'N{n}_{name}_g{gamma}'/'meta.json').read_text())
            eta = .5/meta['L']
            models = [('full_sampled', c.T@c), ('finite_continuous', continuous_gram),
                      ('whole_line_neighbor_block_with_finite_anchor', whole)]
            model_rows = []
            for label, matrix in models:
                h = transform.T@matrix@transform
                eigenvalues = eigh(h, eigvals_only=True)
                admissible = eigenvalues[0] >= -1e-11*max(eigenvalues[-1], 1.) and eta*eigenvalues[-1] < 1
                model_rows.append(dict(name=label,
                    min_eigenvalue=float(eigenvalues[0]), eta_L=float(eta*eigenvalues[-1]),
                    relative_gram_difference=float(np.linalg.norm(h-j.T@j)/np.linalg.norm(j.T@j)),
                    status='coefficient_curvature_diagnostic_only' if admissible else 'invalid_gd_surrogate',
                    # Changed Grams have no implied output realization or borrowed target loadings.
                    target_loading_transferred=False))
            removals = {}
            halo_keep = np.r_[True, geometry.core]
            if name.endswith('_neighbor'):
                halo_keep = np.r_[True, geometry.core[:-1]&geometry.core[1:], geometry.core[-1]]
            for label, keep in [('no_bias', np.arange(j.shape[1]) != 0),
                                ('no_final_native_column', np.arange(j.shape[1]) != j.shape[1]-1),
                                ('no_halo_columns', halo_keep)]:
                reduced = j[:, keep]
                forecast = fg.rectangular_forecast(reduced, y, eta)
                removals[label] = dict(hits=fg.first_hit(forecast), floor=forecast['floor'].tolist(),
                                      actual_eta=eta, target_loadings_recomputed=True)
            unanchored = c.copy(); unanchored[:, -1] = 0.
            forecast = fg.rectangular_forecast(unanchored@transform, y, eta)
            valid = bool(np.max(forecast['rates']) < 1)
            removals['remove_anchor_before_native_map'] = dict(hits=fg.first_hit(forecast) if valid else None,
                floor=forecast['floor'].tolist(), actual_eta=eta, target_loadings_recomputed=True,
                eta_L=eta*forecast['L'], status='forecast' if valid else 'invalid_contraction_clock')
            rows.append(dict(n=n, gamma=gamma, map=name, models=model_rows, removal_forecasts=removals,
                exterior_min_eigenvalue=float(eigh(exterior, eigvals_only=True)[0]),
                decomposition_error=float(np.linalg.norm(c@transform-j)/np.linalg.norm(j))))
            core.write_json(destination/'ablation_checks.json', rows)
            print('ABLATION', gamma, name, flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['periodic', 'finite', 'ablation', 'all'], default='all')
    p.add_argument('--steps', type=int, default=20000)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    if a.stage in ['periodic', 'all']:
        periodic_checks(a.output, a.steps)
    if a.stage in ['finite', 'all']:
        finite_checks(a.archive, a.output)
    if a.stage in ['ablation', 'all']:
        ablation_checks(a.archive, a.output)


if __name__ == '__main__':
    main()
