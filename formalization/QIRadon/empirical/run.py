#!/usr/bin/env python3
"""Run the declared construction experiments and preserve every tested row."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import datetime
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import brentq
from scipy.special import hyp1f1, gammaln, roots_legendre
from construction import (sphere_rule, mixtures, target_values, profiles, test_points,
                          geometry, encode, error_bounds, construct_evaluate,
                          rounded_readout_evaluation)

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/'config.json').read_text())


def log(message):
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    with (ROOT/'findings/scratchpad.md').open('a') as f:
        f.write(f'\n- {stamp}: {message}\n')
    print(message, flush=True)


def save(name, rows):
    (ROOT/'results'/f'{name}.json').write_text(json.dumps(rows, indent=2)+'\n')


def errors(pred, target):
    diff = pred-target
    return np.linalg.norm(diff, axis=0)/np.linalg.norm(target, axis=0), np.max(abs(diff), axis=0)


def validate():
    rows = []
    for d in CONFIG['dimensions']:
        theta, alpha = sphere_rule(d, 4)
        assert np.max(abs(np.linalg.norm(theta, axis=1)-1)) < 2e-15
        assert np.min(alpha) > 0 and abs(alpha.sum()-1) < 2e-15
        for k in range(4):
            # Exact sphere moment E theta_1^(2k) = (1/2)_k/(d/2)_k.
            exact = np.exp(gammaln(.5+k)-gammaln(.5)-gammaln(d/2+k)+gammaln(d/2))
            observed = alpha@theta[:, 0]**(2*k)
            assert abs(exact-observed) < 3e-15, (d, k, exact, observed)
        mixed_error = 0.
        rng = np.random.default_rng(192+d)
        for _ in range(40):
            total = 2*int(rng.integers(0,4))
            powers = np.bincount(rng.integers(0,d,size=total), minlength=d)
            observed = alpha@np.prod(theta**powers, axis=1)
            exact = (0. if np.any(powers%2) else np.exp(
                np.sum(gammaln((powers+1)/2)-gammaln(.5))
                +gammaln(d/2)-gammaln((d+total)/2)))
            mixed_error = max(mixed_error, abs(observed-exact))
        assert mixed_error < 5e-14
        x = test_points(d, 24, CONFIG['evaluation_seed'])
        coarse = construct_evaluate(d, 8, 64, x)
        refined = construct_evaluate(d, 8, 64, x, quadrature_factor=2)
        difference = np.max(abs(coarse['pred']-refined['pred']))
        assert difference < 2e-12
        rows.append(dict(d=d, max_quadrature_refinement_change=float(difference),
                         mixed_monomials_checked=40, mixed_monomial_max_error=float(mixed_error)))
    # Constant data give exactly zero density, correction, and non-bias weights.
    weights, _ = encode(lambda z: np.full(np.asarray(z).shape+(1,), 3.25+0j), 64)
    assert np.max(abs(weights)) == 0
    save('validation', rows)
    log('Validation passed: positive normalized sphere rules and exact moments; constant profile; doubled contour quadrature. Refinement is a diagnostic, not an interval certificate.')


def widths():
    rows = []
    start = time.perf_counter()
    for d in CONFIG['dimensions']:
        x = test_points(d, CONFIG['evaluation_points'], CONFIG['evaluation_seed'])
        y = target_values(x)
        rms = np.sqrt(np.mean(y*y, axis=0))
        seen = set()
        for q in CONFIG['angular_orders']:
            n = max(CONFIG['minimum_cells'], CONFIG['cells_per_angular_order']*q)
            if d == 1 and n in seen:
                continue
            seen.add(n)
            t0 = time.perf_counter()
            model = construct_evaluate(d, q, n, x)
            rel, sup = errors(model['pred'], y)
            angular_rel, _ = errors(model['angular'], y)
            plain_rel, _ = errors(model['uncorrected'], y)
            angular_bound, scalar_bound = error_bounds(d, q, n)
            for i, name in enumerate(CONFIG['targets']):
                rows.append(dict(d=d, target=name, q=q, n_cells=n, width=model['width'],
                                 directions=model['directions'], centers_per_direction=model['centers_per_direction'],
                                 relative_l2=float(rel[i]), sampled_linf=float(sup[i]),
                                 angular_only_relative_l2=float(angular_rel[i]),
                                 uncorrected_relative_l2=float(plain_rel[i]),
                                 angular_absolute_bound=float(angular_bound[i]),
                                 scalar_absolute_bound=float(scalar_bound[i]),
                                 relative_bound=float((angular_bound[i]+scalar_bound[i])/rms[i]),
                                 readout_l1=float(model['mass'][i]), output_bias=float(model['bias'][i]),
                                 seconds=time.perf_counter()-t0, points=len(x)))
            save('width', rows)
            log(f'Width run d={d}, q={q}, cells={n}, W={model["width"]}: relative errors '+np.array2string(rel, precision=2)+'.')
    log(f'Width suite completed in {time.perf_counter()-start:.1f}s. All configurations retained, including roundoff plateaus.')


def precision():
    rows = []
    target_index = CONFIG['targets'].index(CONFIG['main_target'])
    for d in CONFIG['dimensions']:
        x = test_points(d, 96, CONFIG['evaluation_seed']+1)
        q, n = CONFIG['precision_angular_order'], CONFIG['precision_cells']
        model = construct_evaluate(d, q, n, x, return_model=True)
        y = target_values(x)[:, target_index]
        pred = rounded_readout_evaluation(model, x, CONFIG['precision_bits'], target_index)
        baseline = np.linalg.norm(model['pred'][:, target_index]-y)/np.linalg.norm(y)
        rms = np.sqrt(np.mean(y*y))
        mass = model['mass'][target_index]+abs(model['bias'][target_index])
        angular_bound, scalar_bound = error_bounds(d, q, n)
        exact_bound = (angular_bound[target_index]+scalar_bound[target_index])/rms
        for j, bits in enumerate(CONFIG['precision_bits']):
            rel = np.linalg.norm(pred[:, j]-y)/np.linalg.norm(y)
            rows.append(dict(d=d, target=CONFIG['main_target'], bits=bits, relative_l2=float(rel), width=model['width'],
                             baseline_relative_l2=float(baseline), readout_and_bias_mass=float(mass),
                             exact_construction_relative_bound=float(exact_bound),
                             perturbation_bound=float(exact_bound+mass*2.**(-bits)/rms),
                             rounding_contribution=float(mass*2.**(-bits)/rms),
                             scope='Final readouts and bias rounded; geometry and evaluation remain float64.'))
        save('precision', rows)
        log(f'Precision run d={d}, W={model["width"]}; float64 baseline {baseline:.3g}; readout+bias absolute mass {mass:.3g}.')


def algebraic():
    rows = []
    # Slope increments explicitly encode piecewise-linear interpolation.
    # x^2 gives an exact supremum at interval midpoints, with no dense-grid proxy.
    for n in CONFIG['algebraic_cells']:
        nodes = np.linspace(-1, 1, n+1)
        h = 2/n
        slope = np.diff(nodes**2)/h
        x = np.r_[nodes, (nodes[:-1]+nodes[1:])/2]
        # The Lean realization uses two ramps per interval, including endpoints.
        relus = np.maximum(x[:, None]-nodes[:-1], 0)-np.maximum(x[:, None]-nodes[1:], 0)
        pred = 1+relus@slope
        error = np.max(abs(pred-x*x))
        expected = 1/n**2
        assert abs(error-expected) < 5e-14
        rows.append(dict(n_intervals=n, neurons=2*n, linf=float(error),
                         upper_bound=expected, arbitrary_relu_lower_bound=1/(2*(2*n+1)**2)))
    save('algebraic', rows)
    log('Algebraic experiment: explicit ReLU interpolation of x^2 agrees with 1/n^2 to float64 accuracy. This tests the ReLU corollary, not the full finite-smooth Radon theorem.')


def algebraic_kernel():
    """Actual spectral construction for a nonanalytic target, without fitting.

    K(x)=exp(-pi*x*x), p_hat(xi)=(1+xi*xi)^(-2), r=0.
    Its inverse transform is pi/2*(1+2*pi*abs(x))*exp(-2*pi*abs(x)).
    The weighted spectral moment A_2 is exactly pi+2.
    """
    rows = []
    H = 2*np.sqrt((np.pi+4)/np.pi)
    Z = np.sqrt(2*(np.pi+3)/np.pi)
    Cker = 1+(1+2*np.sqrt(np.pi)/(2*np.sqrt(2)))+2*(1+1/np.sqrt(2))
    x = np.linspace(-1, 1, 2001)
    y = np.pi/2*(1+2*np.pi*abs(x))*np.exp(-2*np.pi*abs(x))
    for B in CONFIG['nonanalytic_spectral_cutoffs']:
        L = max(1., np.log1p(B))
        gamma = B/np.sqrt(L)
        h = 1/(H*B)
        D = Z*L/B
        radius = int(np.floor((1+D)/h))
        centers = h*np.arange(-radius, radius+1)
        # Work only on [0,B] using evenness; integrate the cosine exactly
        # up to numerical quadrature error, with no fitting of the target.
        def readouts(order):
            xi, qw = roots_legendre(order)
            xi, qw = (xi+1)*B/2, qw*B/2
            spectrum = np.exp(np.pi*(xi/gamma)**2)/(1+xi*xi)**2
            density = np.empty(len(centers))
            for j in range(0, len(centers), 64):
                density[j:j+64] = 2*(np.cos(2*np.pi*centers[j:j+64, None]*xi) @ (qw*spectrum))
            return h*gamma*density
        order = max(128, int(np.ceil(6*np.pi*B*(1+D))))
        weights = readouts(order)
        pred = np.exp(-np.pi*(gamma*(x[:, None]-centers))**2) @ weights
        error = np.max(abs(pred-y))
        bound = (np.pi+2)*Cker/(1+B)**2
        assert error <= bound
        refinement = None
        if B in [4, 64, 256]:
            refined = readouts(2*order)
            refinement = np.max(abs(np.exp(-np.pi*(gamma*(x[:, None]-centers))**2) @ (weights-refined)))
            assert refinement < 1e-9
        rows.append(dict(B=B, L=L, gamma=gamma, spacing=h, halo_extent=D,
                         neurons=len(centers), sampled_linf=float(error), upper_bound=float(bound),
                         quadrature_order=order, refinement_change=None if refinement is None else float(refinement),
                         spectral_moment_s=2, spectral_moment=float(np.pi+2)))
        save('algebraic_kernel', rows)
        log(f'Nonanalytic spectral construction B={B}, W={len(centers)}: max sampled error {error:.3g}; theorem bound {bound:.3g}; quadrature refinement {refinement}.')


def analytic_constants(d, target, angular_rate=1.):
    # A global exponential majorant derived from the generating series, with
    # constants independent of experimental errors. See the empirical appendix.
    U, B = 0., 0.
    for amp, a, c in mixtures(d)[target]:
        radius = np.linalg.norm(c)
        U += 4*abs(amp)*(hyp1f1(d/2, .5, a*(1+radius)**2*np.exp(angular_rate)/4)
                        +hyp1f1(d/2, .5, a*radius**2*np.exp(angular_rate)/4))
        B += abs(amp)*hyp1f1(d/2, .5, a)
    return U, 2*(32+12)*B, angular_rate, np.pi*.5/4


def allocation():
    d = 2
    x = test_points(d, 128, CONFIG['evaluation_seed']+2)
    y = target_values(x)
    all_rows = []
    for q in CONFIG['allocation_orders']:
        for n in CONFIG['allocation_cells']:
            model = construct_evaluate(d, q, n, x)
            rel, sup = errors(model['pred'], y)
            for index in [1, 4]:
                U, V, a, b = analytic_constants(d, index)
                all_rows.append(dict(target=CONFIG['targets'][index], q=q, n_cells=n,
                                     width=model['width'], sampled_linf=float(sup[index]),
                                     relative_l2=float(rel[index]), model_bound=float(U*np.exp(-a*q)+V*np.exp(-b*n))))
        save('allocation_grid', all_rows)
        log(f'Allocation grid q={q} completed for narrow and signed-mixture targets.')
    chosen = []
    for index in [1, 4]:
        U, V, a, b = analytic_constants(d, index)
        for epsilon in [1e-3, 1e-6, 1e-9]:
            # p=d-1=1, kappa=1. Solve for angular error fraction z in (0,1).
            def balance(z):
                return z*np.log(U/(epsilon*z))-(d-1)*(1-z)*np.log(V/(epsilon*(1-z)))
            z = brentq(balance, 1e-12, 1-1e-12, xtol=1e-14)
            qc = np.log(U/(epsilon*z))/a
            nc = np.log(V/(epsilon*(1-z)))/b
            q, n = max(1, int(np.ceil(qc))), max(32, int(np.ceil(nc)))
            model = construct_evaluate(d, q, n, x)
            rel, sup = errors(model['pred'], y)
            model_bound = U*np.exp(-a*q)+V*np.exp(-b*n)
            assert model_bound <= epsilon*(1+1e-12)
            assert sup[index] <= epsilon+1e-12
            candidates = [r for r in all_rows if r['target']==CONFIG['targets'][index] and r['sampled_linf'] <= epsilon]
            best = min(candidates, key=lambda r:r['width']) if candidates else None
            bound_candidates = [r for r in all_rows if r['target']==CONFIG['targets'][index] and r['model_bound'] <= epsilon]
            bound_best = min(bound_candidates, key=lambda r:r['width']) if bound_candidates else None
            # Verify the continuous optimum against a dense budget-fraction scan.
            scan = np.linspace(.0001, .9999, 20001)
            costs = (np.log(U/(epsilon*scan))/a)**(d-1)*(np.log(V/(epsilon*(1-scan)))/b)
            assert qc**(d-1)*nc <= costs.min()*(1+1e-10)
            chosen.append(dict(target=CONFIG['targets'][index], tolerance=epsilon, q=q, n_cells=n,
                               width=model['width'], angular_budget_fraction=float(z),
                               sampled_linf=float(sup[index]), relative_l2=float(rel[index]),
                               model_bound=float(model_bound), U=float(U), V=float(V), a=a, b=b,
                               best_sampled_grid=best, best_bound_grid=bound_best,
                               width_ratio_to_sampled_best=None if best is None else model['width']/best['width']))
            log(f'Allocation {CONFIG["targets"][index]}, eps={epsilon:g}: theory q={q}, cells={n}, W={model["width"]}, error={sup[index]:.3g}; best tested W={None if best is None else best["width"]}.')
    save('allocation', chosen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['all', 'validation', 'width', 'precision', 'algebraic', 'algebraic_kernel', 'allocation'], default='all')
    args = parser.parse_args()
    suites = dict(validation=validate, width=widths, precision=precision, algebraic=algebraic,
                  algebraic_kernel=algebraic_kernel, allocation=allocation)
    for name, fn in suites.items():
        if args.suite in ['all', name]:
            log(f'Starting {name} suite, declared config in config.json.')
            fn()
