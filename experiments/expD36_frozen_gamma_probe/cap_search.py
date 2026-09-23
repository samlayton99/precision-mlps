"""Search for fast admissible dictionaries; these steps are NOT readout timing.

Binary powering evaluates an affine GD recurrence as a differentiable selection
surrogate. Selected slopes are subsequently frozen and trained with ordinary
updates by cap_campaign. No powered step is counted as an executed GD update.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import minimize

from . import core, cap_campaign as campaign

jax.config.update('jax_enable_x64', True)


def powered_coefficients(h, correlation, eta, steps):
    """The zero-start affine recurrence in O(log steps) matrix products."""
    operator = jnp.eye(h.shape[0])-eta*h
    forcing = eta*correlation
    theta = jnp.zeros_like(correlation)
    remaining = int(steps)
    while remaining:
        if remaining & 1:
            theta = operator@theta+forcing
        remaining >>= 1
        if remaining:
            forcing = operator@forcing+forcing
            operator = operator@operator
            operator = (operator+operator.T)/2
    return theta


def objective(unit_slopes, cap, x, centers, y, steps):
    hidden = jnp.tanh((x[:, None]-centers)*cap*unit_slopes)
    j = jnp.column_stack([jnp.ones(len(x)), hidden])/jnp.sqrt(len(x))
    h = j.T@j
    curvature = jnp.linalg.eigvalsh(h)[-1]
    theta = powered_coefficients(h, j.T@y, .5/curvature, steps)
    residual = j@theta-y
    return jnp.log(jnp.sum(residual*residual)/jnp.sum(y*y)+1e-30)


def save_case(root, n, cap, name, slopes, history):
    case = campaign.make_case(n, cap, name, 0, np.asarray(slopes))
    j, y = campaign.matrices(case)
    model, remainder = campaign.fast_screen(j, y)
    folder = root/'cases'/case['id']
    if (folder/'state.npz').exists():
        raise RuntimeError('Refusing to overwrite a dictionary already used for training')
    core.save_arrays(folder/'parameters.npz', x=case['x'], centers=case['centers'], slopes=case['slopes'])
    meta = {k:case[k] for k in ['id', 'n', 'cap', 'family', 'seed']}
    meta.update(map='raw', initialization='zero', eta=.5/(model['L']*(1+1e-12)),
        L_estimate=model['L'], matrix_hash=core.array_hash(j), target_hash=core.array_hash(y),
        screening_kind='powered_GD_selection_then_fp64_gram',
        screen_hits={str(e):campaign.fg.first_hit(model, e) for e in campaign.EPSILONS[:3]},
        unresolved_mass=remainder.tolist(), search_history=history,
        source_commit=os.environ.get('PROBE_SOURCE_COMMIT', 'local'))
    core.write_json(folder/'meta.json', meta)
    return meta


def lbfgs_search(value_gradient, initial, iterations, deadline, horizon):
    """Bounded line-search refinement; outputs remain selection candidates."""
    best_value, best_unit, history = float('inf'), np.asarray(initial).copy(), []
    evaluations = 0

    def evaluate(unit):
        nonlocal best_value, best_unit, evaluations
        if time.monotonic()+20 >= deadline:
            raise TimeoutError('Selection allocation deadline')
        value, gradient = value_gradient(jnp.asarray(unit))
        value, gradient = float(value), np.asarray(gradient)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError('Nonfinite powered-GD selection objective')
        if value < best_value:
            best_value, best_unit = value, unit.copy()
        evaluations += 1
        if evaluations % 20 == 1:
            history.append(dict(evaluation=evaluations, horizon=horizon,
                surrogate_error=float(np.exp(value/2)), best_error=float(np.exp(best_value/2))))
        return value, gradient

    try:
        result = minimize(evaluate, initial, method='L-BFGS-B', jac=True,
            bounds=[(0., 1.)]*len(initial), options=dict(maxiter=iterations, maxls=30, ftol=1e-12, gtol=1e-7))
        status = str(result.message)
    except (TimeoutError, FloatingPointError) as exc:
        status = str(exc)
    history.append(dict(status=status, evaluations=evaluations, horizon=horizon,
                        best_error=float(np.exp(best_value/2)) if np.isfinite(best_value) else None))
    return best_unit, history


def exploratory_start(centers, start, seed):
    """Admissible phase, sparse-lattice, and bimodal slope starts."""
    if start < 8:
        phase = (start-4)*np.pi/4
        return .03+.97*np.abs(np.sin(10*np.pi*centers+phase))**8
    if start < 12:
        stride = [4, 8, 16, 32][start-8]
        return np.where(np.arange(len(centers)) % stride == 0, 1., .01)
    return np.clip(np.random.default_rng(seed).beta(.25, .25, len(centers)), .001, .999)


def run(root, caps, n, iterations, starts, round_index, seconds, method='adam', explore=False):
    from .train import verify_gpu
    verify_gpu(root, f'search_{os.environ.get("SLURM_JOB_ID", "local")}')
    begun = time.monotonic(); deadline = begun+seconds
    rows = json.loads((root/'development_screen.json').read_text())
    for previous in root.glob('search_round*_cases.json'):
        for name in json.loads(previous.read_text()):
            rows.append(json.loads((root/'cases'/name/'meta.json').read_text()))
    records = []
    for cap in caps:
        candidates = [r for r in rows if r['n'] == n and r['cap'] == cap]
        candidates.sort(key=lambda r:r['screen_hits']['0.01'][0] or float('inf'))
        if not candidates:
            continue
        best_forecast = candidates[0]['screen_hits']['0.01'][0]
        horizon = max(8, min(100000000, int(best_forecast or 200000)))
        geometry = core.geometry(n)
        x = np.linspace(-1, 1, 16*n+1)
        y = campaign.f.target(x, campaign.TARGETS[0])/np.sqrt(len(x))
        value_gradient = jax.jit(jax.value_and_grad(
            lambda unit:objective(unit, float(cap), jnp.asarray(x),
                jnp.asarray(geometry.centers), jnp.asarray(y), horizon)))
        for start in range(min(starts, len(candidates))):
            if time.monotonic()+45 >= deadline:
                break
            if explore and start >= 4:
                initial = exploratory_start(geometry.centers, start, 100000+1000*round_index+10*int(cap)+start)
                initial_label = f'exploratory_pattern_{start}'
            else:
                initial = np.load(root/'cases'/candidates[start]['id']/'parameters.npz')['slopes']/cap
                initial_label = candidates[start]['id']
            if method == 'lbfgs':
                best_unit, history = lbfgs_search(value_gradient, initial, iterations, deadline, horizon)
                history.insert(0, dict(initialization=initial_label))
                name = f'search_r{round_index}_start{start}'
                meta = save_case(root, n, cap, name, cap*best_unit, history)
                records.append(meta['id'])
                core.write_json(root/f'search_round{round_index}_cases.json', records)
                print('SEARCH_SELECTED', method, meta['id'], meta['screen_hits']['0.01'], flush=True)
                continue
            unit = jnp.asarray(initial)
            optimizer = optax.adam(.02)
            state = optimizer.init(unit)
            history = []; best_value = float('inf'); best_unit = np.asarray(unit)
            for iteration in range(iterations+1):
                if time.monotonic()+20 >= deadline:
                    break
                value, gradient = value_gradient(unit)
                jax.block_until_ready(value)
                number = float(value)
                if not np.isfinite(number) or not np.all(np.isfinite(np.asarray(gradient))):
                    history.append(dict(iteration=iteration, status='nonfinite_surrogate'))
                    break
                if number < best_value:
                    best_value, best_unit = number, np.asarray(unit)
                if iteration % 20 == 0 or iteration == iterations:
                    history.append(dict(iteration=iteration, horizon=horizon,
                        surrogate_error=float(np.exp(number/2)), best_error=float(np.exp(best_value/2))))
                    print('SEARCH', cap, start, iteration, horizon, np.exp(best_value/2), flush=True)
                if iteration < iterations:
                    updates, state = optimizer.update(gradient, state, unit)
                    unit = jnp.clip(optax.apply_updates(unit, updates), 0., 1.)
            name = f'search_r{round_index}_start{start}'
            meta = save_case(root, n, cap, name, cap*best_unit, history)
            records.append(meta['id'])
            core.write_json(root/f'search_round{round_index}_cases.json', records)
            print('SEARCH_SELECTED', meta['id'], meta['screen_hits']['0.01'], flush=True)
        if time.monotonic()+45 >= deadline:
            break
    core.write_json(root/f'search_completion_{os.environ["SLURM_JOB_ID"]}.json',
        dict(round=round_index, selected=records, seconds=time.monotonic()-begun,
             iterations=iterations, starts=starts, caps=caps, method=method, explore=explore,
             source_commit=os.environ.get('PROBE_SOURCE_COMMIT', 'local')))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--caps', nargs='+', type=float, default=campaign.CAPS)
    p.add_argument('--n', type=int, default=512)
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--starts', type=int, default=4)
    p.add_argument('--round', type=int, default=0)
    p.add_argument('--seconds', type=int, default=3300)
    p.add_argument('--method', choices=['adam', 'lbfgs'], default='adam')
    p.add_argument('--explore', action='store_true')
    a = p.parse_args()
    run(a.root, a.caps, a.n, a.iterations, a.starts, a.round, a.seconds, a.method, a.explore)


if __name__ == '__main__':
    main()
