"""Retrospective own-state quintic responses to already observed physical kicks.

Two fixed models are anchored at each fork: effective-fine quintic and full
polynomial-loss gradient. No coefficient is fit to future trajectories. CPU
Slurm only; failed/truncated cases remain in every scorecard.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import effective_feedback as ef, mechanism_polynomial as poly
from . import mechanism_persistence_kernel as kernel
from . import mechanism_persistence as io
from .adam_run import atomic_npz
from .run import write_json

MODELS = ('anchored_fine', 'anchored_full')
HORIZONS = (0, 1, 1000, 10000, 20000)


def full_gradient(p, transform, target):
    coefficients, jacobian = poly.coefficients_jacobian(p, 5)
    modal_jacobian = transform@jacobian
    return modal_jacobian.T@(transform@coefficients-target)


def polynomial_field(p, transform, target, model):
    if model == 'anchored_fine':
        return poly.field(p, transform, target, 5)[0]
    if model == 'anchored_full':
        return full_gradient(p, transform, target)
    raise ValueError(model)


def anchored_gradient(p, transform, target, initial_polynomial, initial_exact, model):
    # This evaluation order matches the fork exactly even when polynomial and
    # exact forces have vastly different magnitudes. The correction is frozen.
    return initial_exact+(polynomial_field(p, transform, target, model)-initial_polynomial)


def initial_state(p):
    return dict(p=p, count=jnp.array(0, dtype=jnp.int64), failed=jnp.array(False),
                failure_step=jnp.array(-1, dtype=jnp.int64))


def advance_factory(transform, model, eta):
    def one(state, target, initial_polynomial, initial_exact, steps):
        def step(_, old):
            gradient = anchored_gradient(old['p'], transform, target,
                                         initial_polynomial, initial_exact, model)
            point = old['p']-eta*gradient
            good = ~old['failed'] & jnp.all(jnp.isfinite(point)) & jnp.all(jnp.isfinite(gradient))
            new_failure = ~old['failed'] & ~good
            return dict(p=jnp.where(good, point, old['p']), count=old['count']+good.astype(jnp.int64),
                failed=old['failed'] | ~good,
                failure_step=jnp.where(new_failure, old['count']+1, old['failure_step']))
        return jax.lax.fori_loop(0, steps, step, state)
    return jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0, None)))


def _score_vector(observed, predicted):
    norm = np.linalg.norm(observed)
    error = np.linalg.norm(predicted-observed)
    return dict(actual_norm=float(norm), error_norm=float(error),
                relative_error=float(error/norm) if norm > 0 else None)


def score(cases, initial, predictions, supported, failed, achieved, actual_run):
    """Own-initial displacements, pulse−baseline contrasts, and ± responses."""
    width = (initial.shape[-1]-1)//3
    by_source = {int(case['source_index']): i for i, case in enumerate(cases)}
    rows, contrasts, paired, actual_hashes = [], [], [], {}
    for hi, horizon in enumerate(HORIZONS):
        if horizon == 0:
            continue
        path = Path(actual_run)/'snapshots'/f'{horizon:09d}.npz'
        if not path.exists():
            raise FileNotFoundError(path)
        actual_hashes[str(path)] = ef.digest(path)
        with np.load(path) as data:
            actual, actual_failed = data['p'].copy(), data['failed'].copy()
        movement = actual-initial
        actual_gamma = np.abs(actual[:, :width])-np.abs(initial[:, :width])
        for mi, model in enumerate(MODELS):
            model_movement = predictions[mi, hi]-initial
            model_gamma = np.abs(predictions[mi, hi, :, :width])-np.abs(initial[:, :width])
            valid = supported[mi, hi] & ~actual_failed
            families = {}
            for index, case in enumerate(cases):
                base = by_source[int(case['reference_baseline_index'])]
                families.setdefault(base, {})[float(case['amplitude'])] = index
                row = dict(case, model=model, horizon=horizon, supported=bool(valid[index]),
                    surrogate_failed=bool(failed[mi, hi, index]), actual_failed=bool(actual_failed[index]),
                    achieved_updates=int(achieved[mi, hi, index]),
                    truncated=bool(not failed[mi, hi, index] and achieved[mi, hi, index] < horizon))
                for name, block in (('slope', slice(0, width)), ('readout', slice(2*width, 3*width))):
                    result = _score_vector(movement[index, block], model_movement[index, block]) if valid[index] else dict(actual_norm=float(np.linalg.norm(movement[index, block])), error_norm=None, relative_error=None)
                    row.update({name+'_'+key: value for key, value in result.items()})
                row['actual_mean_lambda_change'] = float(case['h']*actual_gamma[index].mean())
                row['predicted_mean_lambda_change'] = float(case['h']*model_gamma[index].mean()) if valid[index] else None
                rows.append(row)
                if index == base:
                    continue
                available = bool(valid[index] and valid[base])
                contrast = dict(case, model=model, horizon=horizon, supported=available,
                    surrogate_failed=bool(failed[mi, hi, index] or failed[mi, hi, base]),
                    actual_failed=bool(actual_failed[index] or actual_failed[base]),
                    truncated=bool((not failed[mi, hi, index] and achieved[mi, hi, index] < horizon)
                                   or (not failed[mi, hi, base] and achieved[mi, hi, base] < horizon)))
                observed = movement[index, :width]-movement[base, :width]
                predicted = model_movement[index, :width]-model_movement[base, :width]
                contrast.update(_score_vector(observed, predicted) if available else
                    dict(actual_norm=float(np.linalg.norm(observed)), error_norm=None, relative_error=None))
                contrast['actual_mean_lambda'] = float(case['h']*(actual_gamma[index]-actual_gamma[base]).mean())
                contrast['predicted_mean_lambda'] = float(case['h']*(model_gamma[index]-model_gamma[base]).mean()) if available else None
                contrasts.append(contrast)
            for base, family in families.items():
                for amplitude in (.01, .005, .0025):
                    if amplitude not in family or -amplitude not in family:
                        continue
                    plus, minus = family[amplitude], family[-amplitude]
                    available = bool(valid[plus] and valid[minus])
                    observed = (movement[plus]-movement[minus])/(2*amplitude)
                    predicted = (model_movement[plus]-model_movement[minus])/(2*amplitude)
                    pair = dict(cases[plus], model=model, horizon=horizon, supported=available,
                        surrogate_failed=bool(failed[mi, hi, plus] or failed[mi, hi, minus]),
                        actual_failed=bool(actual_failed[plus] or actual_failed[minus]),
                        truncated=bool((not failed[mi, hi, plus] and achieved[mi, hi, plus] < horizon)
                                       or (not failed[mi, hi, minus] and achieved[mi, hi, minus] < horizon)))
                    for name, block in (('slope', slice(0, width)), ('readout', slice(2*width, 3*width))):
                        result = _score_vector(observed[block], predicted[block]) if available else dict(actual_norm=float(np.linalg.norm(observed[block])), error_norm=None, relative_error=None)
                        pair.update({name+'_'+key: value for key, value in result.items()})
                    pair['actual_mean_lambda_response'] = float(cases[plus]['h']*(actual_gamma[plus]-actual_gamma[minus]).mean()/(2*amplitude))
                    pair['predicted_mean_lambda_response'] = float(cases[plus]['h']*(model_gamma[plus]-model_gamma[minus]).mean()/(2*amplitude)) if available else None
                    paired.append(pair)
    return rows, contrasts, paired, actual_hashes


def run(args):
    if not jax.config.x64_enabled or any(device.platform != 'cpu' for device in jax.devices()):
        raise RuntimeError('FP64 CPU-only comparison')
    if not 0 < args.max_seconds <= 540:
        raise ValueError('Use at most540 internal seconds within the10-minute CPU allocation')
    begin = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.predictions/'inputs.npz'
    pp, x, yy, cases = ef.load_inputs(source)
    manifest = json.loads((args.predictions/'manifest.json').read_text())
    if manifest['input_sha256'] != ef.digest(source):
        raise ValueError('Issued physical inputs changed')
    if any(case['arm'] != 'natural' for case in cases):
        raise ValueError('This comparison uses physical ordinary-GD cases only')
    if len({int(case['source_index']) for case in cases}) != len(cases):
        raise ValueError('Physical input rows must have unique source indices')
    actual_manifest = json.loads((args.actual_run/'manifest.json').read_text())
    if actual_manifest['cases'] != cases:
        raise ValueError('Ordinary-GD outcomes do not match the issued physical cases')
    eta = float(manifest['eta'])
    transform, _ = poly.modal_setup(x, yy[0], 5)
    targets = np.stack([poly.modal_setup(x, y, 5)[1] for y in yy])
    transform, targets_jax = jnp.asarray(transform), jnp.asarray(targets)
    @jax.jit
    def exact(point, target):
        state = kernel.decomposition(point, jnp.asarray(x), target)
        return jnp.stack((state['F'], state['g']))
    initial_exact = np.stack([np.asarray(exact(jnp.asarray(p), jnp.asarray(y))) for p, y in zip(pp, yy)], axis=1)
    initial_polynomial, states, advances = [], [], []
    for model in MODELS:
        evaluate = jax.jit(jax.vmap(lambda point, target: polynomial_field(point, transform, target, model)))
        field = evaluate(jnp.asarray(pp), targets_jax)
        initial_polynomial.append(field)
        state = jax.vmap(initial_state)(jnp.asarray(pp))
        invalid = ~jnp.all(jnp.isfinite(field), axis=1)
        state['failed'] = invalid
        state['failure_step'] = jnp.where(invalid, 0, -1)
        states.append(state)
        advances.append(advance_factory(transform, model, eta))
    count = len(pp)
    predictions = np.full((2, len(HORIZONS), *pp.shape), np.nan)
    supported = np.zeros((2, len(HORIZONS), count), dtype=bool)
    failed = np.zeros_like(supported)
    achieved = np.zeros_like(supported, dtype=np.int64)
    executed = [0, 0]
    timed_out = False
    for hi, horizon in enumerate(HORIZONS):
        while min(executed) < horizon and not timed_out:
            for mi in range(2):
                if executed[mi] >= horizon:
                    continue
                if time.monotonic()-begin >= args.max_seconds:
                    timed_out = True
                    break
                steps = min(250, horizon-executed[mi])
                states[mi] = advances[mi](states[mi], targets_jax, initial_polynomial[mi],
                    jnp.asarray(initial_exact[mi]), steps)
                jax.block_until_ready(states[mi])
                executed[mi] += steps
        for mi in range(2):
            host = jax.device_get(states[mi])
            predictions[mi, hi] = host['p']
            failed[mi, hi] = host['failed']
            achieved[mi, hi] = host['count']
            supported[mi, hi] = ~host['failed'] & (host['count'] == horizon)
        print(json.dumps(dict(horizon=horizon, executed=executed,
            supported=supported[:, hi].sum(axis=1).tolist(), timed_out=timed_out,
            elapsed=time.monotonic()-begin)), flush=True)
    atomic_npz(args.output/'predictions.npz', p=predictions, supported=supported, failed=failed,
        achieved_updates=achieved, p0=pp, horizons=np.asarray(HORIZONS), initial_exact=initial_exact,
        initial_polynomial=np.asarray(initial_polynomial),
        failure_step=np.stack([np.asarray(state['failure_step']) for state in states]))
    # Outcomes enter only after all fixed, fork-only model trajectories are saved.
    rows, contrasts, paired, actual_hashes = score(cases, pp, predictions, supported, failed, achieved, args.actual_run)
    io.write_csv(args.output/'states.csv', rows)
    io.write_csv(args.output/'contrasts.csv', contrasts)
    io.write_csv(args.output/'paired.csv', paired)
    write_json(args.output/'manifest.json', dict(
        source_sha256=ef.digest(__file__), polynomial_sha256=ef.digest(poly.__file__),
        effective_kernel_sha256=ef.digest(kernel.__file__), input_sha256=ef.digest(source),
        original_prediction_manifest_sha256=ef.digest(args.predictions/'manifest.json'),
        actual_snapshots=actual_hashes, cases=cases, models=MODELS, horizons=HORIZONS,
        eta=eta, max_seconds=args.max_seconds, elapsed_seconds=time.monotonic()-begin,
        timed_out=timed_out, completed_nominal_updates=executed,
        supported_counts=supported.sum(axis=2).tolist(),
        failed_counts=failed.sum(axis=2).tolist(), issued_utc=datetime.now(timezone.utc).isoformat(),
        role='Retrospective fixed fork-only coupled-model comparison; outcomes already known; no future fit.',
        anchor='Fine model matches full-complement F0; full polynomial model matches ordinary g0.',
        tracking='Fine model omits tracking; full polynomial model evolves its own coarse error.',
        supported_definition='Finite trajectory at the requested update, not validation of a Taylor regime.',
        target='All six polynomial coefficients include coarse output bias; outside target modes have zero polynomial sensitivity.',
        failure_policy='Freeze last finite state, retain failure/count; truncated or failed states never count as supported predictions.',
        numerical_certificate=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--actual-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=540.)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
