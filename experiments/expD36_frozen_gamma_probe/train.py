"""Ordinary frozen-readout GD and Adam; no diagnostic solve enters an update."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import core

jax.config.update('jax_enable_x64', True)
TRACE = ['train_relative_error', 'half_mse', 'gradient_l2', 'native_step_l2',
         'physical_step_l2', 'native_parameter_l2', 'physical_parameter_l2', 'zero_motion', 'rate']


def initialize(j, targets):
    shape = (j.shape[0], j.shape[2], targets)
    return dict(theta=jnp.zeros(shape), mu=jnp.zeros(shape), nu=jnp.zeros(shape),
                count=jnp.array(0, dtype=jnp.int32),
                failed=jnp.zeros((shape[0], targets), dtype=jnp.int32))


def adam_multiplier(update, horizon=50000):
    fraction = (update-.4*horizon)/(.6*horizon)
    return jnp.where(update <= .4*horizon, 1., .001+.999*.5*(1+jnp.cos(jnp.pi*fraction)))


def make_chunk(optimizer, steps, epsilon=1e-12, adam_horizon=50000):
    adam = optax.scale_by_adam(b1=.9, b2=.999, eps=epsilon, eps_root=0., mu_dtype=jnp.float64)

    def chunk(state, j, y, scale, rates):
        norms = jnp.linalg.norm(y, axis=1)

        def step(current, _):
            theta = current['theta']
            residual = j @ theta-y
            grad = jnp.swapaxes(j, 1, 2) @ residual
            count = current['count']+1
            if optimizer == 'adam':
                direction, moments = adam.update(grad, optax.ScaleByAdamState(
                    count=current['count'], mu=current['mu'], nu=current['nu']))
                rate = rates*adam_multiplier(count, adam_horizon)
                mu, nu = moments.mu, moments.nu
            else:
                direction, rate = grad, rates
                mu, nu = current['mu'], current['nu']
            candidate = theta-rate[:, None, None]*direction
            finite = (jnp.all(jnp.isfinite(candidate), axis=1)
                      & jnp.all(jnp.isfinite(grad), axis=1)
                      & jnp.all(jnp.isfinite(residual), axis=1)
                      & jnp.all(jnp.isfinite(mu), axis=1)
                      & jnp.all(jnp.isfinite(nu), axis=1))
            active = (current['failed'] == 0) & finite
            new_theta = jnp.where(active[:, None, :], candidate, theta)
            movement = new_theta-theta
            mse2 = jnp.sum(residual*residual, axis=1)
            stats = jnp.stack((jnp.sqrt(mse2)/norms, .5*mse2,
                jnp.linalg.norm(grad, axis=1), jnp.linalg.norm(movement, axis=1),
                jnp.linalg.norm(movement*scale[:, :, None], axis=1),
                jnp.linalg.norm(theta, axis=1), jnp.linalg.norm(theta*scale[:, :, None], axis=1),
                jnp.all(movement == 0, axis=1).astype(jnp.float64),
                jnp.broadcast_to(rate[:, None], norms.shape)), axis=-1)
            updated = dict(theta=new_theta, count=count,
                mu=jnp.where(active[:, None, :], mu, current['mu']),
                nu=jnp.where(active[:, None, :], nu, current['nu']),
                failed=jnp.where((current['failed'] == 0) & ~finite, count, current['failed']))
            return updated, stats

        return jax.lax.scan(step, state, None, length=steps)

    return jax.jit(chunk)


def verify_gpu(root, name):
    job, step = os.environ.get('SLURM_JOB_ID'), os.environ.get('SLURM_STEP_ID')
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not job or not step or not mask:
        raise RuntimeError('Expected a Slurm GPU step and its nonempty CUDA mask')
    job_info = subprocess.check_output(['scontrol', 'show', 'job', job], text=True)
    step_info = subprocess.check_output(['scontrol', 'show', 'step', f'{job}.{step}'], text=True)
    if 'JobState=RUNNING' not in job_info or 'State=RUNNING' not in step_info or 'gpu' not in step_info.lower():
        raise RuntimeError('No active GPU allocation in this step')
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != 'gpu':
        raise RuntimeError(f'Expected exactly one allocated GPU, got {devices}')
    core.write_json(root/f'environment_{name}.json', dict(job=job_info, step=step_info,
        cuda_mask=mask, devices=[str(d) for d in devices], jax=jax.__version__, optax=optax.__version__,
        source_commit=os.environ.get('PROBE_SOURCE_COMMIT', 'local')))


def load(root, name, cfg):
    common = dict(np.load(root/'data/common.npz'))
    banks = [dict(np.load(root/f'data/{name}_g{gamma}.npz')) for gamma in cfg['gammas']]
    return common, banks


def advance(root, name, optimizer, cfg, common, banks, frontier, deadline, rate=None):
    tag = f'{optimizer}_{name}' + (f'_lr{rate:g}' if rate is not None else '')
    folder = root/'training'/tag
    folder.mkdir(parents=True, exist_ok=True)
    target_indices = [0, 1] if optimizer == 'gd' else [0]
    j_host = np.stack([b['J'] for b in banks])
    hashes = [core.array_hash(j) for j in j_host]
    j = jnp.asarray(j_host)
    y_host = np.broadcast_to(common['y'][:, target_indices], (len(banks), len(common['y']), len(target_indices))).copy()
    y = jnp.asarray(y_host)
    scale = jnp.asarray(np.stack([b['scales'] for b in banks]))
    rates = jnp.asarray([.5/float(b['L']) if optimizer == 'gd' else rate for b in banks])
    state = initialize(j, len(target_indices))
    trace_parts, evaluations = [], []
    if (folder/'state.npz').exists():
        state = {k: jnp.asarray(v) for k, v in dict(np.load(folder/'state.npz')).items()}
        trace_parts = [np.load(folder/'trace.npz')['trace']]
        evaluations = json.loads((folder/'evaluations.json').read_text())
    at = int(state['count'])
    if at >= frontier:
        return True
    core.write_json(folder/'case.json', dict(optimizer=optimizer, map=name, gammas=cfg['gammas'],
        targets=[cfg['targets'][i] for i in target_indices], rates=np.asarray(rates).tolist(),
        epsilon=cfg['adam_epsilon'] if optimizer == 'adam' else None, trace_columns=TRACE,
        trace_convention='row n measures state n before update n+1; endpoint saved separately',
        input_matrix_hashes=hashes, source_commit=os.environ.get('PROBE_SOURCE_COMMIT', 'local')))
    kernel = make_chunk(optimizer, cfg['chunk_size'], cfg['adam_epsilon'], cfg['adam_steps'])
    sample_steps = {0, 1000, 5000, 20000, 50000, 100000}
    if optimizer == 'adam':
        sample_steps |= {40000, 42500, 45000, 47500}
    first_chunk_seconds = None
    begin = time.monotonic()

    def evaluate():
        theta = np.asarray(state['theta'])
        train = np.linalg.norm(j_host@theta-y_host, axis=1)/np.linalg.norm(y_host, axis=1)
        ev, val, spectral = [], [], []
        for index, bank in enumerate(banks):
            ev.append((np.linalg.norm(bank['J_eval']@theta[index]-common['y_eval'][:, target_indices], axis=0)
                       /np.linalg.norm(common['y_eval'][:, target_indices], axis=0)).tolist())
            val.append((np.linalg.norm(bank['J_validation']@theta[index]-common['y_validation'][:, target_indices], axis=0)
                        /np.linalg.norm(common['y_validation'][:, target_indices], axis=0)).tolist())
            if optimizer == 'gd':
                prediction = core.spectral_error(at, bank['singular'][bank['keep']],
                    bank['loadings'], bank['floor_sq'], bank['norm_y'], .5/float(bank['L']))
                spectral.append(prediction.tolist())
                np.testing.assert_allclose(train[index], prediction, rtol=2e-7, atol=1e-11,
                    err_msg=f'GD/spectrum discrepancy: {tag} gamma={cfg["gammas"][index]} step={at}')
        row = dict(step=at, train=train.tolist(), evaluation=ev, validation=val,
                   spectral=spectral, failed=np.asarray(state['failed']).tolist())
        evaluations.append(row)
        core.save_arrays(folder/f'checkpoint_{at:06d}.npz', **jax.tree.map(np.asarray, state))
        print(f'TRAIN {tag} step={at} train={train.ravel().tolist()} eval={ev}', flush=True)

    if at == 0:
        evaluate()
    while at < frontier and time.monotonic() < deadline:
        chunk_start = time.monotonic()
        state, trace = kernel(state, j, y, scale, rates)
        jax.block_until_ready(state)
        if first_chunk_seconds is None:
            first_chunk_seconds = time.monotonic()-chunk_start
        trace_parts.append(np.asarray(trace))
        at = int(state['count'])
        if at in sample_steps:
            evaluate()
        if at % 5000 == 0 or at >= frontier or time.monotonic() >= deadline:
            all_trace = np.concatenate(trace_parts, axis=0)
            core.save_arrays(folder/'trace.npz', trace=all_trace)
            core.save_arrays(folder/'state.npz', **jax.tree.map(np.asarray, state))
            core.write_json(folder/'evaluations.json', evaluations)
            core.write_json(folder/'progress.json', dict(step=at, requested_frontier=frontier,
                elapsed_seconds=time.monotonic()-begin, first_chunk_seconds=first_chunk_seconds,
                complete=at >= frontier, failed=np.asarray(state['failed']).tolist()))
            trace_parts = [all_trace]
    assert hashes == [core.array_hash(j) for j in j_host], 'Frozen matrix changed'
    return at >= frontier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--map', choices=['raw', 'collective'], required=True)
    parser.add_argument('--seconds', type=float, default=3100.)
    parser.add_argument('--require-gpu', action='store_true')
    args = parser.parse_args()
    deadline = time.monotonic()+args.seconds
    cfg = core.config()
    if args.require_gpu:
        verify_gpu(args.root, args.map)
    common, banks = load(args.root, args.map, cfg)
    for frontier in cfg['gd_frontiers']:
        if not advance(args.root, args.map, 'gd', cfg, common, banks, frontier, deadline):
            return
    for rate in cfg['adam_rates']:
        if time.monotonic() >= deadline:
            return
        if not advance(args.root, args.map, 'adam', cfg, common, banks, cfg['adam_steps'], deadline, rate):
            return


if __name__ == '__main__':
    main()
