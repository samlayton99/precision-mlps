"""Matched frozen-readout Adam probe with full-horizon learning-rate schedules."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


def recipes(config):
    return [{'schedule': s, 'learning_rate': lr}
            for s in config['schedules'] for lr in config['learning_rates']]


def make_chunk(features, target, config, size):
    """Return post-update errors; carry residual to avoid duplicate forward calls."""
    phi = jnp.asarray(features, dtype=jnp.float64)
    y = jnp.asarray(target, dtype=jnp.float64)
    rec = recipes(config)
    rates = jnp.asarray([r['learning_rate'] for r in rec])[None, None, :]
    cosine = jnp.asarray([r['schedule'] == 'cosine' for r in rec])[None, None, :]
    horizon = config['horizon']
    epsilon = config['epsilon']
    y2 = jnp.sum(y * y)
    samples = phi.shape[1]

    def step(state, _):
        w, m, v, residual, count = state
        gradient = jnp.matmul(jnp.swapaxes(phi, 1, 2), residual) / samples
        m = .9 * m + .1 * gradient
        v = .999 * v + .001 * gradient * gradient
        t = count + 1
        factor = jnp.where(cosine, .5 * (1 + jnp.cos(jnp.pi * count / horizon)), 1.)
        w = w - rates * factor * (m / (1 - .9 ** t)) / (jnp.sqrt(v / (1 - .999 ** t)) + epsilon)
        residual = jnp.matmul(phi, w) - y[None, :, None]
        error = jnp.sqrt(jnp.sum(residual * residual, axis=1) / y2)
        return (w, m, v, residual, t), error

    return jax.jit(lambda state: jax.lax.scan(step, state, None, length=size))


def initial_state(features, target, config):
    g, samples, p = features.shape
    r = len(recipes(config))
    w = jnp.zeros((g, p, r), dtype=jnp.float64)
    residual = jnp.broadcast_to(-jnp.asarray(target)[None, :, None], (g, samples, r))
    return w, w, w, residual, jnp.asarray(0, dtype=jnp.int64)


def self_test():
    rng = np.random.default_rng(191)
    phi = rng.normal(size=(2, 13, 5))
    y = rng.normal(size=13)
    cfg = dict(horizon=31, learning_rates=[.001, .02], epsilon=1e-8,
               schedules=['constant', 'cosine'])
    state, errors = make_chunk(phi, y, cfg, 10)(initial_state(phi, y, cfg))
    expected = np.zeros((10, 2, 4))
    ws = np.zeros((2, 5, 4))
    for gi in range(2):
        for ri, rec in enumerate(recipes(cfg)):
            w = np.zeros(5)
            m = np.zeros(5)
            v = np.zeros(5)
            for n in range(10):
                grad = phi[gi].T @ (phi[gi] @ w - y) / len(y)
                m = .9*m + .1*grad
                v = .999*v + .001*grad**2
                factor = .5*(1+np.cos(np.pi*n/cfg['horizon'])) if rec['schedule']=='cosine' else 1.
                w -= rec['learning_rate']*factor*(m/(1-.9**(n+1)))/(np.sqrt(v/(1-.999**(n+1)))+cfg['epsilon'])
                expected[n, gi, ri] = np.linalg.norm(phi[gi]@w-y)/np.linalg.norm(y)
            ws[gi, :, ri] = w
    np.testing.assert_allclose(np.asarray(errors), expected, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(np.asarray(state[0]), ws, rtol=2e-13, atol=2e-14)
    first, _ = make_chunk(phi, y, cfg, 4)(initial_state(phi, y, cfg))
    split, _ = make_chunk(phi, y, cfg, 6)(first)
    np.testing.assert_allclose(np.asarray(split[0]), ws, rtol=2e-13, atol=2e-14)
    assert .5*(1+np.cos(0)) == 1.
    assert .5*(1+np.cos(np.pi)) == 0.
    assert .5*(1+np.cos(np.pi*.5)) == .5
    print(json.dumps({'self_test': 'passed', 'max_error_difference': float(np.max(np.abs(np.asarray(errors)-expected)))}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--steps', type=int, help='Actual updates, preserving config horizon for benchmarking')
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.require_gpu:
        for key in ['SLURM_JOB_ID', 'SLURM_STEP_ID', 'CUDA_VISIBLE_DEVICES']:
            if not os.environ.get(key):
                raise RuntimeError(f'Required Slurm GPU environment missing: {key}')
        if not (os.environ.get('SLURM_STEP_GPUS') or os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_GPUS_ON_NODE')):
            raise RuntimeError('Slurm does not report allocated GPUs')
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != 'gpu':
            raise RuntimeError(f'Expected exactly one allocated GPU, got {devices}')
    if args.self_test:
        self_test()
        if args.input is None:
            return
    if not all([args.input, args.config, args.output]):
        parser.error('--input, --config and --output are required for training')
    config = json.loads(args.config.read_text())
    if set(config['schedules']) - {'constant', 'cosine'}:
        raise ValueError('Only constant and cosine schedules are supported')
    horizon = int(config['horizon'])
    steps = args.steps if args.steps is not None else horizon
    if not 0 < steps <= horizon:
        raise ValueError('Actual steps must be in [1,horizon]')
    with np.load(args.input) as data:
        features = np.asarray(data['features'], dtype=np.float64)
        target = np.asarray(data['target'], dtype=np.float64)
    if features.ndim != 3 or target.shape != (features.shape[1],) or not np.all(np.isfinite(features)) or not np.all(np.isfinite(target)) or not np.linalg.norm(target):
        raise ValueError('Expected finite features [G,M,P] and nonzero target [M]')
    width = features.shape[2]-1
    if width < 1 or width & (width-1):
        raise ValueError('Total hidden width must be a power of two, excluding output bias')
    args.output.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    metadata = dict(config=config, config_hash=config_hash, input_sha256=input_hash,
                    recipes=recipes(config), axes=['update', 'geometry', 'recipe'],
                    metric='raw relative output L2 error', loss='0.5 * mean((Phi @ w - target)**2)',
                    feature_shape=list(features.shape), width=width, planned_horizon=horizon, actual_steps=steps,
                    devices=[str(d) for d in jax.devices()], cosine_formula='eta0 * (1 + cos(pi*n/horizon))/2; n=0,...,horizon-1',
                    initialization='all readout weights and output bias zero')
    state = initial_state(features, target, config)
    start = 0
    trace_path = args.output / 'relative_error.npy'
    checkpoint_path = args.output / 'state.npz'
    if args.resume:
        previous = json.loads((args.output/'metadata.json').read_text())
        if previous['config_hash'] != config_hash or previous['input_sha256'] != input_hash or previous['actual_steps'] != steps:
            raise ValueError('Resume metadata differs')
        with np.load(checkpoint_path) as saved:
            start = int(saved['count'])
            w, m, v = [jnp.asarray(saved[k]) for k in ['w', 'm', 'v']]
        residual = jnp.matmul(jnp.asarray(features), w) - jnp.asarray(target)[None, :, None]
        state = w, m, v, residual, jnp.asarray(start, dtype=jnp.int64)
        trace = np.lib.format.open_memmap(trace_path, mode='r+')
    else:
        if trace_path.exists():
            raise FileExistsError(f'{trace_path} already exists; use a separate output or --resume')
        trace = np.lib.format.open_memmap(trace_path, mode='w+', dtype=np.float64,
                                         shape=(steps+1, features.shape[0], len(recipes(config))))
        trace[0] = 1.
    checkpoint_steps = np.unique(np.r_[np.arange(0, steps+1, 1000), steps])
    weights_path = args.output/'readout_checkpoints.npy'
    if args.resume:
        weights = np.lib.format.open_memmap(weights_path, mode='r+')
    else:
        weights = np.lib.format.open_memmap(weights_path, mode='w+', dtype=np.float64,
                                            shape=(len(checkpoint_steps), features.shape[0], len(recipes(config)), features.shape[2]))
        weights[0] = 0.
        np.save(args.output/'checkpoint_steps.npy', checkpoint_steps)
    metadata['readout_checkpoint_axes'] = ['checkpoint', 'geometry', 'recipe', 'parameter']
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    chunk_size = min(int(config.get('chunk_size', 1000)), steps)
    if 1000 % chunk_size and steps > chunk_size:
        raise ValueError('chunk_size must divide 1000 to retain every 1000-update checkpoint')
    chunk = make_chunk(features, target, config, chunk_size)
    t0 = time.perf_counter()
    # Compile with the requested shapes, without advancing the optimization state.
    compiled = chunk.lower(state).compile()
    compilation_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    compute_seconds = 0.
    for left in range(start, steps, chunk_size):
        size = min(chunk_size, steps-left)
        fn = compiled if size == chunk_size else make_chunk(features, target, config, size)
        tick = time.perf_counter()
        state, errors = fn(state)
        host_errors = np.asarray(errors)  # Synchronizes device execution before timing.
        compute_seconds += time.perf_counter()-tick
        count = left+size
        trace[left+1:count+1] = host_errors
        if count in checkpoint_steps:
            wi = int(np.searchsorted(checkpoint_steps, count))
            weights[wi] = np.asarray(state[0]).transpose(0, 2, 1)
        if count % 10000 == 0 or count == steps:
            trace.flush()
            weights.flush()
            w, m, v, _, _ = state
            temporary = args.output/'state.tmp.npz'
            np.savez(temporary, w=np.asarray(w), m=np.asarray(m), v=np.asarray(v), count=count)
            temporary.replace(checkpoint_path)
        print(json.dumps(dict(update=count, elapsed_seconds=time.perf_counter()-t0,
                              finite_cases=int(np.isfinite(host_errors[-1]).sum()),
                              min_error=float(np.nanmin(host_errors[-1])), max_error=float(np.nanmax(host_errors[-1])))), flush=True)
    elapsed = time.perf_counter()-t0
    summary = dict(completed_updates=steps, resumed_at=start, compilation_seconds=compilation_seconds,
                   elapsed_seconds=elapsed, synchronized_compute_seconds=compute_seconds,
                   updates_per_second=(steps-start)/compute_seconds if compute_seconds else None,
                   final_relative_error=np.asarray(trace[steps]).tolist())
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
