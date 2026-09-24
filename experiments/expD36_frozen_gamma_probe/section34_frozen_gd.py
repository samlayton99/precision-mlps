"""Actual FP64 frozen-readout GD for the Section 3.4 width-512 comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

LAMBDAS = [.03125, .0625, .125, .25]
INDICES = [0, 1, 3, 4]


def make_chunk(features, target, rates, size):
    phi, y, eta = map(jnp.asarray, (features, target, rates))
    denom = jnp.sum(y*y)
    def step(state, _):
        weights, residual = state
        gradient = jnp.swapaxes(phi, 1, 2) @ residual / len(target)
        weights = weights-eta[:, None, None]*gradient
        residual = phi @ weights-y[None, :, None]
        error = jnp.sqrt(jnp.sum(residual*residual, axis=1)[:, 0]/denom)
        return (weights, residual), error
    return jax.jit(lambda state: jax.lax.scan(step, state, None, length=size))


def self_test():
    rng = np.random.default_rng(63)
    phi, y = rng.normal(size=(2, 17, 7)), rng.normal(size=17)
    eta = np.array([.05, .1])
    init = (jnp.zeros((2, 7, 1)), jnp.broadcast_to(-jnp.asarray(y)[None, :, None], (2, 17, 1)))
    state, errors = make_chunk(phi, y, eta, 23)(init)
    expected = np.zeros((2, 7, 1))
    for n in range(23):
        expected -= eta[:, None, None]*(phi.swapaxes(1, 2)@(phi@expected-y[None, :, None]))/17
    np.testing.assert_allclose(state[0], expected, rtol=2e-13, atol=2e-14)
    # Independent eigensystem prediction verifies normalization and update indexing.
    for k in range(2):
        u, s, _ = np.linalg.svd(phi[k]/np.sqrt(17), full_matrices=False)
        p = (u.T@y)**2/(y@y)
        floor = np.linalg.norm(y-u@(u.T@y))**2/(y@y)
        predicted = np.sqrt((1-eta[k]*s*s)**46@p+floor)
        np.testing.assert_allclose(errors[-1, k], predicted, rtol=2e-13, atol=2e-14)
    print('self-test passed', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--steps', type=int, default=2_000_000)
    p.add_argument('--chunk', type=int, default=10_000)
    p.add_argument('--require-gpu', action='store_true')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.require_gpu:
        for key in ['SLURM_JOB_ID', 'SLURM_STEP_ID', 'CUDA_VISIBLE_DEVICES']:
            if not os.environ.get(key):
                raise RuntimeError(f'Missing allocated GPU environment: {key}')
        if not any(os.environ.get(k) for k in ['SLURM_STEP_GPUS', 'SLURM_JOB_GPUS', 'SLURM_GPUS_ON_NODE']):
            raise RuntimeError('No Slurm GPU allocation reported')
        if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
            raise RuntimeError(f'Expected one GPU, got {jax.devices()}')
    if args.self_test:
        self_test()
        if args.input is None:
            return
    if args.input is None or args.output is None:
        p.error('--input and --output required')
    data = np.load(args.input)
    phi = data['features'][INDICES]
    y = data['target']
    if phi.shape != (4, 2048, 513):
        raise ValueError('Expected the predeclared W512, M2048 comparison')
    gamma = np.array(LAMBDAS)*467/2
    np.testing.assert_allclose(data['a'][INDICES], np.broadcast_to(gamma[:, None], (4, 512)))
    # One GPU batched SVD supplies step normalization, not a forecast training path.
    singular = np.asarray(jnp.linalg.svd(jnp.asarray(phi/np.sqrt(len(y))), full_matrices=False, compute_uv=False))
    rates = .5/singular[:, 0]**2
    args.output.mkdir(parents=True, exist_ok=True)
    errors = np.lib.format.open_memmap(args.output/'raw_error.npy', mode='w+', dtype='float64', shape=(args.steps+1, 4))
    errors[0] = 1.
    state = (jnp.zeros((4, 513, 1)), jnp.broadcast_to(-jnp.asarray(y)[None, :, None], (4, len(y), 1)))
    checkpoint_steps, checkpoints = [0], [np.asarray(state[0])[:, :, 0]]
    chunk = make_chunk(phi, y, rates, args.chunk)
    started = time.monotonic()
    for begin in range(0, args.steps, args.chunk):
        size = min(args.chunk, args.steps-begin)
        state, trace = (chunk if size == args.chunk else make_chunk(phi, y, rates, size))(state)
        errors[begin+1:begin+size+1] = np.asarray(trace)
        checkpoint_steps.append(begin+size)
        checkpoints.append(np.asarray(state[0])[:, :, 0])
        if (begin+size) % 100_000 == 0 or begin+size == args.steps:
            errors.flush()
            print(json.dumps({'step': begin+size, 'error': errors[begin+size].tolist(), 'seconds': time.monotonic()-started}), flush=True)
    errors.flush()
    np.savez_compressed(args.output/'checkpoints.npz', steps=checkpoint_steps, coefficients=checkpoints, lambdas=LAMBDAS, gammas=gamma, eta=rates)
    summary = dict(steps=args.steps, lambdas=LAMBDAS, gammas=gamma.tolist(), eta=rates.tolist(), final_error=errors[-1].tolist(), seconds=time.monotonic()-started, width=512, samples=len(y), initialization='zero readout', error='raw relative output L2', update='actual feature-matrix gradient; post-update error', input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
