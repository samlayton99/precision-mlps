"""Batched ordinary updates; columns are independent targets or recipes."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from . import full_core

jax.config.update('jax_enable_x64', True)
TRACE = ['train_relative_error', 'half_mse', 'gradient_l2', 'native_step_l2',
         'physical_step_l2', 'native_parameter_l2', 'physical_parameter_l2', 'zero_motion', 'rate']


def initialize(theta, tolerance_count):
    theta = jnp.asarray(theta)
    return dict(theta=theta, mu=jnp.zeros_like(theta), nu=jnp.zeros_like(theta),
                count=jnp.array(0, dtype=jnp.int32),
                failed=jnp.zeros((theta.shape[0], theta.shape[2]), dtype=jnp.int32),
                hits=jnp.full((theta.shape[0], theta.shape[2], tolerance_count), -1, dtype=jnp.int32))


def multiplier(count, horizon=50000):
    fraction = jnp.clip((count-.4*horizon)/(.6*horizon), 0., 1.)
    return .001+.999*.5*(1+jnp.cos(jnp.pi*fraction))


def make_chunk(optimizer, neighbor, steps=500, pilot_steps=50000):
    def chunk(state, j, y, scales, rates, epsilons, tolerances):
        denominator = jnp.linalg.norm(y, axis=1)
        denominator = jnp.where(denominator > 0, denominator, 1.)
        def update(current, _):
            theta = current['theta']
            residual = j@theta-y
            gradient = jnp.swapaxes(j, 1, 2)@residual
            count = current['count']+1
            if optimizer == 'adam':
                mu = .9*current['mu']+.1*gradient
                nu = .999*current['nu']+.001*gradient**2
                direction = (mu/(1-.9**count))/(jnp.sqrt(nu/(1-.999**count))+epsilons[:, None, :])
                rate = rates*multiplier(count, pilot_steps)
            else:
                mu, nu, direction, rate = current['mu'], current['nu'], gradient, rates
            candidate = theta-rate[:, None, :]*direction
            finite = (jnp.all(jnp.isfinite(candidate), axis=1) & jnp.all(jnp.isfinite(residual), axis=1)
                      & jnp.all(jnp.isfinite(gradient), axis=1) & jnp.all(jnp.isfinite(mu), axis=1)
                      & jnp.all(jnp.isfinite(nu), axis=1))
            active = (current['failed'] == 0) & finite
            next_theta = jnp.where(active[:, None, :], candidate, theta)
            delta = next_theta-theta
            physical = full_core.decode(theta, scales, neighbor, jnp)
            physical_delta = full_core.decode(delta, scales, neighbor, jnp)
            sq = jnp.sum(residual**2, axis=1)
            error = jnp.sqrt(sq)/denominator
            hits = jnp.where((current['hits'] < 0) & (error[:, :, None] <= tolerances)
                            & (current['failed'][:, :, None] == 0), current['count'], current['hits'])
            stats = jnp.stack([error, .5*sq, jnp.linalg.norm(gradient, axis=1),
                jnp.linalg.norm(delta, axis=1), jnp.linalg.norm(physical_delta, axis=1),
                jnp.linalg.norm(theta, axis=1), jnp.linalg.norm(physical, axis=1),
                jnp.all(delta == 0, axis=1).astype(jnp.float64), rate], axis=-1)
            state = dict(theta=next_theta, count=count, hits=hits,
                mu=jnp.where(active[:, None, :], mu, current['mu']),
                nu=jnp.where(active[:, None, :], nu, current['nu']),
                failed=jnp.where((current['failed'] == 0) & ~finite, count, current['failed']))
            return state, stats
        return jax.lax.scan(update, state, None, length=steps)
    return jax.jit(chunk)


def joint_initial(width, seeds):
    states = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6/(width+1))
        states.append(np.stack([rng.uniform(-limit, limit, width), np.zeros(width),
                               rng.uniform(-limit, limit, width)]))
    p = dict(hidden=jnp.asarray(np.stack(states)), bias=jnp.zeros(len(seeds)))
    return dict(params=p, mu=jax.tree.map(jnp.zeros_like, p), nu=jax.tree.map(jnp.zeros_like, p),
                count=jnp.array(0, dtype=jnp.int32), failed=jnp.zeros(len(seeds), dtype=jnp.int32))


def joint_predict(params, x):
    a, b, w = params['hidden']
    return jnp.tanh(x[:, None]*a+b)@w+params['bias']


def make_joint_chunk(steps=500):
    def loss(p, x, y):
        return .5*jnp.mean((joint_predict(p, x)-y)**2)
    vg = jax.vmap(jax.value_and_grad(loss), in_axes=(0, None, None))
    def chunk(state, x, y, rate=.001, epsilon=1e-8):
        def update(current, _):
            values, gradient = vg(current['params'], x, y)
            count = current['count']+1
            mu = jax.tree.map(lambda m,g: .9*m+.1*g, current['mu'], gradient)
            nu = jax.tree.map(lambda v,g: .999*v+.001*g*g, current['nu'], gradient)
            candidate = jax.tree.map(lambda p,m,v: p-rate*(m/(1-.9**count))/(jnp.sqrt(v/(1-.999**count))+epsilon),
                                     current['params'], mu, nu)
            finite = jnp.isfinite(values)
            for value in jax.tree.leaves((candidate, mu, nu, gradient)):
                finite &= jnp.all(jnp.isfinite(value).reshape((len(values), -1)), axis=1)
            active = (current['failed'] == 0) & finite
            select = lambda new,old: jnp.where(active.reshape((len(values),)+(1,)*(new.ndim-1)), new, old)
            return dict(params=jax.tree.map(select, candidate, current['params']),
                mu=jax.tree.map(select, mu, current['mu']), nu=jax.tree.map(select, nu, current['nu']),
                count=count, failed=jnp.where((current['failed'] == 0) & ~finite, count, current['failed'])), values
        return jax.lax.scan(update, state, None, length=steps)
    return jax.jit(chunk)
