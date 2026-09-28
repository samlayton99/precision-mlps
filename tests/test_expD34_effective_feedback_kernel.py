"""Independent derivatives and causal-update checks for effective-force probes."""
import os
os.environ.setdefault('JAX_ENABLE_X64', 'true')

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from experiments.expD34_readout_race import effective_feedback_kernel as ef
from experiments.expD34_readout_race import targets, transport


def example(width=7, degree=9):
    rng = np.random.default_rng(934)
    p = np.r_[rng.normal(size=3*width)*.7, .13]
    x = targets.grid(128)
    y = np.sin(2*np.pi*x)+.12*x**2
    q = transport.basis(x, degree)
    return jnp.asarray(p), ef.fork_context(p, x, y, q=q)


def independent(p, context):
    pp = torch.tensor(np.asarray(p), requires_grad=True)
    x, y, q = (torch.tensor(np.asarray(context[k])) for k in ('x', 'y', 'q'))
    width = (len(p)-1)//3
    def output(v):
        a, b, c = v[:-1].reshape(3, width)
        return torch.tanh(x[:, None]*a+b) @ c+v[-1]
    r = output(pp)-y
    loss = .5*torch.mean(r*r)
    g = torch.autograd.grad(loss, pp)[0].numpy()
    J = torch.autograd.functional.jacobian(lambda v: q.T @ output(v)/len(x), pp).numpy()
    e = (q.T @ r/len(x)).detach().numpy()
    C = J[:2] @ J[:2].T
    B = np.linalg.solve(C, J[:2] @ J[2:].T)
    T = J[2:].T-J[:2].T @ B
    tracking = J[:2].T @ (e[:2]+B @ e[2:])
    omitted = g-J.T @ e
    return g, J, T, e, tracking, omitted


@pytest.mark.parametrize('width,degree', [(4, 3), (7, 9), (11, 65)])
def test_matrix_free_force_matches_independent_autograd(width, degree):
    p, context = example(width, degree)
    g, J, T, e, tracking, omitted = independent(p, context)
    actual, ch = ef.field(p, context)
    mats = ef.matrices(p, context)
    np.testing.assert_allclose(actual, g, rtol=3e-13, atol=2e-15)
    np.testing.assert_allclose(mats['J'], J, rtol=4e-12, atol=3e-15)
    np.testing.assert_allclose(mats['T'], T, rtol=4e-12, atol=3e-15)
    for name, expected in [('effective_a', (T @ e[2:])[:width]),
                           ('tracking_a', tracking[:width]), ('omitted_a', omitted[:width])]:
        np.testing.assert_allclose(ch[name], expected, atol=3e-15, rtol=4e-12)
    assert bool(ch['coarse_resolved'])
    assert float(ch['reconstruction_norm']) < 4e-15
    np.testing.assert_allclose(mats['J_C'] @ mats['T'], 0, atol=5e-15)


def test_first_update_matches_and_second_step_is_exact_factor_intervention():
    p0, context = example()
    eta = .02
    initial = {arm: ef.field(p0, context, arm)[0] for arm in ef.ARMS}
    for arm in ef.ARMS:
        np.testing.assert_allclose(initial[arm], initial['joint'], atol=2e-16, rtol=2e-14)
    p1 = p0-eta*initial['joint']
    g1, ch1 = ef.field(p1, context)
    T0 = np.asarray(context['T_a0'])
    T1 = np.asarray(ef.matrices(p1, context)['T_a'])
    e0, e1 = np.asarray(context['eH0']), np.asarray(ch1['eH'])
    width = T0.shape[0]
    for arm, desired, second_difference in [
        ('freeze_map', T0 @ e1, -eta*((T0-T1) @ e1)),
        ('clamp_residual', T1 @ e0, -eta*(T1 @ (e0-e1)))]:
        ga, ch = ef.field(p1, context, arm)
        np.testing.assert_allclose(ch['applied_effective_a'], desired, atol=2e-15)
        np.testing.assert_array_equal(ga[width:], g1[width:])
        np.testing.assert_allclose((p1-eta*ga)[:width]-(p1-eta*g1)[:width],
                                   second_difference, atol=2e-16, rtol=3e-11)
        np.testing.assert_allclose(ch['remainder_a'], g1[:width]-T1 @ e1, atol=2e-15)
    assert np.linalg.norm(T1-T0) > 1e-6
    assert np.linalg.norm(e1-e0) > 1e-6


@pytest.mark.parametrize('arm', ef.ARMS)
def test_simultaneous_own_state_updates_keep_nonslope_gradient_and_fork_fixed(arm):
    p, context = example()
    initial_map = np.asarray(context['T_a0']).copy()
    initial_residual = np.asarray(context['eH0']).copy()
    for _ in range(4):
        g, _, T, e, _, _ = independent(p, context)
        applied, ch = ef.field(p, context, arm)
        width = T.shape[0]//3
        expected = g.copy()
        if arm == 'freeze_map':
            expected[:width] += initial_map @ e[2:]-(T @ e[2:])[:width]
        elif arm == 'clamp_residual':
            expected[:width] += (T @ (initial_residual-e[2:]))[:width]
        np.testing.assert_allclose(applied, expected, atol=3e-15, rtol=3e-12)
        p = p-.03*applied
    np.testing.assert_array_equal(context['T_a0'], initial_map)
    np.testing.assert_array_equal(context['eH0'], initial_residual)


def test_context_and_field_support_vmap_and_jit():
    p, context = example()
    ps = jnp.stack([p, p+.01])
    ys = jnp.stack([context['y'], context['y']+.02*context['x']])
    contexts = jax.vmap(lambda pp, yy: ef.fork_context(pp, context['x'], yy, q=context['q']))(ps, ys)
    batched = jax.jit(jax.vmap(lambda pp, cc: ef.field(pp, cc, 'freeze_map')[0]))(ps, contexts)
    expected = jnp.stack([ef.field(ps[i], {k: v[i] for k, v in contexts.items()}, 'joint')[0] for i in range(2)])
    np.testing.assert_allclose(batched, expected, atol=3e-15, rtol=2e-13)


def test_unresolved_coarse_projection_is_flagged_without_regularization():
    p, context = example()
    p = jnp.zeros_like(p)
    _, ch = ef.field(p, context)
    assert not bool(ch['coarse_resolved'])


def test_realized_step_motion_keeps_zero_crossing_correction():
    p, context = example()
    p = p.at[0].set(0.)
    applied, _ = ef.field(p, context, 'clamp_residual')
    width = (len(p)-1)//3
    a = np.asarray(p[:width])
    increment = -100*np.asarray(applied[:width])
    new = a+increment
    crossing = a*new < 0
    assert crossing.any()
    correction = np.where(a == 0, np.abs(increment),
                          np.where(crossing, 2*np.abs(new), 0.))
    delta = np.abs(new)-np.abs(a)
    np.testing.assert_allclose(delta, np.sign(a)*increment+correction, atol=3e-15)
    np.testing.assert_allclose(np.maximum(delta, 0)-np.maximum(-delta, 0), delta, atol=0)
