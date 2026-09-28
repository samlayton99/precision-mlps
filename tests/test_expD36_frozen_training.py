import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.expD36_frozen_gamma_probe import core, train


def fixture():
    rng = np.random.default_rng(81)
    j = rng.normal(size=(2, 21, 6))/np.sqrt(21)
    y = rng.normal(size=(2, 21, 2))/np.sqrt(21)
    scale = np.ones((2, 6))
    return j, y, scale


def test_normalization_autodiff_and_coordinate_gradient():
    x = np.linspace(-1, 1, 65)
    g = core.geometry(64)
    a = core.design(x, g.centers, 4)
    y = core.target(x, 'sine_mix_2_6_10')/np.sqrt(len(x))
    theta = np.linspace(-.1, .1, g.width+1)
    for name in ['raw', 'collective']:
        scale = core.scales(g, name)
        j = a*scale
        loss = lambda z: .5*jnp.sum((jnp.asarray(j)@z-y)**2)
        np.testing.assert_allclose(jax.grad(loss)(jnp.asarray(theta)), j.T@(j@theta-y), rtol=2e-13, atol=2e-14)
        physical = np.sqrt(len(x))*(a@(theta*scale)-y)
        np.testing.assert_allclose(loss(theta), .5*np.mean(physical**2))


def test_gd_direct_recurrence_spectrum_and_chunk_resume():
    j, y, scale = fixture()
    rate = np.array([.1, .15])
    state = train.initialize(j, 2)
    before = j.copy()
    kernel = train.make_chunk('gd', 20)
    once, trace = kernel(state, j, y, scale, rate)
    twice, _ = kernel(once, j, y, scale, rate)
    whole, _ = train.make_chunk('gd', 40)(state, j, y, scale, rate)
    np.testing.assert_array_equal(twice['theta'], whole['theta'])
    expected = np.zeros_like(state['theta'])
    for n in range(40):
        if n < 20:
            relative = np.linalg.norm(j@expected-y, axis=1)/np.linalg.norm(y, axis=1)
            np.testing.assert_allclose(np.asarray(trace)[n, :, :, 0], relative, rtol=2e-13)
        expected -= rate[:, None, None]*(j.transpose(0, 2, 1)@(j@expected-y))
    np.testing.assert_allclose(whole['theta'], expected, rtol=2e-13, atol=2e-14)
    for b in range(2):
        u, s, _ = np.linalg.svd(j[b], full_matrices=False)
        alpha = u.T@y[b]
        floor = np.sum((y[b]-u@alpha)**2, axis=0)
        predicted = core.spectral_error(40, s, alpha, floor, np.linalg.norm(y[b], axis=0), rate[b])
        actual = np.linalg.norm(j[b]@expected[b]-y[b], axis=0)/np.linalg.norm(y[b], axis=0)
        np.testing.assert_allclose(actual, predicted, rtol=2e-13)
    np.testing.assert_array_equal(j, before)


def test_adam_matches_optax_and_rescaling():
    j, y, scale = fixture()
    rate = np.array([.001, .002])
    initial = train.initialize(j, 2)
    result, _ = train.make_chunk('adam', 30)(initial, j, y, scale, rate)
    theta = initial['theta']
    tx = optax.adam(1., b1=.9, b2=.999, eps=1e-12, eps_root=0.)
    opt = tx.init(theta)
    for _ in range(30):
        grad = jnp.swapaxes(j, 1, 2)@(j@theta-y)
        update, opt = tx.update(grad, opt, theta)
        theta = theta+rate[:, None, None]*update
    np.testing.assert_allclose(result['theta'], theta, rtol=2e-12, atol=2e-14)
    s = 3.
    scaled, _ = train.make_chunk('adam', 30, epsilon=s*1e-12)(initial, s*j, y, s*scale, rate/s)
    np.testing.assert_allclose(s*scaled['theta'], theta, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(train.adam_multiplier(jnp.array([1, 20000, 50000])), [1, 1, .001])
