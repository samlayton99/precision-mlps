"""Full quintic loss, own-fork anchoring, and failed-state accounting."""
import jax
import jax.numpy as jnp
import numpy as np

from experiments.expD34_readout_race import mechanism_persistence_coupled_response as response
from experiments.expD34_readout_race import mechanism_polynomial as poly
from experiments.expD34_readout_race import mechanism_persistence_kernel as kernel


def example():
    p = jnp.asarray([.2, -.4, .3, .1, -.2, .2, .7, -.5, .4, .3])
    x = np.linspace(-.97, .97, 33)
    y = np.sin(4*x)+.1*x*x+.2
    transform, target = poly.modal_setup(x, y, 5)
    return p, jnp.asarray(x), jnp.asarray(y), jnp.asarray(transform), jnp.asarray(target)


def test_full_polynomial_gradient_matches_direct_empirical_autodiff():
    p, x, y, transform, target = example()
    def loss(point):
        a, b, c = point[:-1].reshape(3, -1)
        u = x[:, None]*a+b
        value = (u-u**3/3+2*u**5/15)@c+point[-1]
        return jnp.mean((value-y)**2)/2
    expected = jax.grad(loss)(p)
    np.testing.assert_allclose(response.full_gradient(p, transform, target), expected,
                               atol=2e-15, rtol=3e-13)
    modal_gradient = response.full_gradient(p, transform, target)
    # The bias is part of the coarse gradient and must not be projected away.
    assert abs(float(modal_gradient[-1])) > .01
    np.testing.assert_array_equal(response.polynomial_field(p, transform, target, 'anchored_fine'),
                                  poly.field(p, transform, target, 5)[0])


def test_both_anchors_match_their_exact_force_and_full_model_first_gd_step():
    p, x, y, transform, target = example()
    exact = kernel.decomposition(p, x, y)
    eta = .002
    for model, field in (('anchored_fine', exact['F']), ('anchored_full', exact['g'])):
        polynomial = response.polynomial_field(p, transform, target, model)
        np.testing.assert_array_equal(response.anchored_gradient(p, transform, target,
                                      polynomial, field, model), field)
        state = jax.vmap(response.initial_state)(p[None])
        advance = response.advance_factory(transform, model, eta)
        result = advance(state, target[None], polynomial[None], field[None], 1)
        np.testing.assert_allclose(result['p'][0], p-eta*field, atol=2e-16, rtol=2e-15)
        assert int(result['count'][0]) == 1
        assert not bool(result['failed'][0])
        second = advance(result, target[None], polynomial[None], field[None], 1)
        direct = result['p'][0]-eta*response.anchored_gradient(result['p'][0], transform,
                               target, polynomial, field, model)
        np.testing.assert_allclose(second['p'][0], direct, atol=3e-16, rtol=2e-15)


def test_nonfinite_polynomial_step_freezes_state_and_preserves_first_failure():
    _, _, _, transform, target = example()
    point = jnp.full((1, 10), 1e100)
    state = jax.vmap(response.initial_state)(point)
    advance = response.advance_factory(transform, 'anchored_full', .002)
    result = advance(state, target[None], jnp.zeros_like(point), jnp.zeros_like(point), 3)
    np.testing.assert_array_equal(result['p'], point)
    assert bool(result['failed'][0])
    assert int(result['count'][0]) == 0
    assert int(result['failure_step'][0]) == 1
