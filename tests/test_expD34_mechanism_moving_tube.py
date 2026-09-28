import jax.numpy as jnp
import numpy as np

from experiments.expD34_readout_race import mechanism_moving_tube as mt
from experiments.expD34_readout_race import persistence_theory as pt


def example():
    rng = np.random.default_rng(515)
    return rng.normal(size=13)*.2, np.linspace(-1., 1., 32)


def test_reference_quantities_and_uniform_bounds_match_independent_numpy():
    p, x = example(); y = np.sin(2*x)
    state = mt.reference_state(jnp.asarray(p), jnp.asarray(x), jnp.asarray(y))
    independent = pt.tensors(p, x, y)
    np.testing.assert_allclose(state['g'], independent['g'], atol=1e-15)
    np.testing.assert_allclose(state['jacobian_bound'], independent['jacobian_bound'], atol=1e-15)
    for radius in (0., .001, .1):
        expected = pt.ball_constants(p, independent, radius, eta=.002)
        beta, negative, upper = mt.ball_map_bound(jnp.asarray(p), state, radius, .002)
        np.testing.assert_allclose([beta, negative, upper],
                                   [expected[k] for k in ('beta', 'negative', 'upper')], atol=1e-14)


def test_full_parameter_error_enclosed_against_independent_actual_updates():
    p, x = example(); y = .05*x
    T = np.zeros((13, 2)); T[0, 0] = .03; T[1, 1] = -.02
    result = mt.enclose(*map(jnp.asarray, (p, x, y, T, np.array([.1, -.2]))),
                        .002, 1/64, .25, steps=20, save_updates=tuple(range(1, 21)))
    actual = p.copy()
    for k in range(20):
        actual -= .002*pt.tensors(actual, x, y)['g']
        assert bool(result['evaluated'][k])
        error = np.linalg.norm(actual-np.asarray(result['checkpoint_p'][k]))
        assert error <= float(result['radius'][k])+1e-14


def test_discrete_map_bound_keeps_positive_curvature_endpoint():
    p, x = example(); y = np.sin(x)
    state = mt.reference_state(jnp.asarray(p), jnp.asarray(x), jnp.asarray(y))
    beta, negative, upper = map(float, mt.ball_map_bound(jnp.asarray(p), state, .001, 10.))
    assert beta >= abs(1-10.*upper)
    assert beta >= 1+10.*negative


def test_censoring_and_json_numbers_are_explicit():
    p, x = example(); y = np.sin(x)
    result = mt.enclose(*map(jnp.asarray, (p, x, y, np.zeros((13, 2)), np.zeros(2))),
                        .002, 1., 1e-12, steps=3, save_updates=(1, 2, 3))
    np.testing.assert_array_equal(result['evaluated'], [True, False, False])
    assert np.isnan(np.asarray(result['checkpoint_p'])[1:]).all()
    assert mt.finite_number(np.inf) is None
    assert mt.finite_number(np.nan) is None
    assert mt.finite_number(.5) == .5
