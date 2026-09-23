"""Run on the campaign FP64 CPU allocation with the reduced-field audit."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.expD34_readout_race import mechanism_polynomial as polynomial
from experiments.expD34_readout_race.population_reduced_audit import comparison, gram_status, ratio


@pytest.mark.parametrize('degree', [3, 5])
def test_polynomial_field_matches_direct_sample_jacobian(degree):
    assert jax.config.x64_enabled
    x = np.linspace(-1., 1., 65)
    y = np.sin(4*x)+.4*x*x
    p = jnp.array([.3, -.2, .1, -.15, .4, -.7, .2])
    def output(point):
        a, b, c = point[:-1].reshape(3, -1)
        u = x[:, None]*a+b
        h = u-u**3/3
        if degree == 5:
            h = h+2*u**5/15
        return h@c+point[-1]
    jac = np.asarray(jax.jacfwd(output)(p))
    basis = np.stack([np.ones_like(x), x/np.sqrt(np.mean(x*x))], axis=1)
    jc = basis.T@jac/len(x)
    residual = np.asarray(output(p))-y
    fine = residual-basis@(basis.T@residual/len(x))
    raw = jac.T@fine/len(x)
    expected = raw-jc.T@np.linalg.solve(jc@jc.T, jc@raw)
    transform, target = polynomial.modal_setup(x, y, degree)
    actual, gram = polynomial.field(p, jnp.asarray(transform), jnp.asarray(target), degree)
    np.testing.assert_allclose(actual, expected, atol=2e-16, rtol=2e-12)
    np.testing.assert_allclose(gram, jc@jc.T, atol=2e-15, rtol=2e-13)


def test_linear_network_has_no_effective_fine_force():
    x = np.linspace(-1., 1., 31)
    p = jnp.array([.3, -.2, .1, -.15, .4, -.7, .2])
    transform, target = polynomial.modal_setup(x, x**5, 1)
    actual, _ = polynomial.field(p, jnp.asarray(transform), jnp.asarray(target), 1)
    np.testing.assert_array_equal(actual, np.zeros(len(p)))


def test_zero_denominators_and_singular_gram_are_explicit():
    assert ratio(0., 0.) is None
    assert ratio(1e-100, 1e-100) == 1.
    assert ratio(np.nan, 1.) is None
    status = gram_status(np.diag([1., 0.]))
    assert not status['gram_positive']
    assert status['gram_condition'] is None
    row = comparison(np.ones(2), np.zeros(2), np.ones(2), np.zeros(2))
    assert row['relative_error'] is None
    assert row['cosine'] is None
    assert row['positive_relative_error'] is None


def test_positive_motion_sign_and_zero_slope_derivative():
    a = np.array([1., -1., 0.])
    exact = np.array([-2., 3., -4.])
    row = comparison(a, exact, exact, exact*0)
    assert row['exact_positive_norm'] == np.linalg.norm([2., 3., 4.])
    assert row['positive_error_norm'] == 0.
    assert row['signed_positive_discrepancy'] == 0.
