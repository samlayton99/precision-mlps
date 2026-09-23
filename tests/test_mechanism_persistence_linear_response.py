import numpy as np
import pytest

from experiments.expD34_readout_race.mechanism_persistence_linear_response import spectral_movement, operator_movement


@pytest.mark.parametrize('steps', [0, 1, 2, 17])
def test_quadratic_discrete_response_including_negative_base(steps):
    # eta=0.5 gives positive, unit, zero, negative, and unstable factors.
    eig = np.array([.1, 0., 2., 3., -1.])
    v = np.array([1., -2., .3, 4., -.2])
    response, supported = spectral_movement(eig, np.eye(5), v, steps, eta=.5)
    exact = ((1-.5*eig)**steps-1)*v
    np.testing.assert_allclose(response, exact, rtol=3e-14, atol=2e-14)
    direct, direct_supported = operator_movement(np.diag(eig), v, steps, eta=.5)
    np.testing.assert_allclose(response, direct, rtol=3e-14, atol=2e-14)
    assert supported and direct_supported


def test_tiny_eigenvalue_is_retained_and_overflow_is_unsupported():
    response, supported = spectral_movement(np.array([1e-25]), np.eye(1), np.ones(1), 20000)
    assert supported
    np.testing.assert_allclose(response, [-4e-24], rtol=1e-14, atol=0.)
    _, supported = spectral_movement(np.array([-1e6]), np.eye(1), np.ones(1), 20000)
    assert not supported


def test_nondiagonal_quadratic_eigenbasis():
    basis = np.array([[1., -1.], [1., 1.]])/np.sqrt(2.)
    eig = np.array([.2, -.1])
    direction = np.array([.3, -.7])
    hessian = basis@np.diag(eig)@basis.T
    predicted, supported = spectral_movement(eig, basis, direction, 25, eta=.1)
    actual, direct_supported = operator_movement(hessian, direction, 25, eta=.1)
    np.testing.assert_allclose(predicted, actual, rtol=5e-14, atol=2e-15)
    assert supported and direct_supported
