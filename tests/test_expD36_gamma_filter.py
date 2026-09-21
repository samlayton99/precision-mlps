"""Independent checks of the gamma filter and its finite-model transfer."""
import numpy as np
import pytest
from scipy.integrate import quad

from experiments.expD36_frozen_gamma_probe import core, gamma_filter as gf
from experiments.expD36_frozen_gamma_probe import common_slope_poly as p


def test_convolution_and_transform_normalization():
    gamma = 2.
    rho = lambda t: gamma/2/np.cosh(gamma*t)**2
    for t in [-1.3, 0., .4]:
        convolution = 2*quad(rho, -20, t, epsabs=1e-13)[0]-1
        assert convolution == pytest.approx(np.tanh(gamma*t), abs=2e-13)
    for omega in [0., 1., 7.]:
        actual = quad(lambda t: rho(t)*np.cos(omega*t), -20, 20, epsabs=1e-13)[0]
        assert actual == pytest.approx(float(gf.multiplier(gamma, omega)), abs=2e-13)
    assert np.isfinite(gf.multiplier(gamma, [0., 1e-16, 1e6])).all()


@pytest.mark.parametrize('gamma,harmonics', [(2., 16), (8., 64), (16., 128), (64., 256)])
def test_fixed_geometry_factorization_and_feature_remainder(gamma, harmonics):
    x = np.linspace(-1, 1, 37)
    centers = np.array([-1.2, -.37, .15, .15, 1.1])
    geometry = gf.geometry(x, centers, harmonics)
    f_before, c_before = geometry['f'].copy(), geometry['c'].copy()
    approximate, budget = gf.synthesize(geometry, gamma)
    odd = np.arange(1, 2*harmonics, 2)
    omega = np.pi*odd/8
    amplitudes = 4/(np.pi*odd)*gf.multiplier(gamma, omega)
    independent = np.sum(amplitudes*np.sin((x[:, None, None]-centers[None, :, None])*omega), axis=-1)
    np.testing.assert_allclose(approximate[:, 1:]*np.sqrt(len(x)), independent, atol=4e-15)
    exact = core.design(x, centers, gamma)
    np.testing.assert_array_equal(approximate[:, 0], exact[:, 0])
    assert np.linalg.norm(exact-approximate, 'fro') <= budget['synthesis_remainder']+3e-14
    gf.synthesize(geometry, gamma*1.2)
    np.testing.assert_array_equal(geometry['f'], f_before)
    np.testing.assert_array_equal(geometry['c'], c_before)


def test_distant_transition_bound_is_needed_and_decreases_with_period():
    # With many harmonics the remaining discrepancy is the other square-wave
    # transition, not the omitted Fourier tail.
    x, centers = np.array([-.9, .9]), np.array([-.7, .7])
    errors = []
    for half_period in [2., 4.]:
        g = gf.geometry(x, centers, 128, half_period)
        j, budget = gf.synthesize(g, 1.)
        error = np.max(np.abs(j[:, 1:]*np.sqrt(2)-np.tanh(x[:, None]-centers)))
        assert error > budget['tail']
        assert error <= budget['extension']+budget['tail']+1e-14
        errors.append(error)
    assert errors[1] < errors[0]/20


def test_filter_dynamics_transfer_to_direct_raw_gd():
    x = np.linspace(-1, 1, 31)
    centers = np.array([-1.1, -.6, -.1, .4, .9])
    y = np.column_stack((np.sin(2*x), np.cos(3*x)))/np.sqrt(len(x))
    j = core.design(x, centers, 3.)
    jt, budget = gf.synthesize(gf.geometry(x, centers, 32), 3.)
    eta = .4/max(np.linalg.norm(j, 2)**2, np.linalg.norm(jt, 2)**2)
    model = p.factor(jt, y, eta)
    bounds = p.defect(j, jt, model, eta, budget['synthesis_remainder'])
    theta = np.zeros((len(centers)+1, y.shape[1]))
    for step in range(201):
        if step in [0, 1, 10, 100, 200]:
            actual = np.linalg.norm(y-j@theta, axis=0)/np.linalg.norm(y, axis=0)
            for method in ['analytic', 'action', 'combined']:
                low, high = p.band(model, bounds, step, method)
                assert np.all(low <= actual+1e-12)
                assert np.all(high >= actual-1e-12)
        theta += eta*j.T@(y-j@theta)
    assert np.linalg.eigvalsh(jt@jt.T).min() >= -1e-13


def test_invalid_filter_domain():
    with pytest.raises(ValueError):
        gf.multiplier(0, [1.])
    with pytest.raises(ValueError):
        gf.geometry(np.array([-1., 1.]), np.array([2.]), 8, half_period=2.)
