import numpy as np
import pytest

from experiments.expD36_frozen_gamma_probe import core, full_core as f
from experiments.expD36_frozen_gamma_probe.tighten import (
    log_derivative_tail, log_neighbor_envelope, spectral_tail_bound,
)


def test_neighbor_bound_covers_independent_projected_features_and_derivatives():
    x = np.linspace(-1, 1, 257)
    centers = np.array([-1.4, -.9, -.3, .01, .04, .65, 1.3])
    scales = np.array([.7, .2, .3, .5, .7, .9, 1.1, 1.4])
    degrees = np.array([0, 1, 4, 8, 16, 32])
    for gamma in [.5, 4, 16, 64]:
        a = core.design(x, centers, gamma)
        j = f.design_from_physical(a, scales, True)
        derivative = -gamma/np.cosh(gamma*(x[:, None]-centers))**2/np.sqrt(len(x))
        bounds = np.exp(log_neighbor_envelope(gamma, degrees, centers, scales))
        derivative_bounds = np.exp(log_derivative_tail(gamma, degrees))
        for k, bound, db in zip(degrees, bounds, derivative_bounds):
            # A separate explicit polynomial projector, not the analysis code.
            q, _ = np.linalg.qr(np.polynomial.legendre.legvander(x, k))
            tail = j-q@(q.T@j)
            dtail = derivative-q@(q.T@derivative)
            assert np.linalg.norm(tail, 'fro')**2 <= bound*(1+1e-12)+1e-27
            assert np.max(np.linalg.norm(dtail, axis=0)) <= db*(1+1e-12)+1e-13
        # Including more poles reduces the conservative remainder; it need not
        # agree to machine precision, especially at degree zero.
        assert np.all(log_derivative_tail(gamma, degrees, 256)
                      <= log_derivative_tail(gamma, degrees, 128)+1e-12)


def test_spectral_tail_bound_against_direct_gd_and_partial_spectrum():
    rng = np.random.default_rng(321)
    for _ in range(12):
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        eigenvalues = np.geomspace(.02, 1, 8)
        j = q@np.diag(np.sqrt(eigenvalues))
        y = rng.normal(size=8); weights = (q.T@y/np.linalg.norm(y))**2
        theta = np.zeros(8); eta = .5
        for epsilon in [.3, .05]:
            theta[:] = 0
            for step in range(2000):
                residual = j@theta-y
                if np.linalg.norm(residual)/np.linalg.norm(y) <= epsilon:
                    break
                theta -= eta*j.T@residual
            else:
                raise AssertionError('Direct GD did not reach test tolerance')
            for keep in [slice(None), slice(2, None)]:
                value = spectral_tail_bound(eigenvalues[keep], weights[keep], eta, epsilon)
                assert value['bound'] <= step
                # Jensen also bounds the complete trajectory, before the hit.
                if 'mass' in value:
                    lower = np.sqrt(value['mass'])*(1-eta*value['mean_eigenvalue'])**(step-1)
                    exact = np.sqrt(np.sum(weights*(1-eta*eigenvalues)**(2*(step-1))))
                    assert lower <= exact*(1+1e-12)


def test_spectral_tail_single_mode_nullspace_and_omitted_energy():
    for eigenvalue in [.001, .2, 1.]:
        expected = np.ceil(np.log(.01)/np.log1p(-.5*eigenvalue))
        assert spectral_tail_bound([eigenvalue], [1.], .5, .01)['bound'] == expected
    assert spectral_tail_bound([0, 1], [.2, .8], .5, .01)['status'] == 'exact_zero_mode_obstruction'
    assert spectral_tail_bound([1], [1e-5], .5, .01)['bound'] == 0
    with pytest.raises(ValueError):
        spectral_tail_bound([1], [1], 1, .01)
