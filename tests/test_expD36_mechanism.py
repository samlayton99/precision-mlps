import numpy as np

from experiments.expD36_frozen_gamma_probe import core
from experiments.expD36_frozen_gamma_probe.mechanism import (
    curvature_lower_bound, finite_error_floor, forced_slow_mass,
)


def test_geometric_curvature_lower_bound():
    x = np.linspace(-1, 1, 257)
    for n in [64, 128]:
        g = core.geometry(n)
        for gamma in [.1, 1., 4., 64.]:
            j = core.design(x, g.centers, gamma)
            actual = np.linalg.norm(j, 2)**2
            assert curvature_lower_bound(x, g.centers, gamma) <= actual*(1+1e-12)


def test_finite_time_polynomial_content_against_direct_gd():
    x = np.linspace(-1, 1, 129)
    g = core.geometry(64)
    y = core.target(x, 'sine_mix_2_6_10')/np.sqrt(len(x))
    p, _ = np.linalg.qr(np.polynomial.legendre.legvander(x, 8))
    q = np.eye(len(x))-p@p.T
    for gamma in [1., 4., 16.]:
        j = core.design(x, g.centers, gamma)
        eta = .5/np.linalg.norm(j, 2)**2
        access = np.linalg.norm(q@j, 2)**2
        tail = np.linalg.norm(q@y)/np.linalg.norm(y)
        theta = np.zeros(j.shape[1])
        for step in range(101):
            fitted = j@theta
            error = np.linalg.norm(fitted-y)/np.linalg.norm(y)
            assert np.linalg.norm(q@fitted)/np.linalg.norm(y) <= np.sqrt(step*eta*access)+1e-12
            prediction = finite_error_floor(np.array([tail]), np.log(np.array([access])), step, eta)
            assert prediction['error'] <= error+1e-12
            theta -= eta*j.T@(fitted-y)


def test_gamma_to_slow_mass_angle_bound_and_rank_count():
    rng = np.random.default_rng(724)
    for _ in range(20):
        u, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        values = np.geomspace(1e-6, 1, 12)
        j = u@np.diag(np.sqrt(values))
        direction = u[:, :4]@rng.normal(size=4)
        direction /= np.linalg.norm(direction)
        orthogonal = rng.normal(size=12)
        orthogonal -= direction*(direction@orthogonal)
        orthogonal /= np.linalg.norm(orthogonal)
        residual = np.sqrt(.6)*direction+np.sqrt(.4)*orthogonal
        delta = abs(direction@residual)
        access = np.linalg.norm(j.T@direction)**2
        for threshold in np.geomspace(1e-5, 1, 9):
            actual = np.sum((u[:, values <= threshold].T@residual)**2)
            lower = forced_slow_mass(delta, access/threshold)
            assert lower <= actual+1e-12
        # A rank-d polynomial-like approximation bounds the remaining eigenvalues.
        p, _ = np.linalg.qr(rng.normal(size=(12, 4)))
        tail_access = np.linalg.norm(j-p@(p.T@j), 2)**2
        assert values[-5] <= tail_access+1e-12
        fast = u[:, -4:]
        np.testing.assert_allclose(np.linalg.norm(j-fast@(fast.T@j), 2)**2, values[-5], rtol=1e-12)
    assert forced_slow_mass(.8, 0) == .8**2
    assert forced_slow_mass(.8, 2) == 0
    np.testing.assert_allclose(forced_slow_mass(1., .1), .9)
