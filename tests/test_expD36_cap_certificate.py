import numpy as np
import pytest

pytest.importorskip('flint')
from experiments.expD36_frozen_gamma_probe import cap_certificate as c
from experiments.expD36_frozen_gamma_probe import core


def test_exact_two_sample_certificate_and_jensen():
    x = np.array([-1., 1.]); y = x.copy(); centers = np.zeros(4); cap = .05
    beta = 4*np.tanh(cap)**2
    factor = np.ones((2, 1))*np.sqrt(beta/2)
    result = c.certify(x, centers, cap, y, factor, y, target_witness=True)
    assert result['beta'] >= beta
    assert result['beta'] < beta*(1+1e-12)
    assert c.time_bound([result])['bound'] == np.ceil(np.log(.01)/np.log1p(-.5*beta))


def test_repair_is_valid_for_interior_and_endpoint_maxima():
    rng = np.random.default_rng(8)
    x = np.linspace(-1, 1, 17); centers = np.array([-.8, .3, 1.2])
    y = np.sin(2*np.pi*x); v = rng.normal(size=len(x))
    factor = rng.normal(size=(len(x), 2))*.03
    result = c.certify(x, centers, 4., v, factor, y, max_intervals=12)
    v /= np.linalg.norm(v)
    b = np.ones(len(x))/np.sqrt(len(x))
    xx = factor@factor.T+result['bias_repair']*np.outer(b, b)
    for _ in range(100):
        j = core.design(x, centers, rng.uniform(-4, 4, len(centers)))
        assert np.linalg.norm(j.T@v)**2 <= np.trace(j.T@xx@j)+1e-12
        assert np.linalg.norm(j.T@v)**2 <= result['beta']*np.linalg.norm(j, 2)**2+1e-12


def test_small_cap_baseline_and_invalid_direct_jensen():
    x = np.linspace(-1, 1, 33); y = np.sin(2*np.pi*x)
    result = c.analytic_small_cap(x, 9, .05, y)
    assert 0 < result['beta'] < .01
    assert c.time_bound([result])['bound'] > 100
    with pytest.raises(ValueError, match='exactly the target'):
        c.certify(x, [0.], .1, x, np.zeros((33, 0)), y, target_witness=True)


@pytest.mark.parametrize('cap', [1e-4, .001, .1])
def test_convex_candidate_recovers_two_sample_optimum(cap):
    pytest.importorskip('cvxpy')
    x = np.array([-1., 1.]); y = x.copy()
    result = c.optimize_candidate(x, np.zeros(4), cap, y, y[:, None], grid_size=5)
    certified = c.certify(x, np.zeros(4), cap, y, result['factor'], y,
                          target_witness=True, relative_slack=1e-9)
    optimum = 4*np.tanh(cap)**2
    assert certified['beta'] >= optimum
    assert certified['beta'] < optimum*(1+1e-5)+1e-14


def test_joint_witness_optimization_recovers_exact_resolvent():
    pytest.importorskip('cvxpy')
    x = np.array([-1., 1.]); y = x.copy()
    optimum = 4*np.tanh(.1)**2
    result = c.optimize_joint_candidate(x, np.zeros(4), .1, y, y[:, None], .01, grid_size=5)
    assert result['resolvent_candidate'] == pytest.approx(1/(optimum+.01), rel=1e-5)
    certified = c.certify(x, np.zeros(4), .1, result['witness'], result['factor'], y,
                          relative_slack=1e-8)
    assert certified['beta'] >= optimum-1e-12
    assert certified['beta'] < optimum+1e-6


def test_reflection_reuse_does_not_assume_paired_slopes():
    x = np.linspace(-1, 1, 17); y = np.sin(2*np.pi*x)
    centers = np.array([-.75, 0., .75])
    factor = np.ones((len(x), 1))*.01
    result = c.certify(x, centers, 4., y, factor, y, max_intervals=16)
    assert result['columns'][-1]['reflection_reuse']
    assert result['columns'][-1]['upper'] == result['columns'][0]['upper']
    v = y/np.linalg.norm(y)
    for slopes in [[.1, 4., 2.], [4., -.01, .01], [0., 0., 4.]]:
        j = core.design(x, centers, np.array(slopes))
        assert np.linalg.norm(j.T@v)**2 <= result['beta']*np.linalg.norm(j, 2)**2+1e-12
