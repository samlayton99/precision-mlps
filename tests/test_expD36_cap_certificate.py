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


@pytest.mark.parametrize('order', [2, 4, 6])
def test_repair_is_valid_for_interior_and_endpoint_maxima(order):
    rng = np.random.default_rng(8)
    x = np.linspace(-1, 1, 17); centers = np.array([-.8, .3, 1.2])
    y = np.sin(2*np.pi*x); v = rng.normal(size=len(x))
    factor = rng.normal(size=(len(x), 2))*.03
    result = c.certify(x, centers, 4., v, factor, y, max_intervals=12, taylor_order=order)
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


def test_overlap_search_retains_exact_two_sample_direction():
    pytest.importorskip('cvxpy')
    x = np.array([-1., 1.]); y = x.copy()
    optimum = 4*np.tanh(.1)**2
    result = c.optimize_overlap_candidate(x, np.zeros(4), .1, y, y[:, None], .01, grid_size=5)
    certified = c.certify(x, np.zeros(4), .1, result['witness'], result['factor'], y,
                          relative_slack=1e-8)
    assert certified['delta'] > 1-1e-8
    assert certified['beta'] >= optimum-1e-12
    assert certified['beta'] < optimum+1e-6


def test_long_horizons_are_ranked_and_certified_beyond_float_integer_range():
    from experiments.expD36_frozen_gamma_probe.cap_refine import candidate_score
    assert candidate_score(.2, 1e-19) > 2**53
    assert candidate_score(.2, 1e-19) > candidate_score(.2, 1e-8)
    x = np.array([-1., 1.]); cap = 1e-10
    beta = 4*np.tanh(cap)**2
    proof = c.certify(x, np.zeros(4), cap, x, np.ones((2, 1))*np.sqrt(beta/2), x,
                      target_witness=True)
    result = c.time_bound([proof])
    assert result['status'] == 'interval_certified'
    assert result['bound'] > 10**18
    assert result['bound'] == pytest.approx(np.log(.01)/np.log1p(-.5*beta), rel=1e-12)


def test_interval_tanh_polynomials_match_independent_derivatives():
    import mpmath as mp
    from flint import arb
    with mp.workdps(50):
        for g in [.03, .7, 2.]:
            d = -1.3
            coefficients = c.tanh_taylor_coefficients(arb(d), (arb(g)*arb(d)).tanh(), 7)
            for k, value in enumerate(coefficients):
                reference = mp.diff(lambda t:mp.tanh(mp.mpf(d)*t), mp.mpf(g), k)/mp.factorial(k)
                assert float(value) == pytest.approx(float(reference), rel=1e-12, abs=1e-14)


def test_certificate_dual_identifies_exact_fast_control():
    pytest.importorskip('cvxpy')
    from experiments.expD36_frozen_gamma_probe.cap_dual import slope_probabilities
    x = np.array([-1., 1.]); cap = .1
    proposal = c.optimize_candidate(x, np.zeros(4), cap, x, x[:, None], grid_size=5, include_dual=True)
    probabilities = slope_probabilities(proposal['dual_weights'], proposal['dual_bias'])
    grid = np.r_[proposal['grid'], 0.]
    assert np.all(probabilities >= 0)
    np.testing.assert_allclose(probabilities.sum(axis=0), 1.)
    np.testing.assert_allclose(grid[np.argmax(probabilities, axis=0)], cap)


def test_capacity_witness_checks_real_features_and_binary_coefficients():
    from experiments.expD36_frozen_gamma_probe.cap_capacity import certify_readout
    x = np.array([-1., 1.]); gamma = .1
    theta = np.array([0., 1/np.tanh(gamma)])
    proof = certify_readout(x, np.array([0.]), gamma, theta, x/np.sqrt(2))
    assert proof['relative_error_upper'] < 1e-12
    assert proof['status'] == 'interval_certified_capacity'
