"""Independent transfer-inequality and interpolation fixtures."""
import numpy as np
import pytest

from experiments.expD36_frozen_gamma_probe import core, common_slope_poly as p


def test_interpolation_normalization_and_analytic_remainder():
    x = np.linspace(-1, 1, 257)
    centers = np.array([-1.2, -.3, .4, 1.1])
    for gamma, degree in [(0, 12), (2, 32), (8, 64)]:
        approximate, coefficients, remainder = p.interpolate(x, centers, gamma, degree)
        exact = core.design(x, centers, gamma)
        np.testing.assert_array_equal(approximate[:, 0], exact[:, 0])
        assert np.linalg.norm(exact-approximate, 'fro') <= remainder+2e-14
        nodes = np.cos(np.pi*np.arange(degree+1)/degree)
        np.testing.assert_allclose(np.polynomial.chebyshev.chebval(nodes, coefficients).T,
            np.tanh(gamma*(nodes[:, None]-centers)), atol=2e-14)


@pytest.mark.parametrize('seed', [4, 9, 23])
def test_transfer_bounds_for_noncommuting_kernels(seed):
    rng = np.random.default_rng(seed)
    j = rng.normal(size=(9, 5))*.2
    approximate = j+rng.normal(size=j.shape)*.007
    y = rng.normal(size=(9, 3))
    eta = .3/max(np.linalg.norm(j, 2)**2, np.linalg.norm(approximate, 2)**2)
    model = p.factor(approximate, y, eta)
    bounds = p.defect(j, approximate, model, eta, np.linalg.norm(j-approximate, 'fro'))
    k, kt = j@j.T, approximate@approximate.T
    assert np.linalg.norm(k@kt-kt@k) > 1e-4
    for n in [0, 1, 7, 41, 203]:
        exact = np.linalg.norm(np.linalg.matrix_power(np.eye(9)-eta*k, n)@y, axis=0)/model['norm']
        for method in ['analytic', 'action', 'combined']:
            low, high = p.band(model, bounds, n, method)
            assert np.all(low <= exact+3e-13)
            assert np.all(high >= exact-3e-13)


def test_exact_scalar_hit_and_unreachable_component():
    j = np.array([[.2], [0.]])
    y = np.array([[1., .8], [0., .6]])
    model = p.factor(j, y, 12.5)
    bounds = p.defect(j, j, model, 12.5, 0.)
    expected = int(np.ceil(np.log(.01)/np.log(.5)))
    result = p.crossing_bracket(model, bounds, 0, cap=10**5)
    assert result['necessary'] == result['sufficient'] == expected
    other = p.crossing_bracket(model, bounds, 1, cap=10**5)
    assert other['sufficient'] is None
    assert other['necessary'] == 10**5+1
    np.testing.assert_allclose(p.error(model, 100), [0, .6], atol=1e-14)


def test_cross_couplings_change_acquisition_times():
    j = np.array([[1., 0.], [.9, .1]])
    y = np.array([1., -1.])
    eta = .4/np.linalg.norm(j, 2)**2
    coupled = p.factor(j, y, eta)
    diagonal = p.factor(np.diag(np.linalg.norm(j, axis=1)), y, eta)
    assert p.error(coupled, 100)[0] > .8
    assert p.error(diagonal, 100)[0] < 1e-6


def test_upper_search_keeps_witness_for_nonmonotone_envelope():
    j = np.array([[.2]])
    model = p.factor(j, np.ones(1), 1.)
    bounds = p.defect(j, j, model, 1., 1e-5)
    result = p.crossing_bracket(model, bounds, 0, cap=10**7, method='analytic')
    assert result['necessary'] <= np.ceil(np.log(.01)/np.log(.96)) <= result['sufficient']
    assert p.band(model, bounds, result['sufficient'], 'analytic')[1][0] <= .01
    assert p.band(model, bounds, 10**7, 'analytic')[1][0] > .01


def test_reject_changed_unstable_clock():
    with pytest.raises(ValueError, match='archived step'):
        p.factor(np.eye(2), np.ones(2), 1.01)


def test_interval_audit_matches_direct_small_system():
    pytest.importorskip('flint')
    from flint import arb, arb_mat, ctx
    from experiments.expD36_frozen_gamma_probe import common_slope_audit as audit
    with ctx.workprec(128):
        x = np.linspace(-1,1,17)
        centers = np.array([-.75,0.,.5])
        y = np.sin(x)/np.sqrt(len(x))
        gram, corr, norm = audit.finite_gram(x,centers,2,y)
        root = arb(len(x)).sqrt()
        j = arb_mat([[1/root]+[(2*(arb(float(t))-arb(float(c)))).tanh()/root
                              for c in centers] for t in x])
        direct = j.transpose()*j
        for i in range(gram.nrows()):
            for k in range(gram.ncols()):
                assert (gram[i,k]-direct[i,k]).contains(0)
        evolution = audit.augmented(gram,corr,.1)
        output_step = arb_mat([[arb(int(i==k)) for k in range(len(x))] for i in range(len(x))])
        output_step -= arb(.1)*(j*j.transpose())
        residual = output_step**37*arb_mat([[arb(float(t))] for t in y])
        direct_error = (residual.transpose()*residual)[0,0]/norm
        checked = audit.residual_squared(gram,corr,norm,evolution,37)
        assert (checked-direct_error).contains(0)
        assert float(checked.rad()) < 1e-28


def test_best_bracket_combines_only_same_gamma_and_method():
    from experiments.expD36_frozen_gamma_probe import common_slope_analysis as analysis
    rows = [dict(gamma=g,degree=d,results=dict(combined=[dict(necessary=lo,sufficient=hi)]))
            for g,d,lo,hi in [(8,32,10,None),(8,64,20,28),(8,128,19,26),(16,64,100,101)]]
    assert analysis.best(rows,8) == dict(necessary=20,sufficient=26,lower_degree=64,upper_degree=128)
