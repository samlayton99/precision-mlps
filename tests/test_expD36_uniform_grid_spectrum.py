import numpy as np
from scipy.linalg import svdvals

from experiments.expD36_frozen_gamma_probe import core
from experiments.expD36_frozen_gamma_probe.uniform_grid_spectrum import (
    buffered_mass, bulk, construct, low_rank_gram, secular_count, to_bulk_basis,
)


def test_bulk_half_odd_blocks_and_normalization():
    n, q = 16, 4
    values, eigen, _, _ = bulk(n, q, 3.)
    expected = np.sort(np.sum(np.abs(eigen)**2, axis=0)/(n*q+1))
    actual = np.sort(svdvals(values/np.sqrt(n*q+1))**2)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-10)


def test_finite_boundary_remainder_is_nonvacuous_and_encloses_error():
    # Deliberately coarse expansion: test the analytical tail above roundoff.
    a = construct(32, 4, 3, 2., boundary=3, terms=4)
    exact = core.design(a['x'], a['centers'], a['gamma'])
    defect = np.linalg.norm(exact-a['approximate'], 2)
    assert 1e-8 < defect <= a['feature_error'] < .5
    np.testing.assert_allclose(a['approximate'][-1], exact[-1], atol=1e-14)
    np.testing.assert_allclose(a['approximate'][:, :4], exact[:, :4], atol=1e-14)


def test_raw_gram_signed_correction_and_secular_counts():
    a = construct(32, 2, 2, 5., boundary=2, terms=5)
    d, q, signed = low_rank_gram(a)
    gram = a['approximate'].T@a['approximate']
    # Q^* H Q in the fixed, explicit bulk basis.
    transformed = to_bulk_basis(to_bulk_basis(gram, a).conj().T, a).conj().T
    np.testing.assert_allclose(np.diag(d)+(q*signed)@q.conj().T,
                               transformed, atol=2e-13)
    actual = np.linalg.eigvalsh(gram)
    for cutoff in [.002, .02, .2, 2.]:
        count, pivot, rank = secular_count(d, q, signed, cutoff, drop=1e-12)
        assert count == np.sum(actual < cutoff)
        assert pivot > 0 and rank <= len(gram)


def test_neighbor_differences_preserve_raw_parameters():
    rng = np.random.default_rng(391)
    x = np.linspace(-1, 1, 61)
    centers = np.linspace(-1.1, 1.1, 17)
    phi = np.tanh(4*(x[:, None]-centers))
    w = rng.normal(size=17)
    cumulative = np.cumsum(w)
    np.testing.assert_allclose(phi@w,
        (phi[:, :-1]-phi[:, 1:])@cumulative[:-1]+phi[:, -1]*cumulative[-1], atol=1e-14)
    np.testing.assert_allclose(np.sum(w*w), np.sum(np.diff(np.r_[0., cumulative])**2))


def test_buffered_mass_with_repeated_eigenvalues_and_nullspace():
    rng = np.random.default_rng(927)
    eigenvalues = np.array([0, 0, .05, .05, .2, .2, .4, .4, .7, .7, 1., 1.])
    nontrivial = 0
    for _ in range(30):
        modes, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        original = (modes*eigenvalues)@modes.T
        perturbation = rng.normal(size=(12, 3))*10**rng.uniform(-4, -1)
        approximate = original+perturbation@perturbation.T
        target = rng.normal(size=12)
        target /= np.linalg.norm(target)
        rates, vectors = np.linalg.eigh(approximate)
        model = dict(floor=0., weights=np.abs(vectors.T@target)**2, rates=rates)
        cutoff = rng.uniform(.01, .95)
        buffer = cutoff*rng.uniform(.1, .8)
        error = np.linalg.norm(original-approximate, 2)
        lower, upper = buffered_mass(model, cutoff, buffer, error)
        true_mass = np.sum(np.abs(modes.T@target)[eigenvalues <= cutoff]**2)
        assert lower-1e-12 <= true_mass <= upper+1e-12
        nontrivial += int(lower > 0 and upper < 1)
    assert nontrivial >= 10  # Avoid a test that only checks [0, 1] bounds.


def test_finite_gamma_raw_readout_gd_matches_signed_gram_law():
    a = construct(16, 3, 2, 3., boundary=2, terms=5)
    design = a['approximate']
    diagonal, vectors, signed = low_rank_gram(a)
    gram_in_bulk_basis = np.diag(diagonal)+(vectors*signed)@vectors.conj().T
    rates, modes = np.linalg.eigh(gram_in_bulk_basis)
    target = np.sin(3*a['x'])+.3*np.cos(7*a['x'])
    target /= np.sqrt(len(target))
    step = .4/np.linalg.norm(design, 2)**2
    # Spectral solution of theta_{k+1}=(I-eta H)theta_k+eta J^T y.
    forcing = to_bulk_basis((design.T@target)[:, None], a)[:, 0]
    updates = 61
    gain = np.full_like(rates, updates*step)
    nonzero = np.abs(rates) > 1e-12
    gain[nonzero] = -np.expm1(updates*np.log1p(-step*rates[nonzero]))/rates[nonzero]
    spectral_theta = modes@(gain*(modes.conj().T@forcing))
    theta = np.zeros(design.shape[1])
    for _ in range(updates):
        theta -= step*design.T@(design@theta-target)
    np.testing.assert_allclose(to_bulk_basis(theta[:, None], a)[:, 0],
                               spectral_theta, atol=2e-13, rtol=2e-12)
