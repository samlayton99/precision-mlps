import numpy as np

from experiments.expD34_readout_race.population_frozen_witness import witness_bound


def test_arbitrary_witness_bound_holds_at_every_discrete_update():
    rng = np.random.default_rng(219)
    A = rng.normal(size=(17, 5))*.1
    residual = rng.normal(size=17)
    v = rng.normal(size=17)
    eta, N = .2, 25
    L = np.sum(A*A)
    bound = witness_bound(A, residual, v, eta, N, L)
    assert bound['status'] == 'fp64_evaluation'
    original = residual.copy()
    for _ in range(N+1):
        assert np.linalg.norm(residual)+1e-13 >= bound['lower_norm']
        assert abs(v@(residual-original)) <= bound['movement_allowance']+1e-13
        residual = residual-eta*A@(A.T@residual)


def test_exact_null_witness_proves_persistent_error():
    A = np.array([[1.], [0.]])
    residual = np.array([1., 2.])
    result = witness_bound(A, residual, np.array([0., 3.]), .1, 1000000, 1.)
    assert result['lower_norm'] == 2.
    assert result['movement_allowance'] == 0.


def test_witness_scaling_does_not_change_bound():
    A = np.diag([.2, .001])
    residual = np.array([.5, 1.])
    witness = np.array([0., 1.])
    first = witness_bound(A, residual, witness, .002, 100000, np.sum(A*A))
    second = witness_bound(A, residual, -7*witness, .002, 100000, np.sum(A*A))
    np.testing.assert_allclose(first['lower_norm'], second['lower_norm'])
    assert first['lower_norm'] > .9
