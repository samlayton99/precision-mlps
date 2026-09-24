"""Spectral propagation matches the same fixed-feature iterative GD recurrence."""
import numpy as np
import pytest

from experiments.expD34_readout_race import population_frozen_readout as frozen


@pytest.mark.parametrize('rank_deficient', [False, True])
@pytest.mark.parametrize('eta', [.05, .6])
def test_spectral_matches_iterative_tall_design(rank_deficient, eta):
    rng = np.random.default_rng(314)
    U, _ = np.linalg.qr(rng.normal(size=(9, 3)))
    V, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    singular = np.array([np.sqrt(3.), .3, 0. if rank_deficient else .02])
    design = (U*singular)@V.T
    target = rng.normal(size=9); initial = rng.normal(size=3)
    spectral = frozen.setup(design, target, initial)
    expected = initial.copy()
    for count in range(101):
        got, residual, _ = frozen.propagate(spectral, eta, count)
        np.testing.assert_allclose(got, expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(residual, design@got-target, rtol=2e-12, atol=2e-12)
        expected -= eta*design.T@(design@expected-target)
    if eta == .6:
        assert 1 < eta*spectral['singular'][0]**2 < 2


def test_tiny_singular_modes_are_not_truncated():
    singular = np.array([0., 1e-150, 1e-10])
    power, gain = frozen.spectral_factors(singular, .002, 100000)
    assert gain[0] == 0 and gain[1] > 0 and gain[2] > 0
    np.testing.assert_allclose(gain[1:], 200*singular[1:], rtol=1e-14, atol=0)
    np.testing.assert_allclose(power, np.ones(3), rtol=1e-14)
