import numpy as np
import pytest
from experiments.expD36_frozen_gamma_probe import cap_campaign as c


def test_slope_families_obey_cap_and_selection_is_reproducible():
    for family in ['common', 'uniform', 'log_uniform', 'two_random', 'two_center', 'two_boundary', 'sparse_cap']:
        case = c.make_case(64, 4, family, 3)
        assert np.all((case['slopes'] >= 0)&(case['slopes'] <= 4))
        np.testing.assert_array_equal(case['slopes'], c.make_case(64, 4, family, 3)['slopes'])
    with pytest.raises(ValueError, match='admissible'):
        c.make_case(64, 4, 'common', 0, np.array([5.]))


def test_ordinary_gd_every_iterate_hits_and_resume():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    j = np.diag([1., .4])[None]
    y = np.array([0., 1.])[None, :, None]
    eta = np.array([.5])
    theta = jnp.zeros((1, 2, 1)); hits = jnp.full((1, 1, len(c.EPSILONS)), -1)
    chunk = c.make_gd_chunk(100)
    theta, count, hits = chunk(theta, jnp.array(0), hits, jnp.asarray(j), jnp.asarray(y), jnp.asarray(eta))
    theta, count, hits = chunk(theta, count, hits, jnp.asarray(j), jnp.asarray(y), jnp.asarray(eta))
    expected = np.ceil(np.log(c.EPSILONS)/np.log(.92)).astype(int)
    expected[expected >= 200] = -1
    np.testing.assert_array_equal(np.asarray(hits)[0, 0], expected)
    np.testing.assert_allclose(np.asarray(j@theta-y).ravel(), [0, -.92**200], atol=1e-15)
