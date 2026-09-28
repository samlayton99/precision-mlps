"""Regression checks for width budgeting and the full-run Adam schedule clock."""
import numpy as np
import pytest

pytest.importorskip('jax')
import jax.numpy as jnp
from experiments.expD36_frozen_gamma_probe import adam_feature_probe_prepare as prepare
from experiments.expD36_frozen_gamma_probe import adam_feature_probe_run as frozen
from experiments.expD36_frozen_gamma_probe import adam_joint_probe_run as joint
from experiments.expD36_frozen_gamma_probe import adam_feature_probe_collect as collect
from experiments.expD36_frozen_gamma_probe import adam_feature_probe_spectrum as spectrum


def test_width_budget_includes_halos():
    prepare.self_test()
    centers, info = prepare.geometry(512)
    assert centers.shape == (512,)
    assert info['interior_centers'] == 468
    assert info['halo_each_side'] == 22
    assert info['spacing'] == 2/467


def test_independent_adam_references():
    frozen.self_test()
    joint.self_test()


def test_joint_gd_matches_independent_parameter_updates():
    joint.self_test('gd')


def test_known_kernel_spectrum_and_target_energy():
    spectrum.self_test()


def test_merge_preserves_each_case_and_snapshot():
    collect.self_test()


@pytest.mark.parametrize('count', [0, 50000, 100000, 199999])
@pytest.mark.parametrize('optimizer', ['adam', 'gd'])
def test_full_horizon_clock_in_both_training_modes(count, optimizer):
    config = dict(horizon=200000, learning_rates=[.002], epsilon=1e-8,
                  schedules=['constant', 'cosine'], optimizer=optimizer)
    rng = np.random.default_rng(781)
    x = np.linspace(-1, 1, 13)
    y = np.sin(3*x)
    phi = rng.normal(size=(1, 13, 5))
    state = frozen.initial_state(phi, y, config)
    state = (*state[:-1], jnp.asarray(count, dtype=jnp.int64))
    updated, _ = frozen.make_chunk(phi, y, config, 1)(state)
    factor = .5*(1+np.cos(np.pi*count/config['horizon']))
    weights = np.asarray(updated[0])
    np.testing.assert_allclose(weights[0, :, 1], factor*weights[0, :, 0],
                               rtol=1e-12, atol=1e-17)
    initial = rng.normal(scale=.2, size=(1, 13))
    state, _, _ = joint.initial_state(initial, x, y, config)
    state = (*state[:-1], jnp.asarray(count, dtype=jnp.int64))
    updated, _ = joint.make_chunk(x, y, config, 1)(state)
    changes = np.asarray(updated[0])-initial[:, None, :]
    np.testing.assert_allclose(changes[0, 1], factor*changes[0, 0],
                               rtol=1e-10, atol=6e-17)
