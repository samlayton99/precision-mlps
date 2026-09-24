"""Signed force feedback identities and exact archived-state reconstruction."""
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.expD34_readout_race import mechanism_persistence_kernel as kernel
from experiments.expD34_readout_race.population_reinforcement_audit import load_state, observables

jax.config.update('jax_enable_x64', True)


def fixture():
    p = np.random.default_rng(730).normal(size=22)*.3
    x = np.linspace(-1., 1., 61)
    return p, x, np.sin(4*x)+.2*x*x


def test_exact_signed_identity_and_generated_target_split():
    p, x, y = map(jnp.asarray, fixture())
    row = observables(p, x, y)
    assert row['resolved']
    np.testing.assert_allclose(row['half_norm_squared_dot'], row['half_norm_squared_dot_jvp'], rtol=2e-10, atol=1e-14)
    np.testing.assert_allclose(row['log_force_rate'], row['residual_relaxation_rate']+row['generated_rate']+row['target_rate'], rtol=2e-10, atol=1e-13)
    assert row['residual_relaxation_rate'] <= 0
    assert row['log_force_rate'] <= row['directional_curvature_rate_bound']+1e-12
    assert row['directional_curvature_rate_bound'] <= row['structural_curvature_rate_bound']+1e-12
    assert row['curvature_split_absolute_error'] < 1e-13


def test_log_force_rate_matches_finite_difference_of_current_field():
    p, x, y = map(jnp.asarray, fixture())
    F = kernel.effective(p, x, y)
    dt = 1e-4
    after = jnp.log(jnp.linalg.norm(kernel.effective(p-dt*F, x, y)))
    before = jnp.log(jnp.linalg.norm(kernel.effective(p+dt*F, x, y)))
    finite_difference = (after-before)/(2*dt)
    np.testing.assert_allclose(observables(p, x, y)['log_force_rate'], finite_difference, rtol=2e-6, atol=1e-9)


def test_stationary_force_has_undefined_log_rate():
    p, x, _ = map(jnp.asarray, fixture())
    row = observables(p, x, kernel.output(p, x))
    assert row['F_norm'] == 0
    assert np.isnan(row['log_force_rate'])


def write_input(path, p, x, ys):
    np.savez(path, p=np.stack([p]*len(ys)), x=x, y=np.stack(ys),
             cases=np.array(json.dumps([dict(target=f'test{i}') for i in range(len(ys))])))


def state_row(source, p, x, y, role, index):
    return dict(source=str(source), index=str(index), role=role,
                state_sha256=hashlib.sha256(p.tobytes()+x.tobytes()+y.tobytes()).hexdigest())


def test_static_loader_verifies_state_and_target_hash(tmp_path):
    p, x, y = fixture()
    source = tmp_path/'inputs.npz'
    write_input(source, p, x, [y])
    row = state_row(source, p, x, y, 'static', 0)
    actual = load_state(row)
    for observed, expected in zip(actual[:3], (p, x, y)):
        np.testing.assert_array_equal(observed, expected)
    with pytest.raises(ValueError, match='hash'):
        load_state(dict(row, state_sha256='wrong'))


def test_dilation_loader_uses_manifest_input_index(tmp_path):
    p, x, y = fixture()
    source = tmp_path/'inputs.npz'
    write_input(source, p, x, [y, -y])
    run = tmp_path/'dilation'; run.mkdir()
    (run/'manifest.json').write_text(json.dumps(dict(input=str(source), cases=[dict(input_index=1)])))
    snapshot = run/'000000100.npz'
    np.savez(snapshot, p=p[None, :], failed=np.array([False]))
    actual = load_state(state_row(snapshot, p, x, -y, 'trajectory', 0))
    np.testing.assert_array_equal(actual[2], -y)


def test_natural_loader_uses_existing_prediction_layout(tmp_path):
    p, x, y = fixture()
    prediction = tmp_path/'feedback_example'; prediction.mkdir()
    write_input(prediction/'inputs.npz', p, x, [y])
    snapshots = tmp_path/'feedback_example_run'/'snapshots'; snapshots.mkdir(parents=True)
    snapshot = snapshots/'000000100.npz'
    np.savez(snapshot, p=p[None, :], failed=np.array([False]))
    actual = load_state(state_row(snapshot, p, x, y, 'trajectory', 0))
    np.testing.assert_array_equal(actual[2], y)
