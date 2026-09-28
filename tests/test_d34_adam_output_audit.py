"""Numerical checks for the archive-only Adam output audit."""
import numpy as np

from experiments.expD34_readout_race import adam_output_audit as audit


def fixture():
    rng = np.random.default_rng(71)
    p = rng.normal(size=10)*.3
    x = np.linspace(-1, 1, 41)
    y = np.sin(3*x)
    case = dict(beta1=.9, beta2=.999, epsilon=1e-8, adaptive=True, eta=.002)
    return rng, p, x, y, case


def test_jacobian_against_directional_difference():
    rng, p, x, y, _ = fixture()
    r, J = audit.field(p, x, y)
    direction = rng.normal(size=p.shape)
    h = 1e-5
    rp, _ = audit.field(p+h*direction, x, y)
    rm, _ = audit.field(p-h*direction, x, y)
    np.testing.assert_allclose((rp-rm)/(2*h), J @ direction, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose((rp @ rp-rm @ rm)/(4*h), (J.T @ r) @ direction,
                               rtol=1e-7, atol=1e-10)


def test_spectral_energy_and_weak_direction():
    J = np.array([[3., 0.], [0., 1e-4], [0., 0.]])
    r = np.array([0., 2., 1.])
    summary, modes = audit.spectrum(J, r, .1)
    np.testing.assert_allclose(sum(row['residual_energy'] for row in modes), 5.)
    np.testing.assert_allclose(summary['unresolved_residual_fraction'], .2)
    np.testing.assert_allclose(summary['energy_fraction_eta_lambda_below_1_over_20000'], 1.)
    np.testing.assert_allclose(summary['spectral_gradient_energy'], 4e-8)
    np.testing.assert_allclose(summary['direct_gradient_energy'], 4e-8)


def test_actual_loss_and_momentum_output_identities():
    rng, p, x, y, case = fixture()
    m = rng.normal(size=p.shape)*.02
    v = np.full_like(p, .04)
    state, summary, _ = audit.audit_state(p, m, v, 100, case, x, y)
    assert state['loss_identity_error'] < 1e-15
    assert state['output_step_identity_error'] < 1e-15
    assert state['effective_coarse_defect'] < 1e-13
    assert state['adaptive_coarse_resolved']
    for row in summary:
        np.testing.assert_allclose(row['spectral_gradient_energy'], row['direct_gradient_energy'],
                                   rtol=1e-12, atol=1e-14)


def test_numpy_next_step_replays_training_implementation():
    import jax
    import jax.numpy as jnp
    from experiments.expD34_readout_race import adam_run, targets
    assert jax.config.x64_enabled, 'Run tests with JAX_ENABLE_X64=true'
    rng, p, x, y, case = fixture()
    state = adam_run.initial(jnp.asarray(p))
    state['m'] = jnp.asarray(rng.normal(size=p.shape)*.02)
    state['v'] = jnp.asarray(np.full_like(p, .04))
    state['count'] = jnp.asarray(100, dtype=jnp.int64)
    # Keep auxiliary-channel accounting consistent with the chosen momentum.
    state['channel_m'] = jnp.stack((state['m'], jnp.zeros_like(state['m']), jnp.zeros_like(state['m'])))
    q = np.polynomial.legendre.legvander(x, 9) @ targets.polynomial_map(x)
    settings = jnp.asarray([case[k] for k in ('eta', 'beta1', 'beta2', 'epsilon', 'adaptive')])
    new, _ = adam_run.one_step(state, jnp.asarray(x), jnp.asarray(y), jnp.asarray(q), settings)
    r, J = audit.field(p, x, y)
    delta, _, _ = audit.next_update(J.T @ r, np.asarray(state['m']), np.asarray(state['v']), 100, case)
    np.testing.assert_allclose(np.asarray(new['p']), p+delta, rtol=1e-13, atol=1e-14)


def test_gd_has_identity_mobility_and_no_lag():
    _, p, x, y, case = fixture()
    case.update(beta1=0., adaptive=False)
    r, J = audit.field(p, x, y)
    g = J.T @ r
    delta, P, mh = audit.next_update(g, np.ones_like(p), np.ones_like(p), 7, case)
    np.testing.assert_array_equal(P, np.ones_like(p))
    np.testing.assert_array_equal(mh, g)
    np.testing.assert_allclose(delta, -case['eta']*g)
