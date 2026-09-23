"""Small independent checks of the persistence identity and causal arms."""
import jax
import jax.numpy as jnp
import numpy as np

from experiments.expD34_readout_race import mechanism_persistence_kernel as kernel

jax.config.update('jax_enable_x64', True)


def example():
    x = jnp.linspace(-1., 1., 31)
    p = jnp.asarray([.31, -.42, .23, .12, -.18, .21, .38, -.27, .44, .17])
    y = jnp.sin(2.4*x)+.3*x*x+.1
    return p, x, y


def test_gradient_projection_and_curvature_independently():
    p, x, y = example()
    s = kernel.decomposition(p, x, y)
    loss = lambda v: jnp.mean((kernel.output(v, x)-y)**2)/2
    np.testing.assert_allclose(s['g'], jax.grad(loss)(p), rtol=2e-14, atol=2e-15)
    j = jax.jacfwd(lambda v: kernel.output(v, x))(p)
    np.testing.assert_allclose(s['J'], j, atol=2e-15)
    projector = jnp.eye(len(p))-s['JC'].T@jnp.linalg.solve(s['gram'], s['JC'])
    np.testing.assert_allclose(s['F'], projector@s['g'], atol=2e-15)
    np.testing.assert_allclose(s['R'], s['JC'].T@s['z'], atol=2e-15)
    np.testing.assert_allclose(s['JC']@s['F'], 0., atol=2e-15)
    # Independent dense sample Hessians are used only in this ten-parameter test.
    second = jax.jacfwd(jax.jacrev(lambda v: kernel.output(v, x)))(p)
    contracted = jnp.einsum('i,kij,j->k', s['F'], second, s['F'])
    exact_c = jnp.mean(s['eH']*contracted)-s['balance']@(s['basis'].T@contracted/len(x))
    d = kernel.diagnostics(p, x, y)
    np.testing.assert_allclose(d['C'], exact_c, rtol=2e-13, atol=2e-17)
    np.testing.assert_allclose(d['C'], d['C_generated']-d['C_target'], atol=2e-17)
    np.testing.assert_allclose(d['identity_error'], 0., atol=2e-17)
    np.testing.assert_allclose(d['k'], d['k_jvp'], rtol=2e-12, atol=2e-14)


def test_six_arms_match_and_natural_arm_is_ordinary_gd():
    p, x, y = example()
    result = jax.jit(kernel.fork)(p, x, y)
    reference = result['reference']
    g0 = kernel.ordinary_gradient(p, x, y)
    for arm in kernel.ARMS:
        np.testing.assert_allclose(kernel.gradient(p, x, y, reference, arm), g0, atol=2e-15)
        f = kernel.arm_effective(p, x, y, reference, arm)
        np.testing.assert_allclose(f, reference['F0'], atol=2e-15)
    # Check exact baseline equality away from the fork as well.
    moved = p-.02*g0
    np.testing.assert_array_equal(kernel.gradient(moved, x, y, reference, kernel.ARMS[0]),
                                  kernel.ordinary_gradient(moved, x, y))
    moved_state = kernel.decomposition(moved, x, y)
    for arm in kernel.ARMS:
        np.testing.assert_allclose(moved_state['JC']@kernel.arm_effective(moved, x, y, reference, arm),
                                   0., atol=2e-15)


def test_initial_speed_identity_includes_tracking_for_every_arm():
    p, x, y = example()
    result = kernel.fork(p, x, y)
    diag, arms, reference = result['diagnostics'], result['arm_diagnostics'], result['reference']
    state = kernel.decomposition(p, x, y)
    for i, (kappa, nu) in enumerate(kernel.ARMS):
        predicted = -(nu*diag['D']+kappa*diag['C'])/diag['q_squared']
        np.testing.assert_allclose(arms['k_pure'][i], predicted, rtol=2e-12, atol=2e-14)
        direct = jax.jvp(lambda v: kernel.arm_effective(v, x, y, reference, (kappa, nu)),
                         (p,), (-state['g'],))[1]
        exact = state['F']@direct/diag['q_squared']
        np.testing.assert_allclose(arms['k_actual'][i], exact, rtol=2e-12, atol=2e-14)
        np.testing.assert_allclose(arms['k_actual'][i], arms['k_pure'][i]-arms['loaded'][i]/diag['q_squared'],
                                   rtol=2e-12, atol=2e-14)
    # This fixture deliberately has tracking; suppressing it must change the result.
    assert abs(float(diag['loaded'])) > 1e-8*float(diag['q_squared'])


def test_two_step_slope_contrast_and_step_scaling():
    p, x, y = example()
    result = kernel.fork(p, x, y)
    ref = result['reference']
    accel = result['arm_diagnostics']['slope_acceleration']
    w = (len(p)-1)//3
    g0 = kernel.ordinary_gradient(p, x, y)
    errors = []
    for eta in (.01, .005):
        first = p-eta*g0
        endpoints = jnp.stack([first-eta*kernel.gradient(first, x, y, ref, arm) for arm in kernel.ARMS])
        observed = (endpoints[:, :w]-endpoints[0, :w])/eta**2
        prediction = accel-accel[0]
        errors.append(float(jnp.linalg.norm(observed-prediction)))
    # The O(eta^2) contrast has an O(eta^3) remainder, hence first-order error
    # after normalization by eta^2. No future trajectory is fitted here.
    assert .3 < errors[1]/errors[0] < .7
    assert errors[1] < .05*float(jnp.linalg.norm(prediction))


def test_zero_force_is_explicitly_undefined_not_infinite():
    p, x, _ = example()
    y = kernel.output(p, x)
    result = kernel.fork(p, x, y)
    d = result['diagnostics']
    assert float(d['q_squared']) == 0.
    assert np.isnan(float(d['k']))
    assert np.isnan(float(d['k_pure']))
    assert np.isnan(np.asarray(result['arm_diagnostics']['k_actual'])).all()
    assert float(d['D']) == 0.
    assert float(d['C']) == 0.
    for arm in kernel.ARMS:
        np.testing.assert_array_equal(kernel.gradient(p, x, y, result['reference'], arm), jnp.zeros_like(p))
