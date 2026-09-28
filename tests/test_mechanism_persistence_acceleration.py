import jax
import jax.numpy as jnp
import numpy as np

from experiments.expD34_readout_race.mechanism_persistence_acceleration import directional_terms, two_step_acceleration

jax.config.update('jax_enable_x64', True)


def test_quartic_hessian_drift_and_two_step_limit():
    point, direction = jnp.array([.4]), jnp.array([.3])
    field = lambda p: p**3
    g, hv, h2, drift = directional_terms(field, point, direction)
    np.testing.assert_allclose(h2, 9*point**4*direction, rtol=1e-14)
    np.testing.assert_allclose(drift, 6*point**4*direction, rtol=1e-14)
    errors = []
    for eta in (.002, .001, .0005):
        direct, stable = two_step_acceleration(field, point, direction, eta)
        # Independent differentiation of the actual composed two-update map.
        update = lambda p: p-eta*field(p)
        tangent = jax.jvp(lambda p: update(update(p)), (point,), (direction,))[1]
        np.testing.assert_allclose(direct, (tangent-direction+2*eta*hv)/eta**2, atol=2e-10, rtol=0)
        np.testing.assert_allclose(direct, stable, atol=2e-10, rtol=0)
        errors.append(float(jnp.linalg.norm(stable-h2-drift)))
    assert 1.9 < errors[0]/errors[1] < 2.1
    assert 1.9 < errors[1]/errors[2] < 2.1
