"""Full-complement tracking derivative includes the Jacobian curvature term."""
import jax
import jax.numpy as jnp
import numpy as np

from experiments.expD34_readout_race import adam_forces as af
from experiments.expD34_readout_race import mechanism_adam_feedback as feedback


def test_tracking_and_effective_derivatives_match_autodiff_and_finite_difference():
    rng = np.random.default_rng(49)
    p = rng.normal(size=22)*.4
    x = np.linspace(-1, 1, 67)
    y = np.sin(5*x)+.2*x+.1
    system = feedback.derivatives(p, x, y)
    def channels(point):
        g, _, jc, ec = af.field(point, jnp.asarray(x), jnp.asarray(y))
        return af.split(g, jc, ec)[0]
    derivative = np.asarray(jax.jacrev(channels)(jnp.asarray(p)))
    np.testing.assert_allclose(system['DQ'], derivative[1], atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(system['DF'], derivative[0], atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(system['H'], derivative.sum(axis=0), atol=2e-13, rtol=2e-12)
    direction = rng.normal(size=p.size); direction /= np.linalg.norm(direction)
    finite = (np.asarray(channels(jnp.asarray(p+1e-5*direction)))-
              np.asarray(channels(jnp.asarray(p-1e-5*direction))))/2e-5
    np.testing.assert_allclose(system['DQ']@direction, finite[1], atol=2e-10, rtol=2e-8)
    np.testing.assert_allclose(system['DF']@direction, finite[0], atol=2e-10, rtol=2e-8)
    # Dropping the derivative of JC^T would remove a real contribution.
    _, _, jc, _ = af.field(jnp.asarray(p), jnp.asarray(x), jnp.asarray(y))
    assert np.linalg.norm(system['DQ']-np.asarray(jc).T@system['Dz']) > 1e-3
