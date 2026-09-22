import jax
import jax.numpy as jnp
import numpy as np
from experiments.expD34_readout_race import adam_run as ar, plateau_dense as dense, targets


def test_dense_observation_preserves_updates():
    x=targets.grid(64); modes=np.polynomial.legendre.legvander(x,9) @ targets.polynomial_map(x)
    p=np.r_[np.random.default_rng(1).normal(size=21)*.2,.1]
    state=jax.vmap(ar.initial)(jnp.array(p[None])); y=jnp.array(np.sin(5*x)[None])
    for adaptive in (0.,1.):
        settings=jnp.array([[.002,.9 if adaptive else 0.,.999,1e-8,adaptive]])
        final,(v,rows)=dense.kernel(jnp.array(x),jnp.array(modes),8)(state,y,settings)
        reference,*_=ar.advance_factory(64)(state,y,settings,8)
        for key in reference:
            np.testing.assert_allclose(final[key],reference[key],rtol=2e-12,atol=2e-13)
        for k,name in enumerate(('raw_effective_norm','raw_tracking_norm','step_effective_norm','step_tracking_norm')):
            np.testing.assert_allclose(np.linalg.norm(v[:,:,k],axis=-1),rows[:,:,ar.METRICS.index(name)],rtol=1e-11,atol=1e-13)
