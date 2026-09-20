import os
import numpy as np
import pytest
pytest.importorskip('optimistix')
import jax.numpy as jnp
from experiments.expD06_fixed_center_scales import higher_order as higher
from experiments.expD36_ssb_geometry_switching.memory import update


def test_common_secant_update_is_covariant_and_satisfies_secant():
    source=os.environ.get('SSB_SOURCE')
    if not source:pytest.skip('Pinned SSBroyden required')
    solver=higher.ssb_solver(source,1e-30,1e-15,'accepted_step')
    rng=np.random.default_rng(901);a=rng.normal(size=(7,7));curvature=a.T@a+np.eye(7)
    diagonal=np.geomspace(.01,100,7);h=np.eye(7);transformed=h/diagonal[:,None]/diagonal
    for k in range(12):
        s=rng.normal(size=7);y=curvature@s
        h,ok=update(solver,jnp.asarray(h),jnp.asarray(s),jnp.asarray(y),jnp.array(k==0))
        transformed,ok2=update(solver,jnp.asarray(transformed),jnp.asarray(s/diagonal),jnp.asarray(y*diagonal),jnp.array(k==0))
        assert bool(ok)&bool(ok2)
        np.testing.assert_allclose(np.asarray(h)@y,s,rtol=2e-10,atol=2e-10)
        np.testing.assert_allclose(transformed,np.asarray(h)/diagonal[:,None]/diagonal,rtol=2e-8,atol=2e-9)
        assert np.linalg.eigvalsh(h).min()>0
