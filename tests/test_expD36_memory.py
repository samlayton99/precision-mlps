import os
import numpy as np
import pytest
pytest.importorskip('optimistix')
import jax.numpy as jnp
from experiments.expD06_fixed_center_scales import higher_order as higher
from experiments.expD36_ssb_geometry_switching.memory import update


def test_replay_first_secant_matches_the_pinned_priming_convention():
    import jax
    from experiments.expD35_optimization_exploration import core,run,ssb
    source=os.environ.get('SSB_SOURCE')
    if not source:pytest.skip('Pinned SSBroyden required')
    config=run.case(n=64,optimizer='ssbroyden');g,loss,physical=ssb.problem(config)
    solver=higher.ssb_solver(source,1e-30,1e-15,'accepted_step')
    initial=higher.ssb_initial(solver,loss,core.initialize(config)['z'])
    advanced,_=higher.ssb_step(solver,loss,physical,1e-30)(initial)
    s=advanced['z']-initial['z'];y=jax.grad(loss)(advanced['z'])-jax.grad(loss)(initial['z'])
    replayed,valid=update(solver,jnp.eye(len(s)),s,y,jnp.array(False))
    assert bool(valid)
    np.testing.assert_allclose(replayed,advanced['solver'].f_info.hessian_inv.pytree,rtol=2e-10,atol=2e-10)


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


def test_unobserved_invariant_block_keeps_relative_prior_scales():
    source=os.environ.get('SSB_SOURCE')
    if not source:pytest.skip('Pinned SSBroyden required')
    solver=higher.ssb_solver(source,1e-30,1e-15,'accepted_step')
    rng=np.random.default_rng(105);h=jnp.diag(jnp.array([1.,2.,3.,.01,.1,10.,100.]))
    reference=np.asarray(h)[3:,3:]/float(h[3,3])
    c=np.array([[3.,.2,.1],[.2,4.,.3],[.1,.3,2.]])
    for k in range(8):
        observed=rng.normal(size=3);s=jnp.r_[observed,jnp.zeros(4)];y=jnp.r_[c@observed,jnp.zeros(4)]
        h,ok=update(solver,h,s,y,jnp.array(k==0));assert bool(ok)
        np.testing.assert_allclose(np.asarray(h)[3:,3:]/float(h[3,3]),reference,rtol=2e-12,atol=0.)
        np.testing.assert_array_equal(np.asarray(h)[:3,3:],np.zeros((3,4)))
