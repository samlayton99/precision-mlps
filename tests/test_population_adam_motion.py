"""Verify native restoration, diagnostic identities, and fine-only interventions."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.expD34_readout_race import population_adam as pa, population_adam_motion as motion


def example():
    p=jnp.array([.2,.5,-.1,.3,-.2,.4,.6,.1,-.3,.1])
    x=jnp.linspace(-1,1,41); y=jnp.sin(4*x)+x/2
    config=jnp.array([.002,.9,.999,1e-8,True])
    saved=pa.advance(pa.initial(p),x,y,config,31)
    return saved,x,y,config


def test_native_and_passive_instrumentation():
    saved,x,y,config=example()
    native=pa.advance(saved,x,y,config,1000)
    observed=motion.advance(motion.initialize(saved,x,y,config),x,y,config,0,1000)
    for k in ('p','m','v','cm','count'):
        np.testing.assert_array_equal(observed[k],native[k])
    d=motion.diagnostics(observed,x,y,config)
    for k in ('A_closure','bias_closure','readout_closure'):
        assert abs(float(d[k]))<1e-11
    assert float(d['window_count_100'])==10
    assert float(d['window_count_1000'])==1
    for k in ('slope_coherence_100','hidden_coherence_1000','energy_overlap'):
        assert 0<=float(d[k])<=1+1e-13


@pytest.mark.parametrize('arm',[1,2,3])
def test_only_fine_direction_changes_and_release(arm):
    saved,x,y,config=example(); state=motion.initialize(saved,x,y,config)
    z=motion.proposals(state,x,y,config,state['fixed_inverse'],arm)
    native=motion.proposals(state,x,y,config,state['fixed_inverse'],0)
    np.testing.assert_allclose(np.linalg.norm(z['fine'][:-1]),np.linalg.norm(native['fine'][:-1]),rtol=3e-14)
    np.testing.assert_array_equal(z['components'][3],native['components'][3])
    np.testing.assert_array_equal(z['components'][:,-1],native['components'][:,-1])
    for k in ('m','v','cm'):np.testing.assert_array_equal(z[k],native[k])
    pulse=motion.advance(state,x,y,config,arm,100)
    released=motion.advance(pulse,x,y,config,0,100)
    # Native recurrence from the altered state, retaining all historical buffers.
    reference=pa.initial(pulse['p'],pulse['m'],pulse['v'],pulse['count'])
    reference['cm']=pulse['cm']
    reference=pa.advance(reference,x,y,config,100)
    np.testing.assert_array_equal(released['p'],reference['p'])
    d=motion.diagnostics(released,x,y,config)
    assert abs(float(d['A_closure']))<1e-11
    assert float(d['norm_error'])<1e-14
    assert float(d['zero_candidate'])==0


def test_crossed_geometry_residual_identity():
    saved,x,y,config=example(); old=motion.initialize(saved,x,y,config)
    new=motion.advance(old,x,y,config,0,100)
    d=motion.crossed_forces(old,new,x,y,config)
    assert bool(d['crossed_resolved'])
    assert float(d['crossed_closure'])<1e-14
    assert float(d['current_force_error'])<1e-13
    np.testing.assert_allclose(d['geometry_A_rate_change']+d['residual_A_rate_change'],d['total_A_rate_change'],atol=1e-15)
    a=motion.crossed_forces(old,old,x,y,config)
    assert float(a['geometry_force_norm'])==0
    assert float(a['residual_force_norm'])==0
