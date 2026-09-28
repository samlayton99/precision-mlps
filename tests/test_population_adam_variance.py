import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from experiments.expD34_readout_race import population_adam as pa, population_adam_motion as motion, population_adam_variance as variance


def test_shadow_variance_and_matched_gain():
    p=jnp.array([.2,.5,-.1,.3,-.2,.4,.6,.1,-.3,.1]);x=jnp.linspace(-1,1,41);y=jnp.sin(4*x)+x/2
    config=jnp.array([.002,.9,.999,1e-8,True]);saved=pa.advance(pa.initial(p),x,y,config,31)
    start=variance.initialize(saved,x,y,config)
    state=variance.advance(start,x,y,config,0,100)
    native=pa.advance(saved,x,y,config,100)
    np.testing.assert_array_equal(state['p'],native['p'])
    z0,_,_,_=variance.proposal(state,x,y,config,0)
    candidates=[]
    for arm in (1,2):
        z,alternative,raw,gain=variance.proposal(state,x,y,config,arm)
        expected=.999*state['alternative_v']+.001*(z0['gradient']-.9*z0['raw_tracking'])**2
        np.testing.assert_allclose(alternative,expected,rtol=1e-14)
        np.testing.assert_array_equal(z['components'][3],z0['components'][3])
        np.testing.assert_array_equal(z['components'][:,-1],z0['components'][:,-1])
        np.testing.assert_allclose(np.linalg.norm(z['fine'][:-1]),gain*np.linalg.norm(z0['fine'][:-1]),rtol=1e-13)
        assert float(gain)<=10
        candidates.append(np.linalg.norm(z['fine'][:-1]))
        pulse=variance.advance(state,x,y,config,arm,100)
        release=variance.advance(pulse,x,y,config,0,100)
        reference=pa.initial(pulse['p'],pulse['m'],pulse['v'],pulse['count']);reference['cm']=pulse['cm']
        reference=pa.advance(reference,x,y,config,100)
        np.testing.assert_array_equal(release['p'],reference['p'])
        d=variance.diagnostics(release,x,y,config)
        assert abs(float(d['A_closure']))<1e-11
    np.testing.assert_allclose(candidates[0],candidates[1],rtol=1e-13)
