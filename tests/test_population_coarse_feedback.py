"""Validate the intervention, inherited history, identities, and curvature."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest
from experiments.expD34_readout_race import population_adam as pa, population_coarse_feedback as cf


def example(adaptive=True):
    p=jnp.array([.2,.5,-.1,.3,-.2,.4,.6,.1,-.3,.1])
    x=jnp.linspace(-1,1,41);y=jnp.sin(4*x)+x/2
    config=jnp.array([.002,.9 if adaptive else 0.,.999,1e-8,adaptive])
    saved=pa.advance(pa.initial(p),x,y,config,31)
    return saved,x,y,config


@pytest.mark.parametrize('adaptive',[False,True])
def test_native_replay(adaptive):
    saved,x,y,c=example(adaptive)
    native=pa.advance(saved,x,y,c,100)
    state,_=cf.advance(cf.initialize(saved,x,y,c),x,y,c,steps=100)
    for k in ('p','m','v','count'):np.testing.assert_array_equal(state[k],native[k])
    d=cf.diagnostics(state,x,y,c)
    assert abs(float(d['A_closure']))<1e-12
    assert float(d['coarse_identity_error'])<1e-13


@pytest.mark.parametrize('arm',[1,2,3,4,5])
def test_policies_and_release(arm):
    saved,x,y,c=example();s=cf.initialize(saved,x,y,c)
    z=cf.proposal(s,x,y,c,arm);native=cf.proposal(s,x,y,c)
    for k in ('m','v','cm','count','tracking'):np.testing.assert_array_equal(z[k],native[k])
    np.testing.assert_allclose(jnp.linalg.norm(z['fine'][:-1]),z['target_norm'],rtol=2e-13)
    if arm in (2,3,5):assert float(jnp.linalg.norm(z['jc']@z['fine']))<1e-13
    pulse,_=cf.advance(s,x,y,c,arm,steps=100)
    released,_=cf.advance(pulse,x,y,c,steps=100)
    ref=pa.initial(pulse['p'],pulse['m'],pulse['v'],pulse['count'])
    ref=pa.advance(ref,x,y,c,100)
    np.testing.assert_array_equal(ref['p'],released['p'])
    assert abs(float(cf.diagnostics(released,x,y,c)['A_closure']))<1e-11


def test_gd_tracking_attenuation():
    saved,x,y,c=example(False);s=cf.initialize(saved,x,y,c)
    z=cf.proposal(s,x,y,c,tracking_factor=.1)
    np.testing.assert_allclose(z['delta'],-c[0]*(z['channels'][0]+.1*z['channels'][1]),atol=1e-15)
    s,_=cf.advance(s,x,y,c,tracking_factor=.1,steps=100)
    assert abs(float(cf.diagnostics(s,x,y,c)['A_closure']))<1e-12


def test_coarse_response_and_hessian():
    saved,x,y,c=example();s=cf.initialize(saved,x,y,c);z=cf.proposal(s,x,y,c)
    u=-c[0]*z['channels'][0]
    a=cf.isolated_response(s['p'],u,x,y);b=cf.isolated_response(s['p'],u/2,x,y)
    assert float(a['linear_coarse'])<1e-14
    assert float(a['identity_error'])<1e-13
    np.testing.assert_allclose(a['nonlinear_coarse']/b['nonlinear_coarse'],4.,rtol=.02)
    v=jnp.arange(1,len(s['p'])+1,dtype=jnp.float64)/10
    def loss(p):
        aa,bb,cc=p[:-1].reshape(3,-1)
        return .5*jnp.mean((jnp.tanh(x[:,None]*aa+bb)@cc+p[-1]-y)**2)
    expected=jax.jvp(jax.grad(loss),(s['p'],),(v,))[1]
    actual=cf.hessian_product(s['p'],v,jnp.ones_like(v),x,y)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-13)


def test_frozen_denominator_does_not_use_future_variance():
    saved,x,y,c=example();s=cf.initialize(saved,x,y,c)
    other=dict(s,v=s['v']*100)
    for arm in (4,5):
        np.testing.assert_array_equal(cf.proposal(s,x,y,c,arm)['fine'],cf.proposal(other,x,y,c,arm)['fine'])


def test_two_coordinate_model():
    k,dc,dcf,df,f,eta=3.,2.,.8,1.,.03,.1
    x,s=.2,.0;x0=x;star=-dcf*f/(k*dc);alpha=1-eta*k*dc
    for _ in range(50):x,s=x-eta*(k*dc*x+dcf*f),s-eta*(k*dcf*x+df*f)
    np.testing.assert_allclose(x-star,alpha**50*(x0-star),atol=1e-15)
    expected=-50*eta*(df-dcf**2/dc)*f-dcf/dc*(1-alpha**50)*(x0-star)
    np.testing.assert_allclose(s,expected,atol=1e-14)
    assert abs(1-.9*2)<1 and abs(1-1.1*2)>1
    # An uncoupled fine coordinate has the same motion regardless of coarse oscillation.
    for coarse in (.1,10.):assert -eta*(0.*coarse+f)==-eta*f


def test_sparse_gd_checkpoint_advance(tmp_path):
    from experiments.expD34_readout_race.population_coarse_feedback_run import load
    saved,x,y,c=example(False)
    path=tmp_path/'checkpoint.npz'
    np.savez(path,**{k:np.asarray(saved[k]) for k in ('p','m','v','cm','count')})
    spec=dict(path=str(path),member='',age=36,advance=5,eta=.002,beta1=0.,
              beta2=.999,epsilon=1e-8,adaptive=False)
    actual=load(spec,x,y);expected=pa.advance(saved,x,y,c,5)
    for k in ('p','m','v','cm','count'):np.testing.assert_array_equal(actual[k],expected[k])
