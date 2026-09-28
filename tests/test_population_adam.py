"""Passive Adam instrumentation and exact discrete population accounting."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.expD34_readout_race import population_adam as pa, adam_forces as af


def example():
    p=jnp.array([.2,.5,-.1, .3,-.2,.4, .6,.1,-.3, .1])
    x=jnp.linspace(-1,1,41)
    return p,x,jnp.sin(4*x)+x/2


@pytest.mark.parametrize('adaptive,beta1',[(True,.9),(True,0.),(False,0.)])
def test_native_update_and_ledgers(adaptive,beta1):
    p,x,y=example();state=pa.initial(p)
    settings=jnp.array([.002,beta1,.999,1e-8,adaptive])
    reference=p;m=jnp.zeros_like(p);v=m
    step=jax.jit(pa.step)
    for n in range(1,51):
        g=af.field(reference,x,y)[0]
        m=beta1*m+(1-beta1)*g;v=.999*v+.001*g*g
        inv=1/(jnp.sqrt(v/(1-.999**n))+1e-8) if adaptive else 1.
        reference=reference-.002*inv*m/(1-beta1**n)
        state=step(state,x,y,settings)
    np.testing.assert_allclose(state['p'],reference,rtol=2e-12,atol=2e-13)
    before,after=pa.population(p),pa.population(state['p'])
    ledger=dict(zip(pa.LEDGER,np.asarray(state['ledger'])))
    for q,actual in [('M',after['M']-before['M']),('A',after['A']-before['A']),
                     ('logC6',jnp.log(after['C6']/before['C6']))]:
        expected=sum(ledger[q+'_'+c] for c in pa.COMPONENTS)+ledger[q+'_defect']
        assert float(actual)==pytest.approx(expected,abs=2e-13)
    assert float(jnp.max(state['identity']))<2e-14


def test_components_match_independent_autodiff():
    p,x,y=example()
    def output(z):
        a,b,c=z[:-1].reshape(3,-1)
        return jnp.tanh(x[:,None]*a+b)@c+z[-1]
    basis=jnp.stack((jnp.ones_like(x),x/jnp.sqrt(jnp.mean(x*x))),axis=1)
    fine=lambda z:z-basis@(basis.T@z/len(x))
    g,_,_,parts,_=pa.field(p,x,y)
    np.testing.assert_allclose(parts[0],jax.grad(lambda z:.5*jnp.mean(fine(output(z))**2))(p),atol=2e-15)
    np.testing.assert_allclose(parts[1],jax.grad(lambda z:-jnp.mean(output(z)*fine(y)))(p),atol=2e-15)
    np.testing.assert_allclose(g,jax.grad(lambda z:.5*jnp.mean((output(z)-y)**2))(p),atol=2e-15)
    np.testing.assert_allclose(parts.sum(axis=0),g,atol=2e-15)


def test_restore_inherited_momentum_without_reset():
    p,x,y=example();settings=jnp.array([.002,.9,.999,1e-8,True])
    old=pa.advance(pa.initial(p),x,y,settings,31)
    cm=old['cm']
    original=jnp.stack((cm[0]+cm[1]+cm[2]+cm[5],cm[3],cm[4]))
    restored=pa.initial(old['p'],old['m'],old['v'],old['count'],original)
    end=pa.advance(restored,x,y,settings,100)
    direct=pa.advance(old,x,y,settings,100)
    np.testing.assert_array_equal(end['p'],direct['p'])
    np.testing.assert_array_equal(end['v'],direct['v'])
    np.testing.assert_allclose(end['cm'].sum(axis=0),end['m'],atol=3e-16)
    np.testing.assert_allclose(end['cm'][5],.9**100*original[0],rtol=2e-14,atol=1e-20)


def test_population_score_scale_and_output_identity():
    p,x,y=example();settings=jnp.array([.002,.9,.999,1e-8,True])
    np.testing.assert_allclose(pa.concentration_score(p),jax.grad(lambda z:jnp.log(pa.population(z)['C6']))(p),atol=2e-14)
    assert float(pa.population(10*p)['C6'])==pytest.approx(float(pa.population(p)['C6']),rel=1e-14)
    d=pa.diagnostics(pa.initial(p),x,y,settings)
    assert float(d['actual_loss_change'])==pytest.approx(float(d['linear_loss_change']+d['quadratic_loss_change']+d['nonlinear_loss_change']),abs=1e-16)
    assert 0<=float(d['balanced_adaptive_access'])<=float(d['adaptive_access'])*(1+1e-12)
