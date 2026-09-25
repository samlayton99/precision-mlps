"""Independent aggregate identities, signed balances, and native-step closure."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.expD34_readout_race import population_energy_dynamics as ed
from experiments.expD34_readout_race import mechanism_persistence_kernel as kernel


@pytest.mark.parametrize('k',[4,6,10,14])
def test_concentration_score_and_aggregate_norm(k):
    p = jnp.array([.2,.5,-.1, .3,-.2,.4, .6,.1,-.3, .1])
    analytic = ed.score(p,k)
    automatic = jax.grad(lambda z:jnp.log(ed.shape(z,k)))(p)
    np.testing.assert_allclose(analytic,automatic,rtol=2e-13,atol=2e-13)
    m = p[:-1]@p[:-1]
    expected = k*k/m*(ed.shape(p,2*k-2)/ed.shape(p,k)**2-1)
    assert float(analytic@analytic) == pytest.approx(float(expected),rel=2e-13)
    assert abs(float(analytic@p)) < 3e-14


@pytest.mark.parametrize('kind',['gd','effective'])
def test_signed_split_and_derivative(kind):
    x=jnp.linspace(-1,1,41); y=jnp.sin(3*x)+x/2
    p=jnp.array([.2,.5,-.1, .3,-.2,.4, .6,.1,-.3, .1])
    state,parts=ed.velocities(p,x,y,kind)
    v=jnp.sum(parts,axis=0)
    np.testing.assert_allclose(v,-state['g' if kind=='gd' else 'F'],atol=2e-15)
    direct=jax.jvp(lambda z:jnp.log(ed.shape(z,6)),(p,),(v,))[1]
    assert float(direct) == pytest.approx(float(ed.score(p,6)@v),rel=2e-13)
    eps=1e-5
    finite=(jnp.log(ed.shape(p+eps*v,6))-jnp.log(ed.shape(p-eps*v,6)))/(2*eps)
    assert float(direct) == pytest.approx(float(finite),rel=2e-8,abs=2e-10)


def test_native_ledger_closes_concentration_mass_and_slopes():
    x=jnp.linspace(-1,1,31); y=jnp.sin(2*x)
    p=jnp.array([.2,.5,-.1, .3,-.2,.4, .6,.1,-.3, .1])
    end,ledger=ed.advance(p,jnp.zeros(len(ed.LEDGER)),x,y,.02,20,'gd')
    for i,k in enumerate(ed.ORDERS):
        measured=jnp.log(ed.shape(end,k)/ed.shape(p,k))
        predicted=jnp.sum(ledger[4*i:4*i+4])+ledger[-6+i]
        assert float(measured) == pytest.approx(float(predicted),abs=3e-14)
    assert float(end[:-1]@end[:-1]-p[:-1]@p[:-1]) == pytest.approx(float(jnp.sum(ledger[16:20])+ledger[-2]),abs=3e-14)
    assert float(end[:3]@end[:3]-p[:3]@p[:3]) == pytest.approx(float(jnp.sum(ledger[20:24])+ledger[-1]),abs=3e-14)


def test_equal_energy_has_zero_rate_but_variance_drives_acceleration():
    p=jnp.array([1.,1.,1., 0.,0.,0., 0.,0.,0., 0.])
    v=jnp.array([.1,.2,-.1, 0.,0.,0., 0.,0.,0., 0.])
    rate=lambda z:jax.jvp(lambda u:jnp.log(ed.shape(u,6)),(z,),(v,))[1]
    assert abs(float(rate(p))) < 1e-14
    second=jax.jvp(rate,(p,),(v,))[1]
    assert float(second) == pytest.approx(float(6*jnp.var(2*v[:3])),rel=2e-13)


def test_weighted_cubic_contrast_explains_relative_growth_loading():
    x=jnp.linspace(-1,1,41)
    p=jnp.array([.2,.5,-.1, .3,-.2,.4, .6,.1,-.3, .1])
    basis=kernel._basis(x)
    def cubic(z):
        a,b,c=z[:-1].reshape(3,-1)
        return kernel._fine(-((x[:,None]*a+b)**3)@c/3,basis)
    a,b,c=p[:-1].reshape(3,-1)
    e=a*a+b*b+c*c
    individual=kernel._fine(-c*(x[:,None]*a+b)**3/3,basis)
    contrast=24*(individual@(e*e)/jnp.sum(e**3)-cubic(p)/jnp.sum(e))
    exact=jax.jvp(cubic,(p,),(ed.score(p,6),))[1]
    np.testing.assert_allclose(exact,contrast,rtol=2e-13,atol=2e-14)


def test_joint_energy_concentration_gradient_has_orthogonal_budgets():
    p=jnp.array([.2,.5,-.1, .3,-.2,.4, .6,.1,-.3, .1])
    grad=jax.grad(lambda z:jnp.sum(z[:-1]**2)*jnp.sqrt(ed.shape(z,6)))(p)
    m=jnp.sum(p[:-1]**2);chi=ed.shape(p,6);k=ed.shape(p,10)/chi**2
    assert float(grad@grad)==pytest.approx(float(m*chi*(9*k-5)),rel=2e-13)


def test_projected_scalar_comparison_bounds_positive_product_growth():
    from experiments.expD34_readout_race import population_concentration as pc
    x=jnp.linspace(-1,1,41)
    p=jnp.array([.2,.5,-.1, .3,-.2,.4, .6,.1,-.3, .1])
    grad=jax.grad(lambda z:jnp.sum(z[:-1]**2)*jnp.sqrt(ed.shape(z,6)))(p)
    base=kernel.decomposition(p,x,jnp.zeros_like(x))
    direction=kernel._project(grad,base['JC'],base['gram'])
    y=kernel.output(p,x)+kernel._fine(base['J']@direction,base['basis'])
    state=kernel.decomposition(p,x,y)
    observed=grad@(-state['F'])
    assert observed>0
    m=jnp.sum(p[:-1]**2);chi=ed.shape(p,6);k=ed.shape(p,10)/chi**2
    product=m*jnp.sqrt(chi);w=(len(p)-1)//3
    mu2,mu4,mu6=(jnp.mean(x**i) for i in (2,4,6))
    d3=jnp.sqrt(8*(mu4-mu2**2)/27+3*(mu6-mu4**2/mu2)/16)
    e0=jnp.sqrt(jnp.mean(state['eH']**2))
    upper=e0*jnp.sqrt(9*k-5)*(d3*product**2/w+pc.C5*jnp.sqrt(k)*product**3/w**2)
    assert observed<=upper
