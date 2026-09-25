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
