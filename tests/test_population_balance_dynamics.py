import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.expD34_readout_race import population_balance_dynamics as m


@pytest.mark.parametrize('scale',[.05,.5,3.])
def test_global_bounds_and_independent_moment_derivatives(scale):
    p=jnp.asarray(np.random.default_rng(312).normal(size=22)*scale)
    x=jnp.linspace(-1,1,129); y=.3+.4*x+jnp.sin(5*x)
    d=m.diagnostics(p,x,y)
    V, rates, quadratic=m.flux(p,x,y,'gd')
    for name,index in (('A',5),('C',6),('m',7),('M',8)):
        exact=jax.jvp(lambda z:m.moments(z)[name],(p,),(-V,))[1]
        np.testing.assert_allclose(rates[index],exact,rtol=2e-12,atol=1e-12)
    lr=jax.jvp(lambda z:jnp.log(jnp.abs(m.moments(z)['rho'])),(p,),(-V,))[1]
    np.testing.assert_allclose(d['log_alignment_rate'],lr,rtol=2e-12,atol=1e-12)
    assert float(d['Delta_dot_full']) <= float(d['Delta_upper_full'])+1e-12
    assert float(d['capacity_remainder_check']) <= float(d['remainder_capacity'])+1e-12
    assert float(d['sensitivity_remainder_check']) <= float(d['remainder_sensitivity'])+1e-12
    assert float(d['gradient_poly_error']) <= m.C7*float(d['target_fine'])*float(d['remainder_sensitivity'])/m.C7+1e-12
    bound=m.comparison({k:float(v) for k,v in d.items()},float(d['M']))
    assert float(d['F_norm']) <= bound['force']+1e-12
    eta=.007
    before,after=m.moments(p),m.moments(p-eta*V)
    for name,linear,quad in (('A',5,13),('C',6,14),('m',7,15),('M',8,16)):
        np.testing.assert_allclose(after[name]-before[name],eta*rates[linear]+eta**2*quadratic[quad],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(after['Delta']-before['Delta'],eta*rates[:5].sum()+eta**2*quadratic[12],rtol=1e-10,atol=1e-12)


def test_integrated_ledger_and_no_division_at_zero_alignment():
    p=jnp.array([.1,.1,.05,-.05,.2,-.2,0.])
    assert not bool(m.moments(p)['log_alignment_valid'])
    x=jnp.linspace(-1,1,65); y=jnp.sin(3*x)
    final,acc=m.advance(p,jnp.zeros(len(m.LEDGER)),x,y,.002,'gd',100)
    np.testing.assert_allclose(m.moments(final)['Delta']-m.moments(p)['Delta'],acc[:5].sum()+acc[12],atol=2e-15)
    assert np.isfinite(float(m.moments(final)['alignment_margin']))


def test_exact_fit_has_zero_effective_movement():
    p=jnp.array([.1,-.2,.04,.02,.3,-.1,.05]); x=jnp.linspace(-1,1,33)
    y=m.kernel.output(p,x)
    final,acc=m.advance(p,jnp.zeros(len(m.LEDGER)),x,y,.01,'effective',10)
    np.testing.assert_array_equal(final,p)
    np.testing.assert_allclose(acc[:5].sum(),0.,atol=1e-20)
