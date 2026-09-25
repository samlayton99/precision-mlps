"""Mixing controls preserve population scales without preserving nonlinear output."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest
from experiments.expD34_readout_race import population_energy_interventions as surgery
from experiments.expD34_readout_race import mechanism_persistence_kernel as kernel


@pytest.mark.parametrize('factor',[0,2,4,10])
@pytest.mark.parametrize('path',[0,1])
def test_finite_mixing_preserves_gram_and_achieves_requested_dose(factor,path):
    rng=np.random.default_rng(20)
    p=np.r_[rng.normal(size=3*127)/np.sqrt(127),.13]
    new,record=surgery.redistribute(p,factor,path)
    assert record['valid'],record
    oldh,newh=p[:-1].reshape(3,-1),new[:-1].reshape(3,-1)
    np.testing.assert_allclose(newh@newh.T,oldh@oldh.T,rtol=2e-12,atol=2e-13)
    assert new[-1]==p[-1]
    assert surgery.concentration(newh)==pytest.approx(record['requested_sqrtC6'],rel=2e-11)
    x=np.linspace(-1,1,41)
    old_affine=oldh[2]@oldh[0]*x+oldh[2]@oldh[1]+p[-1]
    new_affine=newh[2]@newh[0]*x+newh[2]@newh[1]+new[-1]
    np.testing.assert_allclose(new_affine,old_affine,rtol=2e-12,atol=2e-13)


def test_permutation_and_sign_symmetries_are_null_controls():
    rng=np.random.default_rng(9); w=13
    p=jnp.asarray(np.r_[rng.normal(size=3*w)/4,.1])
    order=rng.permutation(w);sign=rng.choice([-1,1],w)
    def transform(z):
        return jnp.r_[(z[:-1].reshape(3,-1)[:,order]*sign).reshape(-1),z[-1]]
    x=jnp.linspace(-1,1,33);y=jnp.sin(4*x)
    new=transform(p)
    np.testing.assert_allclose(kernel.output(new,x),kernel.output(p,x),atol=2e-15)
    np.testing.assert_allclose(kernel.ordinary_gradient(new,x,y),
                               transform(kernel.ordinary_gradient(p,x,y)),atol=3e-15)
    # Repeated native steps must commute with the same symmetry.
    for _ in range(10):
        p=p-.002*kernel.ordinary_gradient(p,x,y)
        new=new-.002*kernel.ordinary_gradient(new,x,y)
    np.testing.assert_allclose(new,transform(p),atol=3e-15)
