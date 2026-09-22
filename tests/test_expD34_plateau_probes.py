import jax
import jax.numpy as jnp
import numpy as np
from experiments.expD34_readout_race import adam_forces as af, plateau, plateau_probes as pp, targets


def example():
    x=jnp.array(targets.grid(64));q=jnp.array(np.polynomial.legendre.legvander(x,9) @ targets.polynomial_map(np.asarray(x)))
    p=jnp.array(np.r_[np.random.default_rng(12).normal(size=21)*.3,.12]);y=q[:,9]+.4*q[:,1]
    r=plateau.prediction(p,x)-y;fine=r-q[:,:2] @ (q[:,:2].T @ r/len(x))
    return p,x,q,y,fine


def test_joint_and_initial_clamp_match_true_gradient():
    p,x,q,y,fine=example();g=af.field(p,x,y)[0]
    for arm in (0,2):
        direction,*_=pp.field(p,x,y,q,fine,arm,1.,9)
        np.testing.assert_allclose(direction,g,atol=3e-15,rtol=2e-12)


def test_weighted_penalty_is_gradient_of_stated_loss():
    p,x,q,y,fine=example()
    for weight in (0.,.1,1.,10.):
        def loss(v):
            r=plateau.prediction(v,x)-y;coeff=q[:,2:9].T @ r/len(x)
            return .5*jnp.mean(r*r)+.5*(weight-1)*(coeff @ coeff)
        direction,*_=pp.field(p,x,y,q,fine,4,weight,9)
        np.testing.assert_allclose(direction,jax.grad(loss)(p),atol=3e-15,rtol=2e-12)


def test_freezing_and_tracking_removal_only_change_named_blocks():
    p,x,q,y,fine=example();g,r,jc,ec=af.field(p,x,y);parts,_=af.split(g,jc,ec)
    frozen,*_=pp.field(p,x,y,q,fine,1,1.,9)
    np.testing.assert_array_equal(frozen[14:],0)
    np.testing.assert_allclose(frozen[:14],g[:14],atol=1e-15)
    changed,*_=pp.field(p,x,y,q,fine,3,1.,9)
    np.testing.assert_allclose(changed[:7],parts[0,:7],atol=1e-15)
    np.testing.assert_allclose(changed[7:],g[7:],atol=1e-15)


def test_probe_resume_and_motion_accounting():
    p,x,q,y,fine=example();advance=pp.advance_factory(x,q)
    state=jax.vmap(pp.initial)(p[None]);setting=jnp.array([[1.,1.,9.,.002]])
    whole,*_=advance(state,y[None],fine[None],setting,17)
    part,*_=advance(state,y[None],fine[None],setting,6)
    resumed,*_=advance(part,y[None],fine[None],setting,11)
    for k in whole:np.testing.assert_array_equal(whole[k],resumed[k])
    np.testing.assert_array_equal(whole['p'][0,14:],p[14:])
    change=jnp.mean(abs(whole['p'][0,:7])-abs(p[:7]))
    np.testing.assert_allclose(change,whole['signed'].sum()+whole['crossing'].sum(),atol=1e-15)
