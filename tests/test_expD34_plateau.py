"""Independent derivative and linear-GD checks for the plateau measurements."""
import jax
import jax.numpy as jnp
import numpy as np
from experiments.expD34_readout_race import adam_forces as af, plateau, targets


def state():
    rng=np.random.default_rng(73)
    p=jnp.array(np.r_[rng.normal(0,.4,21),.13])
    x=jnp.array(targets.grid(96)); y=jnp.sin(5*x)+.3*x
    return p,x,y


def test_projection_matches_original_decomposition():
    p,x,y=state(); g,r,jc,ec=af.field(p,x,y); channels,_=af.split(g,jc,ec)
    f=plateau.effective(p,r,x)
    np.testing.assert_allclose(f,channels[0],rtol=2e-11,atol=2e-13)
    np.testing.assert_allclose(jc @ f,0,atol=2e-13)


def test_jvp_matches_centered_force_difference():
    p,x,y=state(); velocity=-af.field(p,x,y)[0]
    fn=lambda q:plateau.effective(q,plateau.prediction(q,x)-y,x)
    derivative=jax.jvp(fn,(p,),(velocity,))[1]; h=1e-4
    finite=(fn(p+h*velocity)-fn(p-h*velocity))/(2*h)
    np.testing.assert_allclose(derivative,finite,rtol=2e-6,atol=1e-10)


def test_derivative_partition_and_target_partition():
    p,x,y=state(); zero=jnp.zeros_like(p)
    for adaptive in (False,True):
        d=plateau.diagnostics(p,zero,zero,jnp.zeros((3,len(p))),jnp.array(0),y,x,
            jnp.array([.002,.9 if adaptive else 0.,.999,1e-8,float(adaptive)]))
        for k in ('force_identity','derivative_identity','target_identity','projection_identity'):
            assert float(d[k])<1e-10


def test_frozen_tangent_matches_explicit_linear_gd():
    p,x,y=state(); J=np.asarray(plateau.tangent(p,x)); p0=np.asarray(p); pn=p0.copy()
    r0=np.asarray(plateau.prediction(p,x)-y); eta=.002
    for _ in range(257): pn-=eta*(J.T @ (r0+J @ (pn-p0)))/len(x)
    force,predicted=plateau.frozen_tangent(p0,np.asarray(x),np.asarray(y),[257],eta)[0]
    np.testing.assert_allclose(predicted,pn,rtol=1e-10,atol=1e-12)
    g=J.T @ (r0+J @ (pn-p0))/len(x); jc=np.asarray(plateau.coarse(x)) @ J/len(x)
    expected=g-jc.T @ np.linalg.solve(jc @ jc.T,jc @ g)
    np.testing.assert_allclose(force,expected[:7],rtol=1e-9,atol=1e-12)
