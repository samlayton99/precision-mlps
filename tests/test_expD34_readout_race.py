"""Independent checks of the discrete dynamics and moment hierarchy."""
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from experiments.expD34_readout_race import core, references, targets


def example(name="sine"):
    z, d = targets.initial(64, 12, 3)
    return jnp.asarray(z), jnp.asarray(d), {k: jnp.asarray(v) for k,v in targets.data(64,name).items()}


@pytest.mark.parametrize("degree", [0,1,3,5,7])
def test_gradients_against_sample_autodiff(degree):
    z,d,data=example()
    pred=lambda z,d: (jnp.tanh(data['x'][:,None]*z[0]+z[1]) @ z[2]+d if degree==0
                      else core.sample_polynomial(z,d,data['x'],degree))
    f=lambda z,d:.5*jnp.mean((pred(z,d)-data['y'])**2)
    expected=jax.grad(f,argnums=(0,1))(z,d)
    loss,_,g,gd,_=core.field(z,d,data,degree)
    np.testing.assert_allclose(loss,f(z,d),rtol=1e-13,atol=1e-15)
    np.testing.assert_allclose(g,expected[0],rtol=1e-11,atol=1e-14)
    np.testing.assert_allclose(gd,expected[1],rtol=1e-11,atol=1e-14)


@pytest.mark.parametrize("kappa", [1e-4,1.,100.])
def test_exact_affine_discrete_closure(kappa):
    z,d,data=example()
    z0=np.asarray(z);G=z0@z0.T;U=np.eye(3);dd=float(d)
    beta=np.asarray(data['ym'][:2])/np.array([1.,float(data['sigma'])])
    step=jax.jit(lambda z,d:core.update(z,d,data,1,.002,kappa))
    for _ in range(100):
        G,dd,U=references.affine_step(G,dd,U,beta,float(data['sigma']),.002,kappa)
        z,d=step(z,d)
    np.testing.assert_allclose(z,U@z0,rtol=1e-11,atol=1e-13)
    np.testing.assert_allclose(G,np.asarray(z)@np.asarray(z).T,rtol=1e-11,atol=1e-13)
    np.testing.assert_allclose(d,dd,atol=1e-14)


def test_matched_targets_and_visibility_orders():
    data=[targets.data(1024,f'moment{p}') for p in (3,5,9)]
    for d in data:
        np.testing.assert_allclose(d['sy'],1.,atol=2e-15)
        np.testing.assert_allclose(d['ym'][:2]/[1,d['sigma']],[.3,.4],atol=2e-15)
        phi=np.polynomial.legendre.legvander(d['x'],9)@d['mapping']
        np.testing.assert_allclose(phi.T@phi/1024,np.eye(10),atol=4e-15)
    np.testing.assert_allclose(data[1]['ym'][:4],data[2]['ym'][:4],atol=1e-15)
    assert abs(data[0]['ym'][3]-data[2]['ym'][3])>.01
    z,b,_=example()
    for degree,pair in ((1,(0,2)),(3,(1,2))):
        fields=[references.polynomial(z,b,degree,jnp.asarray(data[i]['Q']),
                    jnp.asarray(data[i]['ym']),data[i]['sy']) for i in pair]
        np.testing.assert_allclose(fields[0][2],fields[1][2],atol=1e-15)


def test_upstream_initialization_and_simultaneous_torch_steps():
    from experiments.expD28_loss_gradient_decomposition import run as previous
    import torch
    torch.set_default_dtype(torch.float64)
    cfg=previous.config()|dict(resolution=64,halo=12,seed=3,n_train=64,steps=25,diagnostic_snapshots=30)
    old=previous.train('sine','xavier',cfg)
    z,d,data=example()
    step=jax.jit(lambda z,d:core.update(z,d,data,0,.002,1.))
    for i in range(26):
        np.testing.assert_allclose(z,np.stack((old['a'][i],old['b'][i],old['v'][i,:-1])),atol=2e-14,rtol=1e-12)
        np.testing.assert_allclose(d,old['v'][i,-1],atol=2e-14)
        z,d=step(z,d)


def test_pointwise_moment_bound_with_nonlinear_bias():
    z,d,data=example()
    z=z.at[1].add(.8)
    loss,moments,g,_,_=core.field(z,d,data,0)
    R=float(jnp.sqrt(2*loss))
    for order in (2,4,6,8):
        approx,rho=references.slope_interval(z,np.asarray(moments),R,order)
        a,b,c=np.asarray(z)
        bound=(1/np.cos(rho)**2/rho**order * np.sqrt(np.mean(np.asarray(data['x'])**(2*order+2)))
               *np.linalg.norm(c*a**order))
        assert np.linalg.norm(np.asarray(g[0])/R-approx)<=bound+2e-15
