"""Independent checks of characteristic normalization and projected dynamics."""
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from experiments.expD34_readout_race import core, targets, transport as tr


def example():
    rng = np.random.default_rng(21)
    return rng.normal(size=(3, 7))*.4, .12, targets.grid(64), np.linspace(1, 7, 7)/28


@pytest.mark.parametrize("degree", [-1, 9])
def test_weighted_characteristics_against_independent_loss_gradient(degree):
    z, d, x, w = example(); y = np.sin(2*np.pi*x)+.3
    q = tr.basis(x, degree) if degree >= 0 else np.empty((len(x), 0))
    g, gd, *_ = tr.field(jnp.array(z), d, jnp.array(x), jnp.array(y), jnp.array(w), 177, jnp.array(q))
    zz = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    dd = torch.tensor(d, dtype=torch.float64, requires_grad=True)
    xx, yy, ww, qq = map(torch.tensor, (x, y, w, q))
    residual = torch.tanh(xx[:, None]*zz[0]+zz[1]) @ (177*ww*zz[2])+dd-yy
    if degree >= 0: residual = qq @ (qq.T @ residual/len(x))
    (.5*torch.mean(residual**2)).backward()
    np.testing.assert_allclose(np.asarray(g)*(177*w), zz.grad.numpy(), rtol=3e-14, atol=3e-13)
    assert float(gd) == pytest.approx(dd.grad.item(), abs=1e-13)


def test_empirical_particles_match_original_gd_and_discrete_balances():
    z, d, x, _ = example(); width=z.shape[1]; w=np.ones(width)/width
    y=np.sin(2*np.pi*x); kappa=3.; eta=.002
    inputs=dict(x=jnp.array(x), y=jnp.array(y), powers=jnp.array(x[:, None]**np.arange(11)))
    initial=tr.initialize(z[None], np.array([d]))
    out=tr.advance_factory(len(x), width, -1, eta, kappa)(initial,jnp.array(y[None]),jnp.array(w[None]),50)
    zz, dd=jnp.array(z), d
    for _ in range(50): zz,dd=core.update(zz,dd,inputs,0,eta,kappa)
    np.testing.assert_allclose(out['z'][0],zz,atol=2e-15)
    np.testing.assert_allclose(out['d'][0],dd,atol=2e-15)
    balance=lambda a: np.sum(a[0]**2+a[1]**2-a[2]**2/kappa)
    change=balance(np.asarray(out['z'][0]))-balance(z)
    assert change == pytest.approx(float(out['balance_flow'][0]+out['balance_discrete'][0]), abs=1e-14)
    np.testing.assert_allclose(out['positive'][0]-out['negative'][0],abs(zz[0])-abs(z[0]),atol=1e-15)


def test_modal_kernel_and_slope_decomposition():
    z,d,x,w=example(); y=np.sin(2*np.pi*x); kappa=.1
    scalar,arrays=tr.modal_diagnostics(z,d,x,y,w,177,degree=17,kappa=kappa)
    q=tr.basis(x,17)
    g,gd,*_=tr.field(jnp.array(z),d,jnp.array(x),jnp.array(y),jnp.array(w),177,jnp.array(q))
    e=arrays['residual_modes']; K=sum(arrays['K_'+k] for k in ['a','b','c','d'])
    expected=177*np.sum(w*(np.asarray(g[0])**2+np.asarray(g[1])**2+kappa*np.asarray(g[2])**2))+kappa*float(gd)**2
    assert e @ K @ e == pytest.approx(expected,rel=3e-14)
    assert scalar['gradient_accounting_error']<1e-12
    assert scalar['decomposition_error']<1e-12
    assert scalar['schur_min_eigenvalue']>-1e-11
    full_g,full_gd,*_=tr.field(jnp.array(z),d,jnp.array(x),jnp.array(y),jnp.array(w),177,jnp.empty((len(x),0)))
    velocity=-np.asarray(full_g)*np.array([1.,1.,kappa])[:,None]
    h=1e-6
    _,plus=tr.modal_diagnostics(z+h*velocity,d-h*kappa*float(full_gd),x,y,w,177,17,kappa)
    _,minus=tr.modal_diagnostics(z-h*velocity,d+h*kappa*float(full_gd),x,y,w,177,17,kappa)
    finite_difference=sum(plus['K_'+k]-minus['K_'+k] for k in ['a','b','c','d'])/(2*h)
    np.testing.assert_allclose(arrays['K_dot'],finite_difference,rtol=1e-7,atol=1e-7)


def test_quadrature_width_and_modal_basis():
    z,d,w=tr.law_initial(177,8)
    assert z.shape==(3,512) and d==0
    assert w.sum()==pytest.approx(1.,abs=1e-15)
    np.testing.assert_allclose(z @ w,0,atol=1e-16)
    np.testing.assert_allclose((z*w) @ z.T,np.eye(3)*2/178,atol=1e-16)
    x=targets.grid(2048); q=tr.basis(x,65)
    np.testing.assert_allclose(q.T @ q/len(x),np.eye(66),atol=2e-15)
    np.testing.assert_allclose(q[:,:10],tr.basis(x,9),atol=2e-13)


def test_characteristic_euler_converges_at_first_order():
    z,d,x,w=example(); y=np.sin(2*np.pi*x); end=.08
    states=[]
    for eta in (.004,.002,.001,.000125):
        s=tr.initialize(z[None],np.array([d]))
        out=tr.advance_factory(len(x),177,9,eta,1.)(s,jnp.array(y[None]),jnp.array(w[None]),round(end/eta))
        states.append(np.r_[np.asarray(out['z']).ravel(),np.asarray(out['d'])])
    errors=[np.linalg.norm(a-states[-1]) for a in states[:-1]]
    assert errors[0]/errors[1]>1.9 and errors[1]/errors[2]>1.9
