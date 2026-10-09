"""Independent dense, adjoint and nonlinear-Jacobian audits for expF20."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF20_local_ridge_pde'))
from local_operator import LocalBank


def setup(cutoff=22.):
    rng=np.random.default_rng(823)
    theta=np.arange(7)*np.pi/7
    bank=LocalBank(np.c_[np.cos(theta),np.sin(theta)],np.linspace(-1.4,1.4,129),12.,cutoff)
    points=np.r_[rng.uniform(-1,1,(31,2)),[[0.,0.],[2.,2.],[-2.,-2.]]]
    return rng,bank,bank.bind(points)


@pytest.mark.parametrize('order',[0,1,2])
def test_dense_and_transpose(order):
    rng,bank,plan=setup()
    weights=rng.normal(size=(bank.m,bank.n))
    z=bank.gamma*(plan.projection[:,:,None]-bank.centers)
    t=np.tanh(z)
    e=np.exp(-2*abs(z));d=4*e/(1+e)**2
    phi=t if order==0 else bank.gamma*d if order==1 else -2*bank.gamma**2*t*d
    factors=rng.normal(size=bank.m)
    dense=np.einsum('mkj,mj,m->k',phi,weights,factors)
    actual=plan.forward(weights,order,factors)
    np.testing.assert_allclose(actual,dense,rtol=3e-12,atol=3e-10)
    residual=rng.normal(size=plan.k)
    trans=plan.adjoint(residual,order,factors)
    np.testing.assert_allclose(trans,np.einsum('mkj,k,m->mj',phi,residual,factors),rtol=3e-12,atol=3e-10)
    lhs=actual@residual;rhs=np.sum(weights*trans)
    assert abs(lhs-rhs)<2e-12*max(1.,abs(lhs),abs(rhs))


def test_quadratic_residual_jacobian_and_adjoint():
    rng,bank,plan=setup()
    w=rng.normal(size=(bank.m,bank.n))*.01
    v=rng.normal(size=w.shape)*.01
    q=plan.forward(w);lap=plan.forward(w,2)
    jv=lap*plan.forward(v)+q*plan.forward(v,2)
    def residual(a):return plan.forward(a)*plan.forward(a,2)
    delta=1e-4
    finite=(residual(w+delta*v)-residual(w-delta*v))/(2*delta)
    np.testing.assert_allclose(jv,finite,rtol=1e-8,atol=1e-8)
    z=rng.normal(size=plan.k)
    jt=plan.adjoint(lap*z)+plan.adjoint(q*z,2)
    np.testing.assert_allclose(jv@z,np.sum(v*jt),rtol=1e-12,atol=1e-10)


def test_physical_mixed_derivative():
    rng,bank,plan=setup()
    w=rng.normal(size=(bank.m,bank.n))*.01
    step=np.array([0.,1e-5])
    forward=lambda pts:bank.bind(pts).forward(w,1,bank.directions[:,0])
    finite=(forward(plan.points+step)-forward(plan.points-step))/(2e-5)
    mixed=plan.forward(w,2,np.prod(bank.directions,axis=1))
    np.testing.assert_allclose(mixed,finite,rtol=2e-6,atol=2e-6)


def test_cutoff_tails_decrease_and_shapes_guarded():
    rng,bank,plan=setup()
    w=rng.normal(size=(bank.m,bank.n))
    exact=plan.forward(w,2)
    errors=[]
    for cutoff in [5.,10.,18.]:
        b=LocalBank(bank.directions,bank.centers,bank.gamma,cutoff)
        errors.append(np.max(abs(b.bind(plan.points).forward(w,2)-exact)))
    assert errors[0]>errors[1]>errors[2]
    assert errors[-1]<1e-10
    with pytest.raises(ValueError):plan.forward(np.ones(3))
    with pytest.raises(ValueError):plan.forward(w,3)
