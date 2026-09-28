import importlib.util
from pathlib import Path

import numpy as np
import scipy.linalg as sla

spec=importlib.util.spec_from_file_location('d37_diagnostics', Path(__file__).with_name('diagnostics.py'))
d=importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def test_single_mode_sharp_bound_and_damping():
    # The note's two-sample tanh example saturates both inequalities.
    gamma=.7
    B=np.array([[-np.tanh(gamma)], [np.tanh(gamma)]])/np.sqrt(2)
    q=np.array([-1.,1.])/np.sqrt(2)
    U,s,_=sla.svd(B,full_matrices=False)
    eps=.01
    assert np.isclose(d.hitting_time(U,s,q,eps,1e-14), np.log(1/eps))
    rho=np.logspace(-8,4,20)
    np.testing.assert_allclose(d.damping_remaining(U,s,q,rho,1e-14),rho/(1+rho),rtol=1e-7)


def test_damping_matches_direct_normal_equations_with_outside_span():
    rng=np.random.default_rng(8)
    B=rng.normal(size=(11,4)); q=rng.normal(size=11);q/=np.linalg.norm(q)
    U,s,_=sla.svd(B,full_matrices=False)
    rho=np.array([.001,.2,8.])
    expected=[]
    for r in rho:
        step=np.linalg.solve(B.T@B+r*s[0]**2*np.eye(4),-B.T@q)
        expected.append(np.linalg.norm(q+B@step))
    np.testing.assert_allclose(d.damping_remaining(U,s,q,rho,1e-14),expected,atol=1e-14)


def test_polynomial_tails_have_correct_cutoff_and_are_orthogonal():
    x=np.linspace(-.99,.99,101)
    Y=np.c_[x**2,np.sin(12*x)]
    P,D,q=d.polynomial_tails(x,Y,30)
    assert D[2:,0].max()<1e-14
    for k in [0,2,8,16]:
        np.testing.assert_allclose(P[:,:k+1].T@q[k,:,1],0,atol=1e-13)
        np.testing.assert_allclose(np.linalg.norm(q[k,:,1]),1,atol=1e-13)


def test_structural_envelope_covers_actual_arbitrary_bias_features():
    x=np.linspace(-1,1,201)
    slopes=np.array([1.,2.,-3.,4.]);bias=np.array([0.,-.3,1.,-.8])
    B=np.tanh(x[:,None]*slopes+bias)/np.sqrt(len(x))
    P,_=sla.qr(np.polynomial.legendre.legvander(x,20),mode='economic')
    for k in [0,5,10,20]:
        tail=B-P[:,:k+1]@(P[:,:k+1].T@B)
        assert np.linalg.norm(tail,'fro')**2 <= d.structural_envelope(4,k,len(slopes))


def test_missing_capacity_is_not_reported_as_finite_hitting_time():
    B=np.array([[1.],[0.]])
    U,s,_=sla.svd(B,full_matrices=False)
    assert np.isinf(d.hitting_time(U,s,np.array([0.,1.]),.1,1e-14))
