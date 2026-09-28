"""Numerical checks of the integral, GD prediction, and component swapping."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.integrate import quad
from threadpoolctl import threadpool_limits

SPEC=importlib.util.spec_from_file_location('gamma_solve_core',Path(__file__).with_name('core.py'))
core=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=core
SPEC.loader.exec_module(core)


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(1):
        yield


@pytest.mark.parametrize('N,m',[(16,33),(128,257),(512,1025)])
def test_grid_order_and_mass(N,m):
    g=core.Geometry(N=N,m=m,center_jitter=1,data_jitter=1)
    x,c=g.arrays()
    assert np.all(np.diff(x)>0) and np.all(np.diff(c)>0)
    assert x[0]==-1 and x[-1]==1
    a,b=g.bounds
    assert (b-a)/g.h==pytest.approx(len(c))
    assert c[0]==pytest.approx(a+g.h/2)
    assert c[-1]==pytest.approx(b-g.h/2)
    half=core.Geometry(N=N,m=m,center_jitter=.5,data_jitter=.5)
    xh,ch=half.arrays();x0,c0=core.Geometry(N=N,m=m).arrays()
    np.testing.assert_allclose(xh,(x+x0)/2,atol=1e-15)
    np.testing.assert_allclose(ch,(c+c0)/2,atol=1e-15)


@pytest.mark.parametrize('gamma',[.0625,1.,4.,16.,256.])
def test_integral_against_independent_adaptive_quadrature(gamma):
    g=core.Geometry(N=64)
    a,b=g.bounds
    x=np.array([a,-1.,-.19,-.19+1e-8,.4,1.,b])
    k=core.continuum_kernel(x,gamma,a,b,1/g.h)
    for i in range(len(x)):
        for j in range(i+1):
            value=quad(lambda c:np.tanh(gamma*(x[i]-c))*np.tanh(gamma*(x[j]-c)),a,b,
                       points=x[(x>a)&(x<b)],epsabs=1e-12,epsrel=1e-12,limit=200)[0]
            assert k[i,j]==pytest.approx(1+value/g.h,abs=3e-10)
    np.testing.assert_allclose(k,k.T,atol=1e-13)


def test_bias_and_midpoint_convergence():
    a,b=-1.5,1.5
    x=np.linspace(-1,1,17)
    errors=[]
    for width in [16,32,64,128]:
        h=(b-a)/width
        c=a+(np.arange(width)+.5)*h
        discrete=core.discrete_kernel(x,c,2.)
        continuous=core.continuum_kernel(x,2.,a,b,1/h)
        errors.append(np.max(abs(discrete-continuous))/width)
        np.testing.assert_allclose(core.continuum_kernel(x,1e-12,a,b,1/h),1.,atol=1e-14)
    assert np.all(np.array(errors[1:]) < .3*np.array(errors[:-1]))


def test_prediction_matches_executed_coefficient_gd():
    g=core.Geometry(N=32,m=65,center_jitter=.4,data_jitter=.3)
    x,c=g.arrays();gamma=7.
    J=core.features(x,c,gamma)/np.sqrt(g.m)
    u,s,ev=core.decompose(x,c,gamma)
    rates=np.r_[.5*ev/ev[0],0]
    y=core.target(x,'mixed')/np.sqrt(g.m)
    p=core.weights(u,y)
    v=np.zeros(J.shape[1]);eta=.5/ev[0]
    for n in range(401):
        if n in (0,1,2,20,400):
            actual=np.linalg.norm(J@v-y)**2/np.linalg.norm(y)**2
            assert core.error_squared(rates,p,n)==pytest.approx(actual,abs=3e-13)
        v -= eta*J.T@(J@v-y)


def test_counterfactual_reference_and_matching_invariance():
    data=core.sweep(core.Geometry(N=32,m=65),.5,64,13,8.)
    a=core.comparison(data,n=20000,matching='rank')
    b=core.comparison(data,n=20000,matching='overlap')
    i=data['reference_index']
    assert a['actual'][i]==a['fixed_p'][i]==a['fixed_rates'][i]
    assert a['steps_actual'][i]==a['steps_fixed_p'][i]==a['steps_fixed_rates'][i]
    np.testing.assert_allclose(a['actual'],b['actual'],rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(np.sum(a['p'],axis=1),1.,atol=1e-14)
    for label in ['fixed_p','fixed_rates']:
        assert np.all(np.array(a[label])>=np.array(a[label+'_lower'])-1e-14)
        assert np.all(np.array(a[label])<=np.array(a[label+'_upper'])+1e-14)


def test_integer_crossing_and_censoring():
    rates=np.array([.5,.02,0.]);p=np.array([.3,.7,0.])
    n=core.hitting_time(rates,p)
    assert core.error_squared(rates,p,n)<=.01**2
    assert core.error_squared(rates,p,n-1)>.01**2
    assert core.hitting_time(rates,np.array([.2,.7,.1])) is None
    assert core.hitting_time(rates,p,maximum=10) is None


def test_eigenvector_derivative_check():
    x,c=core.Geometry(N=32,m=65,center_jitter=.4,data_jitter=.4).arrays()
    check=core.local_derivative_check(x,c,8.)['isolated_modes']
    assert len(check)>=8
    assert max(v['derivative_relative_error'] for v in check)<3e-5


def test_scaling_kernel_does_not_change_normalized_rates():
    x,c=core.Geometry(N=16,m=33).arrays()
    J=core.features(x,c,8.)
    s=np.linalg.svd(J,compute_uv=False)
    ss=np.linalg.svd(7*J,compute_uv=False)
    np.testing.assert_allclose((s/s[0])**2,(ss/ss[0])**2,atol=1e-15)
