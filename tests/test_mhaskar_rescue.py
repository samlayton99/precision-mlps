import numpy as np
import pytest
g=pytest.importorskip("gmpy2")
from experiments.expC12_mhaskar_comparison import pbit,robust
from experiments.expC12_mhaskar_comparison.construction import round_bits


@pytest.mark.parametrize("p",[8,24,40,53])
def test_compensation_recovers_sum_lost_by_sequential_readout(p):
    phi=np.ones((1,3));weights=np.array([2.**p,1.,-2.**p])
    assert robust.readout(phi,weights,0.,p,"sequential")[0]==0
    for mode in ["neumaier","dot2","sorted_dot2"]:
        assert robust.readout(phi,weights,0.,p,mode)[0]==1


@pytest.mark.parametrize("p",[8,11,24,40,52,53])
def test_dot2_recovers_product_rounding_error_without_fma(p):
    k=(p+2)//2
    phi=np.array([[1+2.**-k,1.]])
    weights=np.array([1-2.**-k,-1.])
    assert robust.readout(phi,weights,0.,p,"neumaier")[0]==0
    assert robust.readout(phi,weights,0.,p,"dot2")[0]==-2.**(-2*k)


@pytest.mark.parametrize("p",[8,24,53])
def test_sequential_mode_reproduces_existing_kernel(p):
    rng=np.random.default_rng(13)
    a,b=rng.normal(size=(11,31)),rng.normal(size=31)
    np.testing.assert_array_equal(robust.readout(a,b,.17,p,"sequential"),pbit.readout(a,b,.17,p))


@pytest.mark.parametrize("degree",[1,2,3,4])
def test_symmetric_stencil_accuracy_and_exact_parity(degree):
    from numpy.polynomial.chebyshev import chebval
    c=np.zeros(degree+1);c[-1]=1
    nodes=np.cos(np.pi*(np.arange(64)+.5)/64)
    _,poly,t,b=robust.polynomial_data(nodes,chebval(nodes,c),degree,53)
    x=np.linspace(-1,1,101)
    errors=[]
    for h in [.04,.02]:
        model=robust.construct(poly[degree],t,degree,h,b,53)
        errors.append(np.max(abs(robust.evaluate(model,x,53)-chebval(x,c))))
    assert 3.4<errors[0]/errors[1]<4.6
    monomial=np.zeros(degree+1);monomial[-1]=1
    model=robust.construct(monomial,t,degree,.2,b,24)
    np.testing.assert_array_equal(model.readout,(-1)**degree*model.readout[::-1])
    np.testing.assert_array_equal(model.readout,round_bits(model.readout,24))


@pytest.mark.parametrize("p",[24,53])
def test_dot2_close_to_independent_exact_dot_oracle(p):
    rng=np.random.default_rng(61)
    a=round_bits(rng.normal(size=(5,41)),p)
    b=round_bits(rng.normal(size=41),p)
    with g.context(precision=250):
        expected=np.array([float(g.mpfr(g.fsum(g.mpfr(x)*g.mpfr(y) for x,y in zip(row,b)),precision=p)) for row in a])
    np.testing.assert_array_equal(robust.readout(a,b,0.,p,"dot2"),expected)


@pytest.mark.parametrize('level',[1,2,3])
def test_richardson_weights_cancel_even_step_errors(level):
    from experiments.expC12_mhaskar_comparison.extrapolation import coefficients
    c=coefficients(level,53)
    assert abs(c.sum()-1)<5e-16
    for power in range(1,level+1):
        assert abs(np.dot(c,2.**(-2*power*np.arange(level+1))))<5e-16


def test_extrapolated_network_fourth_order_accuracy():
    from experiments.expC12_mhaskar_comparison.extrapolation import construct
    nodes=np.cos(np.pi*(np.arange(64)+.5)/64)
    _,poly,t,b=robust.polynomial_data(nodes,nodes,1,53)
    x=np.linspace(-1,1,101)
    errors=[]
    for h in [.1,.05]:
        model=construct(poly[1],t,1,h,b,53,1)
        errors.append(np.max(abs(robust.evaluate(model,x,53)-x)))
        assert model.width<=1024
    assert 14<errors[0]/errors[1]<18
