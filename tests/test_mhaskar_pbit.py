"""Independent per-operation oracles for the strict comparison."""
import numpy as np
import pytest

g = pytest.importorskip("gmpy2")
from experiments.expC12_mhaskar_comparison import pbit
from experiments.expC12_mhaskar_comparison.construction import round_bits, TanhNetwork


@pytest.mark.parametrize("p", [8, 11, 24, 40, 52, 53])
def test_affine_features_against_operation_oracle(p):
    x = np.array([-1., -.123456789, 0., .333333333, 1.])
    w = np.array([-32.1789123, -.21, .73, 3.92747])
    b = np.array([.87654321, -.3, 1.21, -12.25467])
    with g.context(precision=200):
        q = lambda v:g.mpfr(v,precision=p)
        expected = np.array([[float(q(g.tanh(q(q(q(xx)*q(ww))+q(bb)))))
                              for ww,bb in zip(w,b)] for xx in x])
    np.testing.assert_array_equal(pbit.features(x,w,b,p),expected)


@pytest.mark.parametrize("p", [8, 24, 53])
def test_projection_taylor_and_polynomial_conversion(p):
    nodes=np.array([-.8,-.3,.1,.7])
    y=np.array([.21,-.31,.72,.81])
    coeff,poly,taylor,bias=pbit.polynomial_data(nodes,y,5,p)
    with g.context(precision=200):
        q=lambda v:g.mpfr(v,precision=p)
        c=[q(0) for _ in range(6)]
        for xx,yy in zip(nodes,y):
            x,z=q(xx),q(yy)
            t=[q(1),x]
            for k in range(2,6):t.append(q(q(q(2*x)*t[-1])-t[-2]))
            for k in range(6):c[k]=q(c[k]+q(z*t[k]))
        c=[q(q(v*2)/q(4)) if k else q(v/q(4)) for k,v in enumerate(c)]
        np.testing.assert_array_equal(coeff,np.array(c,dtype=float))
        t=[q(g.tanh(q(bias)))]
        for r in range(5):
            total=q(0)
            for j in range(r+1):total=q(total+q(t[j]*t[r-j]))
            t.append(q(q((1 if r==0 else 0)-total)/q(r+1)))
        np.testing.assert_array_equal(taylor,np.array(t,dtype=float))
        # Independently specified integer monomial coefficients of T_0..T_5.
        basis=[[1,0,0,0,0,0],[0,1,0,0,0,0],[-1,0,2,0,0,0],
               [0,-3,0,4,0,0],[1,0,-8,0,8,0],[0,5,0,-20,0,16]]
        acc=[q(0) for _ in range(6)]
        for k in range(6):
            acc=[q(a+q(c[k]*q(v))) for a,v in zip(acc,basis[k])]
            np.testing.assert_array_equal(poly[k],np.array(acc,dtype=float))


@pytest.mark.parametrize("p", [8, 24, 53])
def test_stencil_assembly_operation_oracle(p):
    degree=5
    poly=round_bits(np.array([.21,-.31,.72,.81,-.33,.14]),p)
    taylor=round_bits(np.array([.3333333,.888888,-.29629,-.19753,.16460,.0394]),p)
    h,b=.143719,.34657
    model=pbit.construct(poly,taylor,degree,h,b,p)
    with g.context(precision=200):
        q=lambda v:g.mpfr(v,precision=p)
        weights=[q(0) for _ in range(11)]
        for r,a in enumerate(poly):
            term=q(q(a)/q(taylor[r]))
            for k in range(1,r+1):term=q(term/q(q(h)*q(k)))
            if r%2:term=-term
            for j in range(r+1):
                i=degree+2*j-r
                weights[i]=q(weights[i]+term)
                if j<r:term=-q(term*q(q(r-j)/q(j+1)))
        slopes=[q(q(q(h)/q(2))*q(k)) for k in range(-degree,degree+1)]
    np.testing.assert_array_equal(model.readout,np.array(weights,dtype=float))
    np.testing.assert_array_equal(model.slope,np.array(slopes,dtype=float))


def test_affine_kernel_avoids_double_rounding():
    a,b=1+2.**-26,1+2.**-26-2.**-51
    with g.context(precision=200):
        q=lambda v:g.mpfr(v,precision=52)
        expected=float(q(g.tanh(q(q(q(a)*q(b))-q(1)))))
        wrong=float(q(g.tanh(q(q(a*b)-q(1)))))
    assert expected!=wrong
    assert pbit.features(np.array([a]),np.array([b]),np.array([-1.]),52)[0,0]==expected


@pytest.mark.parametrize("p", [8, 24, 53])
def test_whole_forward_separate_rounded_operations(p):
    x=np.array([-.9,.1234567,.8])
    model=TanhNetwork(np.array([.27,2.31]),np.array([.11,-.72]),np.array([1.81,-.93]),.38)
    with g.context(precision=200):
        q=lambda v:g.mpfr(v,precision=p)
        expected=[]
        for xx in x:
            total=q(model.offset)
            for w,b,a in zip(model.slope,model.bias,model.readout):
                phi=q(g.tanh(q(q(q(xx)*q(w))+q(b))))
                total=q(total+q(q(a)*phi))
            expected.append(float(total))
    np.testing.assert_array_equal(pbit.evaluate(model,x,p),expected)


@pytest.mark.parametrize("p", [8, 16, 24, 40, 53])
def test_quill_parameters_and_rule_at_p_bits(p):
    slope,bias,meta=pbit.quill_geometry(1024,24,p)
    np.testing.assert_array_equal(slope,round_bits(slope,p))
    np.testing.assert_array_equal(bias,round_bits(bias,p))
    assert meta["rule_log_score"]<meta["rule_log_budget"]
    assert meta["distinct_biases"]<=1024


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_complete_strict_polynomial_construction_second_order_accuracy(degree):
    from numpy.polynomial.chebyshev import chebval
    c=np.zeros(degree+1);c[-1]=1.
    nodes=np.cos(np.pi*(np.arange(64)+.5)/64)
    _,poly,taylor,bias=pbit.polynomial_data(nodes,chebval(nodes,c),degree,53)
    x=np.linspace(-1,1,101)
    errors=[]
    for step in [.04,.02]:
        model=pbit.construct(poly[degree],taylor,degree,step,bias,53)
        errors.append(np.max(abs(pbit.evaluate(model,x,53)-chebval(x,c))))
    assert 3.5<errors[0]/errors[1]<4.5
