"""Independent MPFR tests of the product and norm enclosures."""
import math
from pathlib import Path
import sys
import time
import gmpy2 as gp
import numpy as np
import pytest
from scipy.linalg import svd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC14_theory_solver_comparison.tight_products import (
    enclosed_product, frobenius_upper, spectral_upper, _split)


def exact_residual(a, b, target):
    with gp.context(gp.get_context(), precision=1200):
        rows=[]
        for i in range(len(a)):
            row=[]
            for j in range(b.shape[1]):
                total=sum(gp.mpfr(float(a[i,k]))*gp.mpfr(float(b[k,j])) for k in range(a.shape[1]))
                row.append(total-gp.mpfr(float(target[i,j])))
            rows.append(row)
        return rows


@pytest.mark.parametrize('kind',['random','cancellation','exponent_range','gram'])
def test_residual_encloses_mpfr(kind):
    rng=np.random.default_rng(20261109)
    a=rng.normal(size=(7,13));b=rng.normal(size=(13,6))
    if kind=='cancellation':
        a[:,1::2]=-a[:,:12:2]
        b[1::2]=b[:12:2]+np.ldexp(rng.normal(size=(6,6)),-45)
    elif kind=='exponent_range':
        a=np.ldexp(a,rng.integers(-180,180,size=a.shape))
        b=np.ldexp(b,rng.integers(-180,180,size=b.shape))
    elif kind=='gram':
        b=a.T
    target=a@b
    enclosed=enclosed_product(a,b,target,block_rows=3)
    exact=exact_residual(a,b,target)
    with gp.context(gp.get_context(),precision=320):
        diff=sum((exact[i][j]-gp.mpfr(float(enclosed.value[i,j])))**2 for i in range(len(a)) for j in range(b.shape[1]))
        exact_norm=gp.sqrt(sum(v*v for row in exact for v in row))
        assert gp.sqrt(diff)<=enclosed.error_frobenius
        assert exact_norm<=enclosed.norm_upper
    if kind!='exponent_range':
        assert enclosed.error_frobenius < 1e-19


def test_split_identity_and_tiny_tail():
    a=np.array([[1.,-.25,np.nextafter(1.,0),2.**-90,0.,2.**-1000]])
    split=_split(a,5,16)
    with gp.context(gp.get_context(),precision=1200):
        for j in range(a.shape[1]):
            rebuilt=sum(gp.mul_2exp(gp.mpfr(float(d[0,j])),split.exponent-16*i) for i,d in enumerate(split.chunks))
            rebuilt+=gp.mul_2exp(gp.mpfr(float(split.remainder[0,j])),split.exponent-80)
            assert rebuilt==gp.mpfr(float(a[0,j]))


def test_frobenius_upper_handles_subnormals():
    a=np.array([[np.nextafter(0.,1.),1e-300],[1e-250,-1e-260]])
    with gp.context(gp.get_context(),precision=1200):
        expected=gp.sqrt(sum(gp.mpfr(float(x))**2 for x in a.ravel()))
        assert expected<=frobenius_upper(a)
    assert frobenius_upper(np.zeros((2,3)))==0


def test_gram_defect_and_spectral_upper():
    rng=np.random.default_rng(8)
    U=np.linalg.qr(rng.normal(size=(129,32)))[0]
    defect=enclosed_product(U.T,U,np.eye(32),block_rows=16)
    assert defect.error_frobenius < 1e-20
    singular=np.linalg.svd(defect.value,compute_uv=False)
    upper=defect.spectral_upper(2)
    assert upper>=singular[0]
    assert upper<defect.norm_upper
    assert spectral_upper(np.zeros((3,5)))==0


def test_raw_product_vector_and_zero():
    rng=np.random.default_rng(9)
    a=rng.normal(size=(9,17));b=rng.normal(size=(17,1))
    e=enclosed_product(a,b)
    exact=exact_residual(a,b,np.zeros((9,1)))
    with gp.context(gp.get_context(),precision=320):
        difference=gp.sqrt(sum((exact[i][0]-gp.mpfr(float(e.value[i,0])))**2 for i in range(9)))
        assert difference<=e.error_frobenius
    zero=enclosed_product(np.zeros((2,3)),np.zeros((3,4)))
    np.testing.assert_array_equal(zero.value,0)
    assert zero.norm_upper < 1e-300


def test_unsupported_exact_digit_sum_rejected():
    with pytest.raises(ValueError):
        enclosed_product(np.ones((1,2**21)),np.ones((2**21,1)))


def test_nonzero_decomposition_tail_is_included():
    rng=np.random.default_rng(11)
    a=rng.normal(size=(3,7));b=rng.normal(size=(7,4));target=a@b
    e=enclosed_product(a,b,target,digits=2,block_rows=2)
    assert e.tail_error_frobenius>1e-12
    exact=exact_residual(a,b,target)
    with gp.context(gp.get_context(),precision=1200):
        error=gp.sqrt(sum((exact[i][j]-gp.mpfr(float(e.value[i,j])))**2 for i in range(3) for j in range(4)))
        assert error<=e.error_frobenius


def test_spectral_mixed_exponents_falls_back_without_dropping_entries():
    # Dividing the smallest positive subnormal by two loses it, so the exact
    # normalization is unavailable although a valid norm bound still exists.
    a=np.array([[1.,np.nextafter(0.,1.)],[0.,.5]])
    upper=spectral_upper(a,power_steps=3)
    assert upper==frobenius_upper(a)
    assert upper>=1.
    # Exact-digit products must still reject this unsupported normalization.
    with pytest.raises(ValueError,match='exponent range'):
        enclosed_product(a,np.eye(2))
    huge=np.diag([1e200,1e-200])
    assert spectral_upper(huge)>=1e200
