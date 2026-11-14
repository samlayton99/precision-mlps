from pathlib import Path
import sys
import gmpy2 as gp
import numpy as np
import pytest
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expC14_theory_solver_comparison import run as r,option_c as c,affine_grid as ag
from experiments.expC14_theory_solver_comparison.theory_parameters import prepare

CFG=yaml.safe_load((r.HERE/'option_c_32_512_dense.yaml').read_text())


def test_all_dense_widths_admissible_for_original_uniform_route():
    assert CFG['N']==list(range(32,513,16))
    for N in CFG['N']:
        assert all(r.geometry(N,CFG)[-1].values())
    other=dict(CFG);other.pop('input_error_route')
    with pytest.raises(ValueError):r.geometry(48,other)


def test_affine_index_features_match_exact_nominal_geometry():
    N=48;R=r.geometry(N,CFG)[1]
    A,y=ag.training(N,R,8)
    with gp.context(gp.get_context(),precision=256):
        for i,col in [(0,0),(17,R+5),(48*8,R+48)]:
            j=col-R;x=gp.mpfr(2*i-N*8)/(N*8)
            arg=gp.mpfr(N)/8*x+gp.mpfr(N)/8-gp.mpfr(j)/4
            assert A[i,col]==float(gp.tanh(arg))
    assert y.shape==(N*8+1,6)
    N=64;_,R,slope,centers,_=r.geometry(N,CFG)
    A,y=ag.training(N,R,8)
    x=np.linspace(-1,1,N*8+1)
    np.testing.assert_array_equal(A,r.feature_matrix(x,centers,slope))


def test_large_chirp_constants_do_not_overflow_model_or_cutoff():
    params=prepare(32,CFG)
    assert params['B'][3]>np.finfo(float).max
    n=len(params['D']);s=np.linspace(3.,1e-4,n);Vt=np.eye(n);U=np.eye(n)
    p=np.full((n,6),1e-5)
    coeff,info=c.solve_original_cutoff(U,s,Vt,p,params,dict(aW=1e-12,oV=1e-14,Vplus=1.00000000000001),1e-14,3)
    assert np.isfinite(coeff).all() and np.isfinite(info['tau'])
    assert info['cutoff_pass'] and isinstance(info['Ebar'],str)
    assert gp.is_finite(gp.mpfr(info['sample_bound']))


def test_non_dyadic_width_evaluation_matches_direct_affine_network():
    N=48;R=r.geometry(N,CFG)[1];count=128;j=13
    x=-1+2*(np.arange(count)+.5)/count
    with gp.context(gp.get_context(),precision=256):
        yy=np.array([[float(gp.tanh(gp.mpfr(N)/8*gp.mpfr(float(t))+gp.mpfr(N)/8-gp.mpfr(j)/4))] for t in x])
    coefficients=np.zeros((N+2*R+2,1));coefficients[j+R,0]=1
    err,maximum=ag.evaluate(N,R,coefficients,yy)
    assert err[0]==0 and maximum[0]==0
