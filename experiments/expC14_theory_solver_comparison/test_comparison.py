"""Tests of the research recipe, not just plotting or array plumbing."""
import math
from pathlib import Path
import sys
import mpmath as mp
import numpy as np
from scipy.linalg import svd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC14_theory_solver_comparison import run as r

CFG = yaml.safe_load((Path(__file__).parent/'config.yaml').read_text())


def test_admissibility_and_high_precision_boundary_envelope():
    for N in CFG['N']:
        h,R,slope,centers,checks = r.geometry(N,CFG)
        assert all(checks.values())
        alpha,D,L,E = r.envelopes(N,CFG)
        other = r.envelopes(N,CFG,precision=256)
        np.testing.assert_array_equal(alpha,other[0])
        np.testing.assert_array_equal(D,other[1])
        np.testing.assert_array_equal(alpha[:-1],alpha[-2::-1])
        assert np.isclose(alpha[-1],1+alpha[:-1].sum())
        with mp.workdps(65):
            lam=mp.mpf('.25');delta=mp.mpf(1)
            baseline=mp.mpf(h)/(2*(delta-mp.pi/(2*slope)))
            m=(R+1)//2
            for i in (1,m):
                zeta=mp.exp(-2*lam)
                pp=lambda k:mp.fprod([1-zeta**j for j in range(1,k+1)])
                Li=zeta**(mp.mpf(i*(i+1)-1)/2)/pp(i-1)/pp(m-i)
                Li*=mp.fprod([1+zeta**(mp.mpf(j)-mp.mpf('.5')) for j in range(1,m+1) if j!=i])
                expected=baseline+mp.mpf(h)*(mp.pi/(2*lam)+4*mp.log(2)/mp.pi)*Li/2
                assert abs(alpha[i-1]-float(expected)) <= np.spacing(alpha[i-1])
            assert alpha[m] == float(baseline)


def test_scaling_equivalence_without_truncation_and_difference_with_it():
    A=np.diag([1.,1e-4]);y=np.array([1.,1e-4]);D=np.array([1.,100.])
    def solve(F, cutoff):
        U,s,Vt=svd(F)
        return Vt.T @ np.where(s>cutoff*s[0],(U.T@y)/s,0.)
    np.testing.assert_allclose(solve(A,0),D*solve(A*D,0))
    np.testing.assert_allclose(solve(A,1e-3),[1,0])
    np.testing.assert_allclose(D*solve(A*D,1e-3),[1,1])


def test_cutoff_and_coefficient_bound_on_real_dictionary():
    N=128
    h,R,slope,centers,_=r.geometry(N,CFG)
    x=np.linspace(-1,1,N*8+1)
    A=r.feature_matrix(x,centers,slope)
    alpha,D,L,E=r.envelopes(N,CFG)
    F=A*D/math.sqrt(len(x));y=r.target_values(x)[:,0]/math.sqrt(len(x))
    U,s,Vt=svd(F,full_matrices=False)
    bounds=r.factorization_bounds(F,U,s,Vt,L)
    c,d=r.theorem_solve(F,y,U,s,Vt,D,L,E,r.analytic_B(CFG)[0],bounds)
    assert d['cutoff_pass'] and d['transformed_norm_pass']
    assert d['rank']==np.count_nonzero(s>d['tau'])
    assert abs(d['tau']-(d['cutoff_analytic']+d['cutoff_matrix']+d['cutoff_labels'])) < 1e-13*d['tau']
    assert np.sum(abs(c)) <= d['coefficient_bound']
    assert np.linalg.norm(A@c/math.sqrt(len(x))-y) <= d['sample_absolute_bound']


def test_oracle_and_pairwise_sum():
    x=np.array([-1.,-.375,0.,.375,1.])
    np.testing.assert_array_equal(r.target_values(x,160),r.target_values(x,256))
    a=np.arange(21,dtype=float).reshape(7,3)
    np.testing.assert_array_equal(r.pairwise_sum(a),a.sum(axis=0))
    np.testing.assert_array_equal(r.pairwise_sum(a,axis=1),a.sum(axis=1))
    _,_,slope,centers,_=r.geometry(128,CFG)
    # Exact tanh library implementation agrees with independent high precision.
    xx=np.linspace(-1,1,1025)
    mat=r.feature_matrix(xx,centers,slope)
    with mp.workdps(65):
        for i,j in [(0,0),(17,21),(512,70),(1024,len(centers)-1)]:
            expected=float(mp.tanh(mp.mpf(slope)*(mp.mpf(float(xx[i]))-mp.mpf(float(centers[j])))))
            assert mat[i,j]==expected


def test_nonzero_recovery_and_native_export():
    rng=np.random.default_rng(421)
    A=rng.uniform(-1,1,(16,5));A[:,-1]=1
    D=np.linspace(.3,1.,5);L=np.linalg.norm(D)
    F=A*D/4
    a_star=np.array([.5,-.3,.2,.1,.7])
    y=F@a_star
    U,s,Vt=svd(F,full_matrices=False)
    bounds=r.factorization_bounds(F,U,s,Vt,L)
    c,d=r.theorem_solve(F,y,U,s,Vt,D,L,1e-12,2.,bounds)
    assert d['rank']==5 and d['cutoff_pass']
    np.testing.assert_allclose(c,D*a_star,atol=1e-14,rtol=0)
    np.testing.assert_allclose(A@c/4,y,atol=1e-14,rtol=0)
    assert d['export_mass'] >= np.sum(abs(c))
