"""Check original-theorem conditions and sharper nearby-problem bounds."""
from pathlib import Path
import sys
import mpmath as mp
import numpy as np
import pytest
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expC14_theory_solver_comparison import run as r
from experiments.expC14_theory_solver_comparison import option_c as c
from experiments.expC14_theory_solver_comparison.theory_parameters import prepare

CFG=yaml.safe_load((r.HERE/'option_c_config.yaml').read_text())


def test_N64_range_uses_a_single_admissible_design():
    with pytest.raises(ValueError,match='Nonadmissible'):
        r.geometry(64,CFG)
    cfg=yaml.safe_load((r.HERE/'option_c_64_512.yaml').read_text())
    assert cfg['N']==[64,128,256,512]
    for N in cfg['N']:
        p=prepare(N,cfg)
        assert all(p['checks'].values())
        assert np.isfinite(p['B']).all()


def test_original_analytic_bound_and_geometry_not_replaced():
    for N in CFG['N']:
        p=prepare(N,CFG)
        assert all(p['checks'].values())
        assert N>=16*CFG['a_H']**2
        with mp.workdps(75):
            W=len(p['centers']);lam=mp.mpf(1)/4
            capp=min(mp.pi,lam*(mp.mpf(11)/4)**2)/8
            E=128*mp.pi/lam*(mp.exp(-capp*W)+mp.exp(-mp.pi**2/lam)/W)
            assert mp.mpf(p['E_per_B_upper'])>=E
            assert p['E_per_B_upper']<1e-16
        assert p['L_lower']<=p['L_upper']


def polar(A):
    vals,Q=mp.eigsy(A.T*A)
    inv=Q*mp.diag([1/mp.sqrt(x) for x in vals])*Q.T
    return A*inv


def spectral(A):return max(mp.svd(A,compute_uv=False))


def test_weighted_polar_bounds_against_high_precision_nearby_problem():
    rng=np.random.default_rng(812)
    U=np.linalg.qr(rng.normal(size=(12,4)))[0]+rng.normal(size=(12,4))*1e-5
    V=np.linalg.qr(rng.normal(size=(4,4)))[0]+rng.normal(size=(4,4))*1e-5
    s=np.array([4.,1.,1e-3,1e-6]);Vt=V.T
    F=(U*s)@Vt+rng.normal(size=(12,4))*1e-8
    hu,bounds=c.certify_factorization(F,U,s,Vt,dict(eF=0.))
    y=rng.normal(size=(12,3))
    p=np.column_stack([r.projection(U,y[:,j]) for j in range(3)])
    bs,_=c.projection_bounds(U,y,p,hu,bounds,s)
    with mp.workdps(65):
        Um,Vm=mp.matrix(U.tolist()),mp.matrix(V.tolist())
        U0,V0=polar(Um),polar(Vm)
        F0=U0*mp.diag(s.tolist())*V0.T
        assert spectral(mp.matrix(F.tolist())-F0)<=mp.mpf(bounds['aW'])
        for j in range(3):
            q0=U0.T*mp.matrix(y[:,j].tolist())
            for tau in [0.,1e-5,.1,2.,5.]:
                keep=s>tau
                z=np.zeros(4);z[keep]=p[keep,j]/s[keep]
                difference=[mp.mpf(s[i])*mp.mpf(z[i])-q0[i] for i in range(4) if keep[i]]
                assert mp.sqrt(mp.fsum(x*x for x in difference)) <= mp.mpf(bs[j])


def test_original_cutoff_condition_uses_lower_S_and_no_guard():
    params=prepare(128,CFG)
    n=len(params['D']);U=np.eye(n);Vt=np.eye(n);s=np.linspace(3,1e-4,n)
    # Small projections keep this synthetic solve below the witness budget;
    # the check is diagnostic only and never changes the cutoff.
    p=np.full((n,6),1e-5)
    bounds=dict(aW=1e-12,oV=1e-14,Vplus=1.00000000000001)
    _,info=c.solve_original_cutoff(U,s,Vt,p,params,bounds,1e-14,0)
    assert info['cutoff_pass'] and info['cutoff_slack_lower']>=0
    assert info['tau']>=bounds['aW']
    assert info['tau']>2**-52*s[0]
    assert info['rank']==np.count_nonzero(s>info['tau'])


def test_native_jacobi_returns_full_factors_and_correct_scale():
    from experiments.expC14_theory_solver_comparison.jacobi_probe import jacobi_svd
    rng=np.random.default_rng(918)
    base=rng.normal(size=(16,4));base[:,-1]=base[:,0]
    for scale in [1.,1e-280,1e280]:
        U,s,Vt,meta=jacobi_svd(base*scale)
        assert U.shape==(16,4) and Vt.shape==(4,4)
        assert np.linalg.norm(U.T@U-np.eye(4))<1e-12
        assert np.linalg.norm(Vt@Vt.T-np.eye(4))<1e-12
        assert np.linalg.norm((U*(s/scale))@Vt-base)<1e-12
        assert meta['characters']['joba']=='C'
        assert meta['characters']['jobv']=='J'
