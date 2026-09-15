"""Checks for the differentiated numerical profile, not just an identity by subtraction."""
import numpy as np
import torch

from experiments.expD28_loss_gradient_decomposition import run as exp
from experiments.expD26_freeze_and_readout_spectrum.freeze import numpy_gradients


def test_truncated_projector_derivative_matches_autograd_and_finite_difference():
    rng=np.random.default_rng(42)
    U,_=np.linalg.qr(rng.normal(size=(19,5)))
    V,_=np.linalg.qr(rng.normal(size=(5,5)))
    A=U@np.diag([2.,.9,.3,.08,.02])@V.T
    y=rng.normal(size=19)
    out=exp.profiled_matrix(A,y,.1)
    assert out['rank']==3
    ta=torch.tensor(A,dtype=torch.float64,requires_grad=True)
    ty=torch.tensor(y,dtype=torch.float64)
    tu,_,_=torch.linalg.svd(ta,full_matrices=False)
    residual=tu[:,:3]@(tu[:,:3].T@ty)-ty
    loss=.5*residual.square().sum()
    grad=torch.autograd.grad(loss,ta)[0].numpy()
    np.testing.assert_allclose(out['matrix_gradient'],grad,rtol=2e-12,atol=2e-12)
    direction=rng.normal(size=A.shape);direction/=np.linalg.norm(direction)
    eps=1e-6
    fd=(exp.profiled_matrix(A+eps*direction,y,.1)['F']-exp.profiled_matrix(A-eps*direction,y,.1)['F'])/(2*eps)
    np.testing.assert_allclose(np.sum(out['matrix_gradient']*direction),fd,rtol=2e-7,atol=2e-8)
    assert np.linalg.norm(out['matrix_gradient']-out['envelope_matrix_gradient'])>1e-2


def test_untruncated_profile_reduces_to_varpro_envelope():
    rng=np.random.default_rng(6);A=rng.normal(size=(15,4));y=rng.normal(size=15)
    out=exp.profiled_matrix(A,y,1e-13)
    assert out['rank']==4
    np.testing.assert_array_equal(out['matrix_gradient'],out['envelope_matrix_gradient'])
    np.testing.assert_allclose(A.T@out['rstar'],0,atol=1e-13)


def test_geometry_gradients_and_gap_have_independent_directional_checks():
    rng=np.random.default_rng(5)
    x=np.linspace(-1,1,41);y=np.sin(5*x)+.3*np.cos(2*x)
    state=dict(a=np.array([.5,1.,2.,3.]),b=np.array([-.2,.1,.4,-.6]),v=rng.normal(size=5))
    out=exp.one_diagnostic(state,x,y,.02)
    assert 0<out['rank']<5
    expected=numpy_gradients(state,x,y)
    np.testing.assert_allclose(out['gL'],np.r_[expected['a'],expected['b']],rtol=1e-13,atol=1e-13)
    direction=rng.normal(size=8);direction/=np.linalg.norm(direction);eps=1e-5
    plus=state|dict(a=state['a']+eps*direction[:4],b=state['b']+eps*direction[4:])
    minus=state|dict(a=state['a']-eps*direction[:4],b=state['b']-eps*direction[4:])
    dp=exp.one_diagnostic(plus,x,y,.02);dm=exp.one_diagnostic(minus,x,y,.02)
    assert dp['rank']==dm['rank']==out['rank']
    for name in ['L','F','G']:
        np.testing.assert_allclose(np.dot(out['g'+name],direction),(dp[name]-dm[name])/(2*eps),rtol=2e-6,atol=1e-8)
    np.testing.assert_allclose(out['gF']+out['gG'],out['gL'],atol=1e-15)
    assert out['G']==out['L']-out['F']


def test_initializer_pairing_centers_scales_and_zero_readout():
    cfg=exp.config()
    x=exp.initial_state('xavier',cfg);s=exp.initial_state('scaled_xavier',cfg);q=exp.initial_state('qi_zero',cfg)
    np.testing.assert_allclose(-x['b']/x['a'],-s['b']/s['a'],rtol=1e-13,atol=1e-12)
    np.testing.assert_array_equal(x['v'],s['v'])
    np.testing.assert_allclose(np.mean(abs(s['a']))*(2/cfg['resolution']),.25,rtol=1e-14)
    np.testing.assert_array_equal(q['v'],0)
    np.testing.assert_array_equal(q['a'],16)


def test_training_does_not_call_profile_and_zero_readout_has_zero_geometry_gradient(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Readout solve called during ordinary training')
    monkeypatch.setattr(exp,'profiled_matrix',forbidden)
    cfg=exp.config()|dict(steps=2,n_train=32,diagnostic_snapshots=3)
    torch.set_default_dtype(torch.float64)
    case=exp.train('sine','qi_zero',cfg)
    np.testing.assert_array_equal(case['training_geometry_gradient'][0],0)
    np.testing.assert_array_equal(case['a'][1],case['a'][0])
    np.testing.assert_array_equal(case['b'][1],case['b'][0])
    assert np.linalg.norm(case['v'][1])>0
    # This first update is ordinary GD, not the LS optimum.
    x=exp.previous.midpoint_grid(cfg['n_train'])
    y=exp.previous.matched.target_values('sine',x,cfg)
    state={key:case[key][0] for key in ('a','b','v')}
    gradient=numpy_gradients(state,x,y)
    for key in state:
        np.testing.assert_allclose(case[key][1],state[key]-cfg['learning_rate']*gradient[key],rtol=1e-14,atol=1e-15)
