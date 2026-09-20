"""Check the weighted scalar objective, unchanged readout rule, and GD control."""
import numpy as np
import torch
import pytest

from experiments.expD28_loss_gradient_decomposition import run as old
from experiments.expD29_weighted_profile import run as exp


@pytest.mark.parametrize("mu",[10,1000,100000])
def test_weighted_gradient_matches_independent_scalar_autograd(mu):
    x=np.linspace(-1,1,41);y=np.sin(5*x)+.3*np.cos(2*x)
    state=dict(a=np.array([.5,1.,2.,3.]),b=np.array([-.2,.1,.4,-.6]),
               v=np.random.default_rng(5).normal(size=5))
    p={k:torch.nn.Parameter(torch.tensor(v,dtype=torch.float64)) for k,v in state.items()}
    tx,ty=torch.tensor(x),torch.tensor(y)
    h=torch.tanh(tx[:,None]*p['a']+p['b'])
    ordinary=.5*torch.mean((h@p['v'][:-1]+p['v'][-1]-ty).square())
    ordinary.backward()
    readout_gradient=p['v'].grad.clone()
    diagnostic=old.one_diagnostic(state,x,y,.02)
    exp.add_profile_gradient(p,diagnostic['gF'],mu)
    actual={k:v.grad.numpy().copy() for k,v in p.items()}
    np.testing.assert_array_equal(actual['v'],readout_gradient.numpy())
    # Differentiate the scalar objective through the moving SVD subspace.
    q={k:torch.tensor(v,dtype=torch.float64,requires_grad=True) for k,v in state.items()}
    features=torch.tanh(tx[:,None]*q['a']+q['b'])
    A=torch.cat((features,torch.ones((len(x),1),dtype=torch.float64)),dim=1)/np.sqrt(len(x))
    yn=ty/np.sqrt(len(x))
    U,_,_=torch.linalg.svd(A,full_matrices=False)
    rank=diagnostic['rank'];rstar=U[:,:rank]@(U[:,:rank].T@yn)-yn
    weighted=.5*(A@q['v']-yn).square().sum()+(mu-1)*.5*rstar.square().sum()
    expected=torch.autograd.grad(weighted,list(q.values()))
    for k,g in zip(q,expected):
        np.testing.assert_allclose(actual[k],g.numpy(),rtol=3e-10,atol=1e-9)


def test_weight_one_recovers_ordinary_gd():
    torch.set_default_dtype(torch.float64)
    cfg=exp.config()|dict(steps=3,n_train=32,diagnostic_snapshots=4)
    case=exp.train('sine_mixture','xavier',1,cfg)
    reference=old.train('sine_mixture','xavier',cfg)
    for k in ('a','b','v'):
        np.testing.assert_array_equal(case[k],reference[k])
    np.testing.assert_array_equal(case['L'],reference['train_loss'])


def test_first_readout_update_is_gd_for_both_weights():
    torch.set_default_dtype(torch.float64)
    cfg=exp.config()|dict(steps=1,n_train=32,diagnostic_snapshots=2)
    ordinary=exp.train('sine_mixture','xavier',1,cfg)
    weighted=exp.train('sine_mixture','xavier',1000,cfg)
    np.testing.assert_array_equal(weighted['v'][1],ordinary['v'][1])
    assert np.linalg.norm(weighted['a'][1]-ordinary['a'][1])>1e-6
    assert np.linalg.norm(weighted['v'][1]-weighted['v'][0])>0


def test_stopped_run_preserves_last_valid_unscheduled_state(monkeypatch):
    torch.set_default_dtype(torch.float64)
    cfg=exp.config()|dict(steps=6,n_train=32,diagnostic_snapshots=2)
    monkeypatch.setattr(old,'snapshots',lambda steps,count:np.array([0,steps]))
    calls=0
    def inject_failure(parameters,gF,weight):
        nonlocal calls
        calls+=1
        # Keep earlier updates ordinary GD so their endpoint is independently
        # reproducible; inject the first bad gradient at step 4.
        if calls==5:parameters['a'].grad.fill_(float('nan'))
    monkeypatch.setattr(exp,'add_profile_gradient',inject_failure)
    case=exp.train('sine_mixture','xavier',10,cfg)
    assert str(case['status'])=='nonfinite gradient at step 4'
    assert case['step'][-1]==case['saved_steps'][-1]==3
    reference=old.train('sine_mixture','xavier',cfg|dict(steps=3))
    for key in ('a','b','v'):
        np.testing.assert_array_equal(case[key][-1],reference[key][-1])
