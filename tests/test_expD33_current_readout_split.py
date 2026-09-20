"""Independent projector-VJP, current-coefficient, and full Adam-update checks."""
import numpy as np
import pytest
import torch

from experiments.expD33_current_readout_split import run as exp


def initial():
    return dict(a=np.array([.5,1.,2.,3.]),b=np.array([-.2,.1,.4,-.6]),
                v=np.random.default_rng(5).normal(size=5))


def config():
    return exp.config()|dict(n_train=41,n_eval=81,steps=3,diagnostic_snapshots=4,readout_rcond=.02)


def torch_signals(p,x,y,rcond):
    h=torch.tanh(x[:,None]*p['a']+p['b'])
    A=torch.cat((h,torch.ones((len(x),1),dtype=torch.float64)),dim=1)/np.sqrt(len(x))
    r=A@p['v']-y/np.sqrt(len(x))
    U,s,_=torch.linalg.svd(A.detach(),full_matrices=False)
    Ur=U[:,s>rcond*s[0]]
    seed=(r-Ur@(Ur.T@r)).detach()
    loss=.5*r.square().sum()
    gout=torch.cat(torch.autograd.grad((seed*(A@p['v'])).sum(),[p['a'],p['b']],retain_graph=True))
    all_l=torch.autograd.grad(loss,[p['a'],p['b'],p['v']])
    return gout,torch.cat(all_l[:2]),all_l[2]


@pytest.mark.parametrize('cutoff',[1e-13,.1])
def test_projected_seed_matches_independent_autodiff(cutoff):
    state=initial();cfg=config();x=exp.profile.previous.midpoint_grid(cfg['n_train'])
    y=exp.profile.previous.matched.target_values('runge',x,cfg)
    d=exp.projected_signals(state,x,y,cutoff)
    p={k:torch.tensor(v,dtype=torch.float64,requires_grad=True) for k,v in state.items()}
    out,total,_=torch_signals(p,torch.tensor(x),torch.tensor(y),cutoff)
    np.testing.assert_allclose(d['gout'],out.numpy(),rtol=1e-10,atol=3e-14)
    np.testing.assert_allclose(d['gout']+d['gparallel'],total.numpy(),rtol=1e-11,atol=3e-14)
    np.testing.assert_allclose(d['retained_basis'].T@d['rperp'],0,atol=3e-15)
    # A retained projector acts on the actual residual, not just on -y.
    if cutoff==.1:assert d['discarded_prediction_norm']>1e-4


def test_zero_current_readout_annuls_signal_but_not_varpro_gradient():
    state=initial();state['v'][:]=0;cfg=config()
    x=exp.profile.previous.midpoint_grid(cfg['n_train'])
    y=exp.profile.previous.matched.target_values('runge',x,cfg)
    d=exp.projected_signals(state,x,y,1e-13)
    np.testing.assert_array_equal(d['gout'],np.zeros(8))
    varpro=exp.profile.one_diagnostic(state,x,y,1e-13)
    assert np.linalg.norm(varpro['gF'])>1e-6


@pytest.mark.parametrize('mu',[100,1000,25000])
@pytest.mark.parametrize('scheduled',[False,True])
def test_three_updates_against_independent_pytorch_streams(monkeypatch,mu,scheduled):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    cfg=config();state=initial()
    if scheduled:cfg['lr_schedule']=dict(kind='cosine',start_step=0,min_factor=.001)
    monkeypatch.setattr(exp.profile,'initial_state',lambda arm,cfg:{k:v.copy() for k,v in state.items()})
    x=torch.tensor(exp.profile.previous.midpoint_grid(cfg['n_train']))
    y=torch.tensor(exp.profile.previous.matched.target_values('runge',x.numpy(),cfg))
    p={k:torch.nn.Parameter(torch.tensor(v)) for k,v in state.items()}
    virtual=[torch.nn.Parameter(torch.zeros(8)) for _ in range(2)]
    adams=[torch.optim.Adam([q],lr=cfg['learning_rate']*factor,betas=tuple(cfg['adam_betas']),
                            eps=cfg['adam_epsilon']) for q,factor in zip(virtual,(mu,1))]
    readout=torch.optim.Adam([p['v']],lr=cfg['learning_rate'],betas=tuple(cfg['adam_betas']),eps=cfg['adam_epsilon'])
    expected={k:[v.detach().numpy().copy()] for k,v in p.items()}
    for step in range(cfg['steps']):
        eta=cfg['learning_rate']*(.001+.999*.5*(1+np.cos(np.pi*step/cfg['steps']))) if scheduled else cfg['learning_rate']
        for opt,factor in zip(adams,(mu,1)):opt.param_groups[0]['lr']=eta*factor
        readout.param_groups[0]['lr']=eta
        gout,gL,gv=torch_signals(p,x,y,cfg['readout_rcond'])
        movements=[]
        for q,opt,g in zip(virtual,adams,(gout,gL-gout)):
            before=q.detach().clone();q.grad=g;opt.step();movements.append(q.detach()-before)
        with torch.no_grad():
            delta=sum(movements);p['a'].add_(delta[:4]);p['b'].add_(delta[4:])
        p['v'].grad=gv;readout.step()
        for k,v in p.items():expected[k].append(v.detach().numpy().copy())
    actual=exp.train('runge',mu,cfg)
    for k in p:
        np.testing.assert_allclose(actual[k],expected[k],rtol=3e-8,atol=2e-7)
