"""Check the optional shared schedule against independent PyTorch Adam steps."""
import numpy as np
import torch

from experiments.expD31_split_adam import run as exp


def test_cosine_schedule_endpoints_and_monotonicity():
    cfg=exp.config()|dict(steps=10000,lr_schedule=dict(kind='cosine',start_step=0,min_factor=.001))
    rates=np.array([exp.learning_rate_at(cfg,t) for t in range(10001)])
    assert rates[0]==.002
    np.testing.assert_allclose(rates[-1],.000002,rtol=1e-14)
    assert np.all(np.diff(rates)<=0)
    assert np.all(rates>0)
    assert exp.learning_rate_at(exp.config(),10000)==.002


def test_scheduled_split_against_independent_three_adam_updates(monkeypatch):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    cfg=exp.config()|dict(n_train=41,steps=3,diagnostic_snapshots=4,readout_rcond=.02,
                          lr_schedule=dict(kind='cosine',start_step=0,min_factor=.1))
    initial=dict(a=np.array([.5,1.,2.,3.]),b=np.array([-.2,.1,.4,-.6]),
                 v=np.random.default_rng(5).normal(size=5))
    monkeypatch.setattr(exp.profile,'initial_state',lambda arm,cfg:{k:v.copy() for k,v in initial.items()})
    x=torch.tensor(exp.profile.previous.midpoint_grid(cfg['n_train']))
    y=torch.tensor(exp.profile.previous.matched.target_values('sine_mixture',x.numpy(),cfg))/np.sqrt(len(x))
    p={k:torch.nn.Parameter(torch.tensor(v)) for k,v in initial.items()}
    virtual=[torch.nn.Parameter(torch.zeros(8)) for _ in range(2)]
    opts=[torch.optim.Adam([q],lr=.002,betas=tuple(cfg['adam_betas']),eps=cfg['adam_epsilon']) for q in virtual]
    readout=torch.optim.Adam([p['v']],lr=.002,betas=tuple(cfg['adam_betas']),eps=cfg['adam_epsilon'])
    expected={k:[v.detach().numpy().copy()] for k,v in p.items()};mu=3
    for step in range(cfg['steps']):
        h=torch.tanh(x[:,None]*p['a']+p['b'])
        A=torch.cat([h,torch.ones((len(x),1))],dim=1)/np.sqrt(len(x))
        U,s,_=torch.linalg.svd(A,full_matrices=False);rank=int((s>cfg['readout_rcond']*s[0]).sum())
        r=U[:,:rank]@(U[:,:rank].T@y)-y
        gF=torch.cat(torch.autograd.grad(.5*r.square().sum(),[p['a'],p['b']],retain_graph=True))
        all_g=torch.autograd.grad(.5*(A@p['v']-y).square().sum(),list(p.values()))
        gL=torch.cat(all_g[:2]);moves=[]
        # Independently specify the cosine formula and apply its rate to all three optimizers.
        eta=.002*(.1+.9*.5*(1+np.cos(np.pi*step/3)))
        for q,opt,g,factor in zip(virtual,opts,(gF,gL-gF),(mu,1)):
            opt.param_groups[0]['lr']=eta*factor
            before=q.detach().clone();q.grad=g;opt.step();moves.append(q.detach()-before)
        with torch.no_grad():
            delta=sum(moves);p['a'].add_(delta[:4]);p['b'].add_(delta[4:])
        readout.param_groups[0]['lr']=eta;p['v'].grad=all_g[2];readout.step()
        for k,v in p.items():expected[k].append(v.detach().numpy().copy())
    actual=exp.train('sine_mixture','xavier',mu,cfg)
    for k in p:np.testing.assert_allclose(actual[k],expected[k],rtol=1e-10,atol=1e-11)
