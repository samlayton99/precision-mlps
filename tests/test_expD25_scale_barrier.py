"""Independent intervention and loss-decomposition checks."""
import numpy as np
import torch
import pytest
from experiments.expD25_scale_barrier import run as exp


def test_only_geometry_step_changes_and_refit_does_not_enter_training():
    torch.set_num_threads(1)
    cfg=exp.configuration() | dict(resolution=8,halo=2,n_train=64,n_eval=256,steps=5)
    multiplier=100
    case=exp.train('sine_mixture','xavier',multiplier,cfg)
    init=exp.old.initial_state('xavier',cfg)
    x=exp.old.midpoint_grid(cfg['n_train']);y=exp.old.matched.target_values('sine_mixture',x,cfg)
    a,b,v=(init[k].copy() for k in ('a','b','v'))
    for i in range(6):
        for k,value in [('a',a),('b',b),('v',v)]:np.testing.assert_allclose(case[k][i],value,atol=2e-14,rtol=1e-12)
        h=np.tanh(x[:,None]*a+b);r=h@v[:-1]+v[-1]-y
        sensitivity=r[:,None]*v[:-1]*(1-h*h)
        ga=np.mean(x[:,None]*sensitivity,axis=0);gb=np.mean(sensitivity,axis=0)
        gv=np.r_[h.T@r/len(x),r.mean()]
        if i<5:
            a-=cfg['learning_rate']*multiplier*ga
            b-=cfg['learning_rate']*multiplier*gb
            v-=cfg['learning_rate']*gv
    saved={k:case[k].copy() for k in ('a','b','v')}
    exp.evaluate(case,cfg)
    for k in saved:np.testing.assert_array_equal(case[k],saved[k])
    assert np.max(case['evaluation'][:,6])<1e-14


def test_common_scale_derivative_preserves_centers():
    x=exp.old.midpoint_grid(256)
    a=np.array([.8,-1.3,4.]);b=np.array([.2,-.1,.7]);v=np.array([.3,-.7,.4,.1])
    y=np.sin(3*x)
    def loss(q):
        r=exp.old.design(x,np.exp(q)*a,np.exp(q)*b)@v-y
        return .5*np.mean(r*r)
    s=x[:,None]*a+b;h=np.tanh(s);r=h@v[:-1]+v[-1]-y
    analytic=np.mean(r*((s*(1-h*h))@v[:-1]))
    np.testing.assert_allclose(analytic,(loss(1e-5)-loss(-1e-5))/2e-5,rtol=1e-8,atol=1e-11)
    np.testing.assert_allclose(-(b*8)/(a*8),-b/a)


def test_scale_and_shared_scale_updates_fix_centers_without_width_speedup():
    cfg=exp.configuration() | dict(resolution=8,halo=2,n_train=64,n_eval=256,steps=5)
    x=exp.old.midpoint_grid(cfg['n_train']);y=exp.old.matched.target_values('runge',x,cfg)
    for mode in ('scale_only','shared_scale'):
        case=exp.train('runge','gamma_1',100,cfg,geometry_mode=mode)
        z=-case['b'][0]/case['a'][0]
        for i in range(5):
            a,b,v=(case[k][i] for k in ('a','b','v'))
            h=np.tanh((x[:,None]-z)*a);r=h@v[:-1]+v[-1]-y
            gradient=np.mean(r[:,None]*v[:-1]*(1-h*h)*(x[:,None]-z),axis=0)
            if mode=='shared_scale':gradient=np.full_like(gradient,gradient.mean())
            np.testing.assert_allclose(case['a'][i+1],a-cfg['learning_rate']*100*gradient,atol=2e-14)
        np.testing.assert_allclose(-case['b']/case['a'],np.broadcast_to(z,case['a'].shape),atol=1e-15)


def test_exact_geometry_hessian_matches_autodiff():
    from experiments.expD25_scale_barrier.diagnostics import measure
    x=exp.old.midpoint_grid(32);y=np.sin(4*x)
    a=np.array([1.2,3.1]);b=np.array([-.3,.5]);v=np.array([.7,-.4,.1])
    _,actual=measure(a,b,v,x,y)
    tx,ty,tv=map(torch.tensor,(x,y,v))
    def objective(p):
        r=torch.tanh(tx[:,None]*p[:2]+p[2:])@tv[:-1]+tv[-1]-ty
        return .5*torch.mean(r*r)
    expected=torch.autograd.functional.hessian(objective,torch.tensor(np.r_[a,b])).numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)


def test_cache_rejects_changed_training_configuration(tmp_path):
    cfg=exp.configuration();path=tmp_path/'case.npz'
    exp.arrays_save(path,{'geometry_mode':np.array('raw')},cfg)
    exp.load_case(path,cfg)
    with pytest.raises(ValueError,match='seed'):
        exp.load_case(path,cfg | dict(seed=5))


def test_adaptive_shared_scale_keeps_readout_on_plain_gd():
    cfg=exp.configuration() | dict(resolution=8,halo=2,n_train=64,n_eval=256,steps=5)
    c=exp.train('sine_mixture','gamma_1',1,cfg,geometry_mode='shared_scale',geometry_optimizer='adam')
    x=exp.old.midpoint_grid(cfg['n_train']);y=exp.old.matched.target_values('sine_mixture',x,cfg)
    z=-c['b'][0]/c['a'][0];m=0.;s=0.
    for i in range(5):
        a,b,v=(c[k][i] for k in ('a','b','v'))
        h=np.tanh(x[:,None]*a+b);r=h@v[:-1]+v[-1]-y
        g=np.mean(np.mean(r[:,None]*v[:-1]*(1-h*h)*(x[:,None]-z),axis=0))
        m=.9*m+.1*g;s=.999*s+.001*g*g
        update=cfg['learning_rate']*(m/(1-.9**(i+1)))/(np.sqrt(s/(1-.999**(i+1)))+1e-8)
        np.testing.assert_allclose(c['a'][i+1],a-update,atol=2e-14,rtol=1e-12)
        gv=np.r_[h.T@r/len(x),r.mean()]
        np.testing.assert_allclose(c['v'][i+1],v-cfg['learning_rate']*gv,atol=2e-14)
