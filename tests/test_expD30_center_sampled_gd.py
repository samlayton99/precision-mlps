import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('expd30_run',ROOT/'experiments/expD30_center_sampled_gd/run.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def config():
    return yaml.safe_load((ROOT/'experiments/expD30_center_sampled_gd/config.yaml').read_text())


def test_exact_centers_and_initial_interpolation():
    cfg=config();x,y,z,h,a,b=mod.setup(cfg)
    np.testing.assert_array_equal(x,z[24:-24])
    assert len(x)==129 and len(z)==177 and cfg['gamma']*h==2
    sp=mod.spectral_data(cfg,x,y,a,b)
    assert np.max(sp['initial_ls_train_loss'])<1e-25
    assert sp['singular_values'][-1]>1e-4


def test_raw_joint_updates_against_autograd_and_no_cross_target_scaling():
    torch.manual_seed(4)
    x=torch.linspace(-1,1,23,dtype=torch.float64)
    y=torch.randn((3,23),dtype=torch.float64)
    a=torch.randn((3,7),dtype=torch.float64)
    b=torch.randn((3,7),dtype=torch.float64)
    v=torch.randn((3,8),dtype=torch.float64)
    rates=torch.tensor([.002,.008,.001],dtype=torch.float64)
    out=mod.joint_chunk(x,y,a,b,v,rates)
    for t in range(3):
        aa=a[t].clone().requires_grad_();bb=b[t].clone().requires_grad_();vv=v[t].clone().requires_grad_()
        for k,rate in enumerate(rates):
            residual=torch.tanh(x[:,None]*aa+bb)@vv[:-1]+vv[-1]-y[t]
            loss=.5*torch.mean(residual**2)
            np.testing.assert_allclose(loss.item(),out[3][k,t].item(),rtol=2e-14,atol=2e-14)
            grads=torch.autograd.grad(loss,(aa,bb,vv))
            with torch.no_grad():
                aa-=rate*grads[0];bb-=rate*grads[1];vv-=rate*grads[2]
        for actual,expected in zip(out[:3],(aa,bb,vv)):
            np.testing.assert_allclose(actual[t].numpy(),expected.detach().numpy(),rtol=2e-14,atol=2e-14)


def test_zero_readout_first_geometry_step_and_fixed_update():
    cfg=config();x,y,z,h,a,b=mod.setup(cfg)
    tx=torch.tensor(x);ty=torch.tensor(y.T.copy())
    aa=torch.tensor(np.tile(a,(4,1)));bb=torch.tensor(np.tile(b,(4,1)))
    vv=torch.zeros((4,len(a)+1),dtype=torch.float64);rr=torch.tensor([.002],dtype=torch.float64)
    out=mod.joint_chunk(tx,ty,aa,bb,vv,rr)
    torch.testing.assert_close(out[0],aa,rtol=0,atol=0);torch.testing.assert_close(out[1],bb,rtol=0,atol=0)
    H=torch.cat((torch.tanh(tx[:,None]*aa[0]+bb[0]),torch.ones((len(x),1),dtype=torch.float64)),dim=1)
    vfixed,loss=mod.fixed_chunk(H,ty.t(),vv.t().contiguous(),rr)
    torch.testing.assert_close(vfixed.t(),out[2],rtol=1e-13,atol=1e-15)


def test_schedule_and_spectral_prediction():
    cfg=config();cfg['steps']=500;cfg['warmup_steps']=50
    x,y,z,h,a,b=mod.setup(cfg);sp=mod.spectral_data(cfg,x,y,a,b)
    peak=float(sp['peak_rate']);r=mod.learning_rates(cfg,peak,'warmup_cosine')
    assert r[0]==cfg['base_rate'] and np.isclose(r[49],peak)
    assert np.isclose(r[-1],.01*peak) and r.max()<sp['stability_limit']
    H=torch.tensor(mod.feature(x,a,b));ty=torch.tensor(y)
    for rate in (cfg['base_rate'],peak):
        v,loss=mod.fixed_chunk(H,ty,torch.zeros((H.shape[1],4),dtype=torch.float64),torch.full((500,),rate,dtype=torch.float64))
        predicted=mod.constant_prediction(sp['singular_values'],sp['alpha'],rate,np.arange(500))
        np.testing.assert_allclose(loss.numpy(),predicted,rtol=1e-9,atol=2e-14)
