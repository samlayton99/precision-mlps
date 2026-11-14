"""Algebra, update, curvature, and explicit-constant checks. No training runs."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'1')
from pathlib import Path
import json,math
import numpy as np
import sympy as sp
import torch
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/checkpoint_D_optimizers/expD65_radial_loss_budget/structural_audit'

def counterexample():
    z=sp.symbols('z',positive=True)
    F=(1-z*z)**2/(4*((1+z*z)**2+4))
    derivative=-2*z*(1-z*z)*(z*z+3)/(z**4+2*z*z+5)**2
    assert sp.simplify(sp.diff(F,z)-derivative)==0
    W=196;eta=.001;a=.5;t1=math.tanh(a/2);t2=math.tanh(a);c=(t1+t2)/(t1*t1+t2*t2)
    x=torch.tensor([-1.,-.5,.5,1.],dtype=torch.float64);y=(7*x-4*x**3)/3
    np.testing.assert_array_equal(y.numpy(),np.array([-1.,-1.,1.,1.]))
    theta=torch.tensor(np.r_[np.full(W,a),np.zeros(W),np.full(W,c/W),0.],requires_grad=True)
    q=torch.tanh(x[:,None]*theta[:W]+theta[W:2*W])@theta[2*W:3*W]+theta[-1]
    L=.5*torch.mean((q-y)**2);g,=torch.autograd.grad(L,theta)
    reduced=torch.tensor([a,c],dtype=torch.float64,requires_grad=True)
    pred=reduced[1]*torch.tanh(reduced[0]*x);loss=.5*torch.mean((pred-y)**2);gr,=torch.autograd.grad(loss,reduced)
    np.testing.assert_allclose(g[:W].numpy(),np.full(W,gr[0].item()/W),atol=1e-16,rtol=1e-11)
    np.testing.assert_allclose(g[2*W:3*W].numpy(),np.full(W,gr[1].item()),atol=1e-14,rtol=1e-10)
    assert torch.max(torch.abs(g[W:2*W])).item()<1e-16 and abs(g[-1].item())<1e-16
    C=6.;H=W+1+2*C*C/W+(C+1)*(4*C/W+math.sqrt(2))
    assert L.item()<.05 and c/W<1/math.sqrt(W) and eta*W<1 and eta*H<1
    assert eta*C/(3*W)<a/2
    return dict(width=W,learning_rate=eta,initial_slope=a,aggregate_readout=c,individual_readout=c/W,
                initial_loss=L.item(),zero_scale_profile_limit=.05,readout_bound=C,
                segment_hessian_bound=H,eta_hessian=eta*H,
                max_slope_step_under_level_bound=eta*C/(3*W),symbolic_profile_derivative_verified=True,
                full_native_GD_reduction_verified=True,
                scope='Correlated symmetric initialization inside PyTorch default parameter ranges; not an iid-draw or dense-grid counterexample.')

def hessian_and_saturation():
    rows=[];saturation=None
    for name in ['square','sine_4pi','runge_100']:
        p=ROOT/f'results/checkpoint_D_optimizers/expD56_scale_acquisition/sweep/runs/{name}__init1__lr1e-03'
        with np.load(p/'initial.npz') as z:x=z['train_x'];y=z['train_y']
        with np.load(p/'checkpoint.npz') as z:
            steps=z['history'][:,z['keys'].tolist().index('step')];snap=z['snapshots'].copy()
        for step in [30000,200000]:
            theta=snap[np.flatnonzero(steps==step)[0]];a,b,c=np.split(theta[:-1],3);W=len(a);N=len(x)
            h=np.tanh(x[:,None]*a+b);hp=1-h*h;hpp=-2*h*hp;e=h@c+theta[-1]-y
            J=np.column_stack([hp*c*x[:,None],hp*c,h,np.ones(N)]);H=J.T@J/N;inds=np.arange(W)
            aa=np.mean(e[:,None]*hpp*c*x[:,None]**2,axis=0);ab=np.mean(e[:,None]*hpp*c*x[:,None],axis=0);bb=np.mean(e[:,None]*hpp*c,axis=0)
            ac=np.mean(e[:,None]*hp*x[:,None],axis=0);bc=np.mean(e[:,None]*hp,axis=0)
            H[inds,inds]+=aa;H[inds+W,inds+W]+=bb
            for ii,jj,val in [(inds,inds+W,ab),(inds,inds+2*W,ac),(inds+W,inds+2*W,bc)]:H[ii,jj]+=val;H[jj,ii]+=val
            ev=np.linalg.eigvalsh(H)
            t=torch.tensor(theta,requires_grad=True);tx=torch.tensor(x);ty=torch.tensor(y)
            pred=torch.tanh(tx[:,None]*t[:W]+t[W:2*W])@t[2*W:3*W]+t[-1]
            loss=.5*torch.mean((pred-ty)**2);g,=torch.autograd.grad(loss,t,create_graph=True)
            v=torch.tensor(np.random.default_rng(65).normal(size=len(theta)))
            hv,=torch.autograd.grad(g@v,t)
            np.testing.assert_allclose(H@v.numpy(),hv.detach().numpy(),rtol=1e-10,atol=1e-11)
            rows.append(dict(target=name,step=step,min_eigenvalue=float(ev[0]),max_eigenvalue=float(ev[-1]),hvp_autograd_match=True))
            if name=='sine_4pi' and step==30000:
                E=float(np.sqrt(np.mean(e*e)));V=float(c@c+theta[-1]**2);eta=.001;S=170.
                delta=float(np.sin(np.min(np.diff(np.arctan(x)))/2));edge=float(np.max(np.abs(x)));D=E*math.sqrt(eta)
                Phi=np.max(np.abs((a[None,:]-x[:,None]*b[None,:])/np.sqrt(1+x[:,None]**2)),axis=0)
                rate=.5*np.mean(y*y)+eta*(W+1)*E*E
                Z=8*math.sqrt(2)*delta*math.exp(2*math.sqrt(2)*D)*E*S*math.sqrt(V+rate*S)
                ceiling=math.sqrt(1+edge*edge)*np.log(np.exp(2*delta*Phi)+Z)/(2*delta)
                saturation=dict(delta=delta,prefactor=math.sqrt(1+edge*edge)/(2*delta),start=30000,end=200000,
                                maximum_slope_bound=float(ceiling.max()),rms_slope_bound=float(np.sqrt(np.mean(ceiling**2))),
                                quill_reference=18.375)
    return rows,saturation

def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    counter=counterexample();hessian,saturation=hessian_and_saturation()
    result={'counterexample_checks':counter,'local_hessian_checks':hessian,'saturation_bound_evaluation':saturation}
    (OUT/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
