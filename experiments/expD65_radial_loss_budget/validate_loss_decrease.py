"""Independent curvature-integral and 40-digit checks of cancellation-safe loss differences."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'1')
import sys,json,math,time
from pathlib import Path
import numpy as np
import torch
import mpmath as mp
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD65_radial_loss_budget.check_loss_decrease import native_step,stable_drop,OUT
BASE=ROOT/'results/checkpoint_D_optimizers/expD56_scale_acquisition/sweep/runs'

def load(name,step):
    p=BASE/name
    with np.load(p/'initial.npz') as z:x=z['train_x'];y=z['train_y']
    with np.load(p/'checkpoint.npz') as z:
        steps=z['history'][:,z['keys'].tolist().index('step')];theta=z['snapshots'][np.flatnonzero(steps==step)[0]].copy()
    lr=json.loads((p/'status.json').read_text())['learning_rate']
    nxt,g,h,e=native_step(torch.from_numpy(theta),torch.from_numpy(x),torch.from_numpy(y),lr)
    nn=nxt.numpy();gg=g.numpy();denom=lr*math.fsum((gg*gg).tolist())
    ratio=stable_drop(theta,nn,h.numpy().T,e.numpy(),x)/denom
    return theta,nn,gg,x,y,denom,ratio

def curvature_ratio(theta,nxt,g,x,y,denom,order):
    delta=nxt-theta;W=(len(theta)-1)//3;va,vb,vc=np.split(delta[:-1],3)
    nodes,weights=np.polynomial.legendre.leggauss(order);total=0.
    for u,weight in zip((nodes+1)/2,weights/2):
        cur=theta+u*delta;a,b,c=np.split(cur[:-1],3);h=np.tanh(x[:,None]*a+b);hp=1-h*h
        e=h@c+cur[-1]-y;du=x[:,None]*va+vb
        dq=(hp*du)@c+h@vc+delta[-1]
        d2q=(-2*h*hp*du*du)@c+2*(hp*du)@vc
        curv=math.fsum((dq*dq+e*d2q).tolist())/len(x)
        total+=weight*(1-u)*curv
    return (-math.fsum((g*delta).tolist())-total)/denom

def multiprecision_ratio(theta,nxt,x,y,denom):
    with mp.workdps(40):
        W=(len(theta)-1)//3
        p=[mp.mpf(float(v)) for v in theta];pn=[mp.mpf(float(v)) for v in nxt]
        total=mp.mpf(0)
        for xx,yy in zip(x,y):
            xx=mp.mpf(float(xx));yy=mp.mpf(float(yy))
            q=p[-1]+mp.fsum(p[2*W+j]*mp.tanh(p[j]*xx+p[W+j]) for j in range(W))
            qn=pn[-1]+mp.fsum(pn[2*W+j]*mp.tanh(pn[j]*xx+pn[W+j]) for j in range(W))
            total+=((q-yy)**2-(qn-yy)**2)/2
        return float(total/len(x)/mp.mpf(denom))

def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True);rows=[]
    cases=[('legendre_12__init1__lr1e-04',199500),('localized_oscillation__init1__lr1e-03',199500),
           ('square__init4__lr1e-03',0),('softplus_5__init1__lr1e-03',199500)]
    for name,step in cases:
        theta,nxt,g,x,y,denom,stable=load(name,step)
        c3=curvature_ratio(theta,nxt,g,x,y,denom,3);c5=curvature_ratio(theta,nxt,g,x,y,denom,5)
        np.testing.assert_allclose([c3,c5],stable,rtol=2e-7,atol=2e-8)
        rows.append({'case':name,'step':step,'stable_c':stable,'curvature_3point':c3,'curvature_5point':c5,'denominator':denom})
        print(rows[-1],flush=True)
    name,step=cases[0];theta,nxt,g,x,y,denom,stable=load(name,step)
    start=time.monotonic();precise=multiprecision_ratio(theta,nxt,x,y,denom)
    np.testing.assert_allclose(precise,stable,rtol=2e-7,atol=2e-8)
    report={'curvature_checks':rows,'multiprecision_check':{'case':name,'step':step,'digits':40,'stable_c':stable,
              'high_precision_c':precise,'absolute_difference':abs(stable-precise),'seconds':time.monotonic()-start}}
    (OUT/'numerical_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['multiprecision_check'],indent=2))
if __name__=='__main__':main()
