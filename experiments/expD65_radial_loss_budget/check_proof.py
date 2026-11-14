"""Independent autograd spot checks and randomized checks of the telescoping constants."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(k,'1')
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD65_radial_loss_budget.audit import diagnostic,OUT,SOURCES

def main():
    torch.set_num_threads(1);checks=[]
    cases=[(SOURCES[0],'chirp__init1__lr1e-03',200000),(SOURCES[1],'airy_transition__init1__lr1e-03',30000),
           (SOURCES[0],'softplus_5__init4__lr1e-03',200000),(SOURCES[0],'legendre_12__init1__lr1e-04',30000)]
    for root,name,step in cases:
        folder=root/name
        with np.load(folder/'initial.npz') as z:x=z['train_x'];y=z['train_y']
        with np.load(folder/'checkpoint.npz') as z:
            steps=z['history'][:,z['keys'].tolist().index('step')];theta=z['snapshots'][np.flatnonzero(steps==step)[0]].copy()
        lr=json.loads((folder/'status.json').read_text())['learning_rate'];v=diagnostic(theta,x,y,lr)
        t=torch.tensor(theta,requires_grad=True);tx=torch.tensor(x);ty=torch.tensor(y);W=(len(theta)-1)//3
        e=torch.tanh(tx[:,None]*t[:W]+t[W:2*W])@t[2*W:3*W]+t[-1]-ty
        L=.5*torch.mean(e*e);g,=torch.autograd.grad(L,t)
        E=torch.sqrt(2*L);g2=g@g;K=E*torch.clamp(-(t[:W]@g[:W]),min=0)/g2
        np.testing.assert_allclose([E.item(),g2.item(),K.item()],[v[0],v[3],v[4]],rtol=1e-8,atol=1e-11)
        checks.append({'case':name,'step':step,'K':K.item(),'autograd_match':True})
    rng=np.random.default_rng(65);worst=-float('inf')
    for _ in range(10000):
        theta=rng.normal(size=17);g=rng.normal(size=17);a=theta[:7];ga=g[:7];g2=g@g
        eta=.001;E=np.sqrt(1+eta*g2);Enext=np.sqrt(E*E-eta*g2)
        K=E*max(0,-a@ga)/g2
        change=(a-eta*ga)@(a-eta*ga)-a@a
        budget=4*K*(E-Enext)+eta*(E*E-Enext*Enext)
        worst=max(worst,float(change-budget));assert change<=budget+1e-12
    report={'autograd_checks':checks,'randomized_exact_descent_cases':10000,'max_increment_minus_budget':worst}
    (OUT/'proof_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
