"""Small validation-guided tuning of the same uniform tanh/least-squares recipe."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time,math,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from docs.qi_practical_guide.reference import fit_qi,MLP
from experiments.expK06_hard_analytic_suite.common import *
from experiments.expK06_hard_analytic_suite.targets import NAMES,target,MAIN

TUNE=OUT/'quill_tuned'

@torch.no_grad()
def predictions(model,x):
    w=torch.from_numpy(model.weight);b=torch.from_numpy(model.hidden_bias)
    v=torch.from_numpy(model.readout.T.copy());c=torch.from_numpy(model.output_bias)
    rows=[]
    for part in torch.from_numpy(x[:,None]).split(512):
        features=torch.nn.functional.linear(part,w,b).tanh()
        # Score each ordinary scalar network separately (same reduction as export).
        rows.append(torch.cat([torch.nn.functional.linear(features,v[k:k+1],c[k:k+1]) for k in range(len(c))],dim=1).numpy())
    return np.concatenate(rows)


def fit(base_n,width,r,lam,solver,cutoff):
    construction_n=width-2*r-1
    if construction_n<16:return []
    ident=f'W{width}_R{r}_l{lam:.3f}_{solver}_rc{cutoff:.0e}'
    folder=TUNE/ident;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'summary.json'
    if path.exists():return json.loads(path.read_text())
    x=training_grid(base_n);y=np.column_stack([target(name,x) for name in NAMES])
    start=time.monotonic()
    m,info=fit_qi(x,y,directions=np.ones((1,1)),intervals=construction_n,halo=r,lam=lam,rcond=cutoff,solver=solver)
    m.save(folder/'model.npz')
    xx,ww=quadrature(32);true=np.column_stack([target(name,xx) for name in NAMES]);pred=predictions(m,xx)
    errors=np.sqrt(np.sum(ww[:,None]*(pred-true)**2,axis=0)/np.sum(ww[:,None]*true**2,axis=0))
    rows=[dict(target=name,method='QUILL',N=base_n,construction_N=construction_n,width=width,
               parameters=3*width+1,lambda_value=lam,halo=r,solver=solver,rcond=cutoff,rank=info['rank'],
               validation=float(errors[k]),model=str((folder/'model.npz').relative_to(ROOT)),output_index=k,
               training_samples=len(x),seconds=time.monotonic()-start) for k,name in enumerate(NAMES)]
    save(path,rows);save(folder/'diagnostics.json',info)
    print(ident,'main',*[f'{errors[NAMES.index(name)]:.2g}' for name in MAIN],flush=True)
    return rows


def collect():
    rows=[r for p in TUNE.glob('*/summary.json') for r in json.loads(p.read_text())]
    chosen=[]
    for width in sorted(set(r['width'] for r in rows)):
        for name in NAMES:
            chosen.append(min((r for r in rows if r['target']==name and r['width']==width),key=lambda r:r['validation']))
    save(OUT/'quill_tuned_candidates.json',rows);save(OUT/'quill_tuned_selected.json',chosen)
    return chosen


def main():
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['probe','sweep','extend']);args=p.parse_args()
    eps=np.finfo(float).eps
    if args.stage=='probe':
        n=512;width=559;r=halo(n)
        for lam in [.2,.225,.25,.275,.3]:
            for rc in [1e-14,1e-15,eps]:fit(n,width,r,lam,'gelsy',rc)
        for solver in ['gelsd','gelss']:
            for rc in [1e-15,eps]:fit(n,width,r,.25,solver,rc)
        for r in [20,32,46]:fit(n,width,r,.25,'gelsy',eps)
    elif args.stage=='sweep':
        # Narrow follow-up: QR + SVD at the lower cutoffs, within audited bandwidth range.
        for n,width in zip(NS,WIDTHS):
            r=halo(n)
            for lam in [.2,.225,.25,.275,.3]:fit(n,width,r,lam,'gelsy',eps)
            for lam in [.2,.25,.3]:fit(n,width,r,lam,'gelsy',1e-15)
            fit(n,width,r,.25,'gelsd',eps)
            if r!=20:fit(n,width,20,.25,'gelsy',eps)
            collect()
    else:
        n=4096;width=n+2*halo(n)+1
        for lam in [.225,.25,.275]:
            rows=fit(n,width,halo(n),lam,'gelsy',eps)
            if next(r['validation'] for r in rows if r['target']=='elliptic_e')<1e-14:break
    selected=collect()
    print('Selected main best:',{name:min(r['validation'] for r in selected if r['target']==name) for name in MAIN},flush=True)
if __name__=='__main__':main()
