"""Audited sample-fitted QUILL recipe. No derivatives or exterior labels."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.expK06_hard_analytic_suite.targets import NAMES,target
from experiments.expK06_hard_analytic_suite.common import OUT,NS,save,quadrature,training_grid
from docs.qi_practical_guide.reference import fit_qi,MLP

def main():
    rows=[];all_candidates=[]
    for n in NS:
        x=training_grid(n);y=np.column_stack([target(name,x) for name in NAMES])
        folder=OUT/'quill'/f'N{n}';folder.mkdir(parents=True,exist_ok=True)
        # One multi-output solve per geometry: same feature matrix, independent readouts.
        candidates=[]
        for lam in [.2,.25,.3]:
            p=folder/f'lambda{lam}.npz';j=p.with_suffix('.json')
            if j.exists():model=MLP.load(p);info=json.loads(j.read_text())
            else:
                start=time.monotonic()
                model,info=fit_qi(x,y,directions=np.ones((1,1)),intervals=n,lam=lam,rcond=1e-14,solver='gelsd')
                model.save(p);info['seconds']=time.monotonic()-start;save(j,info)
            errors={}
            for order in [32,64,128]:
                xx,w=quadrature(order);true=np.column_stack([target(name,xx) for name in NAMES]);pred=model(xx)
                errors[str(order)]=np.sqrt(np.sum(w[:,None]*(pred-true)**2,axis=0)/np.sum(w[:,None]*true**2,axis=0))
            for k,name in enumerate(NAMES):
                row=dict(target=name,method='QUILL',N=n,width=len(model.weight),parameters=3*len(model.weight)+1,
                         lambda_value=lam,halo=info['halo_per_side'],validation=float(errors['32'][k]),
                         relative_l2=float(errors['128'][k]),quadrature={q:float(e[k]) for q,e in errors.items()},
                         model=str(p.relative_to(ROOT)),output_index=k,training_samples=len(x),rank=info['rank'],rcond=1e-14)
                candidates.append(row)
        all_candidates+=candidates
        for name in NAMES:
            row=min((r for r in candidates if r['target']==name),key=lambda r:r['validation'])
            rows.append(row);print(name,row['width'],f"{row['relative_l2']:.3g}",flush=True)
        save(OUT/'quill_candidates.json',all_candidates);save(OUT/'quill_summary.json',rows)
if __name__=='__main__':main()
