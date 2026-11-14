"""Identical pivoted-QR readout opportunity for all 48 saved 50k baselines."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,copy,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.linalg import lstsq
from experiments.expK06_hard_analytic_suite.common import OUT,save
from experiments.expK06_hard_analytic_suite.train import data
from experiments.expK06_hard_analytic_suite.verify import load_row
from experiments.expK06_hard_analytic_suite.audit_npu_fairness import evaluate

def run(r):
    torch.set_num_threads(1)
    key=f"{r['target']}__{r['method']}__W{r['width']}"
    folder=OUT/'fairness_audit/readout_sweep'/key;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text())
    m,amp,offset=load_row(r);x,y,a,o=data(r);assert a==amp and o==offset
    original=copy.deepcopy(m.state_dict());best=evaluate(r,m,amp,offset);best_state=copy.deepcopy(original)
    phase='original';chosen=None;candidates=[]
    with torch.no_grad():features=torch.cat([m.features(z) for z in x.split(512)]).numpy()
    design=np.column_stack([features,np.ones(len(x))])
    for cutoff in [1e-14,np.finfo(float).eps]:
        coef,_,rank,_=lstsq(design,y.numpy(),cond=cutoff,lapack_driver='gelsy')
        m.load_state_dict(original)
        with torch.no_grad():
            m.head.weight.copy_(torch.from_numpy(coef[:-1].T));m.head.bias.copy_(torch.from_numpy(coef[-1]))
        val=evaluate(r,m,amp,offset)
        candidates.append(dict(cutoff=cutoff,rank=int(rank),validation=val,
                               coefficient_l1=float(np.abs(coef).sum())))
        if np.isfinite(val) and val<best:
            best=val;best_state=copy.deepcopy(m.state_dict());phase='readout_QR';chosen=cutoff
    m.load_state_dict(best_state)
    errors={str(q):evaluate(r,m,amp,offset,q) for q in [128,256]}
    source=ROOT/r['run_directory']/'selected.pt'
    row=dict(target=r['target'],method=r['method'],width=r['width'],variant=r['variant'],
             source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             original_test=r['relative_l2'],validation=best,test_errors=errors,selected_phase=phase,cutoff=chosen,
             quadrature_stable=abs(errors['128']-errors['256'])<=max(2e-13,1e-5*errors['256']),candidates=candidates)
    torch.save(dict(model=best_state,config=r,amplitude=amp,offset=offset,phase=phase),folder/'selected.pt')
    save(folder/'summary.json',row);return row

if __name__=='__main__':
    rows=[r for r in json.loads((OUT/'verified_summary.json').read_text()) if r['method']!='QUILL']
    assert len(rows)==48
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,r) for r in rows]):
            row=f.result();results.append(row);save(OUT/'fairness_audit/readout_sweep_summary.json',results)
            print(len(results),'/48',row['target'],row['method'],row['width'],row['test_errors']['256'],flush=True)
