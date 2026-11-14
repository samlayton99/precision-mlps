"""Bounded diagnostics on saved baselines; never replace the published figures.

Same training values, validation-selected solver/cutoff, independent refined test.
Run the eight selected Adam/NPU networks through readout fitting and up to 200
full-batch L-BFGS iterations. This is extra-compute sensitivity, not a new matched
50k-step comparison.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,copy,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.linalg import lstsq
from experiments.expK06_hard_analytic_suite.common import OUT,save,quadrature
from experiments.expK06_hard_analytic_suite.train import data,score
from experiments.expK06_hard_analytic_suite.targets import target
from experiments.expK06_hard_analytic_suite.verify import load_row
from experiments.expK06_hard_analytic_suite.integration import accurate_score

DEST=OUT/'fairness_audit'

def evaluate(r,m,amp,offset,order=32):
    if r['method']=='NPU':return accurate_score(r['target'],m,amp,offset,order)
    x,w=quadrature(order)
    return score(m,x,target(r['target'],x),w,amp,offset)

def audit(r):
    key=r['target']+'__'+r['method'];folder=DEST/key;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text())
    start=time.monotonic();m,amp,offset=load_row(r);x,y,a,o=data(r)
    assert a==amp and o==offset
    original=copy.deepcopy(m.state_dict());candidates=[]
    best=evaluate(r,m,amp,offset);best_state=copy.deepcopy(original);best_phase='original'
    def retain(phase,**info):
        nonlocal best,best_state,best_phase
        val=evaluate(r,m,amp,offset)
        if not np.isfinite(val):val=None
        candidates.append(dict(phase=phase,validation=val,**info))
        if val is not None and val<best:
            best=val;best_state=copy.deepcopy(m.state_dict());best_phase=phase
    retain('original')
    if hasattr(m,'power'):
        gate=m.power.gate.detach().numpy()
        stats=dict(gates=gate.tolist(),clipped_gates=np.clip(gate,0,1).tolist(),
                   real_range=[float(m.power.real.min()),float(m.power.real.max())],
                   imag_range=[float(m.power.imag.min()),float(m.power.imag.max())])
    else:stats={}
    def fit_readout(prefix):
        state=copy.deepcopy(m.state_dict())
        with torch.no_grad():features=torch.cat([m.features(z) for z in x.split(512)]).numpy()
        design=np.column_stack([features,np.ones(len(x))])
        if not np.isfinite(design).all():return
        for solver in ['gelsy','gelsd','gelss']:
            for cutoff in [1e-14,1e-15,np.finfo(float).eps]:
                m.load_state_dict(state)
                coef,_,rank,_=lstsq(design,y.numpy(),cond=cutoff,lapack_driver=solver)
                with torch.no_grad():
                    m.head.weight.copy_(torch.from_numpy(coef[:-1].T))
                    m.head.bias.copy_(torch.from_numpy(coef[-1]))
                retain(prefix,solver=solver,cutoff=cutoff,rank=int(rank),
                       coefficient_l1=float(np.abs(coef).sum()))
        m.load_state_dict(state)
    fit_readout('readout_only')
    # Start nonlinear refinement from the original saved trained model, not a
    # potentially ill-conditioned fitted readout with very large coefficients.
    m.load_state_dict(original)
    opt=torch.optim.LBFGS(m.parameters(),lr=1,max_iter=5,max_eval=12,history_size=50,
                         tolerance_grad=1e-14,tolerance_change=0.,line_search_fn='strong_wolfe')
    calls=0;status='completed';local_best=evaluate(r,m,amp,offset);local_state=copy.deepcopy(original)
    def closure():
        nonlocal calls
        calls+=1;opt.zero_grad(set_to_none=True);total=0.
        for xx,yy in zip(x.split(512),y.split(512)):
            loss=(m(xx)-yy).square().sum()/len(x)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();total+=float(loss.detach())
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters()):
            raise FloatingPointError('nonfinite gradient')
        return torch.tensor(total,dtype=torch.float64)
    for block in range(40):
        before=copy.deepcopy(m.state_dict())
        try:opt.step(closure)
        except (FloatingPointError,RuntimeError) as exc:
            status=str(exc);m.load_state_dict(before);break
        retain('lbfgs',block=block+1)
        val=candidates[-1]['validation']
        if val is not None and val<local_best:local_best=val;local_state=copy.deepcopy(m.state_dict())
        if all(torch.equal(v,before[k]) for k,v in m.state_dict().items()):status='unchanged';break
    m.load_state_dict(local_state);fit_readout('lbfgs_plus_readout')
    m.load_state_dict(best_state)
    tests={str(q):evaluate(r,m,amp,offset,q) for q in [64,128,256]}
    torch.save(dict(model=best_state,config=r,amplitude=amp,offset=offset,phase=best_phase),folder/'selected.pt')
    source=ROOT/r['run_directory']/'selected.pt'
    row=dict(target=r['target'],method=r['method'],width=r['width'],variant=r['variant'],
             source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             original_test=r['relative_l2'],selected_validation=best,selected_phase=best_phase,
             test_errors=tests,quadrature_stable=abs(tests['128']-tests['256'])<=max(2e-13,1e-5*tests['256']),
             initial_parameter_stats=stats,candidates=candidates,lbfgs_closures=calls,lbfgs_status=status,
             seconds=time.monotonic()-start)
    save(folder/'summary.json',row)
    print(key,'original',row['original_test'],'new',tests['256'],best_phase,flush=True)
    return row

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    DEST.mkdir(parents=True,exist_ok=True)
    save(DEST/'protocol.json',dict(description=__doc__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         solvers=['gelsy','gelsd','gelss'],cutoffs=[1e-14,1e-15,float(np.finfo(float).eps)],
         lbfgs_max_iterations=200,selection='validation only, original model retained as candidate',
         changes_to_existing_figures=False))
    rows=json.loads((OUT/'selected_fits.json').read_text());results=[]
    for r in rows:
        if r['method']=='QUILL':continue
        results.append(audit(r));save(DEST/'refinement_summary.json',results)

if __name__=='__main__':main()
