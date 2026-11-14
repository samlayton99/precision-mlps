"""NPU comparison: saved 50k Adam models followed by full-batch L-BFGS.

Select a starting model by validation at each of the six existing widths.
Include the audited reference-rate complex NPU candidates at W=105. No
RealNPU, readout refitting, target-specific initialization, or extra samples.
Keep the original model if nonlinear refinement does not improve validation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,copy,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from experiments.expK06_hard_analytic_suite.common import OUT,WIDTHS,save
from experiments.expK06_hard_analytic_suite.targets import MAIN
from experiments.expK06_hard_analytic_suite.train import data,identifier
from experiments.expK06_hard_analytic_suite.verify import load_row
from experiments.expK06_hard_analytic_suite.integration import accurate_score

DEST=OUT/'npu_adam_lbfgs'

def refine(r):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    folder=DEST/identifier(r);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text())
    started=time.monotonic();m,amp,offset=load_row(r);x,y,a,o=data(r);assert a==amp and o==offset
    source=ROOT/r['run_directory']/'selected.pt'
    best=accurate_score(r['target'],m,amp,offset,32);original=best
    best_state=copy.deepcopy(m.state_dict());best_iteration=0;calls=0;history=[]
    opt=torch.optim.LBFGS(m.parameters(),lr=1,max_iter=20,max_eval=30,history_size=50,
                         tolerance_grad=1e-14,tolerance_change=0.,line_search_fn='strong_wolfe')
    first=0;status='iteration_budget';no_improvement=0
    checkpoint=folder/'progress.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,weights_only=False)
        m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        best=state['best'];best_state=state['best_state'];best_iteration=state['best_iteration']
        calls=state['calls'];history=state['history'];first=state['next_block'];no_improvement=state['no_improvement']
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
    for block in range(first,50):
        before=copy.deepcopy(m.state_dict())
        try:opt.step(closure)
        except (FloatingPointError,RuntimeError) as exc:
            status=str(exc);m.load_state_dict(before);break
        iteration=opt.state[next(iter(m.parameters()))].get('n_iter',0)
        val=accurate_score(r['target'],m,amp,offset,32)
        history.append(dict(iteration=iteration,validation=float(val) if np.isfinite(val) else None,closures=calls))
        if np.isfinite(val) and val<best:
            no_improvement=0;best=val;best_state=copy.deepcopy(m.state_dict());best_iteration=iteration
        else:no_improvement+=1
        state=dict(model=m.state_dict(),optimizer=opt.state_dict(),best=best,best_state=best_state,
                   best_iteration=best_iteration,calls=calls,history=history,next_block=block+1,no_improvement=no_improvement)
        torch.save(state,checkpoint);save(folder/'history.json',history)
        if block%10==9:print('progress',r['target'],r['width'],iteration,best,flush=True)
        if all(torch.equal(v,before[k]) for k,v in m.state_dict().items()):status='unchanged_parameters';break
        if no_improvement>=25:status='500_iterations_without_validation_improvement';break
    iterations=opt.state[next(iter(m.parameters()))].get('n_iter',0)
    m.load_state_dict(best_state)
    errors={str(q):accurate_score(r['target'],m,amp,offset,q) for q in [128,256]}
    stable=abs(errors['128']-errors['256'])<=max(2e-13,1e-5*errors['256'])
    if not stable:
        errors['512']=accurate_score(r['target'],m,amp,offset,512)
        stable=abs(errors['256']-errors['512'])<=max(2e-13,1e-5*errors['512'])
    phase='Adam+LBFGS' if best_iteration else 'Adam'
    torch.save(dict(model=best_state,config=r,amplitude=amp,offset=offset,phase=phase),folder/'selected.pt')
    row={**r,'source_run_directory':r['run_directory'],'run_directory':str(folder.relative_to(ROOT)),
         'source_checkpoint_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
         'validation':best,'relative_l2':errors[str(max(map(int,errors)))],
         'original_adam_validation':original,'selected_phase':phase,'lbfgs_iterations':iterations,
         'best_lbfgs_iteration':best_iteration,'lbfgs_closures':calls,'lbfgs_status':status,
         'independent_refined_errors':errors,'quadrature_stable':stable,'steps':50000,
         'readout_refit':False,'refinement_seconds':time.monotonic()-started}
    save(folder/'summary.json',row)
    print('done',r['target'],r['width'],row['relative_l2'],phase,flush=True)
    return row

if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True)
    sources=json.loads((OUT/'verified_summary.json').read_text())
    sources+=[r for r in json.loads((OUT/'fairness_audit/fresh_summary.json').read_text()) if r['variant']=='native']
    chosen=[min((r for r in sources if r['target']==t and r['method']=='NPU' and r['width']==w),
                key=lambda r:r['validation']) for t in MAIN for w in WIDTHS]
    save(DEST/'protocol.json',dict(description=__doc__,adam_steps=50000,lbfgs_max_iterations=1000,
         lbfgs_lr=1,history_size=50,line_search='strong_wolfe',batch='full training grid, chunks of 512',
         selection='validation at each width, both starting checkpoint and L-BFGS checkpoints',
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),starting_models=chosen))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(refine,r) for r in chosen]):
            rows.append(f.result());save(DEST/'summary.json',rows)
            print('completed',len(rows),'/24',flush=True)
