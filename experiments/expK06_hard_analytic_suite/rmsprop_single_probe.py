"""One optimizer-only probe: complex gated NPU, Kepler, W105, seed1.

Match the existing Adam run's data, initialization, LR .005 and 50k schedule.
RMSProp uses Flux 0.10.4 defaults rho=.9 and epsilon=1e-8. Follow with the
same up-to-1000-iteration L-BFGS refinement. No sweep or readout refitting.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from experiments.expK06_hard_analytic_suite import train,npu_lbfgs
from experiments.expK06_hard_analytic_suite.accurate_full import full_run
from experiments.expK06_hard_analytic_suite.common import OUT,save
from experiments.expK06_hard_analytic_suite.verify import load_row
from experiments.expK06_hard_analytic_suite.targets import target
from experiments.expK06_hard_analytic_suite.integration import zero_crossings

DEST=OUT/'rmsprop_single_probe'

def main():
    torch.set_num_threads(1);DEST.mkdir(exist_ok=True)
    config=dict(target='kepler',method='NPU',N=64,width=105,variant='native',seed=1,lr=.005,
                schedule_steps=50000,optimizer='RMSProp',rmsprop_alpha=.9,rmsprop_eps=1e-8)
    save(DEST/'protocol.json',dict(description=__doc__,config=config,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference='https://raw.githubusercontent.com/FluxML/Flux.jl/v0.10.4/src/optimise/optimisers.jl'))
    # Reuse the exact frozen training loop; substitute only its optimizer factory
    # in this isolated process and restore it before nonlinear refinement.
    original=torch.optim.Adam
    torch.optim.Adam=lambda params,lr,foreach=False:torch.optim.RMSprop(params,lr=lr,alpha=.9,eps=1e-8,
                                                                      momentum=0,centered=False,foreach=foreach)
    try:r=full_run(config,50000,'rmsprop_single_probe/training',None,False)
    finally:torch.optim.Adam=original
    r['selected_phase']='RMSProp';r['rmsprop_validation']=r.pop('adam_validation',r['validation'])
    folder=ROOT/r['run_directory']
    if (folder/'adam_best.pt').exists():shutil.move(folder/'adam_best.pt',folder/'rmsprop_best.pt')
    state=torch.load(folder/'selected.pt',weights_only=False);state['phase']='RMSProp';torch.save(state,folder/'selected.pt')
    save(folder/'summary.json',r)
    prior=next(a for a in json.loads((OUT/'fairness_audit/fresh_summary.json').read_text())
               if a['target']=='kepler' and a['variant']=='native' and a['seed']==1)
    initial=torch.load(folder/'initial.pt',weights_only=False)
    reference=torch.load(ROOT/prior['run_directory']/'initial.pt',weights_only=False)
    assert all(torch.equal(v,reference['model'][k]) for k,v in initial['model'].items())
    npu_lbfgs.DEST=DEST/'lbfgs'
    refined=npu_lbfgs.refine(r)
    refined['selected_phase']='RMSProp+LBFGS' if refined['best_lbfgs_iteration'] else 'RMSProp'
    refined['original_rmsprop_validation']=refined.pop('original_adam_validation',r['validation'])
    state=torch.load(ROOT/refined['run_directory']/'selected.pt',weights_only=False)
    state['phase']=refined['selected_phase'];torch.save(state,ROOT/refined['run_directory']/'selected.pt')
    save(ROOT/refined['run_directory']/'summary.json',refined)
    m,a,o=load_row(refined)
    xx=np.unique(np.r_[np.linspace(-1,1,65537),0.,np.geomspace(1e-18,.1,233),-np.geomspace(1e-18,.1,233)])
    yp=train.prediction(m,xx,a,o);yy=target('kepler',xx)
    comparison=next(a for a in json.loads((OUT/'npu_adam_lbfgs/summary.json').read_text())
                    if a['target']=='kepler' and a['width']==105)
    result=dict(config=config,identical_initialization=True,
         rmsprop_test=r['relative_l2'],rmsprop_then_lbfgs_test=refined['relative_l2'],
         adam_test=prior['verified_test_errors']['256'],adam_then_lbfgs_test=comparison['relative_l2'],
         rmsprop_max_sampled_absolute_error=float(np.max(abs(yp-yy))),quadrature_stable=refined['quadrature_stable'],
         refined_model=refined,no_figure_changes=True)
    np.savez(DEST/'predictions.npz',x=xx,prediction=yp,truth=yy)
    save(DEST/'result.json',result)
    print({k:v for k,v in result.items() if k not in ['config','refined_model']},flush=True)

if __name__=='__main__':main()
