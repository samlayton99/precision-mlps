"""Economical matched RealNPU versus complex gated NPU comparison.

Reuse existing 50k Adam checkpoints: four targets, W105, seed0, LR .005.
Complete the same up-to-1000-iteration L-BFGS refinement for both families.
No hyperparameter sweep, new training labels or least-squares readout fitting.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from experiments.expK06_hard_analytic_suite.common import OUT,save
from experiments.expK06_hard_analytic_suite.targets import MAIN,target
from experiments.expK06_hard_analytic_suite import verify,npu_lbfgs
from experiments.expK06_hard_analytic_suite.audit_npu_fresh import audit_build
from experiments.expK06_hard_analytic_suite.train import prediction
from experiments.expK06_hard_analytic_suite.integration import accurate_score

DEST=OUT/'npu_family_probe'

def run(r):
    torch.set_num_threads(1);verify.build=audit_build
    source=ROOT/r['run_directory']/'selected.pt';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    existing=json.loads((OUT/'npu_adam_lbfgs/summary.json').read_text())
    matches=[a for a in existing if a['source_checkpoint_sha256']==digest]
    if matches:
        refined=matches[0];reused=True
    else:
        npu_lbfgs.DEST=DEST/'lbfgs';refined=npu_lbfgs.refine(r);reused=False
    m,a,o=verify.load_row(refined)
    errors={str(q):accurate_score(r['target'],m,a,o,q) for q in [128,256]}
    stable=abs(errors['128']-errors['256'])<=max(2e-13,1e-5*errors['256'])
    if not stable:
        errors['512']=accurate_score(r['target'],m,a,o,512)
        stable=abs(errors['256']-errors['512'])<=max(2e-13,1e-5*errors['512'])
    xx=np.unique(np.r_[np.linspace(-1,1,65537),0.,np.geomspace(1e-18,.1,233),-np.geomspace(1e-18,.1,233)])
    yp=prediction(m,xx,a,o);true=target(r['target'],xx)
    family='RealNPU' if r['variant']=='real_native' else 'Complex gated NPU'
    row=dict(target=r['target'],family=family,width=105,seed=0,learning_rate=.005,
             adam_error=r['verified_test_errors']['256'],refined_error=errors[str(max(map(int,errors)))],
             validation=refined['validation'],max_sampled_absolute_error=float(np.max(abs(yp-true))),
             refined_test_errors=errors,quadrature_stable=stable,refinement_reused=reused,
             run_directory=refined['run_directory'],lbfgs_iterations=refined['lbfgs_iterations'])
    np.savez(DEST/f"{r['target']}__{r['variant']}.npz",x=xx,prediction=yp,truth=true)
    save(DEST/f"{r['target']}__{r['variant']}.json",row)
    return row

if __name__=='__main__':
    DEST.mkdir(exist_ok=True)
    sources=json.loads((OUT/'fairness_audit/fresh_summary.json').read_text())
    chosen=[next(r for r in sources if r['target']==t and r['variant']==v and r['seed']==0)
            for t in MAIN for v in ['native','real_native']]
    for t in MAIN:
        pair=[r for r in chosen if r['target']==t]
        init=[torch.load(ROOT/r['run_directory']/'initial.pt',weights_only=False)['model'] for r in pair]
        assert set(init[0])-set(init[1])=={'power.imag'}
        assert all(torch.equal(v,init[0][k]) for k,v in init[1].items())
    save(DEST/'protocol.json',dict(description=__doc__,matched_shared_initial_parameters=True,
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sources=chosen))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(run,r) for r in chosen]):
            row=f.result();rows.append(row);save(DEST/'summary.json',rows)
            print('completed',len(rows),'/8',row['target'],row['family'],row['refined_error'],flush=True)
    assert all(r['quadrature_stable'] for r in rows)
    lines=['| Function | RealNPU | Complex gated NPU |','|:---|---:|---:|']
    for t in MAIN:
        values={r['family']:r['refined_error'] for r in rows if r['target']==t}
        lines.append(f"| {t} | {values['RealNPU']:.2e} | {values['Complex gated NPU']:.2e} |")
    (DEST/'comparison.md').write_text('\n'.join(lines)+'\n')
