"""Reference-rate sensitivity: NPU/RealNPU, W=105, seeds 0/1, 50k steps.

This bounded audit adds the released simple-task rate .005 and omitted RealNPU.
All original data, validation, schedule and numerical precision are retained.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from experiments.expK06_hard_analytic_suite import train
from experiments.expK06_hard_analytic_suite.models import build
from experiments.expK06_hard_analytic_suite.accurate_full import full_run
from experiments.expK06_hard_analytic_suite.integration import accurate_score
from experiments.expK06_hard_analytic_suite.common import OUT,save
from experiments.expK06_hard_analytic_suite.targets import MAIN

def audit_build(method,width,variant,seed):
    m=build(method,width,'native',seed)
    if variant=='real_native':m.power.register_parameter('imag',None)
    return m

def run(c):
    original=train.build;train.build=audit_build
    try:
        row=full_run(c,50000,'fairness_audit/fresh_50k',None,False)
        state=torch.load(ROOT/row['run_directory']/'selected.pt',weights_only=False)
        m=audit_build(c['method'],c['width'],c['variant'],c['seed']);m.load_state_dict(state['model'])
        errors={str(q):accurate_score(c['target'],m,state['amplitude'],state['offset'],q) for q in [128,256]}
        row['verified_test_errors']=errors
        row['quadrature_stable']=abs(errors['128']-errors['256'])<=max(2e-13,1e-5*errors['256'])
        save(ROOT/row['run_directory']/'audit_verified.json',row)
        return row
    finally:train.build=original

if __name__=='__main__':
    configs=[dict(target=t,method='NPU',N=64,width=105,variant=v,seed=s,lr=.005,schedule_steps=50000)
             for t in MAIN for v in ['native','real_native'] for s in [0,1]]
    save(OUT/'fairness_audit/fresh_protocol.json',dict(description=__doc__,configs=configs))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(run,c) for c in configs]):
            row=f.result();rows.append(row)
            save(OUT/'fairness_audit/fresh_summary.json',rows)
            print(len(rows),'/16',row['target'],row['variant'],row['seed'],row['verified_test_errors']['256'],flush=True)
