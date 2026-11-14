"""Bounded zero-crossing fix: positive affine input, learned NPU gate.

Four functions, W=105, seed 0, 50k Adam at .005 then up to 1k L-BFGS.
No new samples or labels, no least-squares readout, no published-figure changes.
Input u=1.5+0.5x stays in [1,2]. Both real and imaginary powers use Glorot
uniform initialization; nonzero imaginary initialization avoids an invariant
zero-gradient imaginary branch on exclusively positive inputs.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from experiments.expK06_hard_analytic_suite import train,verify,npu_lbfgs
from experiments.expK06_hard_analytic_suite.models import build
from experiments.expK06_hard_analytic_suite.accurate_full import full_run
from experiments.expK06_hard_analytic_suite.common import OUT,save
from experiments.expK06_hard_analytic_suite.targets import MAIN,target

DEST=OUT/'npu_spike_probe'

def positive_build(method,width,variant,seed):
    if variant!='positive_learned':return build(method,width,variant,seed)
    m=build('NPU',width,'positive',seed)
    with torch.no_grad():m.power.gate.fill_(.5)
    m.power.gate.requires_grad_(True)
    return m

def run(c):
    train.build=positive_build;verify.build=positive_build;npu_lbfgs.DEST=DEST/'lbfgs'
    r=full_run(c,50000,'npu_spike_probe/adam',None,False)
    refined=npu_lbfgs.refine(r)
    m,a,o=verify.load_row(refined)
    xx=np.unique(np.r_[np.linspace(-1,1,8193),0.,np.geomspace(1e-18,1e-2,200),-np.geomspace(1e-18,1e-2,200)])
    pred=train.prediction(m,xx,a,o);true=target(c['target'],xx)
    refined['probes']=dict(x0_prediction=float(train.prediction(m,np.array([0.]),a,o)[0]),
        x0_truth=float(target(c['target'],np.array([0.]))[0]),
        maximum_sampled_absolute_error=float(np.max(abs(pred-true))),
        gate=m.power.gate.detach().tolist(),finite=bool(np.isfinite(pred).all()))
    np.savez(DEST/f"{c['target']}_predictions.npz",x=xx,prediction=pred,truth=true)
    return refined

if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True)
    configs=[dict(target=t,method='NPU',N=64,width=105,variant='positive_learned',seed=0,lr=.005,schedule_steps=50000) for t in MAIN]
    save(DEST/'protocol.json',dict(description=__doc__,configs=configs,
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(run,c) for c in configs]):
            row=f.result();rows.append(row);save(DEST/'summary.json',rows)
            print('finished',row['target'],row['relative_l2'],row['probes'],flush=True)
