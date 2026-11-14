"""High-precision reference spot checks of each selected QUILL model."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import mpmath as mp
from experiments.expK06_hard_analytic_suite.common import *
from experiments.expK06_hard_analytic_suite.targets import *
from experiments.expK06_hard_analytic_suite.verify import load_row
from experiments.expK06_hard_analytic_suite.train import prediction

def main():
    torch.set_num_threads(1)
    rows=json.loads((OUT/'verified_summary.json').read_text());checks=[]
    x=np.unique(np.r_[-1.,0.,1.,np.random.default_rng(81493).uniform(-1,1,64)])
    with mp.workdps(70):
        for name in NAMES:
            r=min((r for r in rows if r['target']==name and r['method']=='QUILL'),key=lambda r:r['validation'])
            m,amp,offset=load_row(r);yp=prediction(m,x,amp,offset)
            true=np.array([float(mp_target(name,mp.mpf(float(xx)),mp)) for xx in x])
            xq,wq=quadrature(128);rms=np.sqrt(np.dot(wq,target(name,xq)**2)/2)
            checks.append(dict(target=name,width=r['width'],points=len(x),reference_decimal_digits=70,
                               normalized_rms_spot_error=float(np.sqrt(np.mean((yp-true)**2))/rms),
                               maximum_normalized_spot_error=float(np.max(abs(yp-true))/rms),
                               reference_float64_maximum_difference=float(np.max(abs(target(name,x)-true))/rms)))
            print(name,checks[-1]['normalized_rms_spot_error'],flush=True)
    step_checks=[]
    for r in json.loads((OUT/'main50k_summary.json').read_text()):
        state=torch.load(ROOT/r['run_directory']/'last.pt',weights_only=False)
        assert state['step']==50000 and abs(state['optimizer']['param_groups'][0]['lr']-1e-6)<1e-20
        step_checks.append(dict(target=r['target'],method=r['method'],width=r['width'],steps=state['step'],final_lr=state['optimizer']['param_groups'][0]['lr']))
    save(OUT/'reference_spot_audit.json',dict(quill=checks,training_steps=step_checks,
          note='Spot RMS is a reference check, not a second estimate or certificate of continuous L2.'))
if __name__=='__main__':main()
