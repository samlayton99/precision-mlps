"""One native scalar inverse-root diagnostic, not a general-function baseline.

Same samples, loss normalization, Adam schedule/seed and L-BFGS as the W105
control. Differences are explicit: one real power, fixed relevance gate1,
no learned output mixture, exponent initialized at0 (not the correct -.5).
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from src.models.neural_power_unit import NeuralPowerUnit
from experiments.expK06_hard_analytic_suite import train,verify,npu_lbfgs
from experiments.expK06_hard_analytic_suite.common import OUT,save
from experiments.expK06_hard_analytic_suite.arithmetic_control_probe import grid,target,quadrature,exact_score
from experiments.expK06_hard_analytic_suite.audit_npu_fresh import audit_build

DEST=OUT/'arithmetic_precision_diagnosis'
RAW=target('inverse_root',grid(64));AMP=float(abs(RAW).max());OFFSET=float((RAW/AMP).mean())

class ScalarPower(torch.nn.Module):
    def __init__(self):
        super().__init__();self.variant='native'
        self.power=NeuralPowerUnit(1,1,complex_weights=False,generator=torch.Generator().manual_seed(0))
        with torch.no_grad():self.power.real.zero_();self.power.gate.fill_(1.)
        self.power.gate.requires_grad_(False)
    def forward(self,x):return self.power(x)/AMP-OFFSET

def builder(method,width,variant,seed):return ScalarPower()

if __name__=='__main__':
    torch.set_num_threads(1);DEST.mkdir(exist_ok=True)
    controls=json.loads((OUT/'arithmetic_control_probe/summary.json').read_text())
    r=next(r for r in controls if r['target']=='inverse_root' and r['family']=='RealNPU')
    state=torch.load(ROOT/r['run_directory']/'selected.pt',weights_only=False)
    m=audit_build('NPU',105,'real_native',0);m.load_state_dict(state['model'])
    with torch.no_grad():features=m.features(torch.from_numpy(grid(64)[:,None])).numpy()
    design=np.column_stack([features,np.ones(len(features))]);s=np.linalg.svd(design,compute_uv=False)
    diag=dict(width105_error=r['refined_error'],smallest_learned_power=float(m.power.real.detach().min()),
              target_power=-.5,glorot_initial_halfwidth=math.sqrt(6/106),
              design_columns=106,design_rank_at_1e14_cutoff=int(np.sum(s>s[0]*1e-14)),
              design_rank_at_machine_epsilon=int(np.sum(s>s[0]*np.finfo(float).eps)),
              singular_value_ratio=float(s[0]/s[-1]),power_quantiles=np.quantile(m.power.real.detach().numpy(),[0,.25,.5,.75,1]).tolist())
    save(DEST/'wide_model_diagnosis.json',diag)
    train.build=builder;verify.build=builder;train.target=target;train.training_grid=grid;train.quadrature=quadrature
    npu_lbfgs.accurate_score=exact_score;npu_lbfgs.DEST=DEST/'lbfgs'
    c=dict(target='inverse_root',method='NPU',N=64,width=1,variant='scalar_fixed_gate_zero',seed=0,lr=.005,schedule_steps=50000)
    save(DEST/'protocol.json',dict(description=__doc__,config=c,initial_exponent=0.,fixed_gate=1.,
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    row=train.run_one(c,50000,'arithmetic_precision_diagnosis/adam',None,False)
    refined=npu_lbfgs.refine(row);model,amp,offset=verify.load_row(refined)
    errors={str(q):exact_score('inverse_root',model,amp,offset,q) for q in [64,128,256]}
    result=dict(wide_diagnosis=diag,scalar_adam_error=row['relative_l2'],scalar_after_lbfgs=errors,
                learned_exponent=model.power.real.item(),imaginary_parameters=False,
                learned_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                retained_checkpoint=refined['run_directory'],general_function_baseline=False,
                scalar_trainable_exponent_initialized_to_answer=False)
    save(DEST/'result.json',result);print(result,flush=True)
