"""One-width affine-first extension of the paper's NAU -> NPU arrangement.

Linear(1,105,bias=True) -> gated NPU(105,1), real and complex versions.
Four targets, seed0, Xavier weights/zero affine biases, gates .5, imaginary
weights zero. Same samples, output normalization, Adam .005/50k schedule,
and up to1000 full-batch L-BFGS iterations as the matched-family probe.
This biased scalar adaptation is not an exact reproduction of the paper.
No additional nonlinear features, readout fitting or parameter sweep.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from src.models.neural_power_unit import NeuralPowerUnit
from experiments.expK06_hard_analytic_suite import train,verify,npu_lbfgs
from experiments.expK06_hard_analytic_suite.common import OUT,save
from experiments.expK06_hard_analytic_suite.targets import MAIN,target
from experiments.expK06_hard_analytic_suite.accurate_full import full_run
from experiments.expK06_hard_analytic_suite.integration import accurate_score,zero_crossings

DEST=OUT/'npu_affine_first_probe'

class AffineFirst(torch.nn.Module):
    def __init__(self,width,complex_weights,seed):
        super().__init__();self.variant='affine_first'
        g=torch.Generator().manual_seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.linear=torch.nn.Linear(1,width,dtype=torch.float64)
        torch.nn.init.xavier_uniform_(self.linear.weight,generator=g)
        torch.nn.init.zeros_(self.linear.bias)
        self.power=NeuralPowerUnit(width,1,complex_weights=complex_weights,generator=g)
    @property
    def mix(self):return torch.cat((self.linear.weight,self.linear.bias[:,None]),dim=1)
    def forward(self,x):return self.power(self.linear(x))

def build(method,width,variant,seed):return AffineFirst(width,variant=='affine_complex',seed)

def run(c):
    torch.set_num_threads(1);train.build=build;verify.build=build
    npu_lbfgs.DEST=DEST/'lbfgs'
    r=full_run(c,50000,'npu_affine_first_probe/adam',None,False)
    refined=npu_lbfgs.refine(r);m,a,o=verify.load_row(refined)
    roots=zero_crossings(m)
    probes=[np.linspace(-1,1,32769),np.random.default_rng(8319).uniform(-1,1,8192)]
    for root in roots:
        delta=np.geomspace(1e-18,.01,128)
        probes.extend([np.array([root]),root-delta,root+delta])
    xx=np.unique(np.clip(np.concatenate(probes),-1,1))
    yp=train.prediction(m,xx,a,o);true=target(c['target'],xx)
    row=dict(target=c['target'],family='Complex gated NPU' if c['variant']=='affine_complex' else 'RealNPU',
         architecture='Linear(1,105,bias=True) -> NPU(105,1)',width=105,seed=0,
         parameters=sum(p.numel() for p in m.parameters()),adam_steps=r['steps'],
         adam_status=r['status'],adam_error=r['relative_l2'],refined_error=refined['relative_l2'],
         validation=refined['validation'],quadrature_stable=refined['quadrature_stable'],
         independent_refined_errors=refined['independent_refined_errors'],
         maximum_sampled_absolute_error=float(np.max(abs(yp-true))),zero_crossings=len(roots),
         run_directory=refined['run_directory'],lbfgs_iterations=refined['lbfgs_iterations'])
    np.savez(DEST/f"{c['target']}__{c['variant']}.npz",x=xx,prediction=yp,truth=true)
    save(DEST/f"{c['target']}__{c['variant']}.json",row)
    return row

def checks():
    # Positive intermediate inputs: known product of shifted powers, and input
    # derivative check. These weights are only an implementation check, never
    # used to initialize a training run.
    for complex_weights in [False,True]:
        m=AffineFirst(2,complex_weights,0)
        with torch.no_grad():
            m.linear.weight.copy_(torch.tensor([[2.],[.4]],dtype=torch.float64))
            m.linear.bias.copy_(torch.tensor([3.,2.],dtype=torch.float64))
            m.power.real.copy_(torch.tensor([[.5,-1.]],dtype=torch.float64));m.power.gate.fill_(1.)
        x=torch.tensor([[-.5],[.2],[.7]],dtype=torch.float64,requires_grad=True)
        eps=torch.finfo(torch.float64).eps
        expected=(2*x+3+eps).sqrt()/(.4*x+2+eps)
        torch.testing.assert_close(m(x),expected,rtol=3e-15,atol=1e-15)
        assert torch.autograd.gradcheck(m,(x,))
    real=AffineFirst(105,False,0).state_dict();complex_state=AffineFirst(105,True,0).state_dict()
    assert set(complex_state)-set(real)=={'power.imag'}
    assert all(torch.equal(v,complex_state[k]) for k,v in real.items())

if __name__=='__main__':
    checks();DEST.mkdir(exist_ok=True)
    configs=[dict(target=t,method='NPU',N=64,width=105,variant=v,seed=0,lr=.005,schedule_steps=50000)
             for t in MAIN for v in ['affine_real','affine_complex']]
    save(DEST/'protocol.json',dict(description=__doc__,configs=configs,forward_and_gradient_checks=True,
         shared_initial_parameters_match=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(run,c) for c in configs]):
            row=f.result();rows.append(row);save(DEST/'summary.json',rows)
            print('completed',len(rows),'/8',row['target'],row['family'],row['refined_error'],
                  'quadrature_stable',row['quadrature_stable'],flush=True)
    lines=['| Function | RealNPU, affine first | Complex NPU, affine first |','|:---|---:|---:|']
    for t in MAIN:
        values={r['family']:r['refined_error'] for r in rows if r['target']==t}
        lines.append(f"| {t} | {values['RealNPU']:.2e} | {values['Complex gated NPU']:.2e} |")
    (DEST/'comparison.md').write_text('\n'.join(lines)+'\n')
