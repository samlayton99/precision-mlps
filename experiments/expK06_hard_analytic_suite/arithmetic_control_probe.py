"""Positive arithmetic controls under the exact matched-family training recipe.

Changes: target functions and physical domain [1e-4,1]. Retained: width105,
seed0, Glorot real powers, zero imaginary powers, learned gates initially .5,
linear output with bias, 2003 uniform training values, target amplitude/mean
normalization, minibatches100, Adam .005 constant through25k then cosine to1e-6
at50k, and up to1000 full-batch L-BFGS iterations. No readout solve.
Validation/test integrate physical dx on logarithmically spaced panels to
resolve the left endpoint. This changes evaluation resolution, not the objective
measure or training distribution. This is not a literal reproduction of the
paper's joint two-input/four-output arithmetic experiment.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from scipy.special import roots_legendre
from experiments.expK06_hard_analytic_suite import train,verify,npu_lbfgs
from experiments.expK06_hard_analytic_suite.audit_npu_fresh import audit_build
from experiments.expK06_hard_analytic_suite.common import OUT,save

DEST=OUT/'arithmetic_control_probe'
POWERS={'inverse':-1.,'inverse_root':-.5,'square_root':.5,'square':2.}
EPS=1e-4

def target(name,x):return np.asarray(x,dtype=float)**POWERS[name]

def grid(n):return np.linspace(EPS,1,max(2003,4*(n+1)))

def quadrature(order=32):
    z,w=roots_legendre(order);edges=np.geomspace(EPS,1,65)
    x=np.concatenate([(a+b)/2+(b-a)*z/2 for a,b in zip(edges[:-1],edges[1:])])
    weights=np.concatenate([(b-a)*w/2 for a,b in zip(edges[:-1],edges[1:])])
    return x,weights

def exact_score(name,m,amp,offset,order=32):
    x,w=quadrature(order)
    return train.score(m,x,target(name,x),w,amp,offset)

def run(c):
    torch.set_num_threads(1)
    train.build=audit_build;verify.build=audit_build
    train.training_grid=grid;train.target=target;train.quadrature=quadrature
    npu_lbfgs.accurate_score=exact_score;npu_lbfgs.DEST=DEST/'lbfgs'
    r=train.run_one(c,50000,'arithmetic_control_probe/adam',None,False)
    refined=npu_lbfgs.refine(r);m,amp,offset=verify.load_row(refined)
    errors={str(q):exact_score(c['target'],m,amp,offset,q) for q in [64,128,256]}
    xx=np.unique(np.r_[np.linspace(EPS,1,32769),np.geomspace(EPS,1,32769)])
    yp=train.prediction(m,xx,amp,offset);true=target(c['target'],xx)
    family='RealNPU' if c['variant']=='real_native' else 'Complex gated NPU'
    row=dict(target=c['target'],power=POWERS[c['target']],family=family,width=105,seed=0,domain=[EPS,1],
             adam_error=r['relative_l2'],refined_error=errors['256'],validation=refined['validation'],
             quadrature_errors=errors,quadrature_stable=abs(errors['128']-errors['256'])<=max(2e-13,1e-5*errors['256']),
             maximum_sampled_absolute_error=float(np.max(abs(yp-true))),
             maximum_sampled_relative_error=float(np.max(abs(yp/true-1))),
             imaginary_maximum=None if m.power.imag is None else float(m.power.imag.detach().abs().max()),
             gate=m.power.gate.detach().tolist(),lbfgs_iterations=refined['lbfgs_iterations'],
             run_directory=refined['run_directory'],adam_run_directory=r['run_directory'])
    np.savez(DEST/f"{c['target']}__{c['variant']}.npz",x=xx,prediction=yp,truth=true)
    save(DEST/f"{c['target']}__{c['variant']}.json",row)
    return row

if __name__=='__main__':
    DEST.mkdir(exist_ok=True)
    configs=[dict(target=t,method='NPU',N=64,width=105,variant=v,seed=0,lr=.005,schedule_steps=50000)
             for t in POWERS for v in ['native','real_native']]
    # Independent analytic checks of quadrature and reference norms.
    checks=[]
    for name,p in POWERS.items():
        x,w=quadrature(32);exponent=2*p+1
        exact=np.log(1/EPS) if exponent==0 else (1-EPS**exponent)/exponent
        measured=float(np.dot(w,target(name,x)**2))
        assert np.isclose(measured,exact,rtol=3e-14,atol=1e-15)
        checks.append(dict(target=name,exact_squared_norm=exact,quadrature_squared_norm=measured))
    save(DEST/'protocol.json',dict(description=__doc__,configs=configs,domain=[EPS,1],reference_checks=checks,
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         training_grid_sha256=hashlib.sha256(grid(64).tobytes()).hexdigest()))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(run,c) for c in configs]):
            row=f.result();rows.append(row);save(DEST/'summary.json',rows)
            print('completed',len(rows),'/8',row['target'],row['family'],row['adam_error'],row['refined_error'],flush=True)
    assert all(r['quadrature_stable'] for r in rows)
    lines=['| Function | RealNPU | Complex gated NPU |','|:---|---:|---:|']
    for t in POWERS:
        values={r['family']:r['refined_error'] for r in rows if r['target']==t}
        lines.append(f"| {t} | {values['RealNPU']:.2e} | {values['Complex gated NPU']:.2e} |")
    (DEST/'comparison.md').write_text('\n'.join(lines)+'\n')
