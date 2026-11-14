"""Matched Adam counterpart of D62. Original GD recurrence is only a diagnostic comparator."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
from pathlib import Path
import sys,json,hashlib,time,argparse,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD56_scale_acquisition.model import initial_parameters,midpoint_grid,target,diagnostics
from experiments.expD56_scale_acquisition.run import write_json,utc_now,case_id
from experiments.expD64_adam_joint_training_diagnostics.model import train_chunk
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/checkpoint_D_optimizers'/HERE.name
WIDTHS=[64,196,384,768];TARGETS=['square','sine_4pi','runge_100'];RATES=[.001,.0001]

def run_case(case,root,steps=50000):
    torch.set_num_threads(1)
    w,name,lr=case
    folder=Path(root)/f'width{w}'/'runs'/case_id(name,1.,lr);folder.mkdir(parents=True,exist_ok=True)
    x=midpoint_grid(1024);xe=midpoint_grid(4096);norm=np.sqrt(np.mean(target(name,x)**2))
    y=target(name,x)/norm;ye=target(name,xe)/norm;tx=torch.from_numpy(x);ty=torch.from_numpy(y)
    theta=initial_parameters(w,30,1.);m=torch.zeros_like(theta);v=m.clone()
    config={'width':w,'target':name,'learning_rate':lr,'steps':steps,'seed':30,'optimizer':'Adam',
            'beta1':.9,'beta2':.999,'epsilon':1e-8,'weight_decay':0,'train_points':1024,'eval_points':4096}
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if not (folder/'initial.npz').exists():
        np.savez_compressed(folder/'initial.npz',theta=theta.numpy(),train_x=x,train_y=y,eval_x=xe,eval_y=ye,target_rms=norm)
    checkpoint=folder/'checkpoint.npz';history=[];snapshots=[];step=0
    if checkpoint.exists():
        with np.load(checkpoint) as d:
            assert d['config_sha256'].item()==digest
            theta=torch.from_numpy(d['theta'].copy());m=torch.from_numpy(d['adam_first'].copy());v=torch.from_numpy(d['adam_second'].copy())
            step=int(d['step']);keys=d['keys'].tolist();history=[dict(zip(keys,row)) for row in d['history']];snapshots=list(d['snapshots'].copy())
    started=time.monotonic();start_step=step
    def record():
        history.append({'step':step,**diagnostics(theta.numpy(),x,y,xe,ye,2/147)})
        snapshots.append(theta.numpy().copy())
    def save(state):
        keys=list(history[0]);temp=folder/'checkpoint.tmp.npz'
        np.savez_compressed(temp,theta=theta.numpy(),adam_first=m.numpy(),adam_second=v.numpy(),step=step,
                            keys=np.array(keys),history=np.array([[r[k] for k in keys] for r in history]),
                            snapshots=np.array(snapshots),config_sha256=np.array(digest))
        temp.replace(checkpoint)
        status={**config,'initialization_multiplier':1.,'config_sha256':digest,'state':state,'step':step,
                'total_steps':steps,'gamma_rms':history[-1]['gamma_rms'],'eval_rel_l2':history[-1]['eval_rel_l2'],
                'updated_utc':utc_now(),'steps_per_second':(step-start_step)/max(time.monotonic()-started,1e-6)}
        write_json(folder/'status.json',status)
        return status
    if not history:record()
    save('complete' if step==steps else 'running')
    times=sorted({steps,*[n for n in (1,2,5,10,20,50,100,200) if n<=steps],*range(500,steps+1,500)})
    for endpoint in times:
        if endpoint<=step:continue
        theta,m,v,n,finite=train_chunk(theta,m,v,tx,ty,lr,endpoint-step,step);step+=n
        if not finite:
            save('nonfinite');raise RuntimeError(f'Nonfinite Adam update {case} at {step}')
        record()
        if step%5000==0 or step==steps:
            result=save('complete' if step==steps else 'running')
            print(case,step,'gamma',round(result['gamma_rms'],5),'error',round(result['eval_rel_l2'],6),flush=True)
    return save('complete')

def verify():
    torch.set_num_threads(1);base=initial_parameters(64,30,1.);x=torch.from_numpy(midpoint_grid(37));y=torch.sin(4*torch.pi*x)
    ref=base.clone().requires_grad_(True);opt=torch.optim.Adam([ref],lr=.001,foreach=False)
    for _ in range(23):
        opt.zero_grad();w=64
        pred=torch.tanh(x[:,None]*ref[:w]+ref[w:2*w])@ref[2*w:3*w]+ref[-1]
        (.5*((pred-y)**2).mean()).backward();opt.step()
    zero=torch.zeros_like(base)
    actual,m,v,n,finite=train_chunk(base,zero,zero,x,y,.001,23,0)
    first,m1,v1,_,_=train_chunk(base,zero,zero,x,y,.001,7,0)
    resumed,m2,v2,_,_=train_chunk(first,m1,v1,x,y,.001,16,7)
    np.testing.assert_allclose(actual.numpy(),ref.detach().numpy(),atol=1e-13,rtol=1e-13)
    assert torch.equal(actual,resumed) and torch.equal(m,m2) and torch.equal(v,v2) and finite
    print(json.dumps({'torch_adam_max_parameter_error':float((actual-ref.detach()).abs().max()),'resume_exact':True}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.verify:verify();return
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),HERE/'model.py',ROOT/'experiments/expD56_scale_acquisition/model.py']
    manifest={'optimizer':'Adam','beta1':.9,'beta2':.999,'epsilon':1e-8,'weight_decay':0,'activation':'tanh',
              'widths':WIDTHS,'targets':TARGETS,'learning_rates':RATES,'steps':50000,'seed':30,'runs':24,
              'initialization':'PyTorch Linear defaults, identical to D62','loss':'Half mean squared error',
              'training_points':1024,'evaluation_points':4096,'normalization':'Unit training RMS',
              'bound_note':'F.6 is a GD-only comparator, not an Adam guarantee. F and R remain raw gradient components.',
              'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    path=OUT/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:write_json(path,manifest)
    write_json(OUT/'sweep_status.json',{'state':'running','pid':os.getpid(),'started_utc':utc_now()})
    failures=[];completed=0
    cases=[(w,n,lr) for w in WIDTHS[::-1] for n in TARGETS for lr in RATES]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs={pool.submit(run_case,c,OUT):c for c in cases}
        for job in as_completed(jobs):
            try:job.result();completed+=1
            except Exception as exc:failures.append({'case':jobs[job],'error':repr(exc)})
            write_json(OUT/'sweep_status.json',{'state':'running','completed':completed,'failures':failures,'updated_utc':utc_now()})
    write_json(OUT/'sweep_status.json',{'state':'complete' if not failures else 'failed','completed':completed,'failures':failures,'updated_utc':utc_now()})
    if failures:raise RuntimeError(failures)
if __name__=='__main__':main()
