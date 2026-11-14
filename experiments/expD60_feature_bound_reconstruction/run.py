"""Reconstruct the supplied PDF's 705-wide, 25k--125k joint-GD check.

The PDF does not specify its initialization or all 23 target definitions.
This controlled reconstruction uses explicit historical Glorot draws and
three declared targets, plus a matched PyTorch-default control. It is not
claimed to reproduce the original 32/46 count or its exact RNG sequence.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import argparse,json,hashlib,math,multiprocessing,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD56_scale_acquisition.model import gd_chunk,diagnostics,initial_parameters,midpoint_grid,target
from experiments.expD56_scale_acquisition.run import write_json,utc_now,save_checkpoint,config_digest
from experiments.expD60_feature_bound_reconstruction.bound import evaluate

HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/checkpoint_D_optimizers/expD60_feature_bound_reconstruction'
CONFIG={'width':705,'interior_intervals':512,'train_points':2048,'eval_points':8192,
        'learning_rate':.002,'steps':125000,'restart':25000,'seed':30,'log_every':500,'workers':4,
        'activation':'tanh','optimizer':'plain_full_batch_gd','loss':'half_MSE','target_normalization':'training_RMS'}
CASES=[('glorot','legendre_5'),('glorot','localized_oscillation'),('glorot','square'),('pytorch_default','legendre_5')]


def initialize(family,cfg):
    W=cfg['width'];seed=cfg['seed'];N=cfg['interior_intervals']
    if family=='pytorch_default':return initial_parameters(W,seed,1.)
    # Matches the documented C05/D24 draw convention, with the paper's width.
    radius=math.sqrt(6/(W+1));rng=np.random.default_rng([seed,N])
    a=rng.uniform(-radius,radius,W);b=rng.uniform(-radius,radius,W)
    c=np.random.default_rng([seed,N,24]).uniform(-radius,radius,W)
    return torch.from_numpy(np.r_[a,b,c,0.])


def analyze(directory,cfg):
    with np.load(directory/'checkpoint.npz') as d:
        h={k:d['history'][:,i] for i,k in enumerate(d['keys'].tolist())}
    if h['step'][-1]<=cfg['restart']:return
    x=midpoint_grid(cfg['train_points']);reference=.25*cfg['interior_intervals']/2
    results=[evaluate(h,x,cfg['width'],cfg['learning_rate'],reference,cfg['restart'],family)
             for family in ['generic','projected','refined']]
    write_json(directory/'bounds.json',results)
    print(directory.name,'BOUND',[(r['family'],r['below_reference'],r['end_ceiling'],r['end_error_floor']) for r in results],flush=True)


def train(case,cfg,output):
    family,name=case;torch.set_num_threads(1)
    directory=Path(output)/f'{family}__{name}';directory.mkdir(parents=True,exist_ok=True)
    x,xe=midpoint_grid(cfg['train_points']),midpoint_grid(cfg['eval_points'])
    y=target(name,x);normalizer=np.sqrt(np.mean(y*y));y=y/normalizer;ye=target(name,xe)/normalizer
    tx,ty=torch.from_numpy(x),torch.from_numpy(y)
    theta=initialize(family,cfg);digest=config_digest({**cfg,'family':family,'target':name})
    if not (directory/'initial.npz').exists():np.savez(directory/'initial.npz',theta=theta.numpy(),x=x,y=y,normalizer=normalizer)
    history=[];snapshots=[];step=0
    if (directory/'checkpoint.npz').exists():
        with np.load(directory/'checkpoint.npz') as d:
            assert d['config_sha256'].item()==digest
            theta=torch.from_numpy(d['theta'].copy());step=int(d['step'])
            history=[dict(zip(d['keys'].tolist(),row.tolist())) for row in d['history']]
            snapshots=list(d['snapshots'].copy())
    def record():
        history.append({'step':step,**diagnostics(theta.numpy(),x,y,xe,ye,2/cfg['interior_intervals'])})
        snapshots.append(theta.numpy().copy())
    if not history:record()
    start=time.monotonic();initial_step=step
    def checkpoint():
        save_checkpoint(directory,theta,history,snapshots,step,digest)
        write_json(directory/'status.json',{'state':'complete' if step==cfg['steps'] else 'running',
            'family':family,'target':name,'step':step,'width':cfg['width'],'learning_rate':cfg['learning_rate'],
            'gamma_rms':history[-1]['gamma_rms'],'energy':history[-1]['parameter_energy'],
            'error':history[-1]['eval_rel_l2'],'updated_utc':utc_now(),
            'steps_per_second':(step-initial_step)/max(time.monotonic()-start,1e-9)})
    checkpoint()
    endpoints=sorted({cfg['steps'],*[s for s in [1,2,5,10,20,50,100,200] if s<=cfg['steps']],
                      *range(cfg['log_every'],cfg['steps']+1,cfg['log_every'])})
    for endpoint in endpoints:
        if endpoint<=step:continue
        theta,count,finite=gd_chunk(theta,tx,ty,cfg['learning_rate'],endpoint-step);step+=count
        if not finite:raise RuntimeError(f'Nonfinite update: {case}, {step}')
        record()
        if step%5000==0 or step==cfg['steps']:
            checkpoint()
            print(f'{family}/{name}: {step}/{cfg["steps"]}, gamma={history[-1]["gamma_rms"]:.5g}, '
                  f'energy={history[-1]["parameter_energy"]:.5g}, error={history[-1]["eval_rel_l2"]:.5g}',flush=True)
        if step in [50000,75000,100000,125000]:analyze(directory,cfg)
    return directory.name


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','launch','status','pilot']);a=p.parse_args()
    if a.mode=='status':
        for f in sorted(OUT.glob('*/status.json')):print(f.parent.name,f.read_text())
        return
    if a.mode=='pilot':
        cfg={**CONFIG,'steps':100}
        for case in CASES:train(case,cfg,OUT/'pilot')
        return
    OUT.mkdir(parents=True,exist_ok=True)
    if a.mode=='launch':
        path=OUT/'launcher.json'
        if path.exists():
            try:os.kill(json.loads(path.read_text())['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('Existing reconstruction is running')
        with (OUT/'run.log').open('a') as f:
            proc=subprocess.Popen([sys.executable,str(Path(__file__)),'run'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL,start_new_session=True)
        write_json(path,{'pid':proc.pid,'launched_utc':utc_now()});print(proc.pid);return
    sources=[Path(__file__),HERE/'bound.py',ROOT/'experiments/expD56_scale_acquisition/model.py']
    manifest={'config':CONFIG,'cases':CASES,'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
       'scope':'Controlled reconstruction of the PDF settings; initialization and target subset declared explicitly because the PDF omits the full original specification.'}
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:write_json(OUT/'manifest.json',manifest)
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(train,case,CONFIG,str(OUT)) for case in CASES]
        for job in as_completed(jobs):print('COMPLETE',job.result(),flush=True)
    write_json(OUT/'status.json',{'state':'complete','cases':len(CASES),'updated_utc':utc_now()})


if __name__=='__main__':main()
