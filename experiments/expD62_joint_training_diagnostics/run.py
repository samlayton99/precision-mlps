"""24 paired GD cases: four widths, three targets, two rates; reuse matching D56 prefixes."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,multiprocessing,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD56_scale_acquisition.run import train_case,load_config,config_digest,write_json,utc_now,case_id
from experiments.expD56_scale_acquisition.model import initial_parameters

HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/checkpoint_D_optimizers/expD62_joint_training_diagnostics'
WIDTHS=[64,196,384,768]
TARGETS=['square','sine_4pi','runge_100']
RATES=[.001,.0001]

def config(width):
    c=load_config()
    c.update(experiment=HERE.name,width=width,steps=50000,learning_rates=RATES,
             initialization_multipliers=[1.],targets={'selected':TARGETS},workers=4)
    # The shared recorder's legacy lambda field is unused here. This experiment
    # compares native total widths, not constructed geometries or halo layouts.
    return c

def reuse():
    width=196;c=config(width);digest=config_digest(c)
    for name in TARGETS:
        for lr in RATES:
            original=ROOT/'results/checkpoint_D_optimizers/expD56_scale_acquisition/sweep/runs'/case_id(name,1.,lr)
            dest=OUT/f'width{width}'/'runs'/original.name;dest.mkdir(parents=True,exist_ok=True)
            with np.load(original/'initial.npz') as d:
                np.testing.assert_array_equal(d['theta'],initial_parameters(width,c['seed'],1.).numpy())
                assert len(d['train_x'])==c['train_points'] and len(d['eval_x'])==c['eval_points']
            shutil.copy2(original/'initial.npz',dest/'initial.npz')
            with np.load(original/'checkpoint.npz') as d:
                keys=d['keys'].tolist();h=d['history'];step=h[:,keys.index('step')]
                keep=step<=c['steps'];assert step[keep][-1]==c['steps']
                h=h[keep];snap=d['snapshots'][keep]
                np.savez_compressed(dest/'checkpoint.npz',theta=snap[-1],step=c['steps'],keys=np.array(keys),
                                    history=h,snapshots=snap,config_sha256=np.array(digest))
            s=json.loads((original/'status.json').read_text())
            s.update(state='complete',step=c['steps'],total_steps=c['steps'],config_sha256=digest,
                     gamma_rms=float(h[-1,keys.index('gamma_rms')]),eval_rel_l2=float(h[-1,keys.index('eval_rel_l2')]),
                     provenance='Reused exact matched prefix of completed D56 run',source=str(original.relative_to(ROOT)),
                     source_sha256=hashlib.sha256((original/'checkpoint.npz').read_bytes()).hexdigest())
            write_json(dest/'status.json',s)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.pilot:
        for w in [64,768]:
            c=config(w);c.update(steps=500,checkpoint_every=500)
            train_case(('square',1.,.001),c,OUT/'pilot'/f'width{w}')
        return
    sources=[Path(__file__),ROOT/'experiments/expD56_scale_acquisition/model.py',ROOT/'experiments/expD56_scale_acquisition/run.py']
    manifest={'widths':WIDTHS,'functions':TARGETS,'learning_rates':RATES,'steps':50000,'seed':30,
              'initialization':'Default float64 torch Linear initialization; multiplier 1',
              'optimizer':'Native full-batch GD on half MSE; fixed learning rate',
              'training_points':1024,'evaluation_points':4096,'target_normalization':'Unit training RMS, no mean subtraction',
              'total_runs':24,'reused_runs':6,'new_runs':18,'ceiling_restart':1000,
              'width_note':'Total neurons. Native random centers; no designated halos. The shared recorder lambda field is unused.',
              'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}}
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Source or configuration changed since launch'
    else:write_json(mp,manifest)
    reuse()
    cases=[(w,name,lr) for w in [768,384,64] for name in TARGETS for lr in RATES]
    write_json(OUT/'sweep_status.json',{'state':'running','pid':os.getpid(),'started_utc':utc_now()})
    failures=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs={pool.submit(train_case,(name,1.,lr),config(w),OUT/f'width{w}'):(w,name,lr) for w,name,lr in cases}
        for job in as_completed(jobs):
            case=jobs[job]
            try:
                r=job.result()
                if r['state']!='complete':failures.append({'case':case,'state':r['state']})
                print('COMPLETE',case,round(time.monotonic()-start,1),flush=True)
            except Exception as e:
                failures.append({'case':case,'error':repr(e)});print('FAILED',case,repr(e),flush=True)
    write_json(OUT/'sweep_status.json',{'state':'complete' if not failures else 'failed','failures':failures,
                                       'elapsed_seconds':time.monotonic()-start,'updated_utc':utc_now()})
    if failures:raise RuntimeError(failures)

if __name__=='__main__':main()
