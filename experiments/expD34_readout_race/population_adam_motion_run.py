"""Bounded continuation matrix for Adam population motion; execute on Modal GPU."""
import argparse
import csv
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import population_adam_motion as motion
from .population_adam_run import data, error, scalar, settings, write_csv, digest, SIX


def run(args):
    if jax.default_backend()!='gpu':raise RuntimeError('Modal GPU required')
    begun=time.monotonic(); rows=[]; crossed=[]; runs=[]; sources=[]; case_times=[]
    cases=[]
    for folder in sorted(args.inputs.glob('adam_population_wide*')):
        with (folder/'cases.csv').open() as f:
            for index,case in enumerate(csv.DictReader(f)):
                if case['optimizer']!='adam':continue
                case.update(width=int(case['width']),seed=int(case['seed']),case_id=index)
                for key in ('eta','beta1','beta2','epsilon','h'):case[key]=float(case[key])
                case['adaptive']=True
                cases.append((case,folder))
    cases.sort(key=lambda c:(SIX.index(c[0]['target']),c[0]['seed'],c[0]['width']))
    assert len(cases)==24

    def flush():
        write_csv(args.output/'states.csv',rows);write_csv(args.output/'crossed.csv',crossed)
        write_csv(args.output/'runs.csv',runs)
        (args.output/'facts.json').write_text(json.dumps(dict(seconds=time.monotonic()-begun,
            cases_expected=24,runs_completed=sum(r['complete'] for r in runs),runs=runs,
            sources=sources),indent=2)+'\n')

    for ci,(case,folder) in enumerate(cases):
        if case_times and time.monotonic()-begun+1.3*max(case_times)>args.seconds:break
        case_started=time.monotonic();x,y=map(jnp.asarray,data(case['target']))
        xe,ye=map(jnp.asarray,data(case['target'],8192));config=settings(case)
        starts={}
        for age in (25000,125000):
            path=folder/f"case{case['case_id']}_state{age}.npz"
            with np.load(path,allow_pickle=False) as z:
                saved={k:jnp.asarray(z[k]) for k in ('p','m','v','cm','count')}
            starts[age]=motion.initialize(saved,x,y,config)
            sources.append(dict(path=str(path),sha256=digest(path)))
        stopped=False
        for age in (25000,125000):
            for arm in range(4):
                duration=100000 if age==25000 and arm==0 else 20000
                state=starts[age];base=dict(case_index=ci,target=case['target'],width=case['width'],
                    seed=case['seed'],age=age,arm=motion.ARMS[arm],h=float(case['h']))
                runtime=time.monotonic();prior=state;offset=0
                for offset in range(0,duration+1,1000):
                    if offset:
                        policy=arm if offset<=10000 else 0
                        state=motion.advance(state,x,y,config,policy,1000)
                        if arm==0:
                            cr=scalar(motion.crossed_forces(prior,state,x,y,config))
                            crossed.append(dict(**base,offset=offset,**cr))
                        prior=state
                    d=scalar(motion.diagnostics(state,x,y,config))
                    row=dict(**base,offset=offset,step=age+offset,**d,
                        lambda_rms=float(case['h'])*d['slope_rms'])
                    if offset%10000==0:row['relative_eval_error']=float(error(state['p'],xe,ye))
                    rows.append(row)
                    if not all(np.isfinite(d[k]) for k in ('M','C6','relative_error')):
                        raise RuntimeError(f'Nonfinite state: {base}, {offset}')
                    if time.monotonic()-begun>args.seconds-20 and offset<duration:
                        stopped=True;break
                result=dict(**base,end_offset=offset,complete=offset==duration,
                    seconds=time.monotonic()-runtime,component_error=d['component_error'],
                    norm_error=d['norm_error'],unresolved=d['unresolved'],zero_candidate=d['zero_candidate'])
                if arm==0 and age==25000 and offset==100000:
                    ref=starts[125000]['p']
                    result['archive_relative_difference']=float(jnp.linalg.norm(state['p']-ref)/jnp.linalg.norm(ref))
                runs.append(result)
                np.savez_compressed(args.output/f'case{ci}_age{age}_{motion.ARMS[arm]}_end.npz',
                    **jax.device_get({k:state[k] for k in ('p','m','v','cm','count')}))
                print(json.dumps(dict(**result,elapsed=time.monotonic()-begun)),flush=True)
                flush()
                if stopped:break
            if stopped:break
        case_times.append(time.monotonic()-case_started)
        if stopped:break
    flush()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,required=True)
    run(parser.parse_args())
