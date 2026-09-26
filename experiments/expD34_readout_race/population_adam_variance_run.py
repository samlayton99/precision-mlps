"""Paired tracking-variance and gain-only pulses after a shared native burn-in."""
import argparse
import csv
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import population_adam_variance as variance
from .population_adam_run import data,error,scalar,settings,write_csv,digest,SIX


def run(args):
    if jax.default_backend()!='gpu':raise RuntimeError('Modal GPU required')
    begun=time.monotonic();rows=[];runs=[];sources=[];times=[];cases=[]
    for folder in sorted(args.inputs.glob('adam_population_wide*')):
        with (folder/'cases.csv').open() as f:
            for c in csv.DictReader(f):
                if c['optimizer']!='adam':continue
                for k in ('width','seed','case_id'):c[k]=int(c[k])
                for k in ('eta','beta1','beta2','epsilon','h'):c[k]=float(c[k])
                c['adaptive']=True;cases.append((c,folder))
    cases.sort(key=lambda c:(SIX.index(c[0]['target']),c[0]['seed'],c[0]['width']))
    assert len(cases)==24
    def flush():
        write_csv(args.output/'states.csv',rows);write_csv(args.output/'runs.csv',runs)
        (args.output/'facts.json').write_text(json.dumps(dict(cases_expected=24,runs=runs,
            seconds=time.monotonic()-begun,sources=sources),indent=2)+'\n')
    for index,(c,folder) in enumerate(cases):
        if times and time.monotonic()-begun+1.3*max(times)>args.seconds:break
        started=time.monotonic();x,y=map(jnp.asarray,data(c['target']));xe,ye=map(jnp.asarray,data(c['target'],8192));config=settings(c)
        path=folder/f"case{c['case_id']}_state125000.npz"
        with np.load(path,allow_pickle=False) as z:saved={k:jnp.asarray(z[k]) for k in ('p','m','v','cm','count')}
        sources.append(dict(path=str(path),sha256=digest(path)))
        burn=variance.advance(variance.initialize(saved,x,y,config),x,y,config,0,5000)
        fork=variance.initialize(burn,x,y,config);fork['alternative_v']=burn['alternative_v']
        np.savez_compressed(args.output/f'case{index}_fork130000.npz',**jax.device_get({k:fork[k] for k in ('p','m','v','cm','count','alternative_v')}))
        stopped=False
        for arm,name in enumerate(variance.ARMS):
            state=fork;runtime=time.monotonic();status='complete';base=dict(target=c['target'],width=c['width'],seed=c['seed'],arm=name,age=130000,h=c['h'])
            for offset in range(0,20001,1000):
                if offset:state=variance.advance(state,x,y,config,arm if offset<=10000 else 0,1000)
                d=scalar(variance.diagnostics(state,x,y,config))
                row=dict(**base,offset=offset,step=130000+offset,**d,lambda_rms=c['h']*d['slope_rms'])
                if offset%10000==0:row['relative_eval_error']=float(error(state['p'],xe,ye))
                rows.append(row)
                if not all(np.isfinite(d[k]) for k in ('M','C6','relative_error')):status='nonfinite';break
                if time.monotonic()-begun>args.seconds-20 and offset<20000:stopped=True;status='budget_stopped';break
            result=dict(**base,end_offset=offset,complete=offset==20000 and status=='complete',status=status,seconds=time.monotonic()-runtime,
                norm_error=d['norm_error'],component_error=d['component_error'],unresolved=d['unresolved'],zero_candidate=d['zero_candidate'])
            runs.append(result);flush()
            np.savez_compressed(args.output/f'case{index}_{name}_end.npz',**jax.device_get({k:state[k] for k in ('p','m','v','cm','count','alternative_v')}))
            print(json.dumps(dict(**result,elapsed=time.monotonic()-begun)),flush=True)
            if stopped:break
        times.append(time.monotonic()-started)
        if stopped:break
    flush()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,required=True);run(parser.parse_args())
