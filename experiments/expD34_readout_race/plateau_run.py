"""Continue paired GD/Adam states to test persistence of effective-force plateaus."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from . import adam_forces as af, adam_run as ar
from .run import verify_gpu, write_json


def run(args):
    if not jax.config.x64_enabled: raise ValueError('FP64 is required')
    output=args.output/args.optimizer; output.mkdir(parents=True, exist_ok=True); verify_gpu(output)
    states=[]; cases=[]; hashes={}
    for seed in range(3):
        folder=args.root/'curated'/f'primary_{seed}'
        records=json.loads((folder/'manifest.json').read_text())['cases']
        f=np.load(folder/'snapshots.npz'); pos=int(np.flatnonzero(f['steps']==600000)[0])
        hashes[str(folder/'snapshots.npz')]=hashlib.sha256((folder/'snapshots.npz').read_bytes()).hexdigest()
        for i,case in enumerate(records):
            if case['optimizer']!=args.optimizer: continue
            cases.append(case); states.append({k:f[k][i,pos] for k in f.files if k!='steps'})
    state=jax.tree.map(lambda *v:jnp.asarray(np.stack(v)), *states)
    settings=jnp.asarray([[c[k] for k in ('eta','beta1','beta2','epsilon','adaptive')] for c in cases])
    yy=jnp.asarray(np.stack([af.data(c['target'])[1] for c in cases]))
    manifest=dict(cases=cases,input_hashes=hashes,source_commit=os.environ.get('RACE_SOURCE_COMMIT'),
        initial_step=600000,final_step=args.end_step,metrics=ar.METRICS,samples=2048,
        trace_convention='pre-update row at interval end minus one; interval extrema retained')
    if (output/'manifest.json').exists() and json.loads((output/'manifest.json').read_text())!=json.loads(json.dumps(manifest)):
        raise ValueError('Changed continuation protocol')
    write_json(output/'manifest.json',manifest)
    step=600000; snapshots={step:jax.device_get(state)}; starts=[]; ends=[]; rows=[]; lows=[]; highs=[]
    if (output/'state.npz').exists():
        f=dict(np.load(output/'state.npz')); step=int(f.pop('cursor')); state=jax.tree.map(jnp.asarray,f)
        f=dict(np.load(output/'snapshots.npz')); ss=f.pop('steps')
        snapshots={int(s):{k:v[:,j] for k,v in f.items()} for j,s in enumerate(ss) if s<=step}
        f=np.load(output/'trace.npz'); keep=f['ends']<=step
        starts=list(f['starts'][keep]); ends=list(f['ends'][keep])
        rows=list(f['values'][:,keep].transpose(1,0,2)); lows=list(f['minimum'][:,keep].transpose(1,0,2)); highs=list(f['maximum'][:,keep].transpose(1,0,2))
    advance=ar.advance_factory(2048); begun=time.monotonic()
    def save():
        host=jax.device_get(state); snapshots[step]=host; ss=sorted(snapshots)
        ar.atomic_npz(output/'snapshots.npz',steps=np.array(ss),**{k:np.stack([snapshots[s][k] for s in ss],axis=1) for k in host})
        ar.atomic_npz(output/'trace.npz',starts=np.array(starts),ends=np.array(ends),values=np.stack(rows,axis=1),minimum=np.stack(lows,axis=1),maximum=np.stack(highs,axis=1))
        ar.atomic_npz(output/'state.npz',cursor=np.array(step),**host)
        write_json(output/'status.json',dict(cursor=step,complete=step==args.end_step,failed=host['failed'].tolist(),
            unresolved=host['unresolved_steps'].tolist(),identity_max=host['identity_max'].max(axis=0).tolist(),seconds=time.monotonic()-begun))
    for end in range(step+10000,args.end_step+1,10000):
        state,row,lo,hi=jax.device_get(advance(state,yy,settings,end-step))
        starts.append(step);ends.append(end);rows.append(row);lows.append(lo);highs.append(hi);step=end
        if step%100000==0 or step==args.end_step or time.monotonic()-begun>args.max_seconds:
            save();print(json.dumps(dict(optimizer=args.optimizer,step=step,seconds=time.monotonic()-begun)),flush=True)
        if time.monotonic()-begun>args.max_seconds: break


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--optimizer',choices=('gd','adam'),required=True)
    parser.add_argument('--end-step',type=int,default=6000000); parser.add_argument('--max-seconds',type=float,default=3300)
    run(parser.parse_args())
