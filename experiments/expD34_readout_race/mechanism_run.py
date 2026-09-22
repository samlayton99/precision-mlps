"""Paired parameter-block continuations from verified actual D34 states."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import core, targets
from .mechanism import ARMS
from .recovery import clean


@lru_cache(maxsize=8)
def advance_factory(m, eta):
    x=jnp.asarray(targets.grid(m)); powers=x[:,None]**jnp.arange(11)
    def one(state,y,rates,count):
        def step(_,old):
            z,d=old['z'],old['d']
            loss,_,g,gd,_=core.tanh_field(z,d,x,y,powers)
            zn=z-eta*rates[:3,None]*g; dn=d-eta*rates[3]*gd
            delta=jnp.abs(zn[0])-jnp.abs(z[0])
            return dict(z=zn,d=dn,positive=old['positive']+jnp.maximum(delta,0),
                negative=old['negative']+jnp.maximum(-delta,0),
                path=old['path']+eta*rates[0]*jnp.linalg.norm(g[0]),
                energy=old['energy']+eta*jnp.r_[rates[:3]*jnp.sum(g*g,axis=1),rates[3]*gd*gd],
                crossing=old['crossing']+jnp.mean(delta+eta*rates[0]*jnp.sign(z[0])*g[0]),
                previous_loss=loss,loss_increases=old['loss_increases']+(loss>old['previous_loss']+1e-14))
        return jax.lax.fori_loop(0,count,step,state)
    return jax.jit(jax.vmap(one,in_axes=(0,0,0,None)))


def initial(z,d):
    z,d=jnp.asarray(z),jnp.asarray(d)
    return dict(z=z,d=d,positive=jnp.zeros_like(z[:,0]),negative=jnp.zeros_like(z[:,0]),
        path=jnp.zeros_like(d),energy=jnp.zeros((len(d),4)),crossing=jnp.zeros_like(d),
        previous_loss=jnp.full_like(d,jnp.inf),loss_increases=jnp.zeros(len(d),dtype=jnp.int64))


def inputs(archives,seed,fork,arms,eta):
    source={}; hashes=[]
    for path in archives:
        f=dict(np.load(path)); index=int(np.flatnonzero(f['steps']==fork)[0])
        for i,case in enumerate(json.loads(str(f['cases']))):
            if isinstance(case,list): case=dict(seed=case[0],target=case[1])
            if case['seed']==seed:
                source[case['target']]=(f['z'][i,index],f['d'][i,index],f['y'][i])
        hashes.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    cases=[]; zz=[]; dd=[]; yy=[]; rr=[]
    for target in targets.TARGETS:
        z,d,y=source[target]
        for arm in arms:
            cases.append(dict(seed=seed,target=target,arm=arm,fork_step=fork,eta=.002,training_eta=eta))
            zz.append(z); dd.append(d); yy.append(y); rr.append(ARMS[arm])
    return cases,np.stack(zz),np.array(dd),np.stack(yy),np.array(rr),hashes


def run(args):
    if not jax.config.x64_enabled: raise ValueError('Set JAX_ENABLE_X64=true')
    args.output.mkdir(parents=True,exist_ok=True)
    from .run import verify_gpu
    verify_gpu(args.output)
    cases,z,d,y,rates,hashes=inputs(args.archives,args.seed,args.fork_step,args.arms,args.eta)
    factor=round(.002/args.eta)
    if factor<1 or not np.isclose(factor*args.eta,.002,rtol=0,atol=1e-15):
        raise ValueError('Step must divide the reference physical step 0.002')
    manifest=dict(cases=cases,source_archives=hashes,reference_eta=.002,training_eta=args.eta,
        step_convention='reference GD updates; physical time = step * 0.002',
        source_commit=os.environ.get('RACE_SOURCE_COMMIT'),source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    archive=args.output/'compact_states.npz'; mp=args.output/'manifest.json'
    if archive.exists():
        if json.loads(mp.read_text())!=manifest: raise ValueError('Resume configuration changed')
        f=dict(np.load(archive)); snapshots={int(s):{k:f[k][:,j] for k in initial(z,d)} for j,s in enumerate(f['steps'])}
        step=max(snapshots); state=jax.tree.map(jnp.asarray,snapshots[step])
    else:
        mp.write_text(json.dumps(manifest,indent=2)+'\n')
        state=initial(z,d); step=args.fork_step; snapshots={step:jax.device_get(state)}
    offsets=(0,1,2,5,10,20,50,100,200,500,1000)
    schedule=sorted({args.fork_step+k for k in offsets}|set(range(args.fork_step,args.end_step+1,2000))|
                    {args.end_step,20000,100000,600000})
    schedule=[s for s in schedule if step<s<=args.end_step]
    x=targets.grid(len(y[0])); begin=time.monotonic(); start_step=step
    advance=advance_factory(len(x),args.eta)
    def save(complete):
        steps=sorted(snapshots)
        arrays={key:np.stack([snapshots[s][key] for s in steps],axis=1) for key in state}
        tmp=archive.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,steps=np.array(steps),cases=np.array(json.dumps(cases)),x=x,y=y,**arrays)
        tmp.replace(archive)
        status=dict(complete=complete,step=step,requested_end=args.end_step,seconds=time.monotonic()-begin,
            advanced_reference_updates=step-start_step,cases=len(cases),max_loss_increases=int(np.max(arrays['loss_increases'])),
            motion_identity_error=float(np.max(abs(arrays['positive'][:,-1]-arrays['negative'][:,-1]-(abs(arrays['z'][:,-1,0])-abs(arrays['z'][:,0,0]))))))
        (args.output/'status.json').write_text(json.dumps(clean(status),indent=2)+'\n')
        print(json.dumps(status),flush=True)
    for stop in schedule:
        state=advance(state,jnp.asarray(y),jnp.asarray(rates),(stop-step)*factor)
        host=jax.device_get(state)
        if not all(np.all(np.isfinite(v)) for v in host.values()): raise FloatingPointError(f'Nonfinite state at {stop}')
        for i,case in enumerate(cases):
            for block in range(3):
                if not rates[i,block]: np.testing.assert_array_equal(host['z'][i,block],z[i,block])
            if not rates[i,3]: np.testing.assert_array_equal(host['d'][i],d[i])
        step=stop; snapshots[step]=host
        if stop%100000==0 or stop==args.end_step or time.monotonic()-begin>args.max_seconds:
            save(stop==args.end_step)
        if time.monotonic()-begin>args.max_seconds: return
    if not schedule: save(step>=args.end_step)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--fork-step',type=int,choices=(20000,100000),required=True)
    parser.add_argument('--arms',choices=tuple(ARMS),nargs='+',default=list(ARMS))
    parser.add_argument('--eta',type=float,default=.002)
    parser.add_argument('--end-step',type=int,default=600000)
    parser.add_argument('--max-seconds',type=float,default=3400)
    run(parser.parse_args())


if __name__=='__main__': main()
