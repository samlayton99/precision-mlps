"""Paired GD interventions on force replenishment and generated-mode penalties."""
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
from . import adam_forces as af, adam_run as ar, targets
from .run import write_json
from .plateau_runtime import verify_gpu, commit_checkpoint

ANCHORS=('moment3','moment4','moment5','moment9','mixed_sine','localized_sine','chirp')
ARMS=('joint','freeze_readout','clamp_fine','no_slope_tracking','weighted_lower')
METRICS=('relative_mse','mean_gamma','max_gamma','readout_l2','reference_effective_norm',
         'reference_tracking_norm','applied_effective_norm','applied_tracking_norm',
         'reference_outward','applied_outward','hard_residual','coarse_residual_norm',
         'active_balance_effective_norm','force_identity_error','coarse_resolved','applied_unresolved_norm')


def initial(p):
    w=(len(p)-1)//3
    return dict(p=p,positive=jnp.zeros(w),negative=jnp.zeros(w),signed=jnp.zeros(3),crossing=jnp.array(0.),
        path=jnp.array(0.),force_sum=jnp.zeros(w),force_norm_sum=jnp.array(0.),failed=jnp.array(False),unresolved=jnp.array(0))


def field(p,x,y,q,fine0,arm,weight,degree):
    """Reference force uses ordinary GD; applied channels describe this intervention."""
    w=(len(p)-1)//3
    g,r,jc,ec=af.field(p,x,y); ref,info=af.split(g,jc,ec)
    e=q.T @ r/len(x)
    low=q @ (e*((jnp.arange(q.shape[1])>=2)&(jnp.arange(q.shape[1])<degree)))
    active_r=r+(weight-1)*low
    ga,_,_,eca=af.field(p,x,y+r-active_r)
    active,_=af.split(ga,jc,eca)
    gf,_,_,ecf=af.field(p,x,y+r-fine0)
    fixed,_=af.split(gf,jc,ecf)
    applied=jnp.where(arm==4,active,ref)
    applied=applied.at[0,:w].set(jnp.where(arm==2,fixed[0,:w],applied[0,:w]))
    applied=applied.at[1,:w].set(jnp.where(arm==3,0.,applied[1,:w]))
    direction=applied.sum(axis=0)
    mobility=jnp.ones_like(p).at[2*w:].set(jnp.where(arm==1,0.,1.))
    direction=mobility*direction
    balanced,active_info=af.split(g,jc,ec,mobility)
    row=jnp.array([jnp.mean(r*r)/jnp.mean(y*y),jnp.mean(abs(p[:w])),jnp.max(abs(p[:w])),
        jnp.linalg.norm(p[2*w:3*w]),jnp.linalg.norm(ref[0,:w]),jnp.linalg.norm(ref[1,:w]),
        jnp.linalg.norm(applied[0,:w]),jnp.linalg.norm(applied[1,:w]),
        -jnp.mean(jnp.sign(p[:w])*ref[0,:w]),-jnp.mean(jnp.sign(p[:w])*direction[:w]),
        e[jnp.minimum(degree,q.shape[1]-1)],jnp.linalg.norm(ec),
        jnp.linalg.norm(jnp.where(arm==1,balanced[0,:w],applied[0,:w])),
        jnp.linalg.norm(direction[:w]-applied[:,:w].sum(axis=0)),info['resolved']&active_info['resolved'],jnp.linalg.norm(applied[2,:w])])
    return direction,applied,ref,row


def advance_factory(x,q):
    def advance(old,y,fine0,setting,length):
        arm,weight,degree,eta=setting
        def step(_,carry):
            state,_,lo,hi=carry; p=state['p'];w=(len(p)-1)//3
            direction,applied,ref,row=field(p,x,y,q,fine0,arm,weight,degree.astype(int))
            delta=-eta*direction; pn=p+delta; change=abs(pn[:w])-abs(p[:w])
            signed=-eta*jnp.mean(applied[:,:w]*jnp.sign(p[:w]),axis=1)
            new=dict(p=pn,positive=state['positive']+jnp.maximum(change,0),negative=state['negative']+jnp.maximum(-change,0),
                signed=state['signed']+signed,crossing=state['crossing']+jnp.mean(change)-signed.sum(),
                path=state['path']+jnp.linalg.norm(delta[:w]),force_sum=state['force_sum']+ref[0,:w],
                force_norm_sum=state['force_norm_sum']+jnp.linalg.norm(ref[0,:w]),failed=state['failed'],
                unresolved=state['unresolved']+(row[METRICS.index('coarse_resolved')]==0))
            good=jnp.all(jnp.isfinite(pn))&jnp.all(jnp.isfinite(row))&~state['failed']
            new=jax.tree.map(lambda a,b:jnp.where(good,a,b),new,state);new['failed']=~good
            return new,row,jnp.fmin(lo,row),jnp.fmax(hi,row)
        zero=jnp.zeros(len(METRICS))
        return jax.lax.fori_loop(0,length,step,(old,zero,jnp.full_like(zero,jnp.inf),jnp.full_like(zero,-jnp.inf)))
    return jax.jit(jax.vmap(advance,in_axes=(0,0,0,0,None)))


def cases_for(seed,start,half=False):
    rows=[]
    for target in ANCHORS:
        for arm in ARMS[:4]:
            rows.append(dict(target=target,seed=seed,start=start,arm=arm,weight=1.,eta=.001 if half else .002))
        if target in ('moment4','moment5','moment9'):
            for weight in (0.,.1,10.):
                rows.append(dict(target=target,seed=seed,start=start,arm='weighted_lower',weight=weight,eta=.001 if half else .002))
    return rows


def run(args):
    if not jax.config.x64_enabled: raise ValueError('FP64 is required')
    out=args.output;out.mkdir(parents=True,exist_ok=True);verify_gpu(out,args.runtime)
    seed=args.seed;start=args.start
    folder=args.source/f'primary_{seed}'
    manifest=json.loads((folder/'manifest.json').read_text()); f=np.load(folder/'snapshots.npz')
    si=int(np.flatnonzero(f['steps']==start)[0]); cases=cases_for(seed,start,args.half)
    x=targets.grid(args.samples); q=np.polynomial.legendre.legvander(x,9) @ targets.polynomial_map(x)
    pp=[]; yy=[]; ff=[]; settings=[]
    for c in cases:
        i=next(i for i,v in enumerate(manifest['cases']) if v['target']==c['target'] and v['optimizer']=='gd')
        p=f['p'][i,si]; y=af.data(c['target'],args.samples)[1]
        (a,b,readout),d=af.unpack(p);r=np.tanh(x[:,None]*a+b) @ readout+d-y
        fine=r-q[:,:2] @ (q[:,:2].T @ r/len(x))
        degree=int(c['target'][6:]) if c['target'].startswith('moment') else 0
        pp.append(p); yy.append(y);ff.append(fine);settings.append([ARMS.index(c['arm']),c['weight'],degree,c['eta']])
    pp=np.array(pp);yy=np.array(yy);ff=np.array(ff);settings=np.array(settings)
    protocol=dict(cases=cases,reference_updates=args.horizon,samples=args.samples,source_commit=os.environ.get('RACE_SOURCE_COMMIT'),
        input_sha256=hashlib.sha256((folder/'snapshots.npz').read_bytes()).hexdigest(),metrics=METRICS,
        fine_clamp='Replace only slope effective force residual with the fork fine residual; current reference tracking retained',
        weighted_loss='Half mean squared residual plus (weight-1)/2 times squared residual in degrees 2 through k-1')
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=json.loads(json.dumps(protocol)):
        raise ValueError('Changed probe protocol')
    write_json(out/'manifest.json',protocol)
    state=jax.vmap(initial)(jnp.asarray(pp));offset=0;records=[];ss={0:jax.device_get(state)}
    if (out/'state.npz').exists():
        old=dict(np.load(out/'state.npz'));offset=int(old.pop('offset'));state=jax.tree.map(jnp.asarray,old)
        old=np.load(out/'snapshots.npz');ss={int(s):{k:old[k][:,i] for k in state} for i,s in enumerate(old['offsets']) if s<=offset}
        old=np.load(out/'trace.npz');records=list(zip(old['offsets'].tolist(),old['values'].transpose(1,0,2),old['minimum'].transpose(1,0,2),old['maximum'].transpose(1,0,2)))
        records=[r for r in records if r[0]<=offset]
    advance=advance_factory(jnp.asarray(x),jnp.asarray(q));begun=time.monotonic()
    def save():
        host=jax.device_get(state);ss[offset]=host;tt=sorted(ss)
        ar.atomic_npz(out/'snapshots.npz',offsets=np.array(tt),initial_p=pp,**{k:np.stack([ss[t][k] for t in tt],axis=1) for k in host})
        ar.atomic_npz(out/'trace.npz',offsets=np.array([r[0] for r in records]),**{k:np.stack([r[i] for r in records],axis=1) for i,k in enumerate(('values','minimum','maximum'),1)})
        ar.atomic_npz(out/'state.npz',offset=np.array(offset),**host)
        write_json(out/'status.json',dict(offset=offset,complete=offset==args.horizon,failed=int(np.count_nonzero(host['failed'])),
            unresolved_steps=int(np.max(host['unresolved'])),
            motion_identity=float(np.max(abs(host['positive']-host['negative']-(abs(host['p'][:,:177])-abs(pp[:,:177]))))),
            channel_identity=float(np.max(abs(np.mean(abs(host['p'][:,:177])-abs(pp[:,:177]),axis=1)-host['signed'].sum(axis=1)-host['crossing']))),
            seconds=time.monotonic()-begun))
        commit_checkpoint(args.runtime)
    schedule=sorted(s for s in {1,2,5,10,20,50,100,200,500,1000,*range(2000,args.horizon+1,2000),args.horizon} if s<=args.horizon)
    for end in schedule:
        if end<=offset:continue
        state,row,lo,hi=jax.device_get(advance(state,jnp.asarray(yy),jnp.asarray(ff),jnp.asarray(settings),(end-offset)*(2 if args.half else 1)))
        offset=end;records.append((end,row,lo,hi))
        if end%20000==0 or end==args.horizon or time.monotonic()-begun>args.max_seconds:
            save();print(json.dumps(dict(seed=seed,start=start,offset=end,seconds=time.monotonic()-begun)),flush=True)
        if time.monotonic()-begun>args.max_seconds:break


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',choices=('slurm','modal'),default='slurm')
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--start',type=int,required=True)
    p.add_argument('--horizon',type=int,default=500000);p.add_argument('--half',action='store_true')
    p.add_argument('--samples',type=int,default=2048)
    p.add_argument('--max-seconds',type=float,default=1650);run(p.parse_args())
