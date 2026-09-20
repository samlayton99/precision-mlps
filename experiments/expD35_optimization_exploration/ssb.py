"""Pinned accepted-step SSBroyden with explicit inverse-metric restarts."""
from functools import lru_cache
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from experiments.expD06_fixed_center_scales import higher_order as higher
from . import core, run

TRACE=('mse','accepted_count','accepted','status','search_calls','step_size','curvature',
       'guard_active','gradient_readout','gradient_geometry','delta_readout_rms',
       'delta_gamma_rms','secant_relative_error','native_identity_direction_cosine','metric_reset',
       'stored_directional_derivative','metric_max_abs','matrix_directional_derivative')


def problem(c):
    if c.get('architecture')=='affine':raise ValueError('SSB screen is fixed-center only')
    g=core.old.geometry(c['n']);x=jnp.linspace(-1,1,16*c['n']+1);y=core.target(x,c['target'])
    def physical(z): return core.physical(z,g,c['coordinates'])
    def loss(z):
        w,gamma=physical(z)
        r=w[0]+core.old.tanh((x[:,None]-jnp.asarray(g.centers))*gamma)@w[1:]-y
        return .5*jnp.mean(r*r)
    return g,loss,physical


def reset_matrix(matrix,mode,split):
    identity=jnp.eye(len(matrix),dtype=matrix.dtype)
    if mode in ('full','non_descent'): return identity
    if mode=='blend': return .9*matrix+.1*identity
    if mode=='geometry':
        keep=(jnp.arange(len(matrix))<split)
        return jnp.where(keep[:,None]&keep[None,:],matrix,identity)
    if mode=='none': return matrix
    raise ValueError(mode)


def restart(state,solver,loss,matrix):
    fresh=higher.ssb_initial(solver,loss,state['z'])
    fresh['solver']=eqx.tree_at(lambda s:s.f_info.hessian_inv.pytree,fresh['solver'],matrix)
    # Fresh line search/descent history; global accounting and parameters persist.
    return dict(state,solver=fresh['solver'])


@lru_cache(maxsize=32)
def kernel(n,coordinates,target,source,epsilon,search_threshold,mode,interval,length):
    config=dict(n=n,coordinates=coordinates,target=target)
    g,loss,physical=problem(config)
    solver=higher.ssb_solver(source,epsilon,search_threshold,'accepted_step')
    advance=higher.ssb_step(solver,loss,physical,epsilon)
    @eqx.filter_jit
    def chunk(state):
        def step(current,_):
            matrix_before=current['solver'].f_info.hessian_inv.pytree
            stored=current['solver'].f_info.grad
            matrix_slope=stored@(-matrix_before@stored)
            stored_slope=stored@(-current['solver'].descent_state.newton)
            reset=(current['count']>0)&(current['count']%interval==0)&(mode!='none')&(current['status']==0)
            if mode=='non_descent':
                bad=(stored_slope>=0)|(matrix_slope>=0)|~jnp.isfinite(stored_slope)|~jnp.isfinite(matrix_slope)
                reset=(current['count']>0)&(current['status']==0)&bad
            if mode!='none':
                current=jax.lax.cond(reset,lambda st:restart(st,solver,loss,
                    reset_matrix(st['solver'].f_info.hessian_inv.pytree,mode,g.width+1)),lambda st:st,current)
            before=current['z']
            proposed,ev=advance(current)
            active=current['status']==0
            out=jax.tree.map(lambda new,old:jnp.where(active,new,old),proposed,current)
            c0,ga0=physical(before);c1,ga1=physical(out['z'])
            grad1=jax.grad(loss)(out['z']);dy=grad1-ev['gradient'];delta=out['z']-before
            matrix=out['solver'].f_info.hessian_inv.pytree
            den=jnp.linalg.norm(delta)
            secant=jnp.linalg.norm(matrix@dy-delta)/jnp.where(den>0,den,jnp.nan)
            direction=-(current['solver'].f_info.hessian_inv.pytree@ev['gradient'])
            row=jnp.array([2*ev['loss'],out['count'],ev['accepted'],out['status'],ev['attempts'],
                ev['step_size'],ev['curvature'],ev['guard_active'],jnp.linalg.norm(ev['gradient'][:g.width+1]),
                jnp.linalg.norm(ev['gradient'][g.width+1:]),jnp.sqrt(jnp.mean((c1-c0)**2)),
                jnp.sqrt(jnp.mean((ga1-ga0)**2)),secant,core.cosine(direction,-ev['gradient']),reset,
                stored_slope,jnp.max(jnp.abs(matrix_before)),matrix_slope])
            return out,jnp.where(active,row,jnp.nan)
        return jax.lax.scan(step,state,None,length=length)
    return chunk


def save_solver(path,state):
    leaves=jax.tree.leaves(state)
    run.save(path,**{f'leaf_{i:03d}':np.asarray(a) for i,a in enumerate(leaves)})


def load_solver(path,template):
    leaves,structure=jax.tree.flatten(template)
    with np.load(path) as data:
        restored=[jnp.asarray(data[f'leaf_{i:03d}']) for i in range(len(leaves))]
    for a,b in zip(leaves,restored):
        if a.shape!=b.shape or a.dtype!=b.dtype: raise ValueError('Incompatible SSB checkpoint')
    return jax.tree.unflatten(structure,restored)


def advance(root,c,source,frontier,deadline):
    folder,initial,_=run.prepare(root,c)
    g,loss,physical=problem(c)
    epsilon=c.get('curvature_epsilon',1e-30);threshold=c.get('search_threshold',1e-15)
    solver=higher.ssb_solver(source,epsilon,threshold,'accepted_step')
    state=higher.ssb_initial(solver,loss,initial['z'])
    if (folder/'solver.npz').exists():state=load_solver(folder/'solver.npz',state)
    evaluate=core.evaluate(c['n'],c['coordinates'],c['target'])
    started=time.monotonic()
    while int(state['count'])<frontier and int(state['status'])==0 and time.monotonic()<deadline:
        at=int(state['count']);length=min(100,frontier-at)
        state,trace=kernel(c['n'],c['coordinates'],c['target'],source,epsilon,threshold,
            c.get('metric_reset','none'),c.get('reset_interval',1000),length)(state)
        jax.block_until_ready(state);end=int(state['count'])
        run.save(folder/f'ssb_trace_{at:09d}_{end:09d}.npz',trace=np.asarray(trace),columns=np.asarray(TRACE))
        save_solver(folder/'solver.npz',state)
        values=np.asarray(evaluate(state['z'][None]))[0]
        metrics=dict(zip(run.EVAL_COLUMNS,map(run.finite,values)))
        if end%1000==0 or int(state['status']):
            run.save(folder/f'snapshot_{end:09d}.npz',z=np.asarray(state['z']),step=end,
                     inverse_metric=np.asarray(state['solver'].f_info.hessian_inv.pytree))
        run.write_json(folder/f'evaluation_{end:09d}.json',dict(step=end,**metrics))
        latest=dict(step=end,failed_update=end+1 if int(state['status']) else 0,
            status=higher.STATUS[int(state['status'])],complete=end>=frontier,requested_frontier=frontier,
            last_train_mse=run.finite(2*loss(state['z'])),**metrics)
        run.write_json(folder/'latest.json',latest)
        if end%1000==0 or int(state['status']):
            print(json.dumps(dict(id=folder.name,elapsed=time.monotonic()-started,**latest)),flush=True)
        if end%2000==0 or int(state['status']):
            from .history import consolidate
            consolidate(folder)
    return int(state['count'])>=frontier or int(state['status'])!=0


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--source',required=True)
    p.add_argument('--seconds',type=float,default=1700);p.add_argument('--frontier',type=int,default=20000)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--require-gpu',action='store_true');args=p.parse_args()
    if args.frontier<20000: p.error('Use at least 20k accepted steps or record explicit numerical failure')
    args.root.mkdir(parents=True,exist_ok=True)
    if args.require_gpu:run.verify_gpu(args.root)
    run.write_json(args.root/f'ssb_source_{os.environ.get("SLURM_JOB_ID","local")}.json',dict(
        commit=os.environ.get('EXPLORATION_SOURCE_COMMIT'),source_sha=higher.SSB_SOURCE_SHA256,
        integration='accepted_step',sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        [Path(__file__),Path(core.__file__),Path(higher.__file__)]}))
    cases=json.loads(args.manifest.read_text());deadline=time.monotonic()+args.seconds
    for i,c in enumerate(cases):
        if i%args.workers!=args.worker:continue
        if not advance(args.root,c,args.source,args.frontier,deadline):break


if __name__=='__main__':main()
