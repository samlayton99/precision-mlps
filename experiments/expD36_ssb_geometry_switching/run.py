"""Checkpointed SSBroyden comparisons; each policy shares one training kernel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from experiments.expD35_optimization_exploration import core, run as old, ssb
from experiments.expD06_fixed_center_scales import higher_order as higher
from . import accessibility as access, diagnostics, reinitialization

EARLY=(0,1,2,5,10,20,50,100,200,500)


def next_diagnostic(at,frontier):
    return min(next((v for v in EARLY if v>at), (at//500+1)*500),frontier)


def controller(row, state, at, policy):
    candidate=row.get('beta_candidate')
    state=dict(state)
    state['streak']=state.get('streak',0)+1 if candidate is not None else 0
    beta=None
    if policy=='adaptive' and state['streak']>=2 and at-state.get('last_event',-100)>=100:
        beta=candidate
    if policy=='periodic' and at>0 and at%1000==0:beta=.1
    if beta is not None:
        state['last_event']=at;state['streak']=0
    return beta,state


def paths(root,config):
    identity=dict(config);identity['optimizer']='ssbroyden'
    return root/old.key(identity)


def advance(root,config,source,frontier,deadline):
    folder=paths(root,config);folder.mkdir(parents=True,exist_ok=True)
    case_path=folder/'case.json'
    if case_path.exists():assert json.loads(case_path.read_text())==config
    else:old.write_json(case_path,config)
    if (folder/'latest.json').exists():
        previous=json.loads((folder/'latest.json').read_text())
        if previous['step']>=frontier or previous['status']!='continuing':return True
    n,coord,target=config['n'],config['coordinates'],config['target']
    g,loss,physical=ssb.problem(config)
    solver=higher.ssb_solver(source,config.get('curvature_epsilon',1e-30),config.get('search_threshold',1e-15),'accepted_step')
    z=core.initialize(config)['z'];initial_z=z
    state=higher.ssb_initial(solver,loss,z)
    control=dict(streak=0,last_event=-100,done_diagnostic=-1)
    if (folder/'solver.npz').exists():
        state=ssb.load_solver(folder/'solver.npz',state)
        control=json.loads((folder/'controller.json').read_text())
    elif 'parent' in config:
        parent=Path(config['parent'])
        if hashlib.sha256(parent.read_bytes()).hexdigest()!=config['parent_sha256']:
            raise ValueError('Parent solver checkpoint changed')
        state=ssb.load_solver(parent, state)
        control['origin_count']=int(state['count'])
        state=dict(state,count=jnp.array(0),status=jnp.array(0))
        if config.get('fork_policy','continue')!='continue':
            if config.get('reinitialization','none')!='none':raise ValueError('Keep metric and parameter interventions separate')
            metric=state['solver'].f_info.hessian_inv.pytree
            if config['fork_policy']=='metric_mix':
                metric,_=access.mix_metric(metric,jax.grad(loss)(state['z']),config['fork_beta'])
            elif config['fork_policy']!='history_only':raise ValueError(config['fork_policy'])
            state=ssb.restart(state,solver,loss,metric)
            old.write_json(folder/'fork_intervention.json',dict(policy=config['fork_policy'],
                beta=config['fork_beta'],origin_count=control['origin_count'],parameter_change=0.))
        if config.get('reinitialization','none')!='none':
            before=state['z']
            after=jnp.asarray(reinitialization.replace(before,config))
            _,_,residual,_,_=access.problem(n,coord,target)
            delta=residual(after)-residual(before)
            old.write_json(folder/'replacement.json',dict(mode=config['reinitialization'],
                mask=config['reset_mask'],before_mse=float(2*loss(before)),after_mse=float(2*loss(after)),
                function_jump_rms=float(jnp.linalg.norm(delta)),origin_count=control['origin_count']))
            state=dict(state,z=after)
            state=ssb.restart(state,solver,loss,jnp.eye(len(after)))
    g,matrix,_,_,_=access.problem(n,coord,target)
    if 'taus' not in control:
        largest=float(jnp.linalg.svd(matrix(initial_z[g.width+1:]),compute_uv=False)[0])**2
        control['taus']=[20000/largest,100000/largest]
        old.save(folder/'initial.npz',z=np.asarray(state['z']),initial_z=np.asarray(initial_z),taus=np.array(control['taus']))
    evaluator=core.evaluate(n,coord,target)
    while int(state['count'])<frontier and int(state['status'])==0:
        if time.monotonic()>deadline:return False
        at=int(state['count'])
        if control['done_diagnostic']!=at:
            matrix_h=state['solver'].f_info.hessian_inv.pytree
            row,data=diagnostics.diagnostic(state['z'],matrix_h,config,control['taus'],audit=at in (0,10,100,1000,5000,20000))
            beta,control=controller(row,control,at,config['policy'])
            event=None
            if config['policy']=='sham':
                if at in config['replay_events']:event='history_only'
            elif beta is not None:event='metric_mix'
            row.update(step=at,beta=beta,event=event,policy=config['policy'])
            old.save(folder/f'diagnostic_{at:09d}.npz',**data)
            old.save(folder/f'snapshot_{at:09d}.npz',z=np.asarray(state['z']),step=at)
            if at in (*EARLY,1000,2000,5000,10000,20000) or event:
                ssb.save_solver(folder/f'solver_{at:09d}.npz',state)
                old.save(folder/f'metric_{at:09d}.npz',inverse_metric=np.asarray(matrix_h),step=at)
            if event:
                revised=matrix_h
                if event=='metric_mix':revised,_=access.mix_metric(matrix_h,jnp.asarray(data['gradient']),beta)
                state=ssb.restart(state,solver,loss,revised)
            old.write_json(folder/f'diagnostic_{at:09d}.json',row)
            control['done_diagnostic']=at
            ssb.save_solver(folder/'solver.npz',state)
            old.write_json(folder/'controller.json',control)
        end=next_diagnostic(at,frontier)
        traces=[]
        while int(state['count'])<end and int(state['status'])==0:
            length=min(100,end-int(state['count']))
            state,trace=ssb.kernel(n,coord,target,source,config.get('curvature_epsilon',1e-30),
                config.get('search_threshold',1e-15),'non_descent',1000,length)(state)
            jax.block_until_ready(state);traces.append(np.asarray(trace))
        stop=int(state['count'])
        old.save(folder/f'ssb_trace_{at:09d}_{stop:09d}.npz',trace=np.concatenate(traces),columns=np.array(ssb.TRACE))
        metrics=dict(zip(old.EVAL_COLUMNS,map(old.finite,np.asarray(evaluator(state['z'][None]))[0])))
        latest=dict(step=stop,status=higher.STATUS[int(state['status'])],complete=stop>=frontier,
                    requested_frontier=frontier,train_mse=float(2*loss(state['z'])),
                    function_evaluations=int(state['function_evaluations']),gradient_evaluations=int(state['gradient_evaluations']),**metrics)
        old.write_json(folder/f'evaluation_{stop:09d}.json',latest)
        ssb.save_solver(folder/'solver.npz',state);old.write_json(folder/'controller.json',control)
        old.write_json(folder/'latest.json',latest)
        if stop%1000==0 or int(state['status']):print(json.dumps(dict(id=folder.name,**latest)),flush=True)
    at=int(state['count'])
    if control['done_diagnostic']!=at:
        row,data=diagnostics.diagnostic(state['z'],state['solver'].f_info.hessian_inv.pytree,config,control['taus'],audit=True)
        row.update(step=at,event=None,policy=config['policy'])
        old.write_json(folder/f'diagnostic_{at:09d}.json',row);old.save(folder/f'diagnostic_{at:09d}.npz',**data)
        old.save(folder/f'snapshot_{at:09d}.npz',z=np.asarray(state['z']),step=at)
        ssb.save_solver(folder/f'solver_{at:09d}.npz',state)
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--source',required=True);p.add_argument('--frontier',type=int,default=20000)
    p.add_argument('--seconds',type=float,default=1600);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--require-gpu',action='store_true');args=p.parse_args()
    if not jax.config.x64_enabled:p.error('Set JAX_ENABLE_X64=true')
    if args.frontier<20000:p.error('Scientific runs require at least 20k accepted steps or explicit failure')
    args.root.mkdir(parents=True,exist_ok=True)
    if args.require_gpu:old.verify_gpu(args.root)
    source_paths=list(Path(__file__).parent.glob('*.py'))+[Path(ssb.__file__),Path(core.__file__),Path(higher.__file__)]
    old.write_json(args.root/f'source_{os.environ.get("SLURM_JOB_ID","local")}_{args.worker}.json',dict(
        commit=os.environ.get('EXPLORATION_SOURCE_COMMIT'),manifest=json.loads(args.manifest.read_text()),
        source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_paths}))
    deadline=time.monotonic()+args.seconds
    cases=json.loads(args.manifest.read_text())
    for i,config in enumerate(cases):
        if config.get('worker_group',i)%args.workers!=args.worker:continue
        if not advance(args.root,config,args.source,args.frontier,deadline):break


if __name__=='__main__':main()
