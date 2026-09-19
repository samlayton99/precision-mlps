"""Eight-GPU-hour joint conditioning campaign; paired maps and explicit failures."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import core, diagnostics, difference_analysis, difference_training as first
from . import higher_order as higher, ratio, run

MAPS=("parameter_scale","parameter_differences")
OPTIMIZERS=("gd","adam","gn","ssbroyden")
RATES=[factor*10.**power for power in range(-5,0) for factor in (1,3)]
HIGHER_COLUMNS=("half_mse","gradient_norm","delta_c_rms","delta_lambda_rms","attempts",
                "step_size","damping","reduction_ratio","curvature","guard_active",
                "linear_residual","function_evaluations","gradient_evaluations","jacobian_evaluations","elapsed_seconds")


def case(optimizer,coordinates,eta=1.,n=512,seed=0,**kwargs):
    row=dict(campaign="joint_conditioning",optimizer=optimizer,coordinates=coordinates,eta=eta,n=n,seed=seed)
    if optimizer in ("gd","adam"):row["native_epsilon"]=1e-12
    if optimizer=="gn":row["damping_floor"]=1e-24
    if optimizer=="ssbroyden":row.update(curvature_epsilon=1e-24,search_threshold=1e-15)
    return dict(row,**kwargs)


def pilot():
    return ([case(opt,coord,eta) for opt in ("gd","adam") for coord in MAPS for eta in RATES]
            +[case(opt,coord) for opt in ("gn","ssbroyden") for coord in MAPS])


def guard_cases(selected):
    rows=[]
    for row in selected:
        if row['optimizer']=='adam':
            rows.extend(dict(row,native_epsilon=eps) for eps in (1e-8,1e-15))
        elif row['optimizer']=='ssbroyden':
            rows.extend(dict(row,curvature_epsilon=eps) for eps in (float(np.finfo(float).eps),1e-30))
            rows.extend(dict(row,search_threshold=eps) for eps in (1e-6,1e-12))
        elif row['optimizer']=='gn':
            rows.extend(dict(row,damping_floor=eps) for eps in (float(np.finfo(float).eps),1e-30))
    # Audit curvature first, then Adam, GN, and finally search termination.
    return sorted(rows,key=lambda c: (0 if c['optimizer']=='ssbroyden' and c['search_threshold']==1e-15
                                     else 1 if c['optimizer']=='adam' else 2 if c['optimizer']=='gn' else 3,
                                     c['coordinates'],first.case_key(c)))


def snapshots(frontier):
    return set([0,1,10,100,1000,2048,frontier,*range(2000,frontier+1,2000)])


def advance_higher(root,config,frontier,deadline,source,samples=16):
    path=root/first.case_key(config);path.mkdir(parents=True,exist_ok=True)
    metadata=dict(config,target="sine",initialization="xavier_a_reference",samples_per_cell=samples,
                  validation_points=32768,objective="half-MSE",reference_lambda=.25,minimum_updates=20000,
                  ssbroyden_commit=higher.SSB_COMMIT,optimistix_commit=higher.OPTIMISTIX_COMMIT)
    if (path/'case.json').exists() and json.loads((path/'case.json').read_text())!=metadata:
        raise ValueError(f"Configuration changed at {path}")
    run.write_json(path/'case.json',metadata)
    latest=json.loads((path/'latest.json').read_text()) if (path/'latest.json').exists() else {}
    if latest.get('status') in higher.STATUS.values() and latest.get('status')!='continuing':return
    if latest.get('completed_updates',0)>=frontier:return
    started=time.monotonic();old_seconds=latest.get('training_seconds',0.)
    g,residual,jacobian,loss=higher.problem(config['n'],config['coordinates'],samples)
    physical=lambda p:higher.physical(p,g,config['coordinates'])
    z=higher.initial_parameters(g,config['seed'],config['coordinates'])
    if config['optimizer']=='gn':
        state=higher.gn_initial(z)
        advance=higher.gn_step(residual,jacobian,physical,config['damping_floor'])
    else:
        solver=higher.ssb_solver(source,config['curvature_epsilon'],config['search_threshold'])
        state=higher.ssb_initial(solver,loss,z)
        advance=higher.ssb_step(solver,loss,physical,config['curvature_epsilon'])
    if latest:
        state,saved_step=higher.load_state(path/'state_latest.pkl',state)
        if saved_step!=latest['completed_updates'] or int(state['count'])!=saved_step:
            raise ValueError('Checkpoint and progress record disagree')
    at=int(state['count']);written=at;trace=[];ring=deque(maxlen=2048)
    if (path/'dense_latest.npz').exists():
        with np.load(path/'dense_latest.npz') as prior:
            ring.extend({k:prior[k][i] for k in prior.files} for i in range(len(prior['step'])))
    loss_eval=jax.jit(loss)
    xv=jnp.asarray(diagnostics.midpoint_grid(32768))
    @jax.jit
    def validation(p):
        c,gamma=physical(p)
        def block(x):return c[0]+core.tanh((x[:,None]-jnp.asarray(g.centers))*gamma)@c[1:]
        pred=jax.lax.map(block,xv.reshape(-1,512)).reshape(-1)
        return jnp.mean((pred-core.target(xv,'sine'))**2)
    run.save_arrays(path/'reference.npz',centers=g.centers,alpha=g.alpha,d=g.d,h=g.h,core=g.core,corrected_halo=g.corrected_halo)
    def save(final=False,failure=None):
        nonlocal trace,written
        count=int(state['count']);c,gamma=map(np.asarray,physical(state['z']))
        mse=float(2*loss_eval(state['z']));vmse=float(validation(state['z']))
        data=dict(c=c,gamma=gamma,**{'lambda':g.h*gamma},z=np.asarray(state['z'])[:g.width+1],
                  native_parameters=np.asarray(state['z']),train_mse=mse,validation_mse=vmse)
        run.save_arrays(path/f'checkpoint_{count:09d}.npz',**data)
        higher.save_state(path/'state_latest.pkl',state,count)
        if count==0 or final:higher.save_state(path/f'state_{count:09d}.pkl',state,count)
        if trace:
            run.save_arrays(path/f'trace_{written:09d}_{count:09d}.npz',trace=np.asarray(trace),columns=HIGHER_COLUMNS)
        if ring:
            dense={k:np.stack([r[k] for r in ring]) for k in ring[0]}
            run.save_arrays(path/'dense_latest.npz',**dense)
            if count==2048 or final:run.save_arrays(path/f'dense_{count:09d}.npz',**dense)
        if failure is not None:
            run.save_arrays(path/'failure.npz',**{k:np.asarray(v) for k,v in failure.items()},last_accepted=np.asarray(state['z']))
        row=dict(step=count,completed_updates=count,status=higher.STATUS[int(state['status'])],
                 minimum_completed=count>=20000,train_mse=mse if np.isfinite(mse) else None,
                 validation_mse=vmse if np.isfinite(vmse) else None,training_seconds=old_seconds+time.monotonic()-started,
                 **{k:int(state[k]) for k in ('function_evaluations','gradient_evaluations','jacobian_evaluations')})
        run.write_json(path/'latest.json',row)
        print(json.dumps(dict(key=path.name,**row)),flush=True)
        trace=[];written=count
    if not latest:save()
    targets=snapshots(frontier)
    while int(state['count'])<frontier and time.monotonic()<deadline:
        before=state['z'];c0,g0=physical(before);index=int(state['count'])
        state,ev=advance(state)
        ev=jax.device_get(ev);status=int(state['status'])
        if status:
            save(final=True,failure=ev);return
        c1,g1=physical(state['z']);c0,g0,c1,g1=map(np.asarray,(c0,g0,c1,g1))
        row=dict(step=index,c=c0,gamma=g0,gradient_native=np.asarray(ev['gradient']),
                 delta_c=c1-c0,delta_lambda=g.h*(g1-g0))
        ring.append(row)
        values=[ev['loss'],np.linalg.norm(ev['gradient']),np.sqrt(np.mean((c1-c0)**2)),
                np.sqrt(np.mean((g.h*(g1-g0))**2)),ev['attempts'],ev['step_size'],ev['damping'],ev['ratio'],
                ev['curvature'],ev['guard_active'],ev['linear_residual'],state['function_evaluations'],
                state['gradient_evaluations'],state['jacobian_evaluations'],old_seconds+time.monotonic()-started]
        trace.append([float(v) for v in values]);at=int(state['count'])
        if at in targets:save(final=at==frontier)
    if trace or int(state['count'])!=written:save(final=True)


def read_trace(folder,end,optimizer):
    if optimizer in ('gd','adam'):return difference_analysis.read_trace(folder,end)
    by_step={}
    for path in sorted(folder.glob('trace_*.npz')):
        _,lo,hi=path.stem.split('_');lo,hi=int(lo),int(hi)
        if lo>=end:continue
        with np.load(path) as a:
            np.testing.assert_array_equal(a['columns'],HIGHER_COLUMNS)
            assert len(a['trace'])==hi-lo
            by_step.update((i,v) for i,v in zip(range(lo,hi),a['trace']) if i<end)
    if set(by_step)!=set(range(end)):raise ValueError(f'Incomplete trace at {folder}')
    return np.stack([by_step[i] for i in range(end)])


def summarize(root):
    rows=[]
    for path in sorted(root.glob('*/case.json')):
        folder=path.parent
        if not (folder/'latest.json').exists():continue
        config=json.loads(path.read_text());status=json.loads((folder/'latest.json').read_text())
        minimum=100000 if config['optimizer'] in ('gd','adam') else 20000
        row=dict(key=folder.name,case=config,**status,eligible=status['completed_updates']>=minimum)
        if row['eligible']:
            trace=read_trace(folder,minimum,config['optimizer'])
            window=2*trace[-minimum//5:,0]
            row.update(comparison_step=minimum,window_mean_mse=float(window.mean()),
                       window_median_mse=float(np.median(window)),window_max_mse=float(window.max()))
        rows.append(row)
    return rows


def selections(rows):
    selected=[]
    for opt in OPTIMIZERS:
        for coord in MAPS:
            pool=[r for r in rows if r['eligible'] and r['case']['n']==512 and r['case']['seed']==0
                  and (r['case']['optimizer'],r['case']['coordinates'])==(opt,coord)]
            if not pool:
                if opt in ('gn','ssbroyden'):selected.append(case(opt,coord));continue
                raise ValueError(f'No completed first-order comparison for {opt}, {coord}')
            best=min(pool,key=lambda r:(r['window_mean_mse'],r['case']['eta']))['case']
            selected.append({k:best[k] for k in case(opt,coord)})
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--manifest',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--select',action='store_true')
    parser.add_argument('--worker',type=int,choices=(0,1));parser.add_argument('--seconds',type=float,default=1800)
    parser.add_argument('--frontier',type=int);parser.add_argument('--require-gpu',action='store_true')
    parser.add_argument('--benchmark',action='store_true',help='Checkpoint pilot trajectories at 2k first-order / 100 higher-order steps; resume for scientific comparisons')
    parser.add_argument('--ssbroyden-source',type=Path)
    args=parser.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    if args.prepare:run.write_json(args.root/'pilot.json',pilot());return
    if args.select:
        rows=summarize(args.root);selected=selections(rows)
        for name,value in (('summary',rows),('selected',selected),('guards',guard_cases(selected)),
                           ('seed_1',[dict(c,seed=1) for c in selected]),
                           ('width_transfer',[dict(c,n=1024,seed=s) for s in (0,1) for c in selected])):
            run.write_json(args.root/f'{name}.json',value)
        return
    if args.worker is None or args.manifest is None:parser.error('Training requires worker and manifest')
    started=time.monotonic();deadline=started+args.seconds-60
    if args.require_gpu:ratio.gpu_environment(args.root)
    cases=json.loads(args.manifest.read_text())
    cases=[c for c in cases if c['coordinates']==MAPS[args.worker]]
    run.write_json(args.root/f'worker_{args.worker}_manifest.json',dict(cases=cases,seconds=args.seconds,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    groups={}
    for c in cases:
        if c['optimizer'] in ('gd','adam'):
            key=(c['optimizer'],c['n'],c['seed'],c['coordinates'],c['native_epsilon'])
            groups.setdefault(key,[]).append(c)
    for group in groups.values():
        if time.monotonic()>=deadline:return
        first.advance_group(args.root,group,2000 if args.benchmark else args.frontier or 100000,deadline)
    for c in cases:
        if c['optimizer'] not in ('gn','ssbroyden'):continue
        if time.monotonic()>=deadline:return
        advance_higher(args.root,c,100 if args.benchmark else args.frontier or 20000,deadline,args.ssbroyden_source)


if __name__=='__main__':main()
