"""Saved-state audit and fixed paired GD/Adam continuation matrix on Modal."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence

from . import population_coarse_feedback as cf, population_adam as pa, adam_forces as af
from .population_adam_run import data, error, scalar, settings, write_csv, SIX, digest


def registry(root):
    result=[];wide={}
    for folder in sorted(root.glob('adam_population_wide*')):
        with (folder/'cases.csv').open() as f:
            cases=list(csv.DictReader(f))
        for c in cases:
            c=dict(c)
            for k in ('width','seed','case_id'):c[k]=int(c[k])
            for k in ('eta','beta1','beta2','epsilon','h'):c[k]=float(c[k])
            c['origin']=c['optimizer'];c['adaptive']=c['origin']=='adam'
            wide[(folder.name,c['case_id'])]=c
            if c['origin']=='adam' or (c['width']==1409 and c['seed']==30):
                for age in ((25000,) if c['origin']=='adam' else (25000,130000)):
                    actual=125000 if age==130000 else age
                    result.append(dict(c,age=age,path=str(folder/f"case{c['case_id']}_state{actual}.npz"),
                                       member='',advance=age-actual))
    folder=root/'adam_variance_runs_20260926'
    sources=json.loads((folder/'facts.json').read_text())['sources']
    for i,source in enumerate(sources):
        p=Path(source['path']);case_id=int(p.name.split('_')[0].removeprefix('case'))
        c=wide[(p.parent.name,case_id)]
        result.append(dict(c,age=130000,path=str(folder/f'case{i}_fork130000.npz'),member='',advance=0))
    for name in ('window_development_20260925','window_panel31_20260925','window_widths_20260925'):
        folder=root/name;seen=[]
        for c in json.loads((folder/'facts.json').read_text())['trajectories']:
            identity=tuple(c[k] for k in ('target','width','seed'))
            if identity in seen:continue
            seen.append(identity);index=len(seen)-1
            if c['width'] not in (705,1409) or c['target'] not in SIX:continue
            base={k:c[k] for k in ('target','width','seed')}
            base.update(origin='gd',optimizer='gd',h=2/c['nref'],eta=.002,beta1=0.,beta2=.999,epsilon=1e-8,adaptive=False)
            for age,offset in ((25000,5000),(130000,110000)):
                result.append(dict(base,age=age,path=str(folder/'sparse_states.npz'),member=f'case{index}_gd_{offset}',advance=0))
    identities=[tuple(c[k] for k in ('target','width','seed','origin','age')) for c in result]
    assert len(result)==96 and len(set(identities))==96
    return sorted(result,key=lambda c:((c['width'],c['seed'])!=(705,30),c['width'],c['seed'],SIX.index(c['target']),c['origin'],c['age']))


def load(c,x,y):
    with np.load(c['path'],allow_pickle=False) as z:
        if c['member']:
            p=jnp.asarray(z[c['member']]);saved=pa.initial(p,count=c['age'])
        else:
            saved={k:jnp.asarray(z[k]) for k in ('p','m','v','cm','count')}
            if 'alternative_v' in z:saved['alternative_v']=jnp.asarray(z['alternative_v'])
    if c['advance']:
        complete=pa.initial(saved['p'],saved['m'],saved['v'],saved['count'])
        complete['cm']=saved['cm']
        saved=pa.advance(complete,x,y,settings(c),c['advance'])
    assert int(saved['count'])==c['age']
    return saved


def curvature(p,x,y,config,inverse):
    """Full loss Hessian, never substituted by the coarse Gauss--Newton matrix."""
    _,_,jc,_=af.field(p,x,y);jc=np.asarray(jc);gram=jc@jc.T
    rows={}
    metrics=[('raw',jnp.ones_like(p))]
    if bool(config[4]):metrics.append(('adaptive',jnp.sqrt(inverse)))
    for name,root in metrics:
        def mv(v):return np.asarray(cf.hessian_product(p,jnp.asarray(v),root,x,y),dtype=np.float64)
        op=LinearOperator((len(p),len(p)),matvec=mv,dtype=np.float64)
        v0=np.random.default_rng(113).normal(size=len(p))
        values=vectors=None;success=False
        for limit in (160,400):
            try:
                values,vectors=eigsh(op,k=2,which='LA',tol=1e-7,maxiter=limit,ncv=12,v0=v0)
                residual=max(np.linalg.norm(mv(vectors[:,i])-values[i]*vectors[:,i])/max(1.,abs(values[i])) for i in range(2))
                success=residual<5e-6
                if success:break
            except ArpackNoConvergence:pass
        rows[name+'_resolved']=success
        if not success:
            rows[name+'_lambda']=float('nan');continue
        i=int(np.argmax(values));lam=float(values[i]);direction=np.asarray(root)*vectors[:,i]
        overlap=jc.T@np.linalg.solve(gram,jc@direction)
        scaled_jc=jc*np.asarray(root)
        metric_overlap=scaled_jc.T@np.linalg.solve(scaled_jc@scaled_jc.T,scaled_jc@vectors[:,i])
        rows.update({name+'_lambda':lam,name+'_second_lambda':float(values[1-i]),
            name+'_residual':float(residual),name+'_coarse_overlap':float(overlap@overlap/(direction@direction)),
            name+'_metric_coarse_overlap':float(metric_overlap@metric_overlap),
            name+'_coarse_rayleigh_fraction':float((jc@direction)@(jc@direction)/lam)})
        try:
            small=eigsh(op,k=1,which='SA',tol=1e-6,maxiter=160,ncv=12,v0=v0,return_eigenvectors=False)
            rows[name+'_min_lambda']=float(small[0])
        except ArpackNoConvergence:rows[name+'_min_lambda']=float('nan')
        threshold=2*(1+float(config[1]))/(1-float(config[1]))
        rows[name+'_stability_ratio']=float(config[0])*lam/threshold
    if not bool(config[4]):
        rows.update({k.replace('raw_','adaptive_',1):v for k,v in list(rows.items()) if k.startswith('raw_')})
    return rows


def state_probes(state,x,y,config):
    z=cf.proposal(state,x,y,config);eta,b1=config[:2]
    mf=z['cm'][0]/(1-b1**z['count'])
    proposals={'raw':-eta*z['channels'][0],'scaled_current':-eta*z['inverse']*z['channels'][0],
               'momentum':-eta*mf,'processed':z['fine'],
               'balanced':cf.proposal(state,x,y,config,2)['fine']}
    target=jnp.linalg.norm(z['fine'][:-1]);rows=[]
    for name,u in proposals.items():
        for match in (False,True):
            vec=u*cf.divide(target,jnp.linalg.norm(u[:-1])) if match else u
            rows.append(dict(proposal=name,norm_matched=match,**scalar(cf.isolated_response(state['p'],vec,x,y))))
    return rows,z


def run(args):
    if jax.default_backend()!='gpu':raise RuntimeError('Modal GPU required')
    start=time.monotonic();cases=registry(args.inputs)
    if args.cohort!='all':
        width,seed=map(int,args.cohort.split('_'));cases=[c for c in cases if (c['width'],c['seed'])==(width,seed)]
    if args.stage!='audit':cases=[c for c in cases if c['age']==130000 and (args.stage not in ('adam','followup') or c['origin']=='adam')]
    rows=[];probes=[];curves=[];runs=[];bursts=[];sources=[];times=[]
    def flush():
        for name,values in (('states',rows),('probes',probes),('curvature',curves),('runs',runs),('bursts',bursts)):
            write_csv(args.output/(name+'.csv'),values)
        (args.output/'facts.json').write_text(json.dumps(dict(stage=args.stage,cohort=args.cohort,
            seconds=time.monotonic()-start,sources=sources,runs=runs,expected_cases=len(cases)),indent=2)+'\n')
    for ci,c in enumerate(cases):
        if times and time.monotonic()-start+1.2*max(times)>args.seconds-30:break
        begun=time.monotonic();x,y=map(jnp.asarray,data(c['target']));xe,ye=map(jnp.asarray,data(c['target'],8192))
        saved=load(c,x,y);config=settings(c);state=cf.initialize(saved,x,y,config)
        base={k:c[k] for k in ('target','width','seed','origin','age','h')}
        base['case_key']=f"{c['target']}_{c['width']}_{c['seed']}_{c['origin']}"
        sources.append(dict(**base,path=c['path'],member=c['member'],advance=c['advance'],sha256=digest(Path(c['path'])),
            parameter_hash=hashlib.sha256(np.asarray(saved['p']).tobytes()).hexdigest(),data_hash=hashlib.sha256(np.asarray(y).tobytes()).hexdigest()))
        if args.stage=='audit':
            pp,z=state_probes(state,x,y,config)
            probes.extend(dict(**base,offset=0,arm='native',**r) for r in pp)
            curves.append(dict(**base,offset=0,arm='native',eta=float(config[0]),tracking_factor=1.,**curvature(state['p'],x,y,config,z['inverse'])))
            rows.append(dict(**base,offset=0,arm='native',**scalar(cf.diagnostics(state,x,y,config))))
            flush();print(json.dumps(dict(audit=base,seconds=time.monotonic()-begun)),flush=True)
            times.append(time.monotonic()-begun);continue
        if args.stage=='gd':
            config=jnp.array([.002,0.,.999,1e-8,False]);state=cf.initialize(saved,x,y,config)
            cc=curvature(state['p'],x,y,config,jnp.ones_like(state['p']))
            reference=2/cc['raw_lambda'] if cc['raw_resolved'] and cc['raw_lambda']>0 else None
            policies=[('native_rate',.002,1.,0)]
            if reference:
                policies += [(f'rate_{r:g}',r*reference,1.,0) for r in (.25,.5,.9,1.05,1.25)]
                policies += [(f'rate_{r:g}_tracking01',r*reference,.1,0) for r in (.9,1.05)]
            duration=10000
        elif args.stage=='followup':
            policies=[(name,.002,1.,i) for name,i in (('native',0),('fine_off',6),('frozen',7),('frozen_balanced',8))];duration=20000
        else:policies=[(name,.002,1.,i) for i,name in enumerate(cf.ARMS)];duration=20000
        stopped=False;runtime=time.monotonic()
        configs=jnp.stack([config.at[0].set(eta) for _,eta,_,_ in policies])
        initialized=[cf.initialize(saved,x,y,cfg) for cfg in configs]
        batch=jax.tree.map(lambda *v:jnp.stack(v),*initialized)
        arms=jnp.array([arm for _,_,_,arm in policies]);rhos=jnp.array([rho for _,_,rho,_ in policies])
        finished=set();latest={};statuses={};trace=None
        for offset in range(0,duration+1,1000):
            active=arms if offset<=10000 else jnp.zeros_like(arms)
            if offset:
                batch,trace=cf.advance_many(batch,x,y,configs,active,rhos,steps=1000)
                trace=np.asarray(trace)
            for pi,(name,eta,rho,arm) in enumerate(policies):
                if pi in finished:continue
                cfg=configs[pi];policy=int(active[pi]);s=jax.tree.map(lambda v:v[pi],batch)
                b=dict(**base,arm=name,eta=eta,tracking_factor=rho,optimizer=args.stage)
                if offset:
                    for k in range(1000):
                        n=offset-1000+k
                        if n<128 or 9872<=n<10000 or 19872<=n<20000:
                            if np.isfinite(trace[pi,k,0]):bursts.append(dict(**b,offset=n,**dict(zip(cf.BURST,map(float,trace[pi,k])))))
                d=scalar(cf.diagnostics(s,x,y,cfg,policy,rho));last=int(s['offset'])
                latest[pi]=(b,d,last)
                row=dict(**b,offset=last,flow_time=last*eta,**d,lambda_rms=c['h']*d['slope_rms'])
                if offset in (0,10000,20000):row['relative_eval_error']=float(error(s['p'],xe,ye))
                rows.append(row)
                if not bool(s['alive']):
                    statuses[pi]='nonfinite' if int(s['failure'])==1 else 'unresolved';finished.add(pi);continue
                if offset in (0,2000,10000,20000):
                    z=cf.proposal(s,x,y,cfg,policy,rho)
                    curves.append(dict(**b,offset=offset,**curvature(s['p'],x,y,cfg,z['inverse'])))
                if offset in (0,10000):
                    pp,_=state_probes(s,x,y,cfg);probes.extend(dict(**b,offset=offset,**r) for r in pp)
                if offset in (10000,20000):
                    np.savez_compressed(args.output/f"{base['case_key']}_{name}_{offset}.npz",**jax.device_get({k:s[k] for k in ('p','m','v','cm','count','shadow_v')}))
            if time.monotonic()-start>args.seconds-30 and offset<duration:stopped=True;break
        for pi in range(len(policies)):
            b,d,last=latest[pi];status=statuses.get(pi,'budget_stopped' if stopped else 'complete')
            runs.append(dict(**b,status=status,end_offset=last,complete=status=='complete' and last==duration,
                batch_seconds=time.monotonic()-runtime,relative_error=d['relative_error'],slope_rms=d['slope_rms'],
                A_closure=d['A_closure'],component_error=d['component_error'],norm_error=d['norm_error']))
            print(json.dumps(dict(**runs[-1],elapsed=time.monotonic()-start)),flush=True)
        flush()
        times.append(time.monotonic()-begun)
        if stopped:break
    flush()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('audit','gd','adam','followup'),required=True);p.add_argument('--cohort',default='all')
    p.add_argument('--seconds',type=float,required=True);run(p.parse_args())
