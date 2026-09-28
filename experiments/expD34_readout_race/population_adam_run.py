"""Archive audit and short native-Adam runs, executed only inside Modal."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import population_adam as pa, adam_forces as af, targets
from . import effective_feedback_holdout as holdout

BASE=Path('/work/results/checkpoint_D_optimizers/expD34_readout_race')
SIX=('moment5','mixed_sine','gauss_left','bump_right','step_right','kink_abs')


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=keys);out.writeheader();out.writerows(rows)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def data(name,m=2048):
    return (holdout.data if name in holdout.TARGETS else af.data)(name,m)[:2]


@jax.jit
def error(p,x,y):
    a,b,c=p[:-1].reshape(3,-1)
    r=jnp.tanh(x[:,None]*a+b)@c+p[-1]-y
    return jnp.sqrt(jnp.mean(r*r)/jnp.mean(y*y))


pop=jax.jit(pa.population)


def scalar(values):
    return {k:float(v) for k,v in jax.device_get(values).items()}


def settings(case):
    return jnp.array([case.get(k,v) for k,v in
        [('eta',.002),('beta1',.9),('beta2',.999),('epsilon',1e-8),('adaptive',True)]])


def audit(args):
    rows=[];sources=[];begun=time.monotonic()
    for folder in sorted((BASE/'adam_force_extension/raw').glob('primary_*')):
        manifest=json.loads((folder/'manifest.json').read_text())
        path=folder/'snapshots.npz'
        sources.append(dict(path=str(path),sha256=digest(path)))
        with np.load(path,allow_pickle=False) as archive:
            saved={k:archive[k] for k in ('steps','p','m','v','count','channel_m')}
        for i,case in enumerate(manifest['cases']):
            x,y=map(jnp.asarray,data(case['target']));config=settings(case)
            for k,step in enumerate(saved['steps']):
                p=jnp.asarray(saved['p'][i,k]);d=scalar(pop(p))
                d['relative_error']=float(error(p,x,y))
                if int(step) in (20000,100000,600000):
                    state=pa.initial(p,saved['m'][i,k],saved['v'][i,k],saved['count'][i,k],saved['channel_m'][i,k])
                    d.update(scalar(pa.diagnostics(state,x,y,config)))
                rows.append(dict(cohort='archive177',target=case['target'],seed=case['seed'],
                    optimizer=case['optimizer'],width=177,h=2/128,step=int(step),
                    eta=case['eta'],recipe='constant',**d,lambda_rms=2/128*d['slope_rms']))
        print(json.dumps(dict(archive=folder.name,rows=len(rows),seconds=time.monotonic()-begun)),flush=True)
        write_csv(args.output/'archive_states.csv',rows)
        del saved
    # Parameter-only larger-width check. No synthetic historical moment buffers.
    spectrum=Path('/spectrum')
    meta=json.loads((spectrum/'metadata.json').read_text()) if (spectrum/'metadata.json').exists() else json.loads((spectrum/'joint_adam/metadata.json').read_text())
    base=json.loads((spectrum/'base/manifest.json').read_text())
    with np.load(spectrum/'base/joint_input.npz',allow_pickle=False) as z:
        x,y=map(jnp.asarray,(z['x'],z['target']))
    ss=np.load(spectrum/'joint_adam/checkpoint_steps.npy')
    pp=np.load(spectrum/'joint_adam/parameter_checkpoints.npy',mmap_mode='r')
    for k,step in enumerate(ss):
        if int(step)>200000 and int(step) not in (600000,2000000):continue
        for seed in range(pp.shape[1]):
            for ri,recipe in enumerate(meta['recipes']):
                p=jnp.asarray(pp[k,seed,ri]);d=scalar(pop(p))
                rows.append(dict(cohort='archive512',target='mixed_sine',seed=seed,optimizer='adam',
                    width=512,h=base['spacing'],step=int(step),eta=recipe['learning_rate'],
                    recipe=recipe['schedule'],**d,lambda_rms=base['spacing']*d['slope_rms'],
                    relative_error=float(error(p,x,y))))
    sources.append(dict(path=str(spectrum/'joint_adam/parameter_checkpoints.npy'),
                        sha256=digest(spectrum/'joint_adam/parameter_checkpoints.npy')))
    write_csv(args.output/'archive_states.csv',rows)
    # Reuse the previously executed late interventions without forecasting them.
    intervention=[]
    for folder in sorted((BASE/'mechanism_refinement/adam/runs').glob('*')):
        manifest=json.loads((folder/'manifest.json').read_text())
        for path in sorted((folder/'snapshots').glob('*.npz')):
            offset=int(path.stem)
            if offset not in (0,1000,10000,20000):continue
            with np.load(path,allow_pickle=False) as z:pp=z['p']
            for i,case in enumerate(manifest['cases']):
                p=jnp.asarray(pp[i]);x,y=map(jnp.asarray,data(case['target']))
                intervention.append(dict(cohort=folder.name,offset=offset,**case,
                    **scalar(pop(p)),relative_error=float(error(p,x,y))))
    write_csv(args.output/'archived_interventions.csv',intervention)
    # Existing GD controls use the same physical population moment definitions.
    controls=[]
    for path in sorted((BASE/'population_output/evidence').glob('window_*20260925/states.csv')):
        if path.parent.name=='window_refinement_20260925':continue
        with path.open() as f:
            for row in csv.DictReader(f):
                if row['kind']!='gd' or float(row['dt'])!=.002:continue
                if int(row['width']) not in (705,1409) or int(row['seed']) not in (30,31) or row['target'] not in SIX:continue
                step=20000+int(row['offset'])
                if not 20000<=step<=125000:continue
                controls.append(dict(cohort='gd_reference',target=row['target'],seed=int(row['seed']),
                    width=int(row['width']),optimizer='gd',h=float(row['h']),step=step,
                    **{k:float(row[k]) for k in ('M','C6','C10','slope_rms','lambda_rms','relative_error','relative_eval_error')},
                    K=float(row['C10'])/float(row['C6'])**2,source=str(path)))
        sources.append(dict(path=str(path),sha256=digest(path)))
    write_csv(args.output/'gd_reference.csv',controls)
    (args.output/'facts.json').write_text(json.dumps(dict(stage='archive',rows=len(rows),
        interventions=len(intervention),gd_reference_rows=len(controls),sources=sources,
        seconds=time.monotonic()-begun),indent=2)+'\n')


def independent_initial(nref,seed):
    halo={512:96,1024:192}[nref];w=nref+2*halo+1
    bound=float(np.sqrt(6/(w+1)))
    geometry=np.random.default_rng([seed,nref]);readout=np.random.default_rng([seed,nref,24])
    return np.r_[geometry.uniform(-bound,bound,w),geometry.uniform(-bound,bound,w),
                 readout.uniform(-bound,bound,w),0.]


def cases(stage):
    if stage.startswith('wide'):
        _,width,seed=stage.split('_');width,seed=int(width),int(seed)
        nref={705:512,1409:1024}[width]
        for optimizer in (('adam','gd') if width==1409 and seed==30 else ('adam',)):
            for target in SIX:
                case=dict(cohort='wide',target=target,seed=seed,width=width,h=2/nref,
                          optimizer=optimizer,eta=.002,beta1=.9 if optimizer=='adam' else 0.,
                          beta2=.999,epsilon=1e-8,adaptive=optimizer=='adam')
                yield case,pa.initial(independent_initial(nref,seed)),{}
    elif stage.startswith('replay'):
        seed=int(stage.split('_')[1]);folder=BASE/f'adam_force_extension/raw/primary_{seed}'
        manifest=json.loads((folder/'manifest.json').read_text())
        with np.load(folder/'snapshots.npz',allow_pickle=False) as z:
            at=list(z['steps']).index(20000)
            for i,case in enumerate(manifest['cases']):
                if case['optimizer']!='adam':continue
                state=pa.initial(z['p'][i,at],z['m'][i,at],z['v'][i,at],z['count'][i,at],z['channel_m'][i,at])
                refs={int(s):z['p'][i,k].copy() for k,s in enumerate(z['steps']) if 20000<int(s)<=125000}
                yield dict(cohort='replay',width=177,h=2/128,**case),state,refs
    else:raise ValueError(stage)


def run(args):
    if jax.default_backend()!='gpu':raise RuntimeError('Use Modal GPU')
    begun=time.monotonic();rows=[];summaries=[];stopped=False
    for ci,(case,state,refs) in enumerate(cases(args.stage)):
        if time.monotonic()-begun>args.seconds-30:stopped=True;break
        x,y=map(jnp.asarray,data(case['target']));xe,ye=map(jnp.asarray,data(case['target'],8192))
        config=settings(case);start=int(state['count']);initial=scalar(pop(state['p']))
        initial_hash=targets.array_hash(np.asarray(state['p']));case_start=time.monotonic()
        prior=start;replay_errors=[];calibration=None;fork=None;case_rows=[];status='complete'
        times=sorted({start,*range(max(1000,start+1000),125001,1000),20000,25000,125000})
        times=[t for t in times if t>=start]
        for step in times:
            if step>prior:state=pa.advance(state,x,y,config,step-prior)
            d=scalar(pa.diagnostics(state,x,y,config));ledger=np.asarray(state['ledger'])
            if not all(np.isfinite(d[k]) for k in ('M','C6','relative_error')):
                status='nonfinite';break
            row=dict(**case,case_id=ci,step=step,**d,lambda_rms=case['h']*d['slope_rms'],
                **{f'sum_{k}':float(v) for k,v in zip(pa.LEDGER,ledger)},
                minimum_error=float(state['minimum_error']),hits_1pct=int(state['hits'][0]),
                hits_0p1pct=int(state['hits'][1]),hits_1e4=int(state['hits'][2]),
                unresolved=int(state['unresolved']),identity_max=float(jnp.max(state['identity'])))
            if step in (start,20000,25000,50000,75000,100000,125000):
                row['relative_eval_error']=float(error(state['p'],xe,ye))
            if step==20000:calibration=ledger.copy()
            if step==25000:
                fork=dict(d);fork['ledger']=ledger.copy()
                reference=(ledger[pa.LEDGER.index('rootC6_sum')]-calibration[pa.LEDGER.index('rootC6_sum')])/5000
                state=dict(state,minimum_error=jnp.array(jnp.inf),hits=jnp.zeros(3,dtype=jnp.int64))
            if step>25000:
                increment=ledger-fork['ledger'];elapsed=step-25000
                row['accumulated_concentration_ratio']=float(increment[pa.LEDGER.index('rootC6_sum')]/elapsed/reference)
                for q,actual in [('M',d['M']-fork['M']),('A',d['A']-fork['A']),('logC6',np.log(d['C6']/fork['C6']))]:
                    reconstructed=sum(increment[pa.LEDGER.index(q+'_'+c)] for c in pa.COMPONENTS)+increment[pa.LEDGER.index(q+'_defect')]
                    row[q+'_closure']=float(actual-reconstructed)
            if step in refs:
                replay_errors.append(float(np.linalg.norm(np.asarray(state['p'])-refs[step])/np.linalg.norm(refs[step])))
                row['archive_parameter_relative_difference']=replay_errors[-1]
            if step in (20000,25000,125000):
                np.savez_compressed(args.output/f'case{ci}_state{step}.npz',**jax.device_get(state))
            case_rows.append(row);prior=step
            if step%25000==0:print(json.dumps(dict(stage=args.stage,case=ci,target=case['target'],
                optimizer=case['optimizer'],step=step,error=d['relative_error'],C6=d['C6'],
                seconds=time.monotonic()-begun)),flush=True)
            if time.monotonic()-begun>args.seconds-15 and step<125000:
                status='budget_stopped';stopped=True;break
        rows.extend(case_rows)
        summaries.append(dict(**case,case_id=ci,start=start,end=prior,status=status,
            initial_hash=initial_hash,data_hash=targets.array_hash(np.asarray(x),np.asarray(y)),
            seconds=time.monotonic()-case_start,initial_M=initial['M'],
            archive_relative_difference_max=max(replay_errors,default=0.)))
        write_csv(args.output/'states.csv',rows);write_csv(args.output/'cases.csv',summaries)
        (args.output/'facts.json').write_text(json.dumps(dict(stage=args.stage,budget_stopped=stopped,
            cases=summaries,seconds=time.monotonic()-begun),indent=2)+'\n')
        if stopped:break


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=1800)
    args=parser.parse_args()
    if args.stage=='archive':audit(args)
    else:run(args)


if __name__=='__main__':main()
