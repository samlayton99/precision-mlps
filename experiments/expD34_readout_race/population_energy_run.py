"""Saved-state replays and matched concentration interventions; Modal GPU only."""
import argparse
import csv
import json
from pathlib import Path
import time

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from . import population_concentration_analysis as archive
from . import population_energy_dynamics as ed
from . import population_energy_interventions as surgery
from . import mechanism_persistence_kernel as kernel
from .population_window_run import SIX, evaluation
from . import adam_forces as af, effective_feedback_holdout as holdout


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def selected(rows,study):
    r=rows[0]
    if study=='audit':return True
    if r['kind']!='gd' or r['dt']!=.002 or r['target'] not in SIX:return False
    if study in ('development','refinement'):
        return r['study']=='development' and (study!='refinement' or r['target'] in ('moment5','gauss_left','step_right','kink_abs'))
    if study=='validation':return r['width']==705 and r['seed']==33
    if study=='wide':return r['width']==1409 and r['seed']==31
    raise ValueError(study)


def run(args):
    if jax.default_backend()!='gpu':raise RuntimeError('Run on Modal GPU')
    begun=time.monotonic();args.output.mkdir(parents=True,exist_ok=False)
    groups=[r for r in archive.load(args.inputs) if selected(r,args.study)]
    rows_out=[];summaries=[];attempts=[];saved={};budget_stopped=False
    for source_index,rows in enumerate(groups):
        row=next(r for r in rows if r['offset']==5000)
        x,y,target=archive.target_data(row['target'])
        data=holdout.data if row['target'] in holdout.TARGETS else af.data
        xe,ye,_,_=data(row['target'],8192)
        x,y,xe,ye=map(jnp.asarray,(x,y,xe,ye))
        with np.load(row['source'],allow_pickle=False) as inputs:
            prefix=f"case{int(row['case_index'])}_{row['kind']}_"
            p0=inputs[prefix+'5000'].copy()
            references={55000:inputs[prefix+'60000'].copy(),105000:inputs[prefix+'110000'].copy()}
        baseline=kernel.decomposition(jnp.asarray(p0),x,y)
        if args.study=='audit':
            variants=[('baseline',1.,-1,p0,dict(valid=True,gram_relative_error=0.))]
            kinds=[(row['kind'],row['dt'])]
            offsets=sorted({0,1000,5000,*range(10000,105001,5000)})
        else:
            variants=[]
            if args.study!='refinement':variants.append(('baseline',1.,-1,p0,dict(valid=True,gram_relative_error=0.)))
            for path in ((0,) if args.study=='refinement' else (0,1)):
                for factor in ((0,10) if args.study=='refinement' else (0,2,4,10)):
                    p,record=surgery.redistribute(p0,factor,path)
                    arm='dispersed' if factor==0 else f'concentrated{factor}'
                    attempts.append(dict(target=row['target'],width=row['width'],seed=row['seed'],arm=arm,**record))
                    if record['valid']:variants.append((arm,factor,path,p,record))
            kinds=[('gd',.001 if args.study=='refinement' else .002),
                   ('effective',.01 if args.study=='refinement' else .02)]
            offsets=[0,100,500,1000,2000,5000,10000,20000,50000,100000]
        for arm,factor,path,initial,record in variants:
            for kind,dt in kinds:
                if time.monotonic()-begun>args.max_seconds-60:
                    budget_stopped=True;break
                case_id=len(summaries)
                identity=dict(study=args.study,case_id=case_id,source_study=row['study'],
                              target=row['target'],width=row['width'],seed=row['seed'],
                              kind=kind,dt=dt,arm=arm,factor=factor,path=path,h=row['h'])
                p=jnp.asarray(initial);ledger=jnp.zeros(len(ed.LEDGER))
                first=ed.diagnostics(p,x,y,kind)
                state=kernel.decomposition(p,x,y)
                load=kernel._project(state['J'].T@baseline['eH']/len(x),state['JC'],state['gram'])
                controls=dict(gram_relative_error=record['gram_relative_error'],
                    fixed_residual_force=float(jnp.linalg.norm(load)),
                    fixed_residual_slope_force=float(jnp.linalg.norm(load[:int(row['width'])])),
                    fine_output_shift=float(jnp.sqrt(jnp.mean((state['fH']-baseline['fH'])**2))),
                    initial_coarse_shift=float(jnp.linalg.norm(state['basis'].T@(kernel.output(p,x)-kernel.output(jnp.asarray(p0),x))/len(x))))
                start=time.monotonic();status='complete';replay_error=0.;closure=0.
                case_rows=[]
                for i,offset in enumerate(offsets):
                    if i:
                        steps=round(.002*(offset-offsets[i-1])/dt)
                        p,ledger=ed.advance(p,ledger,x,y,dt,steps,kind)
                    d={k:float(v) for k,v in jax.device_get(ed.diagnostics(p,x,y,kind)).items()}
                    if not np.isfinite(d['M']) or d['coarse_min']<=0:
                        status='nonfinite_or_singular';break
                    if args.study=='audit' and offset in references:
                        err=float(np.linalg.norm(np.asarray(p)-references[offset])/np.linalg.norm(references[offset]))
                        replay_error=max(replay_error,err)
                    acc={f'integral_{key}':float(value) for key,value in zip(ed.LEDGER,jax.device_get(ledger),strict=True)}
                    for k in ed.ORDERS:
                        predicted=sum(acc[f'integral_logC{k}_{c}'] for c in ed.COMPONENTS)+acc[f'integral_discrete_logC{k}']
                        defect=abs(np.log(d[f'C{k}']/float(first[f'C{k}']))-predicted)
                        closure=max(closure,float(defect))
                    item=dict(**identity,offset=offset,time=.002*offset,
                              lambda_rms=row['h']*d['slope_rms'],
                              relative_eval_error=float(evaluation(p,xe,ye)),**d,**acc)
                    rows_out.append(item);case_rows.append(item)
                    if args.study!='audit' and offset in (0,100000):saved[f'case{case_id}_{offset}']=np.asarray(p)
                    if time.monotonic()-begun>args.max_seconds-30:
                        status='budget_limit';budget_stopped=True;break
                summary=dict(**identity,**controls,status=status,completed_offset=case_rows[-1]['offset'] if case_rows else 0,
                    seconds=time.monotonic()-start,log_moment_closure=closure,replay_relative_error=replay_error)
                if case_rows:
                    for name in ('M','C6','K6','q','jacobian_exact','lambda_rms','relative_eval_error'):
                        summary[f'initial_{name}']=case_rows[0][name]
                        summary[f'final_{name}']=case_rows[-1][name]
                summaries.append(summary)
                print(json.dumps(summary,allow_nan=False),flush=True)
                # Preserve completed cases before launching the next continuation.
                write_csv(args.output/'states.csv',rows_out)
                (args.output/'facts.json').write_text(json.dumps(dict(study=args.study,seconds=time.monotonic()-begun,
                    budget_stopped=budget_stopped,cases=summaries,attempts=attempts),indent=2,allow_nan=False)+'\n')
                if budget_stopped:break
            if budget_stopped:break
        if budget_stopped:break
    if saved:np.savez_compressed(args.output/'sparse_states.npz',**saved)
    print(json.dumps(dict(study=args.study,cases=len(summaries),seconds=time.monotonic()-begun,budget_stopped=budget_stopped)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--study',required=True)
    parser.add_argument('--max-seconds',type=float,required=True)
    run(parser.parse_args())
