"""Read saved trajectories; export scientific tables and compact plot evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from . import references, targets
from .run import TRACE, DEGREES, save, write_json


def clean(value):
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def table(path, rows):
    if not rows: return
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        writer.writerows(clean(rows))


def load_trace(folder,end):
    arrays=[];expected=0
    for path in sorted(folder.glob('trace_*.npz')):
        with np.load(path) as data:
            start,stop=int(data['start']),int(data['stop'])
            if start>=end: break
            if start!=expected: raise ValueError(f'Noncontiguous trace: {path}')
            arrays.append(data['trace'][:,:min(stop,end)-start])
            expected=min(stop,end)
    return np.concatenate(arrays,axis=1) if arrays else None


def load_snapshots(folder,end):
    blocks={};steps=[]
    for path in sorted(folder.glob('snapshots_*.npz')):
        with np.load(path) as data:
            mask=data['steps']<end
            if not np.any(mask): continue
            steps.extend(data['steps'][mask].tolist())
            for key in data.files:
                if key!='steps': blocks.setdefault(key,[]).append(data[key][:,mask])
    return np.array(steps),{key:np.concatenate(value,axis=1) for key,value in blocks.items()}


def probe_indices(steps):
    desired=np.unique(np.r_[np.arange(21),np.rint(np.geomspace(20,max(20,steps[-1]),61)).astype(int)])
    return np.unique([np.argmin(abs(steps-v)) for v in desired])


def sample_probe(z,d,g,gd,x,y,kappa,degree):
    """Independent sample-space loss and exact local coarse-moment velocities."""
    a,b,c=z;pre=x[:,None]*a+b
    if degree==0:
        h=np.tanh(pre);ee=np.exp(-2*abs(pre));derivative=4*ee/(1+ee)**2
    else:
        h=sum(references.COEFFICIENTS[k]*pre**k for k in range(1,degree+1,2))
        derivative=sum(k*references.COEFFICIENTS[k]*pre**(k-1) for k in range(1,degree+1,2))
    e=h@c+d-y;sigma=np.sqrt(np.mean(x*x));m=len(x)
    rc=h@g[2]+gd
    geometry=derivative@(c*g[1])+x*(derivative@(c*g[0]))
    Vv=-kappa*np.array([np.mean(rc),x@rc/(m*sigma)])
    Vq=-np.array([np.mean(geometry),x@geometry/(m*sigma)])
    weighted=e[:,None]*derivative
    gradient=np.stack((c*(x@weighted)/m,c*np.mean(weighted,axis=0),h.T@e/m))
    return dict(sample_half_mse=.5*np.mean(e*e),Vv=Vv,Vq=Vq,
                gradient=gradient,gradient_d=np.mean(e),
                Vbias=np.array([-kappa*gd,0.]),
                above_half=np.mean(abs(pre)>.5),above_one=np.mean(abs(pre)>1),
                above_radius=np.mean(abs(pre)>np.pi/2))


def analyze_bundle(folder,out,end,probes=True):
    manifest=json.loads((folder/'manifest.json').read_text());cfg=manifest['configuration']
    with np.load(folder/'initial.npz') as data:
        initial={k:data[k] for k in data.files}
    rows=[];errors=[];probe_rows=[];matched=[];curve_arrays={}
    actual=load_trace(folder/'p0',end)
    if actual is None: return [],[],[],[]
    actual_steps,actual_snap=load_snapshots(folder/'p0',end)
    sigma=np.sqrt(np.mean(initial['x']**2));mean0=np.mean(abs(initial['z'][0]))
    selected=np.unique(np.r_[np.arange(min(21,actual.shape[1])),np.rint(np.geomspace(21,actual.shape[1],120)).astype(int)-1])
    curve_arrays['steps']=selected
    curve_arrays['configuration']=np.array(json.dumps(cfg))
    curve_arrays['mean_gamma0']=np.array(mean0)
    curve_arrays['target_energy']=initial['sy']
    curve_arrays['cases']=np.array(json.dumps(manifest['cases']))
    for degree in DEGREES:
        branch=folder/f'p{degree}';trace=actual if degree==0 else load_trace(branch,end)
        if trace is None: continue
        steps,snap=(actual_steps,actual_snap) if degree==0 else load_snapshots(branch,end)
        if not len(steps): continue
        valid_selected=selected[selected<trace.shape[1]]
        curve_arrays[f'p{degree}_trace']=trace[:,valid_selected]
        curve_arrays[f'p{degree}_steps']=valid_selected
        status=json.loads((branch/'status.json').read_text())
        checkpoint=branch/f'state_{end:09d}.npz'
        if not checkpoint.exists(): checkpoint=branch/'state.npz'
        with np.load(checkpoint) as data:
            terminal={k:data[k] for k in data.files}
        for i,case in enumerate(manifest['cases']):
            valid=np.flatnonzero(np.isfinite(trace[i,:,0]))
            if not len(valid): continue
            last=int(valid[-1]);failure=int(status['failed_steps'][i]);finite_until=min(trace.shape[1],failure-1) if failure else trace.shape[1]
            window=trace[i,max(0,last-19999):last+1]
            z=terminal['z'][i];d=float(terminal['d'][i])
            xe=targets.grid(8192);ye=targets.values(case['target'],xe,initial['mapping'])
            with np.errstate(over='ignore',invalid='ignore'):
                evaluation=sample_probe(z,d,np.zeros_like(z),0.,xe,ye,case['kappa'],degree)
            coarse=np.linalg.norm(trace[i,:,31:33]/np.array([1.,sigma]),axis=1)
            R=np.sqrt(np.where(trace[i,:,0]>=0,2*trace[i,:,0],np.nan))
            info={key:case[key] for key in ('target','kappa','seed','n','m','eta_geometry','eta_readout')}
            info.update(bundle=folder.name,degree=degree,requested_end=end,trace_states=trace.shape[1],
                completed_updates=finite_until,failed_update=failure if failure and failure<=end else 0,
                complete=bool(finite_until>=end),checkpoint_step=int(terminal['step']),
                initial_mean_gamma=mean0,mean_gamma=float(np.mean(abs(z[0]))),
                median_gamma=float(np.median(abs(z[0]))),mean_lambda=float(np.mean(abs(z[0]))*2/cfg['n']),
                signed_mean_change=float(np.mean(abs(z[0]))-mean0),
                last_preupdate_mse=float(2*trace[i,last,0]),window_mean_mse=float(2*np.mean(window[:,0])),
                window_median_mse=float(2*np.median(window[:,0])),
                heldout_mse=float(2*evaluation['sample_half_mse']),
                heldout_relative_l2=float(np.sqrt(2*evaluation['sample_half_mse']/np.mean(ye*ye))),
                initial_coarse_norm=coarse[0],last_coarse_norm=coarse[last],
                normalized_slope_gradient=trace[i,last,12]/R[last] if R[last]>0 else np.nan,
                cumulative_coarse_motion=cfg['eta']*np.sum(trace[i,valid,16]),
                cumulative_remainder_motion=cfg['eta']*np.sum(trace[i,valid,17]),
                cumulative_crossing_remainder=np.sum(trace[i,valid,19]),
                recorded_signed_motion=np.sum(trace[i,valid,18]),
                loss_increases=int(np.sum(np.diff(trace[i,valid,0])>0)),
                max_preactivation=np.max(trace[i,valid,11]),
                first_radius_exceedance=next((int(j) for j in valid if trace[i,j,11]>=np.pi/2),None))
            denominator=cfg['eta']*(trace[i,:-1,12]**2+trace[i,:-1,13]**2+
                case['kappa']*(trace[i,:-1,14]**2+trace[i,:-1,15]**2))
            stability_floor=1e-24*max(1.,float(denominator[0]))
            mask=np.isfinite(denominator)&(denominator>stability_floor)&np.isfinite(trace[i,1:,0])
            descent=(trace[i,:-1,0]-trace[i,1:,0])[mask]/denominator[mask]
            info.update(descent_ratio_floor=stability_floor,descent_ratio_count=len(descent),
                descent_ratio_min=float(np.min(descent)) if len(descent) else None,
                descent_ratio_median=float(np.median(descent)) if len(descent) else None)
            for threshold in (.1,.01):
                events=np.flatnonzero(coarse<=threshold*coarse[0]);event=int(events[0]) if len(events) else None
                label='coarse_10pct' if threshold==.1 else 'coarse_1pct'
                info[label+'_step']=event
                info[label+'_persists20']=bool(event is not None and event+20<=len(coarse) and np.all(coarse[event:event+20]<=threshold*coarse[0]))
                info[label+'_mean_gamma']=float(trace[i,event,1]) if event is not None else None
            for t,label in enumerate(TRACE[25:31]):
                info[label]=float(np.mean(abs(z[0])>=([1,4,16,.05*cfg['n']/2,.1*cfg['n']/2,.25*cfg['n']/2][t])))
                for fraction in (.1,.5):
                    crossing=np.flatnonzero(trace[i,:,25+t]>=fraction)
                    info[f'{label}_population{fraction}_step']=int(crossing[0]) if len(crossing) else None
            rows.append(info)

        if degree:
            common,ai,ri=np.intersect1d(actual_steps,steps,return_indices=True)
            gp=snap['gradient'][:,ri,0];gt=actual_snap['gradient'][:,ai,0]
            error=np.linalg.norm(gp-gt,axis=-1)
            floor=1e-12*np.maximum(1,np.linalg.norm(actual_snap['gradient'][:,0,0],axis=-1))
            denom=np.maximum(np.linalg.norm(gt,axis=-1),floor[:,None])
            dactual=np.mean(abs(actual_snap['z'][:,ai,0]),axis=-1)-mean0
            dp=np.mean(abs(snap['z'][:,ri,0]),axis=-1)-mean0
            for i,case in enumerate(manifest['cases']):
                for j,step in enumerate(common):
                    if actual_snap['failed'][i,ai[j]] or snap['failed'][i,ri[j]]: continue
                    errors.append(dict(bundle=folder.name,target=case['target'],seed=case['seed'],n=cfg['n'],
                        kappa=case['kappa'],degree=degree,step=int(step),
                        gradient_absolute_error=error[i,j],gradient_relative_error=error[i,j]/denom[i,j],
                        gradient_floor_active=bool(np.linalg.norm(gt[i,j])<=floor[i]),
                        signed_change_actual=dactual[i,j],signed_change_reference=dp[i,j],
                        signed_change_absolute_error=abs(dp[i,j]-dactual[i,j]),
                        signed_change_relative_error=abs(dp[i,j]-dactual[i,j])/max(abs(dactual[i,j]),1e-12*max(1.,mean0)),
                        signed_change_floor_active=bool(abs(dactual[i,j])<=1e-12*max(1.,mean0)),
                        signed_force_actual=-np.mean(np.sign(actual_snap['z'][i,ai[j],0])*gt[i,j]),
                        signed_force_reference=-np.mean(np.sign(snap['z'][i,ri[j],0])*gp[i,j]),
                        residual_moment_l2_error=np.linalg.norm(snap['moments'][i,ri[j]]-actual_snap['moments'][i,ai[j]]),
                        median_gamma_error=np.median(abs(snap['z'][i,ri[j],0]))-np.median(abs(actual_snap['z'][i,ai[j],0])),
                        slope_parameter_l2_error=np.linalg.norm(snap['z'][i,ri[j],0]-actual_snap['z'][i,ai[j],0])))
            picked=probe_indices(common)
            valid=(actual_snap['failed'][:,ai]==0)&(snap['failed'][:,ri]==0)
            curve_arrays[f'p{degree}_error_steps']=common[picked]
            curve_arrays[f'p{degree}_gradient_error']=np.where(valid[:,picked],error[:,picked]/denom[:,picked],np.nan)
            curve_arrays[f'p{degree}_signed_error']=np.where(valid[:,picked],dp[:,picked]-dactual[:,picked],np.nan)

        lookup={(c['target'],c['kappa']):i for i,c in enumerate(manifest['cases'])}
        for left in ('moment3','moment5'):
            for kappa in cfg['ratios']:
                if (left,kappa) not in lookup or ('moment9',kappa) not in lookup: continue
                a,b=lookup[left,kappa],lookup['moment9',kappa]
                for j,step in enumerate(steps):
                    if snap['failed'][a,j] or snap['failed'][b,j]: continue
                    matched.append(dict(bundle=folder.name,n=cfg['n'],seed=cfg['seed'],degree=degree,
                        left=left,right='moment9',kappa=kappa,step=int(step),
                        signed_mean_gamma_difference=np.mean(abs(snap['z'][a,j,0]))-np.mean(abs(snap['z'][b,j,0])),
                        slope_gradient_difference=np.linalg.norm(snap['gradient'][a,j,0]-snap['gradient'][b,j,0]),
                        slope_parameter_difference=np.linalg.norm(snap['z'][a,j,0]-snap['z'][b,j,0]),
                        coarse_residual_difference=np.linalg.norm((snap['moments'][a,j,:2]-snap['moments'][b,j,:2])/[1,sigma])))

        if probes:
            for j in probe_indices(steps):
                step=int(steps[j])
                for i,case in enumerate(manifest['cases']):
                    if snap['failed'][i,j]: continue
                    z=snap['z'][i,j];d=snap['d'][i,j];g=snap['gradient'][i,j];gd=snap['gradient_d'][i,j]
                    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                        probe=sample_probe(z,d,g,gd,initial['x'],initial['y'][i],case['kappa'],degree)
                    row=dict(bundle=folder.name,target=case['target'],seed=case['seed'],n=cfg['n'],degree=degree,kappa=case['kappa'],step=step,
                             sample_half_mse=probe['sample_half_mse'],moment_or_stored_half_mse=float(snap['loss'][i,j]),
                             loss_discrepancy=probe['sample_half_mse']-float(snap['loss'][i,j]),
                             fraction_preactivation_above_half=probe['above_half'],fraction_preactivation_above_one=probe['above_one'],
                             fraction_preactivation_above_radius=probe['above_radius'])
                    G=z@z.T
                    for a,b,label in ((0,0,'aa'),(1,1,'bb'),(2,2,'cc'),(0,1,'ab'),(0,2,'ac'),(1,2,'bc')):
                        row[f'gram_{label}']=G[a,b]
                    row.update(Kq_00=G[2,2],Kq_11=sigma**2*G[2,2],
                               Kv_00=1+G[1,1],Kv_01=sigma*G[0,1],Kv_11=sigma**2*G[0,0])
                    for name in ('Vv','Vq','Vbias'):
                        row[name+'_0'],row[name+'_1']=probe[name]
                    mcoarse=snap['moments'][i,j,:2]/[1.,sigma]
                    for name in ('Vv','Vq','Vbias'):
                        row[name+'_coarse_energy_removal']=-mcoarse@probe[name]
                    row['sample_gradient_error']=np.linalg.norm(probe['gradient']-g)
                    if degree:
                        with np.errstate(over='ignore',invalid='ignore'):
                            true=sample_probe(z,d,g,gd,initial['x'],initial['y'][i],case['kappa'],0)
                        row['tanh_at_reference_slope_gradient_defect']=np.linalg.norm(true['gradient'][0]-g[0])
                        row['tanh_at_reference_half_mse']=true['sample_half_mse']
                    if step+1<trace.shape[1] and np.isfinite(trace[i,step+1,0]):
                        fd=(trace[i,step+1,31:33]-trace[i,step,31:33])/np.array([1.,sigma])/cfg['eta']
                        row['coarse_discrete_derivative_error']=np.linalg.norm(fd-probe['Vv']-probe['Vq'])
                    if degree==0 and probe['sample_half_mse']>0:
                        R=np.sqrt(2*probe['sample_half_mse']);mom=snap['moments'][i,j]
                        for order in (2,4,6,8):
                            approximation,rho=references.slope_interval(z,mom,R,order)
                            bound=(1/np.cos(rho)**2/rho**order*np.sqrt(np.mean(initial['x']**(2*order+2)))
                                   *np.linalg.norm(z[2]*z[0]**order))
                            row[f'pointwise_p{order}_error']=np.linalg.norm(g[0]/R-approximation)
                            row[f'pointwise_p{order}_bound']=bound
                    probe_rows.append(row)
    save(out/f'{folder.name}_curves.npz',**curve_arrays)
    return rows,errors,probe_rows,matched


def contrasts(rows):
    baseline={(r['bundle'],r['target'],r['degree']):r for r in rows if r['kappa']==1}
    actual={(r['bundle'],r['target'],r['kappa']):r for r in rows if r['degree']==0}
    result=[]
    for row in rows:
        anchor=baseline.get((row['bundle'],row['target'],row['degree']))
        true=actual.get((row['bundle'],row['target'],row['kappa']))
        true_anchor=baseline.get((row['bundle'],row['target'],0))
        if not all(v and v['complete'] for v in (row,anchor,true,true_anchor)): continue
        observed=true['signed_mean_change']-true_anchor['signed_mean_change']
        predicted=row['signed_mean_change']-anchor['signed_mean_change']
        result.append(dict(bundle=row['bundle'],target=row['target'],n=row['n'],seed=row['seed'],
            degree=row['degree'],kappa=row['kappa'],actual_contrast=observed,
            predicted_contrast=predicted,contrast_error=predicted-observed))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--end',type=int,default=20000)
    parser.add_argument('--bundles',nargs='*')
    parser.add_argument('--skip-probes',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[];errors=[];probes=[];matched=[]
    folders=[p.parent for p in sorted(args.root.glob('*/manifest.json'))]
    if args.bundles: folders=[p for p in folders if p.name in args.bundles]
    for folder in folders:
        a,b,c,d=analyze_bundle(folder,args.output,args.end,not args.skip_probes)
        rows.extend(a);errors.extend(b);probes.extend(c);matched.extend(d)
        print(json.dumps(dict(bundle=folder.name,rows=len(a),errors=len(b),probes=len(c))),flush=True)
    table(args.output/'summary.csv',rows)
    table(args.output/'reference_metrics.csv',errors)
    table(args.output/'sample_probes.csv',probes)
    table(args.output/'rate_contrasts.csv',contrasts(rows))
    table(args.output/'matched_target_metrics.csv',matched)
    write_json(args.output/'summary.json',clean(rows))
    write_json(args.output/'analysis_manifest.json',dict(end=args.end,bundles=[p.name for p in folders],
        rows=len(rows),reference_rows=len(errors),probe_rows=len(probes),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        probe_schedule='steps 0 through 20 and nearest saved states to 61 logarithmic times',
        evidence_role='descriptive optimization diagnostics; no checkpoint selection'))


if __name__=='__main__':
    main()
