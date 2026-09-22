"""Curate D34 mechanism tables and figures; scientific prose is authored separately."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import targets
from .mechanism import ARMS, design
from .recovery import clean, concentration, write_table


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def number(row,key):
    return float(row[key]) if row.get(key) not in (None,'') else np.nan


def key(row):
    return row['target'],int(row['seed']),row.get('arm','joint'),int(row.get('fork_step',0))


def summarize(root,archives):
    curves=[r for p in sorted(root.glob('*_geometry/frozen_curves.csv')) for r in read(p)]
    metrics=[r for p in sorted(root.glob('*_metrics/metrics.csv')) for r in read(p)]
    lookup={(key(r),int(r['step'])):r for r in curves if r['kind']=='learned' and r['updates']=='600000'}
    x=targets.grid(2048); xe=targets.grid(8192); mapping=targets.polynomial_map(x)
    endpoints=[]
    for path in archives:
        f=dict(np.load(path)); cases=json.loads(str(f['cases']))
        for i,case in enumerate(cases):
            if isinstance(case,list): case=dict(seed=case[0],target=case[1])
            case=dict(case); case.setdefault('arm','joint'); case.setdefault('fork_step',0)
            z=f['z'][i,-1]; d=f['d'][i,-1]; ye=targets.values(case['target'],xe,mapping)
            error=design(z[0],z[1],xe) @ np.r_[z[2],d]-ye
            gamma=abs(z[0]); pos=f['positive'][i,-1]; neg=f['negative'][i,-1]
            row=dict(**case,step=int(f['steps'][-1]),mean_gamma=gamma.mean(),median_gamma=np.median(gamma),
                max_gamma=gamma.max(),q90_gamma=np.quantile(gamma,.9),readout_l2=np.linalg.norm(z[2]),
                heldout_mse=np.mean(error*error),relative_heldout_mse=np.mean(error*error)/np.mean(ye*ye),
                mean_growth=np.mean(gamma-abs(f['z'][i,0,0])),positive_mean=pos.mean(),negative_mean=neg.mean(),
                path=float(f['path'][i,-1]),**concentration(pos))
            frozen=lookup.get((key(case),int(f['steps'][-1])))
            if frozen: row['frozen_relative_mse']=number(frozen,'relative_heldout_mse')
            endpoints.append(row)
    paired=[]
    for row in endpoints:
        if row['arm']=='joint': continue
        control=next(r for r in endpoints if r['arm']=='joint' and r['fork_step']==row['fork_step']
                     and r['seed']==row['seed'] and r['target']==row['target'] and r['step']==row['step'])
        paired.append(dict(**row,**{k+'_minus_joint':row[k]-control[k] for k in
            ('mean_growth','mean_gamma','heldout_mse','relative_heldout_mse','readout_l2','frozen_relative_mse') if k in row and k in control}))
    summary=[]
    for target in targets.TARGETS:
        for fork in (0,20000,100000):
            for arm in ARMS:
                rr=[r for r in endpoints if r['target']==target and r['fork_step']==fork and r['arm']==arm]
                if not rr: continue
                entry=dict(target=target,fork_step=fork,arm=arm,seeds=len(rr))
                for k in ('mean_gamma','median_gamma','max_gamma','mean_growth','heldout_mse','relative_heldout_mse','readout_l2','frozen_relative_mse'):
                    a=[r[k] for r in rr if k in r]
                    if a: entry.update({k+'_median':np.median(a),k+'_min':min(a),k+'_max':max(a)})
                summary.append(entry)
    write_table(root/'endpoints.csv',endpoints); write_table(root/'paired_interventions.csv',paired)
    write_table(root/'summary.csv',summary)
    geometry={(key(r),int(r['step']),r['kind']):number(r,'relative_heldout_mse') for r in curves
              if r['kind']!='construction_centers' and r['updates']=='600000'}
    effects=[]
    for row in curves:
        if row['kind']!='new_slopes_old_biases' or row['updates']!='600000': continue
        identity=key(row); step=int(row['step']); previous=int(row['previous_step'])
        oo=geometry[identity,previous,'learned']; nn=geometry[identity,step,'learned']
        no=geometry[identity,step,'new_slopes_old_biases']; on=geometry[identity,step,'old_slopes_new_biases']
        effects.append(dict(target=identity[0],seed=identity[1],arm=identity[2],fork_step=identity[3],
            previous_step=previous,step=step,old_geometry_error=oo,new_geometry_error=nn,
            slope_gain_at_old_bias=oo-no,bias_gain_at_old_slope=oo-on,total_gain=oo-nn,
            interaction_gain=no+on-oo-nn))
    write_table(root/'geometry_effects.csv',effects)
    peaks=[]
    for identity in sorted({key(r) for r in metrics}):
        rr=[r for r in metrics if key(r)==identity and int(r['step'])>=20000]
        if not rr: continue
        peak=max(rr,key=lambda r:number(r,'actual_slope_speed'))
        selected={k:peak[k] for k in ('target','seed','arm','fork_step','step','full_slope_norm','effective_slope_norm',
             'transient_slope_norm','effective_outward_alignment','mean_gamma','readout_l2') if k in peak}
        for k in ('readout_gain','shape_change','residual_change'):
            selected['full_force_driver_'+k]=number(peak,'full_force_driver_'+k)
        peaks.append(selected)
    write_table(root/'sampled_force_peaks.csv',peaks)
    checks={}
    for k in ('gradient_accounting_error','decomposition_error','effective_derivative_identity_error'):
        a=[number(r,k) for r in metrics if np.isfinite(number(r,k))]
        checks[k+'_max']=max(a) if a else None
    checks['states']=len(metrics)
    checks['unresolved_coarse_inverse']=sum(r['coarse_inverse_resolved']=='False' for r in metrics)
    windows=[r for p in root.glob('*_metrics/movement_windows.csv') for r in read(p)]
    if windows:
        checks['motion_identity_error_max']=max(number(r,'motion_identity_error') for r in windows)
    (root/'analysis_checks.json').write_text(json.dumps(clean(checks),indent=2)+'\n')
    plot(root,metrics,curves,endpoints)


def plot(root,metrics,curves,endpoints):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9})
    colors=dict(joint='#111827',freeze_readout='#b45309',freeze_slopes='#dc2626',
                freeze_biases='#7c3aed',slow_readout='#059669',fast_readout='#2563eb')
    labels={a:a.replace('_',' ') for a in ARMS}
    fig,axs=plt.subplots(4,5,figsize=(17,11),sharex=True)
    for col,target in enumerate(targets.TARGETS):
        for seed in range(5):
            rr=sorted([r for r in metrics if r['target']==target and int(r['seed'])==seed and
                       r['arm']=='joint' and int(r['fork_step'])==0],key=lambda r:int(r['step']))
            dense=[r for r in metrics if r['target']==target and int(r['seed'])==seed and
                   r['arm']=='joint' and int(r['fork_step'])==20000 and int(r['step'])>20000]
            if dense: rr=sorted([r for r in rr if int(r['step'])<=20000]+dense,key=lambda r:int(r['step']))
            if not rr: continue
            t=np.array([int(r['step'])*.002 for r in rr]); alpha=1 if seed==0 else .3
            for field,color,style in [('full_slope_norm','#111827','-'),('effective_slope_norm','#2563eb','--'),('transient_slope_norm','#d97706',':')]:
                axs[0,col].plot(t,[number(r,field) for r in rr],style,color=color,alpha=alpha,lw=1,
                    label=field.replace('_',' ') if seed==0 else None)
            axs[1,col].plot(t,[number(r,'readout_l2') for r in rr],color='#7c3aed',alpha=alpha)
            axs[2,col].plot(t,[number(r,'actual_outward') for r in rr],color='#059669',alpha=alpha)
            for field,color in [('mean_gamma','#111827'),('q90_gamma','#2563eb')]:
                axs[3,col].plot(t,[number(r,field) for r in rr],color=color,alpha=alpha,label=field if seed==0 else None)
        axs[0,col].set_title(target)
        axs[0,col].set_yscale('log'); axs[1,col].set_yscale('log')
        axs[2,col].set_yscale('symlog',linthresh=1e-7); axs[2,col].axhline(0,color='#d1d5db',lw=.5)
        axs[3,col].set(xlabel='Physical time',xscale='symlog'); axs[3,col].set_xscale('symlog',linthresh=1)
    for i,label in enumerate(('Slope-force norm','Readout norm','Mean outward velocity','Slope magnitude')): axs[i,0].set_ylabel(label)
    axs[0,0].legend(fontsize=6); axs[3,0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(root/'force_and_growth.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(1,5,figsize=(17,3.8),sharey=True)
    for col,target in enumerate(targets.TARGETS):
        for horizon in (20000,100000,600000):
            rr={float(r['gamma']):r for r in curves if r['target']==target and r['kind']=='construction_centers' and int(r['updates'])==horizon}
            xx=sorted(rr)
            axs[col].plot(xx,[number(rr[g],'relative_heldout_mse') for g in xx],'o-',label=f'{horizon:,} updates',ms=3)
        axs[col].axvline(16,color='#6b7280',ls=':',label='Construction reference')
        axs[col].set(title=target,xscale='log',yscale='log',xlabel='Common frozen gamma')
    axs[0].set_ylabel('Independent-grid relative MSE'); axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(root/'frozen_gamma_reference.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(1,5,figsize=(17,3.8))
    for col,target in enumerate(targets.TARGETS):
        for kind,color,label in [('learned','#111827','Current slopes and biases'),('new_slopes_old_biases','#2563eb','New slopes, previous biases'),
                                  ('old_slopes_new_biases','#d97706','Previous slopes, new biases')]:
            for seed in range(5):
                rr=sorted([r for r in curves if r['target']==target and r['kind']==kind and r.get('fork_step')=='0'
                           and r.get('seed')==str(seed) and r['updates']=='600000'],key=lambda r:int(r['step']))
                if rr: axs[col].plot([int(r['step'])*.002 for r in rr],[number(r,'relative_heldout_mse') for r in rr],
                    'o-',color=color,alpha=.85 if seed==0 else .3,ms=3,label=label if seed==0 else None)
        axs[col].set(title=target,xscale='symlog',yscale='log',ylim=(1e-4,1.1),xlabel='Physical time of geometry snapshot')
        axs[col].set_xscale('symlog',linthresh=1)
    axs[0].set_ylabel('Frozen readout relative MSE after 600k'); axs[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(root/'learned_geometry_usefulness.png',dpi=160); plt.close(fig)
    for field,filename,label in [('relative_heldout_mse','intervention_fit','Actual relative MSE'),
                                  ('frozen_relative_mse','intervention_geometry','Frozen readout relative MSE after 600k')]:
        if not any(r['arm']!='joint' for r in endpoints): continue
        fig,axs=plt.subplots(2,5,figsize=(17,7.5))
        for ri,fork in enumerate((20000,100000)):
            for col,target in enumerate(targets.TARGETS):
                for ai,arm in enumerate(ARMS):
                    a=[r[field] for r in endpoints if r['target']==target and r['fork_step']==fork and r['arm']==arm and field in r]
                    if not a: continue
                    axs[ri,col].scatter(np.full(len(a),ai),a,color=colors[arm],s=18,alpha=.65)
                    axs[ri,col].plot([ai-.2,ai+.2],[np.median(a)]*2,color=colors[arm],lw=2)
                axs[ri,col].set(title=f'{target}; fork {fork:,}',yscale='log',xticks=range(len(ARMS)))
                axs[ri,col].set_xticklabels([labels[a] for a in ARMS],rotation=60,ha='right',fontsize=7)
            axs[ri,0].set_ylabel(label)
        fig.tight_layout(); fig.savefig(root/(filename+'.png'),dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--archives',type=Path,nargs='+',required=True)
    args=parser.parse_args(); summarize(args.root,args.archives)


if __name__=='__main__': main()
