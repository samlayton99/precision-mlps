"""Small scalar-only evidence analysis and figures; execute on Modal CPU."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TARGETS=('moment5','mixed_sine','gauss_left','bump_right','step_right','kink_abs')
LABELS=('Degree five','Mixed sine','Gaussian','Bump','Step','Kink')
ARMS=('native','fine10','balanced','balanced10','frozen10','frozen_balanced10')


def read(path):
    if not path.exists():return []
    with path.open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in r.items():
            if v in ('True','False'):r[k]=v=='True'
            else:
                try:r[k]=float(v)
                except (ValueError,TypeError):pass
    return rows


def write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def ratio(a,b):return a/b if np.isfinite(b) and b!=0 else float('nan')


def distribution(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    return dict(n=len(a),minimum=float(np.min(a)),median=float(np.median(a)),maximum=float(np.max(a))) if len(a) else dict(n=0)


def save(fig,output,name):
    fig.savefig(output/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(output/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def analyze(args):
    output=args.output;output.mkdir(exist_ok=True,parents=True)
    states=[];curves=[];probes=[];runs=[];receipts=[]
    newer=args.inputs/'audit_metrics'
    use_new_audit=(newer/'execution.json').exists() and json.loads((newer/'execution.json').read_text())['returncode']==0
    for folder in sorted(args.inputs.iterdir()):
        if not folder.is_dir() or not (folder/'execution.json').exists():continue
        receipt=json.loads((folder/'execution.json').read_text());receipts.append(dict(folder=folder.name,**{k:receipt[k] for k in ('stage','platform','seconds','returncode')}))
        if receipt['returncode']:continue
        if receipt['stage'] not in ('audit','gd','adam','followup'):continue
        if receipt['stage']=='audit' and use_new_audit and folder.name!='audit_metrics':continue
        for name,dest in (('states',states),('curvature',curves),('probes',probes),('runs',runs)):
            dest.extend(dict(stage=receipt['stage'],folder=folder.name,**r) for r in read(folder/(name+'.csv')))
    # Reject duplicate scientific instances rather than quietly counting reruns.
    for rows,fields in ((states,('stage','case_key','age','arm','offset')),
                        (probes,('stage','case_key','age','arm','offset','proposal','norm_matched'))):
        keys=[tuple(r[k] for k in fields) for r in rows]
        assert len(keys)==len(set(keys)), 'Duplicate completed scientific instances'
    audit=[r for r in probes if r['stage']=='audit']
    source=[]
    for r in audit:
        denom=r['z_change']**2
        magnitude=r['linear_coarse']+r['equilibrium_change']+r['nonlinear_coarse']
        source.append(dict(**r,linear_share=ratio(r['linear_z_pairing'],denom),
            equilibrium_share=ratio(r['equilibrium_z_pairing'],denom),
            nonlinear_share=ratio(r['remainder_z_pairing'],denom),
            linear_magnitude_share=ratio(r['linear_coarse'],magnitude),
            equilibrium_magnitude_share=ratio(r['equilibrium_change'],magnitude),
            nonlinear_magnitude_share=ratio(r['nonlinear_coarse'],magnitude)))
    write(output/'tracking_sources.csv',source)
    paired=[]
    for end in (10000,20000):
        subset=[r for r in states if r['stage']=='adam' and r['offset']==end and r['alive']]
        native={r['case_key']:r for r in subset if r['arm']=='native'}
        starts={(r['case_key'],r['arm']):r for r in states if r['stage']=='adam' and r['offset']==0}
        for r in subset:
            if r['case_key'] not in native:continue
            n=native[r['case_key']];s=starts[(r['case_key'],r['arm'])]
            row={k:r[k] for k in ('case_key','target','width','seed','arm','offset')}
            for key in ('lambda_rms','relative_error','minimum_error','fine_slope_path','fine_hidden_path','z2_sum','coarse_fine2_sum','raw_tracking2_sum'):
                row[key]=r[key];row[key+'_ratio']=ratio(r[key],n[key])
            row.update(relative_eval_error=r.get('relative_eval_error',np.nan),
                lambda_gain=r['lambda_rms']-s['lambda_rms'],
                lambda_gain_over_native=r['lambda_rms']-n['lambda_rms'],
                fine_A=r['A_fine'],tracking_A=r['A_tracking'],step_A=r['A_step'],
                tracking_to_fine_A=ratio(r['A_tracking'],r['A_fine']),step_to_fine_A=ratio(r['A_step'],r['A_fine']),
                tracking_lag=ratio(r['z_lag_dot'],r['z2_sum']),
                coherence=r['fine_coherence'],mean_shadow_gain=r['shadow_gain_sum']/end,
                shadow_gain_ratio=ratio(r['shadow_gain_sum'],n['shadow_gain_sum']),
                equilibrium_movement=r['equilibrium_movement'],nonlinear_coarse_movement=r['nonlinear_coarse_movement'],
                hits_1pct=r['hits_1pct'],hits_1e6=r['hits_1e6'],loss_increases=r['loss_increases'])
            paired.append(row)
    write(output/'adam_pairs.csv',paired)
    gd=[]
    for r in runs:
        if r['stage']!='gd':continue
        seq=[s for s in states if s['stage']=='gd' and s['case_key']==r['case_key'] and s['arm']==r['arm']]
        seq.sort(key=lambda s:s['offset']);s,e=seq[0],seq[-1]
        cs=[c for c in curves if c['stage']=='gd' and c['case_key']==r['case_key'] and c['arm']==r['arm']]
        first=min(cs,key=lambda c:c['offset'])
        gd.append(dict(**r,lambda_rms=e['lambda_rms'],lambda_gain=e['lambda_rms']-s['lambda_rms'],
            lambda_ratio=ratio(e['lambda_rms'],s['lambda_rms']),error_ratio=ratio(e['relative_error'],s['relative_error']),
            best_error_ratio=ratio(e['minimum_error'],s['relative_error']),
            relative_eval_error=e.get('relative_eval_error',np.nan),minimum_error=e['minimum_error'],
            fine_A=e['A_fine'],tracking_A=e['A_tracking'],step_A=e['A_step'],
            tracking_to_fine_A=ratio(e['A_tracking'],e['A_fine']),step_to_fine_A=ratio(e['A_step'],e['A_fine']),loss_increases=e['loss_increases'],
            tracking_lag=ratio(e['z_lag_dot'],e['z2_sum']),
            mean_shadow_gain=ratio(e['shadow_gain_sum'],e['offset']),
            initial_stability_ratio=first.get('raw_stability_ratio',np.nan),initial_coarse_overlap=first.get('raw_coarse_overlap',np.nan),
            final_stability_ratio=max(cs,key=lambda c:c['offset']).get('raw_stability_ratio',np.nan),
            flow_time=e['flow_time'],hits_1pct=e['hits_1pct'],hits_1e6=e['hits_1e6']))
    write(output/'gd_endpoints.csv',gd)
    # Common recorded flow-time interval. This is an interpolation diagnostic,
    # not an extra run or an equal-accuracy comparison.
    flow=[]
    for case in sorted(set(r['case_key'] for r in gd)):
        complete=[r for r in gd if r['case_key']==case and r['complete'] and r['tracking_factor']==1]
        if len(complete)<2:continue
        horizon=min(r['flow_time'] for r in complete)
        for r in complete:
            seq=sorted([s for s in states if s['stage']=='gd' and s['case_key']==case and s['arm']==r['arm']],key=lambda s:s['flow_time'])
            ts=np.array([s['flow_time'] for s in seq]);hi=int(np.searchsorted(ts,horizon,side='left'));lo=max(0,hi-1)
            flow.append(dict(case_key=case,arm=r['arm'],target=r['target'],origin=r['origin'],width=r['width'],seed=r['seed'],
                flow_time=horizon,bracket_width=ts[hi]-ts[lo],
                **{key:float(np.interp(horizon,ts,[s[key] for s in seq])) for key in ('lambda_rms','relative_error')}))
    write(output/'gd_common_flow.csv',flow)
    contrasts=[]
    for end in (10000,20000):
        for left,right in (('balanced','native'),('balanced10','fine10'),('frozen_balanced10','frozen10')):
            controls={r['case_key']:r for r in paired if r['offset']==end and r['arm']==right}
            for r in paired:
                if r['offset']!=end or r['arm']!=left or r['case_key'] not in controls:continue
                b=controls[r['case_key']]
                contrasts.append(dict(case_key=r['case_key'],target=r['target'],width=r['width'],seed=r['seed'],
                    offset=end,left=left,right=right,
                    **{key:ratio(r[key],b[key]) for key in ('lambda_rms','relative_error','minimum_error','fine_slope_path','z2_sum','coarse_fine2_sum','mean_shadow_gain')},
                    extra_lambda=r['lambda_rms']-b['lambda_rms']))
    write(output/'adam_balancing_contrasts.csv',contrasts)
    gd_contrasts=[]
    for r in gd:
        if not r['arm'].endswith('_tracking01'):continue
        bb=[b for b in gd if b['case_key']==r['case_key'] and b['arm']==r['arm'].removesuffix('_tracking01')]
        if not bb:continue
        b=bb[0]
        gd_contrasts.append(dict(case_key=r['case_key'],target=r['target'],origin=r['origin'],arm=r['arm'],
            left_status=r['status'],right_status=b['status'],
            lambda_ratio=ratio(r['lambda_rms'],b['lambda_rms']),error_ratio=ratio(r['relative_error'],b['relative_error']),
            left_stability=r['final_stability_ratio'],right_stability=b['final_stability_ratio']))
    write(output/'gd_tracking_contrasts.csv',gd_contrasts)
    followup=[]
    for end in (10000,20000):
        subset=[r for r in states if r['stage']=='followup' and r['offset']==end and r['alive']]
        native={r['case_key']:r for r in subset if r['arm']=='native'}
        for r in subset:
            n=native[r['case_key']]
            followup.append(dict(case_key=r['case_key'],target=r['target'],width=r['width'],seed=r['seed'],arm=r['arm'],offset=end,
                **{key:ratio(r[key],n[key]) for key in ('lambda_rms','relative_error','minimum_error','z2_sum','raw_tracking2_sum','shadow_gain_sum')},
                actual_lambda=r['lambda_rms'],actual_error=r['relative_error'],
                fine_A=r['A_fine'],tracking_A=r['A_tracking'],step_A=r['A_step'],
                tracking_lag=ratio(r['z_lag_dot'],r['z2_sum']),mean_shadow_gain=r['shadow_gain_sum']/end))
    write(output/'followup_pairs.csv',followup)
    # Late-window rates distinguish sustained tracking from a large inherited transient.
    indexed={(r['case_key'],r['arm'],r['offset']):r for r in states if r['stage']=='followup' and r['alive']}
    windows=[]
    for case in sorted(set(r['case_key'] for r in followup)):
        for arm in ('native','fine_off','frozen','frozen_balanced'):
            for left,right in ((0,2000),(8000,10000),(18000,20000)):
                keys=[(case,a,t) for a in (arm,'native') for t in (left,right)]
                if not all(k in indexed for k in keys):continue
                s,e,ns,ne=[indexed[k] for k in keys]
                z=e['z2_sum']-s['z2_sum']; nz=ne['z2_sum']-ns['z2_sum']
                windows.append(dict(case_key=case,target=e['target'],width=e['width'],seed=e['seed'],arm=arm,
                    left=left,right=right,z2_ratio=ratio(z,nz),z2_rate=z/(right-left),
                    endpoint_z_norm=e['z_norm'],
                    tracking_lag=ratio(e['z_lag_dot']-s['z_lag_dot'],z),
                    lambda_ratio=ratio(e['lambda_rms'],indexed[(case,arm,0)]['lambda_rms']),
                    raw_tracking_ratio=ratio(e['raw_tracking2_sum']-s['raw_tracking2_sum'],ne['raw_tracking2_sum']-ns['raw_tracking2_sum'])))
    write(output/'followup_windows.csv',windows)
    outcomes=[dict(stage=stage,arm=arm,target=target,n=len(rr),
        complete=sum(r['complete'] for r in rr),
        failure_updates=distribution([r['end_offset'] for r in rr if not r['complete']]))
        for stage in ('gd','adam','followup') for arm in sorted(set(r['arm'] for r in runs if r['stage']==stage))
        for target in ('all',*TARGETS)
        if (rr:=[r for r in runs if r['stage']==stage and r['arm']==arm and (target=='all' or r['target']==target)])]
    late=[dict(arm=arm,target=target,left=left,right=right,n=len(rr),
        unresolved_z_increment=sum(r['z2_rate']==0 for r in rr),
        endpoint_z_below_1e14=sum(r['endpoint_z_norm']<1e-14 for r in rr),
        **{k:distribution([r[k] for r in rr]) for k in ('z2_ratio','z2_rate','endpoint_z_norm','tracking_lag','lambda_ratio','raw_tracking_ratio')})
        for arm in ('native','fine_off','frozen','frozen_balanced') for left,right in ((0,2000),(8000,10000),(18000,20000))
        for target in ('all',*TARGETS)
        if (rr:=[r for r in windows if r['arm']==arm and r['left']==left and (target=='all' or r['target']==target)])]
    rescues=[]
    for r in runs:
        if r['stage']=='followup' and r['arm']=='frozen' and not r['complete']:
            other=next(v for v in runs if v['stage']=='followup' and v['case_key']==r['case_key'] and v['arm']=='frozen_balanced')
            rescues.append(dict(case_key=r['case_key'],target=r['target'],frozen_stop=r['end_offset'],balanced_stop=other['end_offset'],balanced_complete=other['complete']))
    (output/'completion_details.json').write_text(json.dumps(dict(outcomes=outcomes,windows=late,frozen_rescues=rescues),indent=2)+'\n')
    write(output/'run_outcomes.csv',runs)
    coverage={stage:dict(states=sum(r['stage']==stage for r in states),runs=sum(r['stage']==stage for r in runs),
        complete=sum(r['stage']==stage and r['complete'] for r in runs),
        statuses={status:sum(r['stage']==stage and r['status']==status for r in runs) for status in ('complete','nonfinite','unresolved','budget_stopped')}) for stage in ('audit','gd','adam','followup')}
    summaries=[]
    for offset in (10000,20000):
        for arm in ARMS:
            for target in ('all',*TARGETS):
                rr=[r for r in paired if r['offset']==offset and r['arm']==arm and (target=='all' or r['target']==target)]
                if rr:summaries.append(dict(offset=offset,arm=arm,target=target,n=len(rr),**{k:distribution([r[k] for r in rr]) for k in ('lambda_rms_ratio','relative_error_ratio','fine_slope_path_ratio','z2_sum_ratio','coarse_fine2_sum_ratio','mean_shadow_gain','tracking_to_fine_A','step_to_fine_A','coherence','tracking_lag')}))
    facts=dict(coverage=coverage,receipts=receipts,
        gpu_seconds_with_receipt_reserve=sum(r['seconds']+15 for r in receipts if r['platform']=='Modal GPU'),
        checks={k:distribution([r[k] for r in states if k in r and r['alive']]) for k in ('component_error','norm_error','coarse_identity_error','balanced_response_error','A_closure')},
        scaled_A_closure=distribution([abs(r['A_closure'])/(1+abs(r['A'])+sum(abs(r['A_'+k]) for k in ('fine','tracking','unresolved','step'))) for r in states if r['alive']]),
        audit_identity=distribution([r['identity_error'] for r in audit]),
        sources=[dict(origin=origin,age=age,proposal=proposal,
            **{key:distribution([r[key] for r in source if r['origin']==origin and r['age']==age and r['proposal']==proposal and r['norm_matched']])
               for key in ('linear_coarse','equilibrium_change','nonlinear_coarse','z_change','linear_share','equilibrium_share','nonlinear_share')})
               for origin in ('gd','adam') for age in (25000,130000) for proposal in ('raw','scaled_current','momentum','processed','balanced')],
        curvature=[dict(origin=origin,age=age,metric=metric,
            stability=distribution([r.get(metric+'_stability_ratio',np.nan) for r in curves if r['stage']=='audit' and r['origin']==origin and r['age']==age]),
            **{key:distribution([r[metric+'_'+key] for r in curves if r['stage']=='audit' and r['origin']==origin and r['age']==age and metric+'_'+key in r])
               for key in ('coarse_overlap','metric_coarse_overlap','coarse_rayleigh_fraction')} )
            for origin in ('gd','adam') for age in (25000,130000) for metric in ('raw','adaptive')],adam=summaries)
    (output/'summary.json').write_text(json.dumps(facts,indent=2)+'\n')
    highlights=dict(coverage=coverage,gpu_seconds=facts['gpu_seconds_with_receipt_reserve'],
        scaled_A_closure=facts['scaled_A_closure'],
        followup=[dict(arm=arm,offset=end,target=target,n=len(rr),
            **{key:distribution([r[key] for r in rr]) for key in ('lambda_rms','relative_error','minimum_error','z2_sum','tracking_lag','mean_shadow_gain')})
            for arm in ('native','fine_off','frozen','frozen_balanced') for end in (10000,20000) for target in ('all',*TARGETS)
            if (rr:=[r for r in followup if r['arm']==arm and r['offset']==end and (target=='all' or r['target']==target)])],
        gd=[dict(origin=origin,arm=arm,n=len(rr),**{key:distribution([r[key] for r in rr if r['complete']]) for key in ('lambda_ratio','relative_error','error_ratio','best_error_ratio','final_stability_ratio','loss_increases','tracking_to_fine_A','step_to_fine_A')})
            for origin in ('gd','adam') for arm in sorted(set(r['arm'] for r in gd))
            if (rr:=[r for r in gd if r['origin']==origin and r['arm']==arm])],
        balancing=[dict(offset=end,left=left,right=right,target=target,n=len(rr),
            **{key:distribution([r[key] for r in rr]) for key in ('lambda_rms','relative_error','minimum_error','fine_slope_path','z2_sum','mean_shadow_gain')})
            for end in (10000,20000) for left,right in (('balanced','native'),('balanced10','fine10'),('frozen_balanced10','frozen10')) for target in ('all',*TARGETS)
            if (rr:=[r for r in contrasts if r['offset']==end and r['left']==left and (target=='all' or r['target']==target)])])
    (output/'highlights.json').write_text(json.dumps(highlights,indent=2)+'\n')
    print(json.dumps(dict(coverage=coverage,gpu_seconds=facts['gpu_seconds_with_receipt_reserve'],checks=facts['checks']),indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=plt.get_cmap('tab10').colors
    if curves:
        fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        for ti,target in enumerate(TARGETS):
            rr=[r for r in curves if r['stage']=='audit' and r['origin']=='adam' and r['age']==130000 and r['target']==target]
            vals=[(r.get('adaptive_stability_ratio',np.nan),r.get('adaptive_coarse_rayleigh_fraction',np.nan)) for r in rr]
            if vals:ax[0].scatter(*np.asarray(vals).T,label=LABELS[ti],color=colors[ti])
            rr=[r for r in paired if r['offset']==10000 and r['arm']=='native' and r['target']==target]
            if rr:ax[1].scatter([r['tracking_lag'] for r in rr],[r['tracking_to_fine_A'] for r in rr],color=colors[ti])
        ax[0].set(xlabel='Adaptive sharpness / frozen stability threshold',ylabel='Coarse contribution to top-mode curvature',title='A. Which curvature sets the Adam boundary?')
        ax[0].axvline(1,color='.7',ls=':');ax[0].axhline(1,color='.7',ls=':');ax[0].legend(fontsize=8,loc='upper left')
        ax[1].set(xlabel='One-step correlation of tracking error',ylabel=r'Linear tracking / fine contribution to $\Delta A$',title='B. Oscillation and direct population effect')
        ax[1].axhline(0,color='.7',ls=':');ax[1].set_yscale('symlog',linthresh=.1)
        save(fig,output,'stability_and_tracking')
    if source:
        fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        for oi,origin in enumerate(('gd','adam')):
            for pi,proposal in enumerate(('processed','balanced')):
                rr=[r for r in source if r['origin']==origin and r['age']==130000 and r['proposal']==proposal and r['norm_matched']]
                x=oi*3+pi
                for ki,key in enumerate(('linear_magnitude_share','equilibrium_magnitude_share','nonlinear_magnitude_share')):
                    values=[r[key] for r in rr if np.isfinite(r[key])]
                    ax[0].scatter(np.full(len(values),x+(ki-1)*.18),values,s=12,alpha=.5,color=colors[ki])
                    if values:ax[0].plot([x+(ki-1)*.18-.07,x+(ki-1)*.18+.07],[np.median(values)]*2,color=colors[ki],lw=3)
        for i,label in enumerate(('Coarse-output change','Moving equilibrium','Nonlinear remainder')):ax[0].scatter([],[],color=colors[i],label=label)
        ax[0].set(xticks=[0,1,3,4],xticklabels=['GD\nfine','GD\nbalanced','Adam\nfine','Adam\nbalanced'],ylabel='Magnitude / sum of the three magnitudes',title='A. Sources of an isolated fine-step response',ylim=(-.04,1.2))
        ax[0].axhline(1,color='.7',ls=':');ax[0].legend(fontsize=8,loc='upper center',ncol=1)
        for ti,target in enumerate(TARGETS):
            rr=[r for r in paired if r['target']==target and r['arm']=='balanced10' and r['offset']==10000]
            controls={r['case_key']:r for r in paired if r['arm']=='fine10' and r['offset']==10000}
            pts=[(ratio(r['z2_sum'],controls[r['case_key']]['z2_sum']),ratio(r['mean_shadow_gain'],controls[r['case_key']]['mean_shadow_gain'])) for r in rr if r['case_key'] in controls]
            if pts:ax[1].scatter(*np.asarray(pts).T,color=colors[ti],label=LABELS[ti])
        ax[1].set(xscale='log',yscale='log',xlabel='Tracking energy: balanced / unbalanced',ylabel='Available denominator gain: balanced / unbalanced',title='B. Does balancing relieve the restriction?')
        ax[1].axhline(1,color='.7',ls=':');ax[1].axvline(1,color='.7',ls=':')
        if paired:ax[1].legend(fontsize=8)
        save(fig,output,'tracking_feedback')
    if gd:
        fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        rates=('native_rate','rate_0.25','rate_0.5','rate_0.9','rate_1.05','rate_1.25')
        for ti,target in enumerate(TARGETS):
            for ai,key in enumerate(('lambda_ratio','best_error_ratio')):
                vals=[]
                for arm in rates:
                    rr=[r for r in gd if r['origin']=='gd' and r['target']==target and r['arm']==arm and r['complete']]
                    vals.append(np.median([r[key] for r in rr]) if rr else np.nan)
                ax[ai].plot(range(6),vals,'o-',ms=4,color=colors[ti],label=LABELS[ti])
        for a in ax:a.set(xticks=range(6),xticklabels=['Native','.25','.5','.9','1.05','1.25'],xlabel=r'GD rate / initial $2/\lambda_{\max}(H)$');a.axhline(1,color='.7',ls=':');a.set_yscale('log')
        ax[0].set(ylabel='RMS slope / fork RMS slope',title='A. Population response after 10k GD updates')
        ax[0].yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1,2,5)))
        ax[0].yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:g}'))
        ax[1].set(ylabel='Best raw relative error / fork error',title='B. Best output error within 10k updates');ax[1].legend(fontsize=8)
        save(fig,output,'gd_rate_response')
    if paired:
        fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        names=['Native','Fine ×10','Balanced','Balanced ×10'];shown=ARMS[:4]
        for ti,target in enumerate(TARGETS):
            for ai,key in enumerate(('lambda_rms_ratio','minimum_error_ratio')):
                for end,style in ((10000,'-'),(20000,'--')):
                    vals=[np.median([r[key] for r in paired if r['target']==target and r['arm']==arm and r['offset']==end]) if any(r['target']==target and r['arm']==arm and r['offset']==end for r in paired) else np.nan for arm in shown]
                    if ai==0:vals=100*(np.asarray(vals)-1)
                    ax[ai].plot(range(4),vals,style,marker='o' if end==10000 else '.',ms=4,color=colors[ti],alpha=1 if end==10000 else .55,label=LABELS[ti] if end==10000 else None)
        for i,a in enumerate(ax):a.set(xticks=range(4),xticklabels=names);a.axhline(0 if i==0 else 1,color='.7',ls=':')
        ax[1].set_yscale('log')
        ax[0].set(ylabel='RMS slope change versus native (%)',title='A. Population acquisition: pulse / release')
        ax[1].set(ylabel='Best raw error / native best raw error',title='B. Does extra motion improve output?');ax[1].legend(fontsize=8)
        save(fig,output,'adam_causal_response')
    if followup:
        fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        names=('fine_off','frozen','frozen_balanced')
        for ti,target in enumerate(TARGETS):
            for i,arm in enumerate(names):
                rr=[r for r in windows if r['target']==target and r['arm']==arm and r['left']==8000]
                if rr:ax[0].scatter(np.full(len(rr),i)+(ti-2.5)*.04,[r['z2_ratio'] for r in rr],color=colors[ti],s=20,label=LABELS[ti] if i==0 else None)
                rr=[r for r in runs if r['stage']=='followup' and r['target']==target and r['arm']==arm]
                if rr:ax[1].scatter(np.full(len(rr),i)+(ti-2.5)*.04,[min(r['end_offset'],10000) for r in rr],color=colors[ti],s=20)
        for a in ax:a.set(xticks=range(3),xticklabels=['No fine motion','Frozen fine\ndenominator','Frozen +\nbalanced'])
        ax[0].set_yscale('symlog',linthresh=.05);ax[1].set_yscale('log')
        ax[0].axhline(1,color='.7',ls=':');ax[0].set(ylabel='Tracking energy in final 2k / native',title='A. Does tracking persist without fine motion?');ax[0].set_ylim(bottom=-.003);ax[0].legend(fontsize=8,loc='lower right')
        ax[1].axhline(10000,color='.7',ls=':');ax[1].set(ylabel='Updates reached in the 10k pulse',title='B. Are unit-gain frozen controls stable?')
        save(fig,output,'secondary_controls')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage');p.add_argument('--cohort');p.add_argument('--seconds');analyze(p.parse_args())
