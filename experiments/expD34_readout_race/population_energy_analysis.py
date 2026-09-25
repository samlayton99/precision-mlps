"""Curate aggregate-rate, concentration-intervention, and comparison evidence.

Numerical analysis runs on Modal CPU. This module emits data and figures only.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

from . import population_concentration as pc
from . import population_concentration_analysis as previous
from . import population_energy_dynamics as ed
from .population_window_run import SIX


def cumulative(t,v):return cumulative_trapezoid(v,t,initial=0.)


def cap_clock(cap,clocks,target,e0,width):
    """Three separately valid integrated allowances, followed by their minimum."""
    aa,bb,cc=np.asarray(clocks).T
    return np.minimum(pc.C3*e0/width*aa,
        np.minimum(target['D3']*e0/width*aa+pc.C5*e0*cap**2/width**2*bb,
                   target['D3']*target['tau3']/width*aa+pc.C5*e0*cap**2/width**2*bb
                   +target['D3']*pc.A3*cap**4/width**2*cc))


def load(inputs):
    groups=defaultdict(list);windows=defaultdict(list);case_records=[];attempts=[]
    for path in inputs:
        with path.open() as stream:
            for row in csv.DictReader(stream):
                for k,v in row.items():
                    try:row[k]=float(v)
                    except ValueError:pass
                if 'arm' in row:
                    groups[(row['study'],row['case_id'])].append(row)
                elif row['width']>=705:
                    windows[(row['target'],row['width'],row['seed'],row['kind'],row['dt'])].append(row)
        if path.with_suffix('.json').exists():
            facts=json.loads(path.with_suffix('.json').read_text())
            case_records.extend(facts.get('cases',[]));attempts.extend(facts.get('attempts',[]))
    return list(groups.values()),windows,case_records,attempts


def closure(rows,target):
    """Continuous comparison supplied with measured K and tracking histories.

    Native-GD evaluations are a sampled GF comparison, not discrete certificates.
    The exact discrete identity is audited separately.
    """
    rows=[r for r in rows if r['offset']<=100000]
    t=np.array([r['time'] for r in rows]);start=rows[0]
    b0=math.sqrt(start['M']);z0=math.sqrt(start['C6']);w=start['width']
    e0=start['fine_error'] if start['kind']=='effective' else start['relative_error']*start['target_norm']
    ks=np.array([max(1.,r['K6']) for r in rows])
    tracking=np.array([r['tracking_norm'] if start['kind']=='gd' else 0. for r in rows])
    tracking_shape=np.array([max(0.,r['rate_logC6_tracking']) for r in rows])
    def allowance(b,z,k):
        root6=z*b**3/w;root10=math.sqrt(k)*z*z*b**5/w**2
        cap=min(pc.A3*z*b**4/w,target['B3']*z*b**4/w+pc.A5*z*z*b**6/w**2)
        speed=min(pc.C3*e0*root6,e0*(target['D3']*root6+pc.C5*root10),
                  target['D3']*root6*(target['tau3']+cap)+pc.C5*e0*root10)
        return speed,cap
    def rhs(s,u):
        b=b0*np.exp(min(u[0],20.));z=z0*np.exp(min(u[1],20.))
        k=np.interp(s,t,ks);speed,_=allowance(b,z,k)
        return [(speed+np.interp(s,t,tracking))/b,
                3*math.sqrt(k-1)*speed/b+.5*np.interp(s,t,tracking_shape)]
    def stop(s,u):
        b=b0*np.exp(min(u[0],20.));z=z0*np.exp(min(u[1],20.))
        _,cap=allowance(b,z,np.interp(s,t,ks))
        return min(target['fine_norm']-cap-.01*target['target_norm'],.25-start['h']*b/math.sqrt(w))
    stop.terminal=True;stop.direction=-1
    solution=solve_ivp(rhs,(0,float(t[-1])),[0.,0.],t_eval=t,events=stop,rtol=2e-9,atol=1e-11,max_step=1.)
    if solution.status<0:raise RuntimeError(solution.message)
    horizon=float(solution.t_events[0][0]/.002) if len(solution.t_events[0]) else float(t[-1]/.002)
    final_b=b0*np.exp(solution.y[0,-1]);final_z=z0*np.exp(solution.y[1,-1])
    _,final_capacity=allowance(final_b,final_z,np.interp(solution.t[-1],t,ks))
    return dict(horizon=horizon,bound_offset=float(solution.t[-1]/.002),final_lambda_upper=float(start['h']*final_b/math.sqrt(w)),
                final_output_floor=float(max(0.,target['fine_norm']-final_capacity)/target['target_norm']),
                final_sqrtC6_upper=float(z0*np.exp(solution.y[1,-1])),
                sampled_mass_ratio_max=float(max(rows[i]['M']/(b0*np.exp(solution.y[0,i]))**2 for i in range(len(solution.t)))),
                sampled_shape_ratio_max=float(max(math.sqrt(rows[i]['C6'])/(z0*np.exp(solution.y[1,i])) for i in range(len(solution.t)))))


def scalar_closure(rows,target):
    """Joint radius/shape gradient bound: no separate saturation of directions."""
    rows=[r for r in rows if r['offset']<=100000]
    t=np.array([r['time'] for r in rows]);start=rows[0];w=start['width']
    y0=start['M']*math.sqrt(start['C6'])
    e0=start['fine_error'] if start['kind']=='effective' else start['relative_error']*start['target_norm']
    ks=np.array([max(1.,r['K6']) for r in rows])
    tracking=np.array([max(0.,r['rate_M_tracking']/r['M']+.5*r['rate_logC6_tracking']) for r in rows])
    def capacity(y):return min(pc.A3*y*y/w,target['B3']*y*y/w+pc.A5*y**3/w**2)
    def rhs(s,u):
        y=y0*np.exp(min(u[0],20.));k=np.interp(s,t,ks)
        drive=min(pc.C3*e0*y/w,e0*(target['D3']*y/w+pc.C5*math.sqrt(k)*y*y/w**2),
                  target['D3']*y/w*(target['tau3']+capacity(y))+pc.C5*e0*math.sqrt(k)*y*y/w**2)
        return [math.sqrt(9*k-5)*drive+np.interp(s,t,tracking)]
    def stop(s,u):
        y=y0*np.exp(min(u[0],20.))
        return min(target['fine_norm']-capacity(y)-.01*target['target_norm'],.25-start['h']*math.sqrt(y/w))
    stop.terminal=True;stop.direction=-1
    sol=solve_ivp(rhs,(0,float(t[-1])),[0.],t_eval=t,events=stop,rtol=2e-9,atol=1e-11,max_step=1.)
    if sol.status<0:raise RuntimeError(sol.message)
    horizon=float(sol.t_events[0][0]/.002) if len(sol.t_events[0]) else float(t[-1]/.002)
    end=y0*np.exp(sol.y[0,-1])
    return dict(horizon=horizon,bound_offset=float(sol.t[-1]/.002),final_lambda_upper=float(start['h']*math.sqrt(end/w)),
        final_output_floor=float(max(0.,target['fine_norm']-capacity(end))/target['target_norm']),
        sampled_product_ratio_max=float(max(rows[i]['M']*math.sqrt(rows[i]['C6'])/(y0*np.exp(sol.y[0,i])) for i in range(len(sol.t)))))


def accumulated_caps(rows,window,target):
    """First-exit/Bihari bounds using three population clocks, not mean ODE coefficients."""
    rows=[r for r in rows if r['offset']<=100000]
    window=sorted([r for r in window if r['offset']<=5000],key=lambda r:r['time'])
    if not window:return []
    t=np.array([r['time'] for r in rows]);tw=np.array([r['time'] for r in window])
    coefficients=lambda rs:np.array([[math.sqrt(r['C6']),math.sqrt(r['C10']),r['C4']*math.sqrt(r['C6'])] for r in rs])
    actual=coefficients(rows);baseline=coefficients(window)
    clocks=np.stack([cumulative(t,actual[:,i]) for i in range(3)],axis=1)
    means=np.array([cumulative(tw,baseline[:,i])[-1]/(tw[-1]-tw[0]) for i in range(3)])
    start=rows[0];w=start['width'];b0=math.sqrt(start['M'])
    e0=start['fine_error'] if start['kind']=='effective' else start['relative_error']*start['target_norm']
    reserve=np.array([r['integral_tracking_path'] for r in rows])
    results=[]
    for multiplier in (1,2,4):
        supplied=multiplier*t[:,None]*means
        premise=np.all(clocks<=supplied+1e-12,axis=1)
        prefix=np.logical_and.accumulate(premise)
        best=np.full_like(t,np.inf)
        for factor in (1.25,1.5,2,4,8,16):
            cap=factor*b0
            clock=cap_clock(cap,supplied,target,e0,w)
            seed=b0+reserve;denominator=1-2*seed*seed*clock
            upper=seed/np.sqrt(np.maximum(denominator,1e-300))
            valid=(denominator>0)&(upper<=cap)&prefix
            best=np.minimum(best,np.where(valid,upper,np.inf))
        valid=np.isfinite(best)&(start['h']*best/math.sqrt(w)<.25)
        results.append(dict(multiplier=multiplier,premise_horizon=previous.first_horizon(t,prefix),
            slope_horizon=previous.first_horizon(t,valid),
            largest_clock_ratio=float(np.max(np.divide(clocks,supplied/multiplier,out=np.ones_like(clocks),where=t[:,None]>0))),
            **{f'largest_clock_ratio_{name}':float(np.max(np.divide(clocks[:,i],t*means[i],out=np.ones_like(t),where=t>0))) for i,name in enumerate(('6','10','46'))},
            final_lambda_upper=float(start['h']*best[-1]/math.sqrt(w)) if np.isfinite(best[-1]) else None))
    return results


def analyze(inputs,output):
    output.mkdir(parents=True,exist_ok=False)
    groups,windows,case_records,attempts=load(inputs)
    target_cache={r[0]['target']:previous.target_data(r[0]['target'])[2] for r in groups}
    audit=[];coupled=[];scalar=[];sampling=[];caps=[];endpoints=[];curves={}
    for rows in groups:
        rows.sort(key=lambda r:r['offset']);start=rows[0]
        selected=[r for r in rows if r['offset']<=100000];end=selected[-1]
        ident={k:start[k] for k in ('study','case_id','target','width','seed','kind','dt','arm','factor','path')}
        point=dict(**ident,complete=end['offset']==100000)
        for k in ('M','C6','K6','q','jacobian_exact','lambda_rms','relative_eval_error','relative_error','fine_error'):
            point['initial_'+k]=start[k];point['final_'+k]=end[k]
        point['best_sampled_error']=min(r['relative_eval_error'] for r in selected)
        point['largest_sampled_lambda']=max(r['lambda_rms'] for r in selected)
        for c in ed.COMPONENTS:
            point['logC6_'+c]=end[f'integral_logC6_{c}']
            point['M_'+c]=end[f'integral_M_{c}']
            point['A_'+c]=end[f'integral_A_{c}']
        point['discrete_logC6']=end['integral_discrete_logC6']
        point['discrete_A']=end['integral_discrete_A']
        endpoints.append(point)
        if start['study']=='wide' and start['arm']=='baseline' and start['kind']=='effective':
            coupled.append(dict(**ident,**closure(selected,target_cache[start['target']])))
            scalar.append(dict(**ident,**scalar_closure(selected,target_cache[start['target']])))
        if start['study']!='audit':continue
        curves[(start['target'],start['width'],start['seed'],start['kind'],start['dt'])]=selected
        sum_absolute=sum(abs(point['logC6_'+c]) for c in ed.COMPONENTS)
        item=dict(**ident,logC6_change=math.log(end['C6']/start['C6']),
            net_fraction_of_signed_integrals=abs(math.log(end['C6']/start['C6']))/sum_absolute if sum_absolute else None,
            K6_initial=start['K6'],K6_final=end['K6'],K6_largest_sample=max(r['K6'] for r in selected),
            concentration_acceleration_initial=start['logC6_acceleration'],
            concentration_acceleration_final=end['logC6_acceleration'],
            rate_initial=start['logC6_rate'],rate_final=end['logC6_rate'],
            concentration_ratio=end['C6']/start['C6'],
            score_identity_error=max(abs(r['score_identity_error']) for r in rows),
            decomposition_error=max(r['velocity_identity_error'] for r in rows),
            **{f'logC6_{c}':point[f'logC6_{c}'] for c in ed.COMPONENTS})
        for name in ('jacobian_generic','jacobian_projected','jacobian_cubic_concentration'):
            denominator='jacobian_cubic' if name.endswith('concentration') else 'jacobian_exact'
            item[name+'_median_slack']=float(np.median([r[name]/r[denominator] for r in selected]))
        for name in ('fine_alignment','projection_fraction','slope_fraction','mass_radial_fraction'):
            item[name+'_median']=float(np.median([r[name] for r in selected]))
        t=np.array([r['time'] for r in selected])
        e0=start['fine_error'] if start['kind']=='effective' else start['relative_error']*start['target_norm']
        product0=start['M']*math.sqrt(start['C6'])
        dispersion_clock=cumulative(t,[math.sqrt(max(0.,9*r['K6']-5)) for r in selected])
        denominator=1-pc.C3*e0/start['width']*product0*dispersion_clock
        item['effective_product_comparison_horizon']=previous.first_horizon(t,denominator>0) if start['kind']=='effective' else None
        allowances=[6/math.sqrt(r['M'])*math.sqrt(max(0.,r['K6']-1))*r['q'] for r in selected]
        item['median_relative_advantage_fraction']=float(np.median([abs(r['logC6_rate']-r['rate_logC6_tracking'])/b for r,b in zip(selected,allowances) if b>0]))
        audit.append(item)
        if start['source_study']=='development' or (start['width']==1409):
            comparison=closure(selected,target_cache[start['target']])
            coupled.append(dict(**ident,**comparison))
            joint=scalar_closure(selected,target_cache[start['target']])
            scalar.append(dict(**ident,**joint))
            coarse=selected[::2]
            if coarse[-1] is not selected[-1]:coarse.append(selected[-1])
            check=closure(coarse,target_cache[start['target']])
            joint_check=scalar_closure(coarse,target_cache[start['target']])
            sampling.append(dict(**ident,horizon_change=abs(check['horizon']-comparison['horizon']),
                endpoint_lambda_relative_change=abs(check['final_lambda_upper']/comparison['final_lambda_upper']-1) if comparison['horizon']==100000 and check['horizon']==100000 else None,
                scalar_horizon_change=abs(joint_check['horizon']-joint['horizon']),
                scalar_endpoint_lambda_relative_change=abs(joint_check['final_lambda_upper']/joint['final_lambda_upper']-1) if joint['horizon']==100000 and joint_check['horizon']==100000 else None))
        key=(start['target'],start['width'],start['seed'],start['kind'],start['dt'])
        if key in windows:
            caps.extend(dict(**ident,**r) for r in accumulated_caps(selected,windows[key],target_cache[start['target']]))
    interventions=[];refinement=[]
    for point in endpoints:
        if point['study']=='audit':continue
        baseline=next((s for s in endpoints if s['study']==point['study'] and s['target']==point['target'] and s['kind']==point['kind'] and s['arm']=='baseline'),None)
        if baseline:
            item=dict(point)
            item.update(initial_sensitivity_gain=point['initial_jacobian_exact']/baseline['initial_jacobian_exact'],
                initial_force_gain=point['initial_q']/baseline['initial_q'],
                final_slope_ratio=point['final_lambda_rms']/baseline['final_lambda_rms'],
                final_mass_ratio=point['final_M']/baseline['final_M'],
                final_error_change=point['final_relative_eval_error']-baseline['final_relative_eval_error'],
                achieved_concentration_ratio=math.sqrt(point['initial_C6']/baseline['initial_C6']))
            record=next(r for r in case_records if r['study']==point['study'] and r['case_id']==point['case_id'])
            item.update(fixed_residual_force_gain=record['fixed_residual_force']/baseline['initial_q'],
                        initial_fine_output_shift=record['fine_output_shift'],gram_relative_error=record['gram_relative_error'])
            interventions.append(item)
        if point['study']=='refinement':
            match=next((s for s in endpoints if s['study']=='development' and all(s[k]==point[k] for k in ('target','kind','arm','path'))),None)
            if match:
                refinement.append(dict(target=point['target'],kind=point['kind'],arm=point['arm'],
                    slope_relative_change=abs(point['final_lambda_rms']-match['final_lambda_rms'])/point['final_lambda_rms'],
                    error_absolute_change=abs(point['final_relative_eval_error']-match['final_relative_eval_error']),
                    concentration_relative_change=abs(point['final_C6']-match['final_C6'])/point['final_C6']))
    tables=dict(audit=audit,coupled=coupled,scalar_coupled=scalar,sampling=sampling,accumulated_caps=caps,interventions=interventions,refinement=refinement)
    for name,rows in tables.items():
        if rows:
            with (output/f'{name}.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    native=[r for r in audit if r['kind']=='gd' and r['dt']==.002]
    summary=dict(trajectories=len(groups),audit_native=len(native),intervention_trajectories=len(interventions),
        construction_attempts=len(attempts),construction_failures=sum(not r['valid'] for r in attempts),
        incomplete_cases=[{k:r[k] for k in ('study','case_id','target','status','completed_offset')} for r in case_records if r['status']!='complete'],
        maximum_gram_error=max((r['gram_relative_error'] for r in interventions),default=None),
        maximum_replay_error=max((r['replay_relative_error'] for r in case_records if r['study']=='audit'),default=None),
        maximum_log_moment_closure=max((r['log_moment_closure'] for r in case_records),default=None),
        maximum_score_identity_error=max((r['score_identity_error'] for r in audit),default=None),
        native_concentration_grows=sum(r['logC6_change']>0 for r in native),
        native_generated_opposes=sum(r['logC6_generated']<0 for r in native),
        native_compensation_opposes=sum(r['logC6_compensation']<0 for r in native),
        native_target_drives=sum(r['logC6_target']>0 for r in native),
        native_median_signed_fraction=float(np.median([r['net_fraction_of_signed_integrals'] for r in native])) if native else None)
    def extent(values):
        return dict(min=float(min(values)),median=float(np.median(values)),max=float(max(values))) if values else None
    summary['native_rates']=dict(initial=extent([r['rate_initial'] for r in native]),final=extent([r['rate_final'] for r in native]),
        concentration_ratio=extent([r['concentration_ratio'] for r in native]),dispersion_final=extent([r['K6_final'] for r in native]))
    summary['native_slack']={key:extent([r[key] for r in native]) for key in native[0] if key.endswith(('_median','_median_slack','_fraction'))} if native else {}
    summary['coupled_coverage']=[dict(width=w,kind=k,count=len(rs),horizon=extent([r['horizon'] for r in rs]),
        full=sum(r['horizon']>=99999 for r in rs),mass_violation=max(r['sampled_mass_ratio_max'] for r in rs),
        shape_violation=max(r['sampled_shape_ratio_max'] for r in rs))
        for w,k in ((705,'gd'),(705,'effective'),(1409,'gd'),(1409,'effective')) if (rs:=[r for r in coupled if r['width']==w and r['kind']==k])]
    summary['cap_coverage']=[dict(width=w,multiplier=m,count=len(rs),premise=extent([r['premise_horizon'] for r in rs]),
        horizon=extent([r['slope_horizon'] for r in rs]),full=sum(r['slope_horizon']>=99999 for r in rs),
        clock_ratios={key:extent([r['largest_clock_ratio_'+key] for r in rs]) for key in ('6','10','46')})
        for w in (705,1409) for m in (1,2,4) if (rs:=[r for r in caps if r['width']==w and r['kind']=='gd' and r['dt']==.002 and r['multiplier']==m])]
    summary['scalar_coverage']=[dict(width=w,kind=k,count=len(rs),horizon=extent([r['horizon'] for r in rs]),
        full=sum(r['horizon']>=99999 for r in rs),product_violation=max(r['sampled_product_ratio_max'] for r in rs))
        for w,k in ((705,'gd'),(705,'effective'),(1409,'gd'),(1409,'effective')) if (rs:=[r for r in scalar if r['width']==w and r['kind']==k])]
    summary['sampling_check']=dict(horizon_change=extent([r['horizon_change'] for r in sampling]),
        endpoint_lambda_relative_change=extent([r['endpoint_lambda_relative_change'] for r in sampling if r['endpoint_lambda_relative_change'] is not None]),
        scalar_horizon_change=extent([r['scalar_horizon_change'] for r in sampling]),
        scalar_endpoint_lambda_relative_change=extent([r['scalar_endpoint_lambda_relative_change'] for r in sampling if r['scalar_endpoint_lambda_relative_change'] is not None]))
    summary['doses']=[dict(study=cohort,target=target,factor=factor,kind=kind,count=len(rs),
        sensitivity=extent([r['initial_sensitivity_gain'] for r in rs]),force=extent([r['initial_force_gain'] for r in rs]),
        fixed_residual_force=extent([r['fixed_residual_force_gain'] for r in rs]),
        slope=extent([r['final_slope_ratio'] for r in rs]),error=extent([r['final_relative_eval_error'] for r in rs]))
        for cohort in ('development','validation','wide') for target in SIX for factor in (0,2,4,10) for kind in ('gd','effective')
        if (rs:=[r for r in interventions if r['study']==cohort and r['target']==target and r['factor']==factor and r['kind']==kind and r['complete']])]
    pairs=[]
    for row in interventions:
        if row['kind']!='gd':continue
        other=next((r for r in interventions if r['kind']=='effective' and all(r[key]==row[key] for key in ('study','target','arm','path'))),None)
        if other:pairs.append(dict(study=row['study'],target=row['target'],arm=row['arm'],path=row['path'],
            slope_relative_difference=abs(row['final_lambda_rms']/other['final_lambda_rms']-1),
            error_difference=abs(row['final_relative_eval_error']-other['final_relative_eval_error'])))
    summary['paired_dynamics']=dict(slope_relative_difference=extent([r['slope_relative_difference'] for r in pairs]),
        error_difference=extent([r['error_difference'] for r in pairs]))
    summary['paired_dynamics_cases']=pairs
    summary['refinement']={key:extent([r[key] for r in refinement]) for key in ('slope_relative_change','error_absolute_change','concentration_relative_change')}
    summary['best_sampled_intervention_error']=min((r['best_sampled_error'] for r in interventions),default=None)
    summary['largest_sampled_intervention_lambda']=max((r['largest_sampled_lambda'] for r in interventions),default=None)
    summary['maximum_sampled_residual_ratio']=max(max(r['fine_error' if rows[0]['kind']=='effective' else 'relative_error']/rows[0]['fine_error' if rows[0]['kind']=='effective' else 'relative_error'] for r in rows) for rows in groups)
    executions=[json.loads(p.with_suffix('.execution.json').read_text()) for p in inputs if p.with_suffix('.execution.json').exists()]
    executions=[r for r in executions if r.get('study','').startswith('energy_') and r.get('platform')=='Modal GPU']
    summary['campaign_gpu_seconds']=sum(r['seconds'] for r in executions)
    summary['campaign_gpu_jobs']=[dict(study=r['study'],seconds=r['seconds'],peak_rss_mib=r['child_peak_rss_mib']) for r in executions]
    (output/'facts.json').write_text(json.dumps(dict(summary=summary,**tables),indent=2,allow_nan=False)+'\n')
    plots(curves,interventions,coupled,scalar,groups,output)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('doses','paired_dynamics_cases')}),flush=True)


def plots(curves,interventions,coupled,scalar,groups,output):
    import matplotlib.pyplot as plt
    names=dict(moment5='Degree five',mixed_sine='Mixed sine',gauss_left='Left Gaussian',bump_right='Right bump',step_right='Right step',kink_abs='Absolute-value kink')
    example=[rs for rs in groups if rs[0]['study']=='development' and rs[0]['target']=='gauss_left' and rs[0]['kind']=='gd' and rs[0]['arm'] in ('baseline','concentrated2')]
    if example:
        fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
        for rows in example:
            start=rows[0];baseline=start['arm']=='baseline'
            label='Baseline' if baseline else f'2× dose, mixing path {int(start["path"])+1}'
            color='k' if baseline else f'C{int(start["path"])}'
            t=np.array([r['offset'] for r in rows])/1000
            for ax,key in zip(axes.flat,('C6','K6','lambda_rms','relative_eval_error'),strict=True):
                ax.plot(t,[r[key] for r in rows],color=color,label=label)
        for ax,label in zip(axes.flat,('Concentration χ₆','Dispersion χ₁₀ / χ₆²','Normalized slope RMS','Raw relative output error'),strict=True):
            ax.set(xlabel='Further updates (thousands)',ylabel=label)
        for ax in axes[0]:ax.set_yscale('log')
        axes[0,0].legend(fontsize=8);axes[1,1].axhline(.01,color='k',ls=':',lw=.8)
        fig.savefig(output/'causal_evolution.png',dpi=170);plt.close(fig)
    if coupled:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained',sharey=True)
        for ax,w in zip(axes,(705,1409),strict=True):
            for i,(source,label) in enumerate(((coupled,'Separate radius and concentration'),(scalar,'Joint product: M√χ₆'))):
                points=[r for r in source if r['width']==w and r['kind']=='gd']
                ax.barh(np.arange(len(SIX))+(i-.5)*.32,[next((r['horizon']/1000 for r in points if r['target']==target),0) for target in SIX],height=.3,label=label)
            ax.set(title=f'Width {w}',yticks=np.arange(len(SIX)),yticklabels=[names[t] for t in SIX],xlabel='Useful horizon (thousands of further updates)',xlim=(0,105))
            ax.axvline(100,color='k',lw=.6)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,fontsize=9)
        fig.savefig(output/'coupled_coverage.png',dpi=170);plt.close(fig)
    dev=[(key,rows) for key,rows in curves.items() if key[1:]==(705,30,'gd',.002) and key[0] in SIX]
    if dev:
        fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
        for ax,(key,rows) in zip(axes.flat,dev):
            time=np.array([r['offset'] for r in rows])/1000
            for c in ed.COMPONENTS:ax.plot(time,[r[f'integral_logC6_{c}'] for r in rows],label=c)
            ax.plot(time,[math.log(r['C6']/rows[0]['C6']) for r in rows],'k--',label='Observed total')
            ax.set(title=names[key[0]],xlabel='Further updates (thousands)',ylabel='Contribution to log concentration')
        axes.flat[0].legend(fontsize=7)
        fig.savefig(output/'relative_growth.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        for key,rows in dev:
            t=np.array([r['offset'] for r in rows])/1000
            axes[0].plot(t,[r['logC6_rate'] for r in rows],label=names[key[0]])
            axes[1].plot(t,[r['K6'] for r in rows])
        axes[0].set(ylabel='Relative growth advantage: d log χ₆ / dt')
        axes[1].set(ylabel='Energy dispersion ratio χ₁₀ / χ₆²')
        axes[0].legend(fontsize=7,ncol=2)
        for ax in axes:ax.set_xlabel('Further updates (thousands)')
        fig.savefig(output/'growth_evolution.png',dpi=170);plt.close(fig)
        fig,grid=plt.subplots(2,2,figsize=(11,7),layout='constrained');axes=grid.flat
        for key,rows in dev:
            r=rows[0]
            axes[0].plot(range(4),[r['jacobian_generic']/r['jacobian_exact'],r['jacobian_projected']/r['jacobian_exact'],r['jacobian_cubic_concentration']/r['jacobian_cubic'],1.],'-o',label=names[key[0]])
            axes[1].plot(range(3),[r['fine_alignment'],r['projection_fraction'],r['slope_fraction']],'-o')
            t=np.array([r['offset'] for r in rows])/1000
            axes[2].plot(t,[r['mass_radial_fraction'] for r in rows])
            axes[3].plot(t,[(r['logC6_rate']-r['rate_logC6_tracking'])/(6/math.sqrt(r['M'])*math.sqrt(max(1e-30,r['K6']-1))*r['q']) for r in rows])
        axes[0].set(yscale='log',ylabel='Sensitivity bound / measured sensitivity',xticks=range(4),xticklabels=['Generic','Projection','Cubic only','Exact'])
        axes[1].set(yscale='log',ylabel='Stagewise factor (not cumulative)',xticks=range(3),xticklabels=['Residual alignment','After compensation','Slope block'])
        axes[2].set(ylabel='Fine velocity along radius (signed fraction)')
        axes[3].set(ylabel='Fine velocity along concentration (signed fraction)')
        for ax in (axes[2],axes[3]):
            ax.set_xlabel('Further updates (thousands)');ax.axhline(0,color='k',lw=.5);ax.set_ylim(-1,1)
        axes[0].legend(fontsize=7,ncol=2)
        fig.savefig(output/'slack_sources.png',dpi=170);plt.close(fig)
    if interventions:
        cohorts=('development','validation','wide')
        fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained',sharey='row')
        for column,cohort in enumerate(cohorts):
            for color,target in enumerate(SIX):
                for kind,ls in (('gd','-'),('effective','--')):
                    for path in (0,1):
                        points=[r for r in interventions if r['study']==cohort and r['target']==target and r['kind']==kind and r['path'] in (-1,path)]
                        points.sort(key=lambda r:r['achieved_concentration_ratio'])
                        if not points:continue
                        x=[r['achieved_concentration_ratio'] for r in points]
                        axes[0,column].plot(x,[r['initial_sensitivity_gain'] for r in points],ls,color=f'C{color}',alpha=.7)
                        axes[1,column].plot(x,[r['final_slope_ratio'] for r in points],ls,color=f'C{color}',alpha=.7,label=names[target] if kind=='gd' and path==0 else None)
            axes[0,column].set_title({'development':'Width 705, seed 30','validation':'Width 705, seed 33','wide':'Width 1409, seed 31'}[cohort])
            for ax in axes[:,column]:ax.set(xscale='log',yscale='log',xlabel='Initial √χ₆ / baseline');ax.axhline(1,color='k',lw=.5)
        axes[0,0].set_ylabel('Initial fine sensitivity / baseline')
        axes[1,0].set_ylabel('Final slope RMS / baseline')
        fig.legend(*axes[1,0].get_legend_handles_labels(),loc='outside upper center',ncol=3,fontsize=9)
        fig.savefig(output/'concentration_dose.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        for i,target in enumerate(SIX):
            points=[r for r in interventions if r['study']=='development' and r['target']==target and r['kind']=='gd' and r['path']==0]
            points.sort(key=lambda r:r['achieved_concentration_ratio'])
            if not points:continue
            x=[r['achieved_concentration_ratio'] for r in points]
            axes[0].plot(x,[r['final_relative_eval_error'] for r in points],'-o',label=names[target])
            axes[1].plot(x,[r['A_tracking'] for r in points],'-o')
        axes[0].set(ylabel='Final raw relative output error',xscale='log')
        axes[1].set(ylabel='Tracking contribution to slope squared norm',xscale='log')
        for ax in axes:ax.set_xlabel('Initial √χ₆ / baseline')
        axes[0].legend(fontsize=7,ncol=2)
        fig.savefig(output/'dose_output_tracking.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();analyze(args.inputs,args.output)
