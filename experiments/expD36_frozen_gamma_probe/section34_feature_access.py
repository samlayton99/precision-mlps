"""Frozen-readout assays and target-weighted kernel diagnostics for learned features.

Kernel ratios describe normalized plain GD, not an Adam learning-rate law.
"""
import argparse
import hashlib
import json
from pathlib import Path
from fractions import Fraction
import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from .adam_feature_probe_spectrum import slow_energy, CUTOFFS
    from .section34_analyze import binned_traces, json_safe, style
except ImportError:
    from adam_feature_probe_spectrum import slow_energy, CUTOFFS
    from section34_analyze import binned_traces, json_safe, style



def metric_spectrum(phi, target, residual, diagonal=None):
    """Both vector energies use ||target||², including residual energy."""
    if diagonal is not None:
        phi = phi*np.sqrt(diagonal)[None,:]
    j = phi/np.sqrt(len(target))
    u,s,_ = scipy.linalg.svd(j,full_matrices=False,check_finite=True,lapack_driver='gesdd')
    mu=s*s;rho=mu/mu[0];normalizer=target@target
    weights=[];checks={}
    for name,vector in [('target',target),('residual',residual)]:
        projected=u.T@vector;w=projected**2/normalizer
        complement=float(np.sum((vector-u@projected)**2)/normalizer)
        total=float(vector@vector/normalizer)
        np.testing.assert_allclose(np.sum(w)+complement,total,atol=2e-12*max(total,1e-25),rtol=2e-12)
        direct=float(np.sum((j.T@vector)**2)/normalizer);spectral=float(mu@w)
        np.testing.assert_allclose(direct,spectral,rtol=2e-10,atol=2e-13*mu[0]*max(total,1e-25))
        checks[name]=dict(total_energy=total,complement_energy=complement,energy_closure=float(np.sum(w)+complement),direct_quadratic_form=direct,spectral_quadratic_form=spectral)
        weights.append(w)
    return mu,rho,weights[0],weights[1],checks


def residual_slow_energy(rho, weights, total, cutoffs):
    return np.clip(total-np.array([np.sum(weights[rho>c]) for c in cutoffs]),0.,total)


def analyze(input_path, manifest_path, run_paths, output, kernels=True):
    data=np.load(input_path);manifest=json.loads(manifest_path.read_text())
    a,b=data['a'],data['b'];y=data['target'];x=data['train_x']
    vx,vy=data['validation_x'],data['validation_target'];ex,ey=data['eval_x'],data['eval_target']
    rows=manifest['geometries'];output.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(input_path.read_bytes()).hexdigest()
    if digest!=manifest['input_sha256']:raise ValueError('Input hash differs from manifest')
    runs=[]
    for path in run_paths:
        meta=json.loads((path/'metadata.json').read_text());state=np.load(path/'state.npz')
        if meta['input_sha256']!=digest:raise ValueError('Frozen run uses different input')
        runs.append((path,meta,state['w'],int(state['count']),np.load(path/'relative_error.npy',mmap_mode='r'),state['v']))
    if len({r[3] for r in runs})!=1:raise ValueError('Frozen assay budgets differ')
    summaries=[];curves={};kernel_arrays={};thresholds=np.logspace(-14,0,281)
    for gi,row in enumerate(rows):
        phi=np.column_stack((np.tanh(x[:,None]*a[gi]+b[gi]),np.ones(len(x))))
        vphi=np.column_stack((np.tanh(vx[:,None]*a[gi]+b[gi]),np.ones(len(vx))))
        ephi=np.column_stack((np.tanh(ex[:,None]*a[gi]+b[gi]),np.ones(len(ex))))
        candidates=[]
        for run_index,(path,meta,w,count,raw,v) in enumerate(runs):
            val=np.linalg.norm(vphi@w[gi]-vy[:,None],axis=0)/np.linalg.norm(vy)
            train=np.linalg.norm(phi@w[gi]-y[:,None],axis=0)/np.linalg.norm(y)
            finite=np.isfinite(train)
            np.testing.assert_allclose(train[finite],raw[-1,gi,finite],rtol=1e-8,atol=5e-11)
            for ri,recipe in enumerate(meta['recipes']):
                candidates.append(dict(run_index=run_index,recipe_index=ri,**recipe,validation_error=float(val[ri]),train_error=float(train[ri])))
        good=[r for r in candidates if np.isfinite(r['validation_error'])]
        if not good:
            summaries.append(dict(**row,status='all frozen recipes nonfinite',candidates=candidates));continue
        selected=min(good,key=lambda r:r['validation_error']);run_index=selected['run_index'];ri=selected['recipe_index']
        path,meta,w,count,raw,v=runs[run_index]
        eval_error=float(np.linalg.norm(ephi@w[gi,:,ri]-ey)/np.linalg.norm(ey))
        item=dict(**row,selected=selected,eval_error=eval_error,candidates=candidates,assay_steps=count)
        for field,arr in binned_traces(raw[:,gi:gi+1,:],ri).items():curves[f'g{gi}_{field}']=arr.squeeze() if field!='steps' else arr
        if kernels:
            residual=phi@w[gi,:,ri]-y
            mu,rho,weights,residual_weights,checks=metric_spectrum(phi,y,residual)
            # A numerical resolution convention, not a proof of unrepresentability.
            resolved=rho>1e-18
            unresolved=float(np.clip(1-np.sum(weights[resolved]),0,1))
            item.update(kernel_checks=checks,mu_max=float(mu[0]),slow_energy={f'{c:g}':float(e) for c,e in zip(CUTOFFS,slow_energy(rho,weights,CUTOFFS))},
                        unresolved_target_energy_at_rho_1e_18=unresolved,
                        resolved_projection_residual=float(np.sqrt(unresolved)))
            kernel_arrays[f'g{gi}_relative_rate']=rho;kernel_arrays[f'g{gi}_target_weights']=weights
            kernel_arrays[f'g{gi}_slow_energy']=slow_energy(rho,weights,thresholds)
            residual_total=checks['residual']['total_energy']
            kernel_arrays[f'g{gi}_residual_weights']=residual_weights
            kernel_arrays[f'g{gi}_residual_slow_energy']=residual_slow_energy(rho,residual_weights,residual_total,thresholds)
            item['raw_residual_energy_below_cutoff']={f'{c:g}':float(e) for c,e in zip(CUTOFFS,residual_slow_energy(rho,residual_weights,residual_total,CUTOFFS))}
            epsilon=meta['config']['epsilon']
            diagonal=1/(np.sqrt(v[gi,:,ri]/(1-.999**count))+epsilon)
            amu,arho,aw,arw,achecks=metric_spectrum(phi,y,residual,diagonal)
            item['frozen_second_moment_metric']=dict(mu_max=float(amu[0]),checks=achecks,
                target_energy_below_cutoff={f'{c:g}':float(e) for c,e in zip(CUTOFFS,slow_energy(arho,aw,CUTOFFS))},
                residual_energy_below_cutoff={f'{c:g}':float(e) for c,e in zip(CUTOFFS,residual_slow_energy(arho,arw,residual_total,CUTOFFS))},
                bias_correction_count=count,epsilon=epsilon,learning_rate_included=False,
                schedule=selected['schedule'],interpretation='Endpoint second-moment metric only; momentum and evolving preconditioners remain. Relative spectrum omits learning rate, including at cooled cosine endpoints.')
            kernel_arrays[f'g{gi}_adaptive_relative_rate']=arho
            kernel_arrays[f'g{gi}_adaptive_target_weights']=aw
            kernel_arrays[f'g{gi}_adaptive_residual_weights']=arw
            kernel_arrays[f'g{gi}_adaptive_slow_energy']=slow_energy(arho,aw,thresholds)
            kernel_arrays[f'g{gi}_adaptive_residual_slow_energy']=residual_slow_energy(arho,arw,residual_total,thresholds)
        summaries.append(item)
        print(json.dumps(dict(geometry=row['name'],eval_error=eval_error,recipe=selected)),flush=True)
    comparisons = {}
    for optimizer in sorted({r['optimizer'] for r in summaries if 'optimizer' in r}):
        subset = [r for r in summaries if r.get('optimizer') == optimizer and 'eval_error' in r]
        final = max(r['snapshot_step'] for r in subset)
        baseline = {r['seed']:r['eval_error'] for r in subset if r['snapshot_step'] == final and r.get('slope_multiplier',1.) == 1.}
        stages = {}
        for stage in sorted({r['snapshot_step'] for r in subset}):
            values = [r['eval_error'] for r in subset if r['snapshot_step'] == stage and r.get('slope_multiplier',1.) == 1.]
            stages[str(stage)] = dict(eval_errors=values, median_eval_error=float(np.median(values)))
        ratios = {}
        for factor in [4.,16.]:
            values = [r['eval_error']/baseline[r['seed']] for r in subset if r['snapshot_step'] == final and r.get('slope_multiplier') == factor]
            if values:ratios[str(factor)] = dict(paired_error_ratios=values,median_paired_error_ratio=float(np.median(values)))
        comparisons[optimizer] = dict(snapshot_assays=stages,intervention_to_unscaled_ratios=ratios)
    summary=dict(comparisons=comparisons,input_sha256=digest,assay_steps=runs[0][3],selection='Each geometry chooses one fixed recipe by final validation output error; every geometry receives the same candidate schedules/rates and budget.',
                 kernel_interpretation='mu_i/mu_max and target weights diagnose normalized plain GD. They are not an Adam rate prediction.',
                 numerical_resolution='rho <= 1e-18 target energy is numerically unresolved, not an established capacity floor.',geometries=summaries)
    (output/'summary.json').write_text(json.dumps(json_safe(summary),indent=2,allow_nan=False)+'\n')
    np.savez_compressed(output/'assay_traces.npz',**curves)
    if kernels:np.savez_compressed(output/'kernel_diagnostics.npz',cutoffs=thresholds,**kernel_arrays)
    plots(summaries,kernel_arrays,thresholds,output)
    uniform_trajectories(summaries,curves,output)
    if kernel_arrays:adaptive_plot(summaries,kernel_arrays,thresholds,output)


def plots(rows,kernel_arrays,thresholds,output):
    style();optimizers=list(dict.fromkeys(r.get('optimizer','uniform') for r in rows))
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7})
    learned=[r for r in rows if r.get('family')=='learned' and 'eval_error' in r]
    if learned:
        fig,axes=plt.subplots(1,2,figsize=(5.5,2.65),layout='constrained')
        colors={'adam':'#0072B2','gd':'#D55E00'}
        for optimizer in optimizers:
            subset=[r for r in learned if r.get('optimizer')==optimizer]
            if not subset:continue
            stages=sorted({r['snapshot_step'] for r in subset});color=colors.get(optimizer,'#333333')
            values=[np.array([r['eval_error'] for r in subset if r['snapshot_step']==s]) for s in stages]
            label = ('Adam' if optimizer == 'adam' else 'GD') + ' features'
            axes[0].plot(np.array(stages)/1e6,[np.median(v) for v in values],'-o',color=color,label=label)
            axes[0].fill_between(np.array(stages)/1e6,[min(v) for v in values],[max(v) for v in values],color=color,alpha=.12)
            final=max(stages);interventions=[r for r in rows if r.get('optimizer')==optimizer and r.get('snapshot_step')==final and 'eval_error' in r]
            factors=sorted({r.get('slope_multiplier',1.) for r in interventions})
            vals=[np.array([r['eval_error'] for r in interventions if r.get('slope_multiplier',1.)==s]) for s in factors]
            axes[1].plot(factors,[np.median(v) for v in vals],'-o',color=color,label=label)
            axes[1].fill_between(factors,[min(v) for v in vals],[max(v) for v in vals],color=color,alpha=.12)
        axes[0].set_xlabel('Source joint updates (millions)');axes[0].set_title('Frozen Adam readout restart')
        axes[1].set_xlabel('Slope and intercept multiplier');axes[1].set_xscale('log',base=4);axes[1].set_xticks([1,4,16],['1','4','16']);axes[1].set_title('Same centers, increased slopes')
        for ax in axes:ax.set_yscale('log');ax.set_ylabel('Final relative output error');ax.legend();ax.grid(alpha=.15)
        for ext in ['png','pdf']:fig.savefig(output/f'feature_access_assays.{ext}',dpi=220)
        plt.close(fig)
    if kernel_arrays:
        fig,axes=plt.subplots(1,len(optimizers),figsize=(5.5,2.8),squeeze=False,layout='constrained')
        for ax,optimizer in zip(axes[0],optimizers):
            ids=[i for i,r in enumerate(rows) if r.get('optimizer','uniform')==optimizer and r.get('family')!='slope_intervention' and f'g{i}_slow_energy' in kernel_arrays]
            if ids and all(rows[i]['family']=='uniform' for i in ids):
                for i,color in zip(ids,plt.cm.viridis(np.linspace(.05,.9,len(ids)))):
                    ax.plot(thresholds,kernel_arrays[f'g{i}_slow_energy'],color=color,label=f"λ = {Fraction(rows[i]['lambda_rms']).limit_denominator(32)}")
                stages=[]
            else:stages=sorted({rows[i].get('snapshot_step',0) or 0 for i in ids})
            for stage,color in zip(stages,plt.cm.viridis(np.linspace(.05,.9,len(stages)))):
                chosen=[i for i in ids if (rows[i].get('snapshot_step',0) or 0)==stage]
                values=np.array([kernel_arrays[f'g{i}_slow_energy'] for i in chosen])
                ax.plot(thresholds,np.median(values,axis=0),color=color,label=f'{stage/1e6:g}M source updates')
                ax.fill_between(thresholds,np.min(values,axis=0),np.max(values,axis=0),color=color,alpha=.1)
            ax.set_title('Uniform dictionaries' if optimizer=='uniform' else optimizer.upper());ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel(r'Relative GD rate cutoff $\mu/\mu_{\max}$');ax.set_ylabel('Target energy below cutoff');ax.grid(alpha=.15);ax.legend(fontsize=7)
        for ext in ['png','pdf']:fig.savefig(output/f'target_weighted_kernel.{ext}',dpi=220)
        plt.close(fig)




def uniform_trajectories(rows, curves, output):
    ids=[i for i,r in enumerate(rows) if r.get('family')=='uniform' and 'selected' in r]
    if not ids:return
    style();plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7})
    fig,axes=plt.subplots(2,1,figsize=(5.5,4.5),layout='constrained')
    for i,color in zip(ids,plt.cm.viridis(np.linspace(.05,.9,len(ids)))):
        row=rows[i];t=curves[f'g{i}_steps']/1e6;y=curves[f'g{i}_median']
        fraction=Fraction(row['lambda_rms']).limit_denominator(32)
        label=f'λ = {fraction}'
        axes[0].plot(t,y,color=color,label=label)
        axes[0].fill_between(t,curves[f'g{i}_low'],curves[f'g{i}_high'],color=color,alpha=.1,lw=0)
        if any(np.isclose(row['lambda_rms'],v) for v in [.09375,.125,.25]):
            axes[1].plot(t,y,color=color,label=label)
            axes[1].fill_between(t,curves[f'g{i}_low'],curves[f'g{i}_high'],color=color,alpha=.1,lw=0)
    axes[0].set_title('Uniform dictionaries, tuned frozen Adam')
    axes[1].set_title('Comparing bandwidth near 0.1 with 0.25')
    for ax in axes:
        ax.set_yscale('log');ax.set_xlabel('Readout updates (millions)');ax.set_ylabel('Relative output L2 error');ax.grid(alpha=.15);ax.legend(fontsize=7,ncol=4 if ax==axes[0] else 3,loc='upper right')
    axes[0].set_ylim(top=axes[0].get_ylim()[1]*4)
    for ext in ['png','pdf']:fig.savefig(output/f'uniform_adam_trajectories.{ext}',dpi=220)
    plt.close(fig)


def adaptive_plot(rows, arrays, thresholds, output):
    groups=[]
    for lam in [.03125,.25]:
        ids=[i for i,r in enumerate(rows) if r.get('family')=='uniform' and np.isclose(r['lambda_rms'],lam)]
        if ids:groups.append((f'Uniform λ = {Fraction(lam).limit_denominator(32)}',ids))
    for optimizer in sorted({r['optimizer'] for r in rows if 'optimizer' in r}):
        ids=[i for i,r in enumerate(rows) if r.get('optimizer')==optimizer and r.get('family')=='learned']
        if ids:
            final=max(rows[i]['snapshot_step'] for i in ids)
            groups.append((f'Final {optimizer.upper()} features',[i for i in ids if rows[i]['snapshot_step']==final]))
    if not groups:return
    plt.rcParams.update({'font.size':7.5,'axes.labelsize':7.5,'axes.titlesize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7})
    fig,axes=plt.subplots(2,len(groups),figsize=(5.5,4.2),squeeze=False,layout='constrained')
    for col,(label,ids) in enumerate(groups):
        for row,suffix in enumerate(['slow_energy','residual_slow_energy']):
            ax=axes[row,col]
            for prefix,color,name in [('', '#0072B2','Raw kernel'),('adaptive_','#D55E00','Frozen second moments')]:
                values=np.array([arrays[f'g{i}_{prefix}{suffix}'] for i in ids if f'g{i}_{prefix}{suffix}' in arrays])
                if not len(values):continue
                ax.plot(thresholds,np.median(values,axis=0),color=color,label=name)
                ax.fill_between(thresholds,np.min(values,axis=0),np.max(values,axis=0),color=color,alpha=.12)
            ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Relative metric eigenvalue cutoff')
            ax.set_ylabel(('Target' if row==0 else 'Residual')+' energy / target energy')
            ax.grid(alpha=.15);ax.legend(fontsize=7)
        axes[0,col].set_title(label)
    fig.suptitle('Endpoint second-moment geometry',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(output/f'frozen_second_moment_diagnostic.{ext}',dpi=220)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--run',action='append',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--skip-kernels',action='store_true')
    args=p.parse_args()
    with threadpool_limits(limits=2):analyze(args.input,args.manifest,args.run,args.output,not args.skip_kernels)


if __name__=='__main__':main()
