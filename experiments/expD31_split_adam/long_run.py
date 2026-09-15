"""Long Xavier VarPro split: selected multipliers, constant versus scheduled rate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.expD31_split_adam import run as prior
from experiments.expD31_split_adam import mu_refit_sweep as refits
from experiments.expD33_current_readout_split import long_run as current

RESULTS=prior.RESULTS/'long_run'
SCHEDULES=('constant','cosine')


def config(schedule):
    return prior.config()|dict(arms=['xavier'],mu_values=[50,100,250,500],steps=10000,
                               diagnostic_snapshots=201,save_best_state=True,
                               lr_schedule=None if schedule=='constant' else
                               dict(kind='cosine',start_step=0,min_factor=.001))


def case_path(target,mu,schedule):
    return RESULTS/'data'/f'{target}__{mu}__{schedule}.npz'


def run_case(target,mu,schedule):
    cfg=config(schedule);path=case_path(target,mu,schedule)
    ep=path.with_name(path.stem+'__refits.npz')
    if path.exists():c=prior.gd.load(path,cfg)
    elif not mu and schedule=='constant':
        # The completed 10k ordinary-Adam control has identical physical settings.
        old=current.RESULTS/'data'/f'{target}__0.npz'
        c=prior.gd.load(old,current.config())
        c['learning_rate']=np.full(len(c['step']),cfg['learning_rate'])
        prior.gd.save(c,cfg,path)
    else:
        start=time.perf_counter()
        c=prior.train(target,'xavier',mu,cfg) if mu else current.ordinary_control(target,cfg,prior.learning_rate_at)
        c['seconds']=np.array(time.perf_counter()-start)
        if 'learning_rate' not in c:c['learning_rate']=np.array([prior.learning_rate_at(cfg,t) for t in c['step']])
        prior.gd.save(c,cfg,path)
    initial=prior.profile.initial_state('xavier',cfg)
    for key in ('a','b','v'):np.testing.assert_array_equal(c[key][0],initial[key])
    # Cheap regression check where a constant-rate reference already exists.
    if schedule=='constant' and mu in (0,100,500):
        old=prior.gd.load(refits.RESULTS/'data'/f'{target}__{mu}.npz',refits.config()) if mu else current.exp.reference(target,0)
        for key in ('L','mean_gamma')+(('F',) if mu else ()):
            np.testing.assert_array_equal(c[key][:501],old[key])
    if not ep.exists():
        if not mu and schedule=='constant':
            with np.load(current.RESULTS/'data'/f'{target}__0__refits.npz') as f:e={k:f[k] for k in f.files}
        else:
            records=[refits.refit({k:c[k][i] for k in ('a','b','v')},target,cfg) for i in range(len(c['saved_steps']))]
            e={k:np.asarray([r[k] for r in records]) for k in records[0]}
            e['steps']=c['saved_steps']
        np.savez_compressed(ep,**e)
    print(f'COMPLETE {target}/{schedule}/mu={mu}: {c["status"]}, steps={c["step"][-1]}',flush=True)


def load_cases():
    cases,evaluations={},{}
    for target in config('constant')['targets']:
        for schedule in SCHEDULES:
            for mu in [0]+config(schedule)['mu_values']:
                path=case_path(target,mu,schedule)
                cases[target,mu,schedule]=prior.gd.load(path,config(schedule))
                with np.load(path.with_name(path.stem+'__refits.npz')) as f:
                    evaluations[target,mu,schedule]={k:f[k] for k in f.files}
    return cases,evaluations


def summarize(cases,evaluations):
    cfg=config('constant');rows=[];checks=[]
    qi={t:refits.refit(prior.profile.initial_state('qi_zero',cfg),t,cfg) for t in cfg['targets']}
    for key,c in cases.items():
        target,mu,schedule=key;e=evaluations[key];best=int(np.argmin(e['refit_relative_l2']))
        audits=json.loads(str(c['audits_json'])) if mu else []
        row=dict(target=target,mu=mu,schedule=schedule,status=str(c['status']),final_step=int(c['step'][-1]),
                 final_actual_relative_l2=float(e['actual_relative_l2'][-1]),
                 final_refit_relative_l2=float(e['refit_relative_l2'][-1]),
                 final_gamma=float(c['mean_gamma'][-1]),final_coefficient_norm=float(e['coefficient_norm'][-1]),
                 best_saved_step=int(e['steps'][best]),best_saved_refit_relative_l2=float(e['refit_relative_l2'][best]),
                 best_saved_gamma=float(c['mean_gamma'][e['steps'][best]]),
                 final_rate=float(c['learning_rate'][-1]),
                 maximum_step_variation=max((a['relative_step_variation'] for a in audits),default=0))
        rows.append(row)
        for which,index in [('final',len(e['steps'])-1),('best_saved',best)]:
            state={k:c[k][index] for k in ('a','b','v')}
            checks.append(dict(target=target,mu=mu,schedule=schedule,which=which,step=int(e['steps'][index]),
                               rcond=cfg['readout_rcond'],n_eval=65536,
                               **refits.refit(state,target,cfg,n_eval=65536)))
            if mu:
                for cutoff in (1e-12,1e-14):
                    checks.append(dict(target=target,mu=mu,schedule=schedule,which=which,step=int(e['steps'][index]),
                                       rcond=cutoff,n_eval=cfg['n_eval'],**refits.refit(state,target,cfg,cutoff=cutoff)))
    s=dict(runs=rows,checks=checks,qi_reference=qi)
    (RESULTS/'data/summary.json').write_text(json.dumps(s,indent=2)+'\n')
    return s


def plot(cases,evaluations,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator,MaxNLocator
    cfg=config('constant');colors=dict(zip(cfg['mu_values'],plt.cm.viridis(np.linspace(.05,.9,4))))
    figdir=RESULTS/'figures';figdir.mkdir(parents=True,exist_ok=True)
    all_errors=np.concatenate([e['refit_relative_l2'] for e in evaluations.values()])
    loss_limits={};gamma_limits={}
    for target in cfg['targets']:
        cs=[c for (t,mu,sc),c in cases.items() if t==target]
        ls=np.concatenate([c['L'] for c in cs]);gs=np.concatenate([c['mean_gamma'] for c in cs])
        loss_limits[target]=(float(ls.min())*.6,float(ls.max())*1.6)
        pad=max(float(gs.max()-gs.min()),.01)*.08
        gamma_limits[target]=(max(0,float(gs.min())-pad),float(gs.max())+pad)
    for schedule in SCHEDULES:
        fig,axes=plt.subplots(3,4,figsize=(20,12),dpi=170,sharex=True)
        for col,target in enumerate(cfg['targets']):
            axes[0,col].set_title(prior.profile.LABELS[target],fontsize=16,pad=13)
            for mu in [0]+cfg['mu_values']:
                c,e=cases[target,mu,schedule],evaluations[target,mu,schedule]
                color=colors[mu] if mu else '.1';style='-' if mu else ':'
                axes[0,col].plot(c['step'],c['L'],color=color,ls=style,lw=1.4)
                steps=c['step'] if mu else e['steps']
                values=c['refit_train_relative_l2'] if mu else e['refit_train_relative_l2']
                axes[1,col].plot(steps,values,color=color,ls=style,lw=1.4)
                axes[1,col].plot(e['steps'],e['refit_relative_l2'],ls='none',marker='o',ms=2,
                                 markerfacecolor='none',markeredgewidth=.5,color=color)
                axes[2,col].plot(c['step'],c['mean_gamma'],color=color,ls=style,lw=1.4)
            axes[1,col].axhline(summary['qi_reference'][target]['refit_relative_l2'],color='.65',ls='--',lw=1)
            for row in (0,1):
                axes[row,col].set_yscale('log');axes[row,col].yaxis.set_major_locator(LogLocator(base=10,numticks=5))
            axes[0,col].set_ylim(*loss_limits[target])
            axes[1,col].set_ylim(1e-16,max(2,float(np.max(all_errors))*1.5))
            axes[2,col].set_ylim(*gamma_limits[target])
            axes[2,col].yaxis.set_major_locator(MaxNLocator(nbins=5));axes[2,col].ticklabel_format(axis='y',style='plain',useOffset=False)
            axes[2,col].set_xlabel('Training updates',fontsize=12)
            for ax in axes[:,col]:
                ax.set_xlim(0,10000);ax.set_xticks([0,2000,4000,6000,8000,10000]);ax.tick_params(labelsize=9)
                ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
        axes[0,0].set_ylabel(r'Actual loss $L=\frac{1}{2}\mathrm{mean}(e^2)$'+'\n(log scale)',fontsize=12)
        axes[1,0].set_ylabel(r'Refitted relative $L_2$ error'+'\n(log scale)',fontsize=12)
        axes[2,0].set_ylabel(r'Mean $\gamma=\mathrm{mean}|a_k|$'+'\n(linear scale)',fontsize=12)
        title='Constant learning rate' if schedule=='constant' else 'Cosine learning-rate decay from the start'
        fig.suptitle(r'Xavier · original VarPro $J_*$ split · 10,000 updates'+'\n'+title,fontsize=19,y=.99)
        handles=[Line2D([],[],color=colors[m],lw=2,label=f'μ = {m}') for m in cfg['mu_values']]
        handles += [Line2D([],[],color='.1',ls=':',label='Ordinary Adam: same schedule'),
                    Line2D([],[],color='.65',ls='--',label='QI refit reference'),
                    Line2D([],[],color='.3',ls='none',marker='o',mfc='none',ms=4,label='Independent-grid refit check')]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.915),ncol=4,frameon=False,fontsize=11)
        fig.subplots_adjust(left=.08,right=.985,top=.79,bottom=.15,hspace=.25,wspace=.27)
        rate_text='ηt = 0.002 for the whole run.' if schedule=='constant' else 'ηt = 0.002 × [0.001 + 0.999(1 + cos(πt/10,000))/2], from 0.002 to 0.000002.'
        fig.text(.5,.061,
                 'Geometry update = −ηt[μ Adam(DFτ) + Adam(DGτ)]; separate histories and the original numerical VarPro derivative. Readout uses ordinary Adam.\n'
                 +rate_text+' The common rate applies to both geometry and readout; no moment reset.\n'
                 'μ multiplies the F direction after Adam normalization; ε=10⁻⁸. Seed, width, samples, and SVD cutoff are unchanged. Matching panels have identical axes.\n'
                 'Middle: reconstructed training-refit error each split step; circles score saved refits on 8,192 independent samples. Coefficients are never installed.',
                 ha='center',va='center',fontsize=10)
        fig.savefig(figdir/f'{schedule}.png');plt.close(fig)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--targets',nargs='+');parser.add_argument('--train-only',action='store_true')
    parser.add_argument('--analyze-only',action='store_true');parser.add_argument('--plot-only',action='store_true');args=parser.parse_args()
    cfg=config('constant');(RESULTS/'data').mkdir(parents=True,exist_ok=True)
    torch.set_default_dtype(torch.float64);torch.set_num_threads(cfg['threads'])
    with threadpool_limits(cfg['threads']):
        if not (args.analyze_only or args.plot_only):
            for target in args.targets or cfg['targets']:
                for schedule in SCHEDULES:
                    for mu in [0]+cfg['mu_values']:run_case(target,mu,schedule)
        if args.train_only:return
        cases,evaluations=load_cases()
        s=json.loads((RESULTS/'data/summary.json').read_text()) if args.plot_only else summarize(cases,evaluations)
        plot(cases,evaluations,s)
    for r in s['runs']:print(json.dumps(r),flush=True)


if __name__=='__main__':main()
