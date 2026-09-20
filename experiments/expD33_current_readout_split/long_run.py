"""10,000-step current-J split, preserving the original optimizer and short runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD33_current_readout_split import run as exp
from experiments.expD31_split_adam import mu_refit_sweep as refits

RESULTS = exp.RESULTS / "long_run"


def config():
    return exp.config() | dict(steps=10000, diagnostic_snapshots=201)


def ordinary_control(target, cfg, learning_rate_at=None):
    """Ordinary Adam; expensive refit diagnostics only at saved states."""
    x = exp.profile.previous.midpoint_grid(cfg['n_train'])
    y = exp.profile.previous.matched.target_values(target, x, cfg)
    initial = exp.profile.initial_state('xavier', cfg)
    p = {k: torch.nn.Parameter(torch.tensor(v)) for k, v in initial.items()}
    tx, ty = torch.tensor(x), torch.tensor(y)
    opt = torch.optim.Adam(p.values(), lr=cfg['learning_rate'],
                           betas=tuple(cfg['adam_betas']), eps=cfg['adam_epsilon'])
    chosen = set(exp.profile.snapshots(cfg['steps'], cfg['diagnostic_snapshots'])) | {500,1000,2500,5000,7500}
    history, saved = [], {}
    best_F, best_step = np.inf, 0
    for step in range(cfg['steps']+1):
        if learning_rate_at is not None:
            for group in opt.param_groups:
                group['lr'] = learning_rate_at(cfg, step)
        opt.zero_grad(set_to_none=True)
        f = torch.tanh(tx[:, None]*p['a']+p['b'])@p['v'][:-1]+p['v'][-1]
        loss = .5*(f-ty).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Ordinary Adam became nonfinite at {target}/{step}')
        loss.backward()
        state = {k: v.detach().numpy() for k,v in p.items()}
        row = dict(step=step, L=float(loss.detach()), mean_gamma=float(np.mean(abs(state['a']))),
                   F=np.nan, refit_train_relative_l2=np.nan)
        if step in chosen:
            d = exp.projected_signals(state, x, y, cfg['readout_rcond'])
            row.update(F=d['F'], refit_train_relative_l2=d['refit_train_relative_l2'])
            saved[step] = {k:v.copy() for k,v in state.items()}
            if d['F'] < best_F:
                best_F, best_step = d['F'], step
        history.append(row)
        if step % 2000 == 0:
            print(f'{target}/ordinary Adam, step {step}: L={row["L"]:.5g}, gamma={row["mean_gamma"]:.6g}', flush=True)
        if step < cfg['steps']:
            opt.step()
    steps = sorted(saved)
    c = {k: np.asarray([r[k] for r in history]) for k in history[0]}
    c.update({k: np.asarray([saved[t][k] for t in steps]) for k in p})
    c.update(saved_steps=np.asarray(steps), best_step=np.array(best_step), status=np.array('complete'),
             target=np.array(target), mu=np.array(0))
    return c


def worker(task):
    target, mu = task
    cfg = config()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cfg['threads'])
    path = RESULTS / 'data' / f'{target}__{mu}.npz'
    ep = path.with_name(f'{target}__{mu}__refits.npz')
    with threadpool_limits(cfg['threads']):
        if path.exists():
            c = exp.prior.gd.load(path, cfg)
        else:
            start = time.perf_counter()
            c = exp.train(target, mu, cfg) if mu else ordinary_control(target, cfg)
            c['seconds'] = np.array(time.perf_counter()-start)
            exp.prior.gd.save(c, cfg, path)
        old = exp.reference(target, 0) if not mu else exp.prior.gd.load(
            exp.RESULTS/'data'/f'xavier__{target}__{mu}.npz', exp.config())
        for k in ('L', 'mean_gamma') + (('F',) if mu else ()):
            np.testing.assert_array_equal(c[k][:501], old[k])
        np.testing.assert_array_equal(c['v'][0], old['v'][0])
        if not ep.exists():
            rows = [refits.refit({k:c[k][i] for k in ('a','b','v')}, target, cfg)
                    for i in range(len(c['saved_steps']))]
            e = {k:np.asarray([r[k] for r in rows]) for k in rows[0]}
            np.savez_compressed(ep, steps=c['saved_steps'], **e)
    return dict(target=target, mu=mu, status=str(c['status']), steps=int(c['step'][-1]),
                seconds=float(c['seconds']))


def load_cases(cfg):
    cases, evaluations = {}, {}
    for target in cfg['targets']:
        for mu in [0]+cfg['mu_values']:
            path = RESULTS/'data'/f'{target}__{mu}.npz'
            cases[target,mu] = exp.prior.gd.load(path, cfg)
            with np.load(path.with_name(f'{target}__{mu}__refits.npz')) as f:
                evaluations[target,mu] = {k:f[k] for k in f.files}
    return cases, evaluations


def summarize(cases, evaluations, cfg):
    rows, checks, qi = [], [], {}
    for target in cfg['targets']:
        qi[target] = refits.refit(exp.profile.initial_state('qi_zero', cfg), target, cfg)
        for mu in [0]+cfg['mu_values']:
            c, e = cases[target,mu], evaluations[target,mu]
            best = int(np.argmin(e['refit_relative_l2']))
            r = dict(target=target, mu=mu, status=str(c['status']),
                     final_step=int(c['step'][-1]), final_actual_relative_l2=float(e['actual_relative_l2'][-1]),
                     final_refit_relative_l2=float(e['refit_relative_l2'][-1]),
                     final_refit_train_relative_l2=float(e['refit_train_relative_l2'][-1]),
                     final_gamma=float(c['mean_gamma'][-1]), best_saved_step=int(e['steps'][best]),
                     best_saved_refit_relative_l2=float(e['refit_relative_l2'][best]),
                     final_coefficient_norm=float(e['coefficient_norm'][-1]),
                     max_gamma_mean=float(c['mean_gamma'].max()), seconds=float(c['seconds']))
            if mu:
                ratio=c['out_step_norm'][:-1]/np.maximum(c['rest_step_norm'][:-1],1e-300)
                r.update(median_stream_ratio=float(np.median(ratio)),maximum_stream_ratio=float(ratio.max()),
                         initial_stream_ratio=float(ratio[0]),final_stream_ratio=float(ratio[-1]))
            rows.append(r)
            for which, index in [('final',len(e['steps'])-1),('best_saved',best)]:
                state={k:c[k][index] for k in ('a','b','v')}
                # Retain coefficients fitted on the original grid; increase only scoring density.
                checks.append(dict(target=target,mu=mu,which=which,step=int(e['steps'][index]),
                                   **refits.refit(state,target,cfg,n_eval=8*cfg['n_eval'])))
    summary=dict(runs=rows,dense_grid_checks=checks,qi_reference=qi)
    (RESULTS/'data/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def plot(cases, evaluations, summary, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MaxNLocator, FuncFormatter

    colors=dict(zip(cfg['mu_values'],plt.cm.viridis(np.linspace(.05,.9,len(cfg['mu_values'])))))
    figdir=RESULTS/'figures';figdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,4,figsize=(20,12),dpi=170,sharex=True)
    max_error=max(float(e['refit_relative_l2'].max()) for e in evaluations.values())
    max_error=max(max_error,max(float(np.nanmax(c['refit_train_relative_l2'])) for c in cases.values()))
    for col,target in enumerate(cfg['targets']):
        axes[0,col].set_title(exp.profile.LABELS[target],fontsize=16,pad=14)
        gammas=[]
        for mu in [0]+cfg['mu_values']:
            c,e=cases[target,mu],evaluations[target,mu]
            color,style=('black',':') if not mu else (colors[mu],'-')
            axes[0,col].plot(c['step'],c['L'],color=color,ls=style,lw=1.5)
            steps=c['step'] if mu else e['steps']
            values=c['refit_train_relative_l2'] if mu else e['refit_train_relative_l2']
            axes[1,col].plot(steps,values,color=color,ls=style,lw=1.5)
            axes[1,col].plot(e['steps'],e['refit_relative_l2'],ls='none',marker='o',ms=2.3,
                             markerfacecolor='none',markeredgewidth=.6,color=color)
            axes[2,col].plot(c['step'],c['mean_gamma'],color=color,ls=style,lw=1.5)
            gammas.extend(c['mean_gamma'])
        axes[1,col].axhline(summary['qi_reference'][target]['refit_relative_l2'],color='.65',ls='--',lw=1)
        for row in (0,1):
            axes[row,col].set_yscale('log');axes[row,col].yaxis.set_major_locator(LogLocator(base=10,numticks=5))
        axes[1,col].set_ylim(1e-16,max(2,2*max_error))
        lo,hi=min(gammas),max(gammas);pad=max(hi-lo,.01)*.08
        axes[2,col].set_ylim(max(0,lo-pad),hi+pad)
        axes[2,col].yaxis.set_major_locator(MaxNLocator(nbins=5))
        axes[2,col].ticklabel_format(axis='y',style='plain',useOffset=False)
        axes[2,col].set_xlabel('Training updates',fontsize=12)
        for ax in axes[:,col]:
            ax.set_xlim(0,cfg['steps']);ax.set_xticks([0,2000,4000,6000,8000,10000])
            ax.tick_params(labelsize=9);ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
    axes[0,0].set_ylabel(r'Actual loss $L=\frac{1}{2}\mathrm{mean}(e^2)$'+'\n(log scale)',fontsize=12)
    axes[1,0].set_ylabel(r'Refitted relative $L_2$ error'+'\n(log scale)',fontsize=12)
    axes[2,0].set_ylabel(r'Mean scale $\overline{\gamma}=\mathrm{mean}|a_k|$'+'\n(linear scale)',fontsize=12)
    fig.suptitle(r'Current $J^T r_\perp$ · separate Adam streams · 10,000 updates from Xavier',fontsize=19,y=.98)
    handles=[Line2D([],[],color=colors[mu],lw=2,label=f'μ = {mu:,}') for mu in cfg['mu_values']]
    handles += [Line2D([],[],color='black',ls=':',lw=2,label='Ordinary Adam'),
                Line2D([],[],color='.65',ls='--',label='QI refit reference (middle row)'),
                Line2D([],[],color='.3',ls='none',marker='o',mfc='none',ms=5,label='Independent-grid refit check')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.927),ncol=4,frameon=False,fontsize=11.5)
    fig.subplots_adjust(left=.085,right=.98,top=.82,bottom=.15,hspace=.25,wspace=.28)
    fig.text(.5,.062,
             'θ update = −η[μ Adam(h) + Adam(∇θL−h)], h = Jᵀ(I−Pτ)r at the current readout; independent moment histories.\n'
             'μ stays outside Adam. Unchanged η=0.002, β=(0.9,0.999), εAdam=10⁻⁸, seed 0, width and training samples.\n'
             'Middle: reconstruct the least-squares model on training samples at every split step; circles score saved refits on 8,192 independent samples.\n'
             'Solved coefficients never replace the trained readout. SVD cutoff 10⁻¹³σ₁. Each extended run reproduces its original first 500 steps.',
             ha='center',va='center',fontsize=10)
    fig.savefig(figdir/'training.png');plt.close(fig)

    fig,axes=plt.subplots(1,4,figsize=(19,5.4),dpi=170,sharex=True,sharey=True)
    for ax,target in zip(axes,cfg['targets']):
        for mu in cfg['mu_values']:
            c=cases[target,mu]
            ratio=c['out_step_norm'][:-1]/np.maximum(c['rest_step_norm'][:-1],1e-300)
            ax.plot(c['step'][:-1],np.maximum(ratio,1e-18),color=colors[mu],lw=1.1,alpha=.9)
        ax.axhline(1,color='.35',ls='--',lw=1)
        ax.set_yscale('log');ax.set_title(exp.profile.LABELS[target],fontsize=14,pad=12)
        ax.set_xlabel('Training updates',fontsize=11);ax.set_xlim(0,cfg['steps']);ax.set_xticks([0,5000,10000])
        ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel(r'$\|\eta\mu\,u_h\|_2\,/\,\|\eta\,u_{\rm rest}\|_2$'+'\nRatio of the two proposed step norms',fontsize=12)
    fig.suptitle('Does the amplified projected stream become large enough to affect training?',fontsize=17,y=.99)
    fig.legend(handles=handles[:len(cfg['mu_values'])],loc='upper center',bbox_to_anchor=(.5,.91),ncol=5,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.075,right=.985,top=.73,bottom=.24,wspace=.22)
    fig.text(.5,.075,'Below 1: the amplified projected contribution has smaller norm than the other contribution. Above 1: larger norm.\n'
             'These are the actual post-Adam, post-multiplier contributions, including optimizer history. The ratio does not show their angle or sum.',
             ha='center',fontsize=10)
    fig.savefig(figdir/'stream_strength.png');plt.close(fig)

    fig,axes=plt.subplots(1,4,figsize=(19,5.4),dpi=170,sharex=True,sharey=True)
    low=min(float(c['mean_gamma'].min()) for c in cases.values())/1.3
    high=max(float(c['mean_gamma'].max()) for c in cases.values())*1.3
    for ax,target in zip(axes,cfg['targets']):
        for mu in [0]+cfg['mu_values']:
            c=cases[target,mu]
            ax.plot(c['step'],c['mean_gamma'],color=colors[mu] if mu else 'black',
                    ls='-' if mu else ':',lw=1.5)
        ax.set_yscale('log');ax.set_ylim(low,high)
        ax.set_yticks([.1,1,10,100]);ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
        ax.set_title(exp.profile.LABELS[target],fontsize=14,pad=12)
        ax.set_xlim(0,cfg['steps']);ax.set_xticks([0,5000,10000]);ax.set_xlabel('Training updates',fontsize=11)
        ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel(r'Mean $\gamma=\mathrm{mean}|a_k|$'+'\n(logarithmic y-axis)',fontsize=12)
    fig.suptitle('The same gamma histories, with the smaller-scale trajectories visible',fontsize=17,y=.99)
    fig.legend(handles=handles[:len(cfg['mu_values'])+1],loc='upper center',bbox_to_anchor=(.5,.91),
               ncol=6,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.07,right=.985,top=.73,bottom=.24,wspace=.22)
    fig.text(.5,.08,'Same trajectories and mean across all neurons as the main figure. All four panels share the same logarithmic range.\n'
             'Only the axis changes; no additional training or altered parameterization.',ha='center',fontsize=10)
    fig.savefig(figdir/'gamma_log.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--targets',nargs='+')
    parser.add_argument('--train-only',action='store_true')
    parser.add_argument('--analyze-only',action='store_true')
    parser.add_argument('--plot-only',action='store_true');args=parser.parse_args()
    cfg=config();(RESULTS/'data').mkdir(parents=True,exist_ok=True)
    if not (args.plot_only or args.analyze_only):
        tasks=[(t,mu) for t in (args.targets or cfg['targets']) for mu in [0]+cfg['mu_values']]
        for i,task in enumerate(tasks,1):
            print(f'COMPLETE {i}/{len(tasks)}: '+json.dumps(worker(task)),flush=True)
    if args.train_only:return
    cases,evaluations=load_cases(cfg)
    torch.set_num_threads(cfg['threads'])
    with threadpool_limits(cfg['threads']):
        summary=json.loads((RESULTS/'data/summary.json').read_text()) if args.plot_only else summarize(cases,evaluations,cfg)
        plot(cases,evaluations,summary,cfg)
    for row in summary['runs']:print(json.dumps(row),flush=True)


if __name__=='__main__':main()
