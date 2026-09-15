"""Amplify the current-readout J.T @ projected residual after Adam normalization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.linalg as sla
import torch
from threadpoolctl import threadpool_limits
import yaml

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiments.expD31_split_adam import run as prior

profile=prior.profile
HERE=Path(__file__).resolve().parent
RESULTS=ROOT/'results/checkpoint_D_optimizers/expD33_current_readout_split'


def config():
    return yaml.safe_load((HERE/'config.yaml').read_text())


def projected_signals(state,x,y,rcond,driver='gesvd',backend='numpy'):
    """Project the actual residual; contract with J at current v, with no dP term."""
    a,b,v=(state[k] for k in ('a','b','v'))
    h=profile.hidden(a,b,x,backend)
    A=np.c_[h,np.ones(len(x))]/np.sqrt(len(x));yn=y/np.sqrt(len(x))
    r=A@v-yn
    U,s,Vh=sla.svd(A,full_matrices=False,check_finite=False,lapack_driver=driver)
    keep=s>rcond*s[0];Ur=U[:,keep]
    rperp=r-Ur@(Ur.T@r)
    rstar=Ur@(Ur.T@yn)-yn
    vstar=Vh[keep].T@((Ur.T@yn)/s[keep])
    gout=profile.matrix_gradient_to_geometry(np.outer(rperp,v),h,x)
    gparallel=profile.matrix_gradient_to_geometry(np.outer(r-rperp,v),h,x)
    gL=profile.matrix_gradient_to_geometry(np.outer(r,v),h,x)
    return dict(L=.5*float(r@r),F=.5*float(rstar@rstar),gout=gout,gparallel=gparallel,gL=gL,
                rperp=rperp,rstar=rstar,vstar=vstar,rank=int(keep.sum()),
                refit_train_relative_l2=float(np.linalg.norm(A@vstar-yn)/np.linalg.norm(yn)),
                discarded_prediction_norm=float(np.linalg.norm(rperp-rstar)),
                retained_basis=Ur)


def train(target,mu,cfg):
    x=profile.previous.midpoint_grid(cfg['n_train'])
    y=profile.previous.matched.target_values(target,x,cfg)
    initial=profile.initial_state('xavier',cfg)
    p={k:torch.nn.Parameter(torch.tensor(v,dtype=torch.float64)) for k,v in initial.items()}
    tx,ty=torch.tensor(x),torch.tensor(y);eta=cfg['learning_rate'];width=len(initial['a'])
    readout=torch.optim.Adam([p['v']],lr=eta,betas=tuple(cfg['adam_betas']),eps=cfg['adam_epsilon'])
    streams=[prior.AdamStream(2*width,cfg['adam_betas'],cfg['adam_epsilon']) for _ in range(2)]
    audit_steps={0,1,10,50,100,250,500,1000,2500,5000,7500,cfg['steps']}
    chosen=set(profile.snapshots(cfg['steps'],cfg['diagnostic_snapshots']))|audit_steps
    history,snapshots,audits=[],{},[]
    best_F=np.inf;best_step=None;best_state=None;status='complete'
    for step in range(cfg['steps']+1):
        for value in p.values():value.grad=None
        prediction=torch.tanh(tx[:,None]*p['a']+p['b'])@p['v'][:-1]+p['v'][-1]
        loss=.5*(prediction-ty).square().mean()
        if not torch.isfinite(loss):status=f'nonfinite loss at {step}';break
        loss.backward()
        state={k:v.detach().numpy() for k,v in p.items()}
        gL=np.r_[p['a'].grad.numpy(),p['b'].grad.numpy()]
        try:d=projected_signals(state,x,y,cfg['readout_rcond'])
        except np.linalg.LinAlgError:status=f'SVD failed at {step}';break
        np.testing.assert_allclose(gL,d['gL'],rtol=1e-8,atol=2e-13)
        gout=d['gout'];grest=gL-gout
        uout,urest=[s.direction(g) for s,g in zip(streams,(gout,grest))]
        delta=eta*(mu*uout+urest)
        if not (np.isfinite(delta).all() and all(torch.isfinite(v.grad).all() for v in p.values())):
            status=f'nonfinite gradient or direction at {step}';break
        row=dict(step=step,L=float(loss.detach()),F=d['F'],mean_gamma=float(np.mean(abs(state['a']))),
                 refit_train_relative_l2=d['refit_train_relative_l2'],
                 max_gamma=float(np.max(abs(state['a']))),readout_norm=float(np.linalg.norm(state['v'])),
                 profile_coefficient_norm=float(np.linalg.norm(d['vstar'])),rank=d['rank'],
                 gL_norm=float(np.linalg.norm(gL)),gout_norm=float(np.linalg.norm(gout)),grest_norm=float(np.linalg.norm(grest)),
                 proposed_step_norm=float(np.linalg.norm(delta)),out_step_norm=float(eta*mu*np.linalg.norm(uout)),
                 rest_step_norm=float(eta*np.linalg.norm(urest)),
                 discarded_prediction_norm=d['discarded_prediction_norm'])
        history.append(row)
        if row['F']<best_F:
            best_F=row['F'];best_step=step;best_state={k:v.copy() for k,v in state.items()}
        if step in chosen:snapshots[step]={k:v.copy() for k,v in state.items()}
        last_state={k:v.copy() for k,v in state.items()}
        if step in audit_steps:
            alts=[projected_signals(state,x,y,cfg['readout_rcond'],driver='gesdd'),
                  projected_signals(state,x,y,cfg['readout_rcond'],backend='torch')]
            variation=max(np.linalg.norm(gout-a['gout']) for a in alts)
            delta_variation=max(np.linalg.norm(delta-eta*(mu*streams[0].direction(a['gout'])+
                                    streams[1].direction(gL-a['gout']))) for a in alts)
            audits.append(dict(step=step,ranks_agree=all(a['rank']==d['rank'] for a in alts),
                               signal_variation=float(variation),signal_norm=row['gout_norm'],
                               relative_step_variation=float(delta_variation/max(np.linalg.norm(delta),1e-300))))
        if step in (0,100,250,cfg['steps']) or (step and step%1000==0):
            print(f'{target}/mu={mu}, step {step}: L={row["L"]:.5g}, F={row["F"]:.5g}, gamma={row["mean_gamma"]:.6g}',flush=True)
        if step<cfg['steps']:
            for s,g in zip(streams,(gout,grest)):s.direction(g,advance=True)
            with torch.no_grad():
                p['a'].sub_(torch.as_tensor(delta[:width]));p['b'].sub_(torch.as_tensor(delta[width:]))
            # The readout uses its pre-step ordinary-loss gradient, never vstar.
            readout.step()
    if not history:raise FloatingPointError(status)
    snapshots[history[-1]['step']]=last_state;snapshots[best_step]=best_state
    saved_steps=sorted(snapshots)
    case={k:np.asarray([r[k] for r in history]) for k in history[0]}
    case.update({k:np.asarray([snapshots[t][k] for t in saved_steps]) for k in p})
    case.update(saved_steps=np.asarray(saved_steps),best_step=np.array(best_step),status=np.array(status),
                audits_json=np.array(json.dumps(audits)),target=np.array(target),mu=np.array(mu))
    return case


def reference(target,mu):
    path=prior.RESULTS/'data'/f'xavier__{target}__{mu}.npz'
    with np.load(path) as f:return {k:f[k] for k in f.files if k!='config_json'}


def evaluate(cases,cfg):
    x=profile.previous.midpoint_grid(cfg['n_train']);xe=profile.previous.midpoint_grid(cfg['n_eval'])
    rows=[]
    for (target,mu),c in cases.items():
        old=reference(target,0)
        for key in ('a','b','v'):np.testing.assert_array_equal(c[key][0],old[key][0])
        np.testing.assert_array_equal(c['v'][1],old['v'][1])
        y=profile.previous.matched.target_values(target,x,cfg)
        ye=profile.previous.matched.target_values(target,xe,cfg)
        for label,step in [('final',int(c['step'][-1])),('best',int(c['best_step']))]:
            i=int(np.flatnonzero(c['saved_steps']==step)[0]);state={k:c[k][i] for k in ('a','b','v')}
            for cutoff in (1e-12,1e-13,1e-14):
                d=projected_signals(state,x,y,cutoff)
                he=profile.hidden(state['a'],state['b'],xe)
                actual=he@state['v'][:-1]+state['v'][-1]
                fitted=he@d['vstar'][:-1]+d['vstar'][-1]
                rows.append(dict(target=target,mu=mu,which=label,step=step,cutoff=cutoff,F=d['F'],
                                 sampled_refit_relative_l2=float(np.sqrt(2*d['F']/np.mean(y*y))),
                                 independent_refit_relative_l2=float(np.linalg.norm(fitted-ye)/np.linalg.norm(ye)),
                                 actual_independent_relative_l2=float(np.linalg.norm(actual-ye)/np.linalg.norm(ye)),
                                 mean_gamma=float(np.mean(abs(state['a']))),
                                 coefficient_norm=float(np.linalg.norm(d['vstar'])),rank=d['rank']))
    (RESULTS/'data/evaluation.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def plot(cases,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator,MaxNLocator,ScalarFormatter
    colors=dict(zip(cfg['mu_values'],plt.cm.viridis(np.linspace(.05,.9,len(cfg['mu_values'])))))
    fig,axes=plt.subplots(3,4,figsize=(20,12),dpi=170,sharex=True)
    all_entries=[]
    for col,target in enumerate(cfg['targets']):
        entries=[(reference(target,0),'black',':',1.8),(reference(target,1000),'#999999','--',1.8)]
        entries += [(cases[target,mu],colors[mu],'-',1.7) for mu in cfg['mu_values']]
        all_entries.extend(entries)
        for c,color,style,width in entries:
            for row,key in enumerate(('L','F','mean_gamma')):
                if row==2 and style=='--':continue
                axes[row,col].plot(c['step'],np.maximum(c[key],1e-32) if row<2 else c[key],
                                   color=color,ls=style,lw=width)
        axes[0,col].set_title(profile.LABELS[target],fontsize=15,pad=15)
        for row in range(3):
            ax=axes[row,col];ax.set_xlim(0,cfg['steps']);ax.set_xticks([0,100,200,300,400,500])
            ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
            if row<2:ax.set_yscale('log');ax.yaxis.set_major_locator(LogLocator(base=10,numticks=6))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                fmt=ScalarFormatter(useOffset=False);fmt.set_scientific(False);ax.yaxis.set_major_formatter(fmt)
        axes[2,col].set_xlabel('Training updates',fontsize=12,labelpad=9)
    for row,key in enumerate(('L','F')):
        values=np.concatenate([c[key] for c,_,style,_ in all_entries])
        values=values[np.isfinite(values)]
        limits=(max(values[values>0].min()/3,1e-32),values.max()*2)
        for ax in axes[row]:ax.set_ylim(*limits)
    for col,target in enumerate(cfg['targets']):
        values=np.concatenate([reference(target,0)['mean_gamma']]+
                              [cases[target,mu]['mean_gamma'] for mu in cfg['mu_values']])
        pad=max((values.max()-values.min())*.1,abs(values.max())*1e-5)
        axes[2,col].set_ylim(values.min()-pad,values.max()+pad)
    axes[0,0].set_ylabel(r'Actual squared loss $L=\frac{1}{2}\mathrm{mean}(e^2)$'+'\n(log scale)',fontsize=12)
    axes[1,0].set_ylabel(r'Least-squares refitted loss $F_\tau$'+'\n(log scale; squared loss)',fontsize=12)
    axes[2,0].set_ylabel(r'Mean scale $\mathrm{mean}_j|a_j|$'+'\n(linear scale)',fontsize=12)
    fig.suptitle(r'Xavier: amplify $J_{\rm current}^{T}(I-P_\tau)r$ with separate Adam streams',fontsize=20,y=.98)
    handles=[Line2D([],[],color=colors[mu],lw=2,label=f'μ = {mu:,}') for mu in cfg['mu_values']]
    handles += [Line2D([],[],color='black',ls=':',lw=2,label='Ordinary Adam'),
                Line2D([],[],color='#999999',ls='--',lw=2,label='expD31: VarPro μ = 1,000 (loss rows)')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.925),ncol=4,frameon=False,fontsize=12)
    fig.subplots_adjust(left=.085,right=.98,top=.83,bottom=.15,hspace=.25,wspace=.27)
    fig.text(.5,.06,
             'Amplified stream h = Jᵀ(I−Pτ)r at the current readout. Other stream = ∇θL−h. μ acts after separate Adam normalization.\n'
             'Readout uses ordinary Adam; solved coefficients are used for diagnostics, never installed. Geometry includes slopes and biases.\n'
             'Same expD31 samples, initialization and settings: η=0.002, Adam β=(0.9,0.999), ε=10⁻⁸, N=128, 24 halo per side, 500 steps.\n'
             'Gamma uses a separate expanded linear range per function; the gray expD31 reference appears in the loss rows only. SVD cutoff: 10⁻¹³σ₁.',
             ha='center',va='center',fontsize=10)
    (RESULTS/'figures').mkdir(parents=True,exist_ok=True)
    fig.savefig(RESULTS/'figures/xavier.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--targets',nargs='+');parser.add_argument('--mu',nargs='+',type=int)
    parser.add_argument('--plot-only',action='store_true');parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args();cfg=config();torch.set_default_dtype(torch.float64);torch.set_num_threads(cfg['threads'])
    cases={}
    with threadpool_limits(limits=cfg['threads']):
        for target in args.targets or cfg['targets']:
            for mu in args.mu or cfg['mu_values']:
                path=RESULTS/'data'/f'xavier__{target}__{mu}.npz'
                if path.exists():c=prior.gd.load(path,cfg)
                elif args.plot_only or args.analyze_only:raise FileNotFoundError(path)
                else:
                    start=time.perf_counter();c=train(target,mu,cfg);c['seconds']=np.array(time.perf_counter()-start)
                    prior.gd.save(c,cfg,path)
                cases[target,mu]=c
        if len(cases)==len(cfg['targets'])*len(cfg['mu_values']):
            if not args.plot_only:evaluate(cases,cfg)
            plot(cases,cfg)
        else:print(f'Saved {len(cases)} requested cases.',flush=True)


if __name__=='__main__':main()
