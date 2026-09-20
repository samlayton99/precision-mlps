"""Controlled scale landscape and geometry-rate intervention; offline LS only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD24_gd_residual_spectrum import four_way_comparison as old

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/checkpoint_D_optimizers/expD25_scale_barrier'
FIGURES = RESULTS / 'figures'


def configuration():
    return old.config() | yaml.safe_load((HERE/'config.yaml').read_text())


def snapshots(total):
    return np.unique(np.r_[np.arange(min(10, total)+1),
                           np.rint(np.geomspace(11, total, 40)).astype(int) if total > 10 else [],
                           [2000] if total>=2000 else [], total]).astype(int)


def arrays_save(path, data, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, config_json=json.dumps(cfg), **data)
    temp.replace(path)


def load_case(path,cfg,*,geometry_mode='raw',geometry_optimizer='sgd'):
    keys=('resolution','halo','seed','steps','learning_rate','n_train','n_eval',
          'envelope_sigma','target_modes','target_amplitudes','readout_rcond','domain')
    with np.load(path,allow_pickle=False) as f:
        previous=json.loads(str(f['config_json']))
        different=[k for k in keys if previous.get(k)!=cfg.get(k)]
        if different:raise ValueError(f'Cached configuration differs for {path.name}: {different}')
        case={k:f[k] for k in f.files if k!='config_json'}
    if str(case.get('geometry_mode','raw'))!=geometry_mode:
        raise ValueError(f'Cached geometry mode differs for {path.name}')
    if str(case.get('geometry_optimizer','sgd'))!=geometry_optimizer:
        raise ValueError(f'Cached geometry optimizer differs for {path.name}')
    return case


def train(target, arm, multiplier, cfg, *, seed=None, geometry_mode='raw',geometry_optimizer='sgd'):
    cfg = cfg if seed is None else cfg | dict(seed=seed)
    x=old.midpoint_grid(cfg['n_train']); y=old.matched.target_values(target,x,cfg)
    initial=old.initial_state(arm,cfg)
    p={k: torch.nn.Parameter(torch.tensor(v)) for k,v in initial.items()}
    tx,ty=torch.tensor(x),torch.tensor(y)
    centers=torch.tensor(-initial['b']/initial['a'])
    assert geometry_mode in ('raw','scale_only','shared_scale')
    if geometry_mode=='shared_scale':
        np.testing.assert_array_equal(initial['a'],np.full_like(initial['a'],initial['a'][0]))
    geo_params=[p['a'],p['b']] if geometry_mode=='raw' else [p['a']]
    assert geometry_optimizer in ('sgd','adam')
    if geometry_optimizer=='sgd':
        optim=torch.optim.SGD([dict(params=geo_params,lr=cfg['learning_rate']*multiplier),
                               dict(params=[p['v']],lr=cfg['learning_rate'])])
        geo_optim=None
    else:
        optim=torch.optim.SGD([p['v']],lr=cfg['learning_rate'])
        geo_optim=torch.optim.Adam(geo_params,lr=cfg['learning_rate']*multiplier)
    chosen=snapshots(cfg['steps']); lookup=set(chosen)
    result={k: [] for k in ('steps','a','b','v','raw_scale_gradient','centered_scale_gradient','center_coupling_gradient')}
    loss_history=[]; gamma_history=[]; motion_history=[]; path_history=[]
    ids=np.arange(cfg['halo'],cfg['halo']+cfg['resolution'])
    initial_gamma=abs(initial['a'][ids]); previous_gamma=initial_gamma.copy(); path=np.zeros(len(ids))
    failed=-1
    for step in range(cfg['steps']+1):
        optim.zero_grad(set_to_none=True)
        if geo_optim is not None:geo_optim.zero_grad(set_to_none=True)
        current_b=p['b'] if geometry_mode=='raw' else -p['a']*centers
        hidden=torch.tanh(tx[:,None]*p['a']+current_b)
        r=hidden @ p['v'][:-1]+p['v'][-1]-ty
        loss=.5*torch.mean(r.square())
        if not torch.isfinite(loss) or loss.item()>1e12:
            failed=step; print('FAILED',target,arm,multiplier,step,flush=True); break
        loss.backward()
        a=p['a'].detach().numpy(); b=current_b.detach().numpy()
        gamma=abs(a[ids]); path+=abs(gamma-previous_gamma); previous_gamma=gamma.copy()
        loss_history.append(loss.item()); gamma_history.append([gamma.mean(),np.median(gamma),np.quantile(gamma,.1),np.quantile(gamma,.9)])
        motion_history.append([np.mean(gamma-initial_gamma),np.mean(abs(gamma-initial_gamma))]); path_history.append(np.mean(path))
        if step in lookup:
            result['steps'].append(step)
            for k in ('a','v'): result[k].append(p[k].detach().numpy().copy())
            result['b'].append(b.copy())
            sensitivity=(r[:,None]*p['v'][:-1]*(1-hidden.square())).detach()
            ga=torch.mean(tx[:,None]*sensitivity,dim=0).numpy(); gb=sensitivity.mean(dim=0).numpy()
            z=np.divide(-b,a,out=np.zeros_like(b),where=abs(a)>np.finfo(float).tiny)
            coupling=z*gb; centered=np.sign(a)*(ga-coupling)
            result['raw_scale_gradient'].append(ga)
            result['centered_scale_gradient'].append(centered)
            result['center_coupling_gradient'].append(coupling)
        if step % 500==0:
            print(target,arm,'multiplier',multiplier,'step',step,'relative L2',np.sqrt(2*loss.item()/np.mean(y*y)), 'mean gamma',gamma.mean(),flush=True)
        if step<cfg['steps']:
            if geometry_mode=='shared_scale':
                # Orthogonal projection of the per-neuron scale gradient onto
                # equal-scale displacements. For SGD the scalar LR is eta/width;
                # Adam normalizes this mean gradient with its own moments.
                p['a'].grad.fill_(p['a'].grad.mean().item())
            if geo_optim is not None:geo_optim.step()
            optim.step()
    result={k:np.asarray(v) for k,v in result.items()}
    result.update(loss=np.asarray(loss_history),gamma=np.asarray(gamma_history),motion=np.asarray(motion_history),
                  path=np.asarray(path_history),failed_step=np.array(failed),target=np.array(target),arm=np.array(arm),
                  multiplier=np.array(multiplier),seed=np.array(cfg['seed']),geometry_mode=np.array(geometry_mode),geometry_optimizer=np.array(geometry_optimizer))
    return result


def evaluate(case,cfg):
    target=str(case['target']); x=old.midpoint_grid(cfg['n_train']); xx=old.midpoint_grid(cfg['n_eval'])
    y=old.matched.target_values(target,x,cfg); yy=old.matched.target_values(target,xx,cfg)
    weights=np.full(len(x),1/len(x)); values=[]
    for index,step in enumerate(case['steps']):
        A=old.design(x,case['a'][index],case['b'][index]); Ae=old.design(xx,case['a'][index],case['b'][index])
        v,info=old.matched.rc.solve_readout(A,y,weights)
        pred=A@case['v'][index]; r=pred-y
        rstar=info['optimal_residual']; gap=.5*np.mean((r-rstar)**2)
        identity_error=abs(.5*np.mean(r*r)-info['loss']-gap)
        assert identity_error<1e-10*(1+.5*np.mean(r*r)),identity_error
        values.append([np.linalg.norm(Ae@case['v'][index]-yy)/np.linalg.norm(yy),
                       np.linalg.norm(Ae@v-yy)/np.linalg.norm(yy), info['rank'],info['readout_norm'],
                       info['loss'],gap,identity_error,info['stationarity']])
    case['evaluation']=np.asarray(values)
    return case


def landscape(cfg):
    original,_=old.load(old.RESULTS/'data/comparison.npz')
    factors=np.unique(np.r_[np.geomspace(.25,32,101),1.])
    x=old.midpoint_grid(cfg['n_train']); xx=old.midpoint_grid(cfg['n_eval']); w=np.full(len(x),1/len(x))
    data={'factors':factors}
    for target in old.TARGETS:
        c=original[f'{target}__gamma_1']; a,b,v=(c[k][-1] for k in ('a','b','v'))
        y=old.matched.target_values(target,x,cfg); yy=old.matched.target_values(target,xx,cfg)
        records=[]
        for factor in factors:
            A=old.design(x,a*factor,b*factor); Ae=old.design(xx,a*factor,b*factor)
            vr,info=old.matched.rc.solve_readout(A,y,w)
            records.append([np.linalg.norm(Ae@v-yy)/np.linalg.norm(yy),np.linalg.norm(Ae@vr-yy)/np.linalg.norm(yy),
                            info['readout_norm'],info['rank'],info['stationarity']])
        s=x[:,None]*a+b; h=np.tanh(s); r=h@v[:-1]+v[-1]-y
        j=(s*(1-h*h))@v[:-1]
        derivative=np.mean(r*j)
        def loss(logfactor):
            rr=old.design(x,a*np.exp(logfactor),b*np.exp(logfactor))@v-y
            return .5*np.mean(rr*rr)
        eps=1e-4; fd=(loss(eps)-loss(-eps))/(2*eps)
        np.testing.assert_allclose(derivative,fd,rtol=1e-6,atol=1e-9)
        data[target]=np.asarray(records); data[target+'_directional_gradient']=np.array([derivative,fd])
        print('LANDSCAPE',target,'directional derivative',derivative,'FD',fd,flush=True)
    arrays_save(RESULTS/'data/landscape.npz',data,cfg)
    plot_landscape(data)


def graphics():
    FIGURES.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_landscape(data):
    plt=graphics(); from matplotlib.lines import Line2D
    fig,axes=plt.subplots(4,3,figsize=(16,13),dpi=170,sharex=True)
    f=data['factors']
    for row,target in enumerate(old.TARGETS):
        v=data[target]; ax=axes[row,0]
        ax.plot(f,v[:,0],color='#2274b5',lw=1.7)
        ax.plot(f,v[:,1],color='#20966b',lw=1.7)
        ax.set(yscale='log',ylim=(1e-16,10),ylabel=old.matched.rc.LABELS[target]+'\nRelative L₂ error')
        derivative=data[target+'_directional_gradient'][0]
        ax.text(.03,.08,f'Current ∂L/∂log(scale): {derivative:+.3g}',transform=ax.transAxes,fontsize=9)
        axes[row,1].plot(f,v[:,2],color='#20966b',lw=1.7); axes[row,1].set(yscale='log',ylim=(1e-2,1e12),ylabel='Solved readout norm ‖v*‖₂')
        axes[row,2].plot(f,v[:,3],color='#8554a3',lw=1.7); axes[row,2].set(ylim=(0,185),ylabel='Retained SVD rank')
        for ax in axes[row]:
            ax.set(xscale='log',xlim=(.25,32)); ax.axvline(1,color='#666666',ls=':',lw=1); ax.grid(alpha=.18)
            if row==3: ax.set_xlabel('Common gamma multiplier (centers fixed)')
    for ax,title in zip(axes[0],['Fixed readout versus refitted readout','Coefficient cost of the refitted solution','Numerical feature dimension']):ax.set_title(title,pad=12)
    fig.suptitle('Are better scales available, and does the current loss point toward them?',fontsize=18,y=.986)
    fig.legend([Line2D([],[],color=c,lw=2) for c in ['#2274b5','#20966b']],['Current GD readout held fixed','Readout solved separately at each scale'],loc='upper center',bbox_to_anchor=(.5,.956),ncol=2,frameon=False)
    fig.subplots_adjust(top=.885,bottom=.10,left=.075,right=.98,hspace=.25,wspace=.28)
    fig.text(.5,.028,'Start from each final gamma-1, 2,000-step GD state; multiply all slopes and biases together, preserving centers. The dotted line is the actual state.\n'
             'All functions: same 1,024 training midpoints and 8,192 evaluation midpoints on [−1,1]. Readout solves use SVD cutoff 10⁻¹³. This is a controlled sweep, not training.\n'
             'Derivatives hold the current readout fixed and are checked by finite differences. Near-zero refit errors and large coefficients require numerical caution.',ha='center',fontsize=9)
    fig.savefig(FIGURES/'landscape.png'); plt.close(fig)


def plot_cases(cases,cfg,pilot=False,scale_only=False):
    plt=graphics(); from matplotlib.lines import Line2D
    targets=['sine_mixture'] if pilot else old.TARGETS
    arms=['gamma_1'] if pilot else (['gamma_1','gamma_4'] if scale_only else old.ARMS)
    multipliers=cfg['geometry_lr_multipliers']
    colors=[plt.cm.viridis({1:.12,100:.5,10000:.86}[m]) for m in multipliers]
    for mode in ('errors','motion'):
        fig,axs=plt.subplots(len(targets),len(arms),figsize=(9 if pilot else (12 if scale_only else 19),6 if pilot else 14),dpi=170,squeeze=False,sharex=True,sharey=True)
        for row,target in enumerate(targets):
            for col,arm in enumerate(arms):
                ax=axs[row,col]
                for multiplier,color in zip(multipliers,colors):
                    key=f'{target}__{arm}__{multiplier}'; c=cases.get(key)
                    if c is None:continue
                    if mode=='errors':
                        ax.plot(c['steps'],np.maximum(c['evaluation'][:,0],1e-16),color=color,lw=1.5)
                        ax.plot(c['steps'],np.maximum(c['evaluation'][:,1],1e-16),'--',color=color,lw=1.5)
                    else:
                        ax.plot(np.arange(len(c['motion'])),c['motion'][:,1],color=color,lw=1.5)
                    if c['failed_step']>=0:ax.axvline(int(c['failed_step']),color=color,ls=':',lw=1)
                ax.set_xscale('symlog',linthresh=2);ax.set_xlim(0,cfg['steps']);ax.grid(alpha=.18)
                ax.set_xticks([0,2,10,100,2000],labels=['0','2','10','100','2000'])
                if mode=='errors':ax.set(yscale='log',ylim=(1e-16,3))
                else:ax.set_yscale('symlog',linthresh=1e-8)
                if row==0:ax.set_title(dict(zip(old.ARMS,old.ARM_LABELS))[arm],fontsize=11)
                if col==0:ax.set_ylabel(old.matched.rc.LABELS[target]+('\nRelative L₂ error' if mode=='errors' else '\nMean |γ − γ₀|'))
                if row==len(targets)-1:ax.set_xlabel('GD step (log spacing after 2)')
        title='Centers fixed: change only the scale learning rate' if scale_only else 'Change only the geometry learning rate'
        fig.suptitle(title+(' — error' if mode=='errors' else ' — scale movement'),fontsize=14 if pilot or scale_only else 20,y=.987)
        handles=[Line2D([],[],color=c,lw=2,label=f'Geometry rate ×{m:g}') for c,m in zip(colors,multipliers)]
        if mode=='errors':handles += [Line2D([],[],color='black',lw=1.5,label='Ordinary evaluation'),Line2D([],[],color='black',ls='--',lw=1.5,label='LS evaluation only')]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.95),ncol=2 if pilot else 5,frameon=False,fontsize=9 if pilot else 11)
        fig.subplots_adjust(top=.75 if pilot else .86,bottom=.23 if pilot else .12,left=.15 if pilot else (.10 if scale_only else .07),right=.97,hspace=.23,wspace=.15)
        setting='Only the center-preserving scale rate changes; centers stay fixed.' if scale_only else 'Only the raw slope/bias rate changes.'
        fig.text(.5,.025,'Identical starts and samples; readout rate stays 0.002. '+setting+'\n'
                 'LS is evaluated after training, including step 0; cutoff 10⁻¹³. Scale movement averages 128 fixed interior-index slots.\n'
                 'All targets train and evaluate on [−1,1]. Dotted vertical lines mark failed runs; none are silently retuned.',ha='center',fontsize=7.5 if pilot else 9)
        folder=FIGURES/'pilot' if pilot else FIGURES
        folder.mkdir(parents=True,exist_ok=True)
        fig.savefig(folder/((mode if pilot else ('scale_only_' if scale_only else 'rate_')+mode)+'.png'));plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--landscape',action='store_true');parser.add_argument('--pilot',action='store_true');parser.add_argument('--full',action='store_true');parser.add_argument('--plot-only',action='store_true');parser.add_argument('--scale-only',action='store_true')
    args=parser.parse_args();cfg=configuration();torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True);RESULTS.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=cfg['threads']):
        if args.landscape:landscape(cfg)
        if args.scale_only:cfg=cfg | dict(geometry_lr_multipliers=[1,10000])
        if args.pilot or args.full or args.scale_only:
            cases={}
            for target in (['sine_mixture'] if args.pilot else old.TARGETS):
                for arm in (['gamma_1'] if args.pilot else (['gamma_1','gamma_4'] if args.scale_only else old.ARMS)):
                    for multiplier in cfg['geometry_lr_multipliers']:
                        key=f'{target}__{arm}__{multiplier}';path=RESULTS/'data'/f'{key}{"__scale_only" if args.scale_only else ""}.npz'
                        if path.exists():
                            case=load_case(path,cfg,geometry_mode='scale_only' if args.scale_only else 'raw')
                        else:
                            if args.plot_only:continue
                            case=evaluate(train(target,arm,multiplier,cfg,geometry_mode='scale_only' if args.scale_only else 'raw'),cfg);arrays_save(path,case,cfg)
                        cases[key]=case
                        print('COMPLETE',key,'final GD/refit',case['evaluation'][-1,:2],flush=True)
            plot_cases(cases,cfg,pilot=args.pilot,scale_only=args.scale_only)


if __name__=='__main__':main()
