"""Measured replacements for the two schematics; no fitted schematic curves."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titlesize':13,'axes.labelsize':12,
                     'legend.frameon':False,'savefig.dpi':190})

NAMES=['sine','quadratic','mixed','runge']
TITLES=['Sine','Quadratic','Mixed sine','Runge']
EQUATIONS=[r'$f(x)=\sqrt{2}\sin(2\pi x)$',r'$f(x)=\sqrt{5}\,x^2$',
           r'$f(x)=[\sin(2\pi x)+0.1\sin(20\pi x)]/\sqrt{0.505}$',
           r'$f(x)=1/(1+25x^2)$']
COLORS=['#2563a8','#a455a1','#d77b18','#23866e','#20252d']


def optimizer_figures(cfg, source, output):
    paths=[source,source/'ablations/sqrt_allowance',output/'data/training/alpha_scale',
           source/'ablations/neighbor_unscaled',output/'data/training/neighbor_sqrt_cumulative']
    labels=[r'Direct weights: $c=\theta$',
            r'Weights scaled by $\sqrt{\alpha}$',r'Weights scaled by $\alpha$',
            'Adjacent differences (unscaled)',r'Adjacent differences scaled by $\sqrt{s}$']
    records=[]
    ref=np.load(source/'data/reference.npz')
    gamma=ref['gammas'];h=2/256
    for path in paths:
        z=np.load(path/'data/trajectory.npz')
        assert z['step']==cfg['training_steps']
        late=z['eval_rel_l2'][z['steps']>=19000]
        records.append({'final':z['eval_rel_l2'][-1],
                        'late_min':late.min(axis=0),'late_max':late.max(axis=0),
                        'late_median':np.median(late,axis=0)})
    np.savez_compressed(output/'data/optimizer_figure_values.npz',gammas=gamma,
                        labels=np.asarray(labels),**{k:np.stack([r[k] for r in records]) for k in records[0]})
    for f,name in enumerate(NAMES):
        fig,axs=plt.subplots(1,2,figsize=(12.9,6.2),sharex=True,sharey=True)
        fig.subplots_adjust(left=.09,right=.98,bottom=.34,top=.79,wspace=.13)
        for method,ax in enumerate(axs):
            for i,(r,label) in enumerate(zip(records,labels)):
                ax.plot(gamma,r['final'][method,:,f],'-o',color=COLORS[i],lw=1.8,
                        markersize=4,label=label)
            ax.set_xscale('log');ax.set_yscale('log')
            ax.set_xlabel(r'Frozen slope $\gamma$')
            ax.set_title(f"{'GD' if method==0 else 'Adam'} · 20,000 updates",pad=9)
            ax.set_xticks([gamma[0],1,4,16,32,64,128],
                          ['0.135','1','4','16','32','64','128'])
            ax.set_xlim(gamma[0]*.8,155)
            ax.grid(axis='y',which='major',alpha=.18)
            ax.axvline(32,color='.55',ls=':',lw=1.3,zorder=0)
            ax.text(32,.97,r'$\lambda=0.25$',transform=ax.get_xaxis_transform(),
                    va='top',ha='right',fontsize=10,color='.35',rotation=90)
        values=np.stack([r['final'][:,:,f] for r in records])
        low=10**np.floor(np.log10(values.min())-.12)
        axs[0].set_ylim(low,1.45)
        axs[0].set_ylabel('Test relative $L_2$ error\n'+r'$\|f_{\rm model}-f\|_2/\|f\|_2$')
        fig.suptitle(f'Frozen-geometry readout training — {TITLES[f]}',y=.98,fontsize=17)
        fig.text(.535,.90,EQUATIONS[f],ha='center',fontsize=12)
        handles,labels=axs[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.54,.11),ncol=2,fontsize=10.5,
                   columnspacing=2.5,handlelength=2.6)
        fig.text(.535,.065,r'Uniform centers · $h=1/128$ · 305 tanh neurons · zero readout · 1,021 training / 4,093 test samples',
                 ha='center',fontsize=10)
        fig.text(.535,.028,r'GD: $\eta=1/\|AM\|_2^2$.  Adam: $\eta=10^{-3}$.  Reference scales $\alpha$ fixed; $s_j=\sum_{\ell\leq j}\alpha_\ell$.',
                 ha='center',fontsize=10)
        path=output/'optimizer_comparison.png' if name=='mixed' else output/'companions'/f'{name}_optimizer.png'
        path.parent.mkdir(exist_ok=True,parents=True)
        fig.savefig(path,facecolor='white');plt.close(fig)


def theorem_figures(cfg,output):
    z=np.load(output/'data/theorem_diagnostics.npz')
    gamma=z['gammas'];k=z['k'];rho=z['rho']
    main=int(np.argmin(abs(z['relative_cutoffs']-cfg['primary_svd_relative_cutoff'])))
    epsilon=cfg['relative_tolerance']
    for name in cfg['theorem_targets']:
        f=NAMES.index(name)
        fig,axs=plt.subplots(1,3,figsize=(19.3,7.4))
        fig.subplots_adjust(left=.055,right=.985,bottom=.36,top=.78,wspace=.31)
        ax=axs[0]
        needed=z['necessary_energy'][:,f]
        ax.plot(k,np.where(needed>0,needed,np.nan),color='#20252d',lw=2,
                label=r'$[D_k-\epsilon]_+^2$: required tail')
        for g,col in zip(cfg['access_panel_gammas'],['#d77b18','#23866e']):
            i=int(np.flatnonzero(gamma==g)[0])
            values=np.where(z['access_resolved'][i,:,f],z['access'][i,:,f],np.nan)
            ax.plot(k,values,color=col,lw=2,label=rf'$\mu_k/\|B\|_2^2$: measured, $\gamma={g}$')
        ax.plot(k,z['normalized_envelope'][0],color='#be4343',ls='--',lw=1.8,
                label=r'$\mathcal{B}_k/\|B\|_2^2$: upper bound, $\Gamma=4$')
        required=k[needed>0]
        if len(required) and required[-1]<k[-1]:
            stop=int(required[-1])+1
            ax.axvspan(stop,k[-1],color='.95',zorder=-1)
            ax.text((stop+k[-1])/2,.98,'No required tail\n'+r'$(D_k\leq\epsilon)$',
                    transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=9,color='.4')
        ax.set_yscale('log');ax.set_xlim(0,k[-1]);ax.set_ylim(1e-34,10)
        ax.set_xticks(np.arange(0,97,16))
        ax.set_xlabel(r'Polynomial degree cutoff $k$')
        ax.set_ylabel('Tail energy / normalized squared access')
        ax.set_title('(a) Required correction and readout access',pad=12)
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=6))
        ax.legend(loc='upper left',bbox_to_anchor=(-.02,-.24),fontsize=10,handlelength=2.5)

        ax=axs[1]
        for g,col in zip(cfg['damping_panel_gammas'],['#2563a8','#d77b18','#23866e']):
            i=int(np.flatnonzero(gamma==g)[0])
            values=z['tail_damping_remainder'][i,f]
            ax.plot(rho,values[main],color=col,lw=2,label=rf'$\gamma={g}$')
            ax.fill_between(rho,values.min(axis=0),values.max(axis=0),color=col,alpha=.15)
        ax.axhline(.5,color='.6',lw=1,ls=':')
        ax.set_xscale('log');ax.set_xlim(rho[0],rho[-1]);ax.set_ylim(-.025,1.025)
        ax.set_xticks([1e-24,1e-18,1e-12,1e-6,1,100])
        ax.set_xlabel(r'Relative damping $\rho=\zeta/\|B\|_2^2$')
        ax.set_ylabel('Residual norm remaining\n'+r'$\mathcal{R}(q_{16};\zeta)$')
        ax.set_title('(b) One damped Gauss–Newton step',pad=12)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=3,fontsize=11)
        ax.text(.5,-.43,'Same unit residual for every curve:\n'+r'$q_{16}=Q_{16}y/\|Q_{16}y\|_2$',
                transform=ax.transAxes,ha='center',va='top',fontsize=10)

        ax=axs[2]
        times=z['flow_time'][:,f,main]
        ax.plot(gamma,times,'-o',color='#2563a8',lw=2,ms=5,
                label='Spectral gradient-flow prediction · Eq. (11)')
        ax.fill_between(gamma,z['flow_time'][:,f].min(axis=1),z['flow_time'][:,f].max(axis=1),
                        color='#2563a8',alpha=.15)
        ax.plot(gamma,z['directional_bound'][:,f],'--s',color='#d77b18',lw=1.8,ms=5,
                label=r'Lower bound using measured $\mu_k$ · Eq. (4)')
        ax.plot(gamma,z['bounded_slope_bound'][:,f],':^',color='#23866e',lw=2,ms=6,
                label=r'Lower bound using slope cap $\Gamma$ · Eqs. (4, 6)')
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks(gamma,[f'{g:g}' for g in gamma])
        ax.set_xlabel(r'Frozen slope $\gamma$ ($\Gamma=\gamma$)')
        ax.set_ylabel('Normalized gradient-flow time\n'+r'$T_\epsilon\|B\|_2^2$')
        ax.set_title(r'(c) Time to $10^{-3}$ training relative error',pad=12)
        ax.set_ylim(1,10**np.ceil(np.log10(times.max())+.1))
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=6))
        ax.legend(loc='upper left',bbox_to_anchor=(-.04,-.24),fontsize=10,handlelength=2.5)
        for ax in axs:
            ax.grid(axis='y',which='major',alpha=.15)
        fig.suptitle(f'Frozen tanh readout: polynomial tails, damping, and the time bound — {TITLES[f]}',
                     y=.98,fontsize=18)
        fig.text(.52,.91,EQUATIONS[f]+r'   ·   Direct readout $M=I$   ·   $h=1/128$, $W=305$, $m=1{,}021$',
                 ha='center',fontsize=12)
        fig.text(.52,.083,r'$Q_k=I-P_k$ removes the best degree-$k$ polynomial; $D_k=\|Q_ky\|_2/\|y\|_2$; '
                 r'$\mu_k=\|B^Tq_k\|_2^2$.  $B$ and $y$ include $1/\sqrt{m}$.',ha='center',fontsize=11)
        fig.text(.52,.042,'Lower bounds maximize over degrees 0–96. Unresolved access points are omitted. '
                 'Faint bands show SVD-cutoff sensitivity; flow time is not a measured iteration count.',ha='center',fontsize=10)
        path=output/'theorem_diagnostics.png' if name=='mixed' else output/'companions'/f'{name}_theorem.png'
        path.parent.mkdir(exist_ok=True,parents=True)
        fig.savefig(path,facecolor='white');plt.close(fig)


def plot_all(cfg,source,output):
    optimizer_figures(cfg,source,output)
    theorem_figures(cfg,output)
    print(f'Wrote two main figures and four companions to {output}',flush=True)
