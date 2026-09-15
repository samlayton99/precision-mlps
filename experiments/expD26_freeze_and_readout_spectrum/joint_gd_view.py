"""Plot completed simultaneous GD updates and parameter histories from saved states."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiments.expD26_freeze_and_readout_spectrum import nudge_correlation as n


def main():
    cfg=n.late.config()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(cfg['threads'])
    x=n.base.original.midpoint_grid(cfg['n_train'])
    y=n.base.original.matched.target_values('runge',x,cfg)
    with np.load(n.RESULTS/'data/nudge_correlation_5000.npz') as source:
        saved={name:{k.split('__',1)[1]:source[k] for k in source.files if k.startswith(name+'__')}
               for name in ('early','fitted')}
    initial={'early':n.base.original.initial_state('xavier',cfg)}
    with np.load(n.RESULTS/'data/late_freeze_runge.npz') as source:
        initial['fitted']={k:source['warmup__'+k][-1] for k in n.KEYS}

    cases,histories,reports,summary={},{},{},{}
    for name,s in saved.items():
        p=initial[name];slots=s['selected_neurons']
        first=n.independent_fork(p,torch.tensor(x),torch.tensor(y),cfg['learning_rate'],set())
        np.testing.assert_array_equal(abs(first['a'][slots]),s['gamma'][0])
        np.testing.assert_array_equal(abs(first['v'][slots]),s['coefficient'][0])
        gamma=np.vstack((abs(p['a'][slots]),s['gamma']))
        coefficient=np.vstack((abs(p['v'][slots]),s['coefficient']))
        dg,dc=np.diff(gamma,axis=0),np.diff(coefficient,axis=0)
        # The earlier plot probed state k -> k+1. This figure uses completed
        # updates k-1 -> k, including update 0 -> 1 and excluding 5000 -> 5001.
        np.testing.assert_array_equal(dg[1:],s['delta_gamma'][:-1])
        np.testing.assert_array_equal(dc[1:],s['delta_coefficient'][:-1])
        error0=np.linalg.norm(np.tanh(x[:,None]*p['a']+p['b'])@p['v'][:-1]+p['v'][-1]-y)/np.linalg.norm(y)
        cases[name]=dict(delta_gamma=dg,delta_coefficient=dc,step=s['step'],actual_step=s['actual_step'])
        reports[name]=n.fit(dg,dc)
        histories[name]=dict(step=np.arange(len(gamma)),gamma=gamma.mean(axis=1),
                             coefficient=coefficient.mean(axis=1),error=np.r_[error0,s['relative_error']])
        summary[name]=dict(initial_error=float(error0),final_error=float(s['relative_error'][-1]),
                           initial_mean_gamma=float(gamma[0].mean()),final_mean_gamma=float(gamma[-1].mean()),
                           initial_mean_readout=float(coefficient[0].mean()),final_mean_readout=float(coefficient[-1].mean()),
                           updates=len(dg),selected_neurons=len(slots),
                           joint_differences_match_prior_probes_bitwise=True,fit=reports[name])
    n.plot(cases,reports,'joint_gd_raw_updates',raw=True,joint=True)
    plot_histories(histories)
    (n.RESULTS/'data/joint_gd_view_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def plot_histories(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.ticker import ScalarFormatter,MaxNLocator

    fig,axes=plt.subplots(3,2,figsize=(14,10.5),dpi=180,sharex=True)
    norm=Normalize(0,5000)
    labels=['Training relative $L_2$ error',r'Mean scale $\mathrm{mean}_j|\gamma_j|$',
            r'Mean readout size $\mathrm{mean}_j|c_j|$']
    for col,name in enumerate(('early','fitted')):
        h=cases[name];t=h['step']
        for row,key in enumerate(('error','gamma','coefficient')):
            ax=axes[row,col];v=h[key]
            points=np.column_stack((t,v))
            segments=np.stack((points[:-1],points[1:]),axis=1)
            line=LineCollection(segments,cmap='viridis',norm=norm,linewidth=2.5)
            line.set_array(t[1:]);ax.add_collection(line)
            ax.set_xlim(0,5000)
            low,high=float(v.min()),float(v.max());pad=max((high-low)*.12,abs(high)*1e-12)
            ax.set_ylim(low-pad,high+pad)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False,useMathText=True))
            ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
            if col==0:ax.set_ylabel(labels[row],fontsize=12,labelpad=12)
        axes[0,col].set_title('Runge: Xavier initialization' if col==0 else 'Runge: already fitted, gamma ≈ 16',fontsize=14,pad=15)
        axes[2,col].set_xlabel('Completed joint-GD updates within this window',fontsize=11,labelpad=10)
    fig.suptitle('Nothing frozen: loss, gamma, and readout evolve together',fontsize=18,y=.97)
    fig.subplots_adjust(left=.14,right=.88,top=.86,bottom=.15,wspace=.40,hspace=.27)
    cax=fig.add_axes([.92,.28,.012,.45]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),cax=cax)
    cb.set_label('Completed update within the window',labelpad=10)
    fig.text(.50,.065,
             'Ordinary joint GD at η=0.002: all slopes, hidden biases, readouts, and output bias update every step.\n'
             'Parameter means use the same 128 selected neurons as the scatter; all 177 neurons participate in training.\n'
             'All axes are linear, with separate vertical ranges to show the motion. Gamma is |a| in tanh(ax+b).\n'
             'Xavier window: total steps 0–5000. Fitted window: total steps 6369–11369. No readout solves enter training.',
             ha='center',va='center',fontsize=10)
    fig.savefig(n.RESULTS/'joint_gd_parameter_evolution.png');plt.close(fig)


if __name__=='__main__':
    main()
