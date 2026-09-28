"""The requested 4 x 6 comparison; identical axes and measured points."""
import csv
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR',os.path.join(tempfile.gettempdir(),'precisionmlps-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np


def draw(cfg,out,include_sqrt_h=False):
    from run import TITLES,EQUATIONS
    meta=json.loads((out/'data/metadata.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titlesize':12,'axes.labelsize':11})
    names=cfg['targets']
    fig,axes=plt.subplots(4,6,figsize=(23.5,13.7),sharex=True,sharey=True)
    fig.subplots_adjust(left=.063,right=.99,bottom=.14 if include_sqrt_h else .12,
                        top=.86,wspace=.17,hspace=.25)
    blue,orange,green='#2866ad','#d47816','#148568'
    selected=[]
    for row,n in enumerate(cfg['interior_resolutions']):
        z=np.load(out/'data'/f'N{n}.npz')
        lam=z['lambdas'];errors=z['eval_rel_l2']
        if include_sqrt_h:
            with np.load(out/'data/sqrt_h'/f'N{n}.npz') as extra:
                np.testing.assert_array_equal(lam,extra['lambdas'])
                sqrt_errors=extra['eval_rel_l2'].copy()
        width=len(z['centers'])
        for col,name in enumerate(names):
            ax=axes[row,col]
            ax.set_xscale('log');ax.set_yscale('log')
            ax.set_ylim(1e-16,2);ax.set_xlim(cfg['lambda_min'],cfg['lambda_max'])
            pred=meta['selections'][row][col]
            invalid=np.pi/(n*pred['delta'])
            ax.axvspan(cfg['lambda_min'],max(cfg['lambda_min'],invalid),
                       color='.94',zorder=-5)
            for method,color in [(0,blue),(1,orange)]:
                ax.plot(lam,errors[:,method,col],color=color,lw=1.55,marker='.',markersize=2.1)
            if include_sqrt_h:
                ax.plot(lam,sqrt_errors[:,col],color=green,lw=1.7,ls='--')
            baseline=int(np.flatnonzero(lam==cfg['standard_lambda'])[0])
            standard=float(errors[baseline,0,col])
            ax.axvline(cfg['standard_lambda'],color=blue,ls=':',lw=.9,alpha=.7)
            ax.scatter([cfg['standard_lambda']],[standard],marker='o',s=41,
                       color=blue,edgecolors='white',linewidths=.6,zorder=7)
            adjusted=None;same_lambda_ratio=None
            if pred['lambda'] is not None and pred['envelope_valid_at_prediction']:
                i=int(np.argmin(abs(lam-pred['lambda'])))
                assert abs(lam[i]-pred['lambda'])<1e-14
                adjusted=float(errors[i,1,col])
                same_lambda_ratio=float(errors[i,0,col]/adjusted)
                ax.axvline(pred['lambda'],color=orange,ls=':',lw=.9,alpha=.8)
                ax.scatter([pred['lambda']],[adjusted],marker='D',s=42,
                           color=orange,edgecolors='white',linewidths=.6,zorder=8)
            else:
                message=('FFT selector:\nno admissible '+r'$\lambda$' if pred['lambda'] is None
                         else 'Envelope invalid\nat predicted '+r'$\lambda$')
                ax.text(.96,.94,message,ha='right',va='top',transform=ax.transAxes,
                        fontsize=9,color='.35',bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':2})
            selected.append({'N':n,'W':width,'target':name,'lambda_standard':cfg['standard_lambda'],
                             'lambda_note':pred['lambda'],'selection_status':pred['status'],
                             'standard_rel_l2':standard,'note_rel_l2':adjusted,
                             'standard_over_note_error':standard/adjusted if adjusted else None,
                             'raw_over_scaled_at_note_lambda':same_lambda_ratio})
            if include_sqrt_h:
                selected[-1].update(sqrt_h_rel_l2_at_025=float(sqrt_errors[baseline,col]),
                                    raw_over_sqrt_h_at_025=float(standard/sqrt_errors[baseline,col]))
            ax.grid(axis='y',which='major',alpha=.18,lw=.5)
            ax.set_yticks([1e-16,1e-12,1e-8,1e-4,1])
            ax.xaxis.set_major_locator(FixedLocator([.05,.1,.25,.5,1]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:g}'))
            ax.xaxis.set_minor_locator(NullLocator())
            if row==0:
                ax.set_title(TITLES[name],pad=32,fontweight='normal')
                ax.text(.5,1.065,EQUATIONS[name],ha='center',va='bottom',
                        transform=ax.transAxes,fontsize=9)
            if row==3:ax.set_xlabel(r'Relative bandwidth $\lambda=\gamma h$')
            if col==0:ax.set_ylabel(rf'$N={n}$ · $W={width}$'+'\nTest relative $L_2$ error',fontsize=11)
    fig.suptitle(r'Raw, envelope, and $\sqrt{h}$ readout scaling' if include_sqrt_h
                 else 'Standard versus envelope-scaled least squares',fontsize=21,y=.987)
    fig.text(.525,.947,rf'Fixed halo $R={cfg["halo_per_side"]}$ per side at every width; same geometry, samples, and SVD cutoff',
             ha='center',fontsize=13)
    handles=[Line2D([],[],color=blue,lw=2,label='Standard: raw readout, $c=a$'),
             Line2D([],[],color=orange,lw=2,label=r'Note scaling: $D_{jj}=\sqrt{\alpha_j(\lambda)}$'),
             Line2D([],[],color=blue,marker='o',ls='',label=r'Standard choice: $\lambda=0.25$'),
             Line2D([],[],color=orange,marker='D',ls='',label=r'Note choice: $\lambda_{\rm pred}$'),
             Patch(facecolor='.94',label='Envelope formula undefined')]
    if include_sqrt_h:
        handles.insert(2,Line2D([],[],color=green,lw=2,ls='--',
                       label=r'$\sqrt{h}$: all hidden columns; bias unscaled'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.525,.047),
               ncol=3 if include_sqrt_h else 5,frameon=False,fontsize=11,columnspacing=1.7)
    fig.text(.525,.025,r'$N$: interior intervals; $W=N+49$: tanh neurons, including 24 halo per side. '
             r'2,049 training points · 8,191 independent test midpoints · $\mathrm{rcond}=10^{-14}$.',
             ha='center',fontsize=10.5)
    fig.text(.525,.006,r'Envelopes use current $\lambda$, with $\delta=0.25$ (Runge: $0.19$). '
             r'Gray region: $\pi/(N\lambda)\geq\delta$. All curves are measured; points joined without smoothing.',
             ha='center',fontsize=10)
    suffix='_sqrt_h' if include_sqrt_h else ''
    destination=out/f'comparison{suffix}.png'
    fig.savefig(destination,dpi=190,facecolor='white')
    plt.close(fig)
    with (out/'data'/f'selected_points{suffix}.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(selected[0]))
        writer.writeheader();writer.writerows(selected)
    comparisons=[r for r in selected if r['note_rel_l2'] is not None]
    summary={'selected_points':selected,
             'valid_note_selections':len(comparisons),
             'note_lower_error_count':sum(r['standard_over_note_error']>1 for r in comparisons),
             'note_more_than_2x_better_count':sum(r['standard_over_note_error']>2 for r in comparisons),
             'note_more_than_2x_worse_count':sum(r['standard_over_note_error']<.5 for r in comparisons)}
    (out/'data'/f'summary{suffix}.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'Saved {destination}',flush=True)
