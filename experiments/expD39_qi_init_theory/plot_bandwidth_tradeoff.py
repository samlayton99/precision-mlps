"""One-factor bandwidth comparisons from the independently audited pilots."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precision_d39_matplotlib')
import json
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from experiments.expD39_qi_init_theory.analyze import plt,save,OUT,base


def main():
    data=json.loads((OUT/'pilot_readout_probes.json').read_text())
    assert data['complete']
    rows={(r['task'],r['scheme']):r for r in data['rows']}
    arms=['soft24','centered','lambda05','lambda10']
    lambdas=[.0875,.25,.5,1.]
    fig,axes=plt.subplots(2,2,figsize=(10.5,8),sharex=True,sharey=True)
    for i,task in enumerate(['airfoil','kin8nm']):
        for j,split in enumerate(['train','val']):
            ax=axes[i,j]
            for kind,style in [('trained','-'),('ls',':')]:
                ys=[rows[task,a]['final'][kind][split] for a in arms]
                ax.plot(lambdas,ys,style+'o',color='tab:orange',lw=2,ms=5)
            ax.set(xscale='log',yscale='log',ylim=(1e-4,1),
                   xticks=lambdas,xticklabels=['.0875','.25','.5','1'],
                   title=base.TITLES[task]+(' · Fit' if split=='train' else ' · Validation')+'\nWidth 512',
                   xlabel=r'Initialization $\lambda$',ylabel='MSE at 10k steps')
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(which='both',alpha=.18)
    fig.suptitle('Increasing bandwidth can improve fitting and worsen generalization\nSame centered 24-bank recipe, seed, readout, batches and learning rate',y=1.04)
    fig.legend(handles=[Line2D([],[],color='tab:orange',ls=s,marker='o',label=l) for s,l in
                        [('-','Trained readout'),(':','Diagnostic LS readout')]],
               loc='upper center',bbox_to_anchor=(.5,.97),ncol=2)
    fig.subplots_adjust(top=.84,hspace=.35,wspace=.17)
    save(fig,'bandwidth_fit_generalization.png')


if __name__=='__main__':
    main()
