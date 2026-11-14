"""Figure 2 fonts/palette/axes, one color per target, linestyle per solver."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/precisionmlps-c14-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator
import numpy as np

LABELS = [r'$e^x$', r'$\sin(4\pi x)$', r'$\sin(24\pi x)$',
          r'$\sin\!\left(8\pi(x+1)^2\right)$',
          r'$\sin(2\pi x)+10^{-3}\sin(40\pi x)$',
          r'$e^{-8(x-0.2)^2}\sin(8\pi x)$']
NAMES = ['exp','sine4','sine24','chirp','weak_high_frequency','packet']
COLORS = ['#0072B2','#E69F00','#8B5FBF','#D55E00','#009E73','#CC79A7']


def draw(out):
    out = Path(out)
    rows = [r for path in out.glob('N[0-9]*.json') for r in json.loads(path.read_text())]
    ns = sorted({r['N'] for r in rows})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.labelsize':18,'axes.titlesize':20,
                         'xtick.labelsize':11.5,'ytick.labelsize':11.5,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig, ax = plt.subplots(figsize=(11,7.4))
    fig.subplots_adjust(left=.105,right=.975,bottom=.13,top=.73)
    for name,color in zip(NAMES,COLORS):
        rr = sorted([r for r in rows if r['target']==name],key=lambda r:r['N'])
        for method,style,marker in [('actual','-','o'),('theory','--','s')]:
            ax.plot([r['N'] for r in rr],[r[f'{method}_error'] for r in rr],
                    color=color,ls=style,lw=2.4,marker=marker,ms=4,
                    mfc=color if method=='actual' else 'white',mew=1.1)
    handles = [Line2D([],[],color=c,lw=2.5,label=l) for c,l in zip(COLORS,LABELS)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.91),ncol=3,
               frameon=False,fontsize=11.5,columnspacing=1.5,handlelength=2.)
    methods=[Line2D([],[],color='.2',lw=2.4,ls='-',label='Actual: existing SVD solver'),
             Line2D([],[],color='.2',lw=2.4,ls='--',label='Theory: reference scaling + bound-based cutoff')]
    fig.legend(handles=methods,loc='upper center',bbox_to_anchor=(.54,.80),ncol=2,
               frameon=False,fontsize=11,handlelength=3.)
    fig.suptitle('Theoretical and actual readout procedures',fontsize=20,fontweight='bold',y=.98)
    ax.set_xscale('log',base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(FixedLocator(ns))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(3e-17,2)
    ax.set_xlim(min(ns)*.94,max(ns)*1.06)
    ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1.,),numticks=100))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: rf'$10^{{{int(round(np.log10(v)))}}}$' if int(round(np.log10(v)))%2==0 else ''))
    ax.yaxis.set_minor_locator(LogLocator(base=10,subs=range(2,10),numticks=200))
    ax.yaxis.set_minor_formatter(NullFormatter())
    for side in ('top','right'):ax.spines[side].set_visible(False)
    for side in ('bottom','left'):ax.spines[side].set_linewidth(1.5)
    ax.tick_params(which='major',direction='out',length=5,width=1.25,pad=7)
    ax.tick_params(which='minor',direction='out',length=2.2,width=.65)
    ax.grid(axis='y',which='major',alpha=.13)
    ax.set_axisbelow(True)
    ax.set_xlabel(r'Interior intervals $N$')
    ax.set_ylabel(r'Relative $L^2$ error')
    fig.text(.54,.025,r'FP64; $\lambda=0.25$; $R=\lceil\sqrt{N}\rceil$; matched neurons and samples; independent evaluation grid.',
             ha='center',fontsize=10,color='.3')
    fig.savefig(out/'comparison.png',dpi=220)
    fig.savefig(out/'comparison.pdf')
    fig.savefig(out/'comparison.svg')
    plt.close(fig)
