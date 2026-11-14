"""Plot the original-theorem comparison using the Figure 2 visual style."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-c14-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expC14_theory_solver_comparison.plot import COLORS,LABELS,NAMES


def draw(out):
    out=Path(out);cfg=yaml.safe_load((out/'config.yaml').read_text())
    rows=[r for p in out.glob('N[0-9]*.json') for r in json.loads(p.read_text())]
    ns=sorted({r['N'] for r in rows})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':18,
                         'xtick.labelsize':11.5,'ytick.labelsize':11.5,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(11,7.4));fig.subplots_adjust(left=.105,right=.975,bottom=.13,top=.73)
    for name,color in zip(NAMES,COLORS):
        rr=sorted([r for r in rows if r['target']==name],key=lambda r:r['N'])
        for arm,style,marker in [('actual','-','o'),('theory','--','s')]:
            ax.plot([r['N'] for r in rr],[r[f'{arm}_error'] for r in rr],color=color,
                    lw=2.4,ls=style,marker=marker,ms=2.5 if len(ns)>10 else 4,mfc=color if arm=='actual' else 'white')
    fig.suptitle('Original theorem recipe versus simple SVD',fontsize=20,fontweight='bold',y=.98)
    fig.legend(handles=[Line2D([],[],color=c,lw=2.5,label=l) for c,l in zip(COLORS,LABELS)],
               loc='upper center',bbox_to_anchor=(.54,.91),ncol=3,frameon=False,fontsize=11.5)
    fig.legend(handles=[Line2D([],[],color='.2',lw=2.4,ls='-',label='Simple SVD (unscaled)'),
                        Line2D([],[],color='.2',lw=2.4,ls='--',label='Theory: reference scaling + original cutoff')],
               loc='upper center',bbox_to_anchor=(.54,.80),ncol=2,frameon=False,handlelength=3.)
    ax.set_xscale('log',base=2);ax.set_yscale('log')
    major=ns if len(ns)<=8 else [x for x in [32,64,128,256,512,1024,2048] if min(ns)<=x<=max(ns)]
    ax.xaxis.set_major_locator(FixedLocator(major));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
    ax.xaxis.set_minor_locator(NullLocator());ax.set_xlabel(r'Interior intervals $N$');ax.set_ylabel(r'Relative $L^2$ error')
    maximum=max(r[f'{a}_error'] for r in rows for a in ['actual','theory'])
    ax.set_ylim(3e-16,max(1e-9,10**np.ceil(np.log10(maximum*1.2))))
    ax.grid(axis='y',which='major',alpha=.13)
    for side in ('top','right'):ax.spines[side].set_visible(False)
    for side in ('bottom','left'):ax.spines[side].set_linewidth(1.5)
    ax.tick_params(which='major',direction='out',length=5,width=1.25,pad=7)
    fig.text(.54,.025,rf'$\lambda={cfg["lambda"]:g}$; $R=\lceil{cfg["a_H"]:g}\sqrt{{N}}\rceil$; $\delta={cfg["delta"]:g}$; matched neurons and samples; independent evaluation grid.',ha='center',fontsize=10,color='.3')
    for ext in ['png','pdf','svg']:fig.savefig(out/f'comparison.{ext}',dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    rr=sorted([r for r in rows if r['target']=='exp'],key=lambda r:r['N'])
    for key,label in [('cutoff_analytic','Analytic approximation term'),('cutoff_matrix','Matrix-error term'),('cutoff_labels','Label/projection term'),('tau','Final cutoff')]:
        ax.loglog(ns,[r['theory'][key] for r in rr],marker='o',label=label)
    ax.set_xlabel(r'Interior intervals $N$');ax.set_ylabel('Absolute cutoff contribution')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'cutoff_diagnostics.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args();draw(args.output)
