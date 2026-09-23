"""All-point prospective comparison; fifth-mode exceptions remain visible."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def draw(root):
    with (root/'scores.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    models = [('constant_effective', 'Constant exact force', '#65758b'), ('poly5', 'Quintic, evolving state', '#ce7532'), ('anchored5', 'Anchored quintic', '#197d82')]
    targets = ['moment5','mixed_sine','gauss_left','bump_right','step_right','kink_abs']
    fig, axes = plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for mi,(name,label,color) in enumerate(models):
        for r in rows:
            ni = [128,512,1024].index(int(r['nref'])); ti = targets.index(r['target'])
            offset = (mi-1)*.24+(ti-2.5)*.012+(int(r['seed'])-32.5)*.015
            marker = '*' if r['target']=='moment5' else 'o'
            size = 85 if marker=='*' else 20
            axes[0].scatter(ni+offset,100*float(r[name+'_relative_error']),s=size,marker=marker,color=color,alpha=.8,edgecolors='none')
            if int(r['nref'])==1024:
                axes[1].scatter(ti+(mi-1)*.21+(int(r['seed'])-32.5)*.04,100*float(r[name+'_relative_error']),s=size,marker=marker,color=color,alpha=.85,edgecolors='none')
    for ax in axes:
        ax.set_yscale('log'); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
        ax.set_ylabel('Slope displacement prediction error (%)')
    axes[0].set_xticks(range(3),['128','512','1024']); axes[0].set_xlabel('Reference width (physical widths 177, 705, 1409)')
    axes[0].set_title('Prospective seeds 32/33: all 12 cases per width')
    axes[1].set_xticks(range(6),['Fifth mode','Mixed sine','Gaussian','Bump','Step','Absolute kink'],rotation=25,ha='right')
    axes[1].set_title('W = 1409: both seeds for every target')
    legend = [Line2D([],[],marker='o',linestyle='',color=color,label=label) for _,label,color in models]
    legend.append(Line2D([],[],marker='*',linestyle='',color='black',label='Fifth-mode target',markersize=10))
    fig.legend(handles=legend,loc='outside lower center',ncol=4,frameon=False)
    fig.savefig(root/'forecast_errors.png',dpi=180); fig.savefig(root/'forecast_errors.pdf'); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root',type=Path)
    draw(parser.parse_args().root)
