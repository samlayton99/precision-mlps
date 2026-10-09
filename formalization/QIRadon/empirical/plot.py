#!/usr/bin/env python3
"""One function family across dimensions; no rate regression or fitted floor."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/'config.json').read_text())
TARGET = CONFIG['main_target']
COLORS = ['#2166ac', '#b35806', '#1b7837', '#762a83']
MARKERS = ['o', 's', '^', 'D']
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'axes.edgecolor': '#5c626b',
    'xtick.color': '#42474f', 'ytick.color': '#42474f',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'savefig.facecolor': 'white', 'pdf.fonttype': 42,
})


def read(name):
    return json.loads((ROOT/'results'/f'{name}.json').read_text())


def save(fig, name):
    fig.savefig(ROOT/'figures'/f'{name}.png', dpi=240, bbox_inches='tight', pad_inches=.04)
    fig.savefig(ROOT/'figures'/f'{name}.pdf', bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def finish_axis(ax, error_ticks=None):
    ax.grid(axis='y', color='#e3e6ea', linewidth=.6)
    ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    if error_ticks is not None:
        ax.yaxis.set_major_locator(FixedLocator(error_ticks))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())


def main():
    width = [r for r in read('width') if r['target']==TARGET]
    precision = read('precision')
    assert all(r.get('target')==TARGET for r in precision), 'Rerun precision for the chosen family.'
    algebraic, kernel, allocation = map(read, ['algebraic', 'algebraic_kernel', 'allocation'])
    allocation = [r for r in allocation if r['target']==TARGET]
    # Physical dimensions match the printed figure, so labels remain readable.
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for d in range(1,5):
        color, marker = COLORS[d-1], MARKERS[d-1]
        rows = [r for r in width if r['d']==d]
        axes[0].loglog([r['width'] for r in rows], [r['relative_l2'] for r in rows],
                       '-'+marker, color=color, ms=3, lw=1.15,
                       markerfacecolor='white', markeredgewidth=.8, label=f'$d={d}$')
        axes[0].loglog([r['width'] for r in rows], [r['relative_bound'] for r in rows],
                       '--', color=color, lw=.95, alpha=.8)
        rows = [r for r in precision if r['d']==d]
        axes[1].semilogy([r['bits'] for r in rows], [r['relative_l2'] for r in rows],
                         '-'+marker, color=color, ms=3, lw=1.15,
                         markerfacecolor='white', markeredgewidth=.8)
        axes[1].semilogy([r['bits'] for r in rows], [r['perturbation_bound'] for r in rows],
                         '--', color=color, lw=.95, alpha=.8)
    axes[0].set(xlabel='Neurons $W$', ylabel='Relative error',
                ylim=(1e-16,1), xlim=(40,5e6), xticks=[1e2,1e4,1e6])
    axes[1].set(xlabel='Readout bits', ylim=(1e-16,.1), xticks=[8,24,40,53])
    axes[0].set_title('(a) Width', loc='left', pad=6)
    axes[1].set_title('(b) Precision', loc='left', pad=6)
    for ax in axes:
        finish_axis(ax, [1,1e-4,1e-8,1e-12,1e-16])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(.55,1.025), handlelength=1.6, columnspacing=1.5)
    fig.subplots_adjust(left=.10, right=.99, bottom=.19, wspace=.30, top=.81)
    save(fig,'construction_rates')

    # Separate robustness question: function variation at fixed construction
    # rules. No precision axis, and no reuse as a dimension-scaling claim.
    all_width = read('width')
    targets = ['broad','narrow','shifted','two_peaks','signed_mixture']
    labels = ['Broad','Narrow','Shifted','Two peaks','Signed mixture']
    colors = COLORS+['#c51b7d']
    fig, axes = plt.subplots(2,2,figsize=(6.5,3.5),sharey=True)
    for d,ax in enumerate(axes.flat,1):
        for name,label,color in zip(targets,labels,colors):
            rows=[r for r in all_width if r['target']==name and r['d']==d]
            ax.loglog([r['width'] for r in rows],[r['relative_l2'] for r in rows],
                      '-o',color=color,ms=2,lw=1,label=label)
        ax.set_title(f'$d={d}$',loc='left',pad=4)
        ax.set_ylim(1e-16,1)
        finish_axis(ax,[1,1e-8,1e-16])
        if d==1:
            ax.set_xticks([50,100,200]); ax.xaxis.set_major_formatter(ScalarFormatter())
        else:
            ax.set_xticks({2:[1e2,1e3],3:[1e3,1e4,1e5],4:[1e3,1e4,1e5,1e6]}[d])
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,loc='upper center',bbox_to_anchor=(.54,1.02),
               frameon=False,fontsize=7.5,handlelength=1.4,columnspacing=1.2)
    fig.supxlabel('Neurons $W$',fontsize=8.5,y=.005)
    fig.supylabel('Relative error',fontsize=8.5,x=.005)
    fig.subplots_adjust(left=.12,right=.99,bottom=.13,top=.84,hspace=.49,wspace=.18)
    save(fig,'robustness')

    fig, axes=plt.subplots(1,3,figsize=(6.5,2.25))
    w=[r['neurons'] for r in algebraic]
    axes[0].loglog(w,[r['linf'] for r in algebraic],'o',ms=3,color=COLORS[0],label='Measured')
    axes[0].loglog(w,[r['upper_bound'] for r in algebraic],'--',lw=1,color=COLORS[0],label='Upper bound')
    axes[0].loglog(w,[r['arbitrary_relu_lower_bound'] for r in algebraic],':',lw=1,color='#777777',label='Lower bound')
    axes[0].set(xlabel='Neurons',ylabel='Uniform error',xticks=[10,100,1000])
    axes[0].set_title('(a) ReLU',loc='left')
    w=[r['neurons'] for r in kernel]
    axes[1].loglog(w,[r['sampled_linf'] for r in kernel],'-o',ms=3,lw=1,color=COLORS[2],label='Measured')
    axes[1].loglog(w,[r['upper_bound'] for r in kernel],'--',lw=1,color=COLORS[2],label='Bound')
    axes[1].set(xlabel='Neurons',ylabel='Sampled max. error',xticks=[100,1000])
    axes[1].set_title('(b) Finite smoothness',loc='left')
    x=[r['tolerance'] for r in allocation]
    axes[2].loglog(x,[r['width'] for r in allocation],'-o',ms=3,lw=1,color=COLORS[1],label='Bound-based')
    axes[2].loglog(x,[r['best_sampled_grid']['width'] for r in allocation],'-s',ms=3,lw=1,color=COLORS[0],label='Best tested')
    axes[2].set(xlabel='Tolerance',ylabel='Neurons',xticks=[1e-3,1e-6,1e-9],yticks=[300,1000,3000])
    axes[2].invert_xaxis();axes[2].yaxis.set_major_formatter(ScalarFormatter())
    axes[2].set_title('(c) Allocation',loc='left')
    for i,ax in enumerate(axes):
        finish_axis(ax)
        ax.legend(fontsize=6.8,loc='upper right' if i<2 else 'upper left',frameon=False,
                  handlelength=1.3,handletextpad=.5,borderpad=.15,labelspacing=.25)
    axes[0].yaxis.set_major_locator(FixedLocator([1e-1,1e-3,1e-5]))
    axes[1].yaxis.set_major_locator(FixedLocator([1,1e-3,1e-6]))
    fig.subplots_adjust(left=.085,right=.99,bottom=.24,wspace=.55,top=.88)
    save(fig,'additional_tests')

    lines=['# Results for the displayed experiments\n',
           'The scaling plot uses only f_d(x)=exp(-3*sum(x_k^2)), d=1..4. A separate robustness figure displays all five Gaussian-based families using the same construction schedule, with no precision panel.\n',
           '| d | Best relative L2 | Width at best | Last relative L2 |\n|---|---:|---:|---:|']
    for d in range(1,5):
        rows=[r for r in width if r['d']==d];best=min(rows,key=lambda r:r['relative_l2'])
        lines.append(f'| {d} | {best["relative_l2"]:.3g} | {best["width"]:,} | {rows[-1]["relative_l2"]:.3g} |')
    lines+=['\n## Allocation\n','| Tolerance | Predicted width | Best tested width | Ratio | Max sampled error |\n|---|---:|---:|---:|---:|']
    for r in allocation:
        lines.append(f'| {r["tolerance"]:.0e} | {r["width"]:,} | {r["best_sampled_grid"]["width"]:,} | {r["width_ratio_to_sampled_best"]:.2f} | {r["sampled_linf"]:.3g} |')
    lines+=['\nThe allocation meets every tolerance but is conservative. It is not an empirical optimality result.\n',
            'The nonanalytic Gaussian-kernel experiment uses p_hat(xi)=(1+xi^2)^(-2), s=2, and A_s=pi+2. Max sampled error decreases from 0.00968 (43 neurons) to 3.97e-8 (1611 neurons). The sufficient bound is algebraic and conservative. Doubled spectral quadrature changes the largest run by about 2.0e-10, so the last digits are not claimed accurate.\n',
            'The precision experiment rounds final readouts and bias only; geometry and evaluation remain float64. Dashed curves include the exact construction bound and the readout-rounding bound, with no measured error floor added. Unaccounted float64 errors limit measured target accuracy.\n',
            'The full multidimensional nonanalytic theorem is not tested end to end. Width bounds do not assert efficient profile acquisition or a PDE solver.\n']
    (ROOT/'results/summary.md').write_text('\n'.join(lines))
    print('Rendered the single-family figures and result summary.')


if __name__=='__main__':
    main()
