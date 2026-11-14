"""Five functions, two panels: sharp theorem recovery versus raw TSVD.

Both stored fits use FP64 DGEJSV. Raw relative cutoff is fixed at 2**-52;
the scaled arm uses A.65 with the sharpened bound for the same witness.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from threadpoolctl import threadpool_limits
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expC14_theory_solver_comparison import run as r
from experiments.expC14_theory_solver_comparison.affine_grid import evaluate
from experiments.expC14_theory_solver_comparison.plot import COLORS

TARGETS=['exp','sine4','sine24','chirp','packet']
LABELS=[r'$e^x$',r'$\sin(4\pi x)$',r'$\sin(24\pi x)$','Chirp','Wave packet']
PANEL_COLORS=[COLORS[r.NAMES.index(name)] for name in TARGETS]
OUT=r.OUT/'five_function_error_norm_panel'


def draw(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter,LogLocator,NullFormatter,NullLocator
    # Match the 4320x1344 Figure 2 embedded in submission (18), rendered by
    # expC09_bandwidth_figures/combined.py, true_precision_data branch with
    # runge_rows=None and paper_style=False. Use a compact two-panel canvas,
    # with the paper's exact axis/tick/legend font sizes and a compact layout.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.labelsize':20,'axes.titlesize':23,
                         'xtick.labelsize':13,'ytick.labelsize':13,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.5),sharex=True)
    fig.subplots_adjust(left=.09,right=.99,bottom=.17,top=.77,wspace=.24)
    for name,color in zip(TARGETS,PANEL_COLORS):
        rr=sorted([x for x in rows if x['target']==name],key=lambda x:x['W'])
        for arm,ls,marker in [('theory','--','o'),('raw','-','o')]:
            for ax,metric in zip(axes,['error','l1']):
                ax.plot([x['W'] for x in rr],[x[f'{arm}_{metric}'] for x in rr],
                    color=color,ls=ls,marker=marker,lw=2.8 if arm=='theory' else 3.2,
                    ms=5.,mew=1.5,mfc='white',mec=color,markevery=2,zorder=3)
    for ax in axes:
        ax.set_xscale('linear')
        ax.set_yscale('log')
        ax.set_xlim(64,864)
        ax.set_xticks([128,256,384,512,640,768])
        ax.set_xlabel(r'Total width $W$',fontsize=19)
        ax.grid(False,axis='x',which='both')
        ax.grid(axis='y',which='major',alpha=.13,lw=.65)
        ax.set_axisbelow(True)
        for side in ['top','right']:ax.spines[side].set_visible(False)
        for side in ['bottom','left']:ax.spines[side].set_linewidth(2.6)
        ax.tick_params(which='major',direction='out',length=6,width=1.6,pad=7,
                       labelsize=13,top=False,right=False)
        ax.tick_params(which='minor',direction='out',length=2.8,width=.8,top=False,right=False)
        ax.tick_params(axis='y',which='minor',length=2.2,width=.65)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1.,),numticks=100))
        ax.yaxis.set_minor_locator(LogLocator(base=10,subs=range(2,10),numticks=200))
        ax.yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel(r'$L^2$ error')
    axes[0].set_ylim(3e-17,10)
    weight_label=axes[1].set_ylabel(r'Weights $\ell^1$')
    axes[1].set_ylim(.7,1e11)
    for ax,every in zip(axes,[4,2]):
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value,_,every=every:
            rf'$10^{{{int(round(np.log10(value)))}}}$' if int(round(np.log10(value)))%every==0 else ''))
        for tick,value in zip(ax.yaxis.get_major_ticks(),ax.yaxis.get_majorticklocs()):
            tick.gridline.set_visible(int(round(np.log10(value)))%2==0)
    fig.suptitle('Practical Implementation vs Theorem',fontsize=26,fontweight='bold',y=1.01)
    axes[0].set_title('a) Relative Error',loc='center',y=1.02,fontweight='normal')
    axes[1].set_title('b) Solved Magnitude',loc='center',y=1.02,fontweight='normal')
    handles=[Line2D([],[],color=c,lw=3.2,label=l) for c,l in zip(PANEL_COLORS,LABELS)]
    axes[0].legend(handles=handles,loc='upper right',framealpha=.95,fontsize=13,
                   handlelength=1.7,labelspacing=.30,borderpad=.35,handletextpad=.5)
    fig.legend(handles=[Line2D([],[],color='.2',lw=3.2,ls='-',label='Practical (Raw SVD)'),
                        Line2D([],[],color='.2',lw=2.8,ls='--',label='Theorem Implementation')],
               loc='center',bbox_to_anchor=(.54,.885),ncol=2,frameon=False,
               fontsize=13,handlelength=2.8,columnspacing=2.2)
    # Keep the explanatory sum smaller than the main axis label.
    sum_label=fig.text(0,0,r'$\sum_i |w_i|$',fontsize=13,rotation=90,
                       ha='center',va='center')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    main_box=weight_label.get_window_extent(renderer)
    sum_box=sum_label.get_window_extent(renderer)
    gap=12*fig.dpi/72
    center_y=axes[1].transAxes.transform((0,.5))[1]
    weight_label.set_y(axes[1].transAxes.inverted().transform(
        (0,center_y-(sum_box.height+gap)/2))[1])
    sum_label.set_position(fig.transFigure.inverted().transform(
        ((main_box.x0+main_box.x1)/2,center_y+(main_box.height+gap)/2)))
    for ext in ['png','pdf','svg']:fig.savefig(out/f'comparison.{ext}',dpi=240,bbox_inches='tight',pad_inches=.18)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if args.plot_only:
        draw(OUT,json.loads((OUT/'metrics.json').read_text()));return
    sharp=r.OUT/'option_c_sharp_bound_64_768'
    raw=r.OUT/'readout_l1_three_methods_64_768'
    cfg=yaml.safe_load((sharp/'config.yaml').read_text())
    raw_cfg=yaml.safe_load((raw/'config.yaml').read_text())
    assert cfg['theory_driver']==raw_cfg['driver_all_arms']=='gejsv'
    assert raw_cfg['fixed_relative_cutoff']==2**-52
    for k in ['N','lambda','a_H','delta','Delta','sampling_density','feature_parameterization']:
        assert cfg[k]==raw_cfg['geometry'][k],k
    OUT.mkdir(exist_ok=False)
    (OUT/'config.yaml').write_text(yaml.safe_dump(dict(targets=TARGETS,geometry=cfg,
        driver_both_arms='gejsv',raw_relative_cutoff=2**-52,
        theoretical_cutoff='A.65 with sharper existing same-witness approximation bound',
        input_sources=dict(theory=str(sharp),raw=str(raw)),x_axis='Actual total hidden width W=N+2R+1, linear numeric positions.',
        norms_include_bias=True),sort_keys=False))
    snap=OUT/'source_snapshot';snap.mkdir();hashes={}
    for f in Path(__file__).parent.glob('*.py'):
        shutil.copy2(f,snap/f.name);hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    inputs={};rows=[]
    idx=[r.NAMES.index(name) for name in TARGETS];count=len(TARGETS)
    xe=-1+2*(np.arange(cfg['eval_points'])+.5)/cfg['eval_points']
    xv=-1+2*(np.arange(cfg['validation_points'])+.5)/cfg['validation_points']
    ye=r.target_values(xe)[:,idx];yv=r.target_values(xv)[:,idx]
    with threadpool_limits(limits=cfg['threads']):
        for N in cfg['N']:
            tf=sharp/f'coefficients_N{N}.npz';rf=raw/f'coefficients_N{N}.npz';jf=sharp/f'N{N}.json'
            for f in [tf,rf,jf]:inputs[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
            with np.load(tf) as t,np.load(rf) as s:
                assert np.array_equal(t['D'],s['D'])
                theory=t['theory'][:,idx].copy();direct=s['unscaled_fixed'][:,idx].copy()
            old={x['target']:x for x in json.loads(jf.read_text())}
            R=old['exp']['R'];W=old['exp']['W']
            coeff=np.column_stack((theory,direct))
            errors,_=evaluate(N,R,coeff,np.tile(ye,(1,2)))
            valid,_=evaluate(N,R,coeff,np.tile(yv,(1,2)))
            for j,name in enumerate(TARGETS):
                previous=old[name]
                assert np.isclose(errors[j],previous['theory_error'],rtol=1e-12,atol=0)
                assert all(previous['theory'][k] for k in ['cutoff_pass','transformed_norm_pass','sample_bound_pass','coefficient_bound_pass'])
                rows.append(dict(N=N,W=W,target=name,theory_error=float(errors[j]),raw_error=float(errors[j+count]),
                    theory_validation_error=float(valid[j]),raw_validation_error=float(valid[j+count]),
                    theory_l1=float(np.sum(abs(theory[:,j]))),raw_l1=float(np.sum(abs(direct[:,j]))),
                    theory_absolute_cutoff=previous['theory']['tau'],raw_relative_cutoff=2**-52))
            print(f'N={N}, W={W}: evaluated {count} functions',flush=True)
    (OUT/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (OUT/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'provenance.json').write_text(json.dumps(dict(sources=hashes,inputs=inputs,
        fits_reused_without_modification=True,errors_recomputed_on_independent_grids=True),indent=2)+'\n')
    summary=dict(pairs=len(rows),max_error_N_ge_256={a:max(x[f'{a}_error'] for x in rows if x['N']>=256) for a in ['theory','raw']},
        max_validation_relative_change=max(abs(x[f'{a}_validation_error']/x[f'{a}_error']-1) for x in rows for a in ['theory','raw']))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    draw(OUT,rows);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
