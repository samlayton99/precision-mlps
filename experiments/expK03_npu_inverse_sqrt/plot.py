"""Overlay matched-halo NPU baselines on the saved QUILL width sweep."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/precisionmlps-mpl')
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/checkpoint_K_practical_recipes/expK03_npu_inverse_sqrt'
QI = ROOT/'results/checkpoint_K_practical_recipes/expK02_inverse_sqrt_extension'


def main():
    qi_source = QI/'scaling_1k_5k.json'
    npu_source = OUT/'summary.json'
    adam_source = OUT/'adam_mlp/summary.json'
    rows = json.loads(qi_source.read_text())
    baselines = [r for r in json.loads(npu_source.read_text()) if r['training_mode']=='matched_halo']
    assert {r['model'] for r in baselines} == {'NPU','RealNPU'}
    adam = json.loads(adam_source.read_text())
    assert len(adam)==1 and adam[0]['model']=='Adam MLP' and adam[0]['training_mode']=='matched_halo'
    baselines += adam
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':7,
                         'mathtext.rm':'DejaVu Sans:bold','mathtext.it':'DejaVu Sans:italic:bold',
                         'mathtext.bf':'DejaVu Sans:bold','mathtext.cal':'DejaVu Sans'})
    fig,ax=plt.subplots(figsize=(4.8,4.0))
    fig.subplots_adjust(left=.15,right=.97,bottom=.31,top=.81)
    fig.suptitle('QUILL and trained networks',x=.56,y=.95,fontsize=12,fontweight='bold')
    color=plt.colormaps['viridis'](.55)
    indices=[i for i,r in enumerate(rows) if 1000<=r['width']<=5000]
    ax.plot([r['width'] for r in rows],[r['relative_l2'] for r in rows],
            color=color,lw=1.3,marker='o',markersize=3.8,markerfacecolor='white',
            markeredgecolor=color,markeredgewidth=.9,markevery=indices,label='QUILL',zorder=3)
    displayed=[]
    for row,position,style in zip(baselines,[.08,.30,.83],['--',':','-.']):
        error=row.get('best_trained_validation_test_rel_l2',row['best_validation_test_rel_l2'])
        selected_step=row.get('best_trained_validation_step',row['best_validation_step'])
        assert selected_step>0
        ax.axhline(error,color=plt.colormaps['viridis'](position),ls=style,lw=1.35,
                   label=f'{row["model"]} ({row["hidden_units"]:,} units)\n{error:.2e}',zorder=2)
        displayed.append({'model':row['model'],'relative_l2':error,
                          'checkpoint_step':selected_step,
                          'selection':'minimum logged order-32 validation error after at least one update',
                          'hidden_units':row['hidden_units'],'parameters':row['parameters'],
                          'training_mode':row['training_mode']})
    ax.set(xscale='log',yscale='log',xlim=(1000,5000),ylim=(1e-15,1e4),
           xlabel='QUILL hidden width (neurons)',ylabel=r'Relative $L^2$ error')
    ax.set_xticks([1000,2000,4000],['1k','2k','4k'])
    ax.xaxis.set_minor_locator(FixedLocator([1500,3000,5000]));ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([1e3,1,1e-3,1e-6,1e-9,1e-12,1e-15])
    ax.yaxis.set_minor_locator(LogLocator(base=10,subs=(2.,5.),numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(r'$f(x)=1/\sqrt{x},\quad x\in[10^{-4},1]$',fontsize=9.2,
                 fontweight='bold',math_fontfamily='custom',pad=8)
    last=max((r for r in rows if r['width']<=5000),key=lambda r:r['width'])
    ax.annotate(r'$\approx5\times10^{-14}$',xy=(last['width'],last['relative_l2']),
                xytext=(0,5),textcoords='offset points',ha='right',va='bottom',fontsize=7,color='#444444')
    for side in ['top','right']:ax.spines[side].set_visible(False)
    for side in ['left','bottom']:ax.spines[side].set_linewidth(1.)
    ax.tick_params(which='major',labelsize=7,length=3,width=.85,pad=2)
    ax.tick_params(which='minor',length=1.3,width=.35)
    ax.xaxis.label.set_fontsize(8);ax.yaxis.label.set_fontsize(8)
    ax.xaxis.labelpad=2;ax.yaxis.labelpad=1.5
    ax.grid(axis='y',which='major',color='#D8D8D8',lw=.45);ax.set_axisbelow(True)
    handles,labels=ax.get_legend_handles_labels()
    order=[0,1,3,2]
    fig.legend([handles[i] for i in order],[labels[i] for i in order],
               loc='lower center',bbox_to_anchor=(.56,.065),
               ncol=2,frameon=False,fontsize=7,handlelength=1.8,handletextpad=.5,columnspacing=1.5)
    fig.text(.56,.025,'Matched C17 halo; trained baselines use best validation checkpoints',
             ha='center',fontsize=6.3,color='#555555')
    for extension in ['png','pdf']:
        fig.savefig(OUT/f'inverse_sqrt_npu_comparison.{extension}',dpi=600,bbox_inches='tight',pad_inches=.13)
    plt.close(fig)
    provenance={'qi_source':str(qi_source.relative_to(ROOT)),
                'qi_sha256':hashlib.sha256(qi_source.read_bytes()).hexdigest(),
                'npu_source':str(npu_source.relative_to(ROOT)),
                'npu_sha256':hashlib.sha256(npu_source.read_bytes()).hexdigest(),
                'adam_source':str(adam_source.relative_to(ROOT)),
                'adam_sha256':hashlib.sha256(adam_source.read_bytes()).hexdigest(),
                'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'baselines':displayed,'smoothing':None,'x_limits':[1000,5000],
                'x_scale':'log','npu_lines_are_fixed_six_unit_references':True,
                'adam_line_is_fixed_4815_unit_reference':True,
                'halo_reference_width':4815,'interval_only_controls_displayed':False}
    (OUT/'figure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(OUT/'inverse_sqrt_npu_comparison.png')


if __name__=='__main__':main()
