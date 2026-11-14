"""Actual NPU capacity curves, sharing a total-parameter axis with QUILL."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import hashlib
import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,NullFormatter,FixedLocator

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/checkpoint_K_practical_recipes/expK03_npu_inverse_sqrt'
OUT=BASE/'width_sweep'
QI=ROOT/'results/checkpoint_K_practical_recipes/expK02_inverse_sqrt_extension'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--matched-widths',action='store_true')
    args=parser.parse_args();matched=args.matched_widths
    output=BASE/'matched_construction_widths' if matched else OUT
    adam_source=BASE/'adam_standard/selected_reference.json' if matched else BASE/'adam_mlp/summary.json'
    paths=[output/'summary.json',QI/'scaling_1k_5k.json',adam_source]
    rows,qi,adam=[json.loads(p.read_text()) for p in paths]
    if isinstance(adam,dict):adam=[adam]
    spec=json.loads((output/'sweep_specification.json').read_text())
    expected=({name:[r['width'] for r in spec['construction_points']] for name in spec['models']}
              if matched else spec['widths'])
    for name,widths in expected.items():
        assert sorted(r['hidden_units'] for r in rows if r['model']==name)==widths
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':7,
                         'mathtext.rm':'DejaVu Sans:bold','mathtext.it':'DejaVu Sans:italic:bold',
                         'mathtext.bf':'DejaVu Sans:bold','mathtext.cal':'DejaVu Sans'})
    fig,ax=plt.subplots(figsize=(5.6,3.8))
    fig.subplots_adjust(left=.13,right=.97,bottom=.23,top=.81)
    fig.suptitle('QUILL and trained networks' if matched else 'QUILL and NPU capacity scaling',
                 x=.55,y=.95,fontsize=12,fontweight='bold')
    cmap=plt.colormaps['viridis'];displayed={}
    for name,position in [('NPU',.08),('RealNPU',.30)]:
        selected=sorted([r for r in rows if r['model']==name],key=lambda r:r['parameters'])
        x=[r['hidden_units'] if matched else r['parameters'] for r in selected]
        y=[r['best_trained_validation_test_rel_l2'] for r in selected]
        ax.plot(x,y,color=cmap(position),lw=1.3,marker='o',markersize=3.8,
                markerfacecolor='white',markeredgewidth=.9,label=name,zorder=3)
        displayed[name]=[{'x':a,'parameters':r['parameters'],'relative_l2':b,'width':r['hidden_units'],
                          'checkpoint_step':r['best_trained_validation_step']}
                         for a,b,r in zip(x,y,selected)]
    ax.plot([r['width'] if matched else 3*r['width']+1 for r in qi],[r['relative_l2'] for r in qi],
            color=cmap(.55),lw=1.3,marker='o',markersize=3.8,markerfacecolor='white',
            markeredgewidth=.9,label='QUILL',zorder=4)
    a=adam[0]
    if matched:
        assert a['steps']==100000 and a['best_trained_validation_step']>=50000
        assert a['hidden_units']>=64
        assert a.get('initializer') in ('pytorch','xavier_uniform_tanh')
        assert a.get('output_scale',1)==1
        ax.axhline(a['best_trained_validation_test_rel_l2'],color=cmap(.83),lw=1.35,
                   linestyle=':',label='Adam baseline',zorder=2)
    else:
        ax.plot([a['parameters']],[a['best_trained_validation_test_rel_l2']],color=cmap(.83),
                marker='D',markersize=4.5,markerfacecolor='white',markeredgewidth=1.1,
                linestyle='none',label='Adam MLP',zorder=5)
    displayed['Adam MLP']={'parameters':a['parameters'],'width':a['hidden_units'],
                          'relative_l2':a['best_trained_validation_test_rel_l2'],
                          'checkpoint_step':a['best_trained_validation_step'],
                          'rendering':'dotted horizontal reference' if matched else 'single point'}
    adam_directory=ROOT/a['run_directory'] if matched else BASE/'adam_mlp'
    adam_config=json.loads((adam_directory/'config.json').read_text())
    displayed['Adam MLP']['halo_reference_N']=adam_config['halo_reference_N']
    displayed['Adam MLP']['halo_reference_R']=adam_config['halo_reference_R']
    displayed['Adam MLP']['halo_reference_width']=adam_config['halo_reference_width']
    displayed['Adam MLP']['training_steps']=a['steps']
    displayed['Adam MLP']['learning_rate']=a['learning_rate']
    displayed['Adam MLP']['minimum_selection_step']=a.get('minimum_selection_step',1)
    displayed['Adam MLP']['initializer']=a.get('initializer','pytorch')
    displayed['Adam MLP']['batch_size']=a.get('batch_size',100)
    displayed['Adam MLP']['final_learning_rate']=a.get('final_learning_rate',a['learning_rate'])
    ax.set(xscale='log',yscale='log',xlim=(1000,5000) if matched else (10,18000),ylim=(1e-15,1e4),
           xlabel='Hidden width' if matched else 'Total trainable parameters',ylabel=r'Relative $L^2$ error')
    # QUILL's parameters are constructed but all are ordinary trainable MLP parameters.
    if matched:
        ax.set_xticks([1000,2000,4000],['1k','2k','4k'])
        ax.xaxis.set_minor_locator(FixedLocator([1500,3000,5000]))
    else:
        ax.set_xticks([10,100,1000,10000],['10','100','1k','10k'])
        ax.xaxis.set_minor_locator(LogLocator(base=10,subs=(2.,5.)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([1e3,1,1e-3,1e-6,1e-9,1e-12,1e-15])
    ax.yaxis.set_minor_locator(LogLocator(base=10,subs=(2.,5.),numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(r'$f(x)=1/\sqrt{x},\quad x\in[0.0001,1]$',fontsize=9.2,
                 fontweight='bold',math_fontfamily='custom',pad=8)
    ax.annotate(r'$\approx5\times10^{-14}$',xy=(4815 if matched else 14446,5e-14),xytext=(-3,6),
                textcoords='offset points',ha='right',va='bottom',fontsize=7,color='#444444')
    for side in ['top','right']:ax.spines[side].set_visible(False)
    for side in ['left','bottom']:ax.spines[side].set_linewidth(1.)
    ax.tick_params(which='major',labelsize=7,length=3,width=.85,pad=2)
    ax.tick_params(which='minor',length=1.3,width=.35)
    ax.xaxis.label.set_fontsize(8);ax.yaxis.label.set_fontsize(8)
    ax.xaxis.labelpad=2;ax.yaxis.labelpad=1.5
    ax.grid(axis='y',which='major',color='#D8D8D8',lw=.45);ax.set_axisbelow(True)
    handles,labels=ax.get_legend_handles_labels();order=[2,0,1,3]
    fig.legend([handles[i] for i in order],[labels[i] for i in order],
               loc='lower center',bbox_to_anchor=(.55,.065),ncol=4,frameon=False,fontsize=7,
               handlelength=1.8,handletextpad=.5,columnspacing=1.3)
    if not matched:
        fig.text(.55,.025,'Same C17 target and halo; one seed; best trained validation checkpoints',
                 ha='center',fontsize=6.3,color='#555555')
    stem='matched_width_scaling' if matched else 'npu_capacity_sweep'
    for ext in ['png','pdf']:
        fig.savefig(output/f'{stem}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.13)
    plt.close(fig)
    provenance={'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'displayed':displayed,'x_axis':'hidden width, logarithmic' if matched else 'total parameters, logarithmic','smoothing':None,
                'selection':'best positive-step checkpoint selected by order-32 validation',
                'scored_measure':'physical dx relative L2 on [1e-4,1]',
                'halo_reference_width':None if matched else 4815,'per_width_halo_matching':matched,
                'description_below_legend':not matched,
                'interval_only_controls_displayed':False}
    (output/'figure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(output/f'{stem}.png')


if __name__=='__main__':main()
