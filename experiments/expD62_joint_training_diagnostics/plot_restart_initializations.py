"""Matched default/smaller initialization restart norms: shared absolute axes, 2 x 3."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import sys,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD56_scale_acquisition.plot import apply_style,style_axis,COLORS
from experiments.expD56_scale_acquisition.run import case_id
from experiments.expD62_joint_training_diagnostics.plot_restart_norm import BASE,OUT,TARGETS,RATES,WIDTH


def main():
    OUT.mkdir(exist_ok=True);apply_style()
    fig,axes=plt.subplots(2,3,figsize=(11.2,5.5),sharex=True,sharey=True)
    fig.subplots_adjust(left=.095,right=.985,bottom=.18,top=.84,wspace=.16,hspace=.25)
    fig.suptitle(r'GD: Starting Bound $B_0$',fontsize=13,fontweight='bold',y=.98)
    fig.text(.5,.918,'Width 196 · Shared linear axes',ha='center',fontsize=8,color='#555555')
    report={'width':WIDTH,'optimizer':'GD','steps':50000,'shared_axes':True,
            'definition':'Euclidean norm of slopes, hidden biases, readout weights at each potential restart; output bias excluded',
            'initialization_small':'All three blocks uniform +/-sqrt(6/(196+1)), output bias zero; same family as the useful D60 bound reconstruction',
            'curves':[]};handles=[];allvalues=[]
    for row,family in enumerate(['default','smaller']):
        initial=None
        for col,(name,title) in enumerate(TARGETS):
            ax=axes[row,col];style_axis(ax)
            if row==0:ax.set_title(title,pad=9,fontsize=10)
            if col==0:ax.set_ylabel(('Default Initialization' if row==0 else 'Smaller Initialization')+'\n'+r'$B_0$',fontsize=8,labelpad=8)
            for lr in RATES:
                default=BASE/'width196'/'runs'/case_id(name,1.,lr)
                folder=default if row==0 else BASE/'matched_small_initialization'/f'lr{lr:.0e}'/f'glorot__{name}'
                status=json.loads((folder/'status.json').read_text());assert status['state']=='complete' and status['step']==50000
                path=folder/'checkpoint.npz'
                with np.load(path) as d:
                    keys=d['keys'].tolist();steps=d['history'][:,keys.index('step')];snap=d['snapshots']
                    values=np.linalg.norm(snap[:,:-1],axis=1)
                    np.testing.assert_allclose(values,np.sqrt(d['history'][:,keys.index('parameter_energy')]),rtol=1e-14)
                    if initial is None:initial=snap[0].copy()
                    else:np.testing.assert_array_equal(initial,snap[0])
                if row==1:
                    with np.load(folder/'initial.npz') as small,np.load(default/'initial.npz') as big:
                        np.testing.assert_array_equal(small['x'],big['train_x'])
                        np.testing.assert_array_equal(small['y'],big['train_y'])
                assert steps[0]==0 and steps[-1]==50000;allvalues.extend(values)
                line,=ax.plot(steps,values,color=COLORS[lr],lw=1.25,ls='-' if lr==.001 else (0,(4,2)))
                if row==0 and col==0:handles.append(line)
                report['curves'].append({'initialization':family,'function':name,'learning_rate':lr,
                                        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                        'step':steps.tolist(),'B0':values.tolist()})
            if col==0:
                ax.annotate(f'Initial: {np.linalg.norm(initial[:-1]):.3f}',xy=(0,np.linalg.norm(initial[:-1])),
                            xytext=(5,9),textcoords='offset points',fontsize=7,color='#555555')
            ax.set_xlim(0,50000);ax.set_xticks([0,10000,20000,30000,40000,50000])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda n,_:'0' if n==0 else f'{n/1000:g}k'))
            if row==1:ax.set_xlabel('Restart step',labelpad=6)
    axes[0,0].set_ylim(0,12.6);axes[0,0].set_yticks([0,3,6,9,12])
    assert max(allvalues)<12.6
    fig.legend(handles,[f'LR = {lr:g}' for lr in RATES],ncol=2,frameon=False,loc='lower center',
               bbox_to_anchor=(.5,.065),handlelength=3,columnspacing=3,fontsize=8)
    fig.text(.5,.022,r'$B_0(t)=\sqrt{\sum_j[\gamma_j(t)^2+\beta_j(t)^2+w_j(t)^2]}$',ha='center',va='bottom',fontsize=9,color='#444444')
    fig.savefig(OUT/'gd_restart_norm_initializations.png',dpi=240)
    fig.savefig(OUT/'gd_restart_norm_initializations.svg');plt.close(fig)
    (OUT/'gd_restart_norm_initializations.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in report['curves']:
        print(row['initialization'],row['function'],row['learning_rate'],row['B0'][0],row['B0'][-1])
    print(OUT/'gd_restart_norm_initializations.png')
if __name__=='__main__':main()
