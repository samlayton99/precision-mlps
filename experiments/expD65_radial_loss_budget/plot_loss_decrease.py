"""Function-wise loss-decrease ratios for each initialization and learning rate."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD65_radial_loss_budget.check_loss_decrease import OUT
from experiments.expD56_scale_acquisition.plot import apply_style,style_axis,COLORS
from experiments.expD65_radial_loss_budget.plot_distributions import summary,draw

def main():
    report=json.loads((OUT/'results.json').read_text());apply_style()
    fig,axes=plt.subplots(1,3,figsize=(11.2,3.8),sharex=True,sharey=True)
    fig.subplots_adjust(left=.075,right=.975,top=.76,bottom=.29,wspace=.16)
    fig.suptitle('GD Loss-Decrease Condition: 24 Functions',fontsize=13,fontweight='bold',y=.97)
    fig.text(.5,.864,r'$c_n=(L_n-L_{n+1})/(\eta\|g_n\|^2)$',ha='center',fontsize=10,color='#444444')
    groups=[];stats={}
    with np.load(OUT/'checks.npz') as data:
        for ax,mult,label in zip(axes,[1.,2.,4.],['a) Default Initialization','b) 2x Initialization','c) 4x Initialization']):
            style_axis(ax);ax.set_title(label,fontsize=9.5,pad=8)
            for lr in [.001,.0001]:
                cases=[r for r in report['cases'] if r['initialization']==mult and r['lr']==lr];assert len(cases)==24
                arr=np.stack([data[r['case']] for r in cases]);t=arr[0,:,0]
                for a in arr:np.testing.assert_array_equal(a[:,0],t)
                q=summary(arr[:,:,4]);draw(ax,t,q,COLORS[lr])
                row={'initialization':mult,'lr':lr,'checks':int(arr.shape[0]*arr.shape[1]),'min_c':float(q[0].min()),
                     'max_c':float(q[4].max()),'min_c_after_30k':float(q[0,t>=30000].min()),'failures':int(np.sum(arr[:,:,4]<.5))}
                groups.append(row);stats[f'init{mult:g}_lr{lr:g}']={'steps':t.tolist(),'quantiles':q.tolist()}
            ax.axhline(.5,color='#555555',lw=1,ls=(0,(5,3)))
            ax.text(195000,.515,r'Required: $c_n\geq0.5$',ha='right',va='bottom',fontsize=7,color='#555555')
            minimum=min(r['min_c'] for r in groups if r['initialization']==mult)
            ax.text(.04,.85,f'Minimum: {minimum:.5f}',transform=ax.transAxes,fontsize=7.5,color='#555555')
            ax.set_xlim(0,200000);ax.set_xticks([0,50000,100000,150000,200000]);ax.set_xlabel('GD step')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda n,_:'0' if n==0 else f'{n/1000:g}k'))
    axes[0].set_ylim(.45,max(1.02,report['global_maximum_c']+.015));axes[0].set_yticks([.5,.6,.7,.8,.9,1.]);axes[0].set_ylabel(r'Loss-decrease ratio $c_n$')
    fig.legend([Line2D([],[],color=COLORS[lr],lw=1.4) for lr in [.001,.0001]],['LR = 0.001','LR = 0.0001'],
               loc='lower center',bbox_to_anchor=(.5,.13),ncol=2,frameon=False,fontsize=8,handlelength=3,columnspacing=3)
    fig.legend([Line2D([],[],color='#333333',lw=1.2),Patch(facecolor='#888888',alpha=.25,edgecolor='none'),Line2D([],[],color='#333333',lw=.75,ls=(0,(1.8,2.2)))],
               ['Median','25th–75th percentiles','Minimum / maximum'],loc='lower center',bbox_to_anchor=(.5,.067),ncol=3,frameon=False,fontsize=7.5,handlelength=2.8,columnspacing=2)
    fig.text(.5,.02,'One GD update per saved checkpoint. Summaries across functions, not confidence intervals.',ha='center',fontsize=7,color='#666666')
    for ext in ['png','svg']:fig.savefig(OUT/f'loss_decrease_condition.{ext}',dpi=240)
    plt.close(fig)
    (OUT/'plot_statistics.json').write_text(json.dumps({'probabilities':[0,.25,.5,.75,1],'quantile_method':'linear','groups':groups,'curves':stats},indent=2)+'\n')
    lines=['# Direct GD loss-decrease check','',
           f"Checked {report['checks']:,} adjacent-update probes from 144 saved runs: 24 functions, three initializations, two learning rates, width 196. Each probe uses the saved state at n and computes exactly one native GD update to n+1. No run was continued and no checkpoint was overwritten.",'',
           '| Initialization | LR | Checks | Minimum c | Maximum c | Minimum after 30k | Violations of c >= 0.5 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in groups:lines.append(f"| {r['initialization']:g}x | {r['lr']:g} | {r['checks']:,} | {r['min_c']:.8f} | {r['max_c']:.8f} | {r['min_c_after_30k']:.8f} | {r['failures']} |")
    lines+=['',
            'The numerator is evaluated as -mean(e*delta_q)-mean(delta_q²)/2. The tanh addition identity evaluates delta_q without subtracting nearly equal predictions. Naive subtraction of the two rounded losses is retained only as a numerical diagnostic. Gradients include slopes, hidden biases, readout weights, and output bias; loss is half MSE.',
            '',f"The first two updates of every run match the original training kernel and saved next states ({report['matched_saved_early_updates']} checks; largest saved-state parameter discrepancy {report['max_saved_update_difference']:.3g}).",
            '', 'Regular checkpoints are 500 steps apart; this checks only the single next step at each saved checkpoint, not every intervening update. The last probe is 199500 -> 199501, within the original 200k horizon. Losses separated by 500 updates are never substituted for adjacent losses. All ratios are floating-point evaluations, not interval certificates.']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(groups,indent=2));print(OUT/'loss_decrease_condition.png')
if __name__=='__main__':main()
