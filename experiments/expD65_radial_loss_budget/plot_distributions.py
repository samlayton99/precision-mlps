"""Function-wise min, quartiles, median, and max for the K-theory figure."""
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
from experiments.expD65_radial_loss_budget.audit import OUT
from experiments.expD56_scale_acquisition.plot import apply_style,style_axis,COLORS
PROBS=[0,.25,.5,.75,1]

def summary(a):
    assert a.shape[0]==24 and np.isfinite(a).all()
    q=np.quantile(a,PROBS,axis=0,method='linear')
    np.testing.assert_array_equal(q[0],a.min(axis=0));np.testing.assert_array_equal(q[-1],a.max(axis=0))
    np.testing.assert_allclose(q[2],np.median(a,axis=0));assert np.all(np.diff(q,axis=0)>=-1e-12)
    return q

def draw(ax,t,q,color):
    ax.fill_between(t,q[1],q[3],color=color,alpha=.18,lw=0,zorder=1)
    ax.plot(t,q[0],color=color,lw=.75,ls=(0,(1.8,2.2)),alpha=.9,zorder=2)
    ax.plot(t,q[4],color=color,lw=.75,ls=(0,(1.8,2.2)),alpha=.9,zorder=2)
    ax.plot(t,q[2],color=color,lw=1.2,zorder=3)

def main():
    report=json.loads((OUT/'results.json').read_text());K=report['K_A']
    apply_style();plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5})
    fig=plt.figure(figsize=(7.35,8.0));grid=fig.add_gridspec(2,2,left=.095,right=.95,top=.86,bottom=.20,hspace=.43,wspace=.30,height_ratios=[1.1,1])
    axK=fig.add_subplot(grid[0,:]);axS=fig.add_subplot(grid[1,0]);axB=fig.add_subplot(grid[1,1])
    fig.suptitle('Conditional Scale Bound: GD, Default Initialization',fontsize=11,fontweight='bold',y=.98)
    fig.text(.5,.933,'24 functions · Width 196',ha='center',fontsize=8,color='#555555')
    axK.axvspan(20000,30000,color='#dddddd',alpha=.55,zorder=0)
    rows={}
    with np.load(OUT/'diagnostics.npz') as data:
        for lr in [.001,.0001]:
            cases=[r for r in report['cases'] if r['lr']==lr];assert len(cases)==24
            arrays=[data[r['case']] for r in cases];t=arrays[0][:,0]
            for a in arrays:np.testing.assert_array_equal(a[:,0],t)
            stack=np.stack(arrays);keep=t>=30000
            k=summary(stack[:,:,5]);slopes=summary(np.sqrt(stack[:,keep,2]/196))
            ceilings=np.array([r['prospective_slope_cap'] for r in cases])
            # Independently recalculate every Eq. (3) ceiling from the restart state.
            j=int(np.flatnonzero(t==30000)[0]);E=stack[:,j,1];A=stack[:,j,2]
            np.testing.assert_allclose(ceilings,np.sqrt((A+4*K*E+lr*E*E)/196),rtol=1e-14)
            cq=summary(np.repeat(ceilings[:,None],keep.sum(),axis=1))
            draw(axK,t,k,COLORS[lr]);draw(axS,t[keep],slopes,COLORS[lr]);draw(axB,t[keep],cq,COLORS[lr])
            rows[str(lr)]={'functions':[r['target'] for r in cases],'diagnostic_steps':t.tolist(),'diagnostic_quantiles':k.tolist(),
                           'continuation_steps':t[keep].tolist(),'measured_rms_quantiles':slopes.tolist(),'ceiling_quantiles':cq.tolist()}
    axK.axhline(K,color='#333333',lw=.9,ls=(0,(5,3)),zorder=2)
    axK.text(198000,K+18,f'Fixed K = {K:.1f}',ha='right',va='bottom',fontsize=7.5,color='#333333')
    axK.text(32000,1055,'Gray: 20k–30k calibration',fontsize=6.7,color='#777777',va='top')
    axK.set(title=r'a) Gradient-direction diagnostic $k_n$',ylabel=r'$k_n$',xlim=(20000,200000),ylim=(-20,1100))
    axS.set(title='b) Measured RMS slope',ylabel='RMS slope',xlim=(30000,200000),ylim=(.54,.79))
    axB.set(title='c) Conditional RMS ceiling',ylabel='RMS-slope ceiling',xlim=(30000,200000),ylim=(-.2,20))
    axB.axhline(18.375,color='#777777',ls=(0,(5,3)),lw=.8)
    axB.text(198000,18.375-.65,'QUILL: 18.375',ha='right',va='top',fontsize=7,color='#666666')
    axB.set_yticks([0,5,10,15,20])
    for ax in [axK,axS,axB]:
        style_axis(ax);ax.set_xlabel('GD step');ax.set_xticks([30000,100000,200000]);ax.xaxis.set_major_formatter(FuncFormatter(lambda n,_:f'{n/1000:g}k'))
    fig.legend([Line2D([],[],color=COLORS[lr],lw=1.5) for lr in [.001,.0001]],['LR = 0.001','LR = 0.0001'],
               loc='lower center',bbox_to_anchor=(.5,.105),ncol=2,frameon=False,fontsize=8,handlelength=3,columnspacing=3)
    fig.legend([Line2D([],[],color='#333333',lw=1.2),Patch(facecolor='#888888',alpha=.25,edgecolor='none'),
                Line2D([],[],color='#333333',lw=.8,ls=(0,(1.8,2.2)))],
               ['Median','25th–75th percentiles','Minimum / maximum'],loc='lower center',bbox_to_anchor=(.5,.062),
               ncol=3,frameon=False,fontsize=7.5,handlelength=2.8,columnspacing=2)
    fig.text(.5,.026,'Summaries across functions at each checkpoint; not confidence intervals.',ha='center',fontsize=7,color='#666666')
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'radial_loss_budget_distributions.{ext}',dpi=240)
    plt.close(fig)
    output={'probabilities':PROBS,'quantile_method':'linear','aggregation':'Across 24 equally weighted functions at each checkpoint, separately by learning rate; one paired seed.',
            'initialization':'Default only; larger initializations remain summarized in the note table.',
            'ceiling':'Equation (3) evaluated at 30k and constant thereafter, conditional on both premises.',
            'K':K,'learning_rates':rows}
    (OUT/'distribution_figure_data.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Verified min/median/max, ordered quartiles, common checkpoints, and independently recalculated ceilings for all 48 runs.')
    print(OUT/'radial_loss_budget_distributions.png')
if __name__=='__main__':main()
