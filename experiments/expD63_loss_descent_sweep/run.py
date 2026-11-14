"""Evaluate the proposed loss-descent slope bound on all 216 saved D56/D58 runs."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter,NullLocator

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD62_joint_training_diagnostics.loss_descent_bound import evaluate
from experiments.expD56_scale_acquisition.plot import apply_style,style_axis,COLORS,function_shades

BASE=ROOT/'results/checkpoint_D_optimizers'
OUT=BASE/'expD63_loss_descent_sweep'
SOURCES=[BASE/'expD56_scale_acquisition/sweep',BASE/'expD58_extra_hard_scale_acquisition/sweep']
RATES=[.01,.001,.0001]
INITS=[1,2,4]

def analyze():
    rows=[];curves={};source_hashes={}
    for source in SOURCES:
        for path in sorted((source/'runs').glob('*/checkpoint.npz')):
            meta=json.loads((path.parent/'status.json').read_text())
            assert meta['state']=='complete' and meta['step']==200000 and meta['width']==196
            with np.load(path) as z,np.load(path.parent/'initial.npz') as initial:
                theta=z['snapshots'];h={k:z['history'][:,j] for j,k in enumerate(z['keys'].tolist())}
                steps=h['step'].astype(int);x=initial['train_x'];y=initial['train_y']
                np.testing.assert_array_equal(theta[0],initial['theta'])
                assert len(steps)==409 and steps[0]==0 and steps[-1]==200000 and np.all(np.diff(steps)>0)
                assert np.max(np.abs(x))<=1 and np.isfinite(theta).all()
                W=meta['width'];eta=meta['learning_rate']
                result=evaluate(theta[0],x,y,W,eta,int(steps[-1]))
                actual=np.linalg.norm(theta[:,:W],axis=1)/np.sqrt(W)
                np.testing.assert_allclose(actual,h['gamma_rms'],rtol=1e-13)
                distance=np.linalg.norm(theta-theta[0],axis=1)
                distance_bound=result['initial_residual_rms']*np.sqrt(eta*steps)
                bound=result['initial_gamma_rms']+distance_bound/np.sqrt(W)
                # Independently obtain the same curve directly from the starting half-MSE.
                loss0=.5*h['train_rel_l2'][0]**2*np.mean(y*y)
                np.testing.assert_allclose(bound,actual[0]+np.sqrt(2*eta*steps*loss0/W),rtol=1e-13)
                slope_bad=actual>bound+1e-10;distance_bad=distance>distance_bound+1e-10
                if result['step_condition_passes']:
                    assert not slope_bad.any() and not distance_bad.any()
                row={**result,'run':path.parent.name,'function':meta['target'],
                     'initialization':meta['initialization_multiplier'],'learning_rate':eta,
                     'width':W,'start':0,'end':200000,'checkpoints':409,
                     'actual_gamma_rms_final':float(actual[-1]),'actual_gamma_rms_max':float(actual.max()),
                     'actual_parameter_distance_final':float(distance[-1]),
                     'gamma_bound_violations':int(slope_bad.sum()),'distance_bound_violations':int(distance_bad.sum()),
                     'max_recorded_loss_increase':float(max(0,np.diff(.5*h['train_rel_l2']**2*np.mean(y*y)).max())),
                     'source':str(path.relative_to(ROOT))}
                rows.append(row)
                curves[row['run']]={'steps':steps,'actual':actual,'bound':bound,'distance':distance,'distance_bound':distance_bound}
            source_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            source_hashes[str((path.parent/'initial.npz').relative_to(ROOT))]=hashlib.sha256((path.parent/'initial.npz').read_bytes()).hexdigest()
    assert len(rows)==216 and len({r['function'] for r in rows})==24
    return rows,curves,source_hashes

def draw(rows,curves,names,title):
    fig,axes=plt.subplots(1,3,figsize=(10.3,3.9),sharex=True,sharey=True)
    fig.subplots_adjust(left=.065,right=.985,bottom=.30,top=.76,wspace=.16)
    fig.suptitle(title,fontsize=12,fontweight='bold',y=.97)
    for ax,init,label in zip(axes,INITS,['a) Default Initialization','b) 2x Initialization','c) 4x Initialization']):
        style_axis(ax);ax.set_title(label,fontweight='bold',pad=8)
        chosen=[r for r in rows if r['initialization']==init]
        for eta in RATES:
            for name,color in zip(names,function_shades(eta,len(names))):
                matches=[r for r in chosen if r['learning_rate']==eta and r['function']==name]
                if not matches:continue
                r=matches[0];c=curves[r['run']]
                ax.plot(c['steps'],c['actual'],color=color,lw=.65,alpha=.95)
                ax.plot(c['steps'],c['bound'],color=color,lw=.7,
                        ls=(0,(5,3)) if r['step_condition_passes'] else (0,(1,3)),
                        alpha=.85 if r['step_condition_passes'] else .6,dash_capstyle='round')
        ax.text(.03,.94,f'Step-size check: {sum(r["step_condition_passes"] for r in chosen)}/{len(chosen)} pass',
                transform=ax.transAxes,fontsize=7,va='top',color='#444444')
        ax.set(xlim=(0,200000),ylim=(0,7),xlabel='GD step')
        ax.set_xticks([0,50000,100000,150000,200000]);ax.set_yticks([0,1,2,3,4,5,6,7])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k' if x else '0'))
        ax.xaxis.set_minor_locator(NullLocator())
    axes[0].set_ylabel(r'RMS slope $\gamma$')
    fig.legend([Line2D([],[],color=COLORS[lr],lw=1.5) for lr in RATES],[f'LR = {lr:g}' for lr in RATES],
               ncol=3,frameon=False,loc='lower center',bbox_to_anchor=(.5,.15),columnspacing=3.)
    fig.legend([Line2D([],[],color='#333333',lw=.9,ls=ls) for ls in ['solid',(0,(5,3)),(0,(1,3))]],
               ['Measured RMS slope','Bound: step-size check passes','Formula only: check does not pass'],
               ncol=3,frameon=False,loc='lower center',bbox_to_anchor=(.5,.07),columnspacing=2.,handlelength=3.)
    fig.text(.5,.025,'Each shade is one function. Dotted curves are not justified as bounds by this sufficient check.',ha='center',fontsize=7,color='#555555')
    return fig

def plot(rows,curves):
    apply_style();destination=OUT/'figures';destination.mkdir(exist_ok=True)
    names=[]
    for source in SOURCES:
        config=json.loads((source/'manifest.json').read_text())['config']
        names.extend(n for group in config['targets'].values() for n in group)
    fig=draw(rows,curves,names,'Loss-Descent Bound: 24 functions, 200k steps')
    for ext in ['png','pdf']:fig.savefig(destination/f'gamma_bound_comparison.{ext}',dpi=230)
    plt.close(fig)
    with PdfPages(destination/'all_24_functions.pdf') as pdf:
        for name in names:
            selected=[r for r in rows if r['function']==name]
            fig=draw(selected,curves,[name],f'RMS Scale: {name.replace("_"," ")}')
            pdf.savefig(fig);fig.savefig(destination/f'{name}.png',dpi=160);plt.close(fig)
    # Show the sufficient-condition statistic, including every failed case.
    fig,axes=plt.subplots(1,3,figsize=(10.3,3.2),sharey=True)
    fig.subplots_adjust(left=.075,right=.985,bottom=.20,top=.76,wspace=.16)
    fig.suptitle('Step-size Condition: 24 functions',fontsize=12,fontweight='bold',y=.97)
    for ax,init,title in zip(axes,INITS,['Default Initialization','2x Initialization','4x Initialization']):
        style_axis(ax);ax.set_title(title,fontweight='bold')
        for i,lr in enumerate(RATES):
            chosen=sorted([r for r in rows if r['initialization']==init and r['learning_rate']==lr],key=lambda r:r['function'])
            offsets=np.linspace(-.15,.15,len(chosen))
            ax.scatter(i+offsets,[r['eta_L_bound'] for r in chosen],color=COLORS[lr],s=13,alpha=.8,linewidths=0)
        ax.axhline(1,color='#333333',ls='--',lw=.9)
        ax.text(.98,1,'Pass at or below 1',transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=7)
        ax.set(yscale='log',xlim=(-.4,2.4),xlabel='Learning rate')
        ax.set_xticks(range(3),['0.01','0.001','0.0001'])
    axes[0].set_ylabel(r'Step-size statistic $\eta\widehat L$')
    for ext in ['png','pdf']:fig.savefig(destination/f'step_size_condition.{ext}',dpi=230)
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True);rows,curves,hashes=analyze()
    summaries=[]
    for init in INITS:
        for lr in RATES:
            group=[r for r in rows if r['initialization']==init and r['learning_rate']==lr]
            summaries.append({'initialization':init,'learning_rate':lr,'runs':len(group),
                              'passes':sum(r['step_condition_passes'] for r in group),
                              'eta_L_min':min(r['eta_L_bound'] for r in group),'eta_L_max':max(r['eta_L_bound'] for r in group),
                              'gamma_bound_violations':sum(r['gamma_bound_violations'] for r in group)})
    report={'start':0,'end':200000,'runs':216,'functions':24,'condition':'eta * L_hat <= 1, with the starting-state sufficient tanh bound from D62',
            'qualification':'The sufficient condition and bounds are evaluated in float64, not outward-rounded interval arithmetic. A failed sufficient check does not establish a violation of the underlying descent condition.',
            'passed':sum(r['step_condition_passes'] for r in rows),'source_sha256':hashes,
            'evaluator_sha256':hashlib.sha256((ROOT/'experiments/expD62_joint_training_diagnostics/loss_descent_bound.py').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'groups':summaries,'cases':rows}
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    fields=[k for k in rows[0] if k!='step_check_constants']
    with (OUT/'run_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    np.savez_compressed(OUT/'curves.npz',run_names=np.array([r['run'] for r in rows]),steps=curves[rows[0]['run']]['steps'],
                        actual_gamma_rms=np.array([curves[r['run']]['actual'] for r in rows]),
                        formula_gamma_ceiling=np.array([curves[r['run']]['bound'] for r in rows]),
                        condition_passes=np.array([r['step_condition_passes'] for r in rows]))
    print(json.dumps({'passes':report['passed'],'groups':summaries,
                      'failed_at_lr_001':[r['run'] for r in rows if r['learning_rate']==.001 and not r['step_condition_passes']],
                      'gamma_violations':sum(r['gamma_bound_violations'] for r in rows),
                      'distance_violations':sum(r['distance_bound_violations'] for r in rows)},indent=2),flush=True)
    plot(rows,curves)
    print(OUT/'figures/gamma_bound_comparison.png')

if __name__=='__main__':main()
