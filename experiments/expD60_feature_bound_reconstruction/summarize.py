"""Evaluate the appendix's joint scale/error success criterion on D60 continuations.

Uses an independent normalized-q recurrence, while bound.py uses physical B.
No training or checkpoint modification. Partial reports are explicitly labeled.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter,NullLocator

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD56_scale_acquisition.audit_slope_growth import constants,force_terms
from experiments.expD56_scale_acquisition.plot import apply_style,style_axis
from experiments.expD60_feature_bound_reconstruction.run import CONFIG,CASES,OUT
from experiments.expD60_feature_bound_reconstruction.bound import evaluate as direct_evaluate

FAMILIES={'generic':(0,),'projected':(0,1),'refined':(0,1,2,3)}
COLORS={'generic':'#0072B2','projected':'#E69F00','refined':'#009E73'}
START=25000
H=2/512
REFERENCE=.25

def analyze(history,x,y,width,eta,family):
    t=history['step'].astype(int);start=int(np.flatnonzero(t==START)[0]);end=int(t[-1])
    assert end>START
    assert np.isfinite(np.column_stack(list(history.values()))).all()
    assert np.all(history['coarse_kernel_min_eigenvalue'][start:]>0)
    assert np.all(history['parameter_energy'][start:]>0)
    coefficients=constants(x);norm_y=float(np.sqrt(np.mean(y*y)))
    E=float(history['nonaffine_residual_rms'][start:].max())
    target=float(history['target_nonaffine_rms'][start])
    tau3=float(history['tau_3'][start]);tau5=float(history['tau_5'][start])
    chi=np.column_stack([history[f'chi_{p}'] for p in [4,6,10,14]])
    q=float(np.sqrt(history['parameter_energy'][start]/width))
    def floor(q,shape):
        q2=q*q;q4=q2*q2;q6=q4*q2
        Q=width*min(coefficients['A3']*shape[0]*q4,
                    coefficients['B3']*shape[0]*q4+coefficients['A5']*shape[1]*q6)
        return max(0.,target-Q)/norm_y
    initial_floor=floor(q,chi[start])
    loss_step=START if H*q>=REFERENCE or initial_floor<=.01 else None
    scale_crossing=None;force_counts=[0]*4;trace=[[START,H*q,initial_floor]]
    violations=0
    for k in range(start,len(t)-1):
        dt=int(t[k+1]-t[k]);base=chi[k];change=(chi[k+1]-base)/dt
        R0=float(history['tracking_norm'][k]);dR=float(history['tracking_norm'][k+1]-R0)/dt
        for offset in range(dt):
            c=base+offset*change
            terms=force_terms(q,width,c,E,tau3,tau5,coefficients)
            choice=min(FAMILIES[family],key=lambda j:terms[j]);force_counts[choice]+=1
            q+=eta*(terms[choice]+(R0+offset*dR)/np.sqrt(width))
            step=int(t[k]+offset+1)
            next_floor=floor(q,base+(offset+1)*change)
            if loss_step is None and (H*q>=REFERENCE or next_floor<=.01):loss_step=step
            if H*q>=REFERENCE:
                scale_crossing=step;trace.append([step,H*q,next_floor]);break
        if scale_crossing is not None:break
        trace.append([int(t[k+1]),H*q,next_floor])
        violations+=int(q+1e-10<np.sqrt(history['parameter_energy'][k+1]/width))
        assert next_floor<=history['train_rel_l2'][k+1]+1e-8
    useful=end-START if loss_step is None else loss_step-START
    # Natural preceding-window concentration diagnostic; the paper does not give
    # enough detail to claim this is its exact diagnostic implementation.
    before=(t>=20000)&(t<=25000);after=t>=25000
    baseline=float(np.trapezoid(np.sqrt(history['chi_6'][before]),t[before])/5000)
    elapsed=t[after]-25000;vals=np.sqrt(history['chi_6'][after])
    accumulated=np.r_[0.,np.cumsum(.5*(vals[1:]+vals[:-1])*np.diff(t[after]))]
    ratio=np.divide(accumulated,elapsed*baseline,out=np.ones_like(accumulated),where=elapsed>0)
    return {'family':family,'start':START,'end':end,'initial_norm':float(np.sqrt(history['parameter_energy'][start])),
            'E0':E,'initial_error_floor':initial_floor,'useful_updates':int(useful),'first_not_useful_step':loss_step,
            'full_available_interval_useful':loss_step is None,'full_100k_useful':end==125000 and loss_step is None,
            'scale_crossing_step':scale_crossing,'final_scale_ceiling':float(H*q) if scale_crossing is None else None,
            'final_error_floor':float(next_floor) if scale_crossing is None else 0.,
            'force_choices':force_counts,'parameter_envelope_violations':violations,
            'preceding_window_sqrt_chi6_mean':baseline,'max_accumulated_sqrt_chi6_ratio':float(ratio.max()),
            'trace':trace}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    destination=OUT/('preview' if args.partial else 'reconstruction');destination.mkdir(exist_ok=True)
    data=[];source_hashes={}
    for init,name in CASES:
        directory=OUT/f'{init}__{name}';path=directory/'checkpoint.npz'
        with np.load(path) as z,np.load(directory/'initial.npz') as first:
            h={k:z['history'][:,i] for i,k in enumerate(z['keys'].tolist())};x=first['x'];y=first['y']
        if not args.partial:assert h['step'][-1]==125000
        if h['step'][-1]<=START:continue
        results=[analyze(h,x,y,CONFIG['width'],CONFIG['learning_rate'],family) for family in FAMILIES]
        for result in results:
            independent=direct_evaluate(h,x,CONFIG['width'],CONFIG['learning_rate'],REFERENCE/H,START,result['family'])
            assert result['scale_crossing_step']==independent['reference_crossing']
            assert result['force_choices']==independent['force_choices']
            if result['scale_crossing_step'] is None:
                np.testing.assert_allclose(result['final_scale_ceiling'],H*independent['end_ceiling'],rtol=1e-8)
                np.testing.assert_allclose(result['final_error_floor'],independent['end_error_floor']/np.sqrt(np.mean(y*y)),rtol=1e-8,atol=1e-12)
            assert result['parameter_envelope_violations']==0
        data.append({'initialization':init,'function':name,'history':h,'results':results})
        source_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(data)==4
    apply_style();fig,axes=plt.subplots(2,4,figsize=(12.2,6.2),sharex=True,sharey='row')
    fig.subplots_adjust(left=.075,right=.985,bottom=.18,top=.80,wspace=.20,hspace=.30)
    title='Appendix Reconstruction: Joint-training Bound'
    if args.partial:title+=' (partial)'
    fig.suptitle(title,fontsize=13,fontweight='bold',y=.98)
    fig.text(.5,.925,'Width 705 · GD learning rate 0.002 · 2,048 training points · analysis from 25k',ha='center',fontsize=8)
    labels=['$P_5(x)$\nSmaller initialization','Localized oscillation\nSmaller initialization',
            '$x^2$\nSmaller initialization','$P_5(x)$\nPyTorch default']
    for col,(record,title) in enumerate(zip(data,labels)):
        h=record['history'];keep=h['step']>=START;t=h['step'][keep]
        top,bottom=axes[:,col];top.set_title(title,fontweight='bold',fontsize=9,pad=8)
        top.plot(t,H*h['gamma_rms'][keep],color='#222222',lw=1.2)
        bottom.plot(t,h['train_rel_l2'][keep],color='#222222',lw=1.2)
        for result in record['results']:
            trace=np.asarray(result['trace']);color=COLORS[result['family']]
            top.plot(trace[:,0],np.minimum(trace[:,1],REFERENCE),color=color,lw=.95,ls='--')
            if result['scale_crossing_step'] is not None:
                top.plot(result['scale_crossing_step'],REFERENCE,marker='^',color=color,ms=4)
            error_steps=trace[:,0];error_floor=trace[:,2]
            if result['scale_crossing_step'] is not None and error_steps[-1]<t[-1]:
                # After stopping the upper-bound calculation, retain only the
                # universally valid zero error floor; do not extrapolate B.
                error_steps=np.r_[error_steps,t[-1]];error_floor=np.r_[error_floor,0.]
            bottom.plot(error_steps,np.maximum(error_floor,1e-4),color=color,lw=.95,ls='--')
        for ax in (top,bottom):
            style_axis(ax);ax.set(xlim=(25000,125000),yscale='log')
            ax.set_xticks([25000,50000,75000,100000,125000]);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'))
            ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator())
        top.set_ylim(1e-4,.6);top.axhline(REFERENCE,color='#777777',lw=.65,ls=':')
        top.annotate(r'QUILL: $\lambda=0.25$',xy=(.97,REFERENCE),xycoords=('axes fraction','data'),
                     xytext=(0,3),textcoords='offset points',ha='right',fontsize=6.5)
        bottom.set(ylim=(7e-5,2),xlabel='GD step');bottom.axhline(.01,color='#777777',lw=.65,ls=':')
        bottom.set_yticks([1e-4,1e-3,.01,.1,1]);bottom.set_yticklabels([r'$\leq10^{-4}$','$10^{-3}$','1%','10%','100%'])
    axes[0,0].set_ylabel(r'RMS scale $\lambda=h\gamma_{\rm RMS}$')
    axes[1,0].set_ylabel('Relative training error')
    fig.legend([Line2D([],[],color='#222222',lw=1.2)]+[Line2D([],[],color=COLORS[f],lw=1.2,ls='--') for f in FAMILIES],
               ['Measured','Generic bound','Affine-projected bound','Target-refined bound'],ncol=4,frameon=False,
               loc='lower center',bbox_to_anchor=(.5,.075),columnspacing=2.)
    fig.text(.5,.035,'Top: upper scale bounds. Bottom: lower error bounds. Useful requires both scale < 0.25 and error floor > 1%.',
             ha='center',fontsize=7,color='#555555')
    for ext in ['png','pdf']:fig.savefig(destination/f'appendix_reconstruction.{ext}',dpi=230)
    plt.close(fig)
    public=[{'initialization':r['initialization'],'function':r['function'],'results':r['results']} for r in data]
    report={'partial':args.partial,'settings':CONFIG,'source_sha256':source_hashes,
            'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Controlled reconstruction of published settings. Initialization and target subset declared explicitly; not an exact replication of the original 46-case count.',
            'qualifications':'Inputs are interpolated checkpoint records, not interval-certified bounds. Concentration diagnostic uses the preceding-window mean as a declared interpretation.',
            'cases':public}
    (destination/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    table=[]
    for r in public:
        for result in r['results']:
            table.append({'initialization':r['initialization'],'function':r['function'],
                          **{k:v for k,v in result.items() if k not in ['trace','force_choices']}})
    with (destination/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps([{k:v for k,v in row.items() if k in ['initialization','function','family','useful_updates','full_100k_useful','final_scale_ceiling','final_error_floor']} for row in table],indent=2),flush=True)
    print(destination/'appendix_reconstruction.png')

if __name__=='__main__':main()
