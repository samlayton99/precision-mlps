"""Adam diagnostics; original GD recurrence shown only as an unproved comparator."""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter,NullLocator

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD56_scale_acquisition.plot import apply_style,style_axis
from experiments.expD56_scale_acquisition.audit_slope_growth import constants,force_terms
from experiments.expD64_adam_joint_training_diagnostics.run import OUT,WIDTHS,TARGETS,RATES

RESTART=30000
DISPLAY_CAP=1000.
COLORS={'actual':'#0072B2','ceiling':'#CC79A7','chi':'#009E73','tracking':'#D55E00','residual':'#B8860B'}
STYLES={.001:'solid',.0001:(0,(5,3))}
NAMES={'square':r'$x^2$','sine_4pi':r'$\sin(4\pi x)$','runge_100':r'$1/(1+100x^2)$'}

def ceiling(h,width,lr,geometry_constants,cap=DISPLAY_CAP):
    """F.6 using the minimum of all four F.4 forces. Fixed E0; interpolated chi/R.

    This is an empirical checkpoint-based evaluation, not an every-update
    certificate. Stop only the ceiling evaluation at the declared display cap;
    never reset the ceiling to the actual parameter norm.
    """
    steps=h['step'].astype(int);i=int(np.flatnonzero(steps==RESTART)[0])
    e0=float(np.max(h['nonaffine_residual_rms'][i:]))
    b=float(np.sqrt(h['parameter_energy'][i]));initial=b
    ts=[RESTART];bs=[b];crossing=None;choices=[0,0,0,0]
    for k in range(i,len(steps)-1):
        count=int(steps[k+1]-steps[k])
        for offset in range(count):
            fraction=offset/count
            chi=[float((1-fraction)*h[f'chi_{order}'][k]+fraction*h[f'chi_{order}'][k+1]) for order in [4,6,10,14]]
            r=float((1-fraction)*h['tracking_norm'][k]+fraction*h['tracking_norm'][k+1])
            terms=force_terms(b/np.sqrt(width),width,chi,e0,float(h['tau_3'][i]),float(h['tau_5'][i]),geometry_constants)
            choice=int(np.argmin(terms));choices[choice]+=1
            allowance=terms[choice]*np.sqrt(width)
            b+=lr*(allowance+r)
            step=int(steps[k]+offset+1)
            # Dense points retain the rapid ceiling rise between stored model snapshots.
            ts.append(step);bs.append(b)
            if b>cap:
                crossing=step
                break
        if crossing is not None:break
    return {'step':np.asarray(ts),'B':np.asarray(bs),'E0':e0,'initial_B':initial,
            'cap_crossing_step':crossing,'last_computed_B':b,'force_choices':choices}

def load(allow_partial):
    records={};hashes={}
    for path in sorted(OUT.glob('width*/runs/*/checkpoint.npz')):
        status=json.loads((path.parent/'status.json').read_text())
        with np.load(path) as z:h={k:z['history'][:,i].copy() for i,k in enumerate(z['keys'].tolist())}
        assert np.isfinite(np.column_stack(list(h.values()))).all()
        assert np.all(h['parameter_energy']>0) and np.all(h['chi_6']>=1-1e-12)
        assert np.all(h['coarse_kernel_min_eigenvalue']>0)
        if not allow_partial:assert status['state']=='complete' and h['step'][-1]==50000
        if h['step'][-1]<RESTART:continue
        key=(status['width'],status['target'],status['learning_rate'])
        with np.load(path.parent/'initial.npz') as initial:geometry_constants=constants(initial['train_x'])
        records[key]={'h':h,'bound':ceiling(h,status['width'],status['learning_rate'],geometry_constants)}
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    if not allow_partial:assert len(records)==24
    return records,hashes

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');parser.add_argument('--restart',type=int,default=1000);args=parser.parse_args()
    global RESTART
    RESTART=args.restart
    records,hashes=load(args.partial)
    destination=OUT/(('preview' if args.partial else 'figures')+f'_restart_{RESTART}');destination.mkdir(exist_ok=True)
    apply_style()
    fig,axes=plt.subplots(4,3,figsize=(11.2,10.0),sharex=True,sharey=True)
    fig.subplots_adjust(left=.10,right=.985,bottom=.16,top=.90,hspace=.26,wspace=.16)
    title='Joint Training Diagnostics: Adam, 50k steps'
    if args.partial:title+=' (partial data)'
    fig.suptitle(title,fontsize=13,fontweight='bold',y=.975)
    fig.text(.5,.941,'Rows: total width     Columns: target function     Pink: GD recurrence comparator only',ha='center',fontsize=8,color='#555555')
    summaries=[]
    for row,width in enumerate(WIDTHS):
        for col,name in enumerate(TARGETS):
            ax=axes[row,col];style_axis(ax)
            if row==0:ax.set_title(NAMES[name],fontsize=11,fontweight='bold',pad=9)
            if col==0:ax.set_ylabel(f'Width {width}\nMagnitude',fontsize=8)
            for lr in RATES:
                record=records.get((width,name,lr))
                if record is None:continue
                h=record['h'];b=record['bound'];s=h['step'];valid=s>=1
                style=STYLES[lr]
                ax.plot(s[valid],np.sqrt(h['parameter_energy'][valid]),color=COLORS['actual'],ls=style,lw=.9)
                ax.plot(s[valid],h['chi_6'][valid],color=COLORS['chi'],ls=style,lw=.8)
                ax.plot(s[valid],h['tracking_norm'][valid],color=COLORS['tracking'],ls=style,lw=.8)
                ax.plot([RESTART,s[-1]],[b['E0']]*2,color=COLORS['residual'],ls=style,lw=.8)
                visible=b['B']<=DISPLAY_CAP
                ax.plot(b['step'][visible],b['B'][visible],color=COLORS['ceiling'],ls=style,lw=1.)
                ax.plot(RESTART,b['initial_B'],marker='o',ms=2.5,color=COLORS['ceiling'],zorder=5)
                if b['cap_crossing_step'] is not None:
                    # End with an upward marker, never draw a false plateau at the cap.
                    ax.plot([b['step'][-2],b['step'][-1]],[b['B'][-2],DISPLAY_CAP],color=COLORS['ceiling'],ls=style,lw=1.)
                    ax.plot(b['step'][-1],DISPLAY_CAP,marker='^' if lr==.001 else '^',ms=4,
                            markerfacecolor=COLORS['ceiling'] if lr==.001 else 'white',
                            markeredgecolor=COLORS['ceiling'],markeredgewidth=.8,zorder=5)
                late=np.flatnonzero(s>=RESTART);R=h['tracking_norm'][late];t=s[late]
                tracking_path=float(lr*(np.trapezoid(R,t)-.5*(R[-1]-R[0])))
                summaries.append({'width':width,'target':name,'learning_rate':lr,'last_step':int(s[-1]),
                                  'actual_norm_restart':b['initial_B'],'actual_norm_final':float(np.sqrt(h['parameter_energy'][-1])),
                                  'chi6_restart':float(h['chi_6'][late[0]]),'chi6_max_after_restart':float(h['chi_6'][late].max()),
                                  'R_restart':float(R[0]),'R_final':float(R[-1]),'E0':b['E0'],
                                  'tracking_path_estimate':tracking_path,
                                  'force_choices_generic_projected_target3_target5':b['force_choices'],
                                  'ceiling_exceeds_display_cap_step':b['cap_crossing_step']})
            ax.axvline(RESTART,color='#AAAAAA',lw=.55,ls=':',zorder=0)
            ax.set(xscale='log',yscale='log',xlim=(1,50000),ylim=(1e-11,2500))
            ax.set_xticks([1,100,1000,10000,50000])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k' if x>=1000 else f'{x:g}'))
            ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator())
            ax.set_yticks([1e-10,1e-7,1e-4,1e-1,1e2])
            if row==3:ax.set_xlabel('Adam step (log scale)')
            if not any((width,name,lr) in records for lr in RATES):
                ax.text(.5,.5,'Waiting for training',transform=ax.transAxes,ha='center',color='#888888')
    handles=[Line2D([],[],color=COLORS[k],lw=1.4) for k in COLORS]
    labels=[r'Actual parameter norm $\sqrt{M_h}$',r'GD comparator $B_n$ (not an Adam bound)',
            r'Concentration $\bar\chi_{6,n}$',r'Raw-gradient tracking $\bar R_n$',r'Residual bound $E_0$']
    fig.legend(handles,labels,ncol=3,frameon=False,loc='lower center',bbox_to_anchor=(.5,.071),
               handlelength=2.5,columnspacing=2.2,fontsize=8)
    fig.legend([Line2D([],[],color='#333333',ls=STYLES[lr],lw=1.2) for lr in RATES],
               [f'LR = {lr:g}' for lr in RATES],ncol=2,frameon=False,loc='lower center',
               bbox_to_anchor=(.5,.033),handlelength=3.,columnspacing=4.,fontsize=8)
    fig.text(.5,.012,rf'GD comparator starts at {RESTART/1000:g}k; triangles mark $B_n>10^3$. Checkpoint inputs are interpolated; $E_0$ is fixed.',
             ha='center',fontsize=7,color='#555555')
    for ext in ['png','pdf']:fig.savefig(destination/f'joint_training_diagnostics.{ext}',dpi=210)
    plt.close(fig)
    report={'partial':args.partial,'runs':len(records),'restart':RESTART,'display_cap':DISPLAY_CAP,
            'bound':'GD-only F.6 comparator on Adam diagnostics: not a valid Adam ceiling. All four F.4 alternatives used.',
            'E0':'Maximum recorded nonaffine residual RMS over the full available suffix from the selected restart; fixed for each recurrence.',
            'allowances':'chi6 and tracking norms from recorded checkpoints, linearly interpolated per update; empirical, not certified between checkpoints.',
            'stopping_rule':'Stop ceiling evaluation once B exceeds the display cap; actual Adam training continues to 50k.',
            'source_sha256':hashes,'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':summaries}
    (destination/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(destination/'joint_training_diagnostics.png')

if __name__=='__main__':main()
