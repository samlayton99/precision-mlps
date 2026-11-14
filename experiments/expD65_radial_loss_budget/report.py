"""Summarize the fixed-threshold audit and plot the primary default-init result."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import json,csv,sys,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD65_radial_loss_budget.audit import OUT
from experiments.expD56_scale_acquisition.plot import apply_style,style_axis,COLORS

def main():
    primary=json.loads((OUT/'results.json').read_text());extra=json.loads((OUT/'extra_initializations.json').read_text());K=primary['K_A'];allrows=[]
    oldchecks={r['run']:r for r in csv.DictReader((ROOT/'results/checkpoint_D_optimizers/expD63_loss_descent_sweep/run_results.csv').open())}
    for r in primary['cases']:
        assert oldchecks[r['case']]['step_condition_passes']=='True'
        allrows.append({**r,'initialization':1.,'K_max':r['K_heldout_A'],'K_condition_pass':r['A_condition_pass'],
                        'starting_state_horizon_descent_check_pass':True})
    allrows+=extra['cases']
    groups=[]
    for init in [1.,2.,4.]:
        rs=[r for r in allrows if r['initialization']==init];passed=[r for r in rs if r['K_condition_pass']]
        groups.append({'initialization':init,'pass_count':len(passed),'total':len(rs),
                       'largest_K':max(r['K_max'] for r in rs),'max_cap_in_passing_cases':max(r['prospective_slope_cap'] for r in passed),
                       'actual_max_rms':max(r['maximum_actual_slope_rms'] for r in rs),
                       'descent_starting_state_pass':sum(r['starting_state_horizon_descent_check_pass'] for r in rs)})
    mincritical=min((196*18.375**2-196*r.get('restart_slope_rms',0)**2-r['lr']*r['restart_E']**2)/(4*r['restart_E']) for r in primary['cases'])
    result={'K':K,'groups':groups,'minimum_primary_K_critical_for_18_375':mincritical,
            'primary_checkpoint_count':primary['records_per_run']*48,'extra_checkpoint_count':341*96,
            'scope':'RMS slope ceiling, conditional on radial premise at every update. Checkpoint validation is not an intervening-step or indefinite certificate.'}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# A conditional scale bound that remains informative for default initialization','',
           f'The common slope-radial threshold is K={K:.6f}, twice the largest observed value on the default-initialization 20k–30k prefixes. It is frozen before evaluating the later window and the 2x/4x cases. This is exploratory temporal validation; the candidate condition was developed after inspecting some later states.','',
           '| Initialization | Radial condition passes | Largest observed radial gain | Largest conditional RMS ceiling among passing cases |',
           '|---|---:|---:|---:|']
    for g in groups:lines.append(f"| {g['initialization']:g}x | {g['pass_count']}/{g['total']} | {g['largest_K']:.2f} | {g['max_cap_in_passing_cases']:.3f} |")
    lines += ['',
        'All 144 cases pass the independent sufficient loss-descent check over their analyzed finite horizon (48 default cases use the previously evaluated initialization-to-200k check; the 96 added cases use a fresh 30k-to-200k starting-state check). These are floating-point evaluations of analytic inequalities, not interval certificates. The new radial premise is supported only at saved checkpoints every 500 updates.','',
        'The single added-initialization radial failure is Airy, 2x initialization, LR 0.001: observed radial gain reaches 1389.41, above the unchanged threshold 1000.27. Its measured slopes still lie below the candidate ceiling, but that does not make its failed premise pass. No bound is claimed for that run under the fixed threshold.','',
        'The full-parameter version using a separately calibrated threshold passed only 44/48 primary runs. The successful primary statement specifically controls the slope block and RMS slope; it is not a bound on every parameter or the maximum individual slope.','',
        'For the primary 48 cases, the worst constant RMS ceiling is about 4.56, below the QUILL reference 18.375. No step-count term appears in the ceiling. It remains a bound for arbitrarily long continuations only if its premises persist. The saved data stop at 200k and do not prove that persistence; some radial gains are still rising near the endpoint.','',
        f'The smallest primary threshold that would make the conservative ceiling touch the construction reference is approximately {mincritical:.0f}. This indicates headroom in the implication, not validation that any chosen threshold holds forever. The QUILL reference is sufficient for the construction, not necessary for every accurate fit.','',
        'No new training was run. All 50,064 evaluated states come from saved trajectories. Autograd independently checks the diagnostic on four varied states; 10,000 randomized exact-descent cases check the discrete algebraic coefficients. The proof is elementary but has not received independent mathematical review.','',
        'The remaining mathematical task is to derive or weaken the radial-gain premise from structural properties of the tanh dynamics. The present contribution is a useful conditional implication plus a directly falsifiable diagnostic, not an unconditional explanation of small scales.','',
        '[Proof](../../../docs/review_process/iclr_2027/radial_loss_budget_bound.md) · [Figure](radial_loss_budget.png) · [Primary results](results.json) · [Extra initialization results](extra_initializations.json)']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n')
    apply_style();fig,axes=plt.subplots(1,2,figsize=(10.2,3.65));fig.subplots_adjust(left=.075,right=.98,bottom=.25,top=.78,wspace=.27)
    fig.suptitle('Conditional Scale Bound: GD, Default Initialization',fontsize=12,fontweight='bold',y=.98)
    fig.text(.5,.885,'24 functions · Width 196 · Learning rates 0.001 and 0.0001',ha='center',fontsize=8,color='#555555')
    with np.load(OUT/'diagnostics.npz') as data:
        for lr in [.001,.0001]:
            names=[r['case'] for r in primary['cases'] if r['lr']==lr]
            v=np.stack([data[n] for n in names]);s=v[0,:,0]
            axes[0].plot(s,np.max(v[:,:,5],axis=0),color=COLORS[lr],lw=1.1,label=f'LR = {lr:g}')
        allv=np.stack([data[r['case']] for r in primary['cases']]);s=allv[0,:,0];keep=s>=30000
        actual=np.sqrt(np.max(allv[:,keep,2],axis=0)/196)
    axes[0].axvspan(20000,30000,color='#dddddd',alpha=.6,zorder=-1)
    axes[0].text(.015,.96,'Gray: 20k–30k calibration',transform=axes[0].transAxes,fontsize=6.5,color='#777777',va='top')
    axes[0].axhline(K,color='#333333',ls=(0,(4,2)),lw=1,label=f'Fixed threshold: {K:.0f}')
    axes[0].set(xlim=(20000,200000),ylim=(0,1100),ylabel='Largest radial gain across 24 functions',title='Gradient-direction condition')
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,-.20),frameon=False,fontsize=7,ncol=2)
    cap=groups[0]['max_cap_in_passing_cases']
    axes[1].plot(s[keep],actual,color='#333333',lw=1.2,label='Largest measured RMS slope')
    axes[1].axhline(cap,color='#0072B2',lw=1.3,label=f'Largest conditional ceiling: {cap:.2f}')
    axes[1].axhline(18.375,color='#888888',ls=(0,(4,2)),lw=1,label='QUILL reference: 18.375')
    axes[1].set(xlim=(30000,200000),ylim=(0,20),ylabel='RMS slope',title='Bound from the remaining loss budget')
    axes[1].legend(loc='upper left',bbox_to_anchor=(0,-.20),frameon=False,fontsize=7)
    for ax in axes:
        style_axis(ax);ax.set_xlabel('GD step');ax.set_xticks([30000,100000,200000]);ax.xaxis.set_major_formatter(FuncFormatter(lambda n,_:f'{n/1000:g}k'))
    fig.savefig(OUT/'radial_loss_budget.png',dpi=220);fig.savefig(OUT/'radial_loss_budget.svg');plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
