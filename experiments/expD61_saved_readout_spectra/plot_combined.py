"""Matched scale and readout-rate panels from the completed 24-function sweeps."""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expD56_scale_acquisition.plot import apply_style,style_axis,function_shades,COLORS

BASE=ROOT/'results/checkpoint_D_optimizers'
SCALE=BASE/'expD58_extra_hard_scale_acquisition/combined_24_functions'
SPECTRA=BASE/'expD61_saved_readout_spectra'
OUT=SPECTRA/'combined_figure'

def main():
    OUT.mkdir(exist_ok=True)
    scale_path=SCALE/'scale_acquisition_displayed.json'
    scales=json.loads(scale_path.read_text())
    provenance=json.loads((SCALE/'figure_provenance.json').read_text())
    assert not scales['synthetic'] and not scales['partial']
    names=provenance['functions'];assert len(names)==24
    curves=scales['individual_trajectories']
    records={};source_hashes={}
    for path in sorted((SPECTRA/'runs').glob('*/spectra.npz')):
        meta=json.loads((path.parent/'metadata.json').read_text())
        assert meta['source_sha256']==provenance['source_sha256'][meta['source']]
        key=f"init{meta['initialization_multiplier']:g}/lr{meta['learning_rate']:.0e}/{meta['target']}"
        with np.load(path) as z:
            records[key]={k:z[k].copy() for k in ('steps','percentiles','quantiles','resolution_floor')}
        np.testing.assert_array_equal(records[key]['steps'],curves[key]['step'])
        source_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(records)==set(curves) and len(records)==216
    floor=float(next(iter(records.values()))['resolution_floor'])
    assert all(float(r['resolution_floor'])==floor for r in records.values())
    reference=provenance['config']['quill_reference']
    gamma_ref=reference['lambda']*reference['interior_intervals']/2
    # (1-rho/2)**T = exp(-1), using the theorem's eta=1/(2*mu_max).
    horizon=200000
    rate_ref=float(-2*np.expm1(-1/horizon))
    np.testing.assert_allclose(horizon*np.log1p(-rate_ref/2),-1,rtol=1e-14)
    apply_style()
    fig,axes=plt.subplots(2,3,figsize=(9.4,6.15),sharex=True,sharey='row')
    fig.subplots_adjust(left=.105,right=.985,bottom=.17,top=.84,wspace=.20,hspace=.54)
    fig.suptitle('Scale Acquisition Ablation: 24 functions',fontsize=12,fontweight='bold',y=.98)
    fig.text(.105,.915,'Scales remain small',fontsize=9,fontweight='bold')
    fig.text(.105,.475,'Slow readout directions persist',fontsize=9,fontweight='bold')
    titles=['a) Default Initialization','b) 2x Initialization','c) 4x Initialization']
    plotted=0
    for col,(init,title) in enumerate(zip([1,2,4],titles)):
        upper,lower=axes[:,col]
        upper.set_title(title,fontweight='bold',pad=8)
        final_medians=[]
        for lr in [.01,.001,.0001]:
            for name,color in zip(names,function_shades(lr,24)):
                key=f'init{init}/lr{lr:.0e}/{name}'
                slope=curves[key];spectrum=records[key]
                steps=np.asarray(slope['step'])
                upper.fill_between(steps,slope['p1'],slope['p99'],color=color,alpha=.025,lw=0)
                for boundary in ('p1','p99'):
                    upper.plot(steps,slope[boundary],color=color,alpha=.65,lw=.45,
                               ls=(0,(1,3)),dash_capstyle='round')
                upper.plot(steps,slope['median'],color=color,lw=.6)
                final_medians.append(slope['median'][-1])
                index=int(np.flatnonzero(spectrum['percentiles']==90)[0])
                lower.plot(steps,np.maximum(spectrum['quantiles'][:,index],floor),
                           color=color,lw=.60,alpha=.9)
                plotted+=1
        for ax in (upper,lower):
            style_axis(ax)
            ax.set_xlim(0,horizon)
            ax.set_xticks([0,50000,100000,150000,200000])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k' if x else '0'))
            ax.xaxis.set_minor_locator(NullLocator())
        upper.set_ylim(0,20.8);upper.set_yticks([0,5,10,15,20])
        upper.axhline(gamma_ref,color='#333333',ls='--',lw=1.)
        upper.annotate(rf'QUILL: $\gamma$ = {gamma_ref:g}',xy=(.98,gamma_ref),
                       xycoords=('axes fraction','data'),xytext=(0,3),textcoords='offset points',
                       ha='right',va='bottom',fontsize=7)
        median=float(np.median(final_medians))
        upper.annotate(f'Median: {median:.2f}',xy=(.97,median),
                       xycoords=('axes fraction','data'),xytext=(0,2.5),textcoords='offset points',
                       ha='right',va='bottom',fontsize=7,color='#222222',
                       bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1})
        lower.set(yscale='log',ylim=(floor/6,1e-3),xlabel='GD step')
        lower.set_yticks([1e-5,1e-10,1e-15,1e-20,floor])
        lower.set_yticklabels(['$10^{-5}$','$10^{-10}$','$10^{-15}$','$10^{-20}$','Unresolved'])
        lower.yaxis.set_minor_locator(NullLocator())
        lower.axhspan(floor/6,floor,color='#EEEEEE',zorder=0)
        lower.axhline(floor,color='#888888',lw=.5,ls=':')
        lower.axhline(rate_ref,color='#333333',lw=1.,ls='--')
        lower.annotate('200k-step timescale',xy=(.98,rate_ref),
                       xycoords=('axes fraction','data'),xytext=(0,3),textcoords='offset points',
                       ha='right',va='bottom',fontsize=7)
    axes[0,0].set_ylabel(r'$\gamma_i$ distribution (98% interval)')
    axes[1,0].set_ylabel(r'90th-percentile rate $\rho_i$')
    fig.legend([Line2D([],[],color=COLORS[lr],lw=1.6) for lr in [.01,.001,.0001]],
               ['LR = 0.01','LR = 0.001','LR = 0.0001'],loc='lower center',
               bbox_to_anchor=(.5,.065),ncol=3,frameon=False,columnspacing=3.)
    fig.text(.5,.032,'Each shade is one function; colors match across rows. 90% of modes lie below each spectral curve.',
             ha='center',fontsize=7,color='#444444')
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'scale_and_spectrum.{ext}',dpi=240)
    plt.close(fig)
    assert plotted==216
    maximum=max(float(r['quantiles'][-1,int(np.flatnonzero(r['percentiles']==90)[0])]) for r in records.values())
    manifest={'runs':216,'functions':names,'checkpoints_per_run':409,
              'scale_display_sha256':hashlib.sha256(scale_path.read_bytes()).hexdigest(),
              'spectrum_sha256':source_hashes,'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'quill_gamma_reference':gamma_ref,'spectral_percentile':90,
              'rate_reference':rate_ref,'rate_reference_steps':horizon,
              'rate_reference_definition':'Residual component reduced by exp(-1) under frozen-readout GD with eta=1/(2*mu_max)',
              'resolution_floor':floor,'largest_final_p90':maximum,
              'line_counts_per_panel':72,'aggregation':'No across-function aggregation in either row; upper median labels summarize the 72 final neuron medians.'}
    (OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'caption.md').write_text('''# Scale acquisition and readout rates across 24 functions

Each column uses a different initialization scale; colors identify the joint-training learning rate, and matching shades identify functions in both rows. All 216 runs use 196 tanh neurons and 200,000 full-batch GD updates.

**Top:** median absolute neuron slope and central 98% interval for each run. Dotted lines mark the 1st and 99th percentiles. The QUILL reference is gamma = 18.375. Each median annotation summarizes the 72 final per-run medians in its panel.

**Bottom:** the 90th percentile of the normalized readout rates, rho_i = sigma_i^2 / sigma_max^2, computed from the actual hidden geometry at each saved checkpoint. Each curve is one run, with no pooling across functions. Rates below the numerical resolution (5.17e-26) are displayed on the bottom line. These are spectral percentiles, not confidence intervals or residual-energy percentiles.

The dashed 200k-step reference is rho = 2(1 - exp(-1/200000)), approximately 1e-5. At the theorem's reference frozen-readout GD step size, eta = 1/(2 mu_max), this rate takes 200,000 updates to reduce a residual component to exp(-1), or about 37%, of its starting magnitude. This reference is separate from the joint-training learning rates in the legend.

At the final checkpoint, every run's 90th-percentile rate is below 2.8e-9. Training keeps scales small while slow readout directions persist; the spectral curves alone do not measure how much residual error occupies those directions.
''')
    print(OUT/'scale_and_spectrum.png')

if __name__=='__main__':main()
