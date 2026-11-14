"""Original scale figure and separate spectral percentiles with learning rate as line style."""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator
from plot_combined import ROOT, SCALE, SPECTRA, apply_style, style_axis

OUT=SPECTRA/'separate_figures'
STYLES={.01:'solid',.001:(0,(5,3)),.0001:(0,(1,3))}
TITLES=['a) Default Initialization','b) 2x Initialization','c) 4x Initialization']

def setup(title,spectral=False):
    fig,axes=plt.subplots(1,3,figsize=(9.4,3.85 if spectral else 3.3),sharex=True,sharey=True)
    fig.subplots_adjust(left=.105,right=.985,bottom=.34 if spectral else .24,
                        top=.77,wspace=.20)
    fig.suptitle(title,fontsize=12,fontweight='bold',y=.98)
    for ax,title in zip(axes,TITLES):
        style_axis(ax);ax.set_title(title,fontweight='bold',pad=8)
        ax.set(xlim=(0,200000),xlabel='GD step')
        ax.set_xticks([0,50000,100000,150000,200000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k' if x else '0'))
        ax.xaxis.set_minor_locator(NullLocator())
    fig.legend([Line2D([],[],color='#333333',lw=.8,ls=ls) for ls in STYLES.values()],
               [f'LR = {lr:g}' for lr in STYLES],ncol=3,frameon=False,
               loc='lower center',bbox_to_anchor=(.5,.02),handlelength=3.,columnspacing=3.)
    return fig,axes

def save(fig,name):
    if name=='readout_spectrum' and '--scale-only' in sys.argv:
        plt.close(fig)
        return
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=240)
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    source=SCALE/'scale_acquisition_displayed.json'
    data=json.loads(source.read_text());assert not data['synthetic'] and not data['partial']
    curves=data['individual_trajectories']
    provenance=json.loads((SCALE/'figure_provenance.json').read_text())
    names=provenance['functions'];assert len(names)==24
    records={};hashes={}
    for path in sorted((SPECTRA/'runs').glob('*/spectra.npz')):
        meta=json.loads((path.parent/'metadata.json').read_text())
        assert meta['source_sha256']==provenance['source_sha256'][meta['source']]
        key=f"init{meta['initialization_multiplier']:g}/lr{meta['learning_rate']:.0e}/{meta['target']}"
        with np.load(path) as z:records[key]={k:z[k].copy() for k in ('steps','percentiles','quantiles','resolution_floor')}
        np.testing.assert_array_equal(records[key]['steps'],curves[key]['step'])
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(records)==set(curves) and len(records)==216
    apply_style()
    reference=provenance['config']['quill_reference']
    gamma_ref=reference['lambda']*reference['interior_intervals']/2
    # Preserve the original approved scale plot exactly, including its color legend.
    for ext in ['png','pdf']:
        shutil.copy2(SCALE/f'scale_acquisition.{ext}',OUT/f'scale_acquisition.{ext}')

    floor=float(next(iter(records.values()))['resolution_floor'])
    percentiles=np.arange(10,100,10)
    colors=plt.colormaps['viridis'](percentiles/100)
    rate_ref=float(-2*np.expm1(-1/200000))
    fig,axes=setup('Readout Spectrum: 24 functions',True)
    lines=0
    for ax,init in zip(axes,[1,2,4]):
        # Draw one color at a time; every function remains a separate trajectory.
        for percentile,color in zip(percentiles,colors):
            for lr,ls in STYLES.items():
                for name in names:
                    r=records[f'init{init}/lr{lr:.0e}/{name}']
                    index=int(np.flatnonzero(r['percentiles']==percentile)[0])
                    ax.plot(r['steps'],np.maximum(r['quantiles'][:,index],floor),color=color,
                            lw=.42,alpha=.65,ls=ls,dash_capstyle='round');lines+=1
        ax.set(yscale='log',ylim=(floor/6,1e-3))
        ax.set_yticks([1e-5,1e-10,1e-15,1e-20,floor])
        ax.set_yticklabels(['$10^{-5}$','$10^{-10}$','$10^{-15}$','$10^{-20}$','Unresolved'])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.axhspan(floor/6,floor,color='#EEEEEE',zorder=0)
        ax.axhline(floor,color='#888888',lw=.5,ls=':')
        ax.axhline(rate_ref,color='#333333',lw=.9,ls='--')
        ax.annotate('200k-step timescale',xy=(.98,rate_ref),xycoords=('axes fraction','data'),
                    xytext=(0,3),textcoords='offset points',ha='right',va='bottom',fontsize=7)
    assert lines==216*9
    axes[0].set_ylabel(r'Normalized rate $\rho_i$')
    cax=fig.add_axes([.25,.17,.5,.025])
    cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(5,100,10),len(colors))
    cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,
                    orientation='horizontal',ticks=percentiles)
    cb.set_label('Percentile',fontsize=7,labelpad=2);cb.ax.tick_params(labelsize=6.5,length=2,pad=2)
    cb.outline.set_linewidth(.4)
    save(fig,'readout_spectrum')
    manifest={'functions':names,'runs':216,'spectral_percentiles':percentiles.tolist(),
              'spectral_lines_per_panel':648,'aggregation':'None: every function, learning rate, and percentile is drawn separately.',
              'spectrum_learning_rate_styles':{'0.01':'solid','0.001':'dashed','0.0001':'dotted'},
              'spectrum_colors':'viridis by percentile','scale_colors':'Original blue/orange/purple learning-rate families with function shades',
              'resolution_floor':floor,'rate_reference':rate_ref,
              'scale_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'spectrum_source_sha256':hashes,'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'caption.md').write_text('''# Scale acquisition and readout spectrum: 24 functions

Both figures use three initialization panels. Every function is plotted individually.

**Scale acquisition:** restored exactly from the approved original figure. Blue, orange, and purple identify learning rates 0.01, 0.001, and 0.0001; shades distinguish functions within each family. Solid lines show per-run median absolute neuron slopes, dotted boundaries show the central 98% interval, and translucent fills show the interval. QUILL's reference slope is 18.375. Median annotations summarize the 72 final per-run medians in each panel.

**Readout spectrum:** learning rates 0.01, 0.001, and 0.0001 use solid, dashed, and dotted lines, respectively. Viridis colors identify the 10th, 20th, ..., 90th spectral percentiles (50th is the median). Each panel contains 24 functions × 3 learning rates × 9 percentiles. There is no aggregation across functions. Rates below the numerical resolution (5.17e-26) overlap at the bottom line; they are not assigned that value in the saved data. These percentiles describe the 197 readout modes, not confidence intervals or residual-energy distributions.

The 200k-step reference is rho = 2(1 - exp(-1/200000)), approximately 1e-5. Under frozen-readout GD with eta = 1/(2 mu_max), a residual component at this rate takes 200,000 updates to decrease to about 37% of its starting magnitude. This reference is separate from the joint-training learning rates indicated by line style.
''')
    print(OUT)

if __name__=='__main__':main()
