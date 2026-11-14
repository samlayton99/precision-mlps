"""Publication-style 3x4 diagnostics and 1x2 supplementary scaling panel."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter,LogLocator,FixedLocator
from experiments.expK06_hard_analytic_suite.common import OUT,WIDTHS,NS,save
from experiments.expK06_hard_analytic_suite.targets import MAIN,EXTRA,LABELS,NAMES as TARGET_NAMES
from experiments.expK06_hard_analytic_suite.add_gamma_bump import ADDED_NAMES,ADDED_LABELS,ADDED_DESCRIPTIONS

SUPPLEMENT_NAMES=[('inverse_sqrt_multiscale' if name=='chirp' else name) for name in TARGET_NAMES]+ADDED_NAMES
LABELS={**LABELS,**ADDED_LABELS,'inverse_sqrt_multiscale':r'$1/\sqrt{x}$'}

METHODS=['Adam','NPU','QUILL']
FUNCTION_COLORS=dict(zip(SUPPLEMENT_NAMES,plt.colormaps['tab20'].colors))
COLORS={'Adam':'#90cf72','NPU':'#1f77b4','QUILL':'#d62728'}
NAMES={'Adam':'Adam','NPU':'Gated NPU','QUILL':'QUILL'}
STYLES={method:'-' for method in METHODS}
FIT_LW=.5
RESIDUAL_LW=1.05
TARGET_LW=2.4
SCALING_LW=2.6
SCALING_MS=4.5
SCALING_MEW=1.2
FUNCTION_FONT=11
LABEL_FONT=10
TICK_FONT=8
# Figures contain functions, axes and keys only; protocol prose belongs in captions.


def style(ax):
    for side in ['top','right']:ax.spines[side].set_visible(False)
    for side in ['left','bottom']:ax.spines[side].set_linewidth(.85)
    ax.tick_params(which='major',labelsize=TICK_FONT,length=3,width=.75,pad=2)
    ax.tick_params(which='minor',length=1.3,width=.35)
    ax.set_axisbelow(True)


def curve(rows,name,method):
    r=sorted([r for r in rows if r['target']==name and r['method']==method and r['width'] in WIDTHS],key=lambda r:r['width'])
    assert [r['width'] for r in r]==WIDTHS
    return r


def scaling_axes(ax):
    ax.set(xscale='log',yscale='log',xlim=(95,2300),ylim=(1e-16,2))
    ax.set_xticks([100,300,1000,2000],['100','300','1k','2k'])
    ax.xaxis.set_minor_locator(FixedLocator([200,500]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([1,1e-4,1e-8,1e-12,1e-16])
    ax.yaxis.set_minor_locator(LogLocator(base=10,subs=(2,5),numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis='y',color='#dddddd',lw=.45)
    style(ax)


def main_panel(rows,preview=False,selected=None):
    fig,axs=plt.subplots(3,4,figsize=(13.3,7.4))
    fig.subplots_adjust(left=.075,right=.985,bottom=.105,top=.895,wspace=.20,hspace=.32)
    fig.suptitle('Preview' if preview else 'Hard Function Approximation',x=.53,y=.985,fontsize=14,fontweight='bold')
    if selected is None:selected=json.loads((OUT/'selected_fits.json').read_text())
    for j,name in enumerate(MAIN):
        ax=axs[0,j]
        final={method:next(r for r in selected if r['target']==name and r['method']==method) for method in METHODS}
        values={method:np.load(ROOT/r['predictions']) for method,r in final.items()}
        x=values['QUILL']['x'];truth=values['QUILL']['truth']
        ax.plot(x,truth,color='black',lw=TARGET_LW,linestyle='-',label='Target',zorder=2)
        for order,method in enumerate(METHODS,start=3):
            d=values[method]
            ax.plot(d['x'],d['prediction'],color=COLORS[method],lw=FIT_LW,label=NAMES[method],zorder=order)
        # A very narrow NPU excursion can dominate autoscaling. Use the target
        # range for this shape view; retain all values in the full residual view.
        lo,hi=float(truth.min()),float(truth.max());margin=.08*max(hi-lo,1e-12)
        ax.set_ylim(lo-margin,hi+margin)
        ax.set_title(LABELS[name],fontsize=FUNCTION_FONT,fontweight='bold',math_fontfamily='custom',pad=10)
        ax.set_xlim(-1,1);ax.set_xticks([-1,0,1]);style(ax)
        ax=axs[1,j]
        ax.axhline(0,color='black',lw=1.15,zorder=2)
        for order,method in enumerate(METHODS,start=3):
            d=values[method];ax.plot(d['x'],d['normalized_residual'],color=COLORS[method],lw=RESIDUAL_LW,zorder=order)
        ax.set(xlim=(-1,1),yscale='symlog',xlabel='$x$')
        ax.set_yscale('symlog',linthresh=1e-16,linscale=.6)
        peak=max(float(np.max(abs(values[method]['normalized_residual']))) for method in METHODS)
        bound=max(1,10**np.ceil(np.log10(max(peak,1e-14))))
        ax.set_ylim(-bound,bound)
        ticks=[-1,-1e-4,-1e-8,-1e-12,0,1e-12,1e-8,1e-4,1]
        if bound>1:ticks=[-bound,*ticks[1:-1],bound]
        ax.set_yticks(ticks)
        ax.set_xticks([-1,0,1]);ax.grid(axis='y',color='#e2e2e2',lw=.35);style(ax)
        ax.tick_params(labelsize=TICK_FONT)
        ax.xaxis.label.set_fontsize(LABEL_FONT)
        ax.xaxis.labelpad=1.5
        ax=axs[2,j]
        for order,method in enumerate(METHODS,start=3):
            r=curve(rows,name,method)
            ax.plot([a['width'] for a in r],[a['relative_l2'] for a in r],color=COLORS[method],
                    ls=STYLES[method],lw=SCALING_LW,marker='o',ms=SCALING_MS,mfc='white',mew=SCALING_MEW,zorder=order)
        scaling_axes(ax);ax.set_xlabel('Hidden width $W$',fontsize=LABEL_FONT,labelpad=3)
        for i in range(3):
            if j>0 and i==2:axs[i,j].tick_params(labelleft=False)
    axs[0,0].set_ylabel('Function value',fontsize=LABEL_FONT,fontweight='bold',labelpad=6)
    axs[1,0].set_ylabel('Signed residual / RMS',fontsize=LABEL_FONT,fontweight='bold',labelpad=6)
    axs[2,0].set_ylabel(r'Relative $L^2$ error',fontsize=LABEL_FONT,fontweight='bold',labelpad=6)
    handles=[Line2D([],[],color='black',lw=TARGET_LW,ls='-',label='Target')]+[
        Line2D([],[],color=COLORS[m],lw=RESIDUAL_LW,label=NAMES[m]) for m in METHODS]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.018),ncol=4,frameon=False,fontsize=9,columnspacing=2)
    for ext in ['png','pdf']:fig.savefig(OUT/f'hard_functions_3x4.{ext}',dpi=400,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def supplementary(rows):
    fig,(left,right)=plt.subplots(1,2,figsize=(13.3,5.9),gridspec_kw={'width_ratios':[1.2,1.]})
    fig.subplots_adjust(left=.025,right=.985,bottom=.14,top=.91,wspace=.14)
    fig.suptitle('QUILL Width Scaling',x=.51,y=.985,fontsize=14,fontweight='bold')
    left.set_axis_off();left.set_xlim(0,1);left.set_ylim(0,1)
    descriptions={
        'airy':'Airy function','kepler':'Kepler inverse',
        'elliptic_k':'Elliptic integral, first kind','chirped_division':'Oscillatory quotient',
        'erf':'Error function','chirp':'Quadratic chirp','localized':'Localized oscillation',
        'inverse_sqrt_multiscale':r'Inverse square root; $x\in[0.0001,1]$',
        'dawson':'Dawson integral','bessel0':'Bessel function, order 0','bessel6':'Bessel function, order 6',
        'sine_integral':'Sine integral','voigt':'Voigt profile','kummer':'Kummer function',
        'elliptic_e':'Elliptic integral, second kind','fresnel':'Fresnel sine integral',
        'log_ratio':'Logarithmic quotient',
        **ADDED_DESCRIPTIONS,
    }
    function_colors=FUNCTION_COLORS
    for i,name in enumerate(SUPPLEMENT_NAMES):
        column,row=divmod(i,9)
        y=1-(row+.5)/9;c=function_colors[name];offset=.51*column
        left.plot([offset+.005,offset+.055],[y+.015,y+.015],color=c,lw=SCALING_LW,solid_capstyle='round')
        left.text(offset+.285,y+.015,LABELS[name].replace(r'\frac12',r'\frac{1}{2}'),
                  ha='center',va='center',fontsize=FUNCTION_FONT,fontweight='bold',math_fontfamily='custom')
        left.text(offset+.285,y-.020,descriptions[name],ha='center',va='top',
                  fontsize=7.5,color='#555555')
        r=sorted([a for a in rows if a['target']==name and a['method']=='QUILL'],key=lambda a:a['width'])
        right.plot([a['width'] for a in r],[a['relative_l2'] for a in r],color=c,lw=SCALING_LW,
                   marker='o',ms=SCALING_MS,mfc='white',mew=SCALING_MEW)
    scaling_axes(right)
    right.set_xlabel('Hidden width $W$',fontsize=LABEL_FONT,labelpad=6)
    right.set_ylabel(r'Relative $L^2$ error',fontsize=LABEL_FONT,fontweight='bold',labelpad=5)
    right.tick_params(labelsize=TICK_FONT)
    right.set_xscale('linear')
    right.set_xlim(0,max(r['width'] for r in rows if r['method']=='QUILL' and r['target'] in SUPPLEMENT_NAMES)*1.05)
    right.set_xticks([0,1000,2000,3000,4000,5000],['0','1k','2k','3k','4k','5k'])
    right.set_ylim(1e-16,1e4)
    right.set_yticks([1e4,1,1e-4,1e-8,1e-12,1e-16])
    right.xaxis.set_minor_locator(FixedLocator([]))
    for ext in ['png','pdf']:fig.savefig(OUT/f'additional_functions_1x2.{ext}',dpi=400,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'mathtext.fontset':'dejavusans',
                         'mathtext.rm':'DejaVu Sans:bold','mathtext.it':'DejaVu Sans:italic:bold',
                         'mathtext.bf':'DejaVu Sans:bold','mathtext.cal':'DejaVu Sans'})
    updated=OUT/'npu_adam_lbfgs'
    result_dir=updated if (updated/'verified_summary.json').exists() else OUT
    rows=json.loads((result_dir/'verified_summary.json').read_text())
    selected=json.loads((result_dir/'selected_fits.json').read_text())
    rows+=json.loads((OUT/'quill_extension_verified.json').read_text())
    rows+=json.loads((OUT/'inverse_sqrt_multiscale_curve.json').read_text())
    assert sum(r['method']!='QUILL' for r in rows)==48
    main_panel(rows,selected=selected);supplementary(rows)
    paths=[result_dir/'verified_summary.json',OUT/'quill_extension_verified.json',OUT/'inverse_sqrt_multiscale_curve.json',OUT/'inverse_sqrt_multiscale_import.json',result_dir/'selected_fits.json',result_dir/'verification.json',OUT/'specification_main50k.json',Path(__file__)]
    if result_dir==updated:paths.append(updated/'protocol.json')
    selection='best validation error over the six shared widths; widths documented outside figure'
    if (result_dir/'display_selection.json').exists():
        paths.append(result_dir/'display_selection.json')
        selection=json.loads((result_dir/'display_selection.json').read_text())['description']
    save(OUT/'figure_provenance.json',dict(sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
         main_targets=MAIN,supplementary_targets=SUPPLEMENT_NAMES,function_rows_selection=selection,smoothing=False,
         residual='(prediction-truth)/RMS(f); symmetric log with linear threshold 1e-16',
         supplementary='18 QUILL targets: 17 sample-fitted uniform networks and the earlier 15-block inverse-square-root direct construction with C17 extension; replaces chirp, retains erf; includes nonanalytic compact bump',
         style=dict(method_colors=COLORS,draw_order=['Target',*METHODS],target_linewidth=TARGET_LW,
                    fit_linewidth=FIT_LW,residual_linewidth=RESIDUAL_LW,zero_line_color='black',
                    scaling_linewidth=SCALING_LW,scaling_marker_size=SCALING_MS,scaling_marker_face='white',
                    function_font=FUNCTION_FONT,label_font=LABEL_FONT,tick_font=TICK_FONT,
                    supplementary_function_grid=[9,2],supplementary_xscale='linear',
                    supplementary_xticks=[0,1000,2000,3000,4000,5000],supplementary_ylim=[1e-16,1e4],function_colors=FUNCTION_COLORS,
                    function_value_limits='target min/max plus 8% of target range; excursions retained in residuals and scoring',
                    titles=['Hard Function Approximation','QUILL Width Scaling'],
                    descriptive_text='short function names beneath supplementary formulas only'),
         baselines=('MLP: original 50k Adam; NPU: 50k Adam followed by up to 1000 full-batch L-BFGS iterations; no readout refit or RealNPU'
                    if result_dir==updated else 'validation-best 50k-step Adam training; no polishing or extra seeds')))
    print(OUT/'hard_functions_3x4.png');print(OUT/'additional_functions_1x2.png')
if __name__=='__main__':main()
