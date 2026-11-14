"""Two-panel QUILL width scaling with explicitly fixed-width NPU references."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/precisionmlps-mpl')
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expK05_singular_function_comparison.run import OUT, SIZES, save
from experiments.expK05_singular_function_comparison.construction import TARGETS


def main():
    paths = [OUT/'scaling.json', OUT/'verification.json']
    rows, verification = [json.loads(p.read_text()) for p in paths]
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8, 'mathtext.fontset':'dejavusans',
                         'mathtext.rm':'DejaVu Sans:bold', 'mathtext.it':'DejaVu Sans:italic:bold',
                         'mathtext.bf':'DejaVu Sans:bold', 'mathtext.cal':'DejaVu Sans'})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.65), sharey=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.22, top=.79, wspace=.14)
    fig.suptitle('QUILL and trained NPU', x=.535, y=.97, fontsize=13, fontweight='bold')
    fig.text(.535, .88, r'$x\in[0.0001,1]$', ha='center', fontsize=9)
    titles = [r'$f(x)=1/(\sqrt{x}+x)$', r'$f(x)=e^{-x}/\sqrt{x}$']
    colors = plt.colormaps['viridis']
    displayed = {}
    for ax, name, title in zip(axes, TARGETS, titles):
        curve = sorted([r for r in rows if r['target']==name], key=lambda r:r['N'])
        assert [r['N'] for r in curve] == SIZES
        ax.plot([r['width'] for r in curve], [r['relative_l2'] for r in curve],
                color=colors(.58), lw=1.45, marker='o', ms=3.6, mfc='white', mew=.9, label='QUILL', zorder=4)
        v = verification[name]
        baseline = v['npu_reference']['quadrature']['256']
        ax.axhline(baseline, color=colors(.12), lw=1.35, linestyle='--',
                   label='NPU reference (width 32; Adam + L-BFGS)', zorder=3)
        ax.set(xscale='log', yscale='log', xlim=(990,5055), ylim=(1e-15, 1e4), xlabel='QUILL hidden width')
        ax.set_xticks([1000,2000,4000], ['1k','2k','4k'])
        ax.xaxis.set_minor_locator(FixedLocator([1500,3000,5000]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks([1e3,1,1e-3,1e-6,1e-9,1e-12,1e-15])
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2.,5.), numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontsize=10.5, fontweight='bold', math_fontfamily='custom', pad=8)
        val = v['quadrature']['256']
        ax.annotate(r'$\approx '+f'{val/1e-14:.1f}'+r'\times10^{-14}$', (4815,val),
                    xytext=(-1,7), textcoords='offset points', ha='right', va='bottom', fontsize=8, color='#444444')
        for side in ['top','right']: ax.spines[side].set_visible(False)
        for side in ['left','bottom']: ax.spines[side].set_linewidth(.9)
        ax.tick_params(which='major', labelsize=8, length=3, width=.85, pad=2)
        ax.tick_params(which='minor', length=1.4, width=.4)
        ax.xaxis.labelpad=3
        ax.grid(axis='y', which='major', color='#D8D8D8', lw=.45)
        ax.set_axisbelow(True)
        displayed[name] = dict(quill_curve=curve, npu_reference=v['npu_reference'])
    axes[0].set_ylabel(r'Relative $L^2$ error', fontsize=9, labelpad=4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.535,.035), ncol=2,
               frameon=False, fontsize=8, handlelength=2., columnspacing=2.)
    for suffix in ['png','pdf']:
        fig.savefig(OUT/f'singular_functions_1x2.{suffix}', dpi=450, bbox_inches='tight', pad_inches=.10)
    plt.close(fig)
    save(OUT/'figure_provenance.json', dict(
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        displayed=displayed, smoothing=False,
        note='NPU horizontal lines are saved fixed-width references, not capacity-matched width sweeps.'))
    print(OUT/'singular_functions_1x2.png')


if __name__ == '__main__':
    main()
