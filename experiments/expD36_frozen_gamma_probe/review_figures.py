"""Plot five-target readout evidence for the collaborator note, without training.

Run from the repository root with ``python -m
experiments.expD36_frozen_gamma_probe.review_figures``. Only numerical data
and figure artifacts are written; the report is authored separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np


TARGETS = ['sine_mix_2_6_10', 'exp_sin_3pi', 'runge_25', 'quadratic', 'sine_2pi']
LABELS = ['Sine mixture', 'Exponential of sine', 'Runge', 'Quadratic', 'Single sine']
COLORS = ['#c65525', '#7557a5', '#278064', '#aa6d16', '#245c9f']
GAMMAS = [8, 12, 16, 64]
DEFAULT = Path('results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/'
               'full_sweep/refinements/gamma_factorized_kernel')


def collect(folder):
    sources = {}

    def read(name):
        raw = (folder/name).read_bytes()
        sources[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    summary, audit = read('summary.json'), read('interval_audit.json')
    assert summary['targets'] == TARGETS and summary['gammas'] == GAMMAS
    assert summary['width'] == 559 and summary['samples'] == 8193
    assert summary['epsilon'] == .01
    cases, ratios = [], []
    for ti, target in enumerate(TARGETS):
        rows = []
        for gamma in GAMMAS:
            dictionary = next(d for d in summary['dictionaries'] if d['gamma'] == gamma)
            comparison = next(c for c in summary['comparisons'] if c['gamma'] == gamma)
            bounds = comparison['selected']['combined'][ti]
            low, high = bounds['necessary'], bounds['sufficient']
            hit = dictionary['executed_hits'][ti]
            assert 0 < low <= high
            if hit < 0:
                assert (target, gamma, hit) == ('quadratic', 12, -1)
                assert low > 200000
            else:
                assert low <= hit <= high
            certified = target == TARGETS[0]
            if certified:
                check = next(a for a in audit['results'] if a['gamma'] == gamma)
                certificate = check['methods']['combined']
                assert check['target'] == target and check['eta'] == dictionary['eta']
                assert certificate['status'] == 'interval_certified_endpoints'
                assert (low, high) == (certificate['necessary'], certificate['sufficient'])
            row = dict(target=target, gamma=gamma, necessary=low, sufficient=high,
                       executed=hit if hit >= 0 else None,
                       censored_at=200000 if hit < 0 else None,
                       verification='interval_certified' if certified else 'fp64_checked')
            cases.append(row)
            rows.append(row)
        ratios.append(dict(target=target,
                           necessary=rows[0]['necessary']/rows[-1]['sufficient'],
                           sufficient=rows[0]['sufficient']/rows[-1]['necessary'],
                           executed=rows[0]['executed']/rows[-1]['executed']))
    assert sum(r['executed'] is not None for r in cases) == 19
    return dict(cases=cases, ratios=ratios, source_sha256=sources,
                plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                evidence_role='Archived training results; four primary intervals certified, sixteen controls numerically checked',
                gpu_hours=0)


def save(fig, output, name):
    fig.savefig(output/(name+'.png'), dpi=240)
    fig.savefig(output/(name+'.pdf'), metadata={'CreationDate': None})
    plt.close(fig)


def slowdown(data, output):
    fig, ax = plt.subplots(figsize=(7.2, 2.8), layout='constrained')
    order = [4, 2, 3, 1, 0]
    for y, ti in enumerate(order):
        row = data['ratios'][ti]
        low, high, hit = (row[k] for k in ['necessary', 'sufficient', 'executed'])
        mid = (low+high)/2
        ax.hlines(y, 1, hit, color=COLORS[ti], alpha=.2, lw=3)
        ax.errorbar(mid, y, xerr=[[mid-low], [high-mid]], fmt='none',
                    ecolor=COLORS[ti], capsize=6, elinewidth=2, zorder=3)
        ax.scatter(hit, y, s=34, color=COLORS[ti], zorder=4)
        ax.text(hit*1.17, y, f'{hit:.2f}'+r'$\times$', va='center', fontsize=9,
                color=COLORS[ti], fontweight='bold')
    ax.axvline(1, color='.5', ls=':', lw=1)
    ax.set(xscale='log', xlim=(.8, 2400), ylim=(4.7, -.6),
           xlabel=r'Updates at $\gamma=8$ / updates at $\gamma=64$')
    ax.set_yticks(range(5), [LABELS[i] for i in order])
    ax.set_xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis='x', alpha=.18)
    ax.set_title('Same gamma intervention, different target-dependent delays', loc='left', pad=11)
    ax.text(.99, .99, 'Dots: executed ratios\nWhiskers: predicted intervals',
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color='.3')
    save(fig, output, 'target_slowdown')


def all_cases(data, output):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.5), layout='constrained')
    for ti, ax in enumerate(axes.flat[:5]):
        rows = [r for r in data['cases'] if r['target'] == TARGETS[ti]]
        low = np.array([r['necessary'] for r in rows])
        high = np.array([r['sufficient'] for r in rows])
        mid = (low+high)/2
        ax.plot(GAMMAS, mid, color=COLORS[ti], lw=1.3)
        ax.errorbar(GAMMAS, mid, yerr=[mid-low, high-mid], fmt='none',
                    ecolor=COLORS[ti], capsize=4, elinewidth=1.5)
        for row in rows:
            if row['executed'] is not None:
                ax.scatter(row['gamma'], row['executed'], s=24, c='.12', zorder=4)
            else:
                ax.scatter(row['gamma'], row['censored_at'], marker='^',
                           facecolors='white', edgecolors='.15', s=34, zorder=4)
                ax.annotate('GD stopped\nat 200,000',
                            (row['gamma'], row['censored_at']), xytext=(17, 430000),
                            fontsize=6.5, ha='left', arrowprops=dict(arrowstyle='-', lw=.6))
        if ti == 0:
            ax.text(.03, .06, 'Interval-certified', transform=ax.transAxes,
                    fontsize=6.5, color=COLORS[ti])
        else:
            ax.text(.03, .06, 'Numerically checked', transform=ax.transAxes,
                    fontsize=6.5, color=COLORS[ti])
        ax.set(title=f'{chr(65+ti)}  {LABELS[ti]}', xlabel=r'Common slope $\gamma$',
               ylabel='Updates to 1% residual', xscale='log', yscale='log',
               xlim=(7, 75), ylim=(float(min(low))*.7, float(max(high))*2.5))
        ax.set_xticks(GAMMAS, [str(g) for g in GAMMAS])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        if ti == 4:
            ax.set_yticks([2000, 5000, 10000], ['2,000', '5,000', '10,000'])
        ax.grid(alpha=.18)

    ax = axes.flat[5]
    ax.axvline(0, color='.35', ls=':', lw=1)
    offsets = [-.24, -.08, .08, .24]
    markers = ['o', 's', '^', 'D']
    for ti, target in enumerate(TARGETS):
        for gi, gamma in enumerate(GAMMAS):
            row = next(r for r in data['cases'] if (r['target'], r['gamma']) == (target, gamma))
            if row['executed'] is None:
                continue
            low, high = [100*(row[k]/row['executed']-1) for k in ['necessary', 'sufficient']]
            mid = (low+high)/2
            ax.errorbar(mid, ti+offsets[gi], xerr=[[mid-low], [high-mid]],
                        fmt=markers[gi], color=COLORS[ti], ms=3, capsize=2, lw=1)
    ax.set(title='F  Prediction tightness', xscale='symlog', xlim=(-.16, .16),
           ylim=(4.6, -.6), xlabel='Interval endpoint / hit - 1 (%)')
    ax.set_xscale('symlog', linthresh=1e-5)
    ax.set_xticks([-.1, -.001, 0, .001, .1], ['-0.1', '-0.001', '0', '0.001', '0.1'])
    ax.tick_params(axis='x', labelsize=6.1)
    ax.set_yticks(range(5), ['Mixture', 'Exp. sine', 'Runge', 'Quadratic', 'Single sine'])
    ax.grid(axis='x', alpha=.18)
    legend = [Line2D([], [], marker=marker, color='.3', ls='none', markersize=4,
                     label=str(gamma)) for marker, gamma in zip(markers, GAMMAS)]
    ax.legend(handles=legend, title=r'$\gamma$', ncol=4, frameon=False,
              fontsize=5.8, title_fontsize=7, loc='upper center',
              bbox_to_anchor=(.5, -.23), handletextpad=.2, columnspacing=.65)
    fig.suptitle('Five targets: predicted acquisition versus executed readout GD', fontsize=11)
    fig.supxlabel('A-E: colored lines and whiskers are predictions; black dots are executed hits.\n'
                  'Axes use separate vertical ranges. F: all 19 executed hits; symmetric-log horizontal scale.',
                  fontsize=7)
    save(fig, output, 'target_acquisition')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.source
    data = collect(args.source)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.titlelocation': 'left',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    slowdown(data, output)
    all_cases(data, output)
    (output/'review_figure_data.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), cases=len(data['cases']),
                          executed=19, censored=1), indent=2))


if __name__ == '__main__':
    main()
