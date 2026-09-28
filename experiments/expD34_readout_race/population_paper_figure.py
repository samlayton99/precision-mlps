"""Compact manuscript figure from saved scalars and 403 KiB endpoint arrays.

Run on Modal. No training, large archives, or report generation. The fixed
factor-four feedback allowance comes from the existing sensitivity family;
all bounds here are conditional effective-flow evaluations, not certificates.
"""
import argparse
import ast
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .population_feedback_findings import read_csv


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    folder = root/'feedback_flow_100k'
    rows = read_csv(folder/'states.csv')
    facts = json.loads((folder/'facts.json').read_text())
    targets = [r['target'] for r in facts['cases']]
    labels = ['Degree five', 'Mixed sine', 'Gaussian', 'Bump', 'Step', 'Kink']
    assert targets == ['moment5', 'mixed_sine', 'gauss_left', 'bump_right', 'step_right', 'kink_abs']
    with np.load(folder/'endpoints.npz', allow_pickle=False) as data:
        initial = data['p0'].copy()
        endpoints = dict(zip([ast.literal_eval(s) for s in data['labels']], data['endpoints']))
    width, h, eta, alpha = 705, 2/512, .002, 4.
    assert initial.shape == (6, 3*width+1)
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                         'legend.fontsize': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    premise, top = plt.subplots(1, 2, figsize=(5.5, 2.5))
    consequences, bottom = plt.subplots(1, 2, figsize=(5.5, 2.8))
    axes = np.array([top, bottom])
    colors = plt.get_cmap('tab10').colors
    summaries = []
    for j, (target, label) in enumerate(zip(targets, labels)):
        paths = {}
        for kind, dt in (('effective', .01), ('gd', .002)):
            path = sorted([r for r in rows if r['target'] == target and r['kind'] == kind
                           and r['dt'] == dt], key=lambda r: r['time'])
            assert len(path) == 201
            paths[kind] = path
        eff, gd = paths['effective'], paths['gd']
        t = np.array([r['time'] for r in eff])
        d0, f0, y0, ynorm = (eff[0][key] for key in ('rate', 'f', 'Y', 'target_norm'))
        allowance = alpha*d0*t
        ratio = np.array([r['B'] for r in eff])[1:]/allowance[1:]
        force_bound = np.exp(allowance)*f0
        floor = np.sqrt(np.maximum(0., y0*y0-2*t*force_bound**2))/ynorm
        move_bound = h/np.sqrt(width)*t[-1]*force_bound[-1]
        motion = {}
        for kind, dt, marker in (('gd', .002, 'x'), ('effective', .01, 'o')):
            end = endpoints[(target, kind, dt)]
            motion[kind] = h*np.linalg.norm(np.abs(end[:width])-np.abs(initial[j, :width]))/np.sqrt(width)
            axes[1,0].scatter(j, motion[kind], marker=marker, color=colors[j], s=35,
                              facecolors='none' if marker == 'o' else colors[j])
            axes[1,1].scatter(j, 100*paths[kind][-1]['relative_error'], marker=marker,
                              color=colors[j], s=35, facecolors='none' if marker == 'o' else colors[j])
        # Verification of the plotted conditional consequences on retained
        # effective-flow samples. GD is a separate numerical comparison.
        assert max(ratio) <= 1
        assert np.all(np.array([r['f'] for r in eff]) <= force_bound*(1+1e-10))
        assert np.all(floor <= np.array([r['relative_error'] for r in eff])+1e-12)
        assert motion['effective'] <= move_bound
        axes[0,0].plot(t[1:]/eta/1000, ratio, color=colors[j], label=label)
        axes[0,1].plot(t/eta/1000, [r['f']/f0 for r in gd], color=colors[j])
        axes[0,1].plot(t/eta/1000, [r['f']/f0 for r in eff], color=colors[j], ls=':', lw=1)
        axes[1,0].scatter(j, move_bound, color=colors[j], marker='_', s=90, linewidths=1.8)
        axes[1,1].scatter(j, 100*floor[-1], color=colors[j], marker='_', s=90, linewidths=1.8)
        summaries.append(dict(target=target, max_feedback_allowance_fraction=float(max(ratio)),
            baseline_rate=d0, terminal_allowance=float(allowance[-1]), initial_force=f0,
            final_force_ratio=eff[-1]['f']/f0, effective_relative_error=eff[-1]['relative_error'],
            gd_relative_error=gd[-1]['relative_error'], simplified_relative_floor=float(floor[-1]),
            effective_rms_lambda_motion=float(motion['effective']), gd_rms_lambda_motion=float(motion['gd']),
            conditional_rms_lambda_motion_bound=float(move_bound),
            initial_max_lambda=float(h*max(abs(initial[j,:width]))),
            conditional_ever_fraction_for_increment0125=float(min(1., (move_bound/.125)**2))))
    axes[0,0].axhline(1, color='black', lw=.8, ls='--')
    axes[0,0].set(title='(a) Feedback allowance', ylabel='Accumulated feedback / allowance', ylim=(0,1.08))
    axes[0,1].set(title='(b) Effective force evolution', ylabel='Effective force / restart force')
    for ax in axes[0]:
        ax.set_xlabel('Additional updates (thousands)')
        ax.set_xlim(0,100)
    axes[1,0].axhline(.125, color='black', ls='--', lw=.8)
    axes[1,0].text(.02,.86,'Reference increment: 0.125',transform=axes[1,0].transAxes,va='top',fontsize=7)
    axes[1,0].set(title='(a) RMS slope displacement', ylabel='RMS change in relative slope', yscale='log', ylim=(1e-9,.5))
    axes[1,1].axhline(1, color='black', ls='--', lw=.8)
    axes[1,1].text(.02,.045,'1% accuracy requirement',transform=axes[1,1].transAxes,fontsize=7)
    axes[1,1].set(title='(b) Conditional output floor', ylabel='Relative output error (%)', ylim=(0,100))
    for ax in axes[1]:
        ax.set_xticks(range(6), labels, rotation=35, ha='right', fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    handles, legend_labels = axes[0,0].get_legend_handles_labels()
    premise.legend(handles, legend_labels, loc='upper center', ncol=3, frameon=False,
                   columnspacing=1.3, bbox_to_anchor=(.52,1.01))
    premise.subplots_adjust(left=.12,right=.965,top=.77,bottom=.20,wspace=.50)
    consequences.text(.52,.017,'× GD   ○ effective flow   — conditional effective-flow bound',ha='center',fontsize=7)
    consequences.subplots_adjust(left=.105,right=.985,top=.90,bottom=.30,wspace=.44)
    for fig, name in ((premise, 'feedback_force_check'), (consequences, 'population_error_check')):
        for suffix in ('png', 'svg', 'pdf'):
            fig.savefig(output/f'{name}.{suffix}', dpi=300)
        svg_path = output/f'{name}.svg'
        svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
        plt.close(fig)
    with (output/'paper_summary.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    record=dict(scope='Conditional effective-flow evaluations; GD comparison is empirical',
        width=width,h=h,reference_learning_rate=eta,additional_updates=100000,seed=30,restart_age=20000,
        feedback_allowance_factor=alpha,targets=summaries,
        minimum_relative_floor=min(r['simplified_relative_floor'] for r in summaries),
        maximum_feedback_allowance_fraction=max(r['max_feedback_allowance_fraction'] for r in summaries),
        maximum_population_motion_bound=max(r['conditional_rms_lambda_motion_bound'] for r in summaries),
        maximum_ever_fraction_bound=max(r['conditional_ever_fraction_for_increment0125'] for r in summaries))
    (output/'paper_facts.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args(); run(args.root,args.output)
