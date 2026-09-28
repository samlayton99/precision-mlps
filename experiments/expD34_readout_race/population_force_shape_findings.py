"""Sampled force-shape persistence in natural and finite-dilation GD archives.

State derivatives are effective-flow probes; actual archived paths follow GD.
Sampled maxima do not certify the future structural premises of Theorem 12.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np

from . import population_coverage as pc
from .population_reinforcement_summary import number, stats


def values(row):
    get = lambda key: number(row, 'reinforcement_'+key)
    omega, omega2 = get('force_weighted_second_moment'), get('force_weighted_fourth_moment')
    relax, other = get('omega_dot_redistribution_relaxation'), get('omega_dot_redistribution_geometry_compensation')
    transport, redistribution = get('omega_dot_transport'), get('omega_dot_redistribution')
    fraction = lambda a, b: abs(a)/(abs(a)+abs(b)) if abs(a)+abs(b) > 0 else np.nan
    return dict(I_F=get('hidden_force_energy_concentration'), K_F=omega2/omega**2 if omega > 0 else np.nan,
                Omega_F=omega, relaxation_fraction_of_absolute_redistribution=fraction(relax, other),
                transport_fraction_of_absolute_omega_change=fraction(transport, redistribution),
                omega_dot_transport=transport, omega_dot_redistribution=redistribution,
                omega_dot_redistribution_relaxation=relax,
                omega_dot_redistribution_geometry_compensation=other)


def run(args):
    rows, excluded, sources = [], defaultdict(int), {}
    for source in args.source:
        sources[str(source)] = pc.digest(source)
        with source.open() as stream:
            for original in csv.DictReader(stream):
                status = original.get('reinforcement_status', '')
                if status != 'finite':
                    excluded[status] += 1
                    continue
                row = dict(original, dataset=source.parent.name, **values(original))
                if not all(np.isfinite(row[k]) and row[k] > 0 for k in ('I_F', 'K_F', 'Omega_F')):
                    excluded['undefined_shape'] += 1
                    continue
                rows.append(row)
    paths = defaultdict(list)
    for row in rows:
        if row.get('role') != 'trajectory':
            continue
        if row.get('arm') != 'natural' and not row.get('scale'):
            excluded['non_natural_non_dilation_trajectory'] += 1
            continue
        key = tuple(row.get(k, '') for k in ('dataset', 'panel', 'target', 'seed', 'index', 'arm'))
        paths[key].append(row)
    endpoints = []
    for path in paths.values():
        path.sort(key=lambda r: number(r, 'horizon'))
        first, last = path[0], path[-1]
        identity = {k: first.get(k, '') for k in
                    ('dataset', 'panel', 'target', 'seed', 'index', 'arm', 'width', 'start', 'scale', 'reference', 'eta')}
        record = dict(identity, snapshots=len(path), first_horizon=number(first, 'horizon'),
                      last_horizon=number(last, 'horizon'), has_initial_snapshot=number(first, 'horizon') == 0)
        for metric in ('I_F', 'K_F', 'Omega_F'):
            record[metric+'_initial'] = first[metric]
            record[metric+'_final'] = last[metric]
            record[metric+'_sampled_max'] = max(r[metric] for r in path)
            record[metric+'_final_over_initial'] = last[metric]/first[metric]
            record[metric+'_sampled_max_over_initial'] = record[metric+'_sampled_max']/first[metric]
        for metric in ('relaxation_fraction_of_absolute_redistribution',
                       'transport_fraction_of_absolute_omega_change'):
            record[metric+'_initial'] = first[metric]
            record[metric+'_final'] = last[metric]
        endpoints.append(record)
    grouped = defaultdict(list)
    group_fields = ('dataset', 'panel', 'width', 'start', 'scale', 'reference', 'arm', 'eta', 'first_horizon', 'last_horizon')
    for row in endpoints:
        grouped[tuple(row[k] for k in group_fields)].append(row)
    group_facts = []
    for key, group in grouped.items():
        item = dict(zip(group_fields, key), trajectories=len(group), targets=sorted({r['target'] for r in group}),
                    seeds=sorted({r['seed'] for r in group}), all_have_initial_snapshot=all(r['has_initial_snapshot'] for r in group))
        metrics = [k for k in group[0] if k.startswith(('I_F_', 'K_F_', 'Omega_F_',
                    'relaxation_fraction_', 'transport_fraction_'))]
        item.update({metric: stats([r[metric] for r in group]) for metric in metrics})
        group_facts.append(item)
    state_groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k, '') for k in ('dataset', 'role', 'panel', 'width', 'start', 'scale', 'reference', 'arm', 'eta', 'horizon'))
        state_groups[key].append(row)
    state_facts = []
    for key, group in state_groups.items():
        item = dict(zip(('dataset', 'role', 'panel', 'width', 'start', 'scale', 'reference', 'arm', 'eta', 'horizon'), key), states=len(group))
        item.update({metric: stats([r[metric] for r in group]) for metric in values(group[0])})
        state_facts.append(item)
    args.output.mkdir(parents=True, exist_ok=False)
    pc.write_csv(args.output/'trajectories.csv', endpoints)
    facts = dict(trajectory_groups=group_facts, state_groups=state_facts, excluded=dict(excluded),
                 sources=sources, helper_sha256=pc.digest(__file__),
                 interpretation='Sampled GD states; derivatives probe effective flow. No continuous-time or future shape certification.',
                 first_endpoint_rule='First and last retained valid samples; has_initial_snapshot explicitly marks horizon zero.',
                 mechanism_fractions='Ratios of absolute instantaneous terms, not cumulative attribution or signed cancellation.')
    (args.output/'facts.json').write_text(json.dumps(pc.clean(facts), indent=2)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    normalizer = LogNorm(vmin=1, vmax=100)
    for ax, reference in zip(axes, ('natural', 'primary', 'inverse')):
        for group in group_facts:
            if not group['all_have_initial_snapshot'] or group['trajectories'] == 0:
                continue
            if reference == 'natural':
                if group['arm'] != 'natural':
                    continue
                color = plt.get_cmap('tab10')(({177: 0, 705: 1, 1409: 2}.get(int(float(group['width'])), 3)))
            else:
                if group['reference'] != reference or group['last_horizon'] != 20000 or float(group['eta']) != .002:
                    continue
                color = plt.get_cmap('viridis')(normalizer(float(group['scale'])))
            marker = {177: 'o', 705: 's', 1409: '^'}.get(int(float(group['width'])), 'D')
            first = [group['I_F_initial']['median'], group['K_F_initial']['median']]
            last = [group['I_F_final']['median'], group['K_F_final']['median']]
            ax.plot([first[0], last[0]], [first[1], last[1]], color=color, alpha=.75, lw=1.)
            ax.scatter(*first, marker=marker, facecolors='none', edgecolors=[color], s=45)
            ax.scatter(*last, marker=marker, color=color, s=28)
        ax.set(xscale='log', yscale='log', xlabel='Force concentration I_F', ylabel='Force-weighted size spread K_F',
               title='Natural GD continuations' if reference == 'natural' else f'{reference.capitalize()} readout repair · 20k steps')
    handles = [Line2D([], [], marker=marker, color='gray', ls='', label=f'W={width}')
               for width, marker in ((177, 'o'), (705, 's'), (1409, '^'))]
    axes[0].legend(handles=handles, fontsize=9)
    fig.colorbar(plt.cm.ScalarMappable(norm=normalizer, cmap='viridis'), ax=list(axes[1:]), label='Imposed slope multiplier')
    fig.suptitle('Population shape: hollow = initial, filled = final group medians\n'
                 'Groups retain width, restart age, multiplier, reference and run panel; sparse samples are not a persistence proof', fontsize=11)
    fig.savefig(args.output/'force_shape_evolution.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
