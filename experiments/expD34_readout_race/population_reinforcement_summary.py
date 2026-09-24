"""Facts and plots for retrospective population force-reinforcement diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import population_coverage as pc


GROUPS = ((705, 20000), (1409, 20000), (177, 600000))
TERMS = ('residual_relaxation_rate', 'generated_geometry_rate', 'target_geometry_rate',
         'generated_compensation_rate', 'target_compensation_rate', 'generated_rate', 'target_rate',
         'geometry_rate', 'compensation_rate', 'curvature_rate', 'log_force_rate',
         'structural_curvature_rate_bound', 'directional_curvature_rate_bound',
         'identity_absolute_error', 'identity_rate_error', 'curvature_split_absolute_error',
         'archived_force_relative_difference', 'F_norm',
         'force_weighted_second_moment', 'force_weighted_fourth_moment',
         'hidden_force_energy_concentration', 'omega_dot_transport',
         'omega_dot_redistribution', 'omega_dot_effective',
         'weighted_geometry_rate_bound', 'weighted_compensation_rate_bound',
         'weighted_curvature_rate_bound', 'measured_ell_norm', 'ell_M4_bound')
METRICS = TERMS + ('structural_to_directional', 'generated_target_cancellation',
                  'directional_to_positive_curvature', 'structural_to_positive_net_growth',
                  'structural_to_weighted', 'weighted_to_directional')


def number(row, key):
    try:
        return float(row.get(key, ''))
    except (TypeError, ValueError):
        return float('nan')


def derive(row):
    result = {key: row.get(key, '') for key in
              ('role', 'panel', 'target', 'seed', 'index', 'arm', 'source', 'state_sha256')}
    result.update({key: number(row, key) for key in ('width', 'start', 'horizon')})
    result.update({key: number(row, 'reinforcement_'+key) for key in TERMS})
    structural, directional = result['structural_curvature_rate_bound'], result['directional_curvature_rate_bound']
    generated, target = result['generated_rate'], result['target_rate']
    result['structural_to_directional'] = structural/directional if directional > 0 else np.nan
    weighted = result['weighted_curvature_rate_bound']
    result['structural_to_weighted'] = structural/weighted if weighted > 0 else np.nan
    result['weighted_to_directional'] = weighted/directional if directional > 0 else np.nan
    denominator = abs(generated)+abs(target)
    result['generated_target_cancellation'] = 1-abs(generated+target)/denominator if denominator > 0 else np.nan
    curvature, rate = result['curvature_rate'], result['log_force_rate']
    result['directional_to_positive_curvature'] = directional/curvature if curvature > 0 else np.nan
    result['structural_to_positive_net_growth'] = structural/rate if rate > 0 else np.nan
    return result


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0)
    quantiles = np.quantile(values, [0, .1, .5, .9, 1])
    return dict(count=len(values), **dict(zip(('min', 'p10', 'median', 'p90', 'max'), quantiles)),
                positive=int(np.sum(values > 0)), negative=int(np.sum(values < 0)), zero=int(np.sum(values == 0)))


def static_groups(rows):
    return {f'W{width}_age{age}': [r for r in rows if r['role'] == 'static'
                                 and r['width'] == width and r['start'] == age]
            for width, age in GROUPS}


def trajectories(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['role'] == 'trajectory' and row.get('arm') == 'natural':
            key = tuple(row[k] for k in ('panel', 'target', 'seed', 'index'))
            groups[key].append(row)
    return [sorted(group, key=lambda r: r['horizon']) for group in groups.values()]


def trajectory_endpoints(paths):
    result = []
    for path in paths:
        if len(path) < 2:
            continue
        first, last = path[0], path[-1]
        item = {key: first[key] for key in ('panel', 'target', 'seed', 'index', 'width', 'start')}
        item.update(snapshots=len(path), first_retained_horizon=first['horizon'],
                    last_retained_horizon=last['horizon'],
                    force_ratio_last_to_first=last['F_norm']/first['F_norm'] if first['F_norm'] > 0 else np.nan,
                    initial_log_force_rate=first['log_force_rate'], final_log_force_rate=last['log_force_rate'],
                    positive_sampled_rates=sum(r['log_force_rate'] > 0 for r in path),
                    initial_bound_ratio=first['structural_to_directional'],
                    final_bound_ratio=last['structural_to_directional'])
        for key in ('force_weighted_second_moment', 'force_weighted_fourth_moment',
                    'hidden_force_energy_concentration', 'weighted_curvature_rate_bound',
                    'measured_ell_norm', 'ell_M4_bound'):
            initial = first[key]
            available = [r[key] for r in path if np.isfinite(r[key])]
            item['initial_'+key] = initial
            item['final_'+key] = last[key]
            item[key+'_last_to_first'] = last[key]/initial if initial > 0 else np.nan
            item[key+'_sampled_max_to_first'] = max(available)/initial if available and initial > 0 else np.nan
        for key in ('omega_dot_transport', 'omega_dot_redistribution', 'omega_dot_effective'):
            item['initial_'+key] = first[key]
            item['final_'+key] = last[key]
            item[key+'_positive_sampled_rates'] = sum(r[key] > 0 for r in path)
            item[key+'_negative_sampled_rates'] = sum(r[key] < 0 for r in path)
        result.append(item)
    return result


def plot_static(groups, output):
    figures = []
    colors = ('#405a77', '#db9855', '#589985', '#a45168')
    for kind in ('bound_levels', 'signed_balance'):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
        for ax, ((width, age), (name, rows)) in zip(axes, zip(GROUPS, groups.items())):
            if kind == 'bound_levels':
                keys = ('structural_curvature_rate_bound', 'directional_curvature_rate_bound', 'curvature_rate', 'log_force_rate')
                labels = ('Population\nbound', 'Current-direction\nbound', 'Curvature\npositive only', 'Net growth\npositive only')
                ax.set_yscale('log')
            else:
                keys = ('residual_relaxation_rate', 'generated_rate', 'target_rate', 'log_force_rate')
                labels = ('Residual\nrelaxation', 'Generated\n+ compensation', 'Target\n+ compensation', 'Net')
                magnitudes = [abs(r[k]) for r in rows for k in keys if np.isfinite(r[k]) and r[k] != 0]
                ax.set_yscale('symlog', linthresh=max(1e-12, np.median(magnitudes)*.02) if magnitudes else 1e-6)
                ax.axhline(0, color='.55', lw=.8)
            for i, (key, color) in enumerate(zip(keys, colors)):
                values = np.array([r[key] for r in rows])
                values = values[np.isfinite(values)&((values > 0) if kind == 'bound_levels' else np.ones(len(values), dtype=bool))]
                offsets = np.linspace(-.15, .15, len(values)) if len(values) else np.array([])
                ax.scatter(i+offsets, values, s=17, alpha=.55, color=color, linewidths=0)
                if len(values):
                    ax.plot([i-.2, i+.2], [np.median(values)]*2, color='black', lw=2)
                ax.text(i, -.17, f'{len(values)}/{len(rows)} states', transform=ax.get_xaxis_transform(), ha='center', fontsize=8)
            ax.set_xticks(range(4), labels, fontsize=9)
            ax.set_title(f'W={width}; restart {age:,}\n{len(rows)} states, {len(set(r["target"] for r in rows))} targets', fontsize=11)
            ax.grid(axis='y', alpha=.18)
            ax.set_ylabel('Rate per unit gradient-flow time')
        title = ('How much does the population bound exceed current-direction feedback?' if kind == 'bound_levels'
                 else 'Signed force-norm feedback: compensation is included in each error source')
        fig.suptitle(title, fontsize=13)
        path = output/(kind+'.png'); fig.savefig(path, dpi=180); plt.close(fig); figures.append(path.name)
    return figures


def plot_natural(paths, output):
    panels = defaultdict(list)
    for path in paths:
        if len(path) >= 2:
            panels[(path[0]['width'], path[0]['start'])].append(path)
    if not panels:
        return None
    fig, axes = plt.subplots(len(panels), 3, figsize=(15, 3.4*len(panels)), squeeze=False, constrained_layout=True)
    for axes_row, ((width, age), group) in zip(axes, sorted(panels.items())):
        for column, ax in enumerate(axes_row):
            sampled = defaultdict(list)
            for path in group:
                baseline = path[0]['F_norm']
                values = [(r['F_norm']/baseline if baseline > 0 else np.nan) if column == 0 else
                          r['log_force_rate'] if column == 1 else r['structural_to_directional'] for r in path]
                ax.plot([r['horizon'] for r in path], values, color='#526f93', alpha=.17, lw=.8)
                for row, value in zip(path, values):
                    if np.isfinite(value):
                        sampled[row['horizon']].append(value)
            xx = sorted(sampled)
            if xx:
                quantiles = np.array([np.quantile(sampled[x], [.25, .5, .75]) for x in xx])
                ax.plot(xx, quantiles[:, 1], color='#234c80', lw=2)
                ax.fill_between(xx, quantiles[:, 0], quantiles[:, 2], alpha=.18, color='#526f93')
            ax.set_xscale('symlog', linthresh=1)
            if column == 1:
                flat = [abs(v) for values in sampled.values() for v in values if v != 0]
                ax.set_yscale('symlog', linthresh=max(1e-12, np.median(flat)*.02) if flat else 1e-6)
                ax.axhline(0, color='.5', lw=.7)
            else:
                ax.set_yscale('log')
                ax.axhline(1, color='.5', lw=.7)
            ax.set_title(f'W={int(width)}, restart {int(age):,}; {len(group)} trajectories', fontsize=10)
            ax.set_xlabel('Additional GD updates (retained checkpoints)')
            ax.set_ylabel(('F norm / first retained norm', 'Instantaneous effective-flow log-force rate', 'Population / current-direction bound')[column])
            ax.grid(alpha=.15)
    fig.suptitle('Natural continuations: measured force, local effective feedback, and bound conservatism', fontsize=13)
    path = output/'natural_feedback.png'; fig.savefig(path, dpi=180); plt.close(fig)
    return path.name


def plot_force_weighted_population(endpoints, output):
    omega = 'force_weighted_second_moment'
    selected = [r for r in endpoints if np.isfinite(r['initial_'+omega])
                and np.isfinite(r['final_'+omega]) and r['initial_'+omega] > 0]
    if not selected:
        return None
    groups = defaultdict(list)
    for row in selected:
        groups[(row['width'], row['start'])].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for index, ((width, start), group) in enumerate(sorted(groups.items())):
        label = f'W={int(width)}, restart {int(start):,}; n={len(group)}'
        color = f'C{index}'
        axes[0].scatter([r['initial_'+omega] for r in group], [r['final_'+omega] for r in group],
                        label=label, color=color, s=28, alpha=.7)
        values = np.array([r[omega+'_sampled_max_to_first'] for r in group])
        axes[1].scatter(index+np.linspace(-.12, .12, len(values)), values, color=color, s=28, alpha=.7)
        axes[1].plot([index-.2, index+.2], [np.median(values)]*2, color='black', lw=2)
    bounds = [r[key] for r in selected for key in ('initial_'+omega, 'final_'+omega) if r[key] > 0]
    axes[0].plot([min(bounds), max(bounds)], [min(bounds), max(bounds)], '--', color='.5', label='No change')
    axes[0].set(xscale='log', yscale='log', xlabel='Initial force-weighted second moment Ω', ylabel='Final retained Ω')
    axes[0].legend(fontsize=8)
    axes[1].axhline(1, color='.5', lw=.7)
    axes[1].set_xticks(range(len(groups)), [f'W={int(w)}\nrestart {int(s):,}' for w, s in sorted(groups)])
    axes[1].set(yscale='log', ylabel='Maximum sampled Ω / initial Ω')
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle(f'Force-weighted population evolution: {len(selected)} natural trajectories\n'
                 'Retained checkpoints only; sampled maxima do not bound intervening times')
    path = output/'force_weighted_population.png'; fig.savefig(path, dpi=180); plt.close(fig)
    return path.name


def summarize(source, output):
    source, output = Path(source), Path(output)
    with source.open(newline='') as stream:
        original = list(csv.DictReader(stream))
    rows = [derive(r) for r in original if r.get('reinforcement_status') == 'finite']
    groups, paths = static_groups(rows), trajectories(rows)
    output.mkdir(parents=True, exist_ok=False)
    pc.write_csv(output/'derived_states.csv', rows)
    endpoints = trajectory_endpoints(paths)
    pc.write_csv(output/'trajectory_endpoints.csv', endpoints)
    facts, flat = {}, []
    for name, group in groups.items():
        metrics = {key: stats([r[key] for r in group]) for key in METRICS}
        facts[name] = dict(states=len(group), targets=sorted(set(r['target'] for r in group)),
                           seeds=sorted(set(r['seed'] for r in group)), metrics=metrics)
        flat.extend(dict(group=name, metric=key, **value) for key, value in metrics.items())
    pc.write_csv(output/'group_statistics.csv', flat)
    closure = {key: stats([r[key] for r in rows]) for key in
               ('identity_absolute_error', 'identity_rate_error', 'curvature_split_absolute_error', 'archived_force_relative_difference')}
    figures = plot_static(groups, output)
    natural = plot_natural(paths, output)
    if natural:
        figures.append(natural)
    weighted = plot_force_weighted_population(endpoints, output)
    if weighted:
        figures.append(weighted)
    natural_groups = defaultdict(list)
    for row in endpoints:
        natural_groups[(row['width'], row['start'])].append(row)
    weighted_endpoint_keys = [key for key in endpoints[0] if any(part in key for part in
        ('force_weighted_', 'hidden_force_energy_', 'omega_dot_', 'weighted_curvature_', 'ell_'))] if endpoints else []
    natural_population_facts = {f'W{int(width)}_age{int(start)}': dict(
        trajectories=len(group), targets=sorted({r['target'] for r in group}),
        metrics={key: stats([r[key] for r in group]) for key in weighted_endpoint_keys})
        for (width, start), group in natural_groups.items()}
    rate_facts = {}
    rate_groups = defaultdict(list)
    for row in rows:
        rate_groups[(row['role'], row['width'], row['start'])].append(row)
    for (role, width, start), group in rate_groups.items():
        rate_facts[f'{role}_W{int(width)}_age{int(start)}'] = dict(
            states=len(group), metrics={key: stats([r[key] for r in group]) for key in
                ('omega_dot_transport', 'omega_dot_redistribution', 'omega_dot_effective')})
    result = dict(input_states=len(original), finite_states=len(rows), groups=facts, closure=closure,
                  natural_trajectories=len(endpoints), natural_population_endpoints=natural_population_facts,
                  omega_rate_statistics=rate_facts,
                  natural_endpoint_statistics={key: stats([r[key] for r in endpoints])
                  for key in ('force_ratio_last_to_first', 'initial_log_force_rate', 'final_log_force_rate')},
                  interpretation=dict(structural_to_directional='loss from replacing the current direction by population norm bounds',
                                      generated_target_cancellation='1-|G+T|/(|G|+|T|); each source includes its compensation',
                                      positive_ratios='defined only when the corresponding signed denominator is positive',
                                      natural_rates='local effective-flow probes along actual GD states; not integrated as actual GD rates'),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), helper_sha256=pc.digest(__file__), figures=figures)
    (output/'facts.json').write_text(json.dumps(pc.clean(result), indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.source, args.output)
    print(json.dumps(dict(finite_states=result['finite_states'], figures=result['figures'])))
