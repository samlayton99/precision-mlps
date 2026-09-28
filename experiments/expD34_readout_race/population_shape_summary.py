"""Compact conditional effective-flow lifetime facts and one comparison figure."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


P = 'shape_certificate_'
METHODS = {'Initial sensitivity': 'hs_certificate_flow_time',
           'Initial force': 'force_certificate_flow_time',
           'Concentration: explicit': P+'explicit_flow_time',
           'Concentration: initial force': P+'flow_time'}


def number(row, key):
    try:
        return float(row.get(key, 'nan'))
    except (ValueError, TypeError):
        return float('nan')


def statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0)
    quantiles = np.quantile(values, [0, .1, .5, .9, 1])
    return dict(count=len(values), **dict(zip(('min', 'p10', 'median', 'p90', 'max'), map(float, quantiles))))


def summarize(archive, broad, output, force_archive=None, force_broad=None):
    sources = {}
    for name, source in (('archive', archive), ('broad', broad)):
        with Path(source).open(newline='') as stream:
            sources[name] = list(csv.DictReader(stream))
    joins = {}
    for name, source in (('archive', force_archive), ('broad', force_broad)):
        if source is None:
            continue
        with Path(source).open(newline='') as stream:
            force_rows = list(csv.DictReader(stream))
        lookup = {}
        for row in force_rows:
            key = row.get('state_sha256')
            if not key:
                raise ValueError('Force comparison requires exact state hashes.')
            lookup[key] = row
        count = 0
        for row in sources[name]:
            other = lookup.get(row.get('state_sha256'))
            if other is not None:
                for key in ('target', 'width', 'seed'):
                    if row.get(key) != other.get(key):
                        raise ValueError('State-hash join disagrees on '+key)
                row.update({key: value for key, value in other.items() if key.startswith('force_certificate_')})
                count += 1
        joins[name] = dict(matched=count, total=len(sources[name]), source=str(source),
                           sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    groups = {}
    for label, source, width in (('W705 canonical', 'archive', 705),
                                 ('W1409 canonical', 'archive', 1409),
                                 ('W705 broad', 'broad', 705)):
        groups[label] = [r for r in sources[source] if r.get('role') == 'static'
                         and number(r, 'width') == width and number(r, 'start') == 20000]
    facts = dict(scope='Conditional FP64 effective-flow bounds. Future I_F<=32 is assumed, not certified.',
                 reference_physical_time=40., reference='20,000 updates times eta=0.002; comparison only, not a GD certificate',
                 source_statuses={}, groups={}, force_joins=joins)
    for name, rows in sources.items():
        counts = {}
        for row in rows:
            status = row.get(P+'status', 'missing')
            counts[status] = counts.get(status, 0)+1
        facts['source_statuses'][name] = counts
    table = []
    for label, rows in groups.items():
        accepted = [r for r in rows if r.get(P+'status') == 'conditional_FP64_effective_flow']
        common = [r for r in accepted if all(np.isfinite(number(r, field)) and number(r, field) > 0
                                           for field in METHODS.values())]
        identity = lambda subset: sorted({(r.get('target'), r.get('seed')) for r in subset})
        group = dict(total=len(rows), initial_I_within_allowance=sum(number(r, 'reinforcement_hidden_force_energy_concentration') <= 32 for r in rows),
                     accepted=len(accepted), targets=sorted({r.get('target', '') for r in rows}),
                     target_seed_pairs=identity(rows), common_all_methods=len(common),
                     common_target_seed_pairs=identity(common), methods={}, common_methods={})
        for name, field in METHODS.items():
            group['methods'][name] = statistics([number(r, field) for r in accepted])
            group['common_methods'][name] = statistics([number(r, field) for r in common])
        for key in ('relative_output_error_floor', 'relative_fine_retention', 'explicit_relative_fine_retention',
                    'flow_time', 'explicit_flow_time', 'time_relative_tolerance_difference'):
            group[key] = statistics([number(r, P+key) for r in accepted])
            table.append(dict(group=label, quantity=key, **group[key]))
        group['positive_output_floors'] = sum(number(r, P+'relative_output_error_floor') > 0 for r in accepted)
        group['output_floors_above_one_percent'] = sum(number(r, P+'relative_output_error_floor') > .01 for r in accepted)
        group['fine_retention_at_least_half'] = sum(number(r, P+'relative_fine_retention') >= .5 for r in accepted)
        group['refined_time_at_least_40'] = sum(number(r, P+'flow_time') >= 40 for r in accepted)
        for method, field in list(METHODS.items())[:2]:
            paired = [r for r in accepted if np.isfinite(number(r, field)) and number(r, field) > 0]
            group['refined_over_'+method] = statistics([number(r, P+'flow_time')/number(r, field) for r in paired])
        facts['groups'][label] = group
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    colors = ('#2878b5', '#c05235', '#52944b')
    for index, (label, rows) in enumerate(groups.items()):
        accepted = [r for r in rows if r.get(P+'status') == 'conditional_FP64_effective_flow']
        for position, (method, field) in enumerate(METHODS.items()):
            values = np.asarray([number(r, field) for r in accepted])
            values = values[np.isfinite(values) & (values > 0)]
            if len(values):
                low, median, high = np.quantile(values, [.1, .5, .9])
                axes[0].errorbar(position+(index-1)*.18, median, yerr=[[median-low], [high-median]],
                                 fmt='o', capsize=3, color=colors[index], label=label if position == 0 else None)
        values = np.asarray([number(r, P+'relative_output_error_floor') for r in accepted])
        jitter = np.linspace(-.12, .12, len(values))
        axes[1].scatter(index+jitter, values, color=colors[index], alpha=.55, s=22)
        if len(values):
            axes[1].plot([index-.2, index+.2], [np.median(values)]*2, color='black', lw=2)
    axes[0].axhline(40, color='black', ls='--', lw=1, label='Reference: physical time 40')
    axes[0].set_yscale('log')
    axes[0].set_xticks(range(len(METHODS)), ['Initial\nsensitivity', 'Initial\nforce', 'Concentration\nexplicit', 'Concentration\ninitial force'])
    axes[0].set_ylabel('Effective-flow time (median, 10–90% interval)')
    axes[0].set_title('Conditional population bounds versus earlier bounds')
    axes[0].legend(fontsize=8)
    axes[1].axhline(.01, color='black', ls='--', lw=1, label='1% relative error')
    axes[1].set_xticks(range(len(groups)), list(groups), rotation=10)
    axes[1].set_ylabel('Output-error lower bound / target RMS norm')
    axes[1].set_title('Retained output-error floor at the travel endpoint')
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2, axis='y')
    fig.suptitle('Age-20k states; future force concentration I_F ≤ 32 is an assumption', fontsize=12)
    fig.tight_layout()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output/'conditional_lifetimes.png', dpi=180)
    plt.close(fig)
    facts['provenance'] = {name: dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                           for name, path in (('archive', archive), ('broad', broad))}
    (output/'facts.json').write_text(json.dumps(facts, indent=2, allow_nan=False)+'\n')
    with (output/'group_statistics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in table for k in row)))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(facts, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--broad', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--force-archive')
    parser.add_argument('--force-broad')
    args = parser.parse_args()
    summarize(args.archive, args.broad, args.output, args.force_archive, args.force_broad)
