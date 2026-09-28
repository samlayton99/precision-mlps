"""Numerical applicability summary and figure; never generate report prose."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def num(row, key):
    try:
        value = float(row.get(key, ''))
        return value if np.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def yes(row, key):
    return str(row.get(key, '')).lower() == 'true'


def finite_ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    result = numerator/denominator
    return float(result) if np.isfinite(result) else None


def statistics(rows, key):
    values = [v for r in rows if (v := num(r, key)) is not None]
    return dict(n=len(values), min=float(min(values)) if values else None,
                median=float(np.median(values)) if values else None,
                max=float(max(values)) if values else None)


def groups(rows, key):
    result = defaultdict(list)
    for row in rows:
        result[str(row.get(key, ''))].append(row)
    return result


def identity(row):
    return row.get('state_sha256') or (row.get('panel'), row.get('index'))


def eligibility(rows):
    return dict(states=len(rows), effective_initial_eligible=sum(r.get('status') != 'no_initial_region' for r in rows),
                effective_positive_horizon=sum((num(r, 'effective_physical_time') or 0) > 0 or r.get('status') == 'stationary_effective_flow' for r in rows),
                gd_initial_eligible=sum(r.get('gd_initial_eligible', False) for r in rows),
                gd_positive_horizon=sum((num(r, 'gd_updates') or 0) > 0 or r.get('gd_reason') == 'stationary' for r in rows),
                gd_reasons=dict(Counter(r.get('gd_reason', 'missing') for r in rows)),
                effective_limiting_conditions=dict(Counter(r.get('limiting_condition', 'missing') for r in rows)),
                effective_physical_time=statistics(rows, 'effective_physical_time'),
                gd_updates=statistics(rows, 'gd_updates'), gd_physical_time=statistics(rows, 'gd_physical_time'),
                effective_initial_gap=statistics(rows, 'effective_initial_gap'), gd_initial_gap=statistics(rows, 'gd_initial_gap'))


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    files = ('initial_states', 'effective_flow_best', 'effective_flow_margins', 'ordinary_gd_margins',
             'natural_initial_states', 'natural_gd_margins', 'natural_path_checks', 'failures', 'natural_failures')
    data = {name: read(args.evaluation/(name+'.csv')) for name in files}
    manifest = json.loads((args.evaluation/'manifest.json').read_text())
    initials = {identity(r): r for r in data['initial_states']}
    gd_cases, effective_cases = defaultdict(list), defaultdict(list)
    for r in data['ordinary_gd_margins']:
        gd_cases[identity(r)].append(r)
    for r in data['effective_flow_margins']:
        effective_cases[identity(r)].append(r)
    states = []
    for best in data['effective_flow_best']:
        initial = initials[identity(best)]
        row = dict(initial, **best)
        gd = gd_cases[identity(best)]
        chosen = max(gd, key=lambda r: (float('inf') if r.get('reason') == 'stationary' else num(r, 'valid_updates') or 0, yes(r, 'valid'))) if gd else {}
        time, width, eta, count = num(best, 'max_time_lower'), num(initial, 'W'), num(initial, 'eta'), num(chosen, 'valid_updates')
        row.update(effective_physical_time=time*width if time is not None else None,
                   gd_updates=count, gd_physical_time=count*eta if count is not None and eta is not None else None,
                   gd_reason=chosen.get('reason'), gd_initial_eligible=any(yes(r, 'valid') for r in gd),
                   effective_initial_gap=max((num(r, 'initial_gap') for r in effective_cases[identity(best)] if num(r, 'initial_gap') is not None), default=None),
                   gd_initial_gap=max((num(r, 'gap') for r in gd if num(r, 'gap') is not None), default=None))
        row.update({'gd_selected_'+key: num(chosen, key) for key in ('multiplier', 'gap', 'step_loss', 'a', 'kappa', 'Dell', 'B', 'LF')})
        row.update(gd_uniform_force_over_initial=finite_ratio(finite_ratio(num(chosen, 'f'), width), num(initial, 'F0')),
                   gd_actual_coarse_over_bound=finite_ratio(num(initial, 'coarse_k0'), num(chosen, 'kappa')),
                   gd_LF_over_W=finite_ratio(num(chosen, 'LF'), width),
                   gd_Dell_over_W=finite_ratio(num(chosen, 'Dell'), width),
                   gd_initial_force_zero=num(initial, 'F0') == 0,
                   gd_chosen_kappa_zero=num(chosen, 'kappa') == 0)
        states.append(row)
    natural_initials = {identity(r): r for r in data['natural_initial_states']}
    natural = []
    for row in data['natural_path_checks']:
        if num(row, 'horizon') != 20000:
            continue
        first = natural_initials[identity(row)]
        rec = dict(row, W=first['W'])
        for metric in ('R0', 'M0', 'Es0', 'Y0', 'coarse_k0', 'z0', 'F0', 'total_loss0', 'lambda_max0'):
            now, start = num(row, 'observed_'+metric), num(first, metric)
            rec[metric+'_ratio'] = now/start if now is not None and start not in (None, 0.) else None
            rec[metric+'_change'] = now-start if now is not None and start is not None else None
        natural.append(rec)
    metrics = ('R0', 'M0', 'Es0', 'Y0', 'coarse_k0', 'z0', 'F0', 'total_loss0', 'lambda_max0')
    def observed(rows):
        return dict(states=len(rows), endpoint_lambda_max=statistics(rows, 'observed_lambda_max0'),
                    metrics={metric: {kind: statistics(rows, metric+'_'+kind) for kind in ('ratio', 'change')} for metric in metrics})
    inside = [r for r in data['natural_path_checks'] if yes(r, 'within_valid_gd_interval')]
    slacks = ('support_slack', 'moment_upper_slack', 'moment_lower_slack', 'tracking_slack', 'force_slack', 'conditioning_slack', 'slope_travel_slack')
    comparisons = {key: dict(**statistics(inside, key), negative_slacks=sum((num(r, key) or 0) < 0 for r in inside)) for key in slacks}
    violations = [{k: r.get(k) for k in ('panel', 'index', 'target', 'horizon', 'actual_updates')} | {'negative_slacks': {key: num(r, key) for key in slacks if num(r, key) is not None and num(r, key) < 0}} for r in inside if any(num(r, key) is not None and num(r, key) < 0 for key in slacks)]
    natural_gd = defaultdict(list)
    for row in data['natural_gd_margins']:
        natural_gd[identity(row)].append(row)
    summary = dict(role='FP64 applicability evaluation, not rigorous numerical certification',
        static=eligibility(states), by_width={k: eligibility(v) for k, v in groups(states, 'W').items()},
        by_target={k: eligibility(v) for k, v in groups(states, 'target').items()},
        natural_gd_initial_eligible=sum(any(yes(r, 'valid') for r in rows) for rows in natural_gd.values()),
        natural_gd_positive_horizon=sum(any((num(r, 'valid_updates') or 0) > 0 or r.get('reason') == 'stationary' for r in rows) for rows in natural_gd.values()),
        gd_region_reasons=dict(Counter(r.get('reason', 'missing') for r in data['ordinary_gd_margins'])),
        gd_all_region_bottlenecks={key: statistics(data['ordinary_gd_margins'], key) for key in ('gap', 'step_loss', 'a', 'kappa', 'Dell', 'B', 'LF', 'valid_updates')},
        gd_selected_bottlenecks={key: statistics(states, 'gd_selected_'+key) for key in ('gap', 'step_loss', 'a', 'kappa', 'Dell', 'B', 'LF')},
        gd_selected_ratios_by_width={width: {key: statistics(rows, key) for key in ('gd_uniform_force_over_initial', 'gd_actual_coarse_over_bound', 'gd_LF_over_W', 'gd_Dell_over_W')} for width, rows in groups(states, 'W').items()},
        gd_selected_ratio_zero_denominators=dict(initial_force=sum(r['gd_initial_force_zero'] for r in states), chosen_kappa=sum(r['gd_chosen_kappa_zero'] for r in states)),
        gd_selected_ratio_policy='Missing, zero-denominator, or nonfinite ratios are null; no cutoff is applied',
        observed_20k_all_cases_independent_of_applicability=observed(natural),
        observed_20k_by_width={k: observed(v) for k, v in groups(natural, 'W').items()},
        observed_20k_by_target={k: observed(v) for k, v in groups(natural, 'target').items()},
        within_interval_rows=len(inside), within_interval_positive_horizon_rows=sum((num(r, 'horizon') or 0) > 0 for r in inside),
        comparisons=comparisons, negative_slack_rows=violations,
        negative_slack_policy='Raw negative values retained, including floating-point roundoff; no arbitrary tolerance',
        failures=data['failures'], natural_failures=data['natural_failures'],
        missing_optional_files=[name+'.csv' for name in files if not (args.evaluation/(name+'.csv')).exists()],
        evaluator_manifest=manifest, selected_states=states, natural_20k_states=natural,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    widths = sorted({int(float(r['W'])) for r in states})
    for panel, metric, title in ((0, 'effective_physical_time', 'Pure effective flow'), (1, 'gd_physical_time', 'Ordinary GD')):
        ax = axes[panel]
        for i, width in enumerate(widths):
            rows = [r for r in states if int(float(r['W'])) == width]
            values = [num(r, metric) for r in rows if (num(r, metric) or 0) > 0]
            if values:
                ax.plot([i, i], [min(values), max(values)], color='tab:blue')
                ax.scatter(i, np.median(values), color='tab:blue')
            ax.text(i, .96, f'{len(values)}/{len(rows)} positive', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=8)
        ax.set_yscale('log'); ax.set_title(title)
        ax.set_ylabel('Physical time t' if panel == 0 else 'Physical GD time ηN')
    for i, width in enumerate(widths):
        rows = [r for r in states if int(float(r['W'])) == width]
        for shift, key, label, color in ((-.1, 'effective_initial_gap', 'GF initial gap', 'tab:blue'), (.1, 'gd_initial_gap', 'GD initial gap', 'tab:orange')):
            vals = [num(r, key) for r in rows if num(r, key) is not None]
            if vals:
                axes[2].plot([i+shift]*2, [min(vals), max(vals)], color=color, alpha=.5)
                axes[2].scatter(i+shift, np.median(vals), color=color, label=label if i == 0 else None)
    axes[2].axhline(0, color='gray', linestyle='--'); axes[2].set_ylabel('Best initial singular-value gap')
    axes[2].set_yscale('symlog', linthresh=.1)
    axes[2].set_title('Conditioning test'); axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(range(len(widths)), widths); ax.set_xlabel('Physical width W')
        ax.set_xlim(-.5, len(widths)-.5)
    fig.tight_layout(); fig.savefig(args.output/'applicability.png', dpi=180); plt.close(fig)
    print(json.dumps(dict(static_states=len(states), natural_20k_states=len(natural), within_interval_rows=len(inside))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
