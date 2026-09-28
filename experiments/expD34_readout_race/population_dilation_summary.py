"""Archive every finite-dilation outcome and plot output/population responses.

Numerical CSV/JSON/PNG artifacts only. No generated report prose.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .mechanism_dilation_analysis import clean, digest, ratio, stats, write_csv


SCALARS = ('relative_l2', 'relative_eval_l2', 'gamma_mean', 'lambda_mean', 'lambda_rms',
           'M', 'M6', 'C6', 'Q', 'Fa_norm', 'Ra_norm', 'F_norm', 'loss', 'closure_error',
           'activation_argument_rms', 'activation_argument_max', 'effective_norm_integral',
           'diagnostic_unresolved_steps')


def identity(case):
    return tuple(str(case.get(k)) for k in ('target', 'seed', 'start', 'arm'))


def population_key(row):
    return tuple(row[k] for k in ('cohort', 'width', 'start', 'eta', 'requested_steps'))


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    rows, endpoints, attempts, issues, comparisons = [], [], [], [], []
    provenance, snapshots, initial_points = {}, {}, {}
    for directory in args.runs:
        manifest_path = directory/'manifest.json'
        if not manifest_path.exists():
            issues.append(dict(run=str(directory), reason='missing_manifest')); continue
        manifest = json.loads(manifest_path.read_text())
        provenance[str(manifest_path)] = digest(manifest_path)
        source = Path(manifest['input'])
        if not source.exists() and args.input_root:
            source = args.input_root/source.name
        if not source.exists() or digest(source) != manifest['input_sha256']:
            raise ValueError(f'Missing or mismatched prepared input: {source}')
        provenance[str(source)] = digest(source)
        with np.load(source) as data:
            prepared = data['p'].copy()
        cases = manifest['cases']
        for skipped, values in ((False, cases), (True, manifest.get('skipped_invalid', []))):
            for case in values:
                attempts.append({**case, 'run': str(directory), 'skipped': skipped,
                                 'repair': json.dumps(case.get('repair', {})),
                                 'eta': manifest['eta'], 'requested_steps': manifest['steps']})
        files = sorted(directory.glob('[0-9]'*9+'.npz'))
        if not files or files[0].stem != '000000000':
            issues.append(dict(run=str(directory), reason='missing_initial_snapshot')); continue
        with np.load(files[0]) as data:
            initial = {k: data[k].copy() for k in data.files}
        for i, case in enumerate(cases):
            if not np.array_equal(initial['p'][i], prepared[case['input_index']]):
                raise ValueError(f'Start-state mismatch: {directory}/{i}')
            initial_points[str(directory), identity(case)] = initial['p'][i]
        if int(files[-1].stem) != manifest['steps']:
            issues.append(dict(run=str(directory), reason='missing_final_snapshot',
                               last_snapshot=int(files[-1].stem)))
        for path in files:
            provenance[str(path)] = digest(path)
            with np.load(path) as data:
                state = {k: data[k].copy() for k in data.files}
            step = int(path.stem)
            for i, case in enumerate(cases):
                row = {key: case.get(key) for key in
                       ('target', 'seed', 'start', 'cohort', 'scale', 'reference', 'arm', 'input_index')}
                row.update(run=str(directory), width=(state['p'].shape[1]-1)//3,
                           requested_steps=manifest['steps'], eta=manifest['eta'], step=step,
                           flow_time=step*manifest['eta'], completed_steps=int(state['count'][i]),
                           failed=bool(state['failed'][i]), resolved=bool(state['resolved'][i]),
                           input_sha256=manifest['input_sha256'])
                row['status'] = ('nonfinite_training' if row['failed'] else
                                 'incomplete' if row['completed_steps'] != step else
                                 'unresolved_diagnostic' if not row['resolved'] else 'finite')
                for key in SCALARS:
                    if key in state:
                        row[key] = float(state[key][i])
                        row['initial_'+key] = float(initial[key][i])
                for key in ('relative_l2', 'relative_eval_l2', 'gamma_mean', 'Q'):
                    if key in row:
                        row[key+'_learned_change'] = row[key]-row['initial_'+key]
                        row[key+'_own_start_ratio'] = ratio(row[key], row['initial_'+key])
                row['Fa_own_start_ratio'] = (ratio(row['Fa_norm'], row['initial_Fa_norm'])
                                              if row['resolved'] and initial['resolved'][i] else None)
                for j, channel in enumerate(manifest['channels']):
                    row[channel+'_signed_travel'] = float(np.mean(state['signed'][i, j]))
                    row[channel+'_norm_integral'] = float(state['norm_integral'][i, j])
                row['tracking_integrated_norm_to_effective'] = ratio(
                    row['tracking_norm_integral'], row['effective_norm_integral'])
                row['force_integrals_complete'] = row.get('diagnostic_unresolved_steps', 0) == 0
                if not row['force_integrals_complete']:
                    row['tracking_integrated_norm_to_effective'] = None
                for j, threshold in enumerate(manifest.get('error_thresholds', [])):
                    row[f'first_hit_relative_l2_{threshold:g}'] = int(state['error_first_hit'][i, j])
                row['initially_above_lambda025'] = int(np.sum(initial['first_hit'][i] == 0))
                row['learned_new_hits_lambda025'] = int(np.sum(state['first_hit'][i] > 0))
                snapshots[str(directory), identity(case), step] = (row, state['p'][i].copy())
                rows.append(row)
                if path == files[-1]:
                    endpoints.append(row)
    # Pair only within identical restart population and physical time.
    lookup = {(r['run'], r['target'], r['seed'], r['start'], r['arm'], r['step']): r for r in rows}
    for row in rows:
        base = lookup.get((row['run'], row['target'], row['seed'], row['start'], 'repaired', row['step']))
        row['baseline_available'] = bool(base and not base['failed'] and base['completed_steps'] == row['step'])
        if row['baseline_available']:
            for key in ('relative_l2', 'relative_eval_l2', 'gamma_mean', 'Q'):
                if key in row and key in base:
                    row[key+'_minus_repaired'] = row[key]-base[key]
                    row[key+'_initial_minus_repaired'] = row['initial_'+key]-base['initial_'+key]
    paired = defaultdict(list)
    for row in endpoints:
        paired[(row['input_sha256'], row['target'], row['seed'], row['start'], row['arm'],
                round(row['flow_time'], 12))].append(row)
    for values in paired.values():
        for i, first in enumerate(values):
            for second in values[i+1:]:
                if first['eta'] == second['eta']:
                    continue
                coarse, fine = sorted((first, second), key=lambda r: -r['eta'])
                a0 = initial_points[coarse['run'], identity(coarse)]
                b0 = initial_points[fine['run'], identity(fine)]
                if not np.array_equal(a0, b0):
                    raise ValueError('Matched-step control initial states differ')
                aa = snapshots[coarse['run'], identity(coarse), coarse['step']][1]
                bb = snapshots[fine['run'], identity(fine), fine['step']][1]
                w = coarse['width']
                comparisons.append(dict(target=coarse['target'], seed=coarse['seed'], arm=coarse['arm'],
                    start=coarse['start'], coarse_run=coarse['run'], fine_run=fine['run'],
                    flow_time=coarse['flow_time'], input_hash_match=True, initial_state_match=True,
                    either_failed=coarse['failed'] or fine['failed'],
                    equal_completed_flow_time=bool(
                        np.isclose(coarse['completed_steps']*coarse['eta'], fine['completed_steps']*fine['eta'],
                                   rtol=0, atol=1e-12)),
                    endpoint_slope_difference=float(np.linalg.norm(aa[:w]-bb[:w])),
                    relative_to_coarse_displacement=ratio(np.linalg.norm(aa[:w]-bb[:w]),
                                                         np.linalg.norm(aa[:w]-a0[:w]))))
    grouped = defaultdict(list)
    for row in endpoints:
        grouped[(*population_key(row), row['reference'], row['scale'])].append(row)
    groups = []
    for key, values in grouped.items():
        finite = [r for r in values if not r['failed'] and r['completed_steps'] == r['requested_steps']]
        groups.append(dict(group=key, attempted=len(values), finite=len(finite),
                           targets=len({r['target'] for r in values}),
                           statistics={k: stats(finite, k) for k in (
                               'relative_l2', 'relative_eval_l2', 'gamma_mean_own_start_ratio',
                               'Q_own_start_ratio', 'Fa_own_start_ratio',
                               'tracking_integrated_norm_to_effective', 'diagnostic_unresolved_steps')}))
    write_csv(args.output/'states.csv', rows)
    write_csv(args.output/'endpoints.csv', endpoints)
    write_csv(args.output/'attempts.csv', attempts)
    write_csv(args.output/'step_controls.csv', comparisons)
    summary = dict(source_sha256=digest(Path(__file__)), provenance=provenance,
                   attempts=len(attempts), skipped=sum(a['skipped'] for a in attempts),
                   endpoint_count=len(endpoints), nonfinite_endpoints=sum(r['failed'] for r in endpoints),
                   issues=issues, groups=groups, matched_step_controls=len(comparisons))
    (args.output/'summary.json').write_text(json.dumps(clean(summary), indent=2, allow_nan=False)+'\n')
    if not args.no_plots:
        plot(args.output, endpoints, attempts)


def plot(output, endpoints, attempts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    coverage = defaultdict(lambda: defaultdict(int))
    for row in attempts:
        key = row['run']
        coverage[key]['skipped repair'] += int(row['skipped'])
    for row in endpoints:
        category = ('nonfinite training' if row['failed'] else
                    'incomplete' if row['completed_steps'] != row['requested_steps'] else
                    'finite, unresolved visited' if row.get('diagnostic_unresolved_steps', 0) > 0 else
                    'finite, resolved')
        coverage[row['run']][category] += 1
    fig, ax = plt.subplots(figsize=(12, max(3., .4*len(coverage))), constrained_layout=True)
    names = list(coverage); left = np.zeros(len(names))
    for category in ('finite, resolved', 'finite, unresolved visited', 'incomplete',
                     'nonfinite training', 'skipped repair'):
        counts = np.array([coverage[name][category] for name in names])
        ax.barh(np.arange(len(names)), counts, left=left, label=category)
        left += counts
    ax.set(yticks=np.arange(len(names)), yticklabels=[Path(name).name for name in names],
           xlabel='Branches (all attempted outcomes retained)', title='Coverage and diagnostic resolution')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.13), ncol=3)
    fig.savefig(output/'coverage.png', dpi=180); plt.close(fig)
    panels = defaultdict(list)
    for row in endpoints:
        if not row['failed'] and row['completed_steps'] == row['requested_steps']:
            panels[population_key(row)].append(row)
    fields = ('relative_l2', 'relative_eval_l2', 'gamma_mean_own_start_ratio',
              'Q_own_start_ratio', 'Fa_own_start_ratio', 'tracking_integrated_norm_to_effective')
    labels = ('Training relative L2', 'Independent-grid relative L2', 'Slope / post-kick slope',
              'Q / post-kick Q', 'Effective force / post-kick force', 'Integrated tracking / effective norm')
    for index, (key, rows) in enumerate(sorted(panels.items(), key=lambda item: str(item[0]))):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for ax, field, label in zip(axes.flat, fields, labels):
            for reference, color in (('primary', 'C0'), ('inverse', 'C1')):
                selected = [r for r in rows if r['reference'] == reference and r['arm'] != 'original']
                scales = sorted({r['scale'] for r in selected})
                xx, yy = [], []
                for scale in scales:
                    values = [r[field] for r in selected if r['scale'] == scale and
                              r.get(field) is not None and np.isfinite(r[field]) and r[field] > 0]
                    if values:
                        xx.append(scale); yy.append(np.median(values))
                        ax.scatter(np.full(len(values), scale), values, s=9, alpha=.2, color=color)
                ax.plot(xx, yy, 'o-', color=color, label=reference)
                if field in ('relative_l2', 'relative_eval_l2'):
                    start = [np.median([r['initial_'+field] for r in selected if r['scale'] == scale])
                             for scale in scales]
                    ax.plot(scales, start, ':', color=color, label=reference+' post-kick')
            ax.set(xscale='log', yscale='log', xlabel='Injected geometry multiplier', ylabel=label)
            ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'cohort={key[0]}, W={key[1]}, restart={key[2]}, eta={key[3]}, updates={key[4]}\n'
                     f'{len({r["target"] for r in rows})} targets; {len(rows)} finite endpoints; dots=cases, lines=medians')
        fig.savefig(output/f'response_{index:02d}.png', dpi=180)
        plt.close(fig)
    plot_overview(output, endpoints)


def plot_overview(output, endpoints):
    """Keep supplied geometry separate from subsequent learned expansion."""
    import matplotlib.pyplot as plt
    panels = ((177, 600000), (705, 20000), (1409, 20000))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for column, (width, age) in enumerate(panels):
        rows = [r for r in endpoints if r['width'] == width and r['start'] == age
                and r['eta'] == .002 and r['requested_steps'] == 20000
                and not r['failed'] and r['completed_steps'] == 20000 and r['arm'] != 'original']
        if not rows:
            continue
        for reference, color, label in (('primary', 'C0', 'Repair near original readout'),
                                        ('inverse', 'C1', 'Repair near readout / multiplier')):
            selected = [r for r in rows if r['reference'] == reference]
            scales = sorted({r['scale'] for r in selected})
            errors, initial_errors, growth = [], [], []
            for scale in scales:
                group = [r for r in selected if r['scale'] == scale]
                ee = np.array([r['relative_eval_l2'] for r in group])
                gg = 100*(np.array([r['gamma_mean_own_start_ratio'] for r in group])-1)
                errors.append(np.median(ee))
                initial_errors.append(np.median([r['initial_relative_eval_l2'] for r in group]))
                growth.append(np.median(gg))
                axes[0, column].scatter(np.full(len(ee), scale), ee, s=12, alpha=.2, color=color)
                axes[1, column].scatter(np.full(len(gg), scale), gg, s=12, alpha=.2, color=color)
            axes[0, column].plot(scales, errors, 'o-', color=color, label=label)
            axes[0, column].plot(scales, initial_errors, ':', color=color)
            axes[1, column].plot(scales, growth, 'o-', color=color)
        axes[0, column].axhline(.01, color='.45', ls='--', lw=.8)
        axes[0, column].set(yscale='log', ylabel='Independent-grid relative error',
            title=f'W={width}; restart {age:,}\n{len({r["target"] for r in rows})} targets, '
                  f'{len({(r["target"], r["seed"]) for r in rows})} starts')
        axes[1, column].set(yscale='symlog', ylabel='Additional mean-slope change (%)')
        axes[1, column].set_yscale('symlog', linthresh=.01)
        axes[1, column].axhline(0, color='.45', lw=.8)
        for ax in axes[:, column]:
            ax.set(xscale='log', xlabel='Injected geometry multiplier')
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('20,000 further GD updates: output fitting and additional scale acquisition\n'
                 'Solid: endpoint medians; dotted: post-repair error; dots: individual cases; dashed: 1% error', fontsize=12)
    fig.savefig(output/'large_dilation_output_scale.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--no-plots', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
