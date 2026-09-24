"""Numerical summaries and plots of finite-dilation continuations; no report prose."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PANELS = ('late_development_run', 'late_confirmation_run', 'wide_N512_run', 'wide_N1024_run', 'halfstep_run')
ARMS = ('original', 'repaired', 's125_primary', 's2_primary', 's125_inverse', 's2_inverse')
LABELS = dict(original='Original', repaired='Repaired baseline', s125_primary='Primary 1.25×',
              s2_primary='Primary 2×', s125_inverse='Inverse readout 1.25×', s2_inverse='Inverse readout 2×')
RATIO_FLOOR = 1e-14


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(a, b):
    return float(a/b) if np.isfinite(a) and np.isfinite(b) and abs(b) > RATIO_FLOOR else None


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader(); writer.writerows(clean(rows))


def stats(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return dict(n=len(values), min=min(values) if values else None,
                median=float(np.median(values)) if values else None, max=max(values) if values else None)


def identity(case):
    return str(case['target']), str(case['seed']), str(case['source_index'])


def run(args):
    output = args.output or args.root/'analysis'
    output.mkdir(parents=True, exist_ok=False)
    rows, attempts, failures, provenance, snapshots = [], [], [], {}, {}
    initial_states, prepared_hashes = {}, {}
    for panel in PANELS:
        directory = args.root/panel; manifest_path = directory/'manifest.json'
        if not manifest_path.exists():
            failures.append(dict(panel=panel, reason='missing_manifest')); continue
        manifest = json.loads(manifest_path.read_text())
        provenance[str(manifest_path)] = digest(manifest_path)
        input_path = Path(manifest['input'])
        if digest(input_path) != manifest['input_sha256']:
            raise ValueError(f'Input hash mismatch: {input_path}')
        prepared_hashes[panel] = manifest['input_sha256']
        provenance[str(input_path)] = digest(input_path)
        cases = manifest['cases']
        for case in cases+manifest['skipped_invalid']:
            attempts.append(dict(case, panel=panel, repair=json.dumps(case.get('repair', {})),
                                 skipped=case in manifest['skipped_invalid']))
        if not cases:
            continue
        files = sorted(directory.glob('[0-9]'*9+'.npz'))
        if not files or files[0].stem != '000000000':
            failures.append(dict(panel=panel, reason='missing_initial_snapshot')); continue
        if not (directory/f"{int(manifest['steps']):09d}.npz").exists():
            failures.append(dict(panel=panel, reason='missing_final_snapshot'))
        with np.load(files[0]) as data:
            first = {key: data[key].copy() for key in data.files}
        for i, case in enumerate(cases):
            initial_states[panel, identity(case), case['arm']] = first['p'][i].copy()
        for source in files:
            provenance[str(source)] = digest(source)
            with np.load(source) as data:
                state = {key: data[key].copy() for key in data.files}
            step = int(source.stem); time = step*float(manifest['eta'])
            for i, case in enumerate(cases):
                w = (state['p'].shape[1]-1)//3
                row = {k: case.get(k) for k in ('target', 'seed', 'cohort', 'source_index', 'original_case_id', 'arm', 'scale', 'reference', 'start', 'h')}
                row.update(panel=panel, width=w, step=step, flow_time=time, eta=manifest['eta'],
                    failed=bool(state['failed'][i]), count=int(state['count'][i]),
                    resolved=bool(state['resolved'][i]))
                key = panel, identity(case), case['arm'], time
                snapshots[key] = (row, state['p'][i, :w].copy(), first['p'][i, :w].copy())
                if row['failed'] or not row['resolved'] or row['count'] != step:
                    row['status'] = 'failed_or_incomplete'
                    failures.append(dict(row)); rows.append(row); continue
                start_mean = float(first['lambda_mean'][i]); current_mean = float(state['lambda_mean'][i])
                change = current_mean-start_mean
                forecast_mean = float(np.mean(state['forecast_lambda'][i]))
                row.update(status='valid', initial_lambda_mean=start_mean, lambda_mean=current_mean,
                    initial_lambda_rms=float(first['lambda_rms'][i]), lambda_rms=float(state['lambda_rms'][i]),
                    initial_lambda_max=float(first['lambda_quantiles'][i, -1]), lambda_max=float(state['lambda_quantiles'][i, -1]),
                    initial_F_norm=float(first['F_norm'][i]), F_norm=float(state['F_norm'][i]),
                    lambda_change=change, lambda_change_pct=ratio(100*change, start_mean),
                    initial_force_forecast_lambda_change=forecast_mean-start_mean,
                    initial_force_forecast_change_pct=ratio(100*(forecast_mean-start_mean), start_mean),
                    forecast_vector_absolute_error=float(state['forecast_absolute_error'][i]),
                    forecast_vector_relative_error=float(state['forecast_relative_error'][i]),
                    initial_Fa_norm=float(first['Fa_norm'][i]), Fa_norm=float(state['Fa_norm'][i]),
                    Fa_gain_own_start=ratio(float(state['Fa_norm'][i]), float(first['Fa_norm'][i])),
                    Ra_norm=float(state['Ra_norm'][i]),
                    tracking_to_effective_norm=ratio(float(state['Ra_norm'][i]), float(state['Fa_norm'][i])),
                    loss=float(state['loss'][i]), closure_error=float(state['closure_error'][i]),
                    positive_travel=float(np.mean(state['positive'][i])), negative_travel=float(np.mean(state['negative'][i])),
                    crossing=float(np.mean(state['crossing'][i])),
                    initially_above_lambda025=int(np.sum(first['first_hit'][i] == 0)),
                    newly_hit_lambda025=int(np.sum(state['first_hit'][i] > 0)))
                for j, channel in enumerate(('effective', 'direct_fine', 'compensation', 'tracking')):
                    row[channel+'_initial_signed_mean_rate'] = float(first['signed_mean_rates'][i, j])
                    row[channel+'_signed_mean_rate'] = float(state['signed_mean_rates'][i, j])
                    row[channel+'_outward_mean_rate'] = float(state['outward_mean_rates'][i, j])
                    if j:
                        row[channel+'_signed_travel'] = float(np.mean(state['signed'][i, j-1]))
                        row[channel+'_absolute_radial_travel'] = float(np.mean(state['absolute_radial'][i, j-1]))
                        row[channel+'_norm_integral'] = float(state['norm_integral'][i, j-1])
                row['effective_signed_travel'] = row['direct_fine_signed_travel']+row['compensation_signed_travel']
                row['effective_absolute_radial_travel'] = float(np.mean(state['effective_absolute_radial'][i]))
                row['effective_norm_integral'] = float(state['effective_norm_integral'][i])
                row['tracking_absolute_to_effective'] = ratio(row['tracking_absolute_radial_travel'], row['effective_absolute_radial_travel'])
                row['tracking_integrated_norm_to_effective'] = ratio(row['tracking_norm_integral'], row['effective_norm_integral'])
                row['effective_signed_rate_change'] = row['effective_signed_mean_rate']-row['effective_initial_signed_mean_rate']
                for j, degree in enumerate((2, 3)):
                    for field in ('generated', 'target', 'generated_signed_rate', 'target_signed_rate'):
                        name = 'q'+str(degree)+'_'+field
                        row[name] = float(state['q23_'+field][i, j])
                        row[name+'_initial'] = float(first['q23_'+field][i, j])
                        row[name+'_change'] = row[name]-row[name+'_initial']
                rows.append(row)
    pairs = []
    for key, (row, point, start) in snapshots.items():
        if row.get('status') != 'valid':
            continue
        panel, case_id, arm, time = key
        original = snapshots.get((panel, case_id, 'original', time))
        if original is not None:
            originally_above = float(row['h'])*abs(original[2]) >= .25
            branch_above = float(row['h'])*abs(start) >= .25
            row['injected_newly_above_lambda025'] = int(np.sum(branch_above & ~originally_above))
            row['originally_above_lambda025'] = int(np.sum(originally_above))
        base = snapshots.get((panel, case_id, 'repaired', time))
        if base and base[0].get('status') == 'valid':
            baseline = base[0]
            row['initial_Fa_gain_repaired'] = ratio(row['initial_Fa_norm'], baseline['initial_Fa_norm'])
            row['later_Fa_gain_repaired'] = ratio(row['Fa_norm'], baseline['Fa_norm'])
            row['lambda_gap_repaired'] = row['lambda_mean']-baseline['lambda_mean']
            row['initial_lambda_gap_repaired'] = row['initial_lambda_mean']-baseline['initial_lambda_mean']
            row['gap_retention'] = ratio(row['lambda_gap_repaired'], row['initial_lambda_gap_repaired'])
        elif arm != 'repaired':
            row['baseline_status'] = 'missing_or_failed_repaired'
        if arm.endswith('_primary'):
            other = snapshots.get((panel, case_id, arm.replace('_primary', '_inverse'), time))
            pair = dict(panel=panel, target=row['target'], seed=row['seed'], source_index=row['source_index'], scale=row['scale'], flow_time=time)
            if other and other[0].get('status') == 'valid':
                inverse = other[0]
                pair.update(status='paired', primary_minus_inverse_motion=row['lambda_change']-inverse['lambda_change'],
                    primary_minus_inverse_motion_pct=row['lambda_change_pct']-inverse['lambda_change_pct'],
                    primary_minus_inverse_signed_effective_rate=row['effective_signed_mean_rate']-inverse['effective_signed_mean_rate'],
                    primary_to_inverse_initial_force=ratio(row['initial_Fa_norm'], inverse['initial_Fa_norm']),
                    primary_to_inverse_later_force=ratio(row['Fa_norm'], inverse['Fa_norm']))
            else:
                pair['status'] = 'missing_or_failed_inverse'
            pairs.append(pair)
    halfstep = []
    for key, (row, point, start) in snapshots.items():
        panel, case_id, arm, time = key
        if panel != 'halfstep_run' or row.get('status') != 'valid':
            continue
        primary = snapshots.get(('late_development_run', case_id, arm, time))
        record = dict(target=row['target'], seed=row['seed'], source_index=row['source_index'], arm=arm, flow_time=time)
        if primary and primary[0].get('status') == 'valid':
            main, mainpoint, mainstart = primary
            if (prepared_hashes[panel] != prepared_hashes['late_development_run']
                    or not np.array_equal(initial_states[panel, case_id, arm],
                                          initial_states['late_development_run', case_id, arm])):
                raise ValueError('Step-halving initial states differ')
            error = float(np.linalg.norm(point-mainpoint)); motion = float(np.linalg.norm(mainpoint-mainstart))
            record.update(status='paired', slope_endpoint_difference=error, slope_endpoint_relative_to_primary_motion=ratio(error, motion),
                half_minus_primary_lambda_change=row['lambda_change']-main['lambda_change'],
                half_minus_primary_change_pct=row['lambda_change_pct']-main['lambda_change_pct'])
        else:
            record['status'] = 'missing_primary_flow_time'
        halfstep.append(record)
    endpoints = [row for row in rows if row.get('status') == 'valid' and row['flow_time'] == 40.]
    grouped = defaultdict(list)
    for row in endpoints:
        grouped[(row['panel'], row['arm'])].append(row)
    attempted_groups = defaultdict(list)
    for row in attempts:
        attempted_groups[(row['panel'], row['arm'])].append(row)
    summary = {}
    metrics = ('lambda_change', 'lambda_change_pct', 'initial_force_forecast_change_pct', 'initial_Fa_gain_repaired',
        'Fa_gain_own_start', 'later_Fa_gain_repaired', 'effective_initial_signed_mean_rate', 'effective_signed_mean_rate',
        'effective_signed_rate_change', 'tracking_signed_travel', 'tracking_absolute_to_effective',
        'tracking_integrated_norm_to_effective', 'gap_retention', 'closure_error',
        *(f'q{degree}_{field}_change' for degree in (2, 3) for field in ('generated', 'target', 'generated_signed_rate', 'target_signed_rate')))
    for key, attempted in attempted_groups.items():
        selected = grouped[key]
        summary[str(key)] = dict(n=len(selected), attempted=len(attempted),
            skipped_repair_count=sum(r['skipped'] for r in attempted),
            missing_or_failed_endpoint_count=sum(not r['skipped'] for r in attempted)-len(selected),
            inward_count=sum(r['lambda_change'] < 0 for r in selected),
            metrics={field: stats(selected, field) for field in metrics})
    pair_groups = defaultdict(list)
    for row in pairs:
        if row['flow_time'] == 40.:
            pair_groups[(row['panel'], row['scale'])].append(row)
    pair_summary = {str(key): dict(n=len(values), paired=sum(r['status'] == 'paired' for r in values),
        metrics={field: stats(values, field) for field in ('primary_minus_inverse_motion_pct',
            'primary_minus_inverse_signed_effective_rate', 'primary_to_inverse_initial_force', 'primary_to_inverse_later_force')})
        for key, values in pair_groups.items()}
    write_csv(output/'states.csv', rows); write_csv(output/'endpoints.csv', endpoints)
    write_csv(output/'attempts.csv', attempts); write_csv(output/'failures.csv', failures)
    write_csv(output/'reference_pairs.csv', pairs); write_csv(output/'halfstep.csv', halfstep)
    result = dict(arms=summary, reference_pairs=pair_summary,
        attempted_branches=len(attempts), skipped_branches=sum(r['skipped'] for r in attempts),
        failure_records=len(failures), rows=len(rows), endpoint_rows=len(endpoints),
        halfstep_checks={key: stats(halfstep, key) for key in ('slope_endpoint_difference', 'slope_endpoint_relative_to_primary_motion', 'half_minus_primary_change_pct')},
        ratio_absolute_floor=RATIO_FLOOR, source_sha256=digest(Path(__file__)), input_hashes=provenance,
        notes=['Primary and half-step attempt counts are separate repetitions.',
               'Fa norm growth is not signed outward reinforcement.',
               'Signed direct and compensation add; their absolute or positive parts do not.',
               'Gap retention is not actual contraction; actual change is relative to each branch start.',
               'No positive first-hit counter includes the injected initial population.'])
    (output/'summary.json').write_text(json.dumps(clean(result), indent=2, allow_nan=False))
    plots(output, endpoints, rows)


def plots(output, rows, all_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, LogLocator, MaxNLocator, NullLocator, SymmetricalLogLocator
    panels = [('Late W=177', lambda r: r['panel'].startswith('late_')),
              ('Wide W=705', lambda r: r['panel'] == 'wide_N512_run'),
              ('Wide W=1409', lambda r: r['panel'] == 'wide_N1024_run')]
    colors = dict(zip(ARMS, plt.cm.tab10.colors[:len(ARMS)]))
    def equality(ax):
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.plot([lo, hi], [lo, hi], color='gray', linestyle='--', lw=.8, zorder=0)
    def sparse_symlog(ax, threshold):
        for axis in (ax.xaxis, ax.yaxis):
            locator = SymmetricalLogLocator(base=10, linthresh=threshold)
            locator.set_params(numticks=5)
            axis.set_major_locator(locator)
            axis.set_minor_locator(NullLocator())
        ax.tick_params(axis='both', labelsize=8)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (label, predicate) in zip(axes, panels):
        for arm in ARMS:
            selected = [r for r in rows if predicate(r) and r['arm'] == arm and r.get('lambda_change_pct') is not None and r.get('initial_force_forecast_change_pct') is not None]
            ax.scatter([r['initial_force_forecast_change_pct'] for r in selected], [r['lambda_change_pct'] for r in selected], s=18, alpha=.65, color=colors[arm], label=LABELS[arm])
        ax.axhline(0, color='gray', lw=.6); ax.axvline(0, color='gray', lw=.6)
        ax.set_xscale('symlog', linthresh=.01); ax.set_yscale('symlog', linthresh=.01)
        equality(ax)
        sparse_symlog(ax, .01)
        ax.set_title(label); ax.set_xlabel('Own initial-force forecast (% scale change)')
    axes[0].set_ylabel('Actual post-kick mean-scale change (%)'); axes[-1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(output/'motion_vs_initial_force.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for column, (label, predicate) in enumerate(panels):
        for arm in ARMS:
            selected = [r for r in rows if predicate(r) and r['arm'] == arm]
            for row in selected:
                if row.get('initial_Fa_gain_repaired') is not None and row.get('Fa_gain_own_start') is not None:
                    axes[0, column].scatter(row['initial_Fa_gain_repaired'], row['Fa_gain_own_start'], s=17, alpha=.6, color=colors[arm])
                axes[1, column].scatter(row['effective_initial_signed_mean_rate'], row['effective_signed_mean_rate'], s=17, alpha=.6, color=colors[arm])
        axes[0, column].set_xscale('log'); axes[0, column].set_yscale('log'); axes[0, column].axhline(1, color='gray', lw=.7)
        axes[0, column].xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        axes[0, column].xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        axes[0, column].xaxis.set_minor_locator(NullLocator())
        axes[0, column].yaxis.set_minor_locator(NullLocator())
        if column > 0:
            axes[0, column].yaxis.set_major_locator(MaxNLocator(nbins=4))
            axes[0, column].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        else:
            axes[0, column].yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        axes[0, column].tick_params(axis='both', labelsize=8)
        axes[0, column].set_title(label); axes[0, column].set_xlabel('Initial force norm / repaired baseline')
        axes[1, column].set_xscale('symlog', linthresh=1e-10); axes[1, column].set_yscale('symlog', linthresh=1e-10)
        axes[1, column].axhline(0, color='gray', lw=.6); axes[1, column].axvline(0, color='gray', lw=.6)
        axes[1, column].set_xlabel('Initial signed effective mean-scale rate')
        equality(axes[1, column])
        sparse_symlog(axes[1, column], 1e-10)
        # Retain informative outer decades; a generic sparse locator can leave
        # only the near-zero ticks in the wide panels.
        candidates = ((-1e-4, -1e-7, -1e-10, 0, 1e-10, 1e-7, 1e-4),
                      (-1e-7, -1e-9, 0, 1e-9, 1e-7),
                      (-1e-8, -1e-10, 0, 1e-10, 1e-8))[column]
        limits = axes[1, column].get_xlim()
        ticks = [value for value in candidates if limits[0] <= value <= limits[1]]
        axes[1, column].set_xticks(ticks)
        axes[1, column].set_yticks(ticks)
    axes[0, 0].set_ylabel('Final force norm / own initial norm')
    axes[1, 0].set_ylabel('Final signed effective mean-scale rate')
    handles = [Line2D([], [], color=colors[arm], marker='o', linestyle='none', label=LABELS[arm]) for arm in ARMS]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, 1)); fig.savefig(output/'force_norm_and_signed_rates.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    dilation_arms = ARMS[2:]
    for ax, (label, predicate) in zip(axes, panels):
        for j, arm in enumerate(dilation_arms):
            selected = [r for r in rows if predicate(r) and r['arm'] == arm and r.get('gap_retention') is not None]
            ax.scatter(np.full(len(selected), j), [r['gap_retention'] for r in selected], s=17, alpha=.55, color=colors[arm])
        ax.axhline(1, color='gray', linestyle='--'); ax.set_title(label)
        ax.set_xticks(range(len(dilation_arms)), [LABELS[arm] for arm in dilation_arms], rotation=25, fontsize=7)
    axes[0].set_ylabel('Final gap to repaired baseline / initial gap')
    fig.tight_layout(); fig.savefig(output/'gap_retention.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6))
    trajectory_arms = ('repaired', *dilation_arms)
    for ax, (label, predicate) in zip(axes, panels):
        for arm in trajectory_arms:
            groups = defaultdict(list)
            for row in all_rows:
                if predicate(row) and row.get('status') == 'valid' and row['arm'] == arm and row.get('lambda_change_pct') is not None:
                    groups[row['flow_time']].append(row)
            times = sorted(groups)
            if not times:
                continue
            medians = [np.median([r['lambda_change_pct'] for r in groups[t]]) for t in times]
            lower = [np.percentile([r['lambda_change_pct'] for r in groups[t]], 25) for t in times]
            upper = [np.percentile([r['lambda_change_pct'] for r in groups[t]], 75) for t in times]
            forecast = [np.median([r['initial_force_forecast_change_pct'] for r in groups[t]]) for t in times]
            ax.plot(times, medians, color=colors[arm], label=LABELS[arm])
            ax.fill_between(times, lower, upper, color=colors[arm], alpha=.10)
            ax.plot(times, forecast, color=colors[arm], linestyle='--', lw=1)
        ax.axhline(0, color='gray', lw=.6); ax.set_xlim(0, 40)
        ax.set_yscale('symlog', linthresh=.01); ax.set_title(label, fontsize=12)
        ax.set_xlabel('Flow time η × updates', fontsize=12)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.tick_params(axis='both', labelsize=11)
    axes[0].set_ylabel('Mean-scale change from own start (%)', fontsize=12)
    handles = [Line2D([], [], color=colors[arm], label=LABELS[arm]) for arm in trajectory_arms]
    handles += [Line2D([], [], color='black', linestyle='--', label='Own initial-force forecast')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=11)
    fig.suptitle('Median trajectories; shaded interquartile range of valid branches', fontsize=12)
    fig.tight_layout(rect=(0, .16, 1, .94)); fig.savefig(output/'scale_trajectories.png', dpi=170); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    run(parser.parse_args())
