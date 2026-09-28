"""Plots and numerical facts from exact-state population/output audit CSVs."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .mechanism_dilation_analysis import clean, digest, stats


def load(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value in ('True', 'False'):
                row[key] = value == 'True'
            elif value in ('', 'None', 'null'):
                row[key] = None
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def finite(row, key):
    return isinstance(row.get(key), (float, int)) and np.isfinite(row[key])


def ratio(row, numerator, denominator):
    return (row[numerator]/row[denominator] if finite(row, numerator) and
            finite(row, denominator) and row[denominator] != 0 else None)


def usable(row, resolved=False):
    return row.get('status') == 'finite' and (not resolved or row.get('resolved') is True)


def role(row):
    if row.get('role') == 'static':
        return 'static'
    return 'natural' if row.get('arm') == 'natural' else 'dilation'


def run(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import SymmetricalLogLocator, FuncFormatter
    from matplotlib.lines import Line2D
    args.output.mkdir(parents=True, exist_ok=False)
    rows, duplicates, seen = [], [], set()
    for source in args.states:
        for row in load(source):
            row['source_csv'] = str(source)
            row['analysis_role'] = role(row)
            if role(row) == 'static':
                if row.get('state_sha256') in seen:
                    duplicates.append(row); continue
                seen.add(row.get('state_sha256'))
            row['Q_floor_fraction_actual'] = ratio(row, 'Q_error_floor', 'relative_l2')
            floor_names = ['Q_error_floor', *(f'tail{k}_relative_floor' for k in (2, 3, 5, 9))]
            available = [(name, row[name]) for name in floor_names if finite(row, name)]
            if available:
                row['strongest_error_floor_name'], row['strongest_error_floor'] = max(available, key=lambda item: item[1])
                row['strongest_floor_fraction_actual'] = ratio(row, 'strongest_error_floor', 'relative_l2')
            eval_names = ['eval_Q_error_floor', *(f'eval_tail{k}_relative_floor' for k in (2, 3, 5, 9))]
            available_eval = [(name, row[name]) for name in eval_names if finite(row, name)]
            if available_eval:
                row['strongest_eval_error_floor_name'], row['strongest_eval_error_floor'] = max(available_eval, key=lambda item: item[1])
                row['strongest_eval_floor_fraction_actual'] = ratio(row, 'strongest_eval_error_floor', 'relative_eval_l2')
            row['sensitivity_bound_over_actual'] = ratio(row, 'sensitivity_bound', 'residual_sensitivity')
            row['tracking_output_to_effective'] = ratio(row, 'output_speed_tracking', 'output_speed_effective')
            row['direct_output_to_effective'] = ratio(row, 'output_speed_direct', 'output_speed_effective')
            row['C6_log_rate_effective'] = ratio(row, 'C6_dot_effective', 'C6')
            row['M_log_rate_effective'] = ratio(row, 'M_dot_effective', 'M')
            row['M6_log_rate_effective'] = ratio(row, 'M6_dot_effective', 'M6')
            rows.append(row)
    fields = ('relative_l2', 'Q_error_floor', 'Q_floor_fraction_actual',
              'strongest_error_floor', 'strongest_floor_fraction_actual',
              'strongest_eval_error_floor', 'strongest_eval_floor_fraction_actual',
              'M', 'M6', 'C6', 'Q',
              'capacity_top_decile_share', 'sensitivity_bound', 'residual_sensitivity',
              'sensitivity_bound_over_actual', 'tracking_output_to_effective',
              'direct_output_to_effective', 'fine_loss_dot_effective', 'fine_loss_dot_tracking',
              'fine_loss_dot_full', 'Q_dot_effective', 'Q_dot_tracking', 'Q_dot_full',
              'C6_dot_effective', 'C6_dot_tracking', 'C6_dot_full', 'energy_identity',
              'certificate_flow_time', 'certificate_relative_fine_floor',
              'p8_flow_time', 'p8_relative_fine_floor', 'p12_flow_time', 'p12_relative_fine_floor',
              'hs_certificate_flow_time', 'hs_certificate_relative_fine_floor',
              'hs_certificate_initial_sensitivity_ratio', 'hs_certificate_quadrature_absolute_error',
              'force_certificate_flow_time', 'force_certificate_relative_fine_floor',
              'force_certificate_initial_force_ratio', 'force_certificate_quadrature_absolute_error')
    fields += tuple(f'{quantity}_dot_{channel}' for quantity in ('M', 'M6', 'C6', 'Es', 'Q', 'fine_loss')
                    for channel in ('generated', 'target'))
    for row in rows:
        for quantity in ('M', 'M6', 'C6', 'Es', 'Q', 'fine_loss'):
            generated, target = f'{quantity}_dot_generated', f'{quantity}_dot_target'
            effective = f'{quantity}_dot_effective'
            if all(finite(row, field) for field in (generated, target, effective)) and usable(row, True):
                total = abs(row[generated])+abs(row[target])
                row[quantity+'_generated_target_retained_fraction'] = abs(row[effective])/total if total > 0 else None
                row[quantity+'_split_closure_error'] = abs(row[effective]-row[generated]-row[target])
    fields += tuple(f'{quantity}_{suffix}' for quantity in ('M', 'M6', 'C6', 'Es', 'Q', 'fine_loss')
                    for suffix in ('generated_target_retained_fraction', 'split_closure_error'))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['analysis_role'], row.get('width'), row.get('start'))].append(row)
    facts = []
    for key, group in grouped.items():
        good = [r for r in group if usable(r)]
        resolved = [r for r in group if usable(r, True)]
        facts.append(dict(role=key[0], width=key[1], restart=key[2], states=len(group),
                          finite=len(good), resolved=len(resolved),
                          targets=sorted({r['target'] for r in group}),
                          statistics={field: stats(good, field) for field in fields},
                          output_floor_coverage={field: dict(
                              measured=sum(finite(r, field) for r in good),
                              positive=sum(r[field] > 0 for r in good if finite(r, field)),
                              positive_fraction=(sum(r[field] > 0 for r in good if finite(r, field)) /
                                                 sum(finite(r, field) for r in good))
                                                if any(finite(r, field) for r in good) else None)
                              for field in ('Q_error_floor', 'strongest_error_floor', 'strongest_eval_error_floor')},
                          signs={field: dict(positive=sum(r[field] > 0 for r in resolved if finite(r, field)),
                                             negative=sum(r[field] < 0 for r in resolved if finite(r, field)))
                                 for field in ('Q_dot_effective', 'Q_dot_tracking', 'C6_dot_effective',
                                               'fine_loss_dot_effective', 'fine_loss_dot_tracking',
                                               'Q_dot_generated', 'Q_dot_target',
                                               'C6_dot_generated', 'C6_dot_target')},
                          signed_mechanisms={quantity: dict(
                              measured=sum(all(finite(r, f'{quantity}_dot_{c}') for c in ('generated', 'target')) for r in resolved),
                              generated_restrains_target_builds=sum(
                                  r[f'{quantity}_dot_generated'] < 0 < r[f'{quantity}_dot_target']
                                  for r in resolved if all(finite(r, f'{quantity}_dot_{c}') for c in ('generated', 'target'))),
                              generated_builds_target_restrains=sum(
                                  r[f'{quantity}_dot_target'] < 0 < r[f'{quantity}_dot_generated']
                                  for r in resolved if all(finite(r, f'{quantity}_dot_{c}') for c in ('generated', 'target'))),
                              both_build=sum(min(r[f'{quantity}_dot_generated'], r[f'{quantity}_dot_target']) > 0
                                  for r in resolved if all(finite(r, f'{quantity}_dot_{c}') for c in ('generated', 'target'))),
                              both_restrain=sum(max(r[f'{quantity}_dot_generated'], r[f'{quantity}_dot_target']) < 0
                                  for r in resolved if all(finite(r, f'{quantity}_dot_{c}') for c in ('generated', 'target'))))
                              for quantity in ('Q', 'C6', 'M6')}))
    colors = {width: f'C{i}' for i, width in enumerate(sorted({r['width'] for r in rows if finite(r, 'width')}))}

    def sparse_signed_ticks(ax, threshold=1e-12):
        def label(value, position):
            if value == 0:
                return '0'
            if abs(value) <= threshold*1.01:
                return ''
            sign = '-' if value < 0 else ''
            return rf'${sign}10^{{{int(round(np.log10(abs(value))))}}}$'
        for axis in (ax.xaxis, ax.yaxis):
            locator = SymmetricalLogLocator(base=10, linthresh=threshold)
            locator.set_params(numticks=5)
            axis.set_major_locator(locator)
            axis.set_major_formatter(FuncFormatter(label))
        ax.tick_params(labelsize=8)

    def scatter(ax, group, xkey, ykey, resolved=False):
        for width, color in colors.items():
            chosen = [r for r in group if usable(r, resolved) and r.get('width') == width
                      and finite(r, xkey) and finite(r, ykey)]
            ax.scatter([r[xkey] for r in chosen], [r[ykey] for r in chosen],
                       color=color, s=15, alpha=.55, label=f'W={int(width)}; n={len(chosen)}')
        ax.grid(alpha=.2)

    for which in ('static', 'natural', 'dilation'):
        group = [r for r in rows if r['analysis_role'] == which]
        if not group:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        ax = axes[0, 0]; scatter(ax, group, 'relative_l2', 'Q_error_floor')
        ax.plot([0, 1.5], [0, 1.5], ':', color='.4')
        ax.set(xlabel='Actual raw relative L2 error', ylabel='Q-based error lower bound')
        ax = axes[0, 1]; scatter(ax, group, 'Q', 'output_fine_norm')
        ax.set(xscale='symlog', yscale='symlog', xlabel='Readout-weighted capacity Q', ylabel='Actual nonlinear output norm')
        ax = axes[0, 2]; scatter(ax, group, 'residual_sensitivity', 'sensitivity_bound', True)
        ax.set(xscale='log', yscale='log', xlabel='Current residual sensitivity', ylabel='Collective sensitivity upper bound')
        ax = axes[1, 0]; scatter(ax, group, 'M_log_rate_effective', 'M6_log_rate_effective', True)
        ax.set(xscale='symlog', yscale='symlog', xlabel='Effective d(log M)/dt', ylabel='Effective d(log M6)/dt')
        ax = axes[1, 1]; scatter(ax, group, 'Q', 'capacity_top_decile_share')
        ax.set(xscale='symlog', xlabel='Readout-weighted capacity Q', ylabel='Q share held by top decile')
        ax = axes[1, 2]; scatter(ax, group, 'fine_loss_dot_effective', 'fine_loss_dot_tracking', True)
        ax.axhline(0, color='.5', lw=.7); ax.axvline(0, color='.5', lw=.7)
        ax.set(xscale='symlog', yscale='symlog', xlabel='Effective fine-loss rate', ylabel='Tracking fine-loss rate')
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f'{which}: exact evolving-state population/output diagnostics\n'
                     f'{len(group)} states; {len({r["target"] for r in group})} targets; force panels require resolved coarse solve')
        fig.savefig(args.output/f'{which}_population_output.png', dpi=180); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for ax, xkey, floor_key, title in (
                (axes[0], 'relative_l2', 'strongest_error_floor', 'Training-grid error'),
                (axes[1], 'relative_eval_l2', 'strongest_eval_error_floor', 'Independent-grid error')):
            scatter(ax, group, xkey, floor_key)
            measured = [r for r in group if usable(r) and finite(r, xkey) and finite(r, floor_key)]
            limit = max([r[xkey] for r in measured]+[1.])
            ax.plot([0, limit], [0, limit], ':', color='.4', label='Equality')
            ax.set(xlabel='Actual raw relative L2 error', ylabel='Strongest prelisted output-error lower bound',
                   title=f'{title}; n={len(measured)}')
        axes[0].legend(fontsize=7)
        fig.suptitle(f'{which}: maximum of valid output lower bounds\n'
                     'Q and degree 2/3/5/9 tails, evaluated on each grid separately. All states retained.')
        fig.savefig(args.output/f'{which}_strongest_output_floors.png', dpi=180); plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        scatter(axes[0], group, 'output_speed_effective', 'output_speed_tracking', True)
        scatter(axes[1], group, 'output_speed_direct', 'output_speed_compensation', True)
        scatter(axes[2], group, 'Q_dot_effective', 'Q_dot_tracking', True)
        for ax, labels in zip(axes, (('Effective output speed', 'Tracking output speed'),
                                    ('Direct fine output speed', 'Compensation output speed'),
                                    ('Effective Q rate', 'Tracking Q rate'))):
            ax.set(xlabel=labels[0], ylabel=labels[1])
        for ax in axes[:2]:
            ax.set(xscale='log', yscale='log')
        axes[2].set_xscale('symlog', linthresh=1e-12)
        axes[2].set_yscale('symlog', linthresh=1e-12)
        axes[0].legend(fontsize=7)
        fig.suptitle(which+': exact instantaneous output and population rates')
        fig.savefig(args.output/f'{which}_rates.png', dpi=180); plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for column, quantity in enumerate(('Q', 'C6', 'M6')):
            ax = axes[0, column]
            scatter(ax, group, f'{quantity}_dot_target', f'{quantity}_dot_generated', True)
            ax.axhline(0, color='.5', lw=.7); ax.axvline(0, color='.5', lw=.7)
            ax.set_xscale('symlog', linthresh=1e-12); ax.set_yscale('symlog', linthresh=1e-12)
            sparse_signed_ticks(ax)
            ax.set(xlabel=f'Target contribution to {quantity} rate',
                   ylabel=f'Generated-output contribution to {quantity} rate',
                   title=quantity+' production and correction')
            ax = axes[1, column]
            scatter(ax, group, quantity, quantity+'_generated_target_retained_fraction', True)
            ax.set(xscale='log', xlabel=quantity, ylabel='|Effective rate| / (|generated| + |target|)',
                   ylim=(-.03, 1.03))
        fig.suptitle(f'{which}: full-fine generated-output and target loading through the same compensated sensitivity\n'
                     'Lower right: generated correction restrains target loading. Exact current-state split; no modal truncation.')
        present = sorted({r['width'] for r in group if usable(r, True) and finite(r, 'width')})
        axes[0, 2].legend(handles=[Line2D([0], [0], marker='o', ls='', color=colors[width],
                                       label=f'W={int(width)}') for width in present], fontsize=8)
        fig.savefig(args.output/f'{which}_generated_target.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    image = None
    for ax, width, restart, label in ((axes[0], 1409, 20000, 'Wide population at 20,000 updates'),
                                     (axes[1], 177, 600000, 'Narrow population at 600,000 updates')):
        chosen = [r for r in rows if r['analysis_role'] == 'static' and usable(r, True)
                  and r.get('width') == width and r.get('start') == restart
                  and all(finite(r, key) for key in ('Q_dot_generated', 'Q_dot_target',
                                                   'Q_generated_target_retained_fraction'))]
        if not chosen:
            ax.set_visible(False); continue
        xx = np.array([r['Q_dot_target'] for r in chosen])
        yy = np.array([r['Q_dot_generated'] for r in chosen])
        retained = [r['Q_generated_target_retained_fraction'] for r in chosen]
        image = ax.scatter(xx, yy, c=retained, cmap='viridis', vmin=0, vmax=1,
                           s=35, edgecolors='white', linewidths=.3, zorder=3)
        amplitude = max(np.max(abs(xx)), np.max(abs(yy)), 1e-300)
        threshold = 10**(np.floor(np.log10(amplitude))-5)
        ax.set_xscale('symlog', linthresh=threshold); ax.set_yscale('symlog', linthresh=threshold)
        sparse_signed_ticks(ax, threshold)
        extent = max(np.max(abs(xx)), np.max(abs(yy)))*1.2
        ax.plot([-extent, extent], [extent, -extent], '--', color='.5', lw=1,
                label='Exact cancellation')
        ax.axhline(0, color='.7', lw=.6); ax.axvline(0, color='.7', lw=.6)
        ax.set(xlabel='Target contribution to Q growth', ylabel='Generated-output contribution to Q growth',
               title=f'{label}\nW={width}; {len(chosen)} states; {len({r["target"] for r in chosen})} targets')
        ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=.15)
    if image is not None:
        bar = fig.colorbar(image, ax=axes, shrink=.8, ticks=[0, .5, 1])
        bar.set_label('Fraction remaining in net Q rate\n0 = cancellation; 1 = no cancellation')
        fig.suptitle('The same population equation has different balances\n'
                     'Exact full-fine decomposition; rates per unit gradient-flow time')
        fig.savefig(args.output/'population_correction_regimes.png', dpi=180)
    plt.close(fig)

    static = [r for r in rows if r['analysis_role'] == 'static' and usable(r, True)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    certificate_counts = {}
    for prefix, label, color, marker in (('certificate_', 'sixth-moment flow enclosure', 'C0', 'o'),
                                       ('p8_', 'eighth-moment flow enclosure', 'C1', 's'),
                                       ('p12_', 'twelfth-moment flow enclosure', 'C2', '^'),
                                       ('hs_certificate_', 'evolving fine-sensitivity enclosure', 'C3', 'D'),
                                       ('force_certificate_', 'evolving residual-aligned force enclosure', 'C4', 'v')):
        chosen = [r for r in static if finite(r, prefix+'flow_time') and r[prefix+'flow_time'] > 0]
        certificate_counts[label] = len(chosen)
        axes[0].scatter([r['width'] for r in chosen], [r[prefix+'flow_time'] for r in chosen],
                        s=16, alpha=.4, color=color, marker=marker, label=label)
        axes[1].scatter([r[prefix+'flow_time'] for r in chosen], [r[prefix+'relative_fine_floor'] for r in chosen],
                        s=16, alpha=.4, color=color, marker=marker, label=label)
    axes[0].set(xlabel='Width', ylabel='Available physical flow time', xscale='log', yscale='log')
    axes[1].set(xlabel='Available physical flow time', ylabel='Guaranteed fraction of initial fine error', xscale='log')
    axes[0].legend(fontsize=7); fig.suptitle('Effective-flow initial-data enclosures: evolving sensitivity, not a frozen forecast\n'
                                         'FP64 audit and numerical quadrature; not interval certification or discrete GD')
    fig.savefig(args.output/'effective_flow_enclosures.png', dpi=180); plt.close(fig)

    trajectories = defaultdict(list)
    for row in rows:
        if row['analysis_role'] != 'static' and usable(row):
            trajectories[tuple(row.get(k) for k in ('analysis_role', 'panel', 'target', 'seed', 'start', 'arm'))].append(row)
    drift_facts = []
    for which in ('natural', 'dilation'):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        count = 0
        for key, values in trajectories.items():
            if key[0] != which:
                continue
            values.sort(key=lambda r: r['horizon']); first = values[0]; last = values[-1]
            if first['horizon'] != 0 or len(values) < 2:
                continue
            count += 1
            record = dict(role=which, panel=key[1], target=key[2], seed=key[3], start=key[4], arm=key[5],
                          width=first['width'], final_horizon=last['horizon'])
            for ax, field in zip(axes, ('C6', 'Q', 'relative_l2')):
                if first[field] > 0:
                    changes = np.array([r[field]/first[field] for r in values])
                    ax.plot([r['horizon'] for r in values], changes, alpha=.15,
                            color=colors[first['width']])
                    record[field+'_final_ratio'] = float(changes[-1])
                    record[field+'_max_ratio'] = float(max(changes))
            drift_facts.append(record)
        for ax, field in zip(axes, ('C6', 'Q', 'relative_l2')):
            ax.axhline(1, color='.5', lw=.7); ax.set(xscale='symlog', yscale='log', xlabel='Additional GD updates',
                                                  ylabel=field+' / own initial value'); ax.grid(alpha=.2)
            ax.set_xlim(left=0)
        present_widths = sorted({r['width'] for r in drift_facts if r['role'] == which})
        axes[0].legend(handles=[Line2D([0], [0], color=colors[width], label=f'W={int(width)}')
                                for width in present_widths], fontsize=8)
        fig.suptitle(f'{which}: observed population drift; {count} finite retained trajectories')
        if count:
            fig.savefig(args.output/f'{which}_drift.png', dpi=180)
        plt.close(fig)
    certificate_rows = load(args.gd_certificates) if args.gd_certificates else []
    gd_groups = []
    if certificate_rows:
        certificate_panels = defaultdict(list)
        for row in certificate_rows:
            certificate_panels[role(row)].append(row)
        for label, values in certificate_panels.items():
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
            for width in sorted({r['width'] for r in values if finite(r, 'width')}):
                chosen = [r for r in values if r.get('width') == width]
                status_counts = {status: sum(r.get('status') == status for r in chosen)
                                 for status in sorted({str(r.get('status')) for r in chosen})}
                gd_groups.append(dict(role=label, width=width, states=len(chosen),
                                      targets=sorted({r['target'] for r in chosen}),
                                      statuses=status_counts,
                                      accepted=stats(chosen, 'accepted_updates'),
                                      half_initial_error=stats(chosen, 'half_initial_fine_error_through_update')))
                for ax, field in zip(axes[:2], ('accepted_updates', 'half_initial_fine_error_through_update')):
                    valid = [r for r in chosen if finite(r, field)]
                    ax.scatter([width]*len(valid), [r[field] for r in valid], s=18, alpha=.45,
                               color=colors.get(width, 'C0'), label=f'W={int(width)}, n={len(valid)}')
                    if valid:
                        ax.scatter([width], [np.median([r[field] for r in valid])], s=65,
                                   marker='_', color='black', zorder=3)
            axes[0].set(xlabel='Width', ylabel='Updates within initial-data enclosure')
            axes[1].set(xlabel='Width', ylabel='Updates retaining at least half initial fine error')
            for ax in axes[:2]:
                ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1); ax.grid(alpha=.2)
            statuses = sorted({str(r.get('status')) for r in values})
            counts = [sum(str(r.get('status')) == status for r in values) for status in statuses]
            axes[2].barh(np.arange(len(statuses)), counts)
            axes[2].set(yticks=np.arange(len(statuses)), yticklabels=[s.replace('_', ' ') for s in statuses],
                        xlabel='States (including failed enclosure conditions)', title='Stopping or ineligibility reason')
            axes[2].tick_params(axis='y', labelsize=8)
            fig.suptitle(f'{label}: ordinary-GD initial-data recurrence enclosures\n'
                         'FP64 arithmetic audit, not interval certification; dots=states, black ticks=medians')
            fig.savefig(args.output/f'{label}_ordinary_gd_enclosures.png', dpi=180); plt.close(fig)
    source_paths = [*args.states, *([args.gd_certificates] if args.gd_certificates else [])]
    result = dict(sources={str(p): digest(p) for p in source_paths}, rows=len(rows),
                  duplicate_static_states=len(duplicates), groups=facts, trajectory_changes=drift_facts,
                  flow_certificate_counts=certificate_counts, gd_certificate_rows=certificate_rows,
                  gd_certificate_groups=gd_groups,
                  generated_target_split='Full fine output versus full fine target through the same compensated sensitivity; exact current-state derivative attribution',
                  output_floor_rule='Pointwise maximum of prelisted valid Q and degree2/3/5/9 tail lower bounds, on each grid separately. No cases selected by positive result.',
                  source_sha256=digest(Path(__file__)))
    (args.output/'facts.json').write_text(json.dumps(clean(result), indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gd-certificates', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
