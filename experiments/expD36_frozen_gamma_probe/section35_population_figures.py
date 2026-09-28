"""Plot the Section 3.5 discrete-GD population and output-error comparison.

Reads the existing continuation diagnostics; performs no network training.
The native-step recurrence is Eq. (18c) of d34_population_balance_mechanism.md.
Its sampled-coefficient evaluation is checked against all ten saved summaries.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


A7 = (17 / 315) * 2**3.5 * 7**3.5 / 8**4
C7 = 8 * A7
TARGETS = ('moment5', 'mixed_sine', 'gauss_left', 'bump_right', 'step_right', 'kink_abs')
LABELS = ('Degree five', 'Mixed sine', 'Gaussian', 'Bump', 'Smooth step', 'Kink')
COLORS = ('#882255', '#0072B2', '#009E73', '#CC6677', '#D55E00', '#332288')


def read_rows(path):
    with path.open() as stream:
        return [{k: (v if k in ('target', 'kind', 'scalar_complete') else float(v) if v else None)
                 for k, v in row.items()} for row in csv.DictReader(stream)]


def coefficients(row):
    w = row['width']
    remainder = C7 * np.sqrt(row['C14']) / w**3
    return [row['q3']/w, row['q5']/w**2, A7*row['C8']/w**3,
            row['j3']/w, row['j5']/w**2, remainder,
            row['g3']/w, row['g5']/w**2, remainder*row['target_fine'], row['R_norm']]


def evaluate(rows):
    radius = np.sqrt(rows[0]['M'])
    radii = [radius]
    dt = rows[0]['dt']
    for left, right in zip(rows[:-1], rows[1:], strict=True):
        count = round((right['time'] - left['time']) / dt)
        current, end = coefficients(left), coefficients(right)
        increments = [(b-a)/count for a, b in zip(current, end, strict=True)]
        for _ in range(count):
            q3, q5, q7, j3, j5, j7, g3, g5, g7, tracking = current
            r2 = radius*radius; r3 = r2*radius; r4 = r2*r2
            r5 = r3*r2; r6 = r4*r2; r7 = r5*r2; r8 = r6*r2
            speed = ((j3*r3+j5*r5+j7*r7)*(q3*r4+q5*r6+q7*r8)
                     +g3*r3+g5*r5+g7*r7+tracking)
            radius += dt*speed
            current = [v+dv for v, dv in zip(current, increments, strict=True)]
        radii.append(radius)
    radii = np.asarray(radii)
    capacity = np.array([sum(c*r**p for c, p in zip(coefficients(row)[:3], (4, 6, 8)))
                         for row, r in zip(rows, radii, strict=True)])
    actual = np.array([r['relative_error'] for r in rows])
    lower = np.maximum((rows[0]['target_fine']-capacity)/rows[0]['target_norm'], 0)
    mass = np.array([r['M'] for r in rows])
    assert np.all(np.isfinite(radii)) and np.all(radii**2 >= mass-1e-12)
    assert np.all((0 <= lower) & (lower <= actual+1e-12))
    updates = np.rint(np.array([r['time'] for r in rows])/dt).astype(int)
    assert updates[0] == 0 and updates[-1] == 100000
    assert np.all(np.diff(updates) == 500)
    return dict(additional_updates=updates, norm=np.sqrt(mass), norm_bound=radii,
                relative_error=actual, lower_error=lower, bound_fraction=lower/actual,
                lambda_rms=np.array([r['lambda_rms'] for r in rows]),
                lambda_bound=rows[0]['h']*radii/np.sqrt(rows[0]['width']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    groups, summaries, sources = {}, {}, []
    cohorts = [(30, 'balance_flow_20260924', 'balance_evolving_verified_20260924'),
               (31, 'balance_stress_flow_20260924', 'balance_stress_comparison_20260924')]
    for seed, raw, verified in cohorts:
        path = args.evidence / raw / 'states.csv'
        summary_path = args.evidence / verified / 'summary.csv'
        sources.extend([path, summary_path])
        for row in read_rows(path):
            if row['kind'] == 'gd' and row['dt'] == .002:
                assert row['seed'] == seed and row['width'] == 705 and row['start'] == 20000
                groups.setdefault((row['target'], seed), []).append(row)
        for row in read_rows(summary_path):
            if row['kind'] == 'gd' and row['dt'] == .002:
                summaries[(row['target'], seed)] = row
    curves, checks, flat = {}, [], []
    for key, rows in groups.items():
        rows.sort(key=lambda r: r['time'])
        curve = evaluate(rows)
        curves[key] = curve
        saved = summaries[key]
        np.testing.assert_allclose(
            [curve['lower_error'].min(), curve['lambda_bound'][-1], curve['relative_error'][-1]],
            [saved['scalar_output_error_floor_min'], saved['scalar_final_lambda_bound'], saved['relative_error']],
            atol=2e-13, rtol=2e-11)
        assert saved['scalar_complete'] == 'True'
        checks.append(dict(target=key[0], seed=key[1],
                           min_bound_fraction=float(curve['bound_fraction'].min()),
                           max_absolute_error_gap=float(np.max(curve['relative_error']-curve['lower_error'])),
                           min_output_floor=float(curve['lower_error'].min())))
        for i in range(len(rows)):
            flat.append(dict(target=key[0], seed=key[1], **{k: v[i] for k, v in curve.items()}))
    assert len(curves) == 10
    with (args.output/'curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(flat)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                         'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                         'legend.fontsize': 6.5, 'legend.frameon': False,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6, 'lines.linewidth': 1.15, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.85), layout='constrained')
    for target in ('mixed_sine', 'step_right'):
        i = TARGETS.index(target); color = COLORS[i]; curve = curves[(target, 30)]
        x = curve['additional_updates']/1000
        s0 = curve['norm'][0]
        for ax, actual, bound in [(axes[0], curve['norm']/s0, curve['norm_bound']/s0),
                                 (axes[1], curve['relative_error'], curve['lower_error'])]:
            ax.plot(x, actual, color=color, lw=1.2)
            ax.plot(x, bound, color=color if ax is axes[0] else '#333333',
                    ls='--', lw=.95, zorder=3)
            ax.plot(x[::40], actual[::40], color=color, marker='o', ls='none', ms=2.7,
                    mfc='white', mew=.7, zorder=4)
    axes[0].set(ylabel='Joint parameter norm / initial', ylim=(.975, 1.40))
    axes[1].set(ylabel='Relative training error', ylim=(0, 1))
    axes[0].legend(handles=[Line2D([], [], color=COLORS[TARGETS.index(t)], label=LABELS[TARGETS.index(t)])
                            for t in ('mixed_sine', 'step_right')],
                   loc='upper left', handlelength=1.4, labelspacing=.25)
    axes[1].legend(handles=[Line2D([], [], color='#333333', marker='o', mfc='white', ms=2.7, label='Executed GD'),
                            Line2D([], [], color='#333333', ls='--', label='Theorem bound')],
                   loc='lower left', labelspacing=.25)
    for ax in axes:
        ax.set(xlabel='Additional GD updates (thousands)', xlim=(0, 100), xticks=[0, 50, 100])
        ax.grid(axis='y', color='#dddddd', lw=.5); ax.set_axisbelow(True)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output/f'population_output.{suffix}', dpi=600, pad_inches=.02)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 1.85), layout='constrained')
    for i, target in enumerate(TARGETS):
        for seed, style in ((30, '-'), (31, '--')):
            if (target, seed) not in curves:
                continue
            curve = curves[(target, seed)]
            ax.plot(curve['additional_updates']/1000, curve['bound_fraction'],
                    color=COLORS[i], ls=style, label=LABELS[i] if seed == 30 else None)
    ax.axhline(1, color='#777777', lw=.6, ls=':')
    ax.set(xlabel='Additional GD updates (thousands)', ylabel='Error bound / executed error',
           xlim=(0, 100), ylim=(.975, 1.001), xticks=[0, 25, 50, 75, 100])
    ax.grid(axis='y', color='#dddddd', lw=.5); ax.set_axisbelow(True)
    ax.legend(ncol=3, loc='lower left', handlelength=1.5, columnspacing=1.2, labelspacing=.3)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output/f'output_tightness.{suffix}', dpi=600, pad_inches=.02)
    plt.close(fig)
    facts = dict(checks=checks, minimum_sampled_bound_fraction=min(c['min_bound_fraction'] for c in checks),
                 scope='Native-step GD recurrence with interpolated sampled structural coefficients; no certified between-sample enclosure.',
                 selection='Main pair: mixed sine links to Figure 4; smooth step has the largest principal population-growth allowance. Appendix: all ten continuations.',
                 diagnostic_stride_updates=500, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'facts.json').write_text(json.dumps(facts, indent=2)+'\n')
    print(json.dumps(facts, indent=2))


if __name__ == '__main__':
    main()
