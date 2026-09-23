"""Plot completed held-out evidence; never produce report text."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from experiments.expD34_readout_race.effective_feedback_holdout import TARGETS, FAMILIES

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT/'actual_vs_forecast.csv').open()))
modal = list(csv.DictReader((ROOT/'modal_scores.csv').open()))
horizons = [1000, 10000, 50000, 200000]
labels = [f'{FAMILIES[t]} / {t}' for t in TARGETS]
target_summary = []
for target in TARGETS:
    for horizon in horizons:
        for model in ('affine', 'effective_pure', 'effective_with_remainder'):
            subset = [r for r in rows if r['target'] == target and r['arm'] == 'joint'
                      and r['model'] == model and int(r['offset']) == horizon]
            record = dict(target=target, family=FAMILIES[target], offset=horizon, model=model)
            for metric in ('slope_motion_relative_error', 'slope_motion_absolute_error',
                           'slope_motion_skill', 'actual_initial_eval_relative_mse',
                           'actual_eval_relative_mse', 'actual_initial_max_gamma', 'actual_max_gamma',
                           'actual_endpoint_remainder_to_effective_norm_ratio',
                           'actual_integrated_signed_remainder_to_effective_vector_norm_ratio',
                           'new_ever_fraction_1', 'new_ever_fraction_3.2', 'new_ever_fraction_16'):
                values = np.array([float(r[metric]) for r in subset])
                for name, operation in [('mean', np.mean), ('median', np.median), ('min', np.min), ('max', np.max)]:
                    record[f'{metric}_{name}'] = float(operation(values))
            record['positive_skill_cases'] = sum(float(r['slope_motion_skill']) > 0 for r in subset)
            record['cases'] = len(subset)
            target_summary.append(record)
with (ROOT/'target_summary.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(target_summary[0]))
    writer.writeheader()
    writer.writerows(target_summary)
fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True, layout='constrained')
for ax, model, title in zip(axes, ['affine', 'effective_pure', 'effective_with_remainder'],
                           ['Full applied-field affine', 'Fixed effective map', 'Fixed map + fixed remainder']):
    values = np.array([[np.median([float(r['slope_motion_relative_error']) for r in rows
        if r['target'] == t and r['arm'] == 'joint' and r['model'] == model
        and int(r['offset']) == h])*100 for h in horizons] for t in TARGETS])
    im = ax.imshow(values, norm=LogNorm(.001, 100), cmap='viridis')
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, f'{value:.3g}', ha='center', va='center',
                color='black' if value > 3 else 'white', fontsize=9)
    ax.set_xticks(range(4), ['1k', '10k', '50k', '200k'])
    ax.set_yticks(range(10), labels)
    ax.set_xlabel('Additional GD updates after fork')
    ax.set_title(title)
fig.colorbar(im, ax=axes, label='Median relative slope-motion error (%)', shrink=.75)
fig.suptitle('Held-out targets: ordinary-GD forecast error\nEach cell: six forks (two seeds × 100k, 400k, 600k starts); no-motion error = 100%')
fig.savefig(ROOT/'forecast_error.png', dpi=170)
plt.close(fig)

fig, axes = plt.subplots(2, 5, figsize=(16, 7), layout='constrained')
colors = {100000: '#277da8', 400000: '#e09f3e', 600000: '#8b5a9b'}
for ax, target in zip(axes.flat, TARGETS):
    for start, color in colors.items():
        selected = [r for r in rows if r['target'] == target and r['arm'] == 'joint'
                    and r['model'] == 'affine' and int(r['start']) == start]
        for seed in sorted(set(r['seed'] for r in selected)):
            part = sorted([r for r in selected if r['seed'] == seed], key=lambda r: int(r['offset']))
            xx = [start/1000]+[(start+int(r['offset']))/1000 for r in part]
            yy = [float(part[0]['actual_initial_eval_relative_mse'])]+[float(r['actual_eval_relative_mse']) for r in part]
            ax.plot(xx, yy, '-o', color=color, markersize=3, alpha=.8,
                    label=f'{start//1000}k fork' if seed == sorted(set(r['seed'] for r in selected))[0] else None)
    ax.set_title(target)
    ax.set_yscale('log')
    ax.set_xlabel('Total GD updates (thousands)')
    ax.set_ylabel('Evaluation relative MSE')
    ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=8)
fig.suptitle('Held-out ordinary-GD fits over the practical forecast window\nIndependent 8192-point evaluation; curves retain both seeds and all three fork stages')
fig.savefig(ROOT/'evaluation_total_updates.png', dpi=170)
plt.close(fig)
