"""Select complete frozen-Adam recipes for the two geometry interventions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


def analyze(base, run, baseline, output):
    manifest = json.loads((base/'manifest.json').read_text())
    metadata = json.loads((run/'metadata.json').read_text())
    executed = json.loads((run/'summary.json').read_text())
    digest = hashlib.sha256((base/'input.npz').read_bytes()).hexdigest()
    assert digest == manifest['input_sha256'] == metadata['input_sha256']
    baseline_data = json.loads((baseline/'summary.json').read_text())
    assert baseline_data['input_sha256'] == manifest['intervention_source_sha256']
    assert baseline_data['assay_steps'] == 2_000_000
    baseline_rows = [r for r in baseline_data['geometries']
                     if r['snapshot_step'] == 2_000_000 and r['slope_multiplier'] == 1]
    assert len(baseline_rows) == 10
    data = np.load(base/'input.npz')
    state = np.load(run/'state.npz')
    assert int(state['count']) == executed['completed_updates'] == 2_000_000
    assert metadata['config']['horizon'] == 2_000_000
    recipes = metadata['recipes']
    rates = [r['learning_rate'] for r in recipes]
    rows = []
    for gi, geometry in enumerate(manifest['geometries']):
        weights = state['w'][gi]
        targets = [data[k] for k in ('target', 'validation_target', 'eval_target')]
        matrices = [data['features'][gi]]
        for key in ('validation_x', 'eval_x'):
            matrices.append(np.column_stack((np.tanh(
                data[key][:, None]*data['a'][gi]+data['b'][gi]), np.ones(len(data[key])))))
        errors = [np.linalg.norm(matrix@weights-y[:, None], axis=0)/np.linalg.norm(y)
                  for matrix, y in zip(matrices, targets)]
        np.testing.assert_allclose(errors[0], executed['final_relative_error'][gi],
                                   rtol=1e-8, atol=1e-11)
        finite = np.isfinite(errors[1])
        if not np.any(finite):
            raise ValueError(f'No finite validation endpoint for {geometry["name"]}')
        ri = int(np.argmin(np.where(finite, errors[1], np.inf)))
        rows.append(dict(geometry=geometry['name'], optimizer=geometry['optimizer'],
            family=geometry['family'], seed=geometry['seed'], recipe_index=ri,
            recipe=recipes[ri], train_error=float(errors[0][ri]),
            validation_error=float(errors[1][ri]), eval_error=float(errors[2][ri]),
            boundary_winner=recipes[ri]['learning_rate'] in (min(rates), max(rates)),
            all_recipe_errors={name:[float(v) if np.isfinite(v) else None for v in values]
                               for name, values in zip(['train','validation','eval'], errors)},
            final_readout_l2=float(np.linalg.norm(weights[:,ri]))))
    groups = []
    families = ['uniform_centers_learned_slopes', 'learned_centers_common_slope']
    for optimizer in ['adam', 'gd']:
        for family in families:
            subset = sorted([r for r in rows if r['optimizer'] == optimizer and r['family'] == family], key=lambda r:r['seed'])
            assert [r['seed'] for r in subset] == list(range(5))
            groups.append(dict(source_optimizer=optimizer, family=family,
                per_seed_eval_errors=[r['eval_error'] for r in subset],
                median_eval_error=float(np.median([r['eval_error'] for r in subset])),
                boundary_winner_seeds=[r['seed'] for r in subset if r['boundary_winner']]))
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(dict(input_sha256=digest,
        state_sha256=hashlib.sha256((run/'state.npz').read_bytes()).hexdigest(),
        completed_updates=int(state['count']), cases=rows, groups=groups,
        original_geometry_baselines=[dict(optimizer=r['optimizer'],seed=r['seed'],
            eval_error=r['eval_error'],selected=r['selected']) for r in baseline_rows],
        baseline_summary_sha256=hashlib.sha256((baseline/'summary.json').read_bytes()).hexdigest(),
        selection='One complete recipe per dictionary selected by final validation error; no checkpoint selection or seed pooling.',
        evaluation='8192-point midpoint resolution check; not an untouched test set.',
        boundary_policy='Report boundary winners; no further expansion beyond the previously expanded shared rate grid.'),indent=2,allow_nan=False)+'\n')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.1), layout='constrained', sharey=True)
    fig.suptitle('Frozen Adam readout: 2M updates', fontsize=10)
    for ax, optimizer in zip(axes, ['adam','gd']):
        for seed in range(5):
            original = next(r['eval_error'] for r in baseline_rows
                            if r['optimizer']==optimizer and r['seed']==seed)
            values = [original]+[next(r['eval_error'] for r in rows if r['optimizer']==optimizer
                         and r['seed']==seed and r['family']==family) for family in families]
            ax.plot([0,1,2], values, 'o-', lw=1, ms=4, alpha=.8, label=f'Seed {seed}')
        ax.set_xticks([0,1,2], ['Learned\ngeometry', 'Uniform\ncenters', 'Common\nslope'])
        ax.set(yscale='log', title=f'{"Adam" if optimizer=="adam" else "GD"}-acquired features', xlim=(-.15,2.15))
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Relative output error on dense grid')
    axes[1].legend(frameon=False, fontsize=7, loc='best')
    fig.savefig(output/'paired_intervention_errors.pdf')
    fig.savefig(output/'paired_intervention_errors.png', dpi=300)
    plt.close(fig)
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        analyze(args.base, args.run, args.baseline, args.output)
