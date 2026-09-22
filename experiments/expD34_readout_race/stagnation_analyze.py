"""Compare autonomous forecasts and numerical controls; write evidence, not prose."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import stagnation as st, transport as tr
from .stagnation_run import extract, write_json, write_table


def read_table(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return list(csv.DictReader(stream))


def curve(rows, seed, start, key):
    return np.array([float(r[key]) for r in rows if int(r['seed']) == seed and int(r['fork_step']) == start])


def historical_attribution(root):
    """Reproduce the motivating signed-mode observation without re-running GD."""
    curated = root.parent/'useful_slopes'/'curated'
    result = []
    for seed in range(5):
        folder = curated/(f'seed{seed}_fork20000_metrics' if seed < 3 else f'dense_seed{seed}_metrics')
        rows = read_table(folder/'metrics.csv.gz')
        with np.load(folder/'modal_forces.npz') as f:
            np.testing.assert_array_equal(f['mode_labels'], np.arange(2, 10))
            for i, index in enumerate(f['row']):
                row = rows[index]
                if row['target'] != 'moment9' or row['arm'] != 'joint' or int(row['step']) < 20000:
                    continue
                v = f['outward'][i]
                result.append(dict(seed=seed, step=int(row['step']), generated_outward=v[:7].sum(),
                    hard_outward=v[7], quadratic_outward=v[0], cubic_outward=v[1],
                    even_outward=v[[0, 2, 4, 6]].sum(), effective_outward=float(row['effective_outward']),
                    tracking_outward=float(row['tracking_outward']), mean_gamma=float(row['mean_gamma'])))
    write_table(root/'historical_attribution.csv.gz', result)


def analyze(root):
    historical_attribution(root)
    bundles = {}; statuses = {}; tables = {}; hashes = {}
    for arm in st.ARMS:
        for suffix in ('', '_half'):
            name = arm+suffix; folder = root/'runs'/name
            statuses[name] = json.loads((folder/'status.json').read_text())
            if not statuses[name]['complete'] or statuses[name]['reference_updates'] != 20000:
                raise ValueError(f'Incomplete protocol at {folder}')
            bundles[name] = dict(np.load(folder/'compact_states.npz'))
            tables[name] = read_table(folder/'metrics.csv.gz')
            hashes[name] = hashlib.sha256((folder/'compact_states.npz').read_bytes()).hexdigest()
    assert sum(s['cases'] for n, s in statuses.items() if not n.endswith('_half')) == 40
    assert sum(s['cases'] for n, s in statuses.items() if n.endswith('_half')) == 8
    x = bundles['full']['x']; q = tr.basis(x, 65); qfine = tr.basis(x, 129)
    replay = []
    curated = root.parent/'useful_slopes'/'curated'/'states'
    for i, case in enumerate(json.loads(str(bundles['full']['cases']))):
        seed, start = case['seed'], case['fork_step']
        if start == 100000 and seed >= 3:
            continue
        folder = f'seed{seed}_fork{start}' if seed < 3 else f'dense_seed{seed}'
        z, d, _, _ = extract(curated/folder/'compact_states.npz', seed, (start+1,))
        index = int(np.flatnonzero(bundles['full']['offsets'] == 1)[0])
        error = max(np.max(abs(z[0]-bundles['full']['z'][i, index])), abs(d[0]-bundles['full']['d'][i, index]))
        np.testing.assert_allclose(z[0], bundles['full']['z'][i, index], atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(d[0], bundles['full']['d'][i, index], atol=2e-13, rtol=2e-13)
        replay.append(dict(seed=seed, fork_step=start, first_update_max_error=error))
    write_table(root/'baseline_replay.csv', replay)
    endpoints = []; packed = []; comparisons = []; verification = []
    # Cache actual-loss force vectors at each trajectory's own parameter state.
    gradients = {}; bias_gradients = {}
    for name, f in bundles.items():
        cases = json.loads(str(f['cases']))
        arm = name.removesuffix('_half')
        gg = np.empty_like(f['z'])
        gd = np.empty_like(f['d'])
        for i, case in enumerate(cases):
            z0 = f['z'][i, 0]; d0 = f['d'][i, 0]
            initial, _ = st.diagnostics(z0, d0, x, f['y'][i], arm, basis=q)
            final, arr = st.diagnostics(f['z'][i, -1], f['d'][i, -1], x, f['y'][i], arm, basis=q)
            refined, refined_arr = st.diagnostics(f['z'][i, -1], f['d'][i, -1], x, f['y'][i], arm, degree=129, basis=qfine)
            for j in range(len(f['offsets'])):
                _, vectors = st.diagnostics(f['z'][i, j], f['d'][i, j], x, f['y'][i], arm, basis=q)
                gg[i, j] = vectors['gradient']
                gd[i, j] = vectors['gradient_d'][0]
            exact_delta = np.mean(abs(f['z'][i, -1, 0])-abs(z0[0]))
            row = dict(**case, mean_gamma_delta=exact_delta,
                slope_energy_delta=final['slope_energy']-initial['slope_energy'],
                readout_l2_delta=final['readout_l2']-initial['readout_l2'],
                hard_residual_delta=final['hard_residual']-initial['hard_residual'],
                mean_gamma=final['mean_gamma'], max_gamma=final['max_gamma'],
                hard_residual=final['hard_residual'], positive_travel=f['positive'][i, -1].mean(),
                negative_travel=f['negative'][i, -1].mean(), path=f['path'][i, -1],
                max_positive_travel=f['positive'][i, -1].max(),
                net_outward_fraction=np.mean(abs(f['z'][i, -1, 0]) > abs(z0[0])),
                tracking_travel=f['tracking_travel'][i, -1].mean(),
                effective_travel=exact_delta-f['crossing'][i, -1]-f['tracking_travel'][i, -1].mean(),
                terminal_actual_outward=final['actual_outward'], terminal_effective_outward=final['effective_outward'],
                terminal_tracking_outward=final['tracking_outward'],
                terminal_modal_refinement_outward_error=abs(refined['effective_outward']-final['effective_outward']),
                terminal_modal_refinement_force_error=np.linalg.norm(refined_arr['effective_a']-arr['effective_a']),
                full_loss_delta=final['half_mse']-initial['half_mse'], objective_delta=final['objective']-initial['objective'])
            rr = [r for r in tables[name] if int(r['seed']) == case['seed'] and int(r['fork_step']) == case['fork_step']]
            tau = np.array([float(r['offset'])*.002 for r in rr])
            for component in ('generated', 'hard', 'higher', 'tracking', 'omitted', 'actual'):
                row['sampled_integral_'+component] = np.trapezoid([float(r[component+'_outward']) for r in rr], tau)
            for block in 'abcd':
                ratios = [float(r['tracking_'+block+'_norm'])/max(float(r['effective_'+block+'_norm']), 1e-300) for r in rr]
                row['max_tracking_relative_'+block] = max(ratios)
                row['terminal_tracking_relative_'+block] = ratios[-1]
            endpoints.append(row)
            packed.append(dict(initial_z=z0, final_z=f['z'][i, -1], initial_d=d0, final_d=f['d'][i, -1],
                positive=f['positive'][i, -1], negative=f['negative'][i, -1], tracking_travel=f['tracking_travel'][i, -1],
                terminal_effective_a=arr['effective_a'], terminal_gradient=arr['gradient'],
                terminal_gradient_d=arr['gradient_d']))
        gradients[name] = gg
        bias_gradients[name] = gd
    for name, f in bundles.items():
        arm = name.removesuffix('_half')
        reference = arm if name.endswith('_half') else 'full'
        if name == reference:
            continue
        ref = bundles[reference]
        refcases = json.loads(str(ref['cases']))
        for i, case in enumerate(json.loads(str(f['cases']))):
            k = next(k for k, c in enumerate(refcases) if c['seed'] == case['seed'] and c['fork_step'] == case['fork_step'])
            np.testing.assert_array_equal(f['offsets'], ref['offsets'])
            np.testing.assert_array_equal(f['z'][i, 0], ref['z'][k, 0])
            np.testing.assert_array_equal(f['d'][i, 0], ref['d'][k, 0])
            np.testing.assert_array_equal(f['y'][i], ref['y'][k])
            error = f['z'][i]-ref['z'][k]
            r = dict(**case, comparison='half_step' if name.endswith('_half') else 'surrogate',
                reference=reference, max_mean_gamma_error=np.max(abs(np.mean(abs(f['z'][i, :, 0])-abs(ref['z'][k, :, 0]), axis=1))),
                final_mean_gamma_error=abs(np.mean(abs(f['z'][i, -1, 0])-abs(ref['z'][k, -1, 0]))),
                max_velocity_error=np.max(abs(curve(tables[name], case['seed'], case['fork_step'], 'actual_outward')
                    -curve(tables[reference], case['seed'], case['fork_step'], 'actual_outward'))),
                max_hard_residual_error=np.max(abs(curve(tables[name], case['seed'], case['fork_step'], 'hard_residual')
                    -curve(tables[reference], case['seed'], case['fork_step'], 'hard_residual'))),
                max_coarse_output_error=max(np.max(abs(curve(tables[name], case['seed'], case['fork_step'], 'coarse_output_'+str(mode))
                    -curve(tables[reference], case['seed'], case['fork_step'], 'coarse_output_'+str(mode)))) for mode in (0, 1)))
            for block, label in enumerate('abc'):
                r['max_parameter_error_'+label] = np.linalg.norm(error[:, block], axis=1).max()
                r['final_parameter_error_'+label] = np.linalg.norm(error[-1, block])
                displacement = np.linalg.norm(ref['z'][k, -1, block]-ref['z'][k, 0, block])
                r['reference_displacement_'+label] = displacement
                r['max_force_error_'+label] = np.linalg.norm(gradients[name][i, :, block]-gradients[reference][k, :, block], axis=1).max()
            r['max_parameter_error_d'] = abs(f['d'][i]-ref['d'][k]).max()
            r['max_force_error_d'] = abs(bias_gradients[name][i]-bias_gradients[reference][k]).max()
            comparisons.append(r)
    for name, status in statuses.items():
        rr = tables[name]
        verification.append(dict(run=name, **status,
            max_force_decomposition_error=max(float(r['decomposition_error']) for r in rr),
            max_tracking_modal_exact_error=max(abs(float(r['tracking_a_norm'])-float(r['exact_tracking_a_norm'])) for r in rr)))
    write_table(root/'endpoints.csv', endpoints)
    write_table(root/'forecast_comparisons.csv', comparisons)
    write_table(root/'verification.csv', verification)
    np.savez_compressed(root/'endpoints.npz', cases=np.array(json.dumps([{k: r[k] for k in ('seed', 'fork_step', 'arm', 'training_eta')} for r in endpoints])),
                        **{k: np.stack([p[k] for p in packed]) for k in packed[0]})
    write_json(root/'artifact_hashes.json', hashes)
    plot(root, tables, comparisons)
    print(json.dumps(dict(primary_continuations=40, half_step_controls=8,
        max_motion_error=max(s['motion_identity_error'] for s in statuses.values()),
        max_balance_error=max(s['balance_identity_error'] for s in statuses.values()))), flush=True)


def plot(root, tables, comparisons):
    colors = dict(full='#163c66', five_mode='#d27e19', ten_mode='#319270', remove_lower='#ad4665')
    labels = dict(full='Full loss', five_mode='Modes 0,1,2,3,9', ten_mode='Modes 0–9', remove_lower='Remove modes 2–8')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    rr = [r for r in tables['full'] if int(r['seed']) == 0 and int(r['fork_step']) == 20000]
    tau = np.array([float(r['offset'])*.002 for r in rr])
    for seed in range(5):
        for arm in st.ARMS:
            gamma = curve(tables[arm], seed, 20000, 'mean_gamma')
            axes[0].plot(tau, (gamma-gamma[0])*1e6, color=colors[arm], alpha=.55,
                         label=labels[arm] if seed == 0 else None, ls='--' if arm == 'five_mode' else '-')
    axes[0].set(xlabel='Time since 20k checkpoint', ylabel='Change in mean |a| × 10⁶')
    axes[0].legend(fontsize=8)
    for component, label in [('generated', 'Generated modes 2–8'), ('hard', 'Hard mode 9'), ('tracking', 'Coarse tracking')]:
        v = curve(tables['full'], 0, 20000, component+'_outward')
        axes[1].plot(tau, abs(v), label=label)
    axes[1].set(yscale='log', xlabel='Time since 20k checkpoint', ylabel='|Mean-scale velocity contribution|')
    axes[1].legend(fontsize=8)
    for component, label in [('actual', 'Actual GD'), ('effective', 'Effective fine'), ('tracking', 'Coarse tracking')]:
        axes[2].plot(tau, curve(tables['remove_lower'], 0, 20000, component+'_outward'), label=label)
    axes[2].set(yscale='symlog', xlabel='Time since 20k checkpoint', ylabel='Mean-scale velocity after ablation')
    axes[2].set_yscale('symlog', linthresh=1e-12)
    axes[2].legend(fontsize=8)
    fig.savefig(root/'coupled_forecasts.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
    for seed in range(5):
        for key, label, color in [('positive_travel', 'Outward travel', '#319270'), ('negative_travel', 'Inward travel', '#ad4665')]:
            axes[0].plot(tau, curve(tables['full'], seed, 20000, key)*1e6, color=color, alpha=.6,
                         label=label if seed == 0 else None)
    axes[0].set(xlabel='Time since 20k checkpoint', ylabel='Mean cumulative travel × 10⁶')
    axes[0].legend(fontsize=8)
    for i, arm in enumerate(('five_mode', 'ten_mode')):
        values = [r['final_parameter_error_a']/r['reference_displacement_a'] for r in comparisons
                  if r['comparison'] == 'surrogate' and r['arm'] == arm]
        axes[1].scatter(np.full(len(values), i), values, color=colors[arm], alpha=.65)
    values = [r['final_parameter_error_a']/r['reference_displacement_a'] for r in comparisons
              if r['comparison'] == 'half_step' and r['arm'] == 'full']
    axes[1].scatter(np.full(len(values), 2), values, color=colors['full'])
    axes[1].set(yscale='log', ylabel='Slope-vector error / full-GD displacement',
                xticks=[0, 1, 2], xticklabels=['Five modes', 'Ten modes', 'Full half step'])
    fig.savefig(root/'transport_and_accuracy.png', dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    analyze(parser.parse_args().root)


if __name__ == '__main__':
    main()
