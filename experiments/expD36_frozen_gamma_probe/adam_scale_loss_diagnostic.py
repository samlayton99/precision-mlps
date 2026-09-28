"""Personal diagnostic: observed Adam loss versus evolving RMS slopes.

Uses archived joint-run loss checkpoints and the already verified slope export.
Frozen comparisons use their 12 available scalar checkpoints, without smoothing.
The paper figure is not changed and no optimizer is run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT/'results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep'
COLORS = ['#346AA1', '#CB7735', '#32927F', '#9065A9', '#BD5360']


def collect(joint_root, frozen):
    hashes = {}
    def source(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def read(path):
        return json.loads(source(path).read_text())
    paper = read(frozen/'refinements/paper_training_failure/figure_data.json')
    joint = paper['joint']
    steps = np.asarray(joint['steps'])
    losses = []
    checks = []
    for seed in joint['seeds']:
        folder = joint_root/'raw'/f'primary_{seed}'
        manifest = read(folder/'manifest.json')
        ci = next(i for i, c in enumerate(manifest['cases'])
                  if c['target'] == 'mixed_sine' and c['optimizer'] == 'adam' and c['seed'] == seed)
        mi = manifest['metrics'].index('relative_mse')
        with np.load(source(folder/'trace.npz')) as f:
            updates = f['ends']-1
            values = f['values'][ci, :, mi]
        assert np.all(np.isfinite(values)) and np.all(values > 0)
        assert updates[0] == 0 and updates[-1] == 599999
        if seed:
            np.testing.assert_array_equal(updates, common_steps)
        common_steps = updates
        # The trace measures pre-update states, so compare only exactly shared times.
        errors = np.asarray(joint['adam']['train_error'][seed])
        overlap = 0
        for k, n in enumerate(steps):
            indices = np.flatnonzero(updates == n)
            if len(indices):
                np.testing.assert_allclose(values[indices[0]], errors[k]**2, rtol=0, atol=1e-12)
                overlap += 1
        checks.append(dict(seed=seed, shared_snapshot_loss_checks=overlap))
        losses.append(np.r_[values, errors[-1]**2].tolist())
    joint_data = dict(steps=np.r_[common_steps, 600000].tolist(), loss=losses,
                      slope_steps=steps.tolist(), lambda_rms=joint['adam']['lambda_rms'],
                      snapshot_error=joint['adam']['train_error'], seeds=joint['seeds'],
                      protocol='All parameters trained; W=177; h=1/64; target frequencies (2,6,14)*pi; Adam lr=.002, epsilon=1e-8.')
    training = frozen/'training'
    pilot_case = read(training/'N512_raw_adam_pilot/case.json')
    cont_case = read(training/'N512_raw_adam_continue/case.json')
    selection = read(training/'N512_raw_selection.json')
    pilot = read(training/'N512_raw_adam_pilot/evaluations.json')
    continuation = read(training/'N512_raw_adam_continue/evaluations.json')
    assert pilot_case['gammas'] == cont_case['gammas']
    assert pilot_case['matrix_hashes'] == cont_case['matrix_hashes']
    ci = next(i for i, c in enumerate(cont_case['columns'])
              if c['target'] == 'sine_mix_2_6_10' and c['view'] == 'common')
    fixed = []
    for gamma in (8, 16, 24, 32, 64):
        gi = cont_case['gammas'].index(gamma)
        pi = selection['indices'][gi][ci]
        cfg = pilot_case['columns'][pi]
        assert cfg['initialization'] == 'zero' and cfg['target'] == 'sine_mix_2_6_10'
        assert cfg['initial_rate'] == cont_case['rates'][gi][ci] == .001
        assert cfg['epsilon'] == cont_case['epsilons'][gi][ci] == 1e-12
        assert pilot[-1]['step'] == continuation[0]['step'] == 50000
        assert pilot[-1]['train'][gi][pi] == continuation[0]['train'][gi][ci]
        points = []
        for rows, column in ((pilot, pi), (continuation[1:], ci)):
            for row in rows:
                assert not row['failed'][gi][column]
                points.append([row['step'], row['train'][gi][column]])
        times, error = np.asarray(points).T
        assert len(times) == 12 and times[0] == 0 and times[-1] == 200000
        fixed.append(dict(gamma=gamma, slope_scale=gamma/256, steps=times.astype(int).tolist(),
                          error=error.tolist(), loss=(error**2).tolist()))
    summary = []
    for n in (0, 1000, 2000, 20000, 100000, 200000, 400000, 600000):
        k = int(np.flatnonzero(steps == n)[0])
        summary.append(dict(step=n, median_error=float(np.median(np.asarray(joint['adam']['train_error'])[:, k])),
                            median_lambda_rms=float(np.median(np.asarray(joint['adam']['lambda_rms'])[:, k]))))
    return dict(joint=joint_data, frozen=fixed, summary=summary, validation=checks, source_sha256=hashes,
                convention='Loss means relative MSE = ||prediction-target||^2 / ||target||^2; output-error fraction is sqrt(loss).',
                frozen_protocol='Readout only; W=559; h=1/256; target frequencies (2,6,10)*pi; lr=.001 through20k, cosine decay to1e-6 at50k, constant thereafter; epsilon=1e-12.',
                frozen_sampling='Exactly12 archived checkpoints per curve. Straight segments only connect saved states; they do not reveal intervening Adam oscillations.')


def joint_plot(data):
    d = data['joint']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex='col')
    fig.subplots_adjust(left=.095, right=.970, bottom=.18, top=.78, hspace=.24, wspace=.23)
    for col, limit in enumerate((600, 20)):
        for seed, color in zip(d['seeds'], COLORS):
            axes[0, col].plot(np.asarray(d['steps'])/1000, d['loss'][seed], color=color, lw=.85, alpha=.7)
            axes[1, col].plot(np.asarray(d['slope_steps'])/1000, d['lambda_rms'][seed],
                             color=color, lw=.85, alpha=.7, marker='o', ms=2.2)
        axes[0, col].plot(np.asarray(d['steps'])/1000, np.median(d['loss'], axis=0), color='#263442', lw=2)
        axes[1, col].plot(np.asarray(d['slope_steps'])/1000, np.median(d['lambda_rms'], axis=0),
                         color='#263442', lw=2, marker='o', ms=3)
        axes[0, col].set(yscale='log', xlim=(0, limit), ylim=(.0005, 1.6))
        axes[1, col].set(xlim=(0, limit), ylim=(0, .108), xlabel='Updates (thousands)')
        axes[1, col].axhline(.1, color='#9BA3AA', ls=':', lw=1)
        axes[1, col].set_yticks([0, .02, .04, .06, .08, .1])
        axes[0, col].set_title('Full run: 0-600k updates' if col == 0 else 'Early phase: 0-20k updates', loc='left')
        axes[0, col].yaxis.set_minor_locator(NullLocator())
    axes[0, 0].set_ylabel('Relative MSE (normalized loss)')
    axes[0, 1].set_ylim(.025, 1.6)
    axes[1, 0].set_ylabel(r'RMS slope scale $\lambda_{\rm RMS}$')
    axes[0, 1].axvspan(1, 2, color='#CCD3D8', alpha=.25, zorder=0)
    axes[1, 1].axvspan(1, 2, color='#CCD3D8', alpha=.25, zorder=0)
    axes[0, 1].text(.96, .75, '1k to 2k updates:\noutput error 91% to 46%', transform=axes[0, 1].transAxes,
                    ha='right', fontsize=10, color='#263442')
    axes[0, 0].text(.97, .92, 'End: 5.83% output error', transform=axes[0, 0].transAxes, ha='right', fontsize=10)
    fig.suptitle('Joint Adam: loss falls as the feature geometry develops', fontsize=15, y=.975)
    fig.text(.5, .922, 'All parameters trained. Loss = MSE / mean(target squared); output-error fraction = square root of loss.',
             ha='center', fontsize=10)
    handles = [plt.Line2D([], [], color=c, label=f'Seed {i}', lw=1.5) for i, c in enumerate(COLORS)]
    handles.append(plt.Line2D([], [], color='#263442', lw=2, label='Median'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .898), ncol=6, frameon=False, fontsize=10)
    fig.text(.5, .035, 'No smoothing. Loss uses 1,141 saved states per seed; slope markers use 39 saved parameter states.\n'
             'The dotted 0.1 line is a visual reference, not a proved acquisition threshold.', ha='center', fontsize=10)
    return fig


def frozen_plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    fig.subplots_adjust(left=.095, right=.970, bottom=.25, top=.73, wspace=.17)
    colors = ['#81919D', '#346AA1', '#CB7735', '#32927F', '#9065A9']
    for ax, limit in zip(axes, (20, 200)):
        if limit > 20:
            ax.axvspan(20, 50, color='#CCD3D8', alpha=.35, lw=0)
            ax.text(35, .4, 'LR decay', ha='center', fontsize=9, color='#68747D')
        for row, color in zip(data['frozen'], colors):
            ax.plot(np.asarray(row['steps'])/1000, row['loss'], color=color, lw=1,
                    marker='o', ms=4, mfc='white', mew=1, label=rf"$\lambda={row['slope_scale']:.5g}$")
        ax.set(xlim=(0, limit), yscale='log', ylim=(6e-9, 2), xlabel='Updates (thousands)')
        ax.set_title('First 20k updates' if limit == 20 else 'Full 200k updates', loc='left')
        ax.yaxis.set_minor_locator(NullLocator())
    axes[0].set_ylabel('Relative MSE (normalized loss)')
    fig.suptitle('Frozen uniform dictionaries: Adam near 0.1 already fits well', fontsize=15, y=.975)
    fig.text(.5, .89, r'Readout only; $W=559$, $h=1/256$. These are different features and a different target from the joint runs.',
             ha='center', fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .865), ncol=5, frameon=False, fontsize=10)
    fig.text(.5, .07, '12 saved checkpoints per curve; lines only connect those observations. Intermediate Adam oscillations are not shown.\n'
             r'Learning rate: $10^{-3}$ through 20k; cosine decay to $10^{-6}$ at 50k, then fixed.', ha='center', fontsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint-root', required=True, type=Path)
    parser.add_argument('--frozen-root', type=Path, default=FROZEN)
    parser.add_argument('--output', type=Path, default=ROOT/'output/diagnostics/adam_scale_loss')
    args = parser.parse_args()
    data = collect(args.joint_root, args.frozen_root)
    data['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'data.json').write_text(json.dumps(data, indent=2)+'\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#A3ABB1', 'axes.linewidth': .7, 'text.color': '#263442'})
    for name, plot in [('joint_adam_loss_and_scale', joint_plot), ('frozen_adam_checkpoint_losses', frozen_plot)]:
        fig = plot(data)
        fig.savefig(args.output/f'{name}.png', dpi=180)
        plt.close(fig)
    print(json.dumps(dict(summary=data['summary'], validation=data['validation']), indent=2))


if __name__ == '__main__':
    main()
