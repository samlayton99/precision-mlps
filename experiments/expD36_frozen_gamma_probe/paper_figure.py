"""Plot archived scale mismatch, certified acquisition times, and width recovery.

Run from the repository root with ``python -m
experiments.expD36_frozen_gamma_probe.paper_figure``. No training or prediction
is rerun. The JSON companion records plotted values and source checksums.
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

TARGET = 'sine_mix_2_6_10'
BLUE, ORANGE, GREEN = '#245c9f', '#c65525', '#278064'


def collect(root):
    """Check archived protocols and certificates before extracting plot values."""
    sources = {}

    def read(relative):
        raw = (root/relative).read_bytes()
        sources[str(relative)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    cfg = read(Path('manifest.json'))['config']
    assert cfg['widths'] == [128, 256, 512, 1024]
    assert cfg['joint_steps'] == 20000 and cfg['training_steps'] == 200000
    joint, recovery = [], []
    for n in cfg['widths']:
        last = read(Path(f'joint/N{n}/evaluations.json'))[-1]
        assert last['step'] == cfg['joint_steps'] and not any(last['failed'])
        width = n+2*int(np.ceil(np.sqrt(n)))+1
        assert last['n'] == n and last['width'] == width
        quantiles = np.asarray(last['slope_quantiles'])
        assert quantiles.shape == (len(cfg['seeds']), 4)
        assert np.all(np.isfinite(quantiles)) and np.all(quantiles > 0)
        joint.append(dict(n=n, width=width, reference_gamma=n/8,
                          seed_medians=quantiles[:, 1].tolist(),
                          seed_maxima=quantiles[:, 3].tolist()))

        case = read(Path(f'training/N{n}_raw_gd/case.json'))
        rows = read(Path(f'training/N{n}_raw_gd/evaluations.json'))
        assert case['map'] == 'raw' and case['optimizer'] == 'gd'
        assert case['n'] == n and rows[0]['step'] == 0
        assert rows[-1]['step'] == cfg['training_steps']
        ti = next(i for i, c in enumerate(case['columns']) if c['target'] == TARGET)
        assert case['columns'][ti]['initialization'] == 'zero'
        for gamma in [4, n/8]:
            gi = case['gammas'].index(gamma)
            assert not rows[-1]['failed'][gi][ti]
            assert np.isclose(rows[0]['train'][gi][ti], 1, rtol=0, atol=1e-14)
            meta = read(Path(f'dictionaries/N{n}_raw_g{gamma:g}/meta.json'))
            assert meta['complete'] and meta['width'] == width
            assert meta['matrix_hash'] == case['matrix_hashes'][gi]
            eta = case['rates'][gi][ti]
            assert np.isclose(eta*meta['L'], .5, rtol=1e-10, atol=0)
            recovery.append(dict(n=n, width=width, gamma=gamma, eta=eta,
                residual=rows[-1]['train'][gi][ti],
                strategy='fixed' if gamma == 4 else 'matched'))

    folder = Path('refinements/gamma_factorized_kernel')
    summary, audit = read(folder/'summary.json'), read(folder/'interval_audit.json')
    assert summary['n'] == 512 and summary['width'] == 559
    assert summary['epsilon'] == .01
    ti = summary['targets'].index(TARGET)
    timing = []
    for d in summary['dictionaries']:
        gamma = d['gamma']
        comparison = next(c for c in summary['comparisons'] if c['gamma'] == gamma)
        bounds = comparison['selected']['combined'][ti]
        check = next(a for a in audit['results'] if a['gamma'] == gamma)
        assert check['target'] == TARGET and check['eta'] == d['eta']
        assert check['eta_L_gershgorin_upper'] < 1
        certificate = check['methods']['combined']
        assert certificate['status'] == 'interval_certified_endpoints'
        for key in ['necessary', 'sufficient', 'lower_harmonics', 'upper_harmonics']:
            assert bounds[key] == certificate[key]
        assert certificate['excluded_step'] == bounds['necessary']-1
        assert certificate['excluded_error_squared'][0] > summary['epsilon']**2
        assert certificate['sufficient_error_squared'][1] <= summary['epsilon']**2
        hit = d['executed_hits'][ti]
        assert bounds['necessary'] <= hit <= bounds['sufficient']
        timing.append(dict(gamma=gamma, eta=d['eta'], executed=hit, **bounds))
    assert [r['gamma'] for r in timing] == [8, 12, 16, 64]
    slow, fast = timing[0], timing[-1]
    ratio = dict(necessary=slow['necessary']/fast['sufficient'],
                 sufficient=slow['sufficient']/fast['necessary'],
                 executed=slow['executed']/fast['executed'])
    return dict(target=TARGET, joint_steps=cfg['joint_steps'],
        joint_rate=cfg['joint_rate'], joint_epsilon=cfg['joint_epsilon'],
        seeds=cfg['seeds'], readout_steps=cfg['training_steps'], epsilon=.01,
        samples_per_cell=cfg['samples_per_cell'], joint=joint, timing=timing,
        recovery=recovery, delay_ratio=ratio, source_sha256=sources,
        plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence_role='Retrospective training evidence; B endpoints independently interval-certified',
        gpu_hours=0)


def plot(data, output):
    plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), layout='constrained')
    widths = np.array([row['width'] for row in data['joint']])
    ax = axes[0]
    for key, color, marker, label in [
            ('seed_medians', BLUE, 'o', r'Median $|a_j|$'),
            ('seed_maxima', ORANGE, 's', r'Maximum $|a_j|$')]:
        values = np.array([row[key] for row in data['joint']])
        ax.scatter(np.repeat(widths, values.shape[1]), values.ravel(),
                   color=color, marker=marker, alpha=.4, s=12, linewidths=0, zorder=3)
        ax.plot(widths, np.median(values, axis=1), color=color, marker=marker,
                ms=4, lw=1.5, label=label, zorder=4)
    ax.plot(widths, [row['reference_gamma'] for row in data['joint']],
            '--', color=GREEN, lw=1.7, label=r'Reference $\gamma=N/8$')
    ax.set(title='A  Learned slope gap',
           xlabel=r'Hidden width $W$', ylabel='Physical slope magnitude',
           xscale='log', yscale='log', ylim=(.07, 250))
    ax.text(.03, .97, 'Joint Adam, 20k; 5 seeds', transform=ax.transAxes,
            va='top', fontsize=6.5, color='.3')
    ax.legend(loc='center left', bbox_to_anchor=(.015, .33), fontsize=6.5, frameon=False)

    ax = axes[1]
    gammas = np.array([row['gamma'] for row in data['timing']])
    hits = np.array([row['executed'] for row in data['timing']])
    low = np.array([row['necessary'] for row in data['timing']])
    high = np.array([row['sufficient'] for row in data['timing']])
    middle = (low+high)/2
    ax.plot(gammas, middle, color=ORANGE, lw=1.5)
    ax.errorbar(gammas, middle, yerr=[middle-low, high-middle], fmt='none',
                ecolor=ORANGE, capsize=4, elinewidth=1.5, label='Certified interval')
    ax.scatter(gammas, hits, color='.1', s=24, zorder=4, label='Executed GD hit')
    ax.set(title='B  Predicted learning delay', xlabel=r'Common slope $\gamma$',
           ylabel='Updates to 1% residual', xscale='log', yscale='log',
           xlim=(7, 75), ylim=(1e4, 3e7))
    ax.text(.03, .97, r'Raw GD, $W=559$', transform=ax.transAxes,
            va='top', fontsize=6.5, color='.3')
    ax.legend(loc='upper right', bbox_to_anchor=(1.04, .86), fontsize=6, frameon=False)
    ax.set_xticks(gammas, [str(g) for g in gammas])
    # Show interval tightness separately; the main log axis hides the bracket widths.
    inset = ax.inset_axes([.49, .34, .48, .30])
    inset.axhline(1, color='.45', lw=.7, ls=':')
    inset.errorbar(np.arange(4), middle/hits,
                   yerr=[(middle-low)/hits, (high-middle)/hits],
                   fmt='none', ecolor=ORANGE, capsize=4, elinewidth=1.5)
    inset.scatter(np.arange(4), middle/hits, s=8, color=ORANGE, zorder=3)
    inset.set(xlim=(-.5, 3.5), ylim=(.9988, 1.0012))
    inset.set_xticks(np.arange(4), [str(g) for g in gammas])
    inset.set_yticks([.999, 1, 1.001], ['0.999', '1.000', '1.001'])
    inset.set_title('Bound / hit', fontsize=6.5, pad=4)
    inset.tick_params(labelsize=5.5, length=2)
    inset.set_xlabel(r'$\gamma$', fontsize=6, labelpad=0)

    ax = axes[2]
    for strategy, color, label in [('fixed', BLUE, r'Fixed $\gamma=4$'),
                                    ('matched', GREEN, r'Matched $\gamma=N/8$')]:
        rows = [r for r in data['recovery'] if r['strategy'] == strategy]
        ax.plot([r['width'] for r in rows], [r['residual'] for r in rows],
                'o-', color=color, ms=4, lw=1.7, label=label)
    ax.axhline(data['epsilon'], color='.5', ls=':', lw=1, label='1% residual')
    ax.set(title='C  Recovery across widths', xlabel=r'Hidden width $W$',
           ylabel='Residual after 200k updates', xscale='log', yscale='log',
           ylim=(8e-4, 1.1))
    ax.text(.03, .97, 'Raw GD, fixed target', transform=ax.transAxes,
            va='top', fontsize=6.5, color='.3')
    ax.legend(loc='center left', bbox_to_anchor=(.03, .56), fontsize=6.5, frameon=False)
    for ax in axes:
        ax.grid(True, which='major', alpha=.16, linewidth=.6)
        ax.xaxis.set_minor_locator(NullLocator())
    for ax in axes[[0, 2]]:
        ax.set_xticks(widths, [str(w) for w in widths])
    fig.savefig(output/'paper_three_panel.png', dpi=240)
    fig.savefig(output/'paper_three_panel.pdf', metadata={'CreationDate': None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(
        'results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = collect(args.root)
    output = args.output or args.root/'refinements/gamma_factorized_kernel'
    output.mkdir(parents=True, exist_ok=True)
    plot(data, output)
    (output/'paper_figure_data.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), delay_ratio=data['delay_ratio'],
                         checked_sources=len(data['source_sha256'])), indent=2))


if __name__ == '__main__':
    main()
