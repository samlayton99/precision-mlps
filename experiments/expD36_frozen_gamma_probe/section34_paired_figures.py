"""Compact paper pairs: frozen-feature mechanism and joint-feature acquisition."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from threadpoolctl import threadpool_limits

from .section34_figure import reference_display
from .section34_analyze import relative_error, style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['analysis', 'bounds', 'gd', 'base', 'frozen-analysis', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--spectrum-only', action='store_true',
                        help='Update Figure 3 without replacing the pending Figure 4 artwork.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.analysis/'summary.json').read_text())
    manifest = json.loads((args.base/'manifest.json').read_text())
    access = json.loads((args.frozen_analysis/'summary.json').read_text())
    trace = np.load(args.analysis/'selected_traces.npz')
    initial = np.load(args.base/'joint_input.npz')
    actual = np.load(args.gd/'raw_error.npy', mmap_mode='r')
    horizon = summary['horizon']
    assert horizon == access['assay_steps'] == len(actual)-1
    assert manifest['input_sha256'] == access['input_sha256']
    initial_error = np.array([relative_error(p, initial['x'], initial['target'], summary['width'])
                              for p in initial['initial_parameters']])
    indices = np.unique(np.r_[0, np.geomspace(1, horizon, 2300).astype(int),
                              np.linspace(0, horizon, 1800).astype(int)])
    bandwidths = [1/32, 1/16, 1/8, 1/4]
    palette = ['#332288', '#0072B2', '#009E73', '#CC6677']
    sources = [args.analysis/'summary.json', args.analysis/'selected_traces.npz',
               args.base/'joint_input.npz', args.base/'manifest.json', args.gd/'raw_error.npy',
               args.frozen_analysis/'summary.json', args.frozen_analysis/'assay_traces.npz']
    bounds = []
    for bandwidth in bandwidths:
        path = args.bounds/f'lambda{bandwidth:g}_q16_p24.npz'
        data = np.load(path)
        assert data['steps'][-1] == horizon
        assert np.isclose(data['actual_target_weights'].sum(), 1, atol=1e-10, rtol=0)
        bounds.append(data)
        sources.extend([path, path.with_suffix('.json')])

    style()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.labelsize': 8, 'axes.titlesize': 8.5,
                         'axes.titleweight': 'bold', 'axes.titlepad': 7,
                         'axes.linewidth': .7, 'axes.labelpad': 3,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'xtick.major.size': 3, 'ytick.major.size': 3,
                         'xtick.major.width': .7, 'ytick.major.width': .7,
                         'legend.fontsize': 7, 'legend.frameon': False,
                         'lines.linewidth': 1.25, 'pdf.fonttype': 42,
                         'savefig.pad_inches': .02})
    figure_size = (5.5, 2.0)
    png_dpi = 600
    mechanism, axes = plt.subplots(1, 2, figsize=figure_size, layout='constrained')
    acquisition, joint = plt.subplots(1, 2, figsize=(5.5, 1.65), layout='constrained')
    for ax in [*axes, *joint]:
        ax.set_yscale('log')
        ax.grid(axis='y', color='#D8D8D8', linewidth=.45, which='major')
        ax.set_axisbelow(True)

    rows = []
    for rank, marker in zip([4, 14, 30], ['o', 's', 'D']):
        lower, rates, upper = [np.array([b[key][rank-1] for b in bounds])
                               for key in ['rho_lower', 'actual_rho', 'rho_upper']]
        assert np.all(0 < lower) and np.all(lower <= rates) and np.all(rates <= upper)
        axes[0].fill_between(bandwidths, lower, upper, color='#677381', alpha=.14, lw=0)
        axes[0].plot(bandwidths, lower, color='#677381', ls=(0, (3, 2)), lw=.8)
        axes[0].plot(bandwidths, upper, color='#677381', ls=(0, (3, 2)), lw=.8)
        axes[0].plot(bandwidths, rates, color='#242424', marker=marker, ms=4,
                     mfc='#D6DEE5', mec='#242424', mew=.8, zorder=4)
        axes[0].annotate(f'$i={rank}$', (bandwidths[-1], rates[-1]),
                         xytext=(5, 0), textcoords='offset points', va='center', fontsize=7.5)
        rows.append(dict(rank=rank, actual=rates.tolist(), lower=lower.tolist(), upper=upper.tolist(),
                         target_weights=[float(b['actual_target_weights'][rank-1]) for b in bounds]))
    axes[0].set_xscale('log', base=2)
    axes[0].set_xlim(.027, .43)
    axes[0].set_ylim(1e-9, .25)
    axes[0].set_xticks(bandwidths, ['1/32', '1/16', '1/8', '1/4'])
    axes[0].set_yticks([1e-8, 1e-5, 1e-2])
    axes[0].minorticks_off()
    axes[0].set_title('a) Target-relevant rates')
    axes[0].set_xlabel(r'Bandwidth $\lambda=\gamma h$')
    axes[0].set_ylabel(r'$\mu_i/\mu_1$')
    axes[0].legend(handles=[Line2D([], [], color='#242424', marker='o', ms=4,
                                   mfc='#D6DEE5', mec='#242424', mew=.8, label='Kernel'),
                            Line2D([], [], color='#677381', ls=(0, (3, 2)), lw=.8,
                                   label='Theorem interval')],
                   loc='lower right', handlelength=1.8, labelspacing=.25,
                   borderaxespad=.45)

    marker_steps = np.unique(np.rint(np.geomspace(10, horizon, 12)).astype(int))
    for i, (bandwidth, color, bound) in enumerate(zip(bandwidths, palette, bounds)):
        axes[1].plot(indices, actual[indices, i], color=color, lw=1.25)
        axes[1].plot(bound['steps'], bound['lower_error'], color=color,
                     ls=(0, (3, 2)), lw=1.05, zorder=3)
        axes[1].plot(marker_steps, actual[marker_steps, i], color=color, ls='none',
                     marker='o', ms=3.7, mfc='white', mec=color, mew=.85,
                     clip_on=False, zorder=4)
    bandwidth_legend = axes[1].legend(handles=[Line2D([], [], color=c, label=label)
        for c, label in zip(palette, ['1/32', '1/16', '1/8', '1/4'])],
        title='Bandwidth λ', title_fontsize=7, loc='lower left', ncol=2,
        handlelength=1.5, columnspacing=.8, labelspacing=.25, borderaxespad=.4)
    axes[1].add_artist(bandwidth_legend)
    axes[1].legend(handles=[Line2D([], [], color='#242424', marker='o', ms=3.7,
                                   mfc='white', mew=.85, label='GD'),
                            Line2D([], [], color='#242424', ls=(0, (3, 2)), label='Bound')],
                   loc='upper right', ncol=2, handlelength=1.8, columnspacing=.9,
                   borderaxespad=.45)
    axes[1].set_xscale('symlog', linthresh=10)
    axes[1].set_xlim(0, horizon)
    axes[1].set_xticks([0, 100, 10000, horizon], ['0', '$10^2$', '$10^4$', r'$5\!\times\!10^6$'])
    axes[1].set_ylim(1e-4, 1.5)
    axes[1].set_yticks([1, 1e-1, 1e-2, 1e-3, 1e-4])
    axes[1].minorticks_off()
    axes[1].set_title('b) Readout learning')
    axes[1].set_xlabel('Readout updates')
    axes[1].set_ylabel('Relative training error')

    fingerprints = []
    for path in sources:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        fingerprints.append(dict(path=str(path), sha256=digest))
    for suffix in ['pdf', 'png']:
        mechanism.savefig(args.output/f'spectrum_readout.{suffix}', dpi=png_dpi)
    plt.close(mechanism)
    spectrum_record = dict(
        figure_size_inches=list(figure_size), png_dpi=png_dpi,
        horizon=horizon, sources=fingerprints, spectral_rows=rows,
        bandwidths=bandwidths, gd_marker_steps=marker_steps.tolist(),
        selected_target_energy=np.sum([row['target_weights'] for row in rows], axis=0).tolist(),
        panel_titles=['a) Target-relevant rates', 'b) Readout learning'],
        display='Unchanged executed GD samples and saved theorem intervals. '
                'Different marker shapes identify ordered ranks; rank eigenvectors vary with bandwidth. '
                'White-filled GD markers identify executed updates; dashed curves are theorem bounds. '
                'The time axis is symlog with linear threshold 10 and the full 5000000-update endpoint.')
    (args.output/'spectrum_readout_provenance.json').write_text(
        json.dumps(spectrum_record, indent=2) + '\n')
    if args.spectrum_only:
        plt.close(acquisition)
        print(json.dumps(dict(figure='spectrum_readout', checked_spectral_intervals=12,
                              horizon=horizon, png_dpi=png_dpi)))
        return

    for optimizer in summary['selected']:
        color = {'adam': '#0072B2', 'gd': '#D55E00'}[optimizer]
        label = 'Adam' if optimizer == 'adam' else 'GD'
        t = np.r_[0, trace[f'{optimizer}_error_steps'], horizon]/1e6
        endpoint = trace[f'{optimizer}_error_endpoint']
        y, low, high = [np.vstack((initial_error, trace[f'{optimizer}_error_{key}'], endpoint))
                        for key in ['median', 'low', 'high']]
        joint[0].plot(t, np.median(y, axis=1), color=color, label=f'Joint {label}')
        joint[0].fill_between(t, low.min(axis=1), high.max(axis=1), color=color, alpha=.055, lw=0)
        for name, ls in [('rms', '-'), ('q99', '--')]:
            if name == 'rms':
                ts = np.r_[0, trace[f'{optimizer}_rms_steps'], horizon]
                start = trace[f'{optimizer}_lambda_rms_checkpoints'][0]
                end = trace[f'{optimizer}_rms_endpoint']
                ys, low, high = [np.vstack((start, trace[f'{optimizer}_rms_{key}'], end))
                                 for key in ['median', 'low', 'high']]
            else:
                ts = trace[f'{optimizer}_checkpoint_steps']
                ys = trace[f'{optimizer}_lambda_q99']
                low = high = ys
            joint[1].plot(ts/1e6, np.median(ys, axis=1), color=color, ls=ls,
                          label=f'{label} ' + ('RMS' if name == 'rms' else '99th'))
            joint[1].fill_between(ts/1e6, low.min(axis=1), high.max(axis=1), color=color, alpha=.08, lw=0)

    gi = next(i for i, row in enumerate(access['geometries'])
              if row['family'] == 'uniform' and np.isclose(row['lambda_rms'], .25))
    selected = access['geometries'][gi]['selected']
    assert np.isfinite(selected['validation_error'])
    curves = np.load(args.frozen_analysis/'assay_traces.npz')
    t, mid, low, high = reference_display(curves[f'g{gi}_steps'], curves[f'g{gi}_median'],
        curves[f'g{gi}_low'], curves[f'g{gi}_high'], float(curves[f'g{gi}_endpoint']))
    joint[0].plot(t/1e6, mid, color='#333333', ls='--', lw=.8, label='Fixed Adam')
    joint[0].fill_between(t/1e6, low, high, color='#333333', alpha=.055, lw=0)
    joint[0].plot(indices/1e6, actual[indices, -1], color='#777777', ls=':', label='Fixed GD')
    joint[1].axhline(.25, color='#333333', ls=':', lw=.8, label='Reference 1/4')
    joint[0].set_ylabel('Relative output error')
    joint[1].set_ylabel(r'Scaled slope $h|a_j|$')
    for ax in joint:
        ax.set_xlim(0, horizon/1e6)
        ax.set_xlabel('Updates (millions)')
    joint[0].legend(loc='upper right', ncol=2, handlelength=1.4, columnspacing=.7, labelspacing=.2)
    joint[1].legend(loc='lower right', ncol=2, handlelength=1.4, columnspacing=.7, labelspacing=.2)
    for fig, name in [(acquisition, 'joint_acquisition')]:
        for suffix in ['pdf', 'png']:
            fig.savefig(args.output/f'{name}.{suffix}', dpi=png_dpi)
        plt.close(fig)
    (args.output/'provenance.json').write_text(json.dumps(dict(
        figure_sizes_inches=dict(spectrum_readout=list(figure_size),
                                 joint_acquisition=[5.5, 1.65]), png_dpi=png_dpi,
        panel_identification='Figure 3 has inline a/b titles; Figure 4 remains the legacy pair.',
        horizon=horizon, sources=fingerprints,
        spectral_rows=rows, bandwidths=bandwidths, frozen_adam_selection=selected,
        selected_target_energy=np.sum([row['target_weights'] for row in rows], axis=0).tolist(),
        gd_marker_steps=marker_steps.tolist(),
        display='Same executed curves and extrema-preserving display reduction as section34_figure.py; '
                'spectrum uses saved theorem intervals, with ordered ranks recomputed at each bandwidth. '
                'Hollow markers identify executed GD at 12 log-spaced updates; bounds use every '
                'resolved target projection, not only the three ranks displayed in the spectral panel.'),
        indent=2) + '\n')
    print(json.dumps(dict(figure_size_inches=list(figure_size), png_dpi=png_dpi,
                          checked_spectral_intervals=12,
                          frozen_adam_recipe=selected['recipe_index'], horizon=horizon)))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
