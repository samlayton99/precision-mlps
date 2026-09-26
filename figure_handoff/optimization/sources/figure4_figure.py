"""Paper Figure 4: width scaling, output accuracy, and population bandwidth."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from section34_figure import reference_display


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'analysis', 'reference-base', 'frozen-analysis', 'gd', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    widths = np.array([128, 256, 512, 1024])
    summaries = json.loads((args.analysis/'width_summary.json').read_text())
    summary = summaries['512']
    trace = np.load(args.analysis/'w512/selected_traces.npz')
    reference = json.loads((args.reference_base/'manifest.json').read_text())
    access = json.loads((args.frozen_analysis/'summary.json').read_text())
    gd_meta = json.loads((args.gd/'summary.json').read_text())
    assert reference['input_sha256'] == digest(args.reference_base/'input.npz') == access['input_sha256'] == gd_meta['input_sha256']
    with np.load(args.root/'w512/base/input.npz') as current, np.load(args.reference_base/'input.npz') as original:
        for key in ['train_x', 'target', 'validation_x', 'validation_target', 'eval_x', 'eval_target']:
            np.testing.assert_array_equal(current[key], original[key])
    assert reference['spacing'] == summary['spacing'] and reference['width'] == 512
    horizon = 5000000
    assert horizon == access['assay_steps'] == gd_meta['steps']
    assert all(s['horizon'] == horizon for s in summaries.values())
    gi = next(i for i, row in enumerate(access['geometries'])
              if row['family'] == 'uniform' and np.isclose(row['lambda_rms'], .25))
    frozen_adam = access['geometries'][gi]
    curves = np.load(args.frozen_analysis/'assay_traces.npz')
    ft, fm, fl, fh = reference_display(curves[f'g{gi}_steps'], curves[f'g{gi}_median'],
        curves[f'g{gi}_low'], curves[f'g{gi}_high'], float(curves[f'g{gi}_endpoint']))
    gd = np.load(args.gd/'raw_error.npy', mmap_mode='r')
    assert len(gd) == horizon+1
    gd_column = gd_meta['lambdas'].index(.25)
    gd_steps = np.unique(np.r_[0, np.geomspace(1, horizon, 1800).astype(int),
                              np.linspace(0, horizon, 1800).astype(int)])
    sources = [args.analysis/'width_summary.json', args.analysis/'w512/selected_traces.npz',
               args.reference_base/'manifest.json', args.reference_base/'input.npz',
               args.frozen_analysis/'summary.json', args.frozen_analysis/'assay_traces.npz',
               args.gd/'summary.json', args.gd/'raw_error.npy']
    for width in widths:
        sources.extend([args.analysis/f'w{width}/summary.json',
                        args.root/f'w{width}/base/manifest.json'])
        sources.extend(sorted((args.root/'durable'/f'w{width}').glob('*/step*/manifest.json')))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
        'axes.labelsize': 8, 'axes.labelpad': 3, 'axes.titlesize': 8.5,
        'axes.titleweight': 'bold', 'axes.titlepad': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': .7, 'ytick.major.width': .7,
        'legend.fontsize': 7, 'legend.frameon': False,
        'pdf.fonttype': 42, 'savefig.pad_inches': .025})
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.2))
    fig.subplots_adjust(left=.082, right=.989, top=.835, bottom=.29, wspace=.56)
    colors = {'adam': '#0072B2', 'gd': '#D55E00'}
    plot_data = dict(widths=widths)
    endpoint_rows = {}
    for optimizer, marker in [('adam', 'o'), ('gd', 's')]:
        color = colors[optimizer]
        seed_means = np.array([summaries[str(w)]['selected'][optimizer]['endpoint_seed_mean_slope'] for w in widths])
        means = np.mean(seed_means, axis=1)
        for seed in range(seed_means.shape[1]):
            axes[0].plot(widths, seed_means[:, seed], color=color, alpha=.25, lw=.55,
                         marker=marker, ms=2.5, mec=color, mew=.45)
        axes[0].plot(widths, means, color=color, lw=1.4, marker=marker, ms=4.5,
                     mfc='white', mec=color, mew=1, zorder=5)
        plot_data[f'{optimizer}_slope_seed_means'] = seed_means
        plot_data[f'{optimizer}_slope_means'] = means
        endpoint_rows[optimizer] = dict(mean_slopes=means.tolist(),
            mean_bandwidths=[summaries[str(w)]['selected'][optimizer]['endpoint_mean_bandwidth'] for w in widths],
            median_training_errors=[float(np.median(summaries[str(w)]['selected'][optimizer]['train_errors'])) for w in widths])

        chosen = summary['selected'][optimizer]
        initial = np.load(Path(chosen['run'])/'relative_error.npy', mmap_mode='r')[0, :, chosen['recipe_index']]
        t = np.r_[0, trace[f'{optimizer}_error_steps'], horizon]/1e6
        endpoint = trace[f'{optimizer}_error_endpoint']
        typical, low, high = [np.vstack((initial, trace[f'{optimizer}_error_{k}'], endpoint))
                              for k in ['median', 'low', 'high']]
        median = np.median(typical, axis=1)
        axes[1].fill_between(t, low.min(axis=1), high.max(axis=1), color=color, alpha=.075, lw=0)
        axes[1].plot(t, median, color=color, lw=1.35, zorder=4)
        marks = np.unique(np.linspace(0, len(t)-1, 8).astype(int))
        axes[1].plot(t[marks], median[marks], color=color, ls='none', marker=marker,
                     ms=3.1, mfc='white', mec=color, mew=.85, zorder=5, clip_on=False)
        plot_data[f'{optimizer}_error_steps'] = t*1e6
        plot_data[f'{optimizer}_error_median'] = median
        plot_data[f'{optimizer}_error_low'] = low.min(axis=1)
        plot_data[f'{optimizer}_error_high'] = high.max(axis=1)

        steps = trace[f'{optimizer}_checkpoint_steps']
        mean = trace[f'{optimizer}_lambda_mean']; sd = trace[f'{optimizer}_lambda_population_std']
        assert steps[0] == 0 and steps[-1] == horizon and len(mean) == 501
        axes[2].fill_between(steps/1e6, np.maximum(0, mean-sd), mean+sd, color=color, alpha=.17, lw=0)
        axes[2].plot(steps/1e6, mean, color=color, lw=1.35)
        marks = np.arange(0, len(steps), 100)
        axes[2].plot(steps[marks]/1e6, mean[marks], ls='none', marker=marker,
                     ms=3.1, mfc='white', mec=color, mew=.85, zorder=5, clip_on=False)
        plot_data[f'{optimizer}_bandwidth_steps'] = steps
        plot_data[f'{optimizer}_bandwidth_mean'] = mean
        plot_data[f'{optimizer}_bandwidth_std'] = sd

    construction = np.array([.25/summaries[str(w)]['spacing'] for w in widths])
    axes[0].plot(widths, construction, color='#444444', ls=(0,(2,2)), lw=1)
    axes[0].annotate(r'Supplied $\lambda=1/4$', (widths[-1], construction[-1]),
                     xytext=(0, 5), textcoords='offset points', ha='right', fontsize=6.6)
    axes[0].set_xscale('log', base=2); axes[0].set_yscale('log')
    axes[0].set_xticks(widths, [str(w) for w in widths]); axes[0].set_xlim(110, 1200)
    axes[0].set_ylim(.1, 260); axes[0].set_yticks([.1, 1, 10, 100])
    axes[0].set_xlabel('Total width $W$'); axes[0].set_ylabel(r'Mean slope $|\gamma|$')
    axes[1].plot(ft/1e6, fm, color=colors['adam'], ls=(0,(4,2)), lw=1.05)
    axes[1].fill_between(ft/1e6, fl, fh, color=colors['adam'], alpha=.07, lw=0)
    axes[1].plot(gd_steps/1e6, gd[gd_steps, gd_column], color=colors['gd'], ls=(0,(4,2)), lw=1.05)
    highest_error = max(float(np.max(fh)), *(float(np.max(plot_data[f'{o}_error_high'])) for o in colors))
    axes[1].set_yscale('log'); axes[1].set_ylim(1e-7, highest_error*1.3)
    axes[1].set_yticks([1e1, 1e-1, 1e-3, 1e-5, 1e-7]); axes[1].set_ylabel('Relative training error')
    axes[2].axhline(.25, color='#444444', ls=(0,(2,2)), lw=1)
    axes[2].text(4.9, .257, r'Supplied $\lambda=1/4$', ha='right', fontsize=6.6)
    axes[2].set_ylim(0, max(.3, max(float(np.max(plot_data[f'{o}_bandwidth_mean']+plot_data[f'{o}_bandwidth_std'])) for o in colors)*1.05))
    axes[2].set_yticks([0, .1, .2, .3]); axes[2].set_ylabel(r'Bandwidth $\lambda$')
    for ax in axes[1:]:
        ax.set_xlim(0, 5); ax.set_xticks([0, 2, 4, 5]); ax.set_xlabel('Updates (millions)')
    for ax, title in zip(axes, ['a) Slope scaling', 'b) Output accuracy', 'c) Acquired bandwidth']):
        ax.set_title(title); ax.minorticks_off()
        ax.grid(axis='y', color='#D8D8D8', lw=.45); ax.set_axisbelow(True)
    handles = [Line2D([], [], color=colors[o], lw=1.3, marker=m, ms=3.5,
                       mfc='white', mew=.85, label=f'Joint {label}')
               for o, m, label in [('adam','o','Adam'), ('gd','s','GD')]]
    handles += [Line2D([], [], color=colors[o], ls=(0,(4,2)), lw=1.1, label=f'Frozen {label}')
                for o, label in [('adam','Adam'), ('gd','GD')]]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, -.005), ncol=4,
               handlelength=2, columnspacing=1.5, handletextpad=.5)
    for extension in ['pdf', 'png']:
        fig.savefig(args.output/f'joint_acquisition.{extension}', dpi=600, bbox_inches='tight')
    plt.close(fig)
    plot_data.update(construction_slopes=construction, frozen_adam_steps=ft, frozen_adam_error=fm,
                     frozen_adam_low=fl, frozen_adam_high=fh, frozen_gd_steps=gd_steps,
                     frozen_gd_error=np.asarray(gd[gd_steps, gd_column]))
    np.savez_compressed(args.output/'joint_acquisition_plot_data.npz', **plot_data)
    record = dict(figure_size_inches=[5.5,2.2], png_dpi=600, horizon=horizon,
        endpoints=endpoint_rows, frozen_adam=frozen_adam['selected'],
        frozen_gd_endpoint=float(gd[-1, gd_column]),
        selection=summary['selection'], bandwidth_shading='Population standard deviation across all neurons of five equally sized seeds; lower edge clipped at zero.',
        error_display='Median of per-seed bin medians, with exact endpoint; shading spans seed/bin extrema. Frozen Adam retains within-bin extrema.',
        comparison='Frozen training arrays, target, sampling grids, and reference spacing match the rerun exactly; frozen runs are archived executions.',
        sources=[dict(path=str(p), sha256=digest(p)) for p in sources])
    (args.output/'joint_acquisition_provenance.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    main()
