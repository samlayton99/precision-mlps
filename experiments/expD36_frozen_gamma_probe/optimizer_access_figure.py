"""Paper figures from V4 ratio/time intervals and archived GD/Adam evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_optimizer_access'
GAMMAS = [8, 12, 16, 64]
COLORS = ['#276492', '#cd7338', '#238778', '#8564a3']
TARGET_NAMES = ['Sine mixture', 'Exponential of sine', 'Runge function', 'Quadratic', 'Single sine']


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8,
        'ytick.labelsize': 8, 'legend.fontsize': 7.6, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.edgecolor': '#89929a',
        'axes.linewidth': .65, 'lines.linewidth': 1.7, 'pdf.fonttype': 42,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white'})


def clean(ax):
    ax.grid(axis='y', which='major', color='#e5e8eb', linewidth=.65)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.65)


def gamma_axis(ax):
    ax.set_xscale('log')
    ax.set_xticks(GAMMAS, labels=[str(x) for x in GAMMAS])
    ax.minorticks_off()
    ax.set_xlim(7, 75)
    ax.set_xlabel(r'Common slope $\gamma$')


def save(fig, folder, name, manifest):
    for extension in ('pdf', 'png'):
        path = folder/f'{name}.{extension}'
        metadata = {'Creator': __name__}
        if extension == 'pdf':
            metadata.update(CreationDate=None, ModDate=None)
        fig.savefig(path, dpi=220, bbox_inches='tight', metadata=metadata)
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT)
    args = parser.parse_args()
    folder = args.source
    paths = [folder/'analysis.json', folder/'projections.npz', folder/'trace_envelopes.npz',
        folder.parent/'gamma_factorized_kernel/summary.json', Path(__file__),
        folder/'ema_analysis.json', folder/'ema_curves.npz']
    analysis = json.loads(paths[0].read_text())
    projection = np.load(paths[1])
    traces = np.load(paths[2])
    factorized = json.loads(paths[3].read_text())
    ema_analysis = json.loads(paths[5].read_text())
    ema_curves = np.load(paths[6])
    direct_folder = folder.parent/'gamma_direct_ratio'
    direct, direct_arrays = {}, {}
    for gamma in GAMMAS:
        summary_path = direct_folder/f'archive_g{gamma}_q10_p20.json'
        arrays_path = summary_path.with_suffix('.npz')
        direct[gamma] = json.loads(summary_path.read_text())
        direct_arrays[gamma] = np.load(arrays_path)
        paths.extend([summary_path, arrays_path])
    ema_cases = {(c['gamma'], c['view'], c['target']): c for c in ema_analysis['cases']}
    half_life = 100
    ema_index = list(ema_curves['half_lives']).index(half_life)
    def ema_case(gamma, view, target, window=half_life):
        return next(c for c in ema_cases[gamma, view, target]['ema']
                    if c['half_life'] == window)
    cases = {(c['gamma'], c['view'], c['target']): c for c in analysis['cases']}
    targets = analysis['targets']
    first = targets[0]
    style()
    manifest = {}
    evidence = dict(source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in paths},
        protocol=dict(geometry='N512, q16, 8193 samples, 559 tanh features plus bias',
            primary_target=first, adam_horizon=200000, gd_horizon='Longer archived runs, up to17000000',
            sustained_definition=analysis['sustained_definition'], schedule=analysis['schedule'],
            main_adam_metric='First crossing of EMA squared relative error at 1e-4; half-life 100 updates; initialized at actual squared error at update 0.',
            ema_choice='One common post-hoc visualization window; sensitivity at 10,30,100,300,1000,3000 updates.',
            numerical_status='Checked FP64 V4 p=0 theorem intervals, including analytic tails and numerical allowances; Adam projections are empirical.',
            theorem_source='gamma_direct_ratio/archive_g*_q10_p20; same archived geometry and GD steps',
            output_error_lower_bound='Uses theorem rho_upper with actual target weights and archived eta*mu1; unresolved energy is omitted conservatively. Actual rho supplies only the separate reference decay rates and resolution check.',
            positive_modes='Only retained positive SVD modes; omitted numerical directions excluded.',
            energy_normalization=analysis['normalization'],
            connecting_lines='Lines between evaluated gamma values and saved Adam checkpoints are visual guides, not fitted dynamics.'), figures={})

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.15))
    fig.subplots_adjust(left=.065, right=.99, top=.84, bottom=.29, wspace=.38)
    for ax in axes:
        clean(ax)
    ax = axes[0]
    ratio_data = []
    for gamma, color in zip(GAMMAS, COLORS):
        values = projection[f'g{gamma}_rho'][::-1][:40]
        ranks = np.arange(1, len(values)+1)
        low = direct_arrays[gamma]['rho_lower'][:len(values)]
        high = direct_arrays[gamma]['rho_upper'][:len(values)]
        ax.fill_between(ranks, np.maximum(low, 1e-16), high, color=color, alpha=.16,
                        linewidth=0)
        ax.plot(ranks, values, color=color, label=rf'$\gamma={gamma}$')
        ratio_data.append(dict(gamma=gamma, ranks=ranks, eigenvalue_ratios=values,
                               ratio_lower=low, ratio_upper=high))
    ax.set(yscale='log', ylim=(1e-12, 1.5), xlim=(1, 40),
           xlabel='Eigenvalue rank', ylabel=r'Eigenvalue ratio $\mu_i/\mu_1$')
    ax.set_xticks([1, 10, 20, 30, 40])
    ax.set_yticks([1e-12, 1e-8, 1e-4, 1])
    ax.set_title('A  Gamma sets relative learning rates', loc='left', pad=19)
    ax.text(0, 1.035, 'Lines: finite spectrum   Bands: lemma intervals', transform=ax.transAxes,
            fontsize=7.3, color='#53616c')
    ax.legend(ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.28))

    ax = axes[1]
    energy_data = []
    for gamma, color in zip(GAMMAS, COLORS):
        rho = projection[f'g{gamma}_rho']
        initial = projection[f'g{gamma}_initial_cumulative'][:, 0]
        residual = projection[f'g{gamma}_common_residual_cumulative'][-1, :, 0]
        ax.plot(rho, initial, color=color, linestyle='--', linewidth=1.05, alpha=.8)
        ax.plot(rho, residual, color=color, label=rf'$\gamma={gamma}$')
        energy_data.append(dict(gamma=gamma, rho=rho, initial_cumulative=initial,
                               adam_final_cumulative=residual))
    ax.axhline(1e-4, color='#737b82', linestyle=':', linewidth=1)
    ax.axvline(2e-6, color='#a7afb6', linestyle=':', linewidth=.8)
    ax.set(xscale='log', yscale='log', xlim=(1e-10, 1), ylim=(1e-9, 1.5),
           xlabel=r'Ratio cutoff $\rho$', ylabel=r'Energy in modes $0<\mu_i/\mu_1\leq\rho$')
    ax.set_xticks([1e-10, 1e-6, 1e-2, 1])
    ax.set_title('B  Adam leaves slow-mode residual', loc='left', pad=19)
    ax.text(0, 1.035, 'Dashed: initial target   Solid: Adam at 200k', transform=ax.transAxes,
            fontsize=7.3, color='#53616c')
    ax.text(.97, .565, '(1% error)²', transform=ax.transAxes, ha='right', fontsize=7,
            color='#68727b')
    ax.legend(ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.28))

    ax = axes[2]
    archived = {d['gamma']: d for d in factorized['dictionaries']}
    gd_actual = [archived[g]['executed_hits'][0] for g in GAMMAS]
    gd_forecast = [archived[g]['reference_hits'][0] for g in GAMMAS]
    necessary = [direct[g]['necessary_updates'] for g in GAMMAS]
    sufficient = [direct[g]['sufficient_updates'] for g in GAMMAS]
    if any(value is None for value in necessary+sufficient):
        raise ValueError('The main plot requires finite bounds at the four archived slopes.')
    if not all(a <= b <= c for a, b, c in zip(necessary, gd_actual, sufficient)):
        raise ValueError('Recomputed theorem interval does not enclose executed GD.')
    adam = [ema_case(g, 'common', first)['first_hit'] for g in GAMMAS]
    ax.fill_between(GAMMAS, necessary, sufficient, color='#5888ab', alpha=.22,
                    linewidth=0, label='GD time interval')
    ax.plot(GAMMAS, necessary, color='#2874a8', linestyle='--', linewidth=1.5,
            label='Theorem: necessary GD time')
    ax.plot(GAMMAS, gd_forecast, color='#253c51', linewidth=1.6, label='GD spectrum forecast')
    ax.plot(GAMMAS, gd_actual, color='#253c51', linestyle='none', marker='o',
            markerfacecolor='white', markersize=5, label='Executed GD')
    ax.plot(GAMMAS[1:], adam[1:], color='#ce7041', marker='s', markersize=4,
            label='Adam first EMA hit')
    ax.scatter([8], [200000], marker='^', color='#ce7041', s=28, zorder=4)
    ax.annotate('', xy=(8, 4.2e5), xytext=(8, 2.05e5),
                arrowprops=dict(arrowstyle='->', color='#ce7041', lw=1.1))
    ax.text(8, 5e5, '>200k', color='#b65d32', fontsize=7.5)
    gamma_axis(ax)
    ax.set_yscale('log')
    ax.set_ylim(700, 5e7)
    ax.set_ylabel('Updates to 1% error (Adam: smoothed)')
    ax.set_title('C  Error lower bound forces delay', loc='left', pad=19)
    ax.text(0, 1.035, 'Adam: first loss-EMA crossing, half-life 100', transform=ax.transAxes,
            fontsize=7.3, color='#53616c')
    ax.legend(ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(.45, -.28),
              columnspacing=.8, handlelength=1.7, fontsize=7)
    evidence['figures']['optimizer_access_three_panel'] = dict(ratios=ratio_data,
        energy=energy_data, timings=dict(gamma=GAMMAS, gd_actual=gd_actual,
        gd_full_forecast=gd_forecast, gd_forecast_source='True finite-tanh spectrum reference_hits',
        gd_necessary=necessary, gd_sufficient=sufficient,
        gd_bound_source='V4 p=0 intervals at the archived geometry and actual step',
        adam_first_ema=adam,
        adam_ema_half_life=half_life,
        adam_censor_horizon=200000), slow_cutoff=2e-6, error_squared_threshold=1e-4)
    save(fig, folder, 'optimizer_access_three_panel', manifest)

    fig, axes = plt.subplots(2, 2, figsize=(9.3, 5.8), sharey=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.15, top=.9, hspace=.4, wspace=.17)
    bound_curves = []
    for ax, gamma, color, actual_hit in zip(axes.flat, GAMMAS, COLORS, gd_actual):
        data = direct_arrays[gamma]
        steps = data['steps']
        ax.fill_between(steps, data['lower_error'], data['upper_error'], color=color,
                        alpha=.16, linewidth=0, label='Theorem error interval')
        ax.plot(steps, data['lower_error'], color=color, linestyle='--', linewidth=1,
                label='Theorem output-error lower bound')
        ax.plot(steps, data['upper_error'], color=color, linestyle=':', linewidth=1.2,
                label='Companion output-error upper bound')
        ax.plot(steps, data['reference_error'], color='#253c51', linewidth=1.5,
                label='Finite-spectrum error')
        ax.axhline(.01, color='#7a8289', linestyle=':', linewidth=.9)
        ax.scatter([actual_hit], [.01], edgecolor='#253c51', facecolor='white',
                   s=26, zorder=5, label='Executed GD 1% crossing')
        lo, hi = direct[gamma]['necessary_updates'], direct[gamma]['sufficient_updates']
        ax.set_title(rf'$\gamma={gamma}$'+f'   {lo:,} ≤ steps ≤ {hi:,}', loc='left', fontsize=9)
        ax.set(xscale='log', yscale='log', xlim=(1, max(1e5, hi*4)), ylim=(1e-4, 1.3))
        clean(ax)
        bound_curves.append(dict(gamma=gamma, steps=steps, lower_error=data['lower_error'],
            upper_error=data['upper_error'], reference_error=data['reference_error'],
            necessary_updates=lo, sufficient_updates=hi, executed_updates=actual_hit))
    for ax in axes[:, 0]:
        ax.set_ylabel('Relative training error')
    for ax in axes[-1]:
        ax.set_xlabel('GD updates')
    fig.suptitle('Theorem rate caps predict a lower bound on output error', fontsize=11, y=.99)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[1:], labels[1:], loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(.5, -.005))
    evidence['figures']['optimizer_access_gd_bounds'] = bound_curves
    save(fig, folder, 'optimizer_access_gd_bounds', manifest)

    fig, axes = plt.subplots(1, 5, figsize=(11.5, 3.2), sharey=True)
    fig.subplots_adjust(bottom=.26, top=.8, left=.065, right=.99, wspace=.16)
    target_data = []
    for ti, (ax, target, label) in enumerate(zip(axes, targets, TARGET_NAMES)):
        entry = dict(target=target, gamma=GAMMAS)
        for view, color, marker in [('common', '#276492', 'o'), ('selected', '#ce7041', 's')]:
            hits = [ema_case(g, view, target)['first_hit'] for g in GAMMAS]
            plotted = np.array(hits, float); plotted[plotted < 0] = np.nan
            ax.plot(GAMMAS, plotted, color=color, marker=marker, markersize=3.6,
                    label='Common setting' if view == 'common' else 'Pilot-selected setting')
            for g, hit in zip(GAMMAS, hits):
                if hit < 0:
                    ax.scatter([g], [200000], marker='^', color=color, s=27)
                    ax.annotate('', xy=(g, 3.1e5), xytext=(g, 2.05e5),
                                arrowprops=dict(arrowstyle='->', color=color, lw=1))
            entry[view] = hits
        gamma_axis(ax); clean(ax)
        ax.set(yscale='log', ylim=(500, 4e5), title=label)
        ax.axhline(200000, color='#b8c0c6', linewidth=.8, linestyle=':')
        target_data.append(entry)
    axes[0].set_ylabel('First loss-EMA crossing (updates)')
    fig.suptitle('Adam across five targets: first 1% crossing of smoothed loss (half-life 100)', y=.99, fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
               frameon=False, bbox_to_anchor=(.5, -.01))
    evidence['figures']['optimizer_access_targets'] = target_data
    save(fig, folder, 'optimizer_access_targets', manifest)

    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6), sharex=True, sharey=True)
    fig.subplots_adjust(bottom=.14, top=.88, left=.09, right=.98, hspace=.35, wspace=.17)
    schedule_data = []
    edges = traces['bin_edges']; time = traces['steps']
    for gi, (ax, gamma, color) in enumerate(zip(axes.flat, GAMMAS, COLORS)):
        low = traces['bin_min'][:, gi, 5]; high = traces['bin_max'][:, gi, 5]
        error = traces['errors'][:, gi, 5]
        xx = np.repeat(edges, 2)[1:-1]
        ax.fill_between(xx, np.repeat(low, 2), np.repeat(high, 2), color=color, alpha=.22,
                        linewidth=0, label='All-update min–max envelope')
        ema_steps = ema_curves['steps']
        smoothed = np.sqrt(ema_curves['ema_squared'][ema_index, :, gi, 5])
        ax.plot(ema_steps, smoothed, color=color, linewidth=1.4,
                label='Square root of loss EMA (half-life 100)')
        ax.axhline(.01, color='#777f85', linestyle=':', linewidth=1)
        ax.axvspan(20000, 50000, color='#aeb8c1', alpha=.16)
        case = cases[gamma, 'common', first]
        hit = ema_case(gamma, 'common', first)['first_hit']
        if hit >= 0:
            ax.axvline(hit, color='#374655', linestyle='--', linewidth=.8)
            ax.scatter([hit], [.01], color=color, s=22, zorder=5)
        hit_text = f'{hit:,}' if hit >= 0 else '>200,000'
        ax.set_title(rf'$\gamma={gamma}$'+'   First EMA crossing: '+hit_text,
                     loc='left', fontsize=9)
        ax.set(xscale='symlog', yscale='log', xlim=(0, 200000), ylim=(1e-6, 2))
        ax.set_xticks([0, 1000, 20000, 200000], labels=['0', '1k', '20k', '200k'])
        clean(ax)
        schedule_data.append(dict(gamma=gamma, bin_edges=edges, bin_min=low, bin_max=high,
                                  steps=time, errors=error, first_hit=case['first_hit'],
                                  sustained_hit=case['sustained_hit'], ema_steps=ema_steps,
                                  ema_relative_error=smoothed, first_ema_hit=hit))
    for ax in axes[-1]:
        ax.set_xlabel('Adam updates')
    for ax in axes[:, 0]:
        ax.set_ylabel('Relative error')
    fig.suptitle('First EMA crossings resolve early acquisition', y=.98, fontsize=12)
    fig.text(.5, .935, 'Sine mixture · common Adam setting · shaded vertical window: learning-rate decay (20k–50k)',
             ha='center', fontsize=8.5, color='#56636e')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=2,
               frameon=False, bbox_to_anchor=(.5, .015))
    evidence['figures']['optimizer_access_schedule'] = schedule_data
    save(fig, folder, 'optimizer_access_schedule', manifest)

    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.65), sharey=True)
    fig.subplots_adjust(bottom=.26, top=.83, left=.09, right=.98, wspace=.16)
    sensitivity = []
    windows = list(ema_curves['half_lives'])
    memory_floor = np.ceil(np.array(windows)*np.log2(1e4)).astype(int)
    for ax, view in zip(axes, ('common', 'selected')):
        for gamma, color in zip(GAMMAS, COLORS):
            hits = [ema_case(gamma, view, first, int(h))['first_hit'] for h in windows]
            displayed = np.array(hits, float); displayed[displayed < 0] = np.nan
            ax.plot(windows, displayed, color=color, marker='o', markersize=3.6,
                    label=rf'$\gamma={gamma}$')
            censored = np.array(windows)[np.array(hits) < 0]
            ax.scatter(censored, np.full(len(censored), 200000), marker='^', color=color, s=24)
            sensitivity.append(dict(view=view, gamma=gamma, half_lives=windows, first_hits=hits))
        ax.plot(windows, memory_floor, color='#77838d', linestyle='--', linewidth=1.1,
                label='Initial-loss memory floor')
        ax.axvline(half_life, color='#a7afb6', linestyle=':', linewidth=.9)
        ax.axhline(200000, color='#a7afb6', linestyle=':', linewidth=.8)
        ax.set(xscale='log', yscale='log', ylim=(100, 3.5e5),
               xlabel='Loss-EMA half-life (updates)',
               title='Common Adam setting' if view == 'common' else 'Validation-selected Adam setting')
        ax.set_xticks(windows, labels=[str(h) for h in windows]); clean(ax)
    axes[0].set_ylabel('First 1% loss-EMA crossing (updates)')
    fig.suptitle('Window sensitivity: one shared half-life, no per-gamma tuning', y=.98, fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5,
               frameon=False, bbox_to_anchor=(.5, -.005))
    evidence['figures']['optimizer_access_ema_sensitivity'] = sensitivity
    evidence['ema_memory_floor'] = dict(half_lives=windows, earliest_possible_crossings=memory_floor,
        reason='M0=1 and nonnegative later losses imply Mn >= beta^n.')
    save(fig, folder, 'optimizer_access_ema_sensitivity', manifest)
    evidence['artifact_sha256'] = manifest
    (folder/'figure_data.json').write_text(json.dumps(plain(evidence), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
