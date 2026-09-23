"""Evidence-only figure: gamma attenuation, positive slow mass, executed GD.

Reuses archived ordinary-tanh references and the uniform-grid corrected model.
No training, fitted rates, or report generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator
import numpy as np

from .finite_gamma_gram import error, first_hit
from .gamma_filter import multiplier
from .pi_brief_figure import COLORS
from .review_figures import DEFAULT, GAMMAS, TARGETS
from .uniform_grid_spectrum import buffered_mass


def collect(source):
    root = source.parent.parent
    uniform = source.parent/'uniform_grid_spectrum'
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw) if path.suffix == '.json' else dict(np.load(path))

    summary = read(uniform/'summary.json')
    original = read(source/'summary.json')
    campaign = read(source.parent/'capped_kernel/campaign_summary.json')
    certificates = read(source/'interval_audit.json')
    assert summary['targets'][0] == original['targets'][0] == TARGETS[0]
    floor_rate, probe_rate = 1e-9, 1e-6
    cutoffs = np.unique(np.r_[np.geomspace(1.01e-9, .02, 501), probe_rate])
    buffers = np.geomspace(1e-5, .5, 201)
    frequency = np.linspace(0, 8, 401)
    rows = []
    for gamma in GAMMAS:
        row = next(r for r in summary['rows'] if r['n'] == 512 and r['gamma'] == gamma)
        dictionary = next(r for r in original['dictionaries'] if r['gamma'] == gamma)
        model = read(uniform/f'N512_g{gamma}.npz')
        reference = read(source/f'reference_g{gamma}.npz')
        normalized_error = row['eta']*row['total_kernel_allowance']

        def bounds(cutoff):
            intervals = [buffered_mass(model, cutoff, cutoff*f, normalized_error)
                         for f in buffers]
            return (float(max(x[0][0] for x in intervals)),
                    float(min(x[1][0] for x in intervals)))

        lower_floor, upper_floor = bounds(floor_rate)
        low, high, measured_mass = [], [], []
        for cutoff in cutoffs:
            lower, upper = bounds(cutoff)
            low.append(max(0., lower-upper_floor))
            high.append(min(1., upper-lower_floor))
            mask = (reference['rates'] > floor_rate) & (reference['rates'] <= cutoff)
            measured_mass.append(float(reference['weights'][mask, 0].sum()))
        assert np.all(np.asarray(low) <= np.asarray(measured_mass)+1e-10)
        assert np.all(np.asarray(high)+1e-10 >= measured_mass)
        probe = int(np.flatnonzero(cutoffs == probe_rate)[0])
        p = low[probe]
        necessary = (int(np.ceil(np.log(np.sqrt(p)/.01)/-np.log1p(-probe_rate)))
                     if p > 1e-4 else 0)
        predicted_hit = int(first_hit(model)[0])
        actual = next(c for c in campaign['cases'] if c['id'] == dictionary['archived_case'])
        archive = source.parent/'capped_kernel/evidence/gd_trajectories'/actual['id']
        meta, training = read(archive/'meta.json'), read(archive/'training.json')
        checkpoints = [dict(step=c['step'], residual=c['train'][0])
                       for c in read(archive/'curve.json')]
        assert training == actual['training']
        assert meta['matrix_hash'] == actual['matrix_hash'] == dictionary['matrix_hash']
        assert meta['target_hash'] == actual['target_hash']
        assert meta['eta'] == row['eta'] == dictionary['eta']
        assert meta['map'] == 'raw' and meta['initialization'] == 'zero'
        assert all(a['step'] < b['step'] for a, b in zip(checkpoints, checkpoints[1:]))
        hit = int(training['hits'][0][0])
        assert hit == predicted_hit == row['executed_hits'][0]
        assert checkpoints[-1] == dict(step=training['steps'], residual=training['final_train'][0])
        certificate = next(r for r in certificates['results'] if r['gamma'] == gamma)
        inherited = certificate['methods']['combined']
        assert inherited['status'] == 'interval_certified_endpoints'
        interval = row['intervals'][0]
        assert interval['necessary'] <= inherited['necessary'] <= hit
        assert hit <= inherited['sufficient'] <= interval['sufficient']
        checkpoint_gap = max(abs(float(error(model, c['step'])[0])-c['residual'])
                             for c in checkpoints)
        assert checkpoint_gap < 1e-9
        positive = [i for i, c in enumerate(checkpoints) if c['step'] > 0]
        log_steps = np.log([checkpoints[i]['step'] for i in positive])
        marker_indices = np.unique([positive[int(np.argmin(abs(log_steps-s)))]
                                    for s in np.linspace(log_steps[0], log_steps[-1], 18)])
        steps = np.unique(np.r_[0, np.geomspace(1000, training['steps'], 420).astype(int), hit])
        residual = [float(error(model, int(n))[0]) for n in steps]
        rows.append(dict(gamma=gamma, eta=row['eta'], normalized_kernel_allowance=normalized_error,
            multiplier_squared=(multiplier(gamma, 2*np.pi*frequency)**2).tolist(),
            reference_positive_mass=measured_mass, positive_mass_lower=low, positive_mass_upper=high,
            lower_cutoff_mass_bounds=[lower_floor, upper_floor],
            probe_mass_lower=p, probe_mass_reference=measured_mass[probe],
            probe_necessary_updates=necessary, predicted_hit=predicted_hit, executed_hit=hit,
            timing_interval=interval, inherited_primary_certificate=inherited,
            steps=steps.tolist(), predicted_residual=residual, checkpoints=checkpoints,
            marker_indices=marker_indices.tolist(), maximum_checkpoint_gap=float(checkpoint_gap)))
    assert abs(rows[0]['probe_mass_lower']-.04332247241737147) < 1e-12
    assert rows[0]['probe_necessary_updates'] == 3035627
    return dict(target=TARGETS[0], gammas=GAMMAS, frequency_cycles=frequency.tolist(),
        rate_cutoffs=cutoffs.tolist(), positive_lower_cutoff=floor_rate, probe_cutoff=probe_rate,
        buffer_fractions=buffers.tolist(), rows=rows,
        source_sha256=sources, plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        panel_roles=dict(A='Exact analytic multiplier squared, not an empirical curve.',
            B='Original ordinary-tanh target-weighted spectrum and positive-band bounds from corrected-model buffered masses.',
            C='Corrected-model full spectral predictions, actual saved GD checkpoints, and actual 1% crossings.'),
        numerical_status=summary['numerical_status'],
        positive_band_convention='(1e-9,t], excluding all zero and smaller rates; bound is max(0,L(t)-U(a)), min(1,U(t)-L(a)).',
        plotting_convention='All data retained. Positive mass below 1e-8 is outside panel B. Bounds may be thinner than lines. GD markers chosen only by nearest log-update counts.',
        gpu_hours=0)


def plot(data, output):
    ink, gray = '#263442', '#78838C'
    plt.rcParams.update({'font.family':'serif', 'font.serif':['STIXGeneral'],
        'mathtext.fontset':'stix', 'font.size':9, 'axes.titlesize':10,
        'axes.titlelocation':'left', 'axes.titlepad':10, 'text.color':ink,
        'axes.labelcolor':ink, 'xtick.color':ink, 'ytick.color':ink,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.edgecolor':'#9AA3AA',
        'axes.linewidth':.6, 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.55),
                             gridspec_kw={'width_ratios':[1, 1.18, 1.35]})
    fig.subplots_adjust(left=.065, right=.99, bottom=.17, top=.77, wspace=.44)
    fig.legend(handles=[Line2D([], [], color=c, lw=2, label=rf'$\gamma={g}$')
                        for g, c in zip(GAMMAS, COLORS)],
               loc='upper center', bbox_to_anchor=(.5, 1.015), ncol=4,
               frameon=False, fontsize=10, columnspacing=2.5)
    ax = axes[0]
    for row, color in zip(data['rows'], COLORS):
        ax.semilogy(data['frequency_cycles'], row['multiplier_squared'], color=color, lw=1.8)
    for f in [1, 3, 5]:
        ax.axvline(f, color='#D2D6DA', lw=.65, ls=(0, (2, 3)), zorder=0)
    ax.set(title='A  Gamma attenuates frequencies', xlabel=r'Frequency $\omega/(2\pi)$',
           ylabel=r'Squared multiplier $M_\gamma(\omega)^2$', xlim=(0, 8), ylim=(1e-6, 1.4))
    ax.set_xticks([0, 2, 4, 6, 8]); ax.set_yticks([1e-6, 1e-4, 1e-2, 1])
    ax.text(.04, .05, 'Exact smoothing formula', transform=ax.transAxes, fontsize=8, color=gray)

    ax = axes[1]
    cutoffs = np.asarray(data['rate_cutoffs'])
    for row, color in zip(data['rows'], COLORS):
        lower, upper = np.asarray(row['positive_mass_lower']), np.asarray(row['positive_mass_upper'])
        ax.fill_between(cutoffs, np.maximum(lower, 1e-8), upper,
                        where=upper >= 1e-8, color=color, alpha=.22, linewidth=0)
        values = np.asarray(row['reference_positive_mass'])
        ax.plot(cutoffs, np.where(values >= 1e-8, values, np.nan), color=color, lw=1.5)
        ax.scatter(data['probe_cutoff'], row['probe_mass_reference'], color=color, s=14, zorder=4)
    ax.axvline(data['probe_cutoff'], color=gray, lw=.7, ls=(0, (3, 3)), zorder=0)
    ax.set(title='B  Energy in positive slow modes', xlabel=r'Upper rate cutoff $t$',
           ylabel=r'Target energy in $(10^{-9},t]$', xscale='log', yscale='log',
           xlim=(1e-8, .02), ylim=(1e-8, .8))
    ax.set_xticks([1e-8, 1e-6, 1e-4, 1e-2]); ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2])
    ax.legend(handles=[Line2D([], [], color=ink, label='Original tanh spectrum'),
                       Patch(facecolor=gray, alpha=.3, label='Corrected-model bounds')],
              loc='lower right', frameon=False, fontsize=7.5, handlelength=1.6)
    ax.text(.03, .97, 'Zero modes excluded',
            transform=ax.transAxes, va='top', fontsize=8)

    ax = axes[2]
    for row, color in zip(data['rows'], COLORS):
        ax.plot(row['steps'], 100*np.asarray(row['predicted_residual']), color=color, lw=1.6)
        measured = [row['checkpoints'][i] for i in row['marker_indices']]
        ax.scatter([c['step'] for c in measured], [100*c['residual'] for c in measured],
                   s=14, facecolors='white', edgecolors=color, linewidths=.8, zorder=4)
        ax.scatter(row['executed_hit'], 1, marker='D', s=25, color=color,
                   edgecolors='white', linewidths=.5, zorder=5)
        ax.vlines(row['predicted_hit'], .12, 1, color=color, lw=.7, ls=(0, (2, 2)))
        ax.hlines(1, row['timing_interval']['necessary'], row['timing_interval']['sufficient'],
                  color=color, lw=3, zorder=3)
    ax.axhline(1, color=gray, lw=.8, ls=(0, (4, 3)), zorder=0)
    ax.text(1000, .76, '1% residual', fontsize=8, color=gray)
    for row, color, align in [(data['rows'][0], COLORS[0], 'right'),
                               (data['rows'][-1], COLORS[-1], 'left')]:
        ax.text(row['executed_hit'], .14, f"{row['executed_hit']:,}",
                color=color, ha=align, fontsize=8)
    ax.set(title='C  Predicted and measured GD', xlabel='Gradient updates',
           ylabel='Relative residual (%)', xscale='log', yscale='log',
           xlim=(800, 3e7), ylim=(.1, 100))
    ax.set_xticks([1e3, 1e5, 1e7]); ax.set_yticks([.1, 1, 10, 100], ['0.1', '1', '10', '100'])
    ax.legend(handles=[Line2D([], [], color=ink, label='Corrected-model prediction'),
                       Line2D([], [], color=ink, lw=0, marker='o', ms=4,
                              markerfacecolor='white', label='Executed GD checkpoints')],
              loc='upper right', frameon=False, fontsize=7.5, handlelength=1.6)
    for ax in axes:
        ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(length=3, pad=3)
    fig.savefig(output/'gamma_access_three_panel.png', dpi=300)
    fig.savefig(output/'gamma_access_three_panel.pdf', metadata={'CreationDate':None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=DEFAULT.parent/'gamma_access_note')
    args = parser.parse_args()
    data = collect(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    plot(data, args.output)
    (args.output/'gamma_access_data.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(sources=len(data['source_sha256']),
        checkpoints=sum(len(r['checkpoints']) for r in data['rows']),
        largest_checkpoint_gap=max(r['maximum_checkpoint_gap'] for r in data['rows']),
        gamma8_positive_mass_lower=data['rows'][0]['probe_mass_lower'],
        gamma8_necessary_updates=data['rows'][0]['probe_necessary_updates'])))


if __name__ == '__main__':
    main()
