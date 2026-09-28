"""Curate cap-campaign measurements and figures; never generate report prose."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import core, cap_campaign as campaign


def read(path):
    return json.loads(path.read_text())


def collect(root):
    cases = []
    for path in sorted((root/'cases').glob('*/meta.json')):
        meta = read(path)
        row = {k:meta[k] for k in ['id', 'n', 'cap', 'family', 'seed', 'eta',
            'matrix_hash', 'target_hash', 'source_commit']}
        row['screen_hits'] = meta['screen_hits']
        for name in ['training', 'reference']:
            if (path.parent/f'{name}.json').exists():
                row[name] = read(path.parent/f'{name}.json')
        cases.append(row)
    certificates = []
    for path in sorted((root/'certificates').glob('*/result.json')):
        result = read(path)
        row = {k:result[k] for k in ['n', 'cap', 'target', 'rank', 'max_intervals',
            'source_commit', 'grid_hash', 'centers_hash', 'target_hash']}
        row.update(id=path.parent.name, cdf_bounds={k:v['bound'] for k,v in result['bounds'].items()},
            cdf=result['bounds']['0.01'], witnesses=[{k:c.get(k) for k in
            ['label', 'beta', 'delta', 'status', 'bias_repair', 'seconds', 'target_witness']}
            for c in result['certificates']])
        if (path.parent/'resolvent_refinement.json').exists():
            row['resolvent'] = read(path.parent/'resolvent_refinement.json')
        row['bound'] = max(row['cdf_bounds']['0.01'], row.get('resolvent', {}).get('best', {}).get('bound', 0))
        certificates.append(row)
    bounds = []
    violations = []
    for n in sorted(set(c['n'] for c in cases)):
        for cap in sorted(set(c['cap'] for c in cases if c['n'] == n)):
            available = [c for c in cases if c['n'] == n and c['cap'] <= cap and 'training' in c]
            for ti, target in enumerate(campaign.TARGETS):
                # Larger-cap proofs also apply to every smaller-cap dictionary.
                valid = [c for c in certificates if c['n'] == n and c['cap'] >= cap and c['target'] == target]
                winner = max(valid, key=lambda c:c['bound'], default=None)
                bound = winner['bound'] if winner else 0
                reached = [c for c in available if c['training']['hits'][ti][0] >= 0]
                fastest = min(reached, key=lambda c:c['training']['hits'][ti][0], default=None)
                upper = fastest['training']['hits'][ti][0] if fastest else None
                common = next((c for c in available if c['cap'] == cap and c['family'] == 'common'), None)
                row = dict(n=n, cap=cap, target=target, bound=bound,
                    certificate=winner['id'] if winner else None, executed_upper=upper,
                    upper_case=fastest['id'] if fastest else None,
                    upper_reference=fastest.get('reference') if fastest else None,
                    ratio=upper/bound if upper is not None and bound else None,
                    common_hit=common['training']['hits'][ti][0] if common else None,
                    common_steps=common['training']['steps'] if common else None)
                bounds.append(row)
    by_id = {c['id']:c for c in cases}
    development_path = root/'development_cases.json'
    development_hashes = {by_id[name]['matrix_hash'] for name in read(development_path)
                          if name in by_id} if development_path.exists() else set()
    # Audit each actual dictionary once, at every tolerance, including held-out
    # seeds. A censored trajectory cannot contradict a necessary learning time.
    certificate_checks = []
    for case in cases:
        if 'training' not in case:
            continue
        for ti, target in enumerate(campaign.TARGETS):
            valid = [c for c in certificates if c['n'] == case['n']
                     and c['cap'] >= case['cap'] and c['target'] == target]
            for ei, epsilon in enumerate(campaign.EPSILONS):
                def value(c):
                    return c['bound'] if ei == 0 else c['cdf_bounds'][str(epsilon)]
                winner = max(valid, key=value, default=None)
                if winner is None:
                    continue
                hit = case['training']['hits'][ti][ei]
                row = dict(case=case['id'], target=target, epsilon=epsilon,
                    certificate=winner['id'], bound=value(winner), hit=hit,
                    steps=case['training']['steps'],
                    held_out=100 <= case['seed'] <= 104 and case['matrix_hash'] not in development_hashes,
                    violation=hit >= 0 and hit < value(winner))
                certificate_checks.append(row)
                if row['violation']:
                    violations.append(row)
    agreements = []
    for c in cases:
        if 'training' not in c or 'reference' not in c:
            continue
        for ti, target in enumerate(campaign.TARGETS):
            for ei, epsilon in enumerate(campaign.EPSILONS):
                actual = c['training']['hits'][ti][ei]
                predicted = c['reference']['hits'][str(epsilon)][ti]
                agreements.append(dict(case=c['id'], target=target, epsilon=epsilon,
                    actual=actual, predicted=predicted,
                    difference=actual-predicted if actual >= 0 and predicted is not None else None,
                    censored=actual < 0, steps=c['training']['steps']))
    coverage = {}
    for phase in ['development', 'confirmation']:
        manifest = root/f'{phase}_cases.json'
        if manifest.exists():
            names = read(manifest)
            incomplete = [name for name in names
                if by_id.get(name, {}).get('training', {}).get('steps', 0) < 200000]
            coverage[phase] = dict(expected=len(names), completed=len(names)-len(incomplete),
                                   incomplete=incomplete)
            if phase == 'confirmation':
                coverage[phase]['new_dictionaries_vs_development'] = len({by_id[name]['matrix_hash']
                    for name in names if name in by_id and by_id[name]['matrix_hash'] not in development_hashes})
    return dict(cases=cases, certificates=certificates, bounds=bounds, coverage=coverage,
                certificate_checks=certificate_checks, bound_violations=violations,
                forecast_checks=agreements)


def style(ax, xlabel, ylabel, *, logx=True, logy=True):
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)


def save(fig, output, name):
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(output/f'{name}.{extension}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def exact_controls(output):
    from . import cap_certificate as certificate
    rows = []
    x = np.array([-1., 1.]); y = x/np.sqrt(2)
    for cap in [.05, .1, .2]:
        beta = 4*np.tanh(cap)**2
        proof = certificate.certify(x, np.zeros(4), cap, y,
            np.ones((2, 1))*np.sqrt(beta/2), y, target_witness=True)
        bound = certificate.time_bound([proof])['bound']
        j = core.design(x, np.zeros(4), np.full(4, cap))
        eta = .5/np.linalg.norm(j, 2)**2
        theta = np.zeros(5)
        for step in range(10000):
            residual = j@theta-y
            if np.linalg.norm(residual) <= .01:
                break
            theta -= eta*j.T@residual
        else:
            raise AssertionError('Exact two-sample control did not reach tolerance')
        assert step == bound
        rows.append(dict(cap=cap, beta=beta, certified_beta=proof['beta'],
                         bound=bound, executed_hit=step, eta=eta, samples=x.tolist(), width=4))
    core.write_json(output/'two_sample_controls.json', rows)


def time_panel(ax, rows):
    x = [r['cap'] for r in rows]
    lower = [r['bound'] or np.nan for r in rows]
    upper = [r['executed_upper'] or np.nan for r in rows]
    common = [r['common_hit'] if r['common_hit'] is not None and r['common_hit'] >= 0 else np.nan for r in rows]
    ax.plot(x, lower, 's-', color='#438c67', label='Uniform necessary updates')
    ax.plot(x, upper, 'o-', color='#245c9f', label='Fastest executed dictionary')
    ax.plot(x, common, 'x--', color='#d65f28', label='Common-slope executed hit')
    censored_label = 'Common slope: not reached'
    for r in rows:
        if r['common_hit'] == -1:
            ax.scatter(r['cap'], r['common_steps'], marker='^', facecolors='none', edgecolors='#d65f28', label=censored_label)
            censored_label = None
    style(ax, 'Slope cap Γ', 'Updates to 1% training error')
    ax.legend(fontsize=7)


def cdf_envelope(certificates, n, cap, target):
    """Combine compatible guarantees as step functions, preserving cap nesting."""
    valid = [c['cdf'] for c in certificates if c['n'] == n and c['cap'] >= cap
             and c['target'] == target and c['cdf']['thresholds']]
    if not valid:
        return np.array([]), np.array([])
    thresholds = np.unique(np.concatenate([c['thresholds'] for c in valid]))
    mass = np.zeros_like(thresholds)
    for c in valid:
        index = np.searchsorted(c['thresholds'], thresholds, side='right')-1
        available = index >= 0
        mass[available] = np.maximum(mass[available], np.asarray(c['mass'])[index[available]])
    return thresholds, np.maximum.accumulate(mass)


def plot(summary, output):
    rows = [r for r in summary['bounds'] if r['n'] == 512 and r['target'] == campaign.TARGETS[0]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout='constrained')
    for cap in [4, 8, 16, 64]:
        threshold, mass = cdf_envelope(summary['certificates'], 512, cap, campaign.TARGETS[0])
        keep = (mass > 0)&(threshold < 1)
        axes[0].step(threshold[keep], mass[keep], where='post', label=f'Γ = {cap}')
    axes[0].axhline(.01**2, ls=':', color='#555555', label='1% error squared')
    style(axes[0], 'Normalized curvature threshold s', 'Guaranteed target energy F(s)')
    axes[0].set_ylim(1e-6, 1); axes[0].legend(fontsize=7)
    axes[0].set_title('(a) Target energy forced into slow modes')
    time_panel(axes[1], rows)
    axes[1].set_title('(b) Uniform bound and executed learning time')
    valid = [r for r in rows if r['ratio'] is not None]
    axes[2].plot([r['cap'] for r in valid], [r['ratio'] for r in valid], 'o-', color='#245c9f')
    axes[2].axhline(2, ls='--', color='#438c67', label='Factor-two objective')
    style(axes[2], 'Slope cap Γ', 'Executed upper witness / uniform bound')
    axes[2].set_ylim(bottom=1); axes[2].legend(fontsize=8)
    axes[2].set_title('(c) Remaining tightness gap')
    fig.suptitle('Uniform capped-gamma theorem · raw readout · N=512 · sine mixture')
    save(fig, output, 'capped_kernel_three_panel')


def banner(summary, archive, output):
    from . import full_analyze as old
    cfg = read(archive/'manifest.json')['config']
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), layout='constrained')
    widths, slopes, initial = [], [], []
    for n in cfg['widths']:
        data = read(archive/'joint'/f'N{n}'/'evaluations.json')
        assert data[-1]['step'] == cfg['joint_steps'] and not any(data[-1]['failed'])
        widths.append(data[-1]['width'])
        slopes.append(np.array(data[-1]['slope_quantiles'])[:, 1])
        initial.append(np.array(data[0]['slope_quantiles'])[:, 1])
    slopes, initial = np.array(slopes), np.array(initial)
    axes[0].plot(widths, np.median(slopes, axis=1), 'o-', color=old.COLORS[0], label='Trained median |a|')
    axes[0].fill_between(widths, slopes.min(axis=1), slopes.max(axis=1), color=old.COLORS[0], alpha=.16, label='Range across 5 seeds')
    axes[0].plot(widths, np.median(initial, axis=1), ':', color='#777777', label='Xavier initial median')
    axes[0].plot(widths, np.array(cfg['widths'])/8, '--', color='#222222', label='Reference γ = N/8')
    axes[0].set_yscale('log'); old.finish_axis(axes[0], 'Hidden width W', 'Physical slope magnitude')
    axes[0].set_title('(a) End-to-end Adam, 20k updates'); axes[0].legend(fontsize=7.5)
    for optimizer, color, label in [('gd', old.COLORS[0], 'GD, η = 0.5/L'), ('adam', old.COLORS[1], 'Validation-selected Adam')]:
        axes[1].plot(cfg['gammas'], old.precision_value(archive, cfg['n'], 'raw', optimizer), 'o-', color=color, label=label)
    axes[1].axvline(cfg['n']/8, color='#555555', ls=':', lw=.8, label='Reference γ = N/8')
    axes[1].set_yscale('log'); old.finish_axis(axes[1], ylabel='Independent relative L2 error')
    axes[1].set_title('(b) Raw, 200k updates'); axes[1].legend(fontsize=7.5)
    rows = [r for r in summary['bounds'] if r['n'] == 512 and r['target'] == campaign.TARGETS[0]]
    time_panel(axes[2], rows); axes[2].set_title('(c) Cap guarantee and executed upper witness')
    fig.suptitle('Bounded slopes and readout optimization · sine mixture · N=512', fontsize=12)
    save(fig, output, 'banner_capped_kernel')


def validation_figures(root, archive, output):
    from . import fourier_law as law
    periodic = read(root/'fourier_validation/periodic_checks.json')
    finite = read(root/'fourier_validation/finite_checks.json')
    colors = ['#245c9f', '#d65f28', '#438c67']
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), layout='constrained')
    for gamma, color in zip([4, 16, 64], colors):
        spectrum, lo, hi = law.continuous_spectrum(128, gamma, 64)
        frequency = np.arange(1, 65)
        axes[0].plot(frequency, spectrum[1:65]*32, color=color, label=f'γ = {gamma}')
        axes[0].fill_between(frequency, lo[1:65], hi[1:65], color=color, alpha=.2)
        p = next(p for p in periodic if p['n'] == 128 and p['gamma'] == gamma and p['density'] == 16 and p['offset'] == .5)
        checkpoints = [r for r in p['checkpoints'] if r['step'] > 0]
        axes[1].plot([r['step'] for r in checkpoints], [r['prediction'][3] for r in checkpoints], color=color, label=f'γ = {gamma}: forecast')
        axes[1].scatter([r['step'] for r in checkpoints], [r['actual'][3] for r in checkpoints], facecolors='none', edgecolors=color, s=28)
    style(axes[0], 'Periodic Fourier index', 'Curvature / largest curvature', logx=False)
    axes[0].set_ylim(1e-20, 2); axes[0].legend(fontsize=8)
    axes[0].set_title('(a) Exact continuous spectrum and multiplier bracket')
    style(axes[1], 'Ordinary GD updates', 'Sine-mixture relative training error')
    axes[1].scatter([], [], facecolors='none', edgecolors='#555555', label='Executed checkpoints')
    axes[1].set_ylim(1e-16, 2); axes[1].legend(fontsize=8)
    axes[1].set_title('(b) Sampled packets: forecast and executed checkpoints')
    save(fig, output, 'periodic_validation')

    checks = []
    for row in finite:
        folder = archive/'training'/f'N{row["n"]}_{row["map"]}_gd'
        if not (folder/'hitting_audit.npz').exists():
            continue
        config = read(folder/'case.json')
        if row['gamma'] not in config['gammas']:
            continue
        index = config['gammas'].index(row['gamma'])
        actual = np.load(folder/'hitting_audit.npz')['first'][index]
        for column, specification in enumerate(config['columns']):
            ti, target = specification['target_index'], specification['target']
            for ei, epsilon in enumerate(campaign.EPSILONS):
                predicted = row['gram_hits'][str(epsilon)][ti]
                hit = int(actual[column, ei])
                checks.append(dict(n=row['n'], gamma=row['gamma'], map=row['map'], target=target,
                    epsilon=epsilon, actual=hit, predicted=predicted,
                    log10_hit_ratio=float(np.log10(predicted/hit)) if hit > 0 and predicted else None))
    reached = [c for c in checks if c['actual'] > 0 and c['predicted'] is not None]
    speedups = []
    for row in reached:
        if row['gamma'] == 64:
            continue
        anchor = next((r for r in reached if r['gamma'] == 64 and all(r[k] == row[k]
                        for k in ['n', 'map', 'target', 'epsilon'])), None)
        if anchor:
            speedups.append(dict(n=row['n'], gamma=row['gamma'], map=row['map'], target=row['target'],
                epsilon=row['epsilon'], actual=row['actual']/anchor['actual'],
                predicted=row['predicted']/anchor['predicted']))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1), layout='constrained')
    for epsilon, color in [(1e-2, colors[0]), (1e-4, colors[1]), (1e-6, colors[2])]:
        selected = [r for r in reached if r['epsilon'] == epsilon]
        axes[0].scatter([r['actual'] for r in selected], [r['predicted'] for r in selected], s=15, alpha=.6, color=color, label=f'ε = {epsilon:g}')
    axes[0].plot([1, 2e5], [1, 2e5], '--', color='#555555')
    style(axes[0], 'Executed first-hit update', 'Finite-Gram predicted update')
    axes[0].legend(fontsize=8); axes[0].set_title('(a) Resolved reached cases; unchanged clocks')
    selected = [r for r in speedups if r['epsilon'] >= 1e-6]
    axes[1].scatter([r['actual'] for r in selected], [r['predicted'] for r in selected], s=16, alpha=.6, color=colors[0])
    if selected:
        low = min(min(r['actual'], r['predicted']) for r in selected)
        high = max(max(r['actual'], r['predicted']) for r in selected)
        axes[1].plot([low, high], [low, high], '--', color='#555555')
    style(axes[1], 'Executed n(γ) / n(64)', 'Predicted n(γ) / n(64)')
    axes[1].set_title('(b) Matched gamma-dependent learning delays')
    for gamma, color in zip([4, 16, 64], colors):
        row = next(r for r in finite if r['n'] == 512 and r['map'] == 'raw' and r['gamma'] == gamma)
        data = [r for r in row['comparisons'] if r['step'] > 0]
        axes[2].plot([r['step'] for r in data], [r['gram'][0] for r in data], color=color, label=f'γ = {gamma}')
        axes[2].scatter([r['step'] for r in data], [r['actual'][0] for r in data], s=15, facecolors='none', edgecolors=color)
    style(axes[2], 'Ordinary GD updates', 'Raw sine-mixture training error')
    axes[2].legend(fontsize=8); axes[2].set_title('(c) Finite-Gram curves and executed checkpoints')
    save(fig, output, 'finite_kernel_validation')
    core.write_json(output/'validation_summary.json', dict(periodic_cases=len(periodic),
        periodic_max_curve_difference=max(r['max_curve_absolute_difference'] for r in periodic),
        finite_dictionaries=len(finite), finite_max_gram_difference=max(r['gram_relative_difference'] for r in finite),
        hit_checks=checks, gamma_speedups=speedups,
        periodic_aliases=64, periodic_spatial_images=8,
        largest_spatial_tail=max(float(law.spatial_tail(r['n'], r['gamma'], 8)) for r in periodic),
        largest_principal_energy_alias_tail=max(float(law.alias_tail(np.pi, 2*r['gamma']/r['n'], 64)) for r in periodic)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = collect(args.root)
    core.write_json(args.output/'campaign_summary.json', summary)
    exact_controls(args.output)
    plot(summary, args.output)
    if args.archive:
        banner(summary, args.archive, args.output)
        if (args.root/'fourier_validation/finite_checks.json').exists():
            validation_figures(args.root, args.archive, args.output)
    print(json.dumps(dict(cases=len(summary['cases']), certificates=len(summary['certificates']),
                          bound_violations=len(summary['bound_violations']))))


if __name__ == '__main__':
    main()
