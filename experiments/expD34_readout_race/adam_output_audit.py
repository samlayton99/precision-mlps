"""Archive-only output/residual audit using the actual next Adam preconditioner.

Spectra describe instantaneous sensitivity, not a frozen-trajectory forecast.
All output vectors/Jacobians use RMS normalization; squared residual norm is MSE.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import svd


def field(p, x, y):
    a, b, c = p[:-1].reshape(3, -1)
    u = x[:, None] * a + b
    h = np.tanh(u)
    e = np.exp(-2 * abs(u))
    sech2 = 4 * e / (1 + e)**2
    scale = np.sqrt(len(x))
    residual = (h @ c + p[-1] - y) / scale
    jacobian = np.c_[x[:, None] * sech2 * c, sech2 * c, h, np.ones(len(x))] / scale
    return residual, jacobian


def next_update(g, m, v, count, case):
    b1, b2 = case['beta1'], case['beta2']
    mh = (b1 * m + (1-b1) * g) / (1-b1**(count+1))
    vh = (b2 * v + (1-b2) * g*g) / (1-b2**(count+1))
    P = 1 / (np.sqrt(vh) + case['epsilon']) if case['adaptive'] else np.ones_like(g)
    return -case['eta'] * P * mh, P, mh


def spectrum(J, r, eta):
    """Keep residual energy at unresolved singular values together with null energy."""
    u, s, _ = svd(J, full_matrices=False, check_finite=True, lapack_driver='gesdd')
    weight = (u.T @ r)**2
    total = float(r @ r)
    null = max(0., total - float(weight.sum()))
    floor = np.finfo(float).eps * max(J.shape) * s[0] if len(s) else 0.
    resolved = s > floor
    unresolved = null + float(weight[~resolved].sum())
    rates = eta*s*s
    rows = [dict(index=i, singular_value=float(si), eta_sigma_squared=float(ri),
                 residual_energy=float(wi), resolved=bool(ok))
            for i, (si, ri, wi, ok) in enumerate(zip(s, rates, weight, resolved))]
    rows.append(dict(index=-1, singular_value=0., eta_sigma_squared=0.,
                     residual_energy=null, resolved=False))
    summary = dict(residual_energy=total, numerical_rank=int(resolved.sum()),
                   sigma_max=float(s[0]), singular_value_resolution_floor=float(floor),
                   resolved_condition=float(s[0]/s[resolved][-1]) if resolved.any() else np.inf,
                   unresolved_residual_fraction=unresolved/total if total else 0.,
                   weighted_rate=float(weight @ rates)/total if total else 0.,
                   spectral_gradient_energy=float(weight @ (s*s)),
                   direct_gradient_energy=float(np.linalg.norm(J.T @ r)**2),
                   energy_reconstruction_error=abs(float(weight.sum())+null-total))
    # These are instantaneous frozen-eigenvalue labels, NOT predicted hitting times.
    for budget in (20000, 100000, 600000):
        summary[f'energy_fraction_eta_lambda_below_1_over_{budget}'] = (
            (null + float(weight[(rates < 1/budget) | ~resolved].sum()))/total if total else 0.)
    return summary, rows


def audit_state(p, m, v, count, case, x, y):
    r, J = field(p, x, y)
    g = J.T @ r
    delta, P, mh = next_update(g, m, v, count, case)
    linear = J @ delta
    rn, _ = field(p+delta, x, y)
    nonlinear = rn-r-linear
    gradient_linear = -case['eta'] * J @ (P*g)
    lag_linear = -case['eta'] * J @ (P*(mh-g))
    linear_term = float(r @ linear)
    quadratic = float(.5 * linear @ linear)
    remainder = float((r+linear) @ nonlinear + .5 * nonlinear @ nonlinear)
    actual = float(.5 * (rn @ rn-r @ r))
    width = (len(p)-1)//3
    q = np.linalg.qr(np.c_[np.ones(len(x)), x])[0]
    jc = q.T @ J
    jh = J - q @ jc
    eh = r-q @ (q.T @ r)
    C = (jc * P) @ jc.T
    ev = np.linalg.eigvalsh(C)
    resolved = ev[0] > 64*np.finfo(float).eps*max(1., ev[-1])
    result = dict(relative_l2=np.linalg.norm(r)/np.sqrt(np.mean(y*y)),
                  fine_relative_l2=np.linalg.norm(eh)/np.sqrt(np.mean(y*y)),
                  next_relative_l2=np.linalg.norm(rn)/np.sqrt(np.mean(y*y)),
                  mean_gamma=float(np.mean(abs(p[:width]))),
                  max_gamma=float(np.max(abs(p[:width]))),
                  linear_loss_change=linear_term, quadratic_loss_change=quadratic,
                  nonlinear_loss_change=remainder, actual_loss_change=actual,
                  loss_identity_error=abs(actual-linear_term-quadratic-remainder),
                  output_linear_norm=float(np.linalg.norm(linear)),
                  output_nonlinear_norm=float(np.linalg.norm(nonlinear)),
                  current_gradient_linear_loss_change=float(r @ gradient_linear),
                  momentum_lag_linear_loss_change=float(r @ lag_linear),
                  output_step_identity_error=float(np.linalg.norm(linear-gradient_linear-lag_linear)),
                  inverse_min=float(P.min()), inverse_median=float(np.median(P)),
                  inverse_max=float(P.max()), adaptive_coarse_resolved=bool(resolved),
                  coarse_eigen_min=float(ev[0]), coarse_eigen_max=float(ev[-1]))
    if resolved:
        fine_gradient = jh.T @ eh
        z = q.T @ r + np.linalg.solve(C, jc @ (P*fine_gradient))
        tracking = jc.T @ z
        effective = g-tracking
        result.update(tracking_fine_output_norm=float(np.linalg.norm(jh @ (P*tracking))),
                      effective_fine_output_norm=float(np.linalg.norm(jh @ (P*effective))),
                      tracking_fine_loss_rate=float(eh @ (jh @ (P*tracking))),
                      effective_fine_loss_rate=float(eh @ (jh @ (P*effective))),
                      effective_coarse_defect=float(np.linalg.norm(jc @ (P*effective))))
    spectra = []
    summaries = []
    for block, columns in [('readout', slice(2*width, None)), ('joint', slice(None))]:
        for metric in ('raw', 'adaptive'):
            matrix = J[:, columns] * (np.sqrt(P[columns]) if metric == 'adaptive' else 1.)
            summary, modes = spectrum(matrix, r, case['eta'])
            summaries.append(dict(block=block, metric=metric, **summary))
            spectra.extend(dict(block=block, metric=metric, **row) for row in modes)
    return result, summaries, spectra


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze(source, output, selected_steps):
    # Import target definitions only at the archive boundary: kernels above need no JAX.
    from . import adam_forces as af, targets
    manifest = json.loads((source/'manifest.json').read_text())
    status = json.loads((source/'status.json').read_text())
    if not status['complete']:
        raise ValueError('Incomplete source archive')
    output.mkdir(parents=True, exist_ok=True)
    states, summaries, spectra, crossings = [], [], [], []
    with np.load(source/'snapshots.npz') as archive:
        steps = archive['steps']
        for i, case in enumerate(manifest['cases']):
            x, y, mapping, scale = af.data(case['target'], manifest['m'])
            xe = targets.grid(8192)
            ye = af.target_values(case['target'], xe, mapping)/scale
            identity = dict(bundle=source.name, case_index=i, **case)
            errors = []
            for k, step in enumerate(steps):
                if 'failed' in archive and archive['failed'][i, k] != 0:
                    errors.append(np.nan)
                    continue
                p = archive['p'][i, k]
                a, b, c = p[:-1].reshape(3, -1)
                residual = np.tanh(x[:, None]*a+b) @ c+p[-1]-y
                errors.append(float(np.linalg.norm(residual)/np.linalg.norm(y)))
                if int(step) not in selected_steps:
                    continue
                row, summary, modes = audit_state(p, archive['m'][i, k], archive['v'][i, k],
                                                 int(archive['count'][i, k]), case, x, y)
                re = np.tanh(xe[:, None]*a+b) @ c+p[-1]-ye
                row['relative_eval_l2'] = float(np.linalg.norm(re)/np.linalg.norm(ye))
                mark = dict(**identity, step=int(step), count=int(archive['count'][i, k]))
                states.append(dict(**mark, **row))
                summaries.extend(dict(**mark, **s) for s in summary)
                spectra.extend(dict(**mark, **s) for s in modes)
                print(json.dumps(dict(bundle=source.name, case=i, step=int(step))), flush=True)
            for tolerance in (.1, .01, .001, .0001, .000001):
                hit = np.flatnonzero(np.asarray(errors) <= tolerance)
                k = int(hit[0]) if len(hit) else None
                crossings.append(dict(**identity, tolerance=tolerance,
                    first_sampled_crossing=int(steps[k]) if k is not None else '',
                    previous_sample=int(steps[k-1]) if k is not None and k>0 else '',
                    final_step=int(steps[-1]), final_relative_l2=errors[-1],
                    convention='first observed crossing among archived raw training-error snapshots'))
    for name, rows in [('states.csv', states), ('spectral_summary.csv', summaries),
                       ('spectra.csv.gz', spectra), ('sampled_crossings.csv', crossings)]:
        write_csv(output/name, rows)
    (output/'audit.json').write_text(json.dumps(dict(source=str(source), rows=len(states),
        requested_steps=selected_steps, available_steps=steps.tolist(),
        state_role='virtual next update from archived state, including endpoint',
        source_hashes={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                       for name in ('manifest.json', 'snapshots.npz', 'status.json')},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')


def plots(source, output):
    """Combine bundle CSVs into figures only; interpretation is authored separately."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    states, summary = [], []
    for path in sorted(source.glob('*/states.csv')):
        with path.open() as handle:
            states.extend(csv.DictReader(handle))
    for path in sorted(source.glob('*/spectral_summary.csv')):
        with path.open() as handle:
            summary.extend(csv.DictReader(handle))
    if not states:
        raise ValueError('No bundle states.csv files under source')
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for ax, step in zip(axes, (20000, 100000, 600000)):
        selected = [r for r in states if int(r['step']) == step and r['bundle'].startswith('primary_')]
        for optimizer, color in [('gd', 'tab:blue'), ('adam', 'tab:orange')]:
            rows = [r for r in selected if r['optimizer'] == optimizer]
            ax.scatter([float(r['mean_gamma']) for r in rows],
                       [float(r['relative_eval_l2']) for r in rows], label=optimizer.upper(),
                       color=color, alpha=.7, s=24)
        ax.set(xscale='log', yscale='log', xlabel='Mean absolute slope',
               ylabel='Independent-grid relative L2 error', title=f'{step:,} updates')
        ax.axhline(.01, color='gray', ls=':', lw=1)
    axes[0].legend()
    fig.savefig(output/'adam_output_vs_scale.png', dpi=180)
    plt.close(fig)
    key = lambda r: (r['bundle'], r['case_index'], r['step'])
    lookup = {key(r): r for r in states}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, block in zip(axes, ('readout', 'joint')):
        for metric, color in [('raw', 'tab:blue'), ('adaptive', 'tab:orange')]:
            rows = [r for r in summary if r['optimizer']=='adam' and r['block']==block
                    and r['metric']==metric and r['bundle'].startswith('primary_')]
            ax.scatter([float(r['weighted_rate']) for r in rows],
                       [float(lookup[key(r)]['relative_l2']) for r in rows],
                       color=color, alpha=.5, s=18, label=metric)
        ax.set(xscale='log', yscale='log', xlabel='Residual-weighted instantaneous eta × eigenvalue',
               ylabel='Raw training relative L2 error', title=f'Adam: {block} sensitivity')
        ax.legend()
    fig.savefig(output/'adam_residual_weighted_sensitivity.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for bundle_prefix, label, color in [('primary_', 'eta = 0.002', 'tab:orange'),
                                        ('rates_0', 'eta = 0.0002', 'tab:green'),
                                        ('rates_1', 'eta = 0.001', 'tab:purple')]:
        rows = [r for r in states if r['optimizer']=='adam' and r['bundle'].startswith(bundle_prefix)]
        gradient = np.array([-float(r['current_gradient_linear_loss_change']) for r in rows])
        actual = np.array([-float(r['actual_loss_change']) for r in rows])
        lag = np.array([float(r['momentum_lag_linear_loss_change']) for r in rows])
        denominator = np.maximum(gradient, np.finfo(float).tiny)
        axes[0].scatter(gradient, actual, label=label, color=color, alpha=.55, s=19)
        axes[1].scatter([float(r['relative_l2']) for r in rows], lag/denominator,
                        label=label, color=color, alpha=.55, s=19)
    axes[0].set(xscale='log', yscale='symlog', xlabel='Current-gradient predicted loss reduction',
                ylabel='Actual next-step loss reduction', title='Positive = progress; negative = loss increase')
    axes[0].set_yscale('symlog', linthresh=1e-12)
    axes[1].set(xscale='log', yscale='symlog', xlabel='Raw training relative L2 error',
                ylabel='Momentum-lag loss term / gradient reduction', title='Above 1: lag cancels gradient descent')
    axes[1].axhline(1., color='gray', ls=':')
    axes[0].legend()
    fig.savefig(output/'adam_actual_output_progress.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', nargs='+', type=int, default=[20000, 100000, 600000])
    parser.add_argument('--plots', action='store_true', help='Combine CSVs under source/*/')
    args = parser.parse_args()
    if args.plots:
        plots(args.source, args.output)
    else:
        analyze(args.source, args.output, args.steps)


if __name__ == '__main__':
    main()
