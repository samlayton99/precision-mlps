"""Compare theorem lower-error bounds with executed GD, including censoring."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LAMBDAS = [.03125, .0625, .125, .25]
TOLERANCES = [.1, .03, .01, .003, .001]


def bounded_crossing(value, tolerance, horizon):
    """First integer crossing of a monotone prediction, or None within horizon."""
    if value(0) <= tolerance:
        return 0
    if value(horizon) > tolerance:
        return None
    low, high = 0, horizon
    while high-low > 1:
        mid = (low+high)//2
        if value(mid) <= tolerance:
            high = mid
        else:
            low = mid
    return high


def executed_crossings(errors, tolerances, horizon, block=100_000):
    """Inspect every raw update, including spikes; no monotonicity assumed."""
    hits = [None]*len(tolerances)
    for begin in range(0, horizon+1, block):
        chunk = np.asarray(errors[begin:min(begin+block, horizon+1)])
        for i, tolerance in enumerate(tolerances):
            if hits[i] is None:
                locations = np.flatnonzero(chunk <= tolerance)
                if len(locations):
                    hits[i] = int(begin+locations[0])
    return hits


def theorem_value(arrays):
    weights = arrays['actual_target_weights']*arrays['resolved']
    rates = .5*arrays['rho_upper'][:len(weights)]
    logs = np.log1p(-rates)
    return lambda n: float(np.sqrt(np.exp(2*n*logs)@weights))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bounds', type=Path, required=True)
    p.add_argument('--gd', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.bounds/'summary.json').read_text())
    run = json.loads((args.gd/'summary.json').read_text())
    horizon = int(run['steps'])
    if int(summary['horizon']) != horizon:
        raise ValueError('Bound and executed horizons must match')
    raw = np.load(args.gd/'raw_error.npy', mmap_mode='r')
    if raw.shape != (horizon+1, 4) or run['lambdas'] != LAMBDAS:
        raise ValueError('Unexpected executed GD shape or bandwidth order')
    args.output.mkdir(parents=True, exist_ok=True)
    entries, plot_data = [], []
    for k, lam in enumerate(LAMBDAS):
        path = args.bounds/f'lambda{lam:g}_q16_p24.npz'
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        case = next(c for c in summary['cases'] if c['relative_bandwidth'] == lam)
        if case['refined']['input_sha256'] != run['input_sha256']:
            raise ValueError('Bounds and executed GD use different inputs')
        steps = arrays['steps'].astype(np.int64)
        actual = np.asarray(raw[steps, k])
        lower = arrays['lower_error']
        ratio = np.divide(lower, actual, out=np.full_like(lower, np.nan), where=actual>0)
        value = theorem_value(arrays)
        np.testing.assert_allclose([value(int(n)) for n in steps], lower, rtol=2e-12, atol=2e-14)
        predicted_hits = [bounded_crossing(value, tol, horizon) for tol in TOLERANCES]
        actual_hits = executed_crossings(raw[:, k], TOLERANCES, horizon)
        crossings = [dict(tolerance=tol, theorem_necessary_hit=pred, theorem_censored=pred is None,
                          executed_hit=act, executed_censored=act is None,
                          necessary_over_executed=pred/act if pred is not None and act not in (None, 0) else None)
                     for tol, pred, act in zip(TOLERANCES, predicted_hits, actual_hits)]
        finite_ratio = ratio[(steps>0)&np.isfinite(ratio)]
        entries.append(dict(relative_bandwidth=lam, gamma=case['gamma'], horizon=horizon,
            prediction_grid_count=len(steps), prediction_grid='sorted union of 0, 2000 geomspace(1,H) integer points, and 2001 linspace(0,H) integer points',
            lower_over_executed_on_positive_saved_steps=dict(minimum=float(np.min(finite_ratio)), median=float(np.median(finite_ratio)), maximum=float(np.max(finite_ratio)), final=float(ratio[-1]) if np.isfinite(ratio[-1]) else None),
            unresolved_target_energy=case['refined']['unresolved_target_energy'],
            every_update_max_lower_violation=case.get('executed_max_lower_violation'),
            every_update_max_absolute_reference_discrepancy=case.get('executed_max_absolute_reference_discrepancy'),
            every_update_checks_available='executed_max_lower_violation' in case,
            saved_grid_max_lower_violation=float(max(0., np.max(lower-actual))),
            crossings=crossings, arrays_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        np.savez_compressed(args.output/f'lambda{lam:g}_trajectory.npz', steps=steps,
                            lower_error=lower, executed_error=actual, lower_over_executed=ratio)
        plot_data.append((steps, ratio, predicted_hits, actual_hits))
    report = dict(horizon=horizon, tolerances=TOLERANCES, cases=entries,
        interpretation='Theorem rates use gamma-dependent upper endpoints; weights are actual finite-kernel target projections. Unresolved energy is omitted from the lower bound. Actual crossings inspect every executed update. None denotes no crossing within the horizon, not a spectral forecast.',
        numerical_status=summary['numerical_status'],
        grid_statistic_warning='Unweighted ratio summary describes the saved mixed logarithmic/linear grid, not a uniform-in-time average.',
        raw_error_sha256=hashlib.sha256((args.gd/'raw_error.npy').read_bytes()).hexdigest())
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.65), layout='constrained')
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
    for lam, color, (steps, ratio, pred, act) in zip(LAMBDAS, colors, plot_data):
        mask = steps>0
        axes[0].plot(steps[mask], ratio[mask], color=color, lw=1.25, label=rf'$\lambda=1/{round(1/lam)}$')
        axes[1].plot(TOLERANCES, [np.nan if h is None else h for h in act], color=color, marker='o', ms=3, lw=1.1)
        axes[1].plot(TOLERANCES, [np.nan if h is None else h for h in pred], color=color, marker='s', ms=3, mfc='white', lw=1, ls='--')
        for tol, ph, ah in zip(TOLERANCES, pred, act):
            if ph is None:
                axes[1].scatter(tol, horizon, marker='^', s=26, facecolors='none', edgecolors=color, zorder=4)
            if ah is None:
                axes[1].scatter(tol, horizon*1.07, marker='^', s=15, color=color, zorder=4)
    axes[0].set(xscale='log', xlabel='GD updates', ylabel='Lower bound / executed error', title='A  Trajectory tightness')
    axes[0].axhline(1, color='.45', lw=.8, ls=':')
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=7, handlelength=1.6, labelspacing=.25)
    axes[1].set(xscale='log', yscale='log', xlabel='Relative output-error tolerance', ylabel='First crossing (updates)', title='B  Tolerance crossings')
    axes[1].invert_xaxis()
    axes[1].axhline(horizon, color='.6', lw=.8, ls=':')
    axes[1].set_ylim(top=horizon*2)
    handles = [Line2D([], [], color='.25', marker='o', ms=3, label='Executed (filled)'),
               Line2D([], [], color='.25', ls='--', marker='s', mfc='white', ms=3, label='Bound (open)'),
               Line2D([], [], color='.25', lw=0, marker='^', ms=4, label='Beyond horizon')]
    axes[1].legend(handles=handles, frameon=False, fontsize=7, loc='lower right',
                   handlelength=1.6, labelspacing=.25)
    for ax in axes:
        ax.grid(alpha=.15, which='major')
    fig.savefig(args.output/'bound_diagnostics.pdf')
    fig.savefig(args.output/'bound_diagnostics.png', dpi=300)
    plt.close(fig)
    print(json.dumps({'output': str(args.output), 'horizon': horizon, 'cases': len(entries)}), flush=True)


if __name__ == '__main__':
    main()
