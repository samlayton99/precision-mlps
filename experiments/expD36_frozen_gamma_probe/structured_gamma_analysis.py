"""CPU evidence for gamma-explicit small boundary solves; no training fits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.fft import dct

from . import core, finite_gamma_gram as reference
from . import structured_gamma as differences
from . import uniform_grid_spectrum as construction
from .review_figures import DEFAULT, TARGETS
from .uniform_grid_analysis import targets
from .structured_resolvent import signed_lowrank_solver, rational_filter_mass


CASES = [(512, 8), (512, 64)] + [(n, g) for n in [128, 256, 512]
    for g in [8, 12, 16, 64] if (n, g) not in [(512, 8), (512, 64)]]
CASES += [(512, 10), (512, 24), (512, 48), (128, 2), (256, 4), (256, 32)]
PROTOCOL = dict(order=64, margin=1.25, cutoff_min=1e-8, cutoff_max=.5,
    cutoff_count=97, signed_drop=1e-12, ridge_shifts=[1e-10, 1e-12, 1e-14],
    positive_band='(0,b], with capacity-witness upper bound subtracted for the nullspace',
    epsilon=.01, arithmetic_factors=[0, 1, 10],
    selection='Maximize necessary time over the fixed cutoff grid using only small resolvent solves and explicit witness residuals. No reference spectrum or observed GD enters filter selection.',
    fresh_step='0.5/(W+1); twelve existing dictionaries reuse saved steps.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inverse_bulk(values, a):
    result = np.asarray(values, dtype=complex).copy()
    indices = a['core_columns']
    n = len(indices)
    phase = np.exp(1j*np.pi*np.arange(n)/n)
    result[indices] = phase[:, None]*np.fft.ifft(result[indices], axis=0)*np.sqrt(n)
    return result


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def diagnostics(n, gamma, a, j):
    """Finite-difference identity, aliasing, and error after raw whitening."""
    geometry = differences.finite_geometry(n, 16, a['halo'], gamma)
    transform = geometry['transform']
    whitened = differences.whiten(geometry['z'], transform)
    metric_defect = np.linalg.norm(whitened-j, 'fro')/np.linalg.norm(j, 'fro')
    bandwidth = gamma*2/n
    theta = np.linspace(1e-4, np.pi-1e-4, 161)
    discrete = differences.polyphase_symbol(theta, bandwidth, 16, aliases=16, samples=len(j))
    leading = differences.polyphase_symbol(theta, bandwidth, 16, aliases=0, samples=len(j))
    valid = discrete['energy'] > 1e-250
    alias_relative = np.max(np.abs(discrete['energy'][valid]-leading['energy'][valid])
                            /discrete['energy'][valid])
    padding = int(np.ceil(18*16/bandwidth))
    toeplitz = differences.toeplitz_boundary(geometry, padding)
    local_error = toeplitz['approximation']-toeplitz['finite']
    gram_error = np.zeros((j.shape[1], j.shape[1]))
    gram_error[1:-1, 1:-1] = local_error
    left = solve_triangular(transform.T, gram_error, lower=False)
    whitened_error = solve_triangular(transform.T, left.T, lower=False).T
    return dict(raw_metric_reconstruction_relative_frobenius=float(metric_defect),
        bandwidth=bandwidth, symbol_theta=theta, discrete_symbol=discrete['energy'],
        leading_symbol=leading['energy'], alias_max_relative_on_test_grid=float(alias_relative),
        alias_max_relative_below_half_nyquist=float(np.max(
            np.abs(discrete['energy'][theta <= np.pi/2]-leading['energy'][theta <= np.pi/2])
            /discrete['energy'][theta <= np.pi/2])),
        alias_absolute_tail=discrete['energy_error_bound'], toeplitz_padding=padding,
        toeplitz_local_error_frobenius=float(np.linalg.norm(local_error, 'fro')),
        toeplitz_error_after_whitening_frobenius=float(np.linalg.norm(whitened_error, 'fro')),
        toeplitz_analytic_local_error=toeplitz['error_bound'],
        boundary_local_frobenius=float(np.linalg.norm(toeplitz['boundary'], 'fro')),
        bulk_local_frobenius=float(np.linalg.norm(toeplitz['toeplitz'], 'fro')),
        unwhitened_feature_norm_squared=float(np.linalg.norm(geometry['z'], 'fro')**2),
        raw_feature_norm_squared=float(np.linalg.norm(j, 'fro')**2))


def diagonal_benchmarks(j, y, eta):
    """Failure controls only: these are not the compact theorem predictions."""
    norm2 = np.sum(y*y, axis=0)
    cosine_j = dct(j, type=2, norm='ortho', axis=0)
    cosine_y = dct(y, type=2, norm='ortho', axis=0)
    output = {}
    for name, transformed_j, transformed_y in [
        ('full_DCT_II_diagonal', cosine_j, cosine_y),
        ('full_complex_DFT_diagonal', np.fft.fft(j, norm='ortho', axis=0),
         np.fft.fft(y, norm='ortho', axis=0))]:
        model = dict(rates=eta*np.sum(np.abs(transformed_j)**2, axis=1),
            weights=np.abs(transformed_y)**2/norm2, floor=np.zeros(y.shape[1]))
        output[name] = reference.first_hit(model)
    rows = cosine_j[:129]
    gram = rows@rows.T
    loading = cosine_y[:129]
    floor = np.maximum(0., 1-np.sum(loading**2, axis=0)/norm2)
    for block_width in [8, 16]:
        values, weights = [], []
        for start in range(0, len(rows), block_width):
            stop = min(len(rows), start+block_width)
            eigen, modes = np.linalg.eigh(gram[start:stop, start:stop])
            values.extend(np.maximum(eigen, 0.))
            weights.extend((modes.T@loading[start:stop])**2/norm2)
        model = dict(rates=eta*np.array(values), weights=np.array(weights), floor=floor)
        output[f'DCT_first129_block{block_width}'] = reference.first_hit(model)
    return dict(hits=output, DCT_129_projection_floor=floor,
        role='Validation-only diagonal/block approximations using the original tanh geometry; not bounds or fitted models.')


def spectral_brackets(d, q, signed, allowances, eta, reconstruction):
    """Ordered eigenvalues from small signed inertia problems, not full eigh."""
    result = []
    discard = np.max(np.abs(signed[np.abs(signed) <= PROTOCOL['signed_drop']]), initial=0)
    keep = np.abs(signed) > PROTOCOL['signed_drop']
    signs = np.sign(signed[keep])
    weighted = q[:, keep]*np.sqrt(np.abs(signed[keep]))
    for index in [10, 26, 40]:
        low, high = 0., float(len(d))
        minimum_pivot = float('inf')
        uncertain = False
        for iteration in range(60):
            midpoint = (low+high)/2
            if np.any(d == midpoint):
                midpoint = np.nextafter(midpoint, high)
            small = -np.diag(signs)-weighted.conj().T@(weighted/(d-midpoint)[:, None])
            small = (small+small.conj().T)/2
            pivots = np.linalg.eigvalsh(small)
            minimum_pivot = min(minimum_pivot, float(np.min(np.abs(pivots))))
            uncertainty = 64*np.finfo(float).eps*len(signs)*(1+np.linalg.norm(small, 'fro'))
            base = int(np.sum(d < midpoint)-np.sum(signs > 0))
            lower_count = base+np.sum(pivots < -uncertainty)
            upper_count = base+np.sum(pivots < uncertainty)
            if upper_count <= len(d)-index:
                low = midpoint
            elif lower_count > len(d)-index:
                high = midpoint
            else:
                uncertain = True
                break
        intervals = []
        for allowance in allowances:
            radius = allowance['kernel']/eta+discard+reconstruction
            intervals.append([max(0., low-radius), high+radius])
        result.append(dict(index=index, compressed_bracket=[low, high],
            intervals=intervals, small_dimension=len(signs), iterations=iteration+1,
            stopped_for_uncertain_inertia=uncertain,
            minimum_inertia_pivot=minimum_pivot,
            numerical_status='Scaled FP64 signed inertia with disclosed pivot sensitivity; refinement stops at an uncertain sign and retains the previous wider bracket. Not an interval certificate.'))
    return result


def run_case(n, gamma, source, output, existing, archive_summary):
    start = time.monotonic()
    root = source.parent.parent
    sources = {}
    def read_npz(path):
        sources[str(path.relative_to(root))] = digest(path)
        return dict(np.load(path))
    a = construction.construct(n, 16, int(np.ceil(np.sqrt(n))), gamma)
    jhat = a['approximate']
    j = core.design(a['x'], a['centers'], gamma)
    y = targets(a['x'])/np.sqrt(len(j))
    old = next((r for r in existing['rows'] if r['n'] == n and r['gamma'] == gamma), None)
    eta = old['eta'] if old else .5/j.shape[1]
    executed = [None]*len(TARGETS)
    checkpoints = []
    if n == 512:
        arrays = read_npz(root/'common/N512/arrays.npz')
        np.testing.assert_allclose(y, arrays['y_train'], atol=1e-14)
        y = arrays['y_train']
        if gamma in [8, 12, 16, 64]:
            archived = next(r for r in archive_summary['dictionaries'] if r['gamma'] == gamma)
            assert core.array_hash(j) == archived['matrix_hash']
            executed = archived['executed_hits']
            trajectory = source.parent/'capped_kernel/evidence/gd_trajectories'/archived['archived_case']
            for name in ['meta.json', 'training.json', 'curve.json']:
                path = trajectory/name
                sources[str(path.relative_to(root))] = digest(path)
            checkpoints = json.loads((trajectory/'curve.json').read_text())
    norms = np.sum(y*y, axis=0)
    h = eta*(jhat.T@jhat)
    forcing = np.sqrt(eta)*(jhat.T@y)
    d, q, signed = construction.low_rank_gram(a)
    keep = np.abs(signed) > PROTOCOL['signed_drop']
    dropped = np.max(np.abs(signed[~keep]), initial=0)
    solve = signed_lowrank_solver(eta*d, q[:, keep], eta*signed[keep],
        to_basis=lambda v: construction.to_bulk_basis(v, a),
        from_basis=lambda v: inverse_bulk(v, a))
    width = j.shape[1]-1
    arithmetic = 64*np.finfo(float).eps*np.sqrt(width+1)*(1+gamma*2+np.log2(n))
    allowances = []
    for factor in PROTOCOL['arithmetic_factors']:
        eps = a['feature_error']+factor*arithmetic
        allowances.append(dict(factor=factor, feature=eps,
            kernel=eta*(2*np.sqrt(width+1)+eps)*eps))
    transformed_h = construction.to_bulk_basis(
        construction.to_bulk_basis(h/eta, a).conj().T, a).conj().T
    reconstruction = float(np.linalg.norm(transformed_h-np.diag(d)-(q*signed)@q.conj().T, 'fro'))
    spectral = spectral_brackets(d, q, signed, allowances, eta, reconstruction)
    # A capacity witness is valid regardless of how accurately its ridge solve
    # was performed. Its measured feature residual is evaluated independently.
    witnesses = []
    for shift in PROTOCOL['ridge_shifts']:
        w = np.sqrt(eta)*np.real(solve(forcing, -shift))
        residual = np.linalg.norm(y-jhat@w, axis=0)
        wnorm = np.linalg.norm(w, axis=0)
        witnesses.append(dict(shift=shift, relative_residual=residual/np.sqrt(norms),
            weight_norm=wnorm, floor_upper=[np.minimum(1.,
                (residual+allowance['feature']*wnorm)**2/norms) for allowance in allowances]))
    floor_upper = np.min(np.array([w['floor_upper'] for w in witnesses]), axis=0)
    filters = []
    cutoffs = np.geomspace(PROTOCOL['cutoff_min'], PROTOCOL['cutoff_max'], PROTOCOL['cutoff_count'])
    order = PROTOCOL['order']
    poles = np.exp(1j*np.pi*(2*np.arange(order)+1)/order)
    distance = np.where(poles.real > 0, np.abs(poles.imag), 1.)
    sensitivity_constant = np.mean(1/distance**2)
    guard_base = 64*np.finfo(float).eps
    hnorm = np.linalg.norm(h, 'fro')
    gnorm = np.linalg.norm(forcing, axis=0)
    for cutoff in cutoffs:
        scale = cutoff/PROTOCOL['margin']
        value = rational_filter_mass(forcing, norms, scale, order, solve,
            action=lambda v: h@v, kernel_error=0.,
            residual_guard=lambda v, shift, residual: guard_base*(
                (hnorm+abs(shift))*np.linalg.norm(v, axis=0)+gnorm))
        leakage = 1/(1+PROTOCOL['margin']**order)
        low = []
        for allowance in allowances:
            transfer = allowance['kernel']*sensitivity_constant/scale
            radius = (np.asarray(value['solve_radius'])
                      +allowance['factor']*np.asarray(value['arithmetic_radius'])+transfer)
            lower = np.maximum(0., (np.asarray(value['mass'])-radius-leakage)/(1-leakage))
            low.append(np.maximum(0., lower-floor_upper[len(low)]))
        filters.append(dict(cutoff=cutoff, scale=scale, expectation=value,
            positive_mass_lower=np.array(low), construction_sensitivity=sensitivity_constant/scale))
    masses = np.stack([f['positive_mass_lower'] for f in filters], axis=1)
    necessary = np.zeros_like(masses, dtype=np.int64)
    positive = masses > PROTOCOL['epsilon']**2
    logs = np.log(np.maximum(np.sqrt(masses)/PROTOCOL['epsilon'], 1.))
    necessary[positive] = np.ceil((logs/(-np.log1p(-cutoffs))[None, :, None])[positive]).astype(np.int64)
    selected = np.argmax(necessary, axis=1)
    best = np.max(necessary, axis=1)
    analytic_eta = .5/(width+1)
    scale_eta = analytic_eta/eta
    auxiliary = np.ceil(logs/(-np.log1p(-cutoffs*scale_eta))[None, :, None]).astype(np.int64)
    auxiliary_best = np.max(auxiliary, axis=1)

    # References are constructed only after all compact bound selections.
    if n == 512 and gamma in [8, 12, 16, 64]:
        truth = read_npz(source/f'reference_g{gamma}.npz')
    else:
        truth = reference.rectangular_forecast(j, y, eta)
    assert np.max(truth['rates']) < 1
    true_hits = reference.first_hit(truth)
    for spectral_row in spectral:
        value = float(truth['rates'][spectral_row['index']-1]/eta)
        spectral_row['reference_eigenvalue'] = value
        spectral_row['reference_contained'] = [low <= value <= high
                                               for low, high in spectral_row['intervals']]
        spectral_row['relative_width'] = [(high-low)/value if value > 0 else None
                                          for low, high in spectral_row['intervals']]
    aux_truth = dict(truth, rates=truth['rates']*scale_eta)
    aux_hits = reference.first_hit(aux_truth)
    true_mass = np.array([np.sum(truth['weights'][(truth['rates'] > 0)&(truth['rates'] <= t)], axis=0)
                          for t in cutoffs])
    # FP64 reference weights on near-zero rates are not individual certified
    # eigenspaces; the positive-band comparison is a numerical diagnostic.
    violations = np.max(masses-true_mass[None, :, :], initial=0)
    assert violations < 1e-7
    for target, hit in enumerate(true_hits):
        if hit is not None:
            assert np.all(best[:, target] <= hit)
    corrected = reference.rectangular_forecast(jhat, y, eta)
    corrected_hits = reference.first_hit(corrected)
    checkpoint_gap = None
    if checkpoints:
        checkpoint_gap = max(float(np.max(np.abs(reference.error(corrected, c['step'])-c['train'])))
                             for c in checkpoints)
    np.savez_compressed(output/f'N{n}_g{gamma}.npz',
        rates=truth['rates'], weights=truth['weights'], floor=truth['floor'],
        corrected_rates=corrected['rates'], corrected_weights=corrected['weights'],
        corrected_floor=corrected['floor'], cutoffs=cutoffs,
        positive_mass_lower=masses, true_positive_mass=true_mass,
        necessary_times=necessary, auxiliary_necessary_times=auxiliary)
    failure_controls = (diagonal_benchmarks(j, y, eta)
                        if n == 512 and gamma in [8, 12, 16, 64] else None)
    return jsonable(dict(n=n, gamma=gamma, samples=len(j), width=width, eta=eta,
        step_role='archived curvature-normalized' if old else 'prescribed analytic stable',
        targets=TARGETS, signed_rank=int(np.sum(keep)), nominal_feature_rank=a['feature_correction_rank'],
        setup_dimension=min(width+1, 2*a['feature_correction_rank']),
        discarded_signed_norm=float(dropped), boundary=a['boundary'], image_terms=a['terms'],
        analytic_feature_error=a['feature_error'], arithmetic_feature_allowance=arithmetic,
        allowances=allowances, witnesses=witnesses, floor_upper=floor_upper,
        spectral_brackets=spectral, gram_reconstruction_frobenius=reconstruction,
        filters=filters, best_necessary=best, selected_cutoff_indices=selected,
        reference_hits=true_hits, corrected_reference_hits=corrected_hits, executed_hits=executed,
        analytic_eta=analytic_eta, auxiliary_best_necessary=auxiliary_best,
        auxiliary_reference_hits=aux_hits, reference_mass_violation=float(violations),
        checkpoints=checkpoints, maximum_checkpoint_gap=checkpoint_gap,
        diagnostics=diagnostics(n, gamma, a, j), failure_controls=failure_controls,
        source_sha256=sources,
        seconds=time.monotonic()-start))


def figures(summary, audit, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator
    from .pi_brief_figure import COLORS

    gammas = [8, 12, 16, 64]
    rows = [next(r for r in summary['rows'] if r['n'] == 512 and r['gamma'] == g) for g in gammas]
    probes = [next(c for c in next(r for r in audit['records']
        if r['n'] == 512 and r['q'] == 16 and r['gamma'] == g)['cases']
        if c['name'] == 'gaussian_sine_20_pi') for g in gammas]
    plt.rcParams.update({'font.family':'serif', 'font.serif':['STIXGeneral'],
        'mathtext.fontset':'stix', 'font.size':9, 'axes.titlesize':10,
        'axes.titlelocation':'left', 'axes.titlepad':11,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.6,
        'axes.edgecolor':'#9AA3AA', 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.65),
                             gridspec_kw={'width_ratios':[1, 1.25, 1.15]})
    fig.subplots_adjust(left=.065, right=.99, bottom=.19, top=.76, wspace=.45)
    fig.legend(handles=[Line2D([], [], color=c, lw=2, label=rf'$\gamma={g}$')
                        for g, c in zip(gammas, COLORS)],
               loc='upper center', bbox_to_anchor=(.5, 1.015), ncol=4,
               frameon=False, fontsize=10, columnspacing=2.5)
    ax = axes[0]
    upper = [p['general_cap_refined'] for p in probes]
    ax.loglog(gammas, upper, color='#263442', lw=1.4, label='Analytic cap bound')
    for g, p, color in zip(gammas, probes, COLORS):
        ax.scatter(g, p['finite_quadratic'], s=34, facecolors='white', edgecolors=color,
                   linewidths=1.3, zorder=3)
    ax.set(title='A  Explicit gamma-cap bound', xlabel=r'Common slope / cap $\gamma$',
           ylabel=r'Kernel action $v^\mathsf{T}K_\gamma v$', ylim=(3e-8, .2))
    ax.set_xticks(gammas, labels=list(map(str, gammas)))
    ax.set_yticks([1e-7, 1e-5, 1e-3, 1e-1])
    ax.text(.03, .97, 'Windowed sine probe', transform=ax.transAxes, va='top', fontsize=8)
    ax.legend(handles=[Line2D([], [], color='#263442', lw=1.4, label='Analytic cap bound'),
                       Line2D([], [], color='#263442', lw=0, marker='o', ms=4,
                              markerfacecolor='white', label='Original tanh kernel')],
              loc='lower right', frameon=False, fontsize=7.2, handlelength=1.5)

    ax = axes[1]
    for r, color in zip(rows, COLORS):
        saved = np.load(output/f'N512_g{r["gamma"]}.npz')
        t = saved['cutoffs']; actual = saved['true_positive_mass'][:, 0]
        bound = saved['positive_mass_lower'][1, :, 0]
        ax.loglog(t, np.where(actual > 0, actual, np.nan), color=color, lw=1.3)
        ax.loglog(t, np.where(bound > 0, bound, np.nan), color=color, lw=1.3, ls=(0, (4, 2)))
    ax.set(title='B  Positive slow target energy', xlabel=r'Upper normalized rate $b$',
           ylabel=r'Target energy in $(0,b]$', xlim=(1e-8, .02), ylim=(1e-6, 1.2))
    ax.set_xticks([1e-8, 1e-6, 1e-4, 1e-2]); ax.set_yticks([1e-6, 1e-4, 1e-2, 1])
    ax.text(.03, .97, 'Sine-mixture target\nNull energy excluded',
            transform=ax.transAxes, va='top', fontsize=8)
    ax.legend(handles=[Line2D([], [], color='#263442', lw=1.3, label='Original tanh spectrum'),
                       Line2D([], [], color='#263442', lw=1.3, ls='--', label='Compact lower bound (1×)')],
              loc='lower right', frameon=False, fontsize=7.2, handlelength=1.5)

    ax = axes[2]
    full = [r['corrected_reference_hits'][0] for r in rows]
    necessary = [r['best_necessary'][1][0] for r in rows]
    ax.loglog(gammas, full, color='#263442', lw=1.1)
    ax.loglog(gammas, necessary, color='#78838C', lw=.8, ls='--')
    for g, r, color in zip(gammas, rows, COLORS):
        ax.scatter(g, r['executed_hits'][0], s=31, facecolors='white', edgecolors=color,
                   linewidths=1.2, zorder=4)
        low, middle, high = r['best_necessary'][2][0], r['best_necessary'][1][0], r['best_necessary'][0][0]
        ax.errorbar(g, middle, yerr=[[middle-low], [high-middle]], color=color,
                    marker='^', markersize=5, lw=1.1, capsize=2.5, zorder=3)
    ax.set(title='C  Necessary delay vs actual GD', xlabel=r'Common slope $\gamma$',
           ylabel='Updates to 1% relative residual', ylim=(2e3, 4e7))
    ax.set_xticks(gammas, labels=list(map(str, gammas))); ax.set_yticks([1e4, 1e5, 1e6, 1e7])
    ax.text(.03, .97, 'Sine-mixture target', transform=ax.transAxes, va='top', fontsize=8)
    ax.legend(handles=[Line2D([], [], color='#263442', lw=1.1, label='Full constructed forecast'),
                       Line2D([], [], color='#263442', lw=0, marker='o', ms=4,
                              markerfacecolor='white', label='Executed GD crossing'),
                       Line2D([], [], color='#78838C', lw=.8, ls='--', marker='^', ms=4,
                              label='Necessary time (1×)')],
              loc='upper right', bbox_to_anchor=(1., .83), frameon=False,
              fontsize=6.8, handlelength=1.5)
    ax.text(.03, .04, 'Bars: 0–10× numerical sensitivity',
            transform=ax.transAxes, fontsize=7)
    for ax in axes:
        ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(length=3, pad=3)
    fig.savefig(output/'structured_gamma_three_panel.png', dpi=300)
    fig.savefig(output/'structured_gamma_three_panel.pdf', metadata={'CreationDate':None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=DEFAULT.parent/'structured_gamma')
    parser.add_argument('--primary-only', action='store_true')
    parser.add_argument('--figures-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        figures(json.loads((args.output/'summary.json').read_text()),
                json.loads((args.output/'symbol_audit.json').read_text()), args.output)
        return
    existing_path = args.source.parent/'uniform_grid_spectrum/summary.json'
    existing = json.loads(existing_path.read_text())
    archived = json.loads((args.source/'summary.json').read_text())
    code_hashes = {name: digest(Path(__file__).with_name(name)) for name in
        ['structured_gamma_analysis.py', 'structured_resolvent.py', 'structured_gamma.py', 'uniform_grid_spectrum.py']}
    rows = []
    for n, gamma in (CASES[:2] if args.primary_only else CASES):
        path = args.output/f'N{n}_g{gamma}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get('code_sha256') == code_hashes and saved.get('protocol') == PROTOCOL:
                rows.append(saved['result'])
                continue
        result = run_case(n, gamma, args.source, args.output, existing, archived)
        path.write_text(json.dumps(dict(protocol=PROTOCOL, code_sha256=code_hashes, result=result),
                                   indent=2, allow_nan=False)+'\n')
        rows.append(result)
        print(json.dumps({k: result[k] for k in
            ['n', 'gamma', 'signed_rank', 'best_necessary', 'reference_hits', 'seconds']}), flush=True)
    summary = dict(protocol=PROTOCOL, code_sha256=code_hashes, rows=rows,
        source_sha256={str(existing_path): digest(existing_path),
                       str(args.source/'summary.json'): digest(args.source/'summary.json')},
        numerical_status='FP64 diagnostic. Analytic identities, solve-residual formulas, and exact-arithmetic feature tails are proved separately; formation/roundoff guards and spectral references are not interval certificates.',
        development_note='Primary gamma8/64 first used61cutoffs ending1e-3. Before remaining cases, coverage extended to97cutoffs through0.5 to include faster target directions; order, margin, rank cutoff, and witness shifts unchanged.',
        new_gpu_hours=0, expected_full_target_cases=90)
    cap_path = args.output/'symbol_audit.json'
    if cap_path.exists():
        summary['source_sha256']['symbol_audit.json'] = digest(cap_path)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    if cap_path.exists() and not args.primary_only:
        figures(summary, json.loads(cap_path.read_text()), args.output)


if __name__ == '__main__':
    main()
