"""Frozen tanh readout spectra and exact-arithmetic GD predictions.

SVDs and dense evaluations are offline diagnostics. No SVD is used by GD.
Predictions retain singular values above rcond*sigma_max and explicitly hold
all remaining residual components fixed; they are not exact-null claims.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD24_gd_residual_spectrum.run import uniform_geometry
from experiments.expD24_gd_residual_spectrum.first_steps_refit import target_values
from experiments.expD24_gd_residual_spectrum.spectrum import write_gif_frame, frame_durations

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum'
LABELS = {'sine': 'Sine', 'sine_mixture': 'Mixed sine', 'runge': 'Runge',
          'gaussian_envelope': 'Gaussian envelope'}
THRESHOLDS = np.array([.1, .01, .001])
MAX_PREDICTED_STEPS = 10**32


def midpoint_grid(n, domain=(-1., 1.)):
    lo, hi = domain
    return lo+(np.arange(n)+.5)*(hi-lo)/n


def dictionary(x, centers, gamma):
    """Unscaled features including output bias; loss uses their RMS scaling."""
    return np.column_stack((np.tanh(gamma*(x[:, None]-centers)), np.ones(len(x))))


def decompose(A, y, rcond):
    """Return SVD and target loadings; y may contain several target columns."""
    U, s, Vh = svd(A, full_matrices=False, lapack_driver='gesdd')
    keep = s > rcond*s[0] if len(s) and s[0] else np.zeros(len(s), dtype=bool)
    alpha = U[:, keep].T @ y
    residual_floor = y-U[:, keep] @ alpha
    floor_squared = np.sum(residual_floor**2, axis=0)
    return U, s, Vh, keep, alpha, floor_squared


def contraction_logs(s, eta):
    """Log contraction of each retained singular direction for monotone GD."""
    x = eta*np.asarray(s)**2
    if np.any(x < 0) or np.any(x > 1+2e-14):
        raise ValueError('This predictor requires 0 <= eta*sigma^2 <= 1.')
    with np.errstate(divide='ignore'):
        return np.log1p(-np.minimum(x, 1.))


def predicted_relative_error(step, logs, alpha, floor_squared, norm_y):
    """Stable evaluation even when eta*sigma^2 is smaller than machine epsilon."""
    step = float(step)
    decay = np.ones_like(logs) if step == 0 else np.exp(2*step*logs)
    return np.sqrt(floor_squared+np.sum(alpha**2*decay))/norm_y


def steps_to_error(logs, alpha, floor_squared, norm_y, tolerance, max_steps=MAX_PREDICTED_STEPS):
    """Minimum predicted integer step; inf means unattainable in retained model.

    Counts beyond exact fp64 integer resolution are order-of-magnitude estimates.
    The cap is an explicit reporting limit, not a claim of true impossibility.
    """
    threshold2 = (tolerance*norm_y)**2
    if predicted_relative_error(0, logs, alpha, floor_squared, norm_y) <= tolerance:
        return 0.
    if floor_squared > threshold2:
        return np.inf
    if floor_squared == threshold2:
        active = alpha != 0
        return 1. if np.all(np.isneginf(logs[active])) else np.inf
    high = 1
    while high < max_steps and predicted_relative_error(high, logs, alpha, floor_squared, norm_y) > tolerance:
        high *= 2
    high = min(high, max_steps)
    if predicted_relative_error(high, logs, alpha, floor_squared, norm_y) > tolerance:
        return np.inf
    low = 0
    while high-low > 1:
        middle = (low+high)//2
        if predicted_relative_error(middle, logs, alpha, floor_squared, norm_y) <= tolerance:
            high = middle
        else:
            low = middle
    return float(high)


def validate_explicit_gd(A, y, U, s, keep, alpha, floor_squared, eta, steps=2000):
    """Independent matrix-gradient updates; compares four targets simultaneously."""
    v = np.zeros((A.shape[1], y.shape[1]))
    logs = contraction_logs(s[keep], eta)
    norms = np.linalg.norm(y, axis=0)
    max_error = 0.
    selected = {0, 1, 2, 10, 50, 200, steps}
    checkpoints, actuals, predicteds = [], [], []
    for step in range(steps+1):
        residual = A @ v-y
        if step in selected:
            actual = np.linalg.norm(residual, axis=0)/norms
            predicted = np.array([predicted_relative_error(step, logs, alpha[:, j],
                                  floor_squared[j], norms[j]) for j in range(y.shape[1])])
            # Roundoff in explicit GD and in discarded near-null modes is allowed.
            np.testing.assert_allclose(actual, predicted, rtol=2e-9, atol=2e-12)
            max_error = max(max_error, float(np.max(abs(actual-predicted))))
            checkpoints.append(step); actuals.append(actual); predicteds.append(predicted)
        if step < steps:
            v -= eta*(A.T @ residual)
    return max_error, np.array(checkpoints), np.array(actuals), np.array(predicteds)


def run(cfg, output):
    centers, h, _ = uniform_geometry(cfg['resolution'], cfg['halo'])
    x = midpoint_grid(cfg['n_train'], cfg['domain'])
    xx = midpoint_grid(cfg['n_eval'], cfg['domain'])
    y = np.column_stack([target_values(t, x, cfg) for t in cfg['targets']])/np.sqrt(len(x))
    yy = np.column_stack([target_values(t, xx, cfg) for t in cfg['targets']])
    ynorm = np.linalg.norm(y, axis=0)
    lambdas = np.linspace(cfg['lambda_min'], cfg['lambda_max'], cfg['lambda_frames'])
    payload = dict(config_json=json.dumps(cfg), lambdas=lambdas, gammas=lambdas/h,
                   thresholds=THRESHOLDS, centers=centers, target_norm=ynorm,
                   evaluation_steps=np.array([500, 2000, 10000]),
                   rates_description=np.array(['constant 0.002', '1/sigma_max^2']),
                   max_predicted_steps=np.array(str(MAX_PREDICTED_STEPS)))
    accum = {key: [] for key in ('singular_values', 'rank', 'readout_rates', 'alpha',
             'floor_relative', 'eval_floor_relative', 'prediction_steps', 'predicted_errors',
             'coefficient_norm', 'efolding_steps', 'condition_retained')}
    validations = []
    for index, lam in enumerate(lambdas):
        gamma = lam/h
        A = dictionary(x, centers, gamma)/np.sqrt(len(x))
        U, s, Vh, keep, alpha, floors = decompose(A, y, cfg['readout_rcond'])
        rates = np.array([cfg['learning_rate'], 1/s[0]**2])
        coeff = Vh[keep].T @ (alpha/s[keep, None])
        eval_error = np.linalg.norm(dictionary(xx, centers, gamma) @ coeff-yy, axis=0)/np.linalg.norm(yy, axis=0)
        steps = np.empty((len(cfg['targets']), len(rates), len(THRESHOLDS)))
        predicted_errors = np.empty((len(cfg['targets']), len(rates), 3))
        taus = np.full((len(rates), len(s)), np.nan)
        for ir, eta in enumerate(rates):
            logs = contraction_logs(s[keep], eta)
            taus[ir, keep] = -1/logs
            for it in range(len(cfg['targets'])):
                steps[it, ir] = [steps_to_error(logs, alpha[:, it], floors[it], ynorm[it], tol)
                                  for tol in THRESHOLDS]
                for tolerance, count in zip(THRESHOLDS, steps[it, ir]):
                    if np.isfinite(count) and 0 < count < 2**53:
                        assert predicted_relative_error(count, logs, alpha[:, it], floors[it], ynorm[it]) <= tolerance
                        assert predicted_relative_error(count-1, logs, alpha[:, it], floors[it], ynorm[it]) > tolerance
                predicted_errors[it, ir] = [predicted_relative_error(k, logs, alpha[:, it], floors[it], ynorm[it])
                                             for k in payload['evaluation_steps']]
            if any(np.isclose(lam, v, atol=1e-12) for v in (.1, .25, 1., 2.)):
                discrepancy, ck, actual, predicted = validate_explicit_gd(A, y, U, s, keep, alpha, floors, eta)
                validations.append(dict(lambda_=float(lam), eta=float(eta), max_abs_discrepancy=discrepancy,
                                        steps=ck.tolist(), actual=actual.tolist(), predicted=predicted.tolist()))
        allalpha = np.zeros((len(s), len(cfg['targets'])))
        allalpha[keep] = alpha
        values = dict(singular_values=s, rank=keep.sum(), readout_rates=rates, alpha=allalpha,
                      floor_relative=np.sqrt(floors)/ynorm, eval_floor_relative=eval_error,
                      prediction_steps=steps, predicted_errors=predicted_errors,
                      coefficient_norm=np.linalg.norm(coeff, axis=0), efolding_steps=taus,
                      condition_retained=s[0]/s[keep][-1])
        for key, value in values.items(): accum[key].append(value)
        print(f'lambda={lam:.3f}, gamma={gamma:.2f}, rank={keep.sum()}, sigma1={s[0]:.3f}, '
              f'1%/0.1% steps mixed={steps[1,0,1]:.3g}/{steps[1,0,2]:.3g}', flush=True)
    payload.update({key: np.asarray(value) for key, value in accum.items()})
    payload['validation_json'] = json.dumps(validations)
    cap_hits = np.isinf(payload['prediction_steps']) & (payload['floor_relative'][:, :, None, None] < THRESHOLDS[None, None, None, :])
    payload['prediction_cap_hit'] = cap_hits
    assert not np.any(cap_hits), 'A count reached the reporting cap; label it separately from a floor violation.'
    payload['prediction_convention'] = np.array('Exact-arithmetic SVD recurrence in retained span; discarded residual fixed. Training relative L2; v0=0.')
    np.savez_compressed(output, **payload)
    write_summary(payload, cfg, output.with_suffix('.csv'))
    return payload


def write_summary(data, cfg, output):
    with output.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['lambda', 'gamma', 'target', 'numerical_rank', 'sigma_max', 'retained_condition',
                         'train_floor_rel_l2', 'eval_floor_rel_l2', 'rate_kind', 'eta',
                         'steps_to_0.1', 'steps_to_0.01', 'steps_to_0.001'])
        for i, lam in enumerate(data['lambdas']):
            for t, target in enumerate(cfg['targets']):
                for ir, kind in enumerate(('constant', 'spectral_monotone')):
                    writer.writerow([lam, data['gammas'][i], target, data['rank'][i],
                        data['singular_values'][i, 0], data['condition_retained'][i],
                        data['floor_relative'][i,t], data['eval_floor_relative'][i,t], kind,
                        data['readout_rates'][i,ir], *data['prediction_steps'][i,t,ir]])


def style_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(which='major', alpha=.18)


def animate(data, cfg, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(1, 2, figsize=(13, 8.0), dpi=145)
    fig.subplots_adjust(left=.085, right=.97, bottom=.21, top=.66, wspace=.26)
    fig.suptitle('Fixed-geometry readout: singular values and GD learning times', y=.975, fontsize=17)
    title = fig.text(.5, .915, '', ha='center', fontsize=12)
    fig.text(.5, .85, r'$A=[\tanh(\gamma(x_i-z_j)),\,1]/\sqrt{n}$,  $\gamma=\lambda/h$,  $h=1/64$'
             '\nUniform centers; 177 neurons + bias; 1,024 training samples on [−1,1]', ha='center', fontsize=11)
    ax, bx = axes
    n = data['singular_values'].shape[1]; indices = np.arange(1, n+1)
    ref = int(np.argmin(abs(data['lambdas']-.25)))
    ax.plot(indices, np.maximum(data['singular_values'][ref], 1e-16), color='.75', lw=1.4, ls='--', label='QI reference λ = 0.25')
    retained, = ax.plot([], [], color='#2878b5', lw=2, label='Current retained singular values')
    discarded, = ax.plot([], [], color='.55', lw=1.5, label='Below numerical cutoff')
    cutoff = ax.axhline(1e-12, color='#9b599e', ls=':', lw=1.5, label=r'Cutoff $10^{-13}\sigma_1$')
    ax.set_yscale('log'); ax.set_ylim(1e-16, 20)
    ax.set_ylabel(r'Singular value $\sigma_j$', fontsize=12)
    ax.set_title('Dictionary sensitivity', pad=12, fontsize=13)
    const, = bx.plot([], [], color='#2878b5', lw=2, label='Common GD rate η = 0.002')
    norm, = bx.plot([], [], color='#d47b24', lw=2, label=r'Spectral rate $\eta=1/\sigma_1^2$')
    bx.set_yscale('log'); bx.set_ylim(1, 1e28)
    bx.set_ylabel('Steps for one e-fold decrease\nin a singular-mode residual', fontsize=12)
    bx.set_title('Smaller singular values learn more slowly', pad=12, fontsize=13)
    for a in axes:
        a.set_xlim(1,n); a.set_xlabel(r'Singular-direction index $j$ (largest first)', fontsize=11)
        a.set_xticks([1, 40, 80, 120, 160, n]); style_axes(a)
    handles_a, labels_a = ax.get_legend_handles_labels()
    handles_b, labels_b = bx.get_legend_handles_labels()
    fig.legend(handles_a+handles_b, labels_a+labels_b, loc='upper center',
               bbox_to_anchor=(.5,.793), ncol=3, frameon=False, fontsize=9.5)
    fig.text(.5,.055, r'$\tau_j=-1/\log(1-\eta\sigma_j^2)$; points below one step are shown at one.'
             '\nOne e-fold means residual amplitude divided by e. Matrix directions are not Fourier frequencies.'
             '\nDiscarded directions have no reported learning time.',
             ha='center', fontsize=10)
    durations = frame_durations(len(data['lambdas']), cfg['gif_seconds'])
    palette = None
    with output.open('wb') as stream:
        for i, lam in enumerate(data['lambdas']):
            rank = int(data['rank'][i]); s = data['singular_values'][i]
            retained.set_data(indices[:rank], s[:rank]); discarded.set_data(indices[rank:], np.maximum(s[rank:],1e-16))
            cutoff.set_ydata([cfg['readout_rcond']*s[0]]*2)
            const.set_data(indices[:rank], np.maximum(data['efolding_steps'][i,0,:rank],1))
            norm.set_data(indices[:rank], np.maximum(data['efolding_steps'][i,1,:rank],1))
            title.set_text(f'λ = {lam:.3f}      γ = {data["gammas"][i]:.1f}      numerical rank = {rank}/{n}'
                           f'      stable range: 0 < η < {2/s[0]**2:.5f}')
            fig.canvas.draw()
            rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
            palette = write_gif_frame(stream, rgb, palette, int(durations[i]))
        stream.write(b';')
    plt.close(fig)


def convergence_plot(data, cfg, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.5), dpi=155, sharex=True)
    fig.subplots_adjust(left=.075,right=.985, bottom=.235,top=.79,wspace=.25,hspace=.29)
    fig.suptitle('Does increasing λ make the readout easier to learn?', fontsize=19, y=.98)
    fig.text(.5,.933, r'Frozen uniform centers; zero initial readout. Accuracy target: relative $L_2\leq10^{-3}$ on training samples.', ha='center', fontsize=12)
    handles = [Line2D([],[],color='#72519e',lw=2,label='Numerical LS floor (training)'),
               Line2D([],[],color='#72519e',lw=1.5,ls='--',label='LS refit (independent grid)'),
               Line2D([],[],color='#2878b5',lw=2,label='GD: constant η = 0.002'),
               Line2D([],[],color='#d47b24',lw=2,label=r'GD: $\eta=1/\sigma_1^2$')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.897),ncol=4,frameon=False,fontsize=11)
    lam = data['lambdas']; main_threshold_index = 2
    all_counts = data['prediction_steps'][:,:,:,main_threshold_index]
    ymax = 10**np.ceil(np.log10(np.max(all_counts[np.isfinite(all_counts)])))
    ymax = min(max(ymax,1e5),1e14)
    finite_counts=all_counts[np.isfinite(all_counts)]
    assert np.min(finite_counts) >= 100
    for t,target in enumerate(cfg['targets']):
        ax,bx = axes[:,t]
        floor = data['floor_relative'][:,t]
        ax.plot(lam,np.maximum(floor,1e-16),color='#72519e',lw=2)
        ax.plot(lam,np.maximum(data['eval_floor_relative'][:,t],1e-16),color='#72519e',lw=1.5,ls='--')
        ax.axhline(.001,color='.4',lw=1.3,ls=':')
        ax.set_yscale('log'); ax.set_ylim(1e-16,1)
        ax.set_title(LABELS[target],fontsize=14,pad=12)
        for ir,color in enumerate(('#2878b5','#d47b24')):
            values = all_counts[:,t,ir]
            bx.plot(lam,np.where(np.isfinite(values),np.minimum(values,ymax),np.nan),color=color,lw=2)
            overflow=np.isfinite(values)&(values>ymax)
            if np.any(overflow):
                bx.plot(lam[overflow],np.full(overflow.sum(),ymax),'^',color=color,ms=7,clip_on=False)
        unattainable = floor >= .001
        if np.any(unattainable):
            for a in (ax,bx):
                a.fill_between(lam,0,1,where=unattainable,transform=a.get_xaxis_transform(),color='.88',alpha=.8,interpolate=True,zorder=-2)
            bx.text(lam[unattainable].mean(),.50,'Floor exceeds target',transform=bx.get_xaxis_transform(),
                    rotation=90,va='center',ha='center',color='.4',fontsize=9)
        bx.set_yscale('log'); bx.set_ylim(1e2,ymax)
        bx.set_xlabel(r'Initial $\lambda=\gamma h$',fontsize=12)
        for a in (ax,bx):
            a.set_xlim(.1,2);a.set_xticks([.25,.5,1,1.5,2]); a.axvline(.25,color='.7',ls=':',lw=1);style_axes(a)
        if t==0:
            ax.set_ylabel('Relative L₂ after readout solve',fontsize=12)
            bx.set_ylabel('Predicted GD steps\nto relative L₂ ≤ 0.001',fontsize=12)
    fig.text(.5,.137, r'$E_k^2=E_\perp^2+\sum_{j\in\mathrm{retained}}(\alpha_j/\|y\|)^2(1-\eta\sigma_j^2)^{2k},\quad\alpha_j=u_j^Ty$',ha='center',fontsize=13)
    fig.text(.5,.045,'Gray regions: numerical LS floor already exceeds the requested accuracy. Dotted vertical line: QI λ = 0.25.'
             '\nTriangles at λ = 1.45: mixed-sine counts exceed the plot range (5.0×10²³ common rate; 9.7×10²² spectral rate).'
             '\nTimes assume exact arithmetic and use target loadings. Singular values below 10⁻¹³σ₁ are excluded;'
             '\nthis is a retained-span prediction, not a claim that discarded directions are mathematically null.', ha='center',fontsize=10)
    fig.savefig(output);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args();cfg=yaml.safe_load((HERE/'config.yaml').read_text())
    figures=RESULTS/'figures';data_dir=RESULTS/'data'
    figures.mkdir(parents=True,exist_ok=True);data_dir.mkdir(parents=True,exist_ok=True)
    path=data_dir/'readout_spectrum.npz'
    with threadpool_limits(limits=cfg['threads']):
        if args.plot_only:
            with np.load(path,allow_pickle=False) as saved: data={key:saved[key] for key in saved.files}
            if json.loads(str(data['config_json'])) != cfg: raise ValueError('Saved spectrum configuration mismatch.')
        else: data=run(cfg,path)
        convergence_plot(data,cfg,figures/'readout_convergence.png')
        animate(data,cfg,figures/'readout_spectrum.gif')
    print(f'Saved spectrum, predictions and two figures in {RESULTS}',flush=True)

if __name__=='__main__':main()
