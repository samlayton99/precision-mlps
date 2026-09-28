"""Endpoint Jacobian spectra using the actual selected joint optimizer states.

These local metrics omit momentum and subsequent feature/preconditioner motion;
they are diagnostics, never an Adam trajectory or learning-rate prediction.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from .section34_feature_access import metric_spectrum, residual_slow_energy


def jacobians(parameters, x):
    width = (len(parameters)-1)//3
    a, b, c = np.split(parameters[:-1], 3)
    argument = x[:, None]*a+b
    features = np.tanh(argument)
    decay = np.exp(-2*np.abs(argument))
    sech2 = 4*decay/(1+decay)**2
    readout = np.column_stack((features, np.ones(len(x))))
    full = np.column_stack((x[:, None]*c*sech2, c*sech2, readout))
    prediction = features@c+parameters[-1]
    assert full.shape[1] == 3*width+1
    return readout, full, prediction


def analyze(base, analysis, output):
    selected = json.loads((analysis/'summary.json').read_text())['selected']
    data = np.load(base/'input.npz')
    x, y = data['train_x'], data['target']
    joint_hash = hashlib.sha256((base/'joint_input.npz').read_bytes()).hexdigest()
    cutoffs = np.logspace(-14, 0, 281)
    output.mkdir(parents=True, exist_ok=True)
    records, arrays = [], dict(cutoffs=cutoffs)
    for optimizer, recipe in selected.items():
        run = Path(recipe['run'])
        metadata = json.loads((run/'metadata.json').read_text())
        if metadata['input_sha256'] != joint_hash:
            raise ValueError('Actual optimizer state and supplied data do not match')
        state = np.load(run/'state.npz')
        count, ri = int(state['count']), recipe['recipe_index']
        epsilon = metadata['config']['epsilon']
        width = metadata['width']
        for seed, parameters in enumerate(state['p'][:, ri]):
            readout, full, prediction = jacobians(parameters, x)
            residual = prediction-y
            observed_error = float(np.linalg.norm(residual)/np.linalg.norm(y))
            np.testing.assert_allclose(observed_error, recipe['train_errors'][seed], rtol=1e-9, atol=1e-11)
            matrices = [('readout', readout, slice(2*width, None)), ('full', full, slice(None))]
            if optimizer == 'adam':
                diagonal = 1/(np.sqrt(state['v'][seed, ri]/(1-.999**count))+epsilon)
            for family, matrix, block in matrices:
                metrics = [('raw', None)]
                if optimizer == 'adam':
                    metrics.append(('adaptive', diagonal[block]))
                for metric, scaling in metrics:
                    prefix = f'{optimizer}_seed{seed}_{family}_{metric}'
                    mu, rho, target_weights, residual_weights, checks = metric_spectrum(matrix, y, residual, scaling)
                    target_cdf = residual_slow_energy(rho, target_weights, checks['target']['total_energy'], cutoffs)
                    residual_cdf = residual_slow_energy(rho, residual_weights, checks['residual']['total_energy'], cutoffs)
                    arrays.update({prefix+'_eigenvalues':mu, prefix+'_relative_eigenvalues':rho,
                        prefix+'_target_weights':target_weights, prefix+'_residual_weights':residual_weights,
                        prefix+'_target_cdf':target_cdf, prefix+'_residual_cdf':residual_cdf})
                    item = dict(optimizer=optimizer, seed=seed, family=family, metric=metric,
                        checkpoint_step=count, recipe=recipe, train_error=observed_error, checks=checks,
                        mu_max=float(mu[0]), numerical_unresolved_target_energy_at_1e_18=float(residual_slow_energy(rho,target_weights,1.,[1e-18])[0]),
                        epsilon=epsilon if metric=='adaptive' else None,
                        preconditioner_source='actual joint optimizer second moment, bias corrected' if metric=='adaptive' else 'identity',
                        parameter_block='[c,d]' if family=='readout' else '[a,b,c,d]',
                        learning_rate_included=False, prefix=prefix,
                        state_sha256=hashlib.sha256((run/'state.npz').read_bytes()).hexdigest())
                    if scaling is not None:
                        item['diagonal_range']=[float(np.min(scaling)),float(np.max(scaling))]
                    records.append(item)
                    print(json.dumps(dict(case=prefix,error=observed_error,mu_max=float(mu[0]))),flush=True)
    np.savez_compressed(output/'metrics.npz', **arrays)
    (output/'summary.json').write_text(json.dumps(dict(cases=records,
        interpretation='Endpoint local Jacobian metrics. Adaptive diagonals come from the actual joint Adam state. Momentum, changing geometry, and changing preconditioning are omitted; these are not Adam rate predictions.',
        normalization='Jacobians divided by sqrt(m); both target and residual spectral energies divided by target squared norm.',
        numerical_resolution='Unresolved spectral energy is not a proved capacity floor.',
        analysis_sha256=hashlib.sha256((analysis/'summary.json').read_bytes()).hexdigest()),indent=2,allow_nan=False)+'\n')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':7.5,'axes.titlesize':8,'axes.labelsize':7.5,
                         'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(5.5,4.1),layout='constrained',sharex=True)
    colors = {'adam':'#0072B2','gd':'#D55E00'}
    for row,family in enumerate(['readout','full']):
        for optimizer in selected:
            for metric in (['raw','adaptive'] if optimizer=='adam' else ['raw']):
                subset=[r for r in records if (r['optimizer'],r['family'],r['metric'])==(optimizer,family,metric)]
                for col,energy in enumerate(['target','residual']):
                    values=np.array([arrays[r['prefix']+f'_{energy}_cdf'] for r in subset])
                    # Zero energies are simply outside the log display; stored values are unchanged.
                    positive=lambda v: np.where(v>0,v,np.nan)
                    axes[row,col].plot(cutoffs,positive(np.median(values,axis=0)),color=colors[optimizer],ls='--' if metric=='adaptive' else '-',lw=1.2)
                    axes[row,col].fill_between(cutoffs,positive(np.min(values,axis=0)),positive(np.max(values,axis=0)),color=colors[optimizer],alpha=.08)
        axes[row,0].set_ylabel(('Readout' if family=='readout' else 'Full Jacobian')+'\ncumulative energy / target energy')
    axes[0,0].set_title('Target energy in small-eigenvalue modes')
    axes[0,1].set_title('Remaining residual in those modes')
    for ax in axes.flat:
        ax.set(xscale='log',yscale='log',xlim=(1e-14,1),ylim=(1e-12,1.1))
        ax.grid(alpha=.15)
    for ax in axes[1]:
        ax.set_xlabel('Relative eigenvalue cutoff')
    handles=[Line2D([],[],color=colors[o],label=o.upper()) for o in selected]
    handles += [Line2D([],[],color='.3',label='Raw'),Line2D([],[],color='.3',ls='--',label='Actual Adam scaling')]
    axes[0,0].legend(handles=handles,frameon=False,fontsize=7,loc='upper left',handlelength=1.6)
    fig.savefig(output/'joint_endpoint_metrics.pdf')
    fig.savefig(output/'joint_endpoint_metrics.png',dpi=300)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    with threadpool_limits(limits=args.threads):
        analyze(args.base,args.analysis,args.output)


if __name__=='__main__':
    main()
