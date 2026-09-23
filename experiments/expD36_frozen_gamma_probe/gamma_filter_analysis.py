"""Validate the fixed-geometry gamma filter against archived readout runs.

Produces numerical evidence and figures only. Predictions use F D_gamma C;
the original tanh matrix supplies defect checks and an independent reference.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import core, gamma_filter as gf, common_slope_poly as p
from . import common_slope_analysis as previous, finite_gamma_gram as reference

HARMONICS = [64, 128, 256, 512, 1024, 2048]


def best(rows, gamma, target=0, method='combined'):
    return previous.best(rows, gamma, target, method, resolution_key='harmonics')


def scalar_control(anchor, eta, largest_curvature):
    """Rescale the gamma-64 kernel without changing its target allocation."""
    return dict(rates=anchor['rates']/max(anchor['rates'])*(eta*largest_curvature),
                weights=anchor['weights'], floor=anchor['floor'])


def run(root, output):
    start = time.monotonic()
    arrays = np.load(root/'common/N512/arrays.npz')
    x, centers, y = [arrays[k] for k in ['x_train', 'centers', 'y_train']]
    baseline = previous.read(root/'refinements/common_slope_polynomial/summary.json')
    dictionaries = baseline['dictionaries']
    for key, values in [('grid_hash',x),('centers_hash',centers),('target_hash',y)]:
        if core.array_hash(values) != baseline[key]:
            raise ValueError('Archive input mismatch: '+key)
    output.mkdir(parents=True, exist_ok=True)
    references = {}
    for dictionary in dictionaries:
        gamma, eta = dictionary['gamma'], dictionary['eta']
        direct = core.design(x, centers, gamma)
        if core.array_hash(direct) != dictionary['matrix_hash']:
            raise ValueError('The original matrix no longer matches the archive')
        ref = reference.rectangular_forecast(direct, y, eta)
        if reference.first_hit(ref) != dictionary['reference_hits']:
            raise ValueError('Independent reference no longer matches the archive')
        references[gamma] = ref
        core.save_arrays(output/f'reference_g{gamma}.npz', rates=ref['rates'],
                         weights=ref['weights'], floor=ref['floor'])
    rows, geometries, final_models = [], [], {}
    for q in HARMONICS:
        geom = gf.geometry(x, centers, q)
        geometry_record = dict(harmonics=q, half_period=geom['half_period'], radius=geom['radius'],
                              sampling_hash=core.array_hash(geom['f']),
                              center_map_hash=core.array_hash(geom['c']))
        geometries.append(geometry_record)
        for dictionary in dictionaries:
            tick = time.monotonic()
            gamma, eta = dictionary['gamma'], dictionary['eta']
            approximate, budget = gf.synthesize(geom, gamma)
            model = p.factor(approximate, y, eta)
            allowance = (budget['construction_allowance']+model['reconstruction_error']
                         +model['singular'][0]*model['orthogonality_error'])
            direct = core.design(x, centers, gamma)
            bounds = p.defect(direct, approximate, model, eta, budget['synthesis_remainder'], allowance)
            results = {method:[p.crossing_bracket(model,bounds,t,method=method)
                              for t in range(y.shape[1])] for method in ['analytic','action','combined']}
            stress = dict(bounds)
            delta = budget['synthesis_remainder']+10*allowance
            stress['analytic_kernel'] = (2*model['singular'][0]+delta)*delta
            stress['arithmetic_kernel'] = (2*model['singular'][0]+10*allowance)*10*allowance
            stressed = [p.crossing_bracket(model,stress,t) for t in range(y.shape[1])]
            # These comparisons are validation only and cannot select the model.
            hits = np.array(dictionary['reference_hits'])
            times = np.unique(np.r_[0,np.geomspace(1,2*max(hits),100).astype(np.int64),hits,hits-1])
            gaps, violations = [], []
            for n in times:
                exact = reference.error(references[gamma],int(n))
                low, high = p.band(model,bounds,int(n))
                gaps.append(float(max(np.abs(p.error(model,int(n))-exact))))
                if np.any(low > exact+1e-12) or np.any(high < exact-1e-12):
                    violations.append(int(n))
            core.save_arrays(output/f'g{gamma}_Q{q}.npz', rates=model['rates'], weights=model['weights'],
                floor=model['floor'], singular=model['singular'],
                mode_action=bounds['mode_action'], null_action=bounds['null_action'],
                normalized_abs_loading=np.abs(model['loading']/model['norm']),
                frequencies=geom['omega'], multipliers=gf.multiplier(gamma,geom['omega']))
            row = dict(gamma=gamma,harmonics=q,eta=eta,**budget,arithmetic_allowance=float(allowance),
                measured_synthesis_frobenius=bounds['measured_synthesis_frobenius'],
                orthogonality_error=model['orthogonality_error'], reconstruction_error=model['reconstruction_error'],
                approximate_eta_L=float(model['rates'][0]),analytic_kernel_error=bounds['analytic_kernel'],
                arithmetic_kernel_error=bounds['arithmetic_kernel'],
                approximate_hits=reference.first_hit(model),results=results,arithmetic_stress_10x=stressed,
                reference_check_count=len(times)*y.shape[1],reference_band_violations=violations,
                max_approximate_error=max(gaps),seconds=time.monotonic()-tick,
                status='fp64_estimate_not_interval_certified')
            rows.append(row)
            if q == max(HARMONICS):
                final_models[gamma] = dict(rates=model['rates'],weights=model['weights'],floor=model['floor'],
                                          largest_curvature=float(model['singular'][0]**2))
            print(dict(gamma=gamma,harmonics=q,seconds=round(row['seconds'],2),
                       analytic=results['analytic'][0],combined=results['combined'][0]),flush=True)
        if (core.array_hash(geom['f']) != geometry_record['sampling_hash']
                or core.array_hash(geom['c']) != geometry_record['center_map_hash']):
            raise ValueError('Gamma evaluation mutated the fixed geometry')
    comparisons = []
    for dictionary in dictionaries:
        gamma, eta = dictionary['gamma'], dictionary['eta']
        control = scalar_control(final_models[64],eta,final_models[gamma]['largest_curvature'])
        comparisons.append(dict(gamma=gamma,scalar_control_hits=reference.first_hit(control),
            selected={method:[best(rows,gamma,t,method) for t in range(y.shape[1])]
                      for method in ['analytic','combined']},
            polynomial_baseline=[previous.best(baseline['rows'],gamma,t) for t in range(y.shape[1])]))
    summary = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        targets=baseline['targets'],gammas=baseline['gammas'],harmonics=HARMONICS,resolution_key='harmonics',
        epsilon=.01,n=512,samples=len(x),width=len(centers),
        grid_hash=baseline['grid_hash'],centers_hash=baseline['centers_hash'],target_hash=baseline['target_hash'],
        dictionaries=dictionaries,geometries=geometries,rows=rows,comparisons=comparisons,
        evidence_role='Retrospective validation on archived training runs; no fitted rates or trajectory data',
        arithmetic_allowance_formula='short-dot/pairwise allowance + phase allowance + SVD reconstruction + smax*orthogonality',
        numerical_status='FP64 sensitivity estimates; primary endpoints require separate interval audit',
        seconds=time.monotonic()-start,gpu_hours=0)
    core.write_json(output/'summary.json',summary)
    plot(summary,output)
    return summary


def plot(summary, output):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    gammas = summary['gammas']
    colors = plt.cm.viridis(np.linspace(.1,.85,len(gammas)))
    omega = np.linspace(0,40*np.pi,1000)
    for gamma,color in zip(gammas,colors):
        axes[0].semilogy(omega/np.pi,gf.multiplier(gamma,omega)**2,color=color,label=rf'$\gamma={gamma}$')
        selected = best(summary['rows'],gamma)
        q = selected['upper_harmonics'] or selected['lower_harmonics']
        for source,style,width in [(output/f'g{gamma}_Q{q}.npz','-',2),
                                   (output/f'reference_g{gamma}.npz',':',1.2)]:
            model = np.load(source)
            order = np.argsort(model['rates'])
            rates = np.maximum(model['rates'][order],1e-16)
            cumulative = model['floor'][0]+np.cumsum(model['weights'][order,0])
            axes[1].step(rates,cumulative,where='post',color=color,ls=style,lw=width)
    for frequency in [2,6,10]:
        axes[0].axvline(frequency,color='.7',ls=':',lw=.8)
    axes[0].set(xlabel=r'Physical frequency $\omega/\pi$',ylabel=r'Filter power $M_\gamma(\omega)^2$',
                ylim=(1e-12,1.3),title='A  Explicit gamma attenuation')
    axes[0].legend(fontsize=8)
    axes[1].set(xscale='log',yscale='log',xlim=(1e-10,1),ylim=(1e-5,1.2),
                xlabel=r'Per-update rate $\eta_\gamma\lambda$',ylabel='Target energy at or below rate',
                title='B  Finite target-weighted spectrum')
    axes[1].axhline(1e-4,color='.5',ls='--',lw=.8)
    axes[1].plot([],[],'k-',label='gamma filter')
    axes[1].plot([],[],'k:',label='original kernel reference')
    axes[1].legend(fontsize=8,loc='lower right')
    hits = [d['executed_hits'][0] for d in summary['dictionaries']]
    low = [c['selected']['combined'][0]['necessary'] for c in summary['comparisons']]
    high = [c['selected']['combined'][0]['sufficient'] for c in summary['comparisons']]
    axes[2].loglog(gammas,hits,'ko',ms=6,label='executed GD')
    axes[2].loglog(gammas,low,'^-',color='tab:orange',ms=5,label='filter necessary')
    axes[2].loglog(gammas,high,'v:',color='tab:orange',ms=5,label='filter sufficient')
    axes[2].loglog(gammas,[c['polynomial_baseline'][0]['necessary'] for c in summary['comparisons']],
                   '--',color='.55',label='previous polynomial necessary')
    axes[2].loglog(gammas,[c['scalar_control_hits'][0] for c in summary['comparisons']],
                   's--',color='tab:blue',label='scalar-rescaling control')
    axes[2].set(xlabel='Common slope gamma',ylabel='Updates to 1% residual',
                title='C  Predicted and measured delay')
    axes[2].set_xticks(gammas,labels=[str(g) for g in gammas])
    axes[2].minorticks_off()
    axes[2].legend(fontsize=7.5)
    fig.suptitle('One fixed geometry, gamma-dependent filtering, and readout learning\n'
                 'N=512; raw readout; sine-mixture target; archived GD clock',fontsize=12)
    fig.savefig(output/'three_panel.png',dpi=180)
    fig.savefig(output/'three_panel.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep'))
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    run(args.root,args.output or args.root/'refinements/gamma_factorized_kernel')


if __name__ == '__main__':
    main()
