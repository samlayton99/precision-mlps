"""Run the retrospective common-slope probe; emit evidence, never report prose."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import core, common_slope_poly as p, finite_gamma_gram as reference

TARGETS = ['sine_mix_2_6_10', 'exp_sin_3pi', 'runge_25', 'quadratic', 'sine_2pi']
GAMMAS = [8, 12, 16, 64]
DEGREES = [32, 64, 128, 256, 512, 1024, 2048]


def read(path):
    return json.loads(path.read_text())


def run(root, output, gammas=GAMMAS, degrees=DEGREES):
    start = time.monotonic()
    arrays = np.load(root/'common/N512/arrays.npz')
    x, centers, y = [arrays[k] for k in ['x_train', 'centers', 'y_train']]
    campaign = read(root/'refinements/capped_kernel/campaign_summary.json')
    cases = {r['cap']:r for r in campaign['cases'] if r['n'] == 512 and r['family'] == 'common'}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    rows, dictionaries = [], []
    output.mkdir(parents=True, exist_ok=True)
    for gamma in gammas:
        case = cases[gamma]
        eta = case['training']['eta']
        j = core.design(x, centers, gamma)
        if core.array_hash(j) != case['matrix_hash'] or core.array_hash(y) != case['target_hash']:
            raise ValueError('Archive hash mismatch: the comparison would change its dictionary or target')
        old = read(root/f'dictionaries/N512_raw_g{gamma}/certificates.json')
        ref = reference.rectangular_forecast(j, y, eta)
        reference_hits = reference.first_hit(ref)
        if reference_hits != case['reference']['hits']['0.01']:
            raise ValueError('Independent rectangular reference disagrees with the archive')
        dictionary = dict(gamma=gamma, eta=eta, matrix_hash=case['matrix_hash'],
            archived_case=case['id'], archived_steps=case['training']['steps'],
            reference_hits=reference_hits, executed_hits=[r[0] for r in case['training']['hits']],
            old_c2=[{r['kind']:r['bound'] for r in old if r['target']==t
                     and r['epsilon']==.01 and r['kind'] in ['analytic','directional']} for t in TARGETS])
        dictionaries.append(dictionary)
        for degree in degrees:
            tick = time.monotonic()
            approximate, coefficients, remainder = p.interpolate(x, centers, gamma, degree)
            model = p.factor(approximate, y, eta)
            # Deliberately exposed sensitivity estimate, NOT an enclosure.
            allowance = (64*np.finfo(float).eps*(degree+1)*np.linalg.norm(approximate, 'fro')
                         +model['reconstruction_error']
                         +model['singular'][0]*model['orthogonality_error'])
            bounds = p.defect(j, approximate, model, eta, remainder, allowance)
            results = {method:[p.crossing_bracket(model, bounds, ti, method=method)
                for ti in range(len(TARGETS))] for method in ['analytic', 'action', 'combined']}
            # Inflate the arithmetic allowance tenfold to expose precision sensitivity.
            stress = dict(bounds)
            delta = remainder+10*allowance
            stress['analytic_kernel'] = (2*model['singular'][0]+delta)*delta
            stress['arithmetic_kernel'] = (2*model['singular'][0]+10*allowance)*10*allowance
            stressed = [p.crossing_bracket(model, stress, ti) for ti in range(len(TARGETS))]
            oracle_gaps, violations = [], []
            times = np.unique(np.r_[0, np.geomspace(1, max(reference_hits)*2, 100).astype(np.int64),
                                    reference_hits, np.maximum(0,np.array(reference_hits)-1)])
            for n in times:
                exact = reference.error(ref, int(n))
                low, high = p.band(model, bounds, int(n))
                oracle_gaps.append(float(np.max(np.abs(p.error(model, int(n))-exact))))
                if np.any(low > exact+1e-12) or np.any(high < exact-1e-12):
                    violations.append(int(n))
            folder = output/f'g{gamma}_D{degree}'
            core.save_arrays(folder/'model.npz', rates=model['rates'], weights=model['weights'],
                floor=model['floor'], singular=model['singular'], coefficients=coefficients,
                mode_action=bounds['mode_action'], null_action=bounds['null_action'],
                normalized_abs_loading=np.abs(model['loading']/model['norm']))
            row = dict(gamma=gamma, degree=degree, eta=eta,
                analytic_synthesis_remainder=float(remainder), arithmetic_allowance=float(allowance),
                measured_synthesis_frobenius=bounds['measured_synthesis_frobenius'],
                orthogonality_error=model['orthogonality_error'],
                reconstruction_error=model['reconstruction_error'],
                approximate_eta_L=float(model['rates'][0]),
                analytic_kernel_error=bounds['analytic_kernel'],
                arithmetic_kernel_error=bounds['arithmetic_kernel'],
                approximate_hits=reference.first_hit(model), results=results,
                arithmetic_stress_10x=stressed, reference_check_count=len(times)*len(TARGETS),
                reference_band_violations=violations, max_approximate_error=max(oracle_gaps),
                status='fp64_estimate_not_interval_certified', seconds=time.monotonic()-tick)
            rows.append(row)
            core.write_json(folder/'result.json', row)
            print(json.dumps({k:row[k] for k in ['gamma','degree','seconds']}),
                  'primary',results['analytic'][0],results['combined'][0], flush=True)
    tails = np.sqrt(np.cumsum(arrays['y_hat'][::-1]**2,axis=0)[::-1])/np.linalg.norm(y,axis=0)
    summary = dict(source_commit=commit, targets=TARGETS, gammas=list(gammas), degrees=list(degrees),
        epsilon=.01, n=512, samples=len(x), width=len(centers), target_hash=core.array_hash(y),
        grid_hash=core.array_hash(x), centers_hash=core.array_hash(centers),
        evidence_role='retrospective archived training comparison; no fitting to GD hits',
        gpu_hours=0, seconds=time.monotonic()-start, dictionaries=dictionaries, rows=rows,
        target_polynomial_tails=tails[1:257].tolist(),
        arithmetic_allowance_formula='64*eps*(D+1)*||Jtilde||_F + SVD reconstruction + smax*orthogonality',
        numerical_status='FP64 sensitivity estimates; separate endpoint audit required for certification')
    core.write_json(output/'summary.json', summary)
    plot(summary, output)
    return summary


def best(rows, gamma, target=0, method='combined', resolution_key='degree'):
    candidates = [(r,r['results'][method][target]) for r in rows if r['gamma']==gamma]
    low_row, low = max(candidates, key=lambda pair: pair[1]['necessary'])
    uppers = [(r,v) for r,v in candidates if v['sufficient'] is not None]
    high_row, high = min(uppers, key=lambda pair: pair[1]['sufficient']) if uppers else (None,None)
    return dict(necessary=low['necessary'], sufficient=high['sufficient'] if high else None,
        **{'lower_'+resolution_key:low_row[resolution_key],
           'upper_'+resolution_key:high_row[resolution_key] if high_row else None})


def plot(summary, output):
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1,3,figsize=(15,4.4), constrained_layout=True)
    gammas, rows = summary['gammas'], summary['rows']
    colors = plt.cm.viridis(np.linspace(.1,.85,len(gammas)))
    degrees = np.arange(1,257)
    for gamma,color in zip(gammas,colors):
        envelope = np.minimum(1,np.exp(core.log_feature_envelope(gamma,degrees)))
        axes[0].semilogy(degrees,envelope,color=color,label=rf'$\gamma={gamma}$')
    axes[0].semilogy(degrees,np.asarray(summary['target_polynomial_tails'])[:,0],
                     'k--',lw=2,label='target polynomial tail')
    axes[0].axhline(.01,color='.5',lw=.8,ls=':')
    axes[0].set(xlabel='Polynomial degree',ylabel='Relative target tail / feature envelope',
                ylim=(1e-12,2),title='A  Polynomial attenuation')
    axes[0].legend(fontsize=8)
    dictionaries = {r['gamma']:r for r in summary['dictionaries']}
    hits = [dictionaries[g]['executed_hits'][0] for g in gammas]
    axes[1].loglog(gammas,hits,'ko-',label='executed GD hit')
    axes[1].loglog(gammas,[dictionaries[g]['old_c2'][0]['directional'] for g in gammas],
                   's--',color='.5',label='old directional C2')
    for method,marker,color in [('analytic','v','tab:blue'),('combined','^','tab:orange')]:
        winners = [best(rows,g,method=method) for g in gammas]
        axes[1].loglog(gammas,[w['necessary'] for w in winners],marker+'--',color=color,
                       label=method+' necessary')
        axes[1].loglog(gammas,[w['sufficient'] if w['sufficient'] else np.nan for w in winners],
                       marker+':',color=color,alpha=.6,label=method+' sufficient')
    axes[1].set(xlabel='Common slope gamma',ylabel='Updates to 1% residual',
                title='B  Acquisition-time bounds')
    axes[1].legend(fontsize=7)
    for gamma,color in zip(gammas,colors):
        selected = [r for r in rows if r['gamma']==gamma]
        ratios = [r['results']['combined'][0]['necessary']/dictionaries[gamma]['executed_hits'][0]
                  for r in selected]
        axes[2].semilogx([r['degree'] for r in selected],ratios,'o-',color=color,
                        label=rf'$\gamma={gamma}$')
    axes[2].axhline(1,color='k',ls=':',lw=1)
    axes[2].set(xlabel='Retained polynomial degree D',ylabel='Necessary time / executed hit',
                ylim=(0,1.04),title='C  Tightness versus retained action')
    axes[2].legend(fontsize=8)
    fig.suptitle('Common-slope polynomial kernels: fixed centers, raw readout, zero start\n'
                 'Timing curves are FP64 bound evaluations with a sensitivity allowance',fontsize=12)
    fig.savefig(output/'three_panel.png',dpi=180)
    fig.savefig(output/'three_panel.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--gammas',type=int,nargs='+',default=GAMMAS)
    parser.add_argument('--degrees',type=int,nargs='+',default=DEGREES)
    args = parser.parse_args()
    run(args.root,args.output or args.root/'refinements/common_slope_polynomial',args.gammas,args.degrees)


if __name__ == '__main__':
    main()
