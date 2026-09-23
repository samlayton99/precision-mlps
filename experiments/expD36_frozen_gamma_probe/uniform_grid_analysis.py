"""Evaluate proved uniform-grid spectral constructions; emit evidence only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from . import core, common_slope_poly as poly, finite_gamma_gram as ref
from . import uniform_grid_spectrum as theory
from .gamma_filter import multiplier
from .pi_brief_figure import COLORS
from .review_figures import DEFAULT, GAMMAS, TARGETS


def targets(x):
    return np.column_stack((np.sin(2*np.pi*x)+.5*np.sin(6*np.pi*x)+.25*np.sin(10*np.pi*x),
        np.exp(np.sin(3*np.pi*x)), 1/(1+25*x*x), np.sqrt(5)*x*x,
        np.sqrt(2)*np.sin(2*np.pi*x)))


def collect(source, output, widths=(128, 256, 512)):
    sources = {}

    def read(path):
        sources[str(path.relative_to(source.parent.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    summary = json.loads(read(source/'summary.json').read_text())
    interval_audit = json.loads(read(source/'interval_audit.json').read_text())
    archive = np.load(read(source.parent.parent/'common/N512/arrays.npz'))
    assert summary['targets'] == TARGETS
    rows = []
    for n in widths:
        for gamma in GAMMAS:
            a = theory.construct(n, 16, int(np.ceil(np.sqrt(n))), gamma)
            x, centers = a['x'], a['centers']
            y = targets(x)/np.sqrt(len(x))
            exact = core.design(x, centers, gamma)
            if n == 512:
                np.testing.assert_array_equal(x, archive['x_train'])
                np.testing.assert_array_equal(centers, archive['centers'])
                np.testing.assert_allclose(y, archive['y_train'], atol=1e-14)
                y = archive['y_train']
                dictionary = next(d for d in summary['dictionaries'] if d['gamma'] == gamma)
                assert core.array_hash(exact) == dictionary['matrix_hash']
                eta = dictionary['eta']
                reference = dict(np.load(read(source/f'reference_g{gamma}.npz')))
                measured = dictionary['executed_hits']
            else:
                reference = ref.rectangular_forecast(exact, y)
                eta = .5/reference['L']
                measured = [None]*len(TARGETS)
            model = poly.factor(a['approximate'], y, eta)
            # Exposed arithmetic sensitivity model, not an interval enclosure.
            # The rigorous mathematical theorem uses a['feature_error'] alone.
            arithmetic = (64*np.finfo(float).eps*np.sqrt(len(centers)+1)
                          *(1+gamma*a['length']+np.log2(n))
                          +model['reconstruction_error']
                          +model['singular'][0]*model['orthogonality_error'])
            epsilon = a['feature_error']+arithmetic
            delta = (2*model['singular'][0]+epsilon)*epsilon
            bounds = dict(eta=eta, analytic_kernel=delta)
            intervals = [poly.crossing_bracket(model, bounds, t, method='analytic', cap=10**10)
                         for t in range(len(TARGETS))]
            stress_epsilon = a['feature_error']+10*arithmetic
            stress_delta = (2*model['singular'][0]+stress_epsilon)*stress_epsilon
            stress = [poly.crossing_bracket(model, dict(eta=eta, analytic_kernel=stress_delta),
                      t, method='analytic', cap=10**10) for t in range(len(TARGETS))]
            hits = ref.first_hit(reference)
            predicted = ref.first_hit(model)
            inherited = None
            if n == 512:
                certificate = next(r for r in interval_audit['results'] if r['gamma']==gamma)
                assert certificate['eta'] == eta
                certified = certificate['methods']['combined']
                assert certified['status'] == 'interval_certified_endpoints'
                assert intervals[0]['necessary'] <= certified['necessary']
                assert intervals[0]['sufficient'] >= certified['sufficient']
                inherited = dict(necessary=certified['necessary'], sufficient=certified['sufficient'],
                    status='New wider primary interval follows from existing Arb endpoints and monotonicity; this does not certify spectral values or the arithmetic allowance.')
            for hit, interval in zip(hits, intervals):
                assert interval['necessary'] <= hit
                assert interval['sufficient'] is None or hit <= interval['sufficient']
            original_mu = reference['rates']/eta
            estimated_mu = model['singular']**2
            assert np.max(np.abs(original_mu-estimated_mu)) <= delta
            rate_grid = np.geomspace(1e-10, .7, 401)
            buffered_lo, buffered_hi, actual_mass = [], [], []
            for cutoff in rate_grid:
                choices = [theory.buffered_mass(model, cutoff, cutoff*fraction, eta*delta)
                           for fraction in np.geomspace(1e-5, .5, 41)]
                low = np.max([c[0] for c in choices], axis=0)
                high = np.min([c[1] for c in choices], axis=0)
                mass = theory.slow_mass(reference, cutoff)
                assert np.all(low <= mass+1e-10) and np.all(high+1e-10 >= mass)
                buffered_lo.append(float(low[0])); buffered_hi.append(float(high[0]))
                actual_mass.append(float(mass[0]))
            cutoff = 1e-6
            choices = [theory.buffered_mass(model, cutoff, cutoff*f, eta*delta)
                       for f in np.geomspace(1e-5, .5, 201)]
            mass_lo = np.max([c[0] for c in choices], axis=0)
            mass_hi = np.min([c[1] for c in choices], axis=0)
            mass_true = theory.slow_mass(reference, cutoff)
            d, q, signed = theory.low_rank_gram(a)
            drop = 1e-12
            keep = np.abs(signed) > drop
            discarded = float(np.max(np.abs(signed[~keep]), initial=0))
            transformed_gram = theory.to_bulk_basis(
                theory.to_bulk_basis(a['approximate'].T@a['approximate'], a).conj().T, a).conj().T
            full_update = (q*signed)@q.conj().T
            gram_reconstruction = float(np.linalg.norm(transformed_gram-np.diag(d)-full_update, 'fro'))
            corrected = np.diag(d)+(q[:, keep]*signed[keep])@q[:, keep].conj().T
            corrected_mu = eigh((corrected+corrected.conj().T)/2, eigvals_only=True)[::-1]
            spectral_radius = delta+discarded+gram_reconstruction
            assert np.max(np.abs(corrected_mu-original_mu)) <= spectral_radius
            counts = []
            for rate in [1e-7, 1e-6, 1e-5]:
                count, pivot, dimension = theory.secular_count(d, q, signed, rate/eta, drop=drop)
                count_true = int(np.sum(reference['rates'] < rate))
                assert count == count_true
                counts.append(dict(rate=rate, secular_count=count, reference_count=count_true,
                                   minimum_pivot=pivot, dimension=dimension))
            grid_steps = np.unique(np.r_[0, np.geomspace(1, max(hits)*1.2, 120).astype(int), hits])
            curve_difference = max(float(np.max(np.abs(ref.error(reference,int(t))-poly.error(model,int(t))))) for t in grid_steps)
            row = dict(n=n, gamma=gamma, width=len(centers), samples=len(x), eta=eta,
                boundary=a['boundary'], image_terms=a['terms'], harmonics=a['harmonics'],
                feature_correction_rank=a['feature_correction_rank'],
                nominal_kernel_rank=2*a['feature_correction_rank'],
                retained_signed_rank=int(np.sum(keep)), compression_cutoff=drop,
                discarded_signed_norm=discarded, gram_reconstruction_error=gram_reconstruction,
                analytic_feature_error=a['feature_error'], analytic_kernel_error=a['analytic_kernel_error'],
                arithmetic_feature_allowance=float(arithmetic), total_kernel_allowance=float(delta),
                spectral_allowance=float(spectral_radius),
                measured_feature_frobenius=float(np.linalg.norm(exact-a['approximate'], 'fro')),
                intervals=intervals, stress_10x=stress, reference_hits=hits,
                predicted_hits=predicted, executed_hits=measured,
                inherited_primary_certificate=inherited,
                maximum_sampled_curve_discrepancy=curve_difference,
                rate26_true=float(reference['rates'][25]),
                rate26_lower=float(max(0,eta*(corrected_mu[25]-spectral_radius))),
                rate26_upper=float(eta*(corrected_mu[25]+spectral_radius)),
                true_slow_mass=mass_true.tolist(), slow_mass_lower=mass_lo.tolist(), slow_mass_upper=mass_hi.tolist(),
                secular_checks=counts, rate_cutoffs=rate_grid.tolist(), mass_curve=actual_mass,
                mass_curve_lower=buffered_lo, mass_curve_upper=buffered_hi)
            rows.append(row)
            np.savez_compressed(output/f'N{n}_g{gamma}.npz', rates=model['rates'], weights=model['weights'],
                floor=model['floor'], bulk_eigenvalues=d, corrected_eigenvalues=corrected_mu,
                original_eigenvalues=original_mu, correction_vectors=q[:,keep], correction_values=signed[keep])
            print(json.dumps({k:row[k] for k in ['n','gamma','retained_signed_rank','rate26_lower','rate26_true','rate26_upper']}), flush=True)
    scaling = []
    for n in [128,256,512]:
        h = 2/n
        for bandwidth in [.25,1.,4.]:
            gamma = bandwidth/h
            theta = np.pi/4
            scaling.append(dict(n=n, bandwidth=bandwidth, gamma=gamma,
                relative_frequency=theta, attenuation=float(multiplier(gamma, theta/h)**2)))
    return dict(targets=TARGETS, gammas=GAMMAS, rows=rows, width_scaling=scaling,
        source_sha256=sources, source_code_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['uniform_grid_analysis.py','uniform_grid_spectrum.py']},
        numerical_status='FP64 spectral diagnostics with a heuristic arithmetic allowance. Primary timing intervals contain existing independently certified Arb intervals. Spectral values and the arithmetic allowance are not interval certified.',
        evidence_role='Retrospective optimization verification, five archived targets; new width cases use independent spectral references, not executed GD.',
        selection='B and p minimize analytic correction rank over B=1..N/2-1,p=1..256, targeting analytic kernel tail1e-15. Signed correction cutoff1e-12. Buffers chosen from declared geometric grids without consulting true mass.',
        gpu_hours=0)


def figures(data, output):
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],
        'mathtext.fontset':'stix','font.size':9,'axes.titlesize':10,'axes.titlelocation':'left',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42})
    rows = [r for r in data['rows'] if r['n']==512]
    fig, axes = plt.subplots(1,3,figsize=(8.2,3.4),layout='constrained')
    omega = np.linspace(.01,80,400)
    for row,color in zip(rows,COLORS):
        axes[0].semilogy(omega,multiplier(row['gamma'],omega)**2,color=color,label=rf'$\gamma={row["gamma"]}$')
    axes[0].set(title='A  Exact periodic mechanism',xlabel=r'Angular frequency $\omega$',
                ylabel='Eigenvalue / step-reference eigenvalue',ylim=(1e-10,1.3))
    axes[0].legend(frameon=False,fontsize=7.5,loc='lower left')
    gammas = [r['gamma'] for r in rows]
    low = np.array([r['rate26_lower'] for r in rows]); high = np.array([r['rate26_upper'] for r in rows])
    axes[1].loglog(gammas,(low+high)/2,color='#273746',label='Boundary-corrected prediction')
    axes[1].fill_between(gammas,low,high,color='#9aafbc',alpha=.5,label='Spectral enclosure')
    for r,c in zip(rows,COLORS):
        axes[1].scatter(r['gamma'],r['rate26_true'],facecolors='white',edgecolors=c,zorder=3)
    axes[1].set(title='B  Ordinary tanh spectrum',xlabel=r'Common slope $\gamma$',ylabel='26th normalized eigenvalue')
    axes[1].set_xticks(gammas,labels=list(map(str,gammas))); axes[1].minorticks_off()
    axes[1].legend(frameon=False,fontsize=6.6,loc='lower right')
    axes[1].text(.03,.97,'Circles: original tanh\nBounds are thinner than the line.',transform=axes[1].transAxes,va='top',fontsize=7)
    lo = np.array([r['intervals'][0]['necessary'] for r in rows]); hi = np.array([r['intervals'][0]['sufficient'] for r in rows])
    axes[2].loglog(gammas,(lo+hi)/2,color='#273746',label='Predicted acquisition time')
    axes[2].fill_between(gammas,lo,hi,color='#9aafbc',alpha=.5,label='Necessary–sufficient interval')
    for r,c in zip(rows,COLORS):
        axes[2].scatter(r['gamma'],r['executed_hits'][0],facecolors='white',edgecolors=c,zorder=3)
    axes[2].set(title='C  Target acquisition',xlabel=r'Common slope $\gamma$',ylabel='GD updates to 1% residual')
    axes[2].set_xticks(gammas,labels=list(map(str,gammas))); axes[2].minorticks_off()
    axes[2].legend(frameon=False,fontsize=6.6,loc='upper right')
    axes[2].text(.03,.06,'Circles: executed GD\nSine-mixture target',transform=axes[2].transAxes,fontsize=7)
    for ax in axes: ax.tick_params(length=3)
    fig.savefig(output/'uniform_grid_three_panel.png',dpi=300)
    fig.savefig(output/'uniform_grid_three_panel.pdf',metadata={'CreationDate':None}); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(7.4,3.3),layout='constrained')
    for row,color in zip(rows,COLORS):
        grid=np.array(row['rate_cutoffs']); low=np.array(row['mass_curve_lower']); high=np.array(row['mass_curve_upper'])
        axes[0].loglog(grid,row['mass_curve'],color=color,label=rf'$\gamma={row["gamma"]}$')
        axes[0].fill_between(grid,np.maximum(low,1e-12),high,color=color,alpha=.17)
        z=np.load(output/f'N512_g{row["gamma"]}.npz')
        mu=z['original_eigenvalues']; corrected=z['corrected_eigenvalues']; radius=row['spectral_allowance']
        indices=np.arange(1,len(mu)+1); valid=mu>radius
        axes[1].semilogy(indices[valid],2*radius/mu[valid],color=color)
    axes[0].set(title='A  Target-mass bounds and original spectrum',xlabel='Normalized eigenvalue cutoff',
                ylabel='Fraction of target energy',xlim=(1e-8,.1),ylim=(1e-7,1.2))
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].set(title='B  Relative spectral enclosure width',xlabel='Ordered eigenvalue index',
                ylabel='Interval width / original eigenvalue',ylim=(1e-12,2))
    axes[1].axhline(1,color='.6',ls=':',lw=.7)
    fig.savefig(output/'uniform_grid_spectral_detail.png',dpi=300)
    fig.savefig(output/'uniform_grid_spectral_detail.pdf',metadata={'CreationDate':None}); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path,default=DEFAULT.parent/'uniform_grid_spectrum')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    data=collect(args.source,args.output)
    (args.output/'summary.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    figures(data,args.output)


if __name__=='__main__':
    main()
