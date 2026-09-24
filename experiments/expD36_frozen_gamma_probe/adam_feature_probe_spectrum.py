"""Target-weighted frozen GD kernel geometry; not an Adam rate prediction."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CUTOFFS = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])


def spectrum(phi, target):
    """Thin SVD includes every resolved feature direction and target complement."""
    with threadpool_limits(limits=2):
        u, s, _ = scipy.linalg.svd(phi/np.sqrt(len(target)), full_matrices=False,
                                  check_finite=True, lapack_driver='gesdd')
    norm2 = np.dot(target, target)
    mu = s*s
    rho = mu/mu[0]
    weights = np.square(u.T@target)/norm2
    complement = 1-float(np.sum(weights))
    if complement < -2e-12:
        raise AssertionError(f'Target projection energy exceeds total: {complement}')
    closure = float(np.sum(np.square(target-u@(u.T@target)))/norm2 + np.sum(weights))
    np.testing.assert_allclose(closure, 1., rtol=0, atol=2e-12)
    direct = float(np.sum(np.square(phi.T@target))/(len(target)*norm2))
    spectral = float(np.dot(mu, weights))
    np.testing.assert_allclose(spectral, direct, rtol=1e-10, atol=2e-13*max(1.,mu[0]))
    return mu, rho, weights, dict(target_complement=max(0.,complement), energy_closure=closure,
                                  direct_quadratic_form=direct,spectral_quadratic_form=spectral,
                                  relative_quadratic_form_difference=abs(spectral-direct)/max(abs(direct),np.finfo(float).tiny))


def slow_energy(rho, weights, cutoffs):
    # This includes omitted sample-space directions without assigning tiny
    # numerical singular vectors a meaning. Clamp only roundoff in [0,1].
    return np.clip(1-np.array([np.sum(weights[rho>c]) for c in cutoffs]),0.,1.)


def self_test():
    rng=np.random.default_rng(81)
    q,_=np.linalg.qr(rng.normal(size=(13,13)))
    phi=np.sqrt(13)*q[:,:4]*np.array([3.,1.,.1,.001])
    target=.5*q[:,0]+.4*q[:,2]+.7*q[:,8]
    mu,rho,w,checks=spectrum(phi,target)
    np.testing.assert_allclose(mu,[9,1,.01,1e-6],rtol=1e-12)
    norm2=.25+.16+.49
    np.testing.assert_allclose(slow_energy(rho,w,[.1,.0001]),[(.16+.49)/norm2,.49/norm2],atol=2e-13)
    assert abs(checks['target_complement']-.49/norm2)<2e-13
    print('Known-spectrum, slow-energy, complement and quadratic-form tests passed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path);parser.add_argument('--manifest',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        self_test()
        if args.input is None:return
    if not all([args.input,args.manifest,args.output]):parser.error('Required: --input --manifest --output')
    manifest=json.loads(args.manifest.read_text())
    with np.load(args.input) as data:phi=np.asarray(data['features']);y=np.asarray(data['target'])
    rows=manifest['geometries']
    if len(rows)!=len(phi):raise ValueError('Manifest geometry count differs')
    args.output.mkdir(parents=True,exist_ok=True)
    thresholds=np.logspace(-14,0,281)
    mus=[];rhos=[];weights=[];cdfs=[];summaries=[]
    t0=time.perf_counter()
    for i,row in enumerate(rows):
        tick=time.perf_counter();mu,rho,w,checks=spectrum(phi[i],y)
        mus.append(mu);rhos.append(rho);weights.append(w);cdfs.append(slow_energy(rho,w,thresholds))
        summaries.append(dict(**row,**checks,mu_max=float(mu[0]),
                              slow_energy={f'{c:g}':float(e) for c,e in zip(CUTOFFS,slow_energy(rho,w,CUTOFFS))}))
        print(json.dumps(dict(geometry=row['name'],seconds=time.perf_counter()-tick,checks=checks)),flush=True)
    cdfs=np.asarray(cdfs)
    np.savez_compressed(args.output/'spectrum.npz',mu=np.asarray(mus),relative_rate=np.asarray(rhos),
                        target_weights=np.asarray(weights),cutoffs=thresholds,slow_energy=cdfs)
    summary=dict(description=__doc__,input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                 manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                 kernel='Phi Phi^T / samples, including output bias, raw parameter coordinates',
                 rate='mu_i / mu_max; normalized GD rate, not Adam rate',
                 weight='(u_i^T target)^2 / ||target||^2',
                 slow_energy_formula='1 - sum(weights[relative_rate > cutoff])',
                 unresolved_directions='Includes target complement; ratios near machine resolution do not identify individual directions reliably',
                 seconds=time.perf_counter()-t0,geometries=summaries)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.0),sharex=True,sharey=True,layout='constrained')
    uniform=[i for i,r in enumerate(rows) if r['family']=='uniform']
    for i,color in zip(uniform,plt.cm.viridis(np.linspace(.08,.92,len(uniform)))):
        axes[0].plot(thresholds,cdfs[i],color=color,lw=1.8,label=f"λ = {rows[i]['lambda_rms']:g}")
    stages=sorted({r['snapshot_step'] for r in rows if r['family']=='learned'})
    for stage,color in zip(stages,plt.cm.plasma(np.linspace(.12,.8,len(stages)))):
        ids=[i for i,r in enumerate(rows) if r['family']=='learned' and r['snapshot_step']==stage]
        arr=cdfs[ids];label='Initialization' if stage==0 else f'{stage//1000}k joint updates'
        axes[1].plot(thresholds,np.median(arr,axis=0),color=color,lw=1.8,label=label)
        axes[1].fill_between(thresholds,np.min(arr,axis=0),np.max(arr,axis=0),color=color,alpha=.12,lw=0)
    references=[i for i in uniform if np.isclose(rows[i]['lambda_rms'],.25)]
    if references:axes[1].plot(thresholds,cdfs[references[0]],color='black',lw=1.4,ls='--',label='Uniform λ = 0.25')
    axes[0].set_title('Uniform-center slope changes');axes[1].set_title('Features acquired by joint training')
    for ax in axes:
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(1e-14,1);ax.set_ylim(1e-10,1.2)
        ax.axvline(1e-6,color='#888888',ls=':',lw=.8)
        ax.set_xlabel('Relative kernel eigenvalue cutoff  μ / μmax')
        ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8.5,loc='lower right')
    axes[0].set_ylabel('Fraction of target energy below cutoff')
    fig.suptitle('Target energy in slow kernel directions',fontsize=12)
    fig.savefig(args.output/'target_weighted_spectrum.png',dpi=220)
    svg_path=args.output/'target_weighted_spectrum.svg'
    fig.savefig(svg_path);plt.close(fig)
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')


if __name__=='__main__':main()
