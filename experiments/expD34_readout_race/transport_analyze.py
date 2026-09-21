"""Export transport evidence, comparisons, and plots; never author a report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import transport as tr
from .recovery import acquisition_distance, clean, write_table


def observables(z, d, x, y, w, width):
    gamma = abs(z[0])
    r = np.tanh(x[:, None]*z[0]+z[1]) @ (width*w*z[2])+d-y
    return dict(half_mse=float(np.mean(r*r)/2), mean_gamma=float(w @ gamma),
                max_gamma=float(gamma.max()), readout_l2=float(np.sqrt(width*(w @ z[2]**2))),
                gamma_second_moment=float(w @ gamma**2),
                **{f"fraction_{threshold:g}": float(w @ (gamma >= threshold)) for threshold in (1,3.2,16)})


def weighted_distance(a, w, width, threshold, population):
    """Relaxed population distance for a continuum/quadrature measure.

    Quadrature masses may be split; empirical finite counts instead use ceil(pW).
    """
    gap = np.maximum(threshold-abs(a), 0)
    order = np.argsort(gap)
    masses = np.minimum(w[order], np.maximum(population-np.r_[0., np.cumsum(w[order])[:-1]], 0))
    return float(np.sqrt(width*np.sum(masses*gap[order]**2)))


def marginal_w2(a, wa, b, wb):
    """Exact one-dimensional weighted quantile coupling, up to roundoff."""
    ia,ib=np.argsort(a),np.argsort(b)
    ca,cb=np.cumsum(wa[ia]),np.cumsum(wb[ib])
    if abs(ca[-1]-1)>1e-12 or abs(cb[-1]-1)>1e-12:
        raise ValueError('Both distributions must have unit mass')
    ca[-1]=cb[-1]=1.
    edges=np.unique(np.r_[0.,ca,cb]); mid=(edges[1:]+edges[:-1])/2
    gaps=a[ia][np.searchsorted(ca,mid)]-b[ib][np.searchsorted(cb,mid)]
    return float(np.sqrt(np.diff(edges) @ (gaps*gaps)))


def frozen_forecast(z, d, x, y, width, horizon, degree=65, kappa=1.):
    """Initial-kernel linearization in physical coordinates, no future input."""
    w = np.ones(width)/width
    _, arrays = tr.modal_diagnostics(z,d,x,y,w,width,degree,kappa)
    K = sum(arrays['K_'+k] for k in ('a','b','c','d'))
    lam, vectors = np.linalg.eigh(K)
    lam = np.maximum(lam, 0.)
    integrated = np.full_like(lam, horizon)
    np.divide(-np.expm1(-horizon*lam), lam, out=integrated, where=lam>1e-15)
    e = arrays['residual_modes']
    integral = vectors @ (integrated*(vectors.T @ e))
    zn = z-np.stack([arrays['J_'+key].T @ integral*rate for key, rate in [('a',1),('b',1),('c',kappa)]])
    dn = d-kappa*arrays['J_d'] @ integral
    en = vectors @ (np.exp(-horizon*lam)*(vectors.T @ e))
    return zn, dn, float(en @ en/2)


def identity(case):
    return tuple(case[k] for k in ('width','seed','target','kappa','eta'))


def diagnostics(z,d,x,y,w,width,kappa=1.,eta=None):
    scalar, arrays = tr.modal_diagnostics(z,d,x,y,w,width,65,kappa)
    e, ja = arrays['residual_modes'], arrays['J_a']
    u=x[:,None]*z[0]+z[1]; h=np.tanh(u)
    exp=np.exp(-2*abs(u)); s=4*exp/(1+exp)**2
    r=h @ (width*w*z[2])+d-y
    full=np.sqrt(width*w)*z[2]*(x @ (r[:,None]*s))/len(x)
    K=sum(arrays['K_'+key] for key in ('a','b','c','d'))
    for key in ('a','b','c','d'):
        scalar[f'modal_{key}_share']=float(e @ arrays['K_'+key] @ e/(e @ K @ e+1e-300))
    if scalar['coarse_inverse_resolved']:
        values,vectors=np.linalg.eigh(K[:2,:2])
        inverse_root=(vectors/np.sqrt(values)) @ vectors.T
        relative_drift=inverse_root @ arrays['K_dot'][:2,:2] @ inverse_root
        scalar['tracking_metric_decay_lower']=float(values[0]-.5*np.linalg.eigvalsh(relative_drift)[-1])
        scalar['coarse_max_eigenvalue']=float(values[-1])
        B=np.linalg.solve(K[:2,:2],K[:2,2:])
        S=K[2:,2:]-K[2:,:2] @ B
        effective=ja[2:].T-ja[:2].T @ B
        effective_blocks=[]
        for key,rate in [('a',1.),('b',1.),('c',kappa),('d',kappa)]:
            j=np.atleast_2d(arrays['J_'+key]).T if key=='d' else arrays['J_'+key]
            block=j[2:].T-j[:2].T @ B
            gram=rate*(block.T @ block)
            effective_blocks.append(gram)
            scalar[f'effective_{key}_share']=float(e[2:] @ gram @ e[2:]/(e[2:] @ S @ e[2:]+1e-300))
        scalar['effective_kernel_accounting_error']=float(np.linalg.norm(sum(effective_blocks)-S))
        if eta is not None:
            gradient=np.stack([z[2]*(x @ (r[:,None]*s))/len(x),z[2]*(r @ s)/len(x),h.T @ r/len(x)])
            zn=z-eta*np.array([1.,1.,kappa])[:,None]*gradient; dn=d-eta*kappa*r.mean()
            nxt_scalar,nxt=tr.modal_diagnostics(zn,dn,x,y,w,width,65,kappa)
            if nxt_scalar['coarse_inverse_resolved']:
                Kn=sum(nxt['K_'+key] for key in 'abcd'); Cn=Kn[:2,:2]
                Bn=np.linalg.solve(Cn,Kn[:2,2:]); en=nxt['residual_modes']
                defect=en-e+eta*K @ e
                transition=np.eye(2)-eta*(K[:2,:2]+Bn @ K[2:,:2])
                forcing_step=(Bn-B-eta*Bn @ S) @ e[2:]+defect[:2]+Bn @ defect[2:]
                vn,un=np.linalg.eigh(Cn); root_next=(un*np.sqrt(vn)) @ un.T
                factor=np.linalg.norm(root_next @ transition @ inverse_root,2)
                identity_error=np.linalg.norm(en[:2]+Bn @ en[2:]-transition @ (e[:2]+B @ e[2:])-forcing_step)
                scalar.update(discrete_tracking_factor=float(factor),discrete_tracking_identity_error=float(identity_error),
                    discrete_tracking_forcing_per_time=float(np.linalg.norm(root_next @ forcing_step)/eta),
                    modal_step_defect_norm=float(np.linalg.norm(defect)))
        Bdot=np.linalg.solve(K[:2,:2],arrays['K_dot'][:2,2:]-arrays['K_dot'][:2,:2] @ B)
        force=(Bdot-B @ S) @ e[2:]
        omitted=arrays['residual_velocity']+K @ e
        force+=omitted[:2]+B @ omitted[2:]
        scalar['tracking_forcing_norm']=float(np.linalg.norm(force))
        scalar['B_drift_norm']=float(np.linalg.norm(Bdot))
        scalar['effective_outward_alignment']=float(-(np.sqrt(w)*np.sign(z[0])) @ (effective @ e[2:])/(np.linalg.norm(effective @ e[2:])+1e-300))
        for mode in (3,9):
            scalar[f'residual_mode{mode}']=float(e[mode])
            scalar[f'effective_mode{mode}_along_full']=float(full @ (effective[:,mode-2]*e[mode])/(full @ full+1e-300))
    for degree in (9,17,33,65):
        scalar[f'projection{degree}_relative_error'] = float(np.linalg.norm(full-ja[:degree+1].T @ e[:degree+1])/(np.linalg.norm(full)+1e-300))
    return scalar, arrays


def analyze(runs, output, archive=None):
    output.mkdir(parents=True, exist_ok=True)
    loaded, truth = [], {}
    actual_kernels=[]
    if archive:
        f = dict(np.load(archive))
        for i, case in enumerate(json.loads(str(f['cases']))):
            # The verified replay is fixed to equal-rate W177, eta=.002.
            if isinstance(case, list):
                seed, target = case
            else:
                seed, target = case['seed'], case['target']
            truth[(177,seed,target,1.,.002)] = (f, i)
            for step in (0,2000,20000,100000,600000):
                j=int(np.flatnonzero(f['steps']==step)[0])
                scalar,_=diagnostics(f['z'][i,j],f['d'][i,j],f['x'],f['y'][i],np.ones(177)/177,177,eta=.002)
                actual_kernels.append(dict(run='verified_actual',seed=seed,target=target,step=step,**scalar))
    for root in runs:
        if not (root/'states.npz').exists():
            continue
        config = json.loads((root/'manifest.json').read_text())
        state = dict(np.load(root/'states.npz'))
        curve = dict(np.load(root/'curves.npz'))
        status = json.loads((root/'status.json').read_text())
        loaded.append((root, config, state, curve, status))
        for i, case in enumerate(config['cases']):
            if case['degree']==-1 and not case['nodes']:
                truth[identity(case)] = (state,i)
    endpoints, comparisons, kernels, budgets, frozen = [], [], [], [], []
    kernel_arrays = {}
    for root, config, f, curve, status in loaded:
        for i, case in enumerate(config['cases']):
            meta = dict(run=root.name, **case)
            z, d, w, x, y = f['z'][i], f['d'][i], f['weights'][i], f['x'], f['y'][i]
            valid = np.all(np.isfinite(z[-1])) and np.isfinite(d[-1])
            end = dict(**meta, complete=status['complete'], finite=bool(valid), step=int(f['steps'][-1]))
            if not valid:
                endpoints.append(end)
                continue
            end.update(observables(z[-1],d[-1],x,y,w,case['width']))
            invariant = lambda a: case['width']*(w @ (a[0]**2+a[1]**2-a[2]**2/case['kappa']))
            end.update(balance_change=float(invariant(z[-1])-invariant(z[0])),
                balance_error=float(invariant(z[-1])-invariant(z[0])-f['balance_flow'][i,-1]-f['balance_discrete'][i,-1]),
                positive_negative_error=float(np.max(abs(f['positive'][i,-1]-f['negative'][i,-1]-(abs(z[-1,0])-abs(z[0,0]))))),
                path=float(f['path'][i,-1]), integrated_slope_energy=float(f['energy'][i,-1,0]),
                energy_loss_discrepancy=float(f['energy'][i,-1].sum()-(observables(z[0],d[0],x,y,w,case['width'])['half_mse']-end['half_mse'])))
            endpoints.append(end)
            for j, step in enumerate(f['steps']):
                if step*case['eta'] not in (0.,4.,40.,200.,1200.):
                    continue
                # Full modal diagnostics at the actual/forecast state, not a fitted closure.
                scalar, arrays = diagnostics(z[j],d[j],x,y,w,case['width'],case['kappa'],eta=None if case['nodes'] else case['eta'])
                kernels.append(dict(**meta,step=int(step),**scalar))
                # Keep primary matrices; all other matrices are reproducible from saved states.
                if step in (0,600000) and root.name in ('primary33','freshfull'):
                    for key in ('K_a','K_b','K_c','K_d','residual_modes','K_dot','residual_velocity'):
                        kernel_arrays[f'{root.name}_{i}_{step}_{key}'] = arrays[key]
                pair = truth.get(identity(case))
                if pair:
                    reference, ri = pair
                    indices = np.flatnonzero(reference['steps']==step)
                    if len(indices):
                        rj = indices[0]; rz,rd = reference['z'][ri,rj],reference['d'][ri,rj]
                        ref = observables(rz,rd,x,y,np.ones(case['width'])/case['width'],case['width'])
                        predicted = observables(z[j],d[j],x,y,w,case['width'])
                        comparisons.append(dict(**meta,step=int(step),
                            parameter_error=float(np.sqrt(np.sum((z[j]-rz)**2)+(d[j]-rd)**2)),
                            slope_error=float(np.linalg.norm(z[j,0]-rz[0])),
                            **{key+'_error': predicted[key]-ref[key] for key in ref}))
            for start,endtime in ((0.,40.),(40.,200.),(200.,1200.),(40.,1200.)):
                si=np.flatnonzero(f['steps']*case['eta']==start); ei=np.flatnonzero(f['steps']*case['eta']==endtime)
                if not len(si) or not len(ei): continue
                s,t=si[0],ei[0]
                path=float(f['path'][i,t]-f['path'][i,s])
                ea=float(f['energy'][i,t,0]-f['energy'][i,s,0])
                et=float(np.sum(f['energy'][i,t]-f['energy'][i,s]))
                for threshold in (1.,3.2,16.):
                    for population in (.1,.5):
                        distance=(weighted_distance(z[s,0],w,case['width'],threshold,population) if case['nodes'] else
                                  acquisition_distance(z[s,0],threshold,population))
                        budgets.append(dict(**meta,start=start,end=endtime,threshold=threshold,population=population,
                            distance=distance,path=path,energy_bound=np.sqrt((endtime-start)*ea),slope_energy_share=ea/et if et else np.nan,
                            path_excludes=path<distance,energy_excludes=(endtime-start)*ea<distance**2))
            if not case['nodes'] and case['degree']==33:
                zn,dn,linear_loss=frozen_forecast(z[0],d[0],x,y,case['width'],config['end_time'],kappa=case['kappa'])
                frozen.append(dict(**meta,linearized_projected_loss=linear_loss,
                    **observables(zn,dn,x,y,w,case['width'])))
    refinements=[]
    for run,ref in [('degree9','primary33'),('degree17','primary33'),('degree65','primary33'),
                    ('halfstep','primary33'),('law8','law12'),('law12','law16'),('law16','law24'),('law24','law32'),('law12full','law12'),('law32full','law32'),('law12half','law12')]:
        for row in [r for r in endpoints if r['run']==run and r['complete'] and r['finite']]:
            candidates=[r for r in endpoints if r['run']==ref and r['complete'] and r['finite'] and r['seed']==row['seed'] and r['target']==row['target']]
            if not candidates: continue
            reference=candidates[0]
            refinements.append(dict(run=run,reference=ref,seed=row['seed'],target=row['target'],
                **{key+'_difference':row[key]-reference[key] for key in ('half_mse','mean_gamma','max_gamma','readout_l2','path','fraction_1','fraction_3.2','fraction_16')}))
    law_errors=[]
    for root,config,f,curve,status in loaded:
        if not config['cases'][0]['nodes'] or not status['complete']: continue
        for i,case in enumerate(config['cases']):
            for key,(ref,ri) in truth.items():
                if key[0]!=case['width'] or key[2]!=case['target'] or key[3:]!=(case['kappa'],case['eta']): continue
                indices=np.flatnonzero(ref['steps']==round(config['end_time']/case['eta']))
                if not len(indices):continue
                rj=indices[0]
                law_errors.append(dict(run=root.name,target=case['target'],seed=key[1],physical_time=config['end_time'],
                    gamma_wasserstein2=marginal_w2(abs(f['z'][i,-1,0]),f['weights'][i],abs(ref['z'][ri,rj,0]),np.ones(case['width'])/case['width'])))
    for name, rows in [('endpoints',endpoints),('paired_errors',comparisons),('refinement',refinements),('law_errors',law_errors),('kernels',kernels),('actual_kernels',actual_kernels),('budgets',budgets),('frozen_kernel',frozen)]:
        write_table(output/f'{name}.csv',rows)
    np.savez_compressed(output/'kernel_blocks.npz',**kernel_arrays)
    validation=[]
    for row in comparisons:
        if row['run']!='fresh33' or row['step']!=600000: continue
        ref=next((r for r in endpoints if r['run']=='freshfull' and r['target']==row['target'] and r['seed']==row['seed'] and r['complete'] and r['finite']),None)
        if ref is None: continue
        errors={k:abs(row[k+'_error'])/ref[k] for k in ('mean_gamma','readout_l2')}
        errors['half_mse']=abs(row['half_mse_error'])
        errors['population_count']=max(abs(row[f'fraction_{g:g}_error'])*row['width'] for g in (1.,3.2,16.))
        classification=all((ref[f'fraction_{g:g}']>=.1)==(ref[f'fraction_{g:g}']+row[f'fraction_{g:g}_error']>=.1) for g in (1.,3.2,16.))
        passed=errors['mean_gamma']<=.01 and errors['readout_l2']<=.01 and errors['half_mse']<=.001 and errors['population_count']<=1+1e-12 and classification
        validation.append(dict(seed=row['seed'],target=row['target'],errors=errors,barrier_agreement=classification,passed=passed))
    (output/'fresh_validation.json').write_text(json.dumps(clean(dict(expected_cases=15,evaluated_cases=len(validation),
        all_passed=len(validation)==15 and all(v['passed'] for v in validation),cases=validation)),indent=2)+'\n')
    (output/'analysis_manifest.json').write_text(json.dumps(clean(dict(runs=[str(r) for r in runs],archive=str(archive),
        endpoint_count=len(endpoints),paired_count=len(comparisons),kernel_count=len(kernels))),indent=2)+'\n')
    plot(loaded, output)
    plot_refinements(endpoints,kernels,output)
    if actual_kernels:
        plot_actual(actual_kernels,output)
    print(json.dumps(dict(endpoints=len(endpoints),paired=len(comparisons),kernels=len(kernels))))


def plot(loaded, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,3,figsize=(12,9),sharex=True)
    for root, config, _, curve, _ in loaded:
        if root.name not in ('primary33','law12','law32','freshfull'): continue
        law='law' in root.name
        color={'primary33':'#6b7280','law12':'#10b981','law32':'#111827','freshfull':'#dc2626'}[root.name]
        for i, case in enumerate(config['cases']):
            row=('sine','moment3','moment9').index(case['target'])
            for col,key in enumerate(('mean_gamma','xi','readout_l2')):
                v=curve['trace'][i,:,config['trace_columns'].index(key)]
                axes[row,col].plot(curve['steps']*case['eta'],v,color=color,alpha=.85 if law else .4,lw=1.5 if law else .7,
                    label=root.name if case['target']=='sine' and (law or case['seed']==config['cases'][0]['seed']) else None)
                axes[row,col].set(yscale='log',ylabel=f"{case['target']}: {key}")
                axes[row,col].set_xscale('symlog',linthresh=1)
    for ax in axes[-1]: ax.set_xlabel('Physical time')
    if axes[0,0].get_legend_handles_labels()[0]:
        axes[0,0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output/'transport_comparison.png',dpi=160); plt.close(fig)


def plot_actual(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
    for col,target in enumerate(('sine','moment3','moment9')):
        for seed in range(5):
            rr=[r for r in rows if r['target']==target and r['seed']==seed]
            time=np.array([r['step']*.002 for r in rr])
            for key,label,color,style in [('full_slope_norm','full slope force','#111827','-'),
                    ('effective_slope_norm','effective fine-mode force','#2563eb','--'),
                    ('transient_slope_norm','coarse tracking remainder','#d97706',':')]:
                axes[0,col].plot(time,[r[key] for r in rr],style,color=color,alpha=.6,label=label if seed==0 else None)
            axes[1,col].plot(time,[r['modal_slope_share'] for r in rr],color='#7c3aed',alpha=.5)
        axes[0,col].set(title=target,yscale='log',ylabel='Slope-force norm')
        axes[1,col].set(ylabel='Slope share of dissipation',ylim=(0,1),xlabel='Physical time')
        for ax in axes[:,col]: ax.set_xscale('symlog',linthresh=1)
    axes[0,0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output/'actual_effective_force.png',dpi=160); plt.close(fig)


def plot_refinements(endpoints,kernels,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    targets=('sine','moment3','moment9')
    laws=[r for r in endpoints if r['run'] in ('law8','law12','law16','law24','law32') and r['complete'] and r['finite']]
    if laws:
        fig,axes=plt.subplots(3,3,figsize=(11,8),sharex=True)
        for col,target in enumerate(targets):
            rr=sorted([r for r in laws if r['target']==target],key=lambda r:r['nodes'])
            for row,key in enumerate(('mean_gamma','readout_l2','fraction_3.2')):
                axes[row,col].plot([r['nodes'] for r in rr],[r[key] for r in rr],'o-',color='#2563eb')
                axes[row,col].set(ylabel=key)
            axes[0,col].set_title(target); axes[-1,col].set_xlabel('Quadrature order per dimension')
        fig.tight_layout();fig.savefig(output/'law_refinement.png',dpi=160);plt.close(fig)
    rows=[r for r in kernels if r['run'] in ('primary33','width89full','width353full') and r['seed']<3 and r['step']==20000]
    if rows:
        fig,axes=plt.subplots(1,3,figsize=(11,3.5))
        for ax,target,power in zip(axes,targets,(1,1,2)):
            for seed in range(3):
                rr=sorted([r for r in rows if r['target']==target and r['seed']==seed],key=lambda r:r['width'])
                ax.loglog([r['width'] for r in rr],[r['full_slope_norm'] for r in rr],'o-',alpha=.5,color='#6b7280')
            anchor=np.median([r['full_slope_norm'] for r in rows if r['target']==target and r['width']==177])
            width=np.array([89,177,353])
            ax.loglog(width,anchor*(177/width)**power,'--',color='#2563eb',label=f'W^(-{power}), anchored at W=177')
            law=sorted([r for r in kernels if r['run'] in ('law8','law_width89_early','law_width353_early') and r['target']==target and r['step']==20000],key=lambda r:r['width'])
            if law:
                ax.loglog([r['width'] for r in law],[r['full_slope_norm'] for r in law],'s-',color='#111827',label='Initialization law, quadrature order 8')
            ax.set(title=target,xlabel='Physical width',ylabel='Slope-force norm at t=40')
            ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(output/'width_scaling.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.runs,args.output,args.archive)


if __name__=='__main__': main()
