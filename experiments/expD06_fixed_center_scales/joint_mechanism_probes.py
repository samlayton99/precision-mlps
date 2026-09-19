"""Post-hoc fixed-dictionary and damping probes; never advance training states."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag, eigh, eigvalsh, lstsq, svd

from . import core, diagnostics, higher_order, joint_analysis as analysis, joint_conditioning as campaign, run


def native_hessians(g,c,gamma,coordinates,samples=16):
    """Exact half-MSE Hessian, GN matrix, and gradient in the trained coordinates."""
    x,y,a,r,j=analysis.linearize(g,c,gamma,samples)
    jac=np.column_stack((analysis.native_features(a,g,coordinates),j));gn=jac.T@jac
    distance=(x[:,None]-g.centers)/g.h;argument=distance*(g.h*gamma)
    e=np.exp(-2*np.abs(argument));sech=4*e/(1+e)**2
    derivative=distance*sech/np.sqrt(len(x))
    mixed=derivative.T@r
    second=c[1:]*((-2*distance**2*np.tanh(argument)*sech/np.sqrt(len(x))).T@r)
    transform=higher_order.readout_map(g,coordinates);split=g.width+1
    correction=transform[1:].T*mixed
    hessian=gn.copy();hessian[:split,split:]+=correction;hessian[split:,:split]+=correction.T
    hessian[split:,split:]+=np.diag(second)
    return hessian,gn,jac.T@r


def stability_case(task):
    source,record=task;case=record['case'];g=core.geometry(case['n']);folder=source/record['key'];end=record['end']
    with np.load(folder/f'dense_parameters_{end}.npz') as a:
        states=[(int(a['step'][i]),a['c'][i],a['gamma'][i]) for i in (-2,-1)]
    path=folder/f'spectrum_{end}.npz'
    with np.load(path) as a:states.append((end,a['c'],a['gamma']))
    rows=[]
    for step,c,gamma in states:
        hessian,gn,gradient=native_hessians(g,c,gamma,case['coordinates'])
        maximum,vector=eigh(hessian,subset_by_index=(len(hessian)-1,len(hessian)-1));vector=vector[:,0]
        gn_max=eigvalsh(gn,subset_by_index=(len(gn)-1,len(gn)-1))[0]
        minimum=eigvalsh(hessian,subset_by_index=(0,0))[0]
        rows.append(dict(step=step,hessian_min=float(minimum),hessian_max=float(maximum[0]),gn_max=float(gn_max),
            eta_hessian_max=float(case['eta']*maximum[0]),eta_gn_max=float(case['eta']*gn_max),
            gradient_top_curvature_fraction=float((vector@gradient)**2/max(gradient@gradient,1e-300)),
            top_vector_block_energy=dict(bias=float(vector[0]**2),readouts=float(np.sum(vector[1:g.width+1]**2)),
                                         geometry=float(np.sum(vector[g.width+1:]**2)))))
    return dict(key=record['key'],case=case,end=end,rows=rows,source_sha256={
        path.name:hashlib.sha256(path.read_bytes()).hexdigest(),
        f'dense_parameters_{end}.npz':hashlib.sha256((folder/f'dense_parameters_{end}.npz').read_bytes()).hexdigest()})


def frozen_decay(singular,modal,residual_perpendicular,steps,eta):
    """Exact arithmetic recurrence for the specified retained linear dictionary."""
    rates=eta*singular**2;steps=np.asarray(steps)
    with np.errstate(divide='ignore',invalid='ignore'):
        logs=np.where(rates<=1,np.log1p(-rates),np.log(rates-1))
        decay=np.exp(2*steps[:,None]*logs[None,:])
    decay[steps==0]=1
    return np.sum(modal[None,:]**2*decay,axis=1)+residual_perpendicular@residual_perpendicular


def frequency_fits(g,gamma,a,u,s,v,c=None,jacobian_lambda=None):
    """Change only the diagnostic RHS; all fits use the individual-scale basis."""
    frequencies=np.array([1,2,4,8,16,32]);x=np.linspace(-1,1,16*g.n+1)
    targets=np.sqrt(2)*np.sin(2*np.pi*x[:,None]*frequencies)/np.sqrt(len(x))
    xv=diagnostics.midpoint_grid(32768)
    av=diagnostics.features(xv,g.centers,gamma)
    yv=np.sqrt(2)*np.sin(2*np.pi*xv[:,None]*frequencies);rows=[]
    for cutoff in analysis.CUTOFFS:
        keep=s>cutoff*s[0];projection=u[:,keep].T@targets
        coefficients=g.alpha[:,None]*(v[keep].T@(projection/s[keep,None]))
        projected=targets-u[:,keep]@projection
        physical=a@coefficients-targets;validation=av@coefficients-yv
        if c is not None:
            live=a@c[:,None]-targets
            outside=live-u[:,keep]@(u[:,keep].T@live)
            outside_force=jacobian_lambda.T@outside
            total_force=jacobian_lambda.T@live
            growth=np.sign(gamma)*g.core/np.sqrt(g.core.sum())
        for j,k in enumerate(frequencies):
            row=dict(frequency_multiplier=int(k),cycles_over_domain=int(2*k),cutoff=cutoff,rank=int(keep.sum()),
                projected_mse=float(projected[:,j]@projected[:,j]),physical_train_mse=float(physical[:,j]@physical[:,j]),
                midpoint_mse=float(np.mean(validation[:,j]**2)),physical_coefficient_l1=float(np.abs(coefficients[:,j]).sum()))
            if c is not None:
                row.update(live_outside_mse=float(outside[:,j]@outside[:,j]),
                    live_geometry_force_norm=float(np.linalg.norm(total_force[:,j])),
                    live_outside_geometry_force_norm=float(np.linalg.norm(outside_force[:,j])),
                    live_outside_core_growth_force=float(-growth@outside_force[:,j]),
                    live_outside_geometry_force_by_region={name:float(np.linalg.norm(outside_force[mask,j])) for name,mask in g.masks.items()})
            rows.append(row)
    return rows


def frozen_case(task):
    source,output,record=task;key=record['key'];case=record['case'];g=core.geometry(case['n'])
    path=source/key/f"spectrum_{record['end']}.npz"
    with np.load(path) as cp:
        c,gamma=cp['c'],cp['gamma'];x,y,a,r,jacobian_lambda=analysis.linearize(g,c,gamma)
    steps=np.r_[0,np.unique(np.ceil(np.geomspace(1,1e12,181)))];curves=[];rows=[];spectra=[]
    for coord in campaign.MAPS:
        u,s,v=svd(analysis.native_features(a,g,coord),full_matrices=False,lapack_driver='gesdd')
        if coord=='parameter_scale':frequency=frequency_fits(g,gamma,a,u,s,v,c,jacobian_lambda)
        eta=1/s[0]**2;spectra.append(s);pair=[]
        for cutoff in analysis.CUTOFFS:
            keep=s>cutoff*s[0];modal=u[:,keep].T@r
            # Compute the complement as a vector, avoiding subtraction of nearly
            # equal total energies when the remaining residual is very small.
            perpendicular=r-u[:,keep]@modal
            energy=frozen_decay(s[keep],modal,perpendicular,steps,eta);pair.append(energy)
            hits={str(f):float(steps[np.flatnonzero(energy<=f*(r@r))[0]]) if np.any(energy<=f*(r@r)) else None for f in (.1,.01,.001)}
            rows.append(dict(coordinates=coord,cutoff=cutoff,rank=int(keep.sum()),eta=float(eta),
                             complement_mse=float(perpendicular@perpendicular),first_sampled_steps_to_fraction=hits))
        curves.append(pair)
    run.save_arrays(output/f'{key}_frozen.npz',steps=steps,curves=curves,singular=spectra,initial_mse=r@r,cutoffs=analysis.CUTOFFS)
    return dict(key=key,case=case,checkpoint=record['end'],initial_mse=float(r@r),probes=rows,frequency_fits=frequency,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def reference_frequency(n):
    g=core.geometry(n);gamma=np.full(g.width,.25/g.h)
    x,y,a,r,j=analysis.linearize(g,np.zeros(g.width+1),gamma)
    u,s,v=svd(a*g.alpha,full_matrices=False,lapack_driver='gesdd')
    return dict(n=n,lambda_value=.25,frequency_fits=frequency_fits(g,gamma,a,u,s,v))


def damping_probe(source,output,record):
    key=record['key'];g=core.geometry(record['case']['n']);path=source/key/f"spectrum_{record['end']}.npz"
    with np.load(path) as cp:c,gamma=cp['c'],cp['gamma']
    x,y,a,r,j=analysis.linearize(g,c,gamma);original_mu=record['numerics']['final_damping']
    dampings=np.unique(np.r_[np.logspace(-24,4,29),original_mu]);rows=[];controls=[]
    for coord in campaign.MAPS:
        transform=higher_order.readout_map(g,coord)
        jac=np.column_stack((analysis.native_features(a,g,coord),j))
        u,s,v=svd(jac,full_matrices=False,lapack_driver='gesdd');projection=u.T@r
        for mu in dampings:
            dz=-v.T@((s/(s*s+mu))*projection)
            dc=transform@dz[:g.width+1];dl=dz[g.width+1:];jd=jac@dz
            next_r=(diagnostics.prediction(x,g.centers,c+dc,gamma+dl/g.h)-y)/np.sqrt(len(x))
            predicted=-r@jd-.5*(jd@jd);dr=next_r-r;actual=-r@dr-.5*(dr@dr)
            rows.append(dict(coordinates=coord,damping=float(mu),predicted_reduction=float(predicted),actual_reduction=float(actual),
                reduction_ratio=float(actual/predicted) if predicted>0 else None,passes_training_acceptance=bool(predicted>0 and actual/predicted>1e-4),
                linearized_mse=float(np.sum((r+jd)**2)),actual_trial_mse=float(next_r@next_r),
                delta_c_norm=float(np.linalg.norm(dc)),delta_lambda_norm=float(np.linalg.norm(dl))))
        # Express the individual-scale damping penalty in this map. Both
        # augmented QR solves now have the SAME physical objective and metric.
        regularizer=block_diag(transform/g.alpha[:,None],np.eye(g.width))
        augmented=np.vstack((jac,np.sqrt(original_mu)*regularizer))
        rhs=np.r_[-r,np.zeros(jac.shape[1])]
        dz,_,rank,_=lstsq(augmented,rhs,lapack_driver='gelsy')
        dc=transform@dz[:g.width+1];dl=dz[g.width+1:]
        controls.append(dict(coordinates=coord,rank=int(rank),dc=dc,dl=dl,motion=jac@dz,
                             normalized_stationarity=float(np.linalg.norm(augmented.T@(augmented@dz-rhs))/np.linalg.norm(jac.T@r))))
    p,q=controls
    control=dict(damping=original_mu,physical_readout_step_relative_difference=float(np.linalg.norm(p['dc']-q['dc'])/np.linalg.norm(p['dc'])),
        physical_lambda_step_relative_difference=float(np.linalg.norm(p['dl']-q['dl'])/np.linalg.norm(p['dl'])),
        function_step_relative_difference=float(np.linalg.norm(p['motion']-q['motion'])/np.linalg.norm(p['motion'])),
        ranks=[v['rank'] for v in controls],normalized_stationarity=[v['normalized_stationarity'] for v in controls])
    result=dict(key=key,checkpoint=record['end'],initial_mse=float(r@r),original_next_damping=original_mu,rows=rows,
                same_physical_metric_control=control,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    run.write_json(output/'damping_probe.json',result)
    return result


def figures(output,records,damping,references):
    for n in (512,1024):
        fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
        for i,opt in enumerate(('gd','adam')):
            for k,trained_map in enumerate(campaign.MAPS):
                selected=[r for r in records if (r['case']['optimizer'],r['case']['coordinates'],r['case']['n'],r['case']['seed'])==(opt,trained_map,n,0)]
                if not selected:continue
                record=selected[0];ax=axes[i,k]
                with np.load(output/f"{record['key']}_frozen.npz") as data:
                    for j,coord in enumerate(campaign.MAPS):
                        ax.loglog(np.maximum(data['steps'],1),data['curves'][j,1],color=analysis.COLORS[coord],label=analysis.LABELS[coord])
                        ax.fill_between(np.maximum(data['steps'],1),data['curves'][j].min(axis=0),data['curves'][j].max(axis=0),color=analysis.COLORS[coord],alpha=.1)
                ax.set(title=f'{opt.upper()} geometry: {analysis.LABELS[trained_map]}',xlabel='Additional frozen-readout GD steps (analytic prediction)',ylabel='Predicted residual MSE')
                ax.grid(alpha=.2);ax.legend(fontsize=8,title='Readout coordinates for the prediction',title_fontsize=8)
        fig.suptitle(f'N={n}, seed 0, geometry and residual at update 100k\nSpectral step 1 / sigma_max² for each map; line cutoff 1e−12, shading cutoffs 1e−10–1e−14; not an Adam trajectory')
        fig.savefig(output/f'frozen_decay_N{n}.png',dpi=160,bbox_inches='tight');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
        selected=[r for r in records if r['case']['n']==n and r['case']['seed']==0 and r['case']['optimizer']=='gn']
        series=[(r['frequency_fits'],analysis.LABELS[r['case']['coordinates']],analysis.COLORS[r['case']['coordinates']]) for r in selected]
        reference=next(r for r in references if r['n']==n)
        series.append((reference['frequency_fits'],'Uniform construction bandwidth 0.25','#4d9221'))
        for entries,label,color in series:
            rows=[r for r in entries if r['cutoff']==1e-12]
            for ax,field in zip(axes,('midpoint_mse','physical_coefficient_l1')):
                ax.loglog([r['cycles_over_domain'] for r in rows],[r[field] for r in rows],'o-',color=color,label=label)
        for ax,ylabel in zip(axes,('Detached fit midpoint MSE','Physical coefficient L1 norm')):
            ax.set_xscale('log',base=2);ax.set_xticks([2,4,8,16,32,64],['2','4','8','16','32','64'])
            ax.tick_params(axis='x',which='minor',labelbottom=False)
            ax.set(xlabel='Sine cycles across [-1,1]',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'N={n}, seed-0 GN geometries after 20k updates; new targets only in detached fits\nCommon individual-scale SVD basis, relative cutoff 1e−12; no new training')
        fig.savefig(output/f'frequency_capacity_N{n}.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for coord in campaign.MAPS:
        rows=[r for r in damping['rows'] if r['coordinates']==coord];mu=[r['damping'] for r in rows]
        color=analysis.COLORS[coord];label=analysis.LABELS[coord]
        for ax,field in ((axes[0,0],'linearized_mse'),(axes[0,1],'actual_trial_mse'),(axes[1,0],'delta_c_norm'),(axes[1,1],'delta_lambda_norm')):
            ax.loglog(mu,[r[field] for r in rows],color=color,label=label)
        accepted=[r for r in rows if r['passes_training_acceptance']]
        axes[0,1].plot([r['damping'] for r in accepted],[r['actual_trial_mse'] for r in accepted],'o',color=color,ms=4)
    for ax,title,ylabel in ((axes[0,0],'Linearized model','Predicted MSE'),(axes[0,1],'Actual nonlinear trial; dots pass acceptance','Trial MSE'),
                            (axes[1,0],'Physical readout motion','Norm of delta c'),(axes[1,1],'Bandwidth motion','Norm of delta lambda')):
        ax.axvline(damping['original_next_damping'],color='.5',ls=':',label='Stored next damping')
        ax.set(title=title,xlabel='Native damping mu',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax in axes[0]:ax.axhline(damping['initial_mse'],color='.6',ls='--')
    fig.suptitle('Same N=1024 seed-0 physical checkpoint after 20k individual-scale GN updates\nDetached trials with native identity damping; dashed horizontal line = current MSE')
    fig.savefig(output/'damping_trials.png',dpi=160,bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--stability-only',action='store_true',help='Audit local GD curvature at three consecutive saved states, N=512')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=json.loads((args.analysis/'summary.json').read_text())
    if args.stability_only:
        selected=[r for r in records if r['case']['optimizer']=='gd' and r['case']['n']==512]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows=list(pool.map(stability_case,[(args.analysis,r) for r in selected]))
        run.write_json(args.output/'stability.json',dict(records=rows,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training_states_modified=False))
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        frozen=list(pool.map(frozen_case,[(args.analysis,args.output,r) for r in records]))
    run.write_json(args.output/'frozen_summary.json',frozen)
    references=[reference_frequency(n) for n in (512,1024)]
    run.write_json(args.output/'reference_frequency.json',references)
    source=next(r for r in records if (r['case']['n'],r['case']['seed'],r['case']['optimizer'],r['case']['coordinates'])==(1024,0,'gn','parameter_scale'))
    damping=damping_probe(args.analysis,args.output,source)
    figures(args.output,frozen,damping,references)
    run.write_json(args.output/'provenance.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_summary_sha256=hashlib.sha256((args.analysis/'summary.json').read_bytes()).hexdigest(),training_states_modified=False))


if __name__=='__main__':main()
