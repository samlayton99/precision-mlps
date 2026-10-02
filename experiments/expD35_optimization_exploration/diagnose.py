"""Detached conditioning and Fourier evidence; never writes training parameters."""
import argparse
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from . import core,run

BANDS=((0,1),(1,2),(2,4),(4,8),(8,16),(16,32),(32,64),(64,128),(128,256),(256,None))


def forward(z,g,coordinates,x):
    c,gamma=map(np.asarray,core.physical(z,g,coordinates))
    return c[0]+np.tanh((x[:,None]-g.centers)*gamma+np.asarray(core.offsets(z,g)))@c[1:]


def transform(g,coordinates):
    if coordinates=='physical':return np.eye(g.width+1)
    return np.stack([
        np.asarray(core.maps.decode(np.eye(g.width+1)[j],g,core.ALIASES[coordinates]))
        for j in range(g.width+1)],axis=1)


def band_energy(values):
    return band_product(values,values)


def band_components(values):
    spectrum=np.fft.rfft(values)
    parts=[]
    for lo,hi in BANDS:
        selected=np.zeros_like(spectrum);selected[lo:hi]=spectrum[lo:hi]
        parts.append(np.fft.irfft(selected,n=len(values)))
    return np.stack(parts)


def band_product(a,b):
    fa=np.fft.rfft(a)/len(a);fb=np.fft.rfft(b)/len(b)
    energy=np.real(np.conj(fa)*fb)
    multiplicity=np.full(len(energy),2.);multiplicity[0]=1
    if len(a)%2==0:multiplicity[-1]=1
    energy*=multiplicity
    return np.array([np.sum(energy[lo:hi]) for lo,hi in BANDS])


def snapshot(path,config,out,grid_size=None,training_grid=False):
    g=core.old.geometry(config['n'])
    with np.load(path) as data:z=np.asarray(data['z'])
    c,gamma=map(np.asarray,core.physical(z,g,config['coordinates']))
    if training_grid:
        m=16*config['n']+1;x=np.linspace(-1,1,m)
    else:
        m=grid_size or max(2048,4*config['n']);x=-1+2*(np.arange(m)+.5)/m
    beta=np.asarray(core.offsets(z,g))
    y=core.target(x,config['target'],np);pre=(x[:,None]-g.centers)*gamma+beta;phi=np.tanh(pre)
    design=np.column_stack((np.ones(m),phi));residual=design@c-y
    mapping=transform(g,config['coordinates'])
    native=design@mapping/np.sqrt(m)
    u,s,vh=np.linalg.svd(native,full_matrices=False)
    physical_s=np.linalg.svd(design/np.sqrt(m),compute_uv=False)
    projected=u.T@(residual/np.sqrt(m));relative=s/s[0]
    distances=x[:,None]-g.centers
    exponential=np.exp(-2*np.abs(pre))
    gamma_jacobian=c[1:]*distances*(4*exponential/(1+exponential)**2)
    centered=x-np.mean(x)
    coarse=np.mean(residual)+centered*np.mean(residual*centered)/np.mean(centered**2)
    gamma_gradient=gamma_jacobian.T@residual/m
    readout_gradient=design.T@residual/m
    coarse_gradient=gamma_jacobian.T@coarse/m
    keep=relative>1e-12
    removable=u[:,keep]@(u[:,keep].T@residual)
    removable_gradient=gamma_jacobian.T@removable/m
    fourier_residual=band_components(residual)
    fourier_readout=fourier_residual@design/m
    fourier_gamma=fourier_residual@gamma_jacobian/m
    ls=[];coefficients=[]
    xv=-1+2*(np.arange(32768)+.5)/32768;yv=core.target(xv,config['target'],np)
    for cutoff in (np.finfo(float).eps*max(native.shape),1e-12,1e-14,1e-16):
        keep=relative>cutoff
        solution=vh[keep].T@((u[:,keep].T@(y/np.sqrt(m)))/s[keep])
        physical_c=mapping@solution
        pred=physical_c[0]+np.tanh((xv[:,None]-g.centers)*gamma+beta)@physical_c[1:]
        ls.append(dict(relative_cutoff=cutoff,rank=int(np.sum(keep)),
            fit_mse=float(np.mean((design@physical_c-y)**2)),validation_mse=float(np.mean((pred-yv)**2)),
            coefficient_l2=float(np.linalg.norm(physical_c))))
        coefficients.append(physical_c)
    mode_edges=(0.,1e-8,1e-6,1e-4,1e-2,.1,1.0000001)
    mode_energy=np.array([np.sum(projected[(relative>=a)&(relative<b)]**2) for a,b in zip(mode_edges[:-1],mode_edges[1:])])
    arrays=dict(z=z,c=c,gamma=gamma,offsets=beta,centers=g.centers,alpha=g.alpha,h=g.h,x=x,residual=residual,
        native_singular_values=s,physical_singular_values=physical_s,projected_residual=projected,
        mode_edges=mode_edges,mode_residual_energy=mode_energy,band_energy=band_energy(residual),
        physical_gamma_gradient=gamma_gradient,coarse_gamma_gradient=coarse_gradient,
        physical_readout_gradient=readout_gradient,native_readout_gradient=mapping.T@readout_gradient,
        remainder_gamma_gradient=gamma_gradient-coarse_gradient,readout_span_gamma_gradient=removable_gradient,
        orthogonal_gamma_gradient=gamma_gradient-removable_gradient,
        frequency_readout_gradient=fourier_readout,frequency_gamma_gradient=fourier_gamma,
        frequency_native_readout_gradient=fourier_readout@mapping,
        frequency_native_slope_gradient=fourier_gamma/(1. if config['coordinates']=='physical' else g.h),
        least_squares_coefficients=np.stack(coefficients))
    run.save(out.with_suffix('.npz'),**arrays)
    result=dict(snapshot=str(path),config=config,grid_size=m,
        quadrature='training endpoint grid' if training_grid else 'diagnostic midpoint grid',mse=float(np.mean(residual**2)),
        lambda_quantiles=np.quantile(np.abs(g.h*gamma),[0,.1,.5,.9,1]).tolist(),
        resolved_mode_energy=float(np.sum(projected**2)),residual_energy=float(np.mean(residual**2)),
        least_squares=ls,mode_edges=mode_edges,mode_energy=mode_energy.tolist(),
        frequency_bands=[f'{a}–{b-1}' if b else f'{a}+' for a,b in BANDS],
        frequency_mse=arrays['band_energy'].tolist())
    result['geometry_signal']=dict(coarse_residual_mse=float(np.mean(coarse**2)),
        remainder_residual_mse=float(np.mean((residual-coarse)**2)),
        gradient_norm=float(np.linalg.norm(gamma_gradient)),coarse_gradient_norm=float(np.linalg.norm(coarse_gradient)),
        remainder_gradient_norm=float(np.linalg.norm(gamma_gradient-coarse_gradient)),
        readout_span_gradient_norm=float(np.linalg.norm(removable_gradient)),
        orthogonal_gradient_norm=float(np.linalg.norm(gamma_gradient-removable_gradient)),span_relative_cutoff=1e-12)
    result['fourier_gradient']=dict(readout_norms=np.linalg.norm(fourier_readout,axis=1).tolist(),
        gamma_norms=np.linalg.norm(fourier_gamma,axis=1).tolist(),
        readout_sum_error=float(np.linalg.norm(fourier_readout.sum(axis=0)-readout_gradient)),
        gamma_sum_error=float(np.linalg.norm(fourier_gamma.sum(axis=0)-gamma_gradient)))
    run.write_json(out.with_suffix('.json'),result)
    return result


def dense(folder,config,out,grid_size=None):
    g=core.old.geometry(config['n']);m=grid_size or max(2048,4*config['n']);x=-1+2*(np.arange(m)+.5)/m
    y=core.target(x,config['target'],np);rows=[];basis=None;spectrum=None;residuals=[]
    mapping=transform(g,config['coordinates'])
    tx=jnp.linspace(-1,1,16*config['n']+1);ty=core.target(tx,config['target'])
    training_gradient=jax.jit(lambda z:core.field(z,tx,ty,g,config['coordinates'])[1])
    edges=np.array([0.,1e-8,1e-6,1e-4,1e-2,.1,1.0000001])
    for path in sorted(folder.glob('dense_*.npz')):
        with np.load(path) as data:
            zs=np.concatenate((data['initial_z'][None],data['z']));start=int(data['start'])
            recorded=np.asarray(data['gradients']) if 'gradients' in data else None
            recorded_indices=np.asarray(data['gradient_indices']) if 'gradient_indices' in data else np.arange(0,len(zs)-1,16)
            lookup={int(at):k for k,at in enumerate(recorded_indices)} if recorded is not None else {}
        if basis is None:
            _,ga=map(np.asarray,core.physical(zs[0],g,config['coordinates']))
            features=np.column_stack((np.ones(m),np.tanh((x[:,None]-g.centers)*ga+np.asarray(core.offsets(zs[0],g)))))
            basis,spectrum,_=np.linalg.svd(features@transform(g,config['coordinates'])/np.sqrt(m),full_matrices=False)
            relative=spectrum/spectrum[0]
            masks=[(relative>=a)&(relative<b) for a,b in zip(edges[:-1],edges[1:])]
        for i in run.sample_indices(len(zs)-1,start):
            c0,ga0=map(np.asarray,core.physical(zs[i],g,config['coordinates']))
            c1,ga1=map(np.asarray,core.physical(zs[i+1],g,config['coordinates']))
            beta0=np.asarray(core.offsets(zs[i],g));beta1=np.asarray(core.offsets(zs[i+1],g))
            phi=np.tanh((x[:,None]-g.centers)*ga0+beta0)
            f0=c0[0]+phi@c0[1:];fr=c1[0]+phi@c1[1:]
            f1=c1[0]+np.tanh((x[:,None]-g.centers)*ga1+beta1)@c1[1:]
            r=f0-y;dr=fr-f0;dg=f1-fr
            residuals.append(r)
            gc=np.r_[np.mean(r),phi.T@r/m];gr=mapping.T@gc
            pre=(x[:,None]-g.centers)*ga0+beta0;exp=np.exp(-2*np.abs(pre));sech=4*exp/(1+exp)**2
            gg=np.mean(r[:,None]*c0[1:]*(x[:,None]-g.centers)*sech,axis=0)
            gg/=1. if config['coordinates']=='physical' else g.h
            if config.get('architecture')=='affine':gg=np.r_[gg,np.mean(r[:,None]*c0[1:]*sech,axis=0)]
            delta=zs[i+1]-zs[i];dzr=delta[:g.width+1];dzg=delta[g.width+1:]
            actual=recorded[lookup[i]] if i in lookup else np.asarray(training_gradient(zs[i]))
            actual_r=actual[:g.width+1];actual_g=actual[g.width+1:]
            def alignment(a,b):
                den=np.linalg.norm(a)*np.linalg.norm(b)
                return float(a@b/den) if den else np.nan
            # Exact sequential attribution: readout first, then geometry.
            linear_r=2*band_product(r,dr)
            linear_g=2*band_product(r,dg)
            ur=basis.T@r/np.sqrt(m);uw=basis.T@dr/np.sqrt(m);ug=basis.T@dg/np.sqrt(m)
            aggregate=lambda vector:np.array([np.sum(vector[mask]) for mask in masks])
            rows.append(dict(update=start+i+1,residual=band_energy(r),readout_update=band_energy(dr),
                geometry_update=band_energy(dg),readout_descent=-linear_r,geometry_descent=-linear_g,
                update_cross=2*band_product(dr,dg),
                mode_residual=aggregate(ur**2),mode_readout_update=aggregate(uw**2),
                mode_geometry_update=aggregate(ug**2),mode_readout_descent=aggregate(-2*ur*uw),
                mode_geometry_descent=aggregate(-2*ur*ug),outside_fixed_span=max(0.,np.mean(r*r)-np.sum(ur**2)),
                readout_update_outside_span=max(0.,np.mean(dr*dr)-np.sum(uw**2)),
                geometry_update_outside_span=max(0.,np.mean(dg*dg)-np.sum(ug**2)),
                native_readout_gradient=np.linalg.norm(actual_r),native_geometry_gradient=np.linalg.norm(actual_g),
                native_readout_motion=np.linalg.norm(dzr),native_geometry_motion=np.linalg.norm(dzg),
                readout_gradient_alignment=alignment(dzr,-actual_r),geometry_gradient_alignment=alignment(dzg,-actual_g),
                validation_readout_gradient_alignment=alignment(dzr,-gr),validation_geometry_gradient_alignment=alignment(dzg,-gg),
                gradient_recorded_on_device=i in lookup,
                readout_motion=np.sqrt(np.mean((c1-c0)**2)),gamma_motion=np.sqrt(np.mean((ga1-ga0)**2)),
                offset_motion=np.sqrt(np.mean((beta1-beta0)**2))))
    if rows:run.save(out,grid_size=m,singular_values=spectrum,mode_edges=edges,
                     mean_residual= np.mean(residuals,axis=0),mean_residual_frequency_energy=band_energy(np.mean(residuals,axis=0)),
                     temporal_fluctuation_frequency_energy=np.mean([r['residual'] for r in rows],axis=0)-band_energy(np.mean(residuals,axis=0)),
                     **{k:np.stack([r[k] for r in rows]) for k in rows[0]})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--ids',nargs='+',required=True)
    p.add_argument('--grid-size',type=int)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    for identity in args.ids:
        folder=args.root/identity;c=json.loads((folder/'case.json').read_text())
        snapshots=sorted(folder.glob('snapshot_*.npz'))
        result=snapshot(snapshots[-1],c,args.out/identity,args.grid_size)
        dense(folder,c,args.out/(identity+'_dense.npz'),args.grid_size)
        print(json.dumps(dict(id=identity,mse=result['mse'],least_squares=result['least_squares'])),flush=True)


if __name__=='__main__':main()
