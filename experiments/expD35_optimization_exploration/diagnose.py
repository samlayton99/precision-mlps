"""Detached conditioning and Fourier evidence; never writes training parameters."""
import argparse
import json
from pathlib import Path
import numpy as np
from . import core,run

BANDS=((0,1),(1,2),(2,4),(4,8),(8,16),(16,32),(32,64),(64,128),(128,256),(256,None))


def forward(z,g,coordinates,x):
    c,gamma=map(np.asarray,core.physical(z,g,coordinates))
    return c[0]+np.tanh((x[:,None]-g.centers)*gamma)@c[1:]


def transform(g,coordinates):
    if coordinates=='physical':return np.eye(g.width+1)
    return np.stack([
        np.asarray(core.maps.decode(np.eye(g.width+1)[j],g,core.ALIASES[coordinates]))
        for j in range(g.width+1)],axis=1)


def band_energy(values):
    return band_product(values,values)


def band_product(a,b):
    fa=np.fft.rfft(a)/len(a);fb=np.fft.rfft(b)/len(b)
    energy=np.real(np.conj(fa)*fb)
    multiplicity=np.full(len(energy),2.);multiplicity[0]=1
    if len(a)%2==0:multiplicity[-1]=1
    energy*=multiplicity
    return np.array([np.sum(energy[lo:hi]) for lo,hi in BANDS])


def snapshot(path,config,out):
    g=core.old.geometry(config['n'])
    with np.load(path) as data:z=np.asarray(data['z'])
    c,gamma=map(np.asarray,core.physical(z,g,config['coordinates']))
    m=max(2048,4*config['n']);x=-1+2*(np.arange(m)+.5)/m
    y=core.target(x,config['target'],np);phi=np.tanh((x[:,None]-g.centers)*gamma)
    design=np.column_stack((np.ones(m),phi));residual=design@c-y
    mapping=transform(g,config['coordinates'])
    native=design@mapping/np.sqrt(m)
    u,s,vh=np.linalg.svd(native,full_matrices=False)
    physical_s=np.linalg.svd(design/np.sqrt(m),compute_uv=False)
    projected=u.T@(residual/np.sqrt(m));relative=s/s[0]
    ls=[];coefficients=[]
    xv=-1+2*(np.arange(32768)+.5)/32768;yv=core.target(xv,config['target'],np)
    for cutoff in (np.finfo(float).eps*max(native.shape),1e-12,1e-14,1e-16):
        keep=relative>cutoff
        solution=vh[keep].T@((u[:,keep].T@(y/np.sqrt(m)))/s[keep])
        physical_c=mapping@solution
        pred=physical_c[0]+np.tanh((xv[:,None]-g.centers)*gamma)@physical_c[1:]
        ls.append(dict(relative_cutoff=cutoff,rank=int(np.sum(keep)),
            fit_mse=float(np.mean((design@physical_c-y)**2)),validation_mse=float(np.mean((pred-yv)**2)),
            coefficient_l2=float(np.linalg.norm(physical_c))))
        coefficients.append(physical_c)
    mode_edges=(0.,1e-8,1e-6,1e-4,1e-2,.1,1.0000001)
    mode_energy=np.array([np.sum(projected[(relative>=a)&(relative<b)]**2) for a,b in zip(mode_edges[:-1],mode_edges[1:])])
    arrays=dict(z=z,c=c,gamma=gamma,centers=g.centers,alpha=g.alpha,h=g.h,x=x,residual=residual,
        native_singular_values=s,physical_singular_values=physical_s,projected_residual=projected,
        mode_edges=mode_edges,mode_residual_energy=mode_energy,band_energy=band_energy(residual),
        least_squares_coefficients=np.stack(coefficients))
    run.save(out.with_suffix('.npz'),**arrays)
    result=dict(snapshot=str(path),config=config,mse=float(np.mean(residual**2)),
        lambda_quantiles=np.quantile(np.abs(g.h*gamma),[0,.1,.5,.9,1]).tolist(),
        resolved_mode_energy=float(np.sum(projected**2)),residual_energy=float(np.mean(residual**2)),
        least_squares=ls,mode_edges=mode_edges,mode_energy=mode_energy.tolist(),
        frequency_bands=[f'{a}–{b-1}' if b else f'{a}+' for a,b in BANDS],
        frequency_mse=arrays['band_energy'].tolist())
    run.write_json(out.with_suffix('.json'),result)
    return result


def dense(folder,config,out):
    g=core.old.geometry(config['n']);m=max(2048,4*config['n']);x=-1+2*(np.arange(m)+.5)/m
    y=core.target(x,config['target'],np);rows=[]
    for path in sorted(folder.glob('dense_*.npz')):
        with np.load(path) as data:
            zs=np.concatenate((data['initial_z'][None],data['z']));start=int(data['start'])
        for i in range(0,len(zs)-1,16):
            c0,ga0=map(np.asarray,core.physical(zs[i],g,config['coordinates']))
            c1,ga1=map(np.asarray,core.physical(zs[i+1],g,config['coordinates']))
            phi=np.tanh((x[:,None]-g.centers)*ga0)
            f0=c0[0]+phi@c0[1:];fr=c1[0]+phi@c1[1:]
            f1=c1[0]+np.tanh((x[:,None]-g.centers)*ga1)@c1[1:]
            r=f0-y;dr=fr-f0;dg=f1-fr
            # Exact sequential attribution: readout first, then geometry.
            linear_r=2*band_product(r,dr)
            linear_g=2*band_product(r,dg)
            rows.append(dict(update=start+i+1,residual=band_energy(r),readout_update=band_energy(dr),
                geometry_update=band_energy(dg),readout_descent=-linear_r,geometry_descent=-linear_g,
                update_cross=2*band_product(dr,dg),
                readout_motion=np.sqrt(np.mean((c1-c0)**2)),gamma_motion=np.sqrt(np.mean((ga1-ga0)**2))))
    if rows:run.save(out,**{k:np.stack([r[k] for r in rows]) for k in rows[0]})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--ids',nargs='+',required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    for identity in args.ids:
        folder=args.root/identity;c=json.loads((folder/'case.json').read_text())
        snapshots=sorted(folder.glob('snapshot_*.npz'))
        result=snapshot(snapshots[-1],c,args.out/identity)
        dense(folder,c,args.out/(identity+'_dense.npz'))
        print(json.dumps(dict(id=identity,mse=result['mse'],least_squares=result['least_squares'])),flush=True)


if __name__=='__main__':main()
