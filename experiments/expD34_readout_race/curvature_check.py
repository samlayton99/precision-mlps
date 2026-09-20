"""Offline local GD stability at saved states; never changes training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .analyze import table


def hessian(z,d,x,y):
    a,b,c=z;w=len(a);m=len(x);pre=x[:,None]*a+b
    h=np.tanh(pre);ee=np.exp(-2*abs(pre));s=4*ee/(1+ee)**2
    e=h@c+d-y;second=-2*h*s
    J=np.column_stack((x[:,None]*s*c,s*c,h,np.ones(m)))
    H=J.T@J/m;ia=np.arange(w);ib=ia+w;ic=ia+2*w
    for left,right,term in ((ia,ia,c*x[:,None]**2*second),(ia,ib,c*x[:,None]*second),
                           (ib,ib,c*second),(ia,ic,x[:,None]*s),(ib,ic,s)):
        value=np.mean(e[:,None]*term,axis=0)
        H[left,right]+=value
        if not np.array_equal(left,right): H[right,left]+=value
    return H


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=[]
    for folder in sorted(args.root.glob('core_N128_s[0-4]')):
        manifest=json.loads((folder/'manifest.json').read_text())
        with np.load(folder/'initial.npz') as data: x,y=data['x'],data['y']
        for step in (20000,100000,600000):
            with np.load(folder/f'p0/state_{step:09d}.npz') as data:
                for i,case in enumerate(manifest['cases']):
                    if case['kappa'] not in (1.,100.): continue
                    z,d=data['z'][i],data['d'][i];w=z.shape[1]
                    H=hessian(z,d,x,y[i])
                    scale=np.sqrt(np.r_[np.full(2*w,case['eta_geometry']),np.full(w+1,case['eta_readout'])])
                    eigenvalues,vectors=np.linalg.eigh(scale[:,None]*H*scale[None,:]);v=vectors[:,-1]
                    rows.append(dict(bundle=folder.name,target=case['target'],seed=case['seed'],kappa=case['kappa'],
                        step=step,scaled_hessian_max=float(eigenvalues[-1]),scaled_hessian_min=float(eigenvalues[0]),
                        largest_mode_update_eigenvalue=float(1-eigenvalues[-1]),readout_l2=float(np.linalg.norm(z[2])),
                        top_mode_slope_fraction=float(np.sum(v[:w]**2)),top_mode_hidden_bias_fraction=float(np.sum(v[w:2*w]**2)),
                        top_mode_readout_fraction=float(np.sum(v[2*w:3*w]**2)),top_mode_output_bias_fraction=float(v[-1]**2)))
        print(folder.name,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);table(args.output,rows)


if __name__=='__main__':main()
