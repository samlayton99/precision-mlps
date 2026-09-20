"""Width-dependent construction accuracy using the existing QI implementation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
import time
import mpmath as mp
import numpy as np
from . import core, full_core as f, full_precision as precision


# Load the standalone construction without importing the Torch model package.
_spec=importlib.util.spec_from_file_location('_gamma_qi_reference',
    Path(__file__).resolve().parents[2]/'src/construction/qi_mpmath.py')
_qi=importlib.util.module_from_spec(_spec); sys.modules[_spec.name]=_qi
_spec.loader.exec_module(_qi)
construct_qi=_qi.construct_qi


def derivative(x,name):
    if name=='sine_mix_2_6_10':
        return 2*np.pi*np.cos(2*np.pi*x)+3*np.pi*np.cos(6*np.pi*x)+2.5*np.pi*np.cos(10*np.pi*x)
    if name=='runge_25':
        return -50*x/(1+25*x*x)**2
    raise ValueError(name)


def evaluate(qi,x,dtype):
    result=np.empty(len(x),dtype=dtype)
    centers=qi.centers.astype(dtype); coefficient=qi.a_coeffs.astype(dtype)
    for start in range(0,len(x),4096):
        xx=x[start:start+4096].astype(dtype)
        result[start:start+len(xx)]=dtype(qi.c0)+np.tanh(dtype(qi.gamma)*(xx[:,None]-centers))@coefficient
    return result


def extended_target(x,name):
    x=x.astype(np.longdouble); pi=np.arccos(-np.longdouble(1))
    if name=='sine_mix_2_6_10':
        return np.sin(2*pi*x)+np.sin(6*pi*x)/2+np.sin(10*pi*x)/4
    return 1/(1+25*x*x)


def width(task):
    root,n,targets,n_eval,deadline=task; start=time.monotonic(); folder=root/'reference'/f'N{n}'
    folder.mkdir(parents=True,exist_ok=True); x=core.grid(n_eval); rows=[]
    for name in targets:
        previous=None
        for digits in [40,80]:
            if time.monotonic()>deadline:
                raise TimeoutError('Reference construction deadline')
            qi=construct_qi(lambda z:f.target(z,name),lambda z:derivative(z,name),N=n,
                precision='mpmath',lambda_star=.25,Kc=160,halo=int(np.ceil(np.sqrt(n))),
                mp_dps=digits,cache_dir=root/'reference/cardinal_cache')
            ordinary=evaluate(qi,x,np.float64); extended=evaluate(qi,x,np.longdouble)
            y=extended_target(x,name); norm=np.linalg.norm(y)
            row=dict(n=n,width=len(qi.centers),target=name,digits=digits,lambda_value=.25,gamma=qi.gamma,
                relative_error_fp64=float(np.linalg.norm(ordinary-f.target(x,name))/np.linalg.norm(f.target(x,name))),
                relative_error_extended=float(np.linalg.norm(extended-y)/norm),
                evaluation_precision_difference=float(np.linalg.norm(extended-ordinary)/norm),
                construction_precision_difference=float(np.linalg.norm(extended-previous)/norm) if previous is not None else None,
                extended_mantissa_bits=int(np.finfo(np.longdouble).nmant),
                note='Existing constructor samples derivatives in FP64 and returns FP64 coefficients; this is measured recovery accuracy.')
            if digits==80:
                check_x=core.grid(129); high=[]
                with mp.workdps(80):
                    coefficients=[mp.mpf(float(c)) for c in qi.a_coeffs]
                    centers=[mp.mpf(float(c)) for c in qi.centers]
                    for xx in check_x:
                        z=mp.mpf(float(xx))
                        value=mp.mpf(qi.c0)+mp.fsum(a*mp.tanh(qi.gamma*(z-c)) for a,c in zip(coefficients,centers))
                        high.append(value-precision.target_mp(z,name))
                    high_error=mp.sqrt(mp.fsum(v*v for v in high)/mp.fsum(precision.target_mp(mp.mpf(float(z)),name)**2 for z in check_x))
                low=evaluate(qi,check_x,np.longdouble)-extended_target(check_x,name)
                high_float=np.array([np.longdouble(str(v)) for v in high])
                row.update(check_grid_points=129,check_error_mp80=float(high_error),
                    check_extended_mp_difference=float(np.linalg.norm(low-high_float)/np.linalg.norm(extended_target(check_x,name))))
            previous=extended
            core.save_arrays(folder/f'{name}_dps{digits}.npz',centers=qi.centers,coefficients=qi.a_coeffs,
                bias=qi.c0,gamma=qi.gamma,x=x,prediction_fp64=ordinary,prediction_extended=extended)
            rows.append(row); core.write_json(folder/'measurements.json',rows)
    print(f'REFERENCE N={n} seconds={time.monotonic()-start:.2f}',flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=650)
    args=parser.parse_args(); cfg=json.loads((args.root/'manifest.json').read_text())['config']
    deadline=time.monotonic()+args.seconds
    tasks=[(args.root,n,cfg['robust_targets'],cfg['n_eval'],deadline) for n in cfg['widths']]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=[row for group in pool.map(width,tasks) for row in group]
    core.write_json(args.root/'reference/measurements.json',rows)
    core.write_json(args.root/'validation/reference_complete.json',dict(complete=True))


if __name__=='__main__':
    main()
