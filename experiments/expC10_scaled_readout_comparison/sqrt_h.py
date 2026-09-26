"""Add only hidden-column sqrt(h) scaling to the saved C10 sweep."""
import argparse
import hashlib
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

import run as base


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    root=base.OUT/'data'
    meta=json.loads((root/'metadata.json').read_text())
    cfg=meta['config']
    if not args.plot_only:
        data=root/'sqrt_h';data.mkdir(exist_ok=True)
        with np.load(root/'samples.npz') as z:
            x,xe,Y,Ye=[z[key].copy() for key in ('x_train','x_eval','y_train','y_eval')]
        records=[]
        with threadpool_limits(limits=cfg['threads']):
            for n in cfg['interior_resolutions']:
                source=root/f'N{n}.npz'
                digest=hashlib.sha256(source.read_bytes()).hexdigest()
                output=data/f'N{n}.npz'
                if output.exists():
                    with np.load(output) as old:
                        if str(old['source_sha256'])!=digest:raise ValueError('Source changed')
                    print(f'Reusing sqrt(h), N={n}',flush=True)
                    continue
                start=time.perf_counter()
                with np.load(source) as z:
                    centers=z['centers'].copy();lambdas=z['lambdas'].copy()
                p=len(centers)+1
                scales=np.r_[1.,np.full(len(centers),np.sqrt(2/n))]
                coefficients=[];native=[];errors=[];training=[];ranks=[];spectra=[]
                max_discrepancy=0.
                for lam in lambdas:
                    gamma=lam*n/2
                    A=np.c_[np.ones(len(x)),np.tanh(gamma*(x[:,None]-centers))]
                    E=np.c_[np.ones(len(xe)),np.tanh(gamma*(xe[:,None]-centers))]
                    c,a,rank,s=base.scaled_solve(A,Y,scales,cfg['relative_svd_cutoff'])
                    prediction=E@c
                    errors.append(np.linalg.norm(prediction-Ye,axis=0)/np.linalg.norm(Ye,axis=0))
                    training.append(np.linalg.norm(A@c-Y,axis=0)/np.linalg.norm(Y,axis=0))
                    coefficients.append(c);native.append(a);ranks.append(rank);spectra.append(s)
                    # Check actual physical versus native predictions at the reference scale.
                    if lam==cfg['standard_lambda']:
                        discrepancy=np.linalg.norm(prediction-(E*scales)@a,axis=0)/np.linalg.norm(Ye,axis=0)
                        max_discrepancy=float(discrepancy.max())
                errors=np.asarray(errors)
                assert np.isfinite(errors).all() and np.all(errors>0)
                assert scales[0]==1 and np.all(scales[1:]==np.sqrt(2/n))
                assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
                base.save(output,lambdas=lambdas,centers=centers,scales=scales,
                          eval_rel_l2=errors,train_rel_l2=np.asarray(training),
                          physical_coefficients=np.asarray(coefficients),
                          native_coefficients=np.asarray(native),ranks=np.asarray(ranks),
                          singular_values=np.asarray(spectra),source_sha256=digest,
                          max_mapping_discrepancy_at_reference=max_discrepancy)
                elapsed=time.perf_counter()-start
                records.append({'N':n,'source_sha256':digest,'seconds':elapsed,
                                'maximum_mapping_discrepancy_at_lambda_025':max_discrepancy})
                print(f'Completed sqrt(h), N={n}, {len(lambdas)} bandwidths; {elapsed:.1f}s',flush=True)
        if records:
            (data/'metadata.json').write_text(json.dumps({
                'map':'D=diag(1,sqrt(h),...,sqrt(h)); every hidden neuron including halo, bias unscaled',
                'config':cfg,'source':'../N{N}.npz; raw and envelope runs reused unchanged',
                'validation':records},indent=2)+'\n')
    from plot import draw
    draw(cfg,base.OUT,include_sqrt_h=True)


if __name__=='__main__':main()
