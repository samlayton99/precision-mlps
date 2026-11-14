"""Target-free native FP64 DGEJSV probe on the option-C matrices.

SciPy integer mapping is verified against flapack_other.pyf.src:
https://raw.githubusercontent.com/scipy/scipy/v1.17.0/scipy/linalg/flapack_other.pyf.src
LAPACK semantics:
https://netlib.org/lapack/explore-html/d8/d78/group__gejsv_gaca7ba7f1e8002c7a1d5bffa4ccbb541f.html

No target samples, readout fitting, test errors, or precision changes are used.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy.linalg import get_lapack_funcs
from threadpoolctl import threadpool_limits
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expC14_theory_solver_comparison import option_c as oc
from experiments.expC14_theory_solver_comparison import run as r
from experiments.expC14_theory_solver_comparison.theory_parameters import prepare


def jacobi_svd(F, *, joba=0, jobv=1):
    """Complete thin U,V; C/F accuracy modes, no perturbation or range kill."""
    driver=get_lapack_funcs('gejsv',(F,))
    options=dict(joba=joba,jobu=0,jobv=jobv,jobr=0,jobt=0,jobp=0)
    start=time.perf_counter()
    sva,U,V,work,iwork,info=driver(np.array(F,order='F'),**options)
    seconds=time.perf_counter()-start
    if info!=0:raise RuntimeError(f'DGEJSV failed: INFO={info}')
    if work[1]==0:raise RuntimeError('DGEJSV scale denominator is zero')
    scale=float(work[0]/work[1])
    singular=sva*scale
    if not np.all(np.isfinite(singular)):raise RuntimeError('Nonfinite singular values')
    n=F.shape[1]
    if U.shape!=(F.shape[0],n) or V.shape!=(n,n):raise RuntimeError('Not a complete thin SVD')
    meta=dict(options=options,characters=dict(joba='CEFGAR'[joba],jobu='U',jobv='VJWN'[jobv],jobr='N',jobt='N',jobp='N'),
              seconds=seconds,work=work.tolist(),iwork=iwork.tolist(),scale=scale,
              nonzero_singular_values=int(np.count_nonzero(singular)),n=n,
              left_column_norm_min=float(np.min(np.linalg.norm(U,axis=0))),
              right_column_norm_min=float(np.min(np.linalg.norm(V,axis=0))))
    return U,singular,V.T,meta


def main():
    cfg=yaml.safe_load((r.HERE/'option_c_config.yaml').read_text())
    rows=[]
    with threadpool_limits(limits=4):
        for N in [128,256]:
            params=prepare(N,cfg)
            x=np.linspace(-1,1,N*cfg['sampling_density']+1)
            A=r.feature_matrix(x,params['centers'],params['slope'])
            F=A*params['D']/math.sqrt(len(x))
            for joba,jobv in [(0,1),(2,1),(0,0),(2,0)]:
                U,s,Vt,meta=jacobi_svd(F,joba=joba,jobv=jobv)
                row=dict(N=N,shape=F.shape,**meta)
                try:
                    _,bounds=oc.certify_factorization(F,U,s,Vt,params)
                    row.update(bounds);row['valid_nearby_problem']=True
                except AssertionError:
                    row.update(valid_nearby_problem=False,
                        measured_oU=float(np.linalg.norm(U.T@U-np.eye(F.shape[1]),2)),
                        measured_oV=float(np.linalg.norm(Vt@Vt.T-np.eye(F.shape[1]),2)))
                rows.append(row)
                print(json.dumps(row),flush=True)
    dest=r.OUT/'option_c'/'jacobi_probe_20261109.json'
    if dest.exists():raise FileExistsError(f'Preserving previous result {dest}')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
