"""Full-grid MP80 gradients at the two corrected SSBroyden failure checkpoints."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np

from . import core, run

KEYS=('ssbroyden_N512_s0_parameter_scale_08aa13232c',
      'ssbroyden_N512_s0_parameter_differences_1341e8769c')


def gradient_block(task):
    """Unnormalized sums, analytic target; both coefficient interpretations share slopes."""
    x,centers,gamma,coefficients,h=task
    with mp.workdps(80):
        centers=list(map(mp.mpf,centers));gamma=list(map(mp.mpf,gamma));h=mp.mpf(h)
        cs=[[mp.mpf(v) for v in c] for c in coefficients];width=len(gamma)
        gradients=[[mp.mpf(0)]*(2*width+1) for _ in cs];losses=[mp.mpf(0) for _ in cs]
        for xx in x:
            distance=[mp.mpf(xx)-t for t in centers]
            arguments=[v*d for v,d in zip(gamma,distance)]
            exponential=[mp.exp(-2*abs(a)) for a in arguments]
            phi=[mp.sign(a)*(1-e)/(1+e) for a,e in zip(arguments,exponential)]
            derivative=[d*4*e/(1+e)**2/h for d,e in zip(distance,exponential)]
            target=mp.sqrt(2)*mp.sin(2*mp.pi*mp.mpf(xx))
            for c,gradient,index in zip(cs,gradients,range(len(cs))):
                residual=c[0]+mp.fsum(w*p for w,p in zip(c[1:],phi))-target
                losses[index]+=residual**2;gradient[0]+=residual
                for j,(w,p,dp) in enumerate(zip(c[1:],phi,derivative)):
                    gradient[1+j]+=residual*p
                    gradient[width+1+j]+=residual*w*dp
        return gradients,losses


def pullback(gradient,alpha,coordinates):
    width=len(alpha)-1;readout=gradient[:width+1]
    if coordinates=='parameter_scale':native=[mp.mpf(a)*g for a,g in zip(alpha,readout)]
    else:
        cumulative=np.cumsum(alpha[1:])
        native=[mp.mpf(alpha[0])*readout[0]]
        native.extend(mp.mpf(cumulative[j])*(readout[j+1]-(readout[j+2] if j+1<width else 0)) for j in range(width))
    return native+gradient[width+1:]


def audit(root,key,pool):
    folder=root/key;case=json.loads((folder/'case.json').read_text());status=json.loads((folder/'latest.json').read_text())
    end=status['completed_updates'];g=core.geometry(case['n'])
    checkpoint=folder/f'checkpoint_{end:09d}.npz';state_path=folder/f'state_{end:09d}.pkl'
    with np.load(checkpoint) as a:c=a['c'];gamma=a['gamma'];z=a['native_parameters']
    with np.load(folder/'failure.npz') as a:stored_gradient=a['gradient']
    leaves,_=run.load_state(state_path)
    matrix=next(np.asarray(a) for a in leaves if a.shape==(2*g.width+1,2*g.width+1))
    x=np.linspace(-1,1,16*g.n+1)
    with mp.workdps(80):
        if case['coordinates']=='parameter_scale':native_c=[mp.mpf(a)*mp.mpf(v) for a,v in zip(g.alpha,z)]
        else:
            q=[mp.mpf(a)*mp.mpf(v) for a,v in zip(np.cumsum(g.alpha[1:]),z[1:g.width+1])]
            native_c=[mp.mpf(g.alpha[0])*mp.mpf(z[0])]+[v-(q[j-1] if j else 0) for j,v in enumerate(q)]
        parts=list(pool.map(gradient_block,[(chunk,g.centers,gamma,[c,native_c],g.h) for chunk in np.array_split(x,8)]))
        gradients=[[mp.fsum(p[0][k][j] for p in parts)/len(x) for j in range(2*g.width+1)] for k in range(2)]
        losses=[mp.fsum(p[1][k] for p in parts)/len(x) for k in range(2)]
        hessian_inverse=mp.matrix(matrix.tolist());stored=mp.matrix(stored_gradient.tolist())
        exact_direction=-(hessian_inverse*stored);fp_direction=mp.matrix((-matrix@stored_gradient).tolist())
        rows=[]
        for label,gradient,mse in zip(('stored_physical_parameters','exact_native_decode'),gradients,losses):
            native=mp.matrix(pullback(gradient,g.alpha,case['coordinates']))
            rows.append(dict(interpretation=label,mse=float(mse),gradient_norm=float(mp.norm(native)),
                relative_gradient_difference=float(mp.norm(native-stored)/mp.norm(native)),
                derivative_along_exact_stored_matrix_direction=float(mp.fdot(native,exact_direction)),
                derivative_along_fp64_matrix_direction=float(mp.fdot(native,fp_direction)),
                own_exact_metric_derivative=float(-mp.fdot(native,hessian_inverse*native))))
        stored_derivative=float(mp.fdot(stored,exact_direction))
    return dict(key=key,step=end,points=len(x),digits=80,target='analytic sqrt(2) sin(2 pi x), not bitwise GPU labels',
        stored_gradient_exact_metric_derivative=stored_derivative,rows=rows,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (checkpoint,state_path,folder/'failure.npz')})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--keys',nargs='+',default=KEYS,help='Saved failure identities to audit')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records=[]
        for key in args.keys:
            records.append(audit(args.root,key,pool));run.write_json(args.output/'full_gradient.json',records)
            print(json.dumps(records[-1]),flush=True)
    run.write_json(args.output/'provenance.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training_states_modified=False))


if __name__=='__main__':main()
