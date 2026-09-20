"""MP80 full-grid checks of saved Newton curvature directions; no training."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np

from . import core, run


def curvature_block(task):
    x,centers,c,gamma,dc,dg=task
    with mp.workdps(80):
        centers,c,gamma,dc,dg=([mp.mpf(v) for v in a] for a in (centers,c,gamma,dc,dg))
        sums=[mp.mpf(0) for _ in range(5)]
        for xx in x:
            distance=[mp.mpf(xx)-t for t in centers]
            arg=[d*g for d,g in zip(distance,gamma)]
            exp=[mp.exp(-2*abs(a)) for a in arg]
            phi=[mp.sign(a)*(1-e)/(1+e) for a,e in zip(arg,exp)]
            sech=[4*e/(1+e)**2 for e in exp]
            da=[d*g for d,g in zip(distance,dg)]
            target=mp.sqrt(2)*mp.sin(2*mp.pi*mp.mpf(xx))
            residual=c[0]+mp.fdot(c[1:],phi)-target
            first=dc[0]+mp.fdot(dc[1:],phi)+mp.fsum(w*s*v for w,s,v in zip(c[1:],sech,da))
            second=mp.fsum(2*dw*s*v-2*w*p*s*v*v for dw,w,p,s,v in zip(dc[1:],c[1:],phi,sech,da))
            for k,value in enumerate((residual**2,target**2,first**2,residual*second,residual*first)):sums[k]+=value
        return sums


def loss_block(task):
    """Evaluate fixed FP64 trial parameters against the analytic target at MP80."""
    x,centers,c,gamma=task
    with mp.workdps(80):
        centers,c,gamma=([mp.mpf(v) for v in a] for a in (centers,c,gamma))
        total=mp.mpf(0)
        for xx in x:
            xx=mp.mpf(xx)
            residual=c[0]+mp.fsum(w*mp.tanh(s*(xx-t)) for w,s,t in zip(c[1:],gamma,centers))-mp.sqrt(2)*mp.sin(2*mp.pi*xx)
            total+=residual**2
        return total


def line_profiles(analysis,output,records,pool,workers):
    rows=[]
    for record in records:
        if record['case']['optimizer']!='newton' or not record['case'].get('warm_start'):continue
        g=core.geometry(record['case']['n']);folder=analysis/record['key']
        source=folder/f"curvature_direction_{record['end']}.npz"
        with np.load(source) as a:
            c,gamma,v,scales=(a[k] for k in ('c','gamma','minimum_direction','scales'))
        with np.load(folder/'curvature_profile.npz') as a:
            amplitude=float(a['amplitudes'][np.argmin(a['actual_mse'])])
        direction=v*scales;dc,dg=direction[:g.width+1],direction[g.width+1:]
        x=np.linspace(-1,1,16*g.n+1);points=[]
        with mp.workdps(80):
            baseline=None
            for t in (0.,amplitude,-amplitude,2*amplitude,10*amplitude):
                # Match the detached FP64 trial's physical parameter arithmetic.
                cc,gg=c+t*dc,gamma+t*dg
                loss=mp.fsum(pool.map(loss_block,[(block,g.centers,cc,gg) for block in np.array_split(x,workers)]))/len(x)
                if baseline is None:baseline=loss
                points.append(dict(amplitude=t,mse=float(loss),relative_change=float((loss-baseline)/baseline)))
        row=dict(key=record['key'],step=record['end'],points=len(x),digits=80,trials=points,
                 physical_trial_arithmetic='FP64, matching the detached line profile; evaluation MP80 against analytic sine',
                 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        rows.append(row);run.write_json(output/'line_profiles_mp80.json',rows);print(json.dumps(row),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--line-profiles',action='store_true',help='Verify the tiny warm-Newton profile gains at MP80 instead of curvature')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    records=json.loads((a.analysis/'optimizer_summary.json').read_text());rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        if a.line_profiles:
            line_profiles(a.analysis,a.output,records,pool,a.workers);return
        for record in records:
            if record['case']['optimizer']!='newton':continue
            g=core.geometry(record['case']['n']);path=a.analysis/record['key']/f"curvature_direction_{record['end']}.npz"
            if not path.exists():continue
            with np.load(path) as data:
                c,gamma,v,scales=(data[k] for k in ('c','gamma','minimum_direction','scales'))
            # Hold the saved FP64 physical direction fixed in the precision check.
            physical=v*scales;dc,dg=physical[:g.width+1],physical[g.width+1:]
            x=np.linspace(-1,1,16*g.n+1)
            tasks=[(block,g.centers,c,gamma,dc,dg) for block in np.array_split(x,a.workers)]
            with mp.workdps(80):
                parts=list(pool.map(curvature_block,tasks));values=[mp.fsum(part[k] for part in parts)/len(x) for k in range(5)]
                row=dict(key=record['key'],step=record['end'],points=len(x),digits=80,
                         mse=float(values[0]),l2re=float(mp.sqrt(values[0]/values[1])),
                         gauss_newton_curvature=float(values[2]),residual_curvature=float(values[3]),
                         total_curvature=float(values[2]+values[3]),directional_gradient=float(values[4]),
                         source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         target='analytic sqrt(2) sin(2 pi x); fixed stored physical parameters and direction')
            rows.append(row);run.write_json(a.output/'curvature_mp80.json',rows);print(json.dumps(row),flush=True)


if __name__=='__main__':main()
