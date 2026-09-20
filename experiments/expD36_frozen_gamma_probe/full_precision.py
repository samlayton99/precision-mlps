"""Independent nominal-feature precision ladder for selected primary witnesses."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time
import mpmath as mp
import numpy as np
from . import core, full_core as f, full_screen as screen


def mp_scales(n,name):
    radius=int(np.ceil(np.sqrt(n))); width=n+2*radius+1
    h=mp.mpf(2)/n; lam=mp.mpf(1)/4; delta=mp.mpf(1)/4
    alpha=[h/(2*(delta-mp.pi*h/(2*lam)))]*width
    m=(radius+1)//2; zeta=mp.exp(-2*lam)
    p=[mp.mpf(1)]
    for i in range(1,m+1):
        p.append(p[-1]*(1-zeta**i))
    dlam=mp.pi/(2*lam)+4*mp.log(2)/mp.pi
    for i in range(1,m+1):
        li=zeta**(mp.mpf(i*(i+1)-1)/2)/(p[i-1]*p[m-i])
        li*=mp.fprod(1+zeta**(mp.mpf(j)-mp.mpf('.5')) for j in range(1,m+1) if j!=i)
        for slot in [i-1,width-i]:
            alpha[slot]+=h*dlam*li/(2*delta)
    bias=1+mp.fsum(alpha)
    if name=='raw':
        return [mp.mpf(1)]*(width+1)
    if name=='collective_neighbor':
        cumulative=[]; value=mp.mpf(0)
        for a in alpha:
            value+=a; cumulative.append(value)
        return [mp.sqrt(bias)]+[mp.sqrt(a) for a in cumulative]
    raise ValueError(name)


def target_mp(x,name):
    if name=='sine_mix_2_6_10':
        return mp.sin(2*mp.pi*x)+mp.sin(6*mp.pi*x)/2+mp.sin(10*mp.pi*x)/4
    if name=='exp_sin_3pi':
        return mp.exp(mp.sin(3*mp.pi*x))
    if name=='runge_25':
        return 1/(1+25*x*x)
    if name=='quadratic':
        return mp.sqrt(5)*x*x
    if name=='sine_2pi':
        return mp.sqrt(2)*mp.sin(2*mp.pi*x)
    raise ValueError(name)


def witness(task):
    n,m,gamma,k,digits,target,name,deadline=task; start=time.monotonic()
    with mp.workdps(digits):
        x=[mp.mpf(-1)+mp.mpf(2)*i/(m-1) for i in range(m)]
        y=[target_mp(z,target) for z in x]
        previous=[mp.mpf(0)]*m; p=[mp.mpf(1)]*m; tail=y.copy(); previous_a=mp.mpf(0)
        orth=mp.mpf(0)
        for degree in range(k+1):
            orth=max(orth,abs(mp.fsum(v*v for v in p)/m-1),
                     abs(mp.fsum(a*b for a,b in zip(p,previous))/m))
            coefficient=mp.fsum(a*b for a,b in zip(p,y))/m
            tail=[a-coefficient*b for a,b in zip(tail,p)]
            if degree<k:
                d=degree+1; a=mp.sqrt(mp.mpf(d*d)*(m*m-d*d)/((4*d*d-1)*(m-1)**2))
                previous,p=p,[(z*v-previous_a*w)/a for z,v,w in zip(x,p,previous)]
                previous_a=a
        tail_sq=mp.fsum(v*v for v in tail)/m; target_sq=mp.fsum(v*v for v in y)/m
        radius=int(np.ceil(np.sqrt(n))); half=n//2+radius
        gradients={}; exp_values=[mp.exp(2*gamma*z) for z in x]; ratio=mp.exp(-mp.mpf(4)*gamma/n)
        # Reflection permits both center signs without a second set of exponentials.
        for center in range(half+1):
            if time.monotonic()>deadline:
                raise TimeoutError('Precision witness deadline')
            phi=[(v-1)/(v+1) for v in exp_values]
            gradients[center]=mp.fsum(t*v for t,v in zip(tail,phi))/m
            if center:
                gradients[-center]=-mp.fsum(t*v for t,v in zip(reversed(tail),phi))/m
            exp_values=[v*ratio for v in exp_values]
        physical=[mp.fsum(tail)/m]+[gradients[i] for i in range(-half,half+1)]
        scale=mp_scales(n,name)
        if name.endswith('_neighbor'):
            physical=[physical[0]]+[physical[i]-physical[i+1] for i in range(1,len(physical)-1)]+[physical[-1]]
        mu=mp.fsum((a*b)**2 for a,b in zip(scale,physical))/tail_sq
        return dict(n=n,m=m,gamma=gamma,k=k,digits=digits,target=target,map=name,
            E=mp.nstr(mp.sqrt(tail_sq/target_sq),65),mu=mp.nstr(mu,65),
            polynomial_norm_neighbor_error=mp.nstr(orth,12),seconds=time.monotonic()-start,
            status='nominal_real_features_and_map_not_interval_certified')


def run(root,seconds,workers):
    deadline=time.monotonic()+seconds; cfg=json.loads((root/'manifest.json').read_text())['config']
    selected=[('raw',gamma,target) for gamma in [1,4] for target in cfg['targets']]
    selected += [('collective_neighbor',4,target) for target in cfg['robust_targets']]
    rows=[]; tasks=[]
    for name,gamma,target in selected:
        folder=root/'dictionaries'/screen.dictionary_id(cfg['n'],name,gamma)
        certs=json.loads((folder/'certificates.json').read_text())
        row=next(c for c in certs if c['kind']=='directional' and c['target']==target and c['epsilon']==.01)
        assert row['k'] is not None
        rows.append(dict(row,map=name,gamma=gamma))
        tasks.extend((cfg['n'],cfg['samples_per_cell']*cfg['n']+1,gamma,row['k'],digits,target,name,deadline)
                     for digits in [80,120])
    destination=root/'validation/full_precision.json'
    result=dict(witnesses=[],comparisons=[],complete=False)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for value in pool.map(witness,tasks):
            result['witnesses'].append(value); core.write_json(destination,result)
            print(f'PRECISION {value["map"]} {value["target"]} gamma={value["gamma"]} digits={value["digits"]}',flush=True)
    for row in rows:
        pair=[v for v in result['witnesses'] if all(v[k]==row[k] for k in ['map','gamma','target'])]
        with mp.workdps(120):
            relative=abs(mp.mpf(pair[0]['mu'])/mp.mpf(pair[1]['mu'])-1)
            digits=min(65.,float(-mp.log10(relative))) if relative else 65.
        high=float(pair[-1]['mu']); e=float(pair[-1]['E'])
        value=f.bound(np.array([e]),np.log(np.array([high])),.01,row['L'])
        result['comparisons'].append(dict(map=row['map'],gamma=row['gamma'],target=row['target'],k=row['k'],
            mu_fp64=row['mu'],mu_high=high,fp64_relative_difference=abs(row['mu']/high-1),
            agreeing_decimal_digits=digits,high_log10_bound=value['log10_bound'],
            caveat='L is still FP64; high precision agreement is not an interval certificate'))
    result['complete']=True; core.write_json(destination,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=800)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(); run(args.root,args.seconds,args.workers)


if __name__=='__main__':
    main()
