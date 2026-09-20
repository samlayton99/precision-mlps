"""Final-grid evaluation and 80-digit audits of frozen selected parameters."""
import argparse
import json
import math
from pathlib import Path
import mpmath as mp
import numpy as np
from . import core,run,diagnose


def mp_target(x,name):
    if name=='sine':return mp.sqrt(2)*mp.sin(2*mp.pi*x)
    if name=='mixed':return (mp.sin(2*mp.pi*x)+mp.mpf('.1')*mp.sin(20*mp.pi*x))/mp.sqrt(mp.mpf('.505'))
    if name=='runge':return 1/(1+25*x*x)
    if name=='quadratic':return mp.sqrt(5)*x*x
    coefficients=np.zeros(10);coefficients[[0,1,int(name.removeprefix('moment'))]]=[.3,.4,np.sqrt(.75)]
    power=np.polynomial.legendre.leg2poly(core.polynomial_mapping()@coefficients)
    return mp.polyval([mp.mpf(float(v)) for v in power[::-1]],x)


def exact_native_coefficients(z,g,coordinates):
    if coordinates=='physical':return [mp.mpf(float(v)) for v in z[:g.width+1]]
    if coordinates=='neighbor':
        q=[mp.mpf(float(v))*mp.mpf(float(a)) for v,a in zip(z[1:g.width+1],np.cumsum(g.alpha[1:]))]
        return [mp.mpf(float(z[0]))*mp.mpf(float(g.alpha[0]))]+[q[i]-(q[i-1] if i else 0) for i in range(len(q))]
    scale=g.d if coordinates=='collective' else g.alpha
    return [mp.mpf(float(v))*mp.mpf(float(a)) for v,a in zip(z[:g.width+1],scale)]


def evaluate(folder,samples=257):
    config=json.loads((folder/'case.json').read_text());g=core.old.geometry(config['n'])
    checkpoint=sorted(folder.glob('snapshot_*.npz'))[-1]
    with np.load(checkpoint) as data:z=np.asarray(data['z']);step=int(data['step'])
    c,gamma=map(np.asarray,core.physical(z,g,config['coordinates']))
    x=-1+2*(np.arange(65536)+.5)/65536;y=core.target(x,config['target'],np)
    prediction=np.concatenate([diagnose.forward(z,g,config['coordinates'],block) for block in np.array_split(x,16)])
    subset=(np.arange(samples)*251+7919)%len(x)
    errors=[];native_errors=[];roundoff=[];target_roundoff=[]
    with mp.workdps(80):
        coefficients=[mp.mpf(float(v)) for v in c]
        exact_coefficients=exact_native_coefficients(z,g,config['coordinates'])
        slopes=[mp.mpf(float(v)) for v in gamma];centers=[mp.mpf(float(v)) for v in g.centers]
        offsets=[mp.mpf(float(v)) for v in np.asarray(core.offsets(z,g))]
        for index in subset:
            xx=mp.mpf(float(x[index]));truth=mp_target(xx,config['target'])
            features=[mp.tanh(ga*(xx-t)+beta) for ga,t,beta in zip(slopes,centers,offsets)]
            pred=coefficients[0]+mp.fdot(coefficients[1:],features)
            exact=exact_coefficients[0]+mp.fdot(exact_coefficients[1:],features)
            errors.append(float((pred-truth)**2));native_errors.append(float((exact-truth)**2))
            roundoff.append(float((mp.mpf(float(prediction[index]))-pred)**2))
            target_roundoff.append(float((mp.mpf(float(y[index]))-truth)**2))
    return dict(id=folder.name,step=step,final_grid_size=len(x),final_mse=float(np.mean((prediction-y)**2)),
        final_relative_mse=float(np.mean((prediction-y)**2)/np.mean(y*y)),
        final_max_abs_error=float(np.max(np.abs(prediction-y))),mp_dps=80,mp_samples=samples,
        mp_physical_mse=float(np.mean(errors)),mp_native_mse=float(np.mean(native_errors)),
        fp64_prediction_roundoff_mse=float(np.mean(roundoff)),fp64_target_roundoff_mse=float(np.mean(target_roundoff)),
        mp_indices=subset.tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--ids',nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();rows=[]
    for identity in args.ids:
        result=evaluate(args.root/identity);rows.append(result);print(json.dumps(result),flush=True)
    run.write_json(args.out,rows)


if __name__=='__main__':main()
