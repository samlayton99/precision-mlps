"""Final-grid evaluation and 80-digit audits of frozen selected parameters."""
import argparse
import json
import math
from pathlib import Path
import mpmath as mp
import numpy as np
from . import core,run,diagnose,stable


def resolved_quadrature(centers, gamma, order):
    """Gauss nodes split around learned transition widths; weights average [-1,1]."""
    edges=[-1.,1.]
    for center,slope in zip(centers,gamma):
        edges.append(center)
        if abs(slope)>1000:
            edges.extend(center+np.array([-.25,.25,-1.,1.,-4.,4.,-16.,16.])/abs(slope))
    edges=np.unique(np.clip(edges,-1,1));lo=edges[:-1];hi=edges[1:]
    nodes,weights=np.polynomial.legendre.leggauss(order)
    return ((lo[:,None]+hi[:,None])/2+(hi-lo)[:,None]*nodes/2).ravel(), \
        ((hi-lo)[:,None]*weights/4).ravel()


def spatial_audit(folder):
    """Check integration sensitivity without changing or selecting the model."""
    config=json.loads((folder/'case.json').read_text());g=core.old.geometry(config['n'])
    with np.load(sorted(folder.glob('snapshot_*.npz'))[-1]) as data:z=data['z'];step=int(data['step'])
    _,gamma=map(np.asarray,core.physical(z,g,config['coordinates']))
    beta=np.asarray(core.offsets(z,g))
    centers=g.centers-np.divide(beta,gamma,out=np.zeros_like(gamma),where=gamma!=0)
    def residual(x):
        return np.concatenate([diagnose.forward(z,g,config['coordinates'],b)-core.target(b,config['target'],np)
                               for b in np.array_split(x,max(1,int(np.ceil(len(x)/2048))))])
    uniform={}
    for size in (32768,65536,131072):
        x=-1+2*(np.arange(size)+.5)/size;uniform[str(size)]=float(np.mean(residual(x)**2))
    adaptive={}
    for order in (16,32,64):
        x,w=resolved_quadrature(centers,gamma,order)
        adaptive[str(order)]=dict(points=len(x),mse=float(w@(residual(x)**2)))
    return dict(id=folder.name,step=step,max_abs_gamma=float(np.max(abs(gamma))),
                midpoint_mse=uniform,transition_resolved_gauss=adaptive)


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
    errors=[];native_errors=[];roundoff=[];target_roundoff=[];stable_roundoff=[]
    stable_prediction=None
    if config['coordinates']=='neighbor':
        q=z[1:g.width+1]*np.cumsum(g.alpha[1:])
        def block(xx):
            pre=(xx[:,None]-g.centers)*gamma+np.asarray(core.offsets(z,g))
            return z[0]*g.alpha[0]+stable.difference(pre[:,:-1],pre[:,1:],np)@q[:-1]+np.tanh(pre[:,-1])*q[-1]
        stable_prediction=np.concatenate([block(xx) for xx in np.array_split(x,16)])
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
            if stable_prediction is not None:stable_roundoff.append(float((mp.mpf(float(stable_prediction[index]))-exact)**2))
            target_roundoff.append(float((mp.mpf(float(y[index]))-truth)**2))
    result=dict(id=folder.name,step=step,final_grid_size=len(x),final_mse=float(np.mean((prediction-y)**2)),
        final_relative_mse=float(np.mean((prediction-y)**2)/np.mean(y*y)),
        final_max_abs_error=float(np.max(np.abs(prediction-y))),mp_dps=80,mp_samples=samples,
        mp_physical_mse=float(np.mean(errors)),mp_native_mse=float(np.mean(native_errors)),
        fp64_prediction_roundoff_mse=float(np.mean(roundoff)),fp64_target_roundoff_mse=float(np.mean(target_roundoff)),
        mp_indices=subset.tolist())
    if stable_prediction is not None:
        result.update(stable_neighbor_final_mse=float(np.mean((stable_prediction-y)**2)),
                      stable_neighbor_roundoff_mse=float(np.mean(stable_roundoff)))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--ids',nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();rows=[]
    for identity in args.ids:
        result=evaluate(args.root/identity);rows.append(result);print(json.dumps(result),flush=True)
    run.write_json(args.out,rows)


if __name__=='__main__':main()
