"""Predeclared elementary-function suite and physical-dx relative-L2 metric."""
import math
import numpy as np
import torch
from scipy.special import roots_legendre

A=1e-4
B=1.
TARGETS={
    'inv_sqrt':r'$x^{-1/2}$',
    'reciprocal':r'$1/x$',
    'inv_cuberoot':r'$x^{-1/3}$',
    'power_sum':r'$(1+x)/\sqrt{x}$',
    'root_plus_linear':r'$1/(\sqrt{x}+x)$',
    'root_quadratic':r'$1/\sqrt{x+x^2}$',
    'exp_over_root':r'$e^{-x}/\sqrt{x}$',
    'log_over_root':r'$-\log(x)/\sqrt{x}$',
    'neg_log':r'$-\log x$',
    'log1p':r'$\log(1+x)$',
    'exp_decay':r'$e^{-x}$',
    'atan':r'$\arctan x$',
}
POWER_TARGETS=['inv_sqrt','reciprocal','inv_cuberoot']


def target(name,x):
    lib=torch if isinstance(x,torch.Tensor) else np
    if name=='inv_sqrt':return x**-.5
    if name=='reciprocal':return 1/x
    if name=='inv_cuberoot':return x**(-1/3)
    if name=='power_sum':return (1+x)/lib.sqrt(x)
    if name=='root_plus_linear':return 1/(lib.sqrt(x)+x)
    if name=='root_quadratic':return 1/lib.sqrt(x+x*x)
    if name=='exp_over_root':return lib.exp(-x)/lib.sqrt(x)
    if name=='log_over_root':return -lib.log(x)/lib.sqrt(x)
    if name=='neg_log':return -lib.log(x)
    if name=='log1p':return lib.log1p(x)
    if name=='exp_decay':return lib.exp(-x)
    if name=='atan':return torch.atan(x) if isinstance(x,torch.Tensor) else np.arctan(x)
    raise ValueError(name)


def grid(name,order):
    edges=np.r_[A*2.**np.arange(math.ceil(math.log2(B/A))),B]
    z,w=roots_legendre(order);xs=[];weights=[]
    for left,right in zip(np.log(edges[:-1]),np.log(edges[1:])):
        x=np.exp((left+right)/2+(right-left)*z/2)
        xs.append(x);weights.append((right-left)*w*x/2)
    x=np.concatenate(xs)[:,None];weights=np.concatenate(weights)[:,None]
    y=target(name,x)
    return tuple(torch.from_numpy(v) for v in (x,y,weights))


@torch.no_grad()
def score(model,data):
    x,y,w=data
    prediction=torch.cat([model(part) for part in x.split(512)])
    return float(torch.sqrt((w*(prediction-y).square()).sum()/(w*y.square()).sum()))


def sample(name,rng,size,sampling,norm):
    if sampling=='log_importance':
        x=np.exp(rng.uniform(math.log(A),math.log(B),size))
        weights=x*math.log(B/A)/norm
    elif sampling=='uniform':
        x=rng.uniform(A,B,size);weights=np.full(size,(B-A)/norm)
    else:raise ValueError(sampling)
    return tuple(torch.from_numpy(v[:,None]) for v in (x,target(name,x),weights))
