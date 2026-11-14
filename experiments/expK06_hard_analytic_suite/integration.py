"""Physical-dx integration refined around NPU input zero crossings."""
import numpy as np
import torch
from scipy.special import roots_legendre
from experiments.expK06_hard_analytic_suite.common import quadrature
from experiments.expK06_hard_analytic_suite.targets import target


def zero_crossings(m):
    if not hasattr(m,'power') or m.variant=='positive':return []
    if m.variant=='native':return [0.]
    a=m.mix.detach().numpy()
    return [float(-b/s) for s,b in a if s!=0 and -1<=-b/s<=1]


def zero_refined_grid(roots,order):
    edges=list(np.linspace(-1,1,65))
    for r in roots:
        edges.append(r);offsets=np.geomspace(1e-18,.1,36)
        edges.extend(r-offsets);edges.extend(r+offsets)
    edges=np.unique(np.clip(edges,-1,1));z,w=roots_legendre(order)
    x=np.concatenate([(a+b)/2+(b-a)*z/2 for a,b in zip(edges[:-1],edges[1:])])
    weights=np.concatenate([(b-a)*w/2 for a,b in zip(edges[:-1],edges[1:])])
    return x,weights


@torch.no_grad()
def accurate_score(name,m,amp,offset,order=32):
    roots=zero_crossings(m)
    x,w=zero_refined_grid(roots,order) if roots else quadrature(order)
    y=target(name,x)
    yp=torch.cat([m(p) for p in torch.from_numpy(x[:,None]).split(512)]).numpy().ravel()*amp+amp*offset
    return float(np.sqrt(np.dot(w,(yp-y)**2)/np.dot(w,y*y)))
