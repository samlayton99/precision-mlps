import json,math
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_K_practical_recipes/expK06_hard_analytic_suite'
NS=[64,128,256,512,1024,2048]
def halo(n):return max(20,math.ceil(math.sqrt(n)))
WIDTHS=[n+2*halo(n)+1 for n in NS]
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def quadrature(order=32,panels=64):
    z,w=roots_legendre(order);edges=np.linspace(-1,1,panels+1)
    x=np.concatenate([(a+b)/2+(b-a)*z/2 for a,b in zip(edges[:-1],edges[1:])])
    return x,np.tile(w/panels,panels)
def training_grid(n):return np.linspace(-1,1,max(2003,4*(n+1)))
