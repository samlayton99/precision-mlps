"""Exact native hidden weights for arbitrary integer N at lambda=1/4.

gamma=N/8 and beta_j=N/8-j/4 are dyadic, even when centers are not.
Ideal training preactivations equal (i/s-j)/4. On independent dyadic
evaluation grids the affine preactivations are also exactly representable.
The lookup tables are correctly rounded MPFR tanh at 53 bits; returned
matrices and all readout arithmetic remain FP64.
"""
import math
import gmpy2 as gp
import numpy as np
from experiments.expC14_theory_solver_comparison import run as r
from experiments.expC14_theory_solver_comparison.posterior import truth


def cr_tanh_table(arguments):
    with gp.context(gp.get_context(),precision=53):
        return np.array([float(gp.tanh(gp.mpfr(float(a)))) for a in arguments])


def training(N,R,s):
    assert s>0 and s&(s-1)==0
    js=np.arange(-R,N+R+1)
    phase=np.arange(N*s+1)[:,None]-s*js[None,:]
    lower,upper=int(phase.min()),int(phase.max())
    table=cr_tanh_table(np.arange(lower,upper+1)/(4*s))
    A=np.column_stack((table[phase-lower],np.ones(N*s+1)))
    with gp.context(gp.get_context(),precision=160):
        y=np.array([[float(v) for v in truth(gp.mpfr(2*i-N*s)/(N*s))]
                    for i in range(N*s+1)])
    return A,y


def evaluate(N,R,coefficients,truth_values):
    count=len(truth_values)
    assert count>0 and count&(count-1)==0
    js=np.arange(-R,N+R+1)
    # Numerators over 8*count, exact in binary64 at the supported sizes.
    lower=N-2*count*int(js[-1])
    upper=N*(2*count-1)-2*count*int(js[0])
    stride=2*math.gcd(N,count)
    lattice=np.arange((upper-lower)//stride+1,dtype=np.int64)*stride+lower
    assert max(abs(lower),abs(upper))<2**53
    table=cr_tanh_table(lattice/(8*count))
    residual=np.empty_like(truth_values)
    for start in range(0,count,512):
        stop=min(start+512,count)
        numer=N*(2*np.arange(start,stop)+1)[:,None]-2*count*js[None,:]
        indices=(numer-lower)//stride
        assert np.all((numer-lower)%stride==0)
        phi=np.column_stack((table[indices],np.ones(stop-start)))
        for j in range(coefficients.shape[1]):
            pred=r.pairwise_sum(phi*coefficients[:,j],axis=1)
            residual[start:stop,j]=pred-truth_values[start:stop,j]
    return np.linalg.norm(residual,axis=0)/np.linalg.norm(truth_values,axis=0),np.max(abs(residual),axis=0)
