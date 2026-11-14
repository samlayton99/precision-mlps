"""Solver-independent continuous RMS bounds for stored FP64 networks.

Fits are untouched. The verification grid is composite Gauss--Legendre,
with enclosed roots/weights and an analytic local-ellipse remainder.
The bound is for the exact real network with the saved coefficients; it
does not certify every possible FP64 implementation of its forward pass.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import gmpy2 as gp
import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expC14_theory_solver_comparison import run as r
from experiments.expC14_theory_solver_comparison.ablation import OUT as ABLATION

PREC=256
DEGREE=24
RHO=8
OUT=r.OUT/'posterior_verification'


def directed(fn,rounding):
    with gp.context(gp.get_context(),precision=PREC,round=rounding):return fn()


def add(a,b):
    return (directed(lambda:a[0]+b[0],gp.RoundDown),directed(lambda:a[1]+b[1],gp.RoundUp))


def neg(a):return (-a[1],-a[0])
def sub(a,b):return add(a,neg(b))


def mul(a,b):
    pairs=[(x,y) for x in a for y in b]
    return (min(directed(lambda:x*y,gp.RoundDown) for x,y in pairs),
            max(directed(lambda:x*y,gp.RoundUp) for x,y in pairs))


def div(a,b):
    assert not b[0]<=0<=b[1]
    inv=(directed(lambda:1/b[1],gp.RoundDown),directed(lambda:1/b[0],gp.RoundUp))
    return mul(a,inv)


def point(x):
    if isinstance(x,int):return (gp.mpfr(x),gp.mpfr(x))
    return (directed(lambda:gp.mpfr(x),gp.RoundDown),directed(lambda:gp.mpfr(x),gp.RoundUp))


def legendre_interval(n,x):
    prev=point(1);cur=x
    if n==0:return prev,point(0)
    for k in range(1,n):
        prev,cur=cur,div(sub(mul(point(2*k+1),mul(x,cur)),mul(point(k),prev)),point(k+1))
    derivative=div(mul(point(n),sub(prev,mul(x,cur))),sub(point(1),mul(x,x)))
    return cur,derivative


def quadrature(n=DEGREE+1):
    """Each of n disjoint intervals contains a root by opposite signs.

    The degree is n, so these intervals contain all roots. The Gauss weight
    formula 2/((1-z^2)P'_n(z)^2) is enclosed with directed MPFR arithmetic.
    """
    with mp.workdps(95):
        approximations,_=mp.gauss_quadrature(n,'legendre')
        guesses=[mp.nstr(v,90) for v in approximations]
    nodes=[];weights=[];intervals=[]
    with gp.context(gp.get_context(),precision=PREC):
        radius=gp.mpfr(2)**-180
        for guess in guesses:
            z=gp.mpfr(guess);lo=z-radius;hi=z+radius
            pl,_=legendre_interval(n,(lo,lo));ph,_=legendre_interval(n,(hi,hi))
            assert (pl[1]<0<ph[0]) or (ph[1]<0<pl[0])
            if intervals:assert intervals[-1][1]<lo
            _,dp=legendre_interval(n,(lo,hi))
            w=div(point(2),mul(sub(point(1),mul((lo,hi),(lo,hi))),mul(dp,dp)))
            assert w[0]>0
            nodes.append((lo+hi)/2);weights.append(div(w,point(2)));intervals.append((lo,hi))
        assert sum(w[0] for w in weights)<=1<=sum(w[1] for w in weights)
    return nodes,weights,intervals


def truth(x):
    pi=gp.const_pi()
    def sinpi(t):return gp.mpfr(0) if t==gp.floor(t) else gp.sin(pi*t)
    return [gp.exp(x),sinpi(4*x),sinpi(24*x),sinpi(8*(x+1)**2),
            sinpi(2*x)+sinpi(40*x)/1000,
            gp.exp(-8*(x-gp.mpfr(1)/5)**2)*sinpi(8*x)]


def local_bounds(N):
    # Enclosing rectangle of every core-cell ellipse (rho=8).
    with gp.context(gp.get_context(),precision=PREC,round=gp.RoundUp):
        h=gp.mpfr(2)/N;rho=gp.mpfr(RHO);pi=gp.const_pi()
        a=h*(rho+1/rho)/4;b=h*(rho-1/rho)/4
        xmax=1-h/2+a
        B=[gp.exp(xmax),gp.cosh(4*pi*b),gp.cosh(24*pi*b),
           gp.cosh(16*pi*(xmax+1)*b),gp.cosh(2*pi*b)+gp.cosh(40*pi*b)/1000,
           gp.exp(8*b*b)*gp.cosh(8*pi*b)]
        # tanh is bounded by one for |Im preactivation| <= pi/4.
        assert gp.mpfr(N)/8*b < pi/4
        return B


def verify(N,nodes,weights,intervals):
    start=time.perf_counter()
    source=ABLATION/f'N{N}_lambda0.25_s8_rc2.2204e-16.npz'
    saved=np.load(source)
    coeff=np.column_stack((saved['raw'],saved['scaled']))
    W=len(saved['centers']);n=W+1;R=(W-N-1)//2
    jindices=np.arange(-R,N+R+1)
    assert float(saved['gamma'])==N/8
    np.testing.assert_array_equal(saved['centers'],-1+2*jindices/N)
    diffmin=-N-R;diffmax=N-1+R
    # Exact integer upper bounds avoid any default-context rounding ambiguity.
    realmax=[gp.mpfr(v) for v in [3,1,1,1,2,1]]
    B=local_bounds(N)
    with gp.context(gp.get_context(),precision=PREC,round=gp.RoundUp):
        T=[sum(gp.mpfr(abs(float(v))) for v in coeff[:,j]) for j in range(12)]
        u=gp.mpfr(2)**-53
        k=1+math.ceil(math.log2(n));gam=k*u/(1-k*u)
        # Includes CR feature storage, rounded products/pairwise sums,
        # target storage, residual subtraction, and high-precision oracle/root
        # uncertainty. 1e-40 dominates the explicitly checked tiny last term.
        oracle_allowance=gp.mpfr('1e-40')
        eval_error=[(3*u+gam*(1+u))*T[j]+3*u*realmax[j%6]+oracle_allowance for j in range(12)]
        target_error=[u*v+oracle_allowance for v in realmax]
        # Root radius < 2^-179; mapping into a cell reduces it by 1/N.
        # |q'| <= gamma*T, |f'| < 300 on the real interval for these targets.
        root_error=[(gp.mpfr(N)/8*T[j]+300)/N*gp.mpfr(2)**-179 for j in range(12)]
        # MPFR oracle propagation: <100 operations per expression; real target
        # derivatives <=300; sin arguments <=128; exp outputs <=e; tanh's
        # derivative <=1. Even 1e6*(N+1)*(T+1)*2^-256 is conservative.
        # We use 2^-230 (another 2^26 margin), covering constant preparation
        # and the tanh lattice arguments. Subnormal FP64 allowances are also
        # negligible compared with the reserved 1e-40 absolute budget.
        mpfr_error=[gp.mpfr(2)**-230*(N+1)*(T[j]+1)*gp.mpfr('1e6') for j in range(12)]
        assert max(root_error[j]+mpfr_error[j] for j in range(12)) < oracle_allowance
    residual_squares=[gp.mpfr(0) for _ in range(12)]
    target_squares_lower=[gp.mpfr(0) for _ in range(6)]
    observed_squares=np.zeros(12)
    with gp.context(gp.get_context(),precision=PREC):
        for l,(z,weight) in enumerate(zip(nodes,weights)):
            # gamma*(x_k(z)-center_j)=lambda*(k+1/2-j+z/2).
            table=np.array([float(gp.tanh((gp.mpfr(d)+gp.mpfr('.5')+z/2)/4))
                            for d in range(diffmin,diffmax+1)])
            yy=np.array([[float(v) for v in truth(-1+(2*gp.mpfr(k)+1+z)/N)] for k in range(N)])
            residual=np.empty((N,12))
            for lo in range(0,N,128):
                kk=np.arange(lo,min(lo+128,N))
                phi=np.column_stack((table[kk[:,None]-jindices[None,:]-diffmin],np.ones(len(kk))))
                for j in range(12):
                    pred=r.pairwise_sum(phi*coeff[:,j],axis=1)
                    residual[lo:lo+len(kk),j]=pred-yy[lo:lo+len(kk),j%6]
            observed_squares+=float((weight[0]+weight[1])/2)*np.mean(residual**2,axis=0)
            with gp.context(gp.get_context(),precision=PREC,round=gp.RoundUp):
                for j in range(12):
                    ss=sum((gp.mpfr(abs(float(v)))+eval_error[j])**2 for v in residual[:,j])
                    residual_squares[j]+=weight[1]*ss/N
            with gp.context(gp.get_context(),precision=PREC,round=gp.RoundDown):
                for j in range(6):
                    ss=sum(max(gp.mpfr(0),gp.mpfr(abs(float(v)))-target_error[j])**2 for v in yy[:,j])
                    target_squares_lower[j]+=weight[0]*ss/N
    rows=[]
    with gp.context(gp.get_context(),precision=PREC):
        for j in range(12):
            with gp.context(gp.get_context(),precision=PREC,round=gp.RoundUp):
                remainder=4*(B[j%6]+T[j])/(RHO-1)*gp.mpfr(RHO)**-DEGREE
                target_remainder=4*B[j%6]/(RHO-1)*gp.mpfr(RHO)**-DEGREE
                absolute=gp.sqrt(residual_squares[j])+remainder
            with gp.context(gp.get_context(),precision=PREC,round=gp.RoundDown):
                denominator=gp.sqrt(target_squares_lower[j%6])-target_remainder
            assert denominator>0
            with gp.context(gp.get_context(),precision=PREC,round=gp.RoundUp):
                relative=absolute/denominator
                # Controlled forward pipeline: exact power-of-two slope,
                # rounded centered subtraction, CR tanh, pairwise readout.
                # sup_y |y| sech^2(y)<1 bounds affine-rounding activation
                # error independently of gamma. Arbitrary real inputs are
                # rounded once; |q'| <= gamma*T handles this last conversion.
                forward_allowance=(3*u+gam*(1+3*u))*T[j]+gp.mpfr('1e-40')
                input_allowance=gp.mpfr(N)/8*T[j]*u+gp.mpfr('1e-40')
                fp64_relative=(absolute+forward_allowance+input_allowance)/denominator
            rows.append(dict(N=N,target=r.NAMES[j%6],method='raw' if j<6 else 'scaled',
                continuous_relative_l2_upper=r.up(relative),absolute_rms_upper=r.up(absolute),
                controlled_fp64_relative_l2_upper=r.up(fp64_relative),
                controlled_forward_absolute_allowance=r.up(forward_allowance),
                input_rounding_absolute_allowance=r.up(input_allowance),
                target_rms_lower=float(np.nextafter(float(denominator),-np.inf)),
                quadrature_observed_absolute_rms=float(math.sqrt(observed_squares[j])),
                residual_evaluation_allowance=r.up(eval_error[j]),analytic_remainder=r.up(remainder),
                coefficient_l1=r.up(T[j]),local_target_bound=r.up(B[j%6]),
                root_error_bound=r.up(root_error[j]),mpfr_oracle_error_allowance=r.up(mpfr_error[j]),
                quadrature_points=N*len(nodes),degree=DEGREE,rho=RHO,
                coefficients_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                scope='exact stored network; separate controlled-FP64 bound covers CR tanh, centered arithmetic and pairwise readout, not arbitrary libm/BLAS evaluators'))
    (OUT/f'N{N}.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(f'N={N} posterior bounds ({time.perf_counter()-start:.1f}s): '+', '.join(
        f"{x['target']} {x['continuous_relative_l2_upper']:.2e}" for x in rows[:6]),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--N',nargs='+',type=int,default=[128,256,512,1024,2048]);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    nodes,weights,intervals=quadrature()
    (OUT/'quadrature.json').write_text(json.dumps(dict(precision=PREC,degree=DEGREE,rho=RHO,
        root_enclosures=[[str(a),str(b)] for a,b in intervals],
        normalized_weight_enclosures=[[str(a),str(b)] for a,b in weights]),indent=2))
    with threadpool_limits(limits=4):
        for N in args.N:
            if not (OUT/f'N{N}.json').exists():verify(N,nodes,weights,intervals)


if __name__=='__main__':main()
