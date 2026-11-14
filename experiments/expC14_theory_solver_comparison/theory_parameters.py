"""Outward scalar enclosures of the ORIGINAL tanh theorem parameters.

No measured target error or fitted coefficient appears in these quantities.
Interval helpers are reused purely as arithmetic utilities, not as a change
to the construction theorem or a posterior verification substitute.
"""
from __future__ import annotations
import gmpy2 as gp
import numpy as np
from experiments.expC14_theory_solver_comparison.posterior import (
    PREC, directed, point, add, sub, mul, div, neg)
from experiments.expC14_theory_solver_comparison import run as r


def exp(x):
    return (directed(lambda:gp.exp(x[0]),gp.RoundDown),directed(lambda:gp.exp(x[1]),gp.RoundUp))


def sqrt(x):
    return (directed(lambda:gp.sqrt(x[0]),gp.RoundDown),directed(lambda:gp.sqrt(x[1]),gp.RoundUp))


def cosh(x):
    assert x[0]>=0
    return (directed(lambda:gp.cosh(x[0]),gp.RoundDown),directed(lambda:gp.cosh(x[1]),gp.RoundUp))


def power(x,k):
    # All uses are positive bases and integer exponents.
    assert x[0]>0
    if k<0:return div(point(1),power(x,-k))
    return (directed(lambda:x[0]**k,gp.RoundDown),directed(lambda:x[1]**k,gp.RoundUp))


def total(items):
    ans=point(0)
    for item in items:ans=add(ans,item)
    return ans


def fhi(value):return float(np.nextafter(float(value),np.inf))
def flo(value):return float(np.nextafter(float(value),-np.inf))


def prepare(N,cfg):
    h,R,slope,centers,checks=r.geometry(N,cfg)
    with gp.context(gp.get_context(),precision=PREC):
        pi=(directed(gp.const_pi,gp.RoundDown),directed(gp.const_pi,gp.RoundUp))
        log2=(directed(lambda:gp.log(2),gp.RoundDown),directed(lambda:gp.log(2),gp.RoundUp))
        lam,delta,Delta,aH=map(point,[cfg['lambda'],cfg['delta'],cfg['Delta'],cfg['a_H']])
        d=div(pi,point(2*slope));hh=div(point(2),point(N))
        alpha=[div(hh,mul(point(2),sub(delta,d))) for _ in centers]
        m=(R+1)//2
        zeta=exp(neg(mul(point(2),lam)))
        products=[point(1)]
        for k in range(1,m+1):products.append(mul(products[-1],sub(point(1),power(zeta,k))))
        scalar_D=add(div(pi,mul(point(2),lam)),div(mul(point(4),log2),pi))
        for i in range(1,m+1):
            li=div(exp(neg(mul(lam,point(i*(i+1)-1)))),mul(products[i-1],products[m-i]))
            for j in range(1,m+1):
                if i!=j:
                    li=mul(li,add(point(1),exp(neg(mul(lam,point(2*j-1))))))
            correction=div(mul(mul(hh,scalar_D),li),mul(point(2),delta))
            alpha[i-1]=add(alpha[i-1],correction)
            alpha[-i]=add(alpha[-i],correction)
        alpha.append(add(point(1),total(alpha)))
        sumalpha=total(alpha);L=sqrt(sumalpha)
        Dinterval=[sqrt(x) for x in alpha]
        D=np.array([float((x[0]+x[1])/2) for x in Dinterval])
        Derr=np.array([fhi(max(abs(gp.mpfr(float(v))-x[0]),abs(gp.mpfr(float(v))-x[1]))) for v,x in zip(D,Dinterval)])
        d_hi=np.array([fhi(x[1]) for x in Dinterval])
        Dhi_norm=sqrt(total([mul(point(float(v)),point(float(v))) for v in d_hi]))[1]
        Derr_norm=sqrt(total([mul(point(float(v)),point(float(v))) for v in Derr]))[1]
        cap1=mul(pi,delta);cap2=mul(lam,mul(aH,aH))
        capp=div((min(cap1[0],cap2[0]),min(cap1[1],cap2[1])),point(8))
        W=len(centers)
        pref=div(mul(point(128),pi),mul(lam,delta))
        alias=exp(neg(div(mul(pi,pi),lam)))
        EperB=mul(pref,add(exp(neg(mul(capp,point(W)))),div(alias,point(W))))
        # B is selected as an upward rounded FP64 number, hence thereafter
        # an exact positive constant bounding the target class.
        Bs=[exp(add(add(point(1),Delta),delta)),cosh(mul(mul(point(4),pi),delta)),
            cosh(mul(mul(point(24),pi),delta)),
            cosh(mul(mul(mul(point(16),pi),add(add(point(2),Delta),delta)),delta)),
            add(cosh(mul(mul(point(2),pi),delta)),div(cosh(mul(mul(point(40),pi),delta)),point(1000))),
            mul(exp(mul(point(8),mul(delta,delta))),cosh(mul(mul(point(8),pi),delta)))]
        B=([x[1] for x in Bs] if cfg.get('extended_scalar_bounds')
           else np.array([fhi(x[1]) for x in Bs]))
        # Four relative rounding factors: tanh, sqrt(M), product, division.
        # D storage error is enclosed separately rather than assumed exact.
        u=gp.mpfr(2)**-53
        with gp.context(gp.get_context(),precision=PREC,round=gp.RoundUp):
            formation=(Derr_norm+(4*u/(1-4*u))*Dhi_norm)+gp.mpfr('1e-40')
        result=dict(N=N,h=h,R=R,slope=slope,centers=centers,checks=checks,
                    D=D,Derr=Derr,B=B,L_lower=flo(L[0]),L_upper=fhi(L[1]),
                    L_interval=L,E_per_B_upper=fhi(EperB[1]),
                    sumalpha_upper=fhi(sumalpha[1]),eF=fhi(formation),
                    alpha=np.array([float((x[0]+x[1])/2) for x in alpha]))
    return result
