"""Finite neuron mixing preserving the complete hidden parameter Gram matrix."""
import numpy as np
from scipy.optimize import brentq


def concentration(h):
    e=np.sum(h*h,axis=0)
    return h.shape[1]*np.sqrt(np.sum(e**3)/np.sum(e)**3)


def pair(h,i,j,angle):
    u,v=h[:,i].copy(),h[:,j].copy()
    c,s=np.cos(angle),np.sin(angle)
    h[:,i],h[:,j]=c*u+s*v,-s*u+c*v


def redistribute(p,factor,path):
    """factor=0 equalizes energy; other factors multiply sqrt(chi6).

    The two paths use different seeded pair orders. No readout fitting or
    direct scale dilation is performed. Nonlinear outputs need not match.
    """
    original=np.asarray(p,dtype=float)
    h=original[:-1].reshape(3,-1).copy()
    reference=h@h.T
    w=h.shape[1]
    rng=np.random.default_rng(1800+int(path))
    before=concentration(h)
    rotations=0
    if factor==0:
        mean=np.sum(h*h)/w
        for _ in range(2*w):
            e=np.sum(h*h,axis=0)
            high=np.flatnonzero(e>mean*(1+2e-13))
            low=np.flatnonzero(e<mean*(1-2e-13))
            if not len(high) or not len(low):break
            i,j=int(rng.choice(high)),int(rng.choice(low))
            u,v=h[:,i].copy(),h[:,j].copy()
            objective=lambda angle:np.sum((np.cos(angle)*u+np.sin(angle)*v)**2)-mean
            angle=brentq(objective,0,np.pi/2,xtol=5e-15)
            pair(h,i,j,angle);rotations+=1
        goal=1.
    else:
        goal=factor*before
        anchor=int(rng.choice(np.argsort(np.sum(h*h,axis=0))[-min(8,w):]))
        done=False
        for _ in range(4):
            for j in rng.permutation(w):
                if j==anchor:continue
                u,v=h[:,anchor].copy(),h[:,j].copy()
                angle=.5*np.arctan2(2*u@v,u@u-v@v)
                pair(h,anchor,j,angle)
                if concentration(h)>=goal:
                    h[:,anchor],h[:,j]=u,v
                    def objective(t):
                        h[:,anchor],h[:,j]=u,v
                        pair(h,anchor,j,t*angle)
                        return concentration(h)-goal
                    fraction=brentq(objective,0,1,xtol=5e-15)
                    objective(fraction)
                    done=True;rotations+=1;break
                rotations+=1
            if done:break
    after=concentration(h)
    gram_error=float(np.linalg.norm(h@h.T-reference)/np.linalg.norm(reference))
    achieved=abs(after-goal)<=2e-11*max(goal,1.)
    valid=bool(achieved and gram_error<2e-12)
    result=original.copy();result[:-1]=h.reshape(-1)
    return result,dict(valid=valid,factor=factor,path=path,rotations=rotations,
                       initial_sqrtC6=float(before),requested_sqrtC6=float(goal),
                       achieved_sqrtC6=float(after),gram_relative_error=gram_error,
                       reason='accepted' if valid else 'concentration_or_gram_mismatch')
